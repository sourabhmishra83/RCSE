"""
08c_generate_frozen_baseline_predictions.py

Generate definitive calibrated prediction artifacts from the frozen RCSE
predictive baseline.

Frozen predictive specification
-------------------------------
Feature set : NO_MISMATCH
Model       : Multinomial logistic regression
Solver      : L-BFGS
Penalty     : L2
C           : 1.0
max_iter    : 5000
tol         : 1e-4
Calibration : one-vs-rest isotonic regression
Splits      : grouped IID, temporal, R4 exposure OOD

Input
-----
rcse_experimental_with_splits.parquet

Outputs
-------
frozen_predictions_grouped_iid.parquet
frozen_predictions_grouped_iid.csv
frozen_predictions_temporal.parquet
frozen_predictions_temporal.csv
frozen_predictions_ood_exposure.parquet
frozen_predictions_ood_exposure.csv
frozen_baseline_metrics.csv
frozen_baseline_per_class.csv
frozen_baseline_calibration.csv
frozen_baseline_convergence.csv
frozen_baseline_feature_manifest.json
frozen_baseline_metadata.json

Methodological boundary
-----------------------
transaction_exposure_eur and transaction_risk_stratum are NOT used as model
features. They are copied into the prediction artifacts only AFTER prediction
so the subsequent RCSE decision layer can combine predictive uncertainty with
business consequence.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLASSES = ["EXECUTE", "GATHER", "ESCALATE", "ABSTAIN"]

SPLIT_CONFIG = {
    "grouped_iid": {
        "column": "split_grouped_iid",
        "train": "train",
        "calibration": "calibration",
        "test": "test",
    },
    "temporal": {
        "column": "split_temporal",
        "train": "train",
        "calibration": "calibration",
        "test": "test",
    },
    "ood_exposure": {
        "column": "split_ood_exposure",
        "train": "train",
        "calibration": "calibration",
        "test": "ood_test_r4",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 08c: generate frozen RCSE baseline predictions."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <input parent>/frozen_baseline_results",
    )
    p.add_argument(
        "--regimes",
        nargs="+",
        default=["grouped_iid", "temporal", "ood_exposure"],
        choices=list(SPLIT_CONFIG.keys()),
    )
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument(
        "--no-csv-predictions",
        action="store_true",
        help="Skip CSV prediction exports; Parquet is always written.",
    )
    return p.parse_args()


def no_mismatch_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        # Explicitly excluded:
        # "inv_po_abs_rel_diff",
        # "inv_gr_abs_rel_diff",
        "events_total_case",
        "events_available_at_t0",
        "n_goods_receipts_before_t0",
        "n_cancel_gr_before_t0",
        "n_invoice_receipts_at_t0_history",
        "n_cancel_invoice_before_t0",
        "n_price_change_before_t0",
        "n_quantity_change_before_t0",
        "n_set_payment_block_before_t0",
        "n_remove_payment_block_before_t0",
        "n_service_entry_before_t0",
        "n_delete_po_before_t0",
        "n_block_po_before_t0",
        "n_reactivate_po_before_t0",
        "n_change_approval_before_t0",
        "n_release_po_before_t0",
        "po_to_invoice_days",
        "vendor_invoice_to_recorded_invoice_days",
        "last_gr_to_invoice_days",
    ]

    categorical_candidates = [
        "item_category",
        "company",
        "source_system",
        "document_type",
        "document_category",
        "item_type",
        "spend_classification",
        "spend_area",
        "sub_spend_area",
        "gr_based_invoice_verification",
        "goods_receipt_required",
        "gr_available_at_t0",
        "vendor_invoice_seen_at_t0",
        "po_created_at_t0",
        "had_gr_cancellation_before_t0",
        "had_invoice_cancellation_before_t0",
        "had_price_change_before_t0",
        "had_quantity_change_before_t0",
        "had_payment_block_before_t0",
    ]

    numeric = [c for c in numeric_candidates if c in df.columns]
    categorical = [c for c in categorical_candidates if c in df.columns]
    return numeric, categorical


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    C: float,
    max_iter: int,
    tol: float,
) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def fit_isotonic_calibrators(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
) -> dict[str, IsotonicRegression | None]:
    calibrators: dict[str, IsotonicRegression | None] = {}

    for j, cls in enumerate(CLASSES):
        y_bin = (y_cal == cls).astype(int)

        if len(np.unique(y_bin)) < 2:
            calibrators[cls] = None
            continue

        iso = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        iso.fit(p_cal[:, j], y_bin)
        calibrators[cls] = iso

    return calibrators


def apply_isotonic(
    p: np.ndarray,
    calibrators: dict[str, IsotonicRegression | None],
) -> np.ndarray:
    out = np.zeros_like(p, dtype=float)

    for j, cls in enumerate(CLASSES):
        iso = calibrators[cls]
        out[:, j] = p[:, j] if iso is None else iso.predict(p[:, j])

    sums = out.sum(axis=1, keepdims=True)
    zero_rows = sums.squeeze() <= 1e-15

    if np.any(zero_rows):
        out[zero_rows] = p[zero_rows]
        sums = out.sum(axis=1, keepdims=True)

    return out / sums


def multiclass_brier(
    y_true: np.ndarray,
    p: np.ndarray,
) -> float:
    lookup = {c: i for i, c in enumerate(CLASSES)}
    onehot = np.zeros_like(p, dtype=float)

    for i, label in enumerate(y_true):
        onehot[i, lookup[label]] = 1.0

    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def binary_ece(
    y_bin: np.ndarray,
    p: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    n = len(p)

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (
            (p >= lo) & (p <= hi)
            if i == bins - 1
            else (p >= lo) & (p < hi)
        )

        if not np.any(mask):
            continue

        total += (
            mask.sum() / n
            * abs(
                float(y_bin[mask].mean())
                - float(p[mask].mean())
            )
        )

    return float(total)


def evaluate_regime(
    df: pd.DataFrame,
    regime: str,
    numeric: list[str],
    categorical: list[str],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    cfg = SPLIT_CONFIG[regime]
    split_col = cfg["column"]

    train = df.loc[df[split_col] == cfg["train"]].copy()
    cal = df.loc[df[split_col] == cfg["calibration"]].copy()
    test = df.loc[df[split_col] == cfg["test"]].copy()

    features = numeric + categorical

    X_train = train[features]
    y_train = train["expected_action_reference"].astype(str).to_numpy()

    X_cal = cal[features]
    y_cal = cal["expected_action_reference"].astype(str).to_numpy()

    X_test = test[features]
    y_test = test["expected_action_reference"].astype(str).to_numpy()

    pipe = build_pipeline(
        numeric,
        categorical,
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print(
        f"   [{regime}] train={len(train):,} "
        f"cal={len(cal):,} test={len(test):,}"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipe.fit(X_train, y_train)

    convergence_warning = any(
        issubclass(w.category, ConvergenceWarning)
        for w in caught
    )

    lr = pipe.named_steps["model"]
    n_iter_values = np.asarray(lr.n_iter_, dtype=int).ravel()
    iterations_required = int(n_iter_values.max())

    converged = bool(
        iterations_required < args.max_iter
        and not convergence_warning
    )

    print(
        f"      iterations={iterations_required:,}/{args.max_iter:,} "
        f"warning={convergence_warning} converged={converged}"
    )

    if not converged:
        raise RuntimeError(
            f"Frozen baseline failed convergence in regime '{regime}'. "
            "Do not generate final prediction artifacts."
        )

    model_classes = list(lr.classes_)
    missing_classes = [c for c in CLASSES if c not in model_classes]
    if missing_classes:
        raise RuntimeError(
            f"{regime}: missing trained classes {missing_classes}"
        )

    order = [model_classes.index(c) for c in CLASSES]

    raw_cal = pipe.predict_proba(X_cal)[:, order]
    raw_test = pipe.predict_proba(X_test)[:, order]

    calibrators = fit_isotonic_calibrators(
        y_cal,
        raw_cal,
    )
    p_calibrated = apply_isotonic(
        raw_test,
        calibrators,
    )

    pred = np.array(CLASSES)[
        np.argmax(p_calibrated, axis=1)
    ]

    # Overall metrics
    overall = {
        "regime": regime,
        "solver": "lbfgs",
        "feature_config": "NO_MISMATCH",
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "C": float(args.C),
        "iterations_required": iterations_required,
        "convergence_warning": convergence_warning,
        "converged": converged,
        "train_episodes": len(train),
        "calibration_episodes": len(cal),
        "test_episodes": len(test),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            pred,
        ),
        "macro_f1": f1_score(
            y_test,
            pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_test,
            pred,
            labels=CLASSES,
            average="weighted",
            zero_division=0,
        ),
        "log_loss_raw": log_loss(
            y_test,
            raw_test,
            labels=CLASSES,
        ),
        "log_loss_calibrated": log_loss(
            y_test,
            p_calibrated,
            labels=CLASSES,
        ),
        "brier_raw": multiclass_brier(
            y_test,
            raw_test,
        ),
        "brier_calibrated": multiclass_brier(
            y_test,
            p_calibrated,
        ),
    }

    # Per-class metrics
    report = classification_report(
        y_test,
        pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    calibration_rows = []

    for j, cls in enumerate(CLASSES):
        row = report[cls]

        per_class_rows.append(
            {
                "regime": regime,
                "class": cls,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1-score"],
                "support": int(row["support"]),
            }
        )

        y_bin = (y_test == cls).astype(int)

        calibration_rows.append(
            {
                "regime": regime,
                "class": cls,
                "ece_raw": binary_ece(
                    y_bin,
                    raw_test[:, j],
                    args.ece_bins,
                ),
                "ece_calibrated": binary_ece(
                    y_bin,
                    p_calibrated[:, j],
                    args.ece_bins,
                ),
                "mean_raw_probability": float(
                    raw_test[:, j].mean()
                ),
                "mean_calibrated_probability": float(
                    p_calibrated[:, j].mean()
                ),
                "empirical_prevalence": float(
                    y_bin.mean()
                ),
            }
        )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        pred,
        labels=CLASSES,
    )

    pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    ).to_csv(
        out_dir / f"frozen_confusion_{regime}.csv"
    )

    # -------------------------------------------------------------
    # Definitive prediction artifact
    # -------------------------------------------------------------
    requested_columns = [
        "benchmark_episode_id",
        "source_case_id",
        "case_id",
        "expected_action_reference",
        "benchmark_arm",
        "intervention_type",
        "intervention_strength",
        "synthetic_intervention",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "decision_time",
        "item_category",
        split_col,
    ]

    artifact_cols = [
        c for c in requested_columns
        if c in test.columns
    ]

    artifact = test[artifact_cols].copy()

    artifact["evaluation_regime"] = regime
    artifact["frozen_feature_config"] = "NO_MISMATCH"
    artifact["frozen_solver"] = "lbfgs"

    for j, cls in enumerate(CLASSES):
        lower = cls.lower()
        artifact[f"p_raw_{lower}"] = raw_test[:, j]
        artifact[f"p_cal_{lower}"] = p_calibrated[:, j]

    artifact["predicted_action_frozen"] = pred
    artifact["max_calibrated_probability"] = (
        p_calibrated.max(axis=1)
    )

    eps = 1e-15
    entropy = -np.sum(
        p_calibrated
        * np.log(
            np.clip(
                p_calibrated,
                eps,
                1.0,
            )
        ),
        axis=1,
    )
    artifact["predictive_entropy"] = (
        entropy / np.log(len(CLASSES))
    )

    # Binary execution-risk quantities useful for Step 09.
    # This is NOT yet the RCSE decision.
    artifact["p_not_execute"] = (
        1.0 - artifact["p_cal_execute"]
    )

    artifact["p_requires_nonexecute_response"] = (
        artifact["p_cal_gather"]
        + artifact["p_cal_escalate"]
        + artifact["p_cal_abstain"]
    )

    parquet_path = (
        out_dir
        / f"frozen_predictions_{regime}.parquet"
    )
    artifact.to_parquet(
        parquet_path,
        index=False,
    )

    if not args.no_csv_predictions:
        artifact.to_csv(
            out_dir
            / f"frozen_predictions_{regime}.csv",
            index=False,
        )

    convergence_row = {
        "regime": regime,
        "solver": "lbfgs",
        "iterations_required": iterations_required,
        "max_iter": int(args.max_iter),
        "convergence_warning": convergence_warning,
        "converged": converged,
    }

    return (
        overall,
        per_class_rows,
        calibration_rows,
        convergence_row,
    )


def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    ).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent
        / "frozen_baseline_results"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("RCSE STEP 08c - FROZEN BASELINE PREDICTION GENERATOR")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")
    print("Model : NO_MISMATCH + L-BFGS + isotonic calibration")
    print(f"C={args.C}, tol={args.tol}, max_iter={args.max_iter}")

    df = pd.read_parquet(
        input_path
    )

    required = [
        "benchmark_episode_id",
        "source_case_id",
        "expected_action_reference",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    ]

    for regime in args.regimes:
        required.append(
            SPLIT_CONFIG[regime]["column"]
        )

    missing = [
        c for c in required
        if c not in df.columns
    ]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - "
            + "\n  - ".join(missing)
        )

    numeric, categorical = no_mismatch_features(
        df
    )

    # Hard guardrail: no consequence or rejected mismatch features.
    forbidden = {
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    }

    leaked = forbidden.intersection(
        numeric + categorical
    )
    if leaked:
        raise AssertionError(
            f"Forbidden features present: {leaked}"
        )

    print(f"\nRows: {len(df):,}")
    print(
        f"Numeric features    : {len(numeric)}"
    )
    print(
        f"Categorical features: {len(categorical)}"
    )

    overall_rows = []
    per_class_rows = []
    calibration_rows = []
    convergence_rows = []

    for regime in args.regimes:
        (
            overall,
            pc,
            cal,
            conv,
        ) = evaluate_regime(
            df=df,
            regime=regime,
            numeric=numeric,
            categorical=categorical,
            args=args,
            out_dir=out_dir,
        )

        overall_rows.append(overall)
        per_class_rows.extend(pc)
        calibration_rows.extend(cal)
        convergence_rows.append(conv)

    overall_df = pd.DataFrame(
        overall_rows
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    convergence_df = pd.DataFrame(
        convergence_rows
    )

    overall_df.to_csv(
        out_dir / "frozen_baseline_metrics.csv",
        index=False,
    )

    per_class_df.to_csv(
        out_dir / "frozen_baseline_per_class.csv",
        index=False,
    )

    calibration_df.to_csv(
        out_dir / "frozen_baseline_calibration.csv",
        index=False,
    )

    convergence_df.to_csv(
        out_dir / "frozen_baseline_convergence.csv",
        index=False,
    )

    manifest = {
        "feature_configuration": "NO_MISMATCH",
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_features": sorted(forbidden),
        "post_prediction_fields": [
            "transaction_exposure_eur",
            "transaction_risk_stratum",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
        ],
    }

    with open(
        out_dir / "frozen_baseline_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    metadata: dict[str, Any] = {
        "step": "08c",
        "input_file": str(input_path),
        "predictive_baseline_status": "FROZEN",
        "feature_config": "NO_MISMATCH",
        "solver": "lbfgs",
        "penalty": "l2",
        "C": float(args.C),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "calibration": (
            "one-vs-rest isotonic regression "
            "on dedicated calibration partition"
        ),
        "classes": CLASSES,
        "regimes": args.regimes,
        "convergence_required": True,
        "architectural_boundary": (
            "transaction exposure and risk stratum are excluded from "
            "predictive fitting and are attached only to final prediction "
            "artifacts for the downstream RCSE consequence layer."
        ),
        "step09_inputs": [
            f"frozen_predictions_{r}.parquet"
            for r in args.regimes
        ],
    }

    with open(
        out_dir / "frozen_baseline_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 08c COMPLETE")
    print("=" * 80)

    print("\nFrozen baseline metrics:")
    print(
        overall_df[
            [
                "regime",
                "iterations_required",
                "converged",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "brier_calibrated",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nPrediction artifacts:")
    for regime in args.regimes:
        print(
            "  - "
            + str(
                out_dir
                / f"frozen_predictions_{regime}.parquet"
            )
        )
        if not args.no_csv_predictions:
            print(
                "  - "
                + str(
                    out_dir
                    / f"frozen_predictions_{regime}.csv"
                )
            )

    print(
        "\nThese frozen prediction files are the direct inputs "
        "for Step 09 RCSE."
    )


if __name__ == "__main__":
    main()
