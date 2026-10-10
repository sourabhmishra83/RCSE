"""
08b_validate_solver_convergence.py

Validate SAGA versus L-BFGS convergence for the NO_MISMATCH RCSE baseline.

Goal
----
Test whether the high-exposure R4 degradation observed in Step 07 persists
when potentially ambiguous BPI monetary-comparison features are removed.

Input
-----
rcse_experimental_with_splits.parquet

Feature configurations
----------------------
FULL
    Current Step-07 predictive features.

NO_MISMATCH
    FULL minus:
        inv_po_abs_rel_diff
        inv_gr_abs_rel_diff

PROCESS_ONLY
    NO_MISMATCH minus raw monetary observation fields:
        po_value_initial_eur
        invoice_value_t0_eur
        latest_gr_value_before_t0_eur

Outputs
-------
solver_validation_metrics.csv
solver_validation_per_class.csv
solver_validation_calibration.csv
solver_validation_r4_dangerous_errors.csv
ablation_confusion_matrix_<feature_set>_<regime>.csv
solver_validation_feature_manifest.json
solver_validation_metadata.json

Method
------
For each feature configuration and each regime:
1. Fit multinomial logistic regression on train partition.
2. Calibrate on dedicated calibration partition using one-vs-rest isotonic.
3. Evaluate on test partition.
4. Compare discrimination, calibration, and specifically dangerous
   ESCALATE -> EXECUTE errors in R4 OOD.

Important
---------
Transaction exposure and risk stratum remain excluded from the predictive
model in every feature configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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

FEATURE_CONFIGS = ["FULL", "NO_MISMATCH", "PROCESS_ONLY"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RCSE baseline feature-ablation robustness checks."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <input parent>/ablation_results",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["grouped_iid", "temporal", "ood_exposure"],
        choices=list(SPLIT_CONFIG.keys()),
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["saga", "lbfgs"],
        choices=["saga", "lbfgs"],
        help="Logistic-regression solvers to compare.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Optimizer convergence tolerance.",
    )
    return parser.parse_args()


def base_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
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


def get_feature_config(
    df: pd.DataFrame,
    config: str,
) -> tuple[list[str], list[str]]:
    numeric, categorical = base_feature_sets(df)

    if config == "FULL":
        pass

    elif config == "NO_MISMATCH":
        numeric = [
            c for c in numeric
            if c not in {
                "inv_po_abs_rel_diff",
                "inv_gr_abs_rel_diff",
            }
        ]

    elif config == "PROCESS_ONLY":
        numeric = [
            c for c in numeric
            if c not in {
                "inv_po_abs_rel_diff",
                "inv_gr_abs_rel_diff",
                "po_value_initial_eur",
                "invoice_value_t0_eur",
                "latest_gr_value_before_t0_eur",
            }
        ]

    else:
        raise ValueError(f"Unknown feature config: {config}")

    return numeric, categorical


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    C: float,
    max_iter: int,
    solver: str,
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
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )

    kwargs = dict(
        solver=solver,
        penalty="l2",
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=42,
    )
    if solver == "saga":
        kwargs["n_jobs"] = -1
    model = LogisticRegression(**kwargs)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def fit_isotonic_calibrators(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
    classes: list[str],
) -> dict[str, IsotonicRegression | None]:
    calibrators: dict[str, IsotonicRegression | None] = {}

    for j, cls in enumerate(classes):
        y_bin = (y_cal == cls).astype(int)

        if len(np.unique(y_bin)) < 2:
            calibrators[cls] = None
            continue

        iso = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        iso.fit(proba_cal[:, j], y_bin)
        calibrators[cls] = iso

    return calibrators


def apply_isotonic(
    proba: np.ndarray,
    calibrators: dict[str, IsotonicRegression | None],
    classes: list[str],
) -> np.ndarray:
    calibrated = np.zeros_like(proba, dtype=float)

    for j, cls in enumerate(classes):
        cal = calibrators[cls]
        calibrated[:, j] = (
            proba[:, j]
            if cal is None
            else cal.predict(proba[:, j])
        )

    sums = calibrated.sum(axis=1, keepdims=True)
    zero = sums.squeeze() <= 1e-15

    if np.any(zero):
        calibrated[zero] = proba[zero]
        sums = calibrated.sum(axis=1, keepdims=True)

    return calibrated / sums


def multiclass_brier_score(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba, dtype=float)

    for i, y in enumerate(y_true):
        onehot[i, idx[y]] = 1.0

    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def binary_ece(
    y_true_binary: np.ndarray,
    prob: np.ndarray,
    n_bins: int,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(prob)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]

        if i == n_bins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)

        if not np.any(mask):
            continue

        conf = float(prob[mask].mean())
        acc = float(y_true_binary[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def evaluate_one(
    df: pd.DataFrame,
    solver: str,
    regime: str,
    C: float,
    max_iter: int,
    ece_bins: int,
    tol: float,
    out_dir: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    split_cfg = SPLIT_CONFIG[regime]
    split_col = split_cfg["column"]

    train = df.loc[df[split_col] == split_cfg["train"]].copy()
    cal = df.loc[df[split_col] == split_cfg["calibration"]].copy()
    test = df.loc[df[split_col] == split_cfg["test"]].copy()

    feature_config = "NO_MISMATCH"
    numeric, categorical = get_feature_config(df, feature_config)
    features = numeric + categorical

    forbidden = {
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    }
    if forbidden.intersection(features):
        raise AssertionError(
            f"Consequence features leaked into {feature_config}: "
            f"{forbidden.intersection(features)}"
        )

    X_train = train[features]
    y_train = train["expected_action_reference"].astype(str).to_numpy()

    X_cal = cal[features]
    y_cal = cal["expected_action_reference"].astype(str).to_numpy()

    X_test = test[features]
    y_test = test["expected_action_reference"].astype(str).to_numpy()

    model = build_pipeline(
        numeric,
        categorical,
        C=C,
        max_iter=max_iter,
        solver=solver,
        tol=tol,
    )

    print(
        f"   [{solver} | {feature_config} | {regime}] "
        f"train={len(train):,} cal={len(cal):,} test={len(test):,}"
    )

    model.fit(X_train, y_train)

    fitted_lr = model.named_steps["model"]
    n_iter_values = np.asarray(fitted_lr.n_iter_, dtype=int).ravel()
    iterations_required = int(n_iter_values.max())
    converged = bool(iterations_required < max_iter)

    print(
        f"      optimizer iterations: {iterations_required:,} / "
        f"{max_iter:,} | converged={converged}"
    )

    model_classes = list(fitted_lr.classes_)
    missing = [c for c in CLASSES if c not in model_classes]
    if missing:
        raise RuntimeError(
            f"{feature_config}/{regime}: missing training classes {missing}"
        )

    reorder = [model_classes.index(c) for c in CLASSES]

    p_cal_raw = model.predict_proba(X_cal)[:, reorder]
    p_test_raw = model.predict_proba(X_test)[:, reorder]

    calibrators = fit_isotonic_calibrators(
        y_cal,
        p_cal_raw,
        CLASSES,
    )

    p_test = apply_isotonic(
        p_test_raw,
        calibrators,
        CLASSES,
    )

    pred = np.array(CLASSES)[np.argmax(p_test, axis=1)]

    overall = {
        "solver": solver,
        "feature_config": feature_config,
        "regime": regime,
        "max_iter": int(max_iter),
        "iterations_required": iterations_required,
        "converged": converged,
        "numeric_feature_count": len(numeric),
        "categorical_feature_count": len(categorical),
        "train_episodes": len(train),
        "calibration_episodes": len(cal),
        "test_episodes": len(test),
        "accuracy": accuracy_score(y_test, pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
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
            p_test_raw,
            labels=CLASSES,
        ),
        "log_loss_calibrated": log_loss(
            y_test,
            p_test,
            labels=CLASSES,
        ),
        "brier_raw": multiclass_brier_score(
            y_test,
            p_test_raw,
            CLASSES,
        ),
        "brier_calibrated": multiclass_brier_score(
            y_test,
            p_test,
            CLASSES,
        ),
    }

    report = classification_report(
        y_test,
        pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class = []
    for cls in CLASSES:
        r = report[cls]
        per_class.append(
            {
                "solver": solver,
                "solver": solver,
                "feature_config": feature_config,
                "regime": regime,
                "class": cls,
                "precision": r["precision"],
                "recall": r["recall"],
                "f1": r["f1-score"],
                "support": int(r["support"]),
            }
        )

    cal_rows = []
    for j, cls in enumerate(CLASSES):
        y_bin = (y_test == cls).astype(int)

        cal_rows.append(
            {
                "feature_config": feature_config,
                "regime": regime,
                "class": cls,
                "ece_raw": binary_ece(
                    y_bin,
                    p_test_raw[:, j],
                    ece_bins,
                ),
                "ece_calibrated": binary_ece(
                    y_bin,
                    p_test[:, j],
                    ece_bins,
                ),
                "mean_raw_probability": float(
                    p_test_raw[:, j].mean()
                ),
                "mean_calibrated_probability": float(
                    p_test[:, j].mean()
                ),
                "empirical_prevalence": float(
                    y_bin.mean()
                ),
            }
        )

    cm = confusion_matrix(
        y_test,
        pred,
        labels=CLASSES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )

    safe_feature_name = feature_config.lower()
    cm_df.to_csv(
        out_dir
        / f"solver_validation_confusion_{solver}_{regime}.csv"
    )

    dangerous_summary = None
    if regime == "ood_exposure":
        actual_escalate = y_test == "ESCALATE"
        n_actual_escalate = int(actual_escalate.sum())

        escalate_to_execute = int(
            np.sum(
                actual_escalate
                & (pred == "EXECUTE")
            )
        )

        correct_escalate = int(
            np.sum(
                actual_escalate
                & (pred == "ESCALATE")
            )
        )

        dangerous_summary = {
            "solver": solver,
            "feature_config": feature_config,
            "regime": regime,
            "actual_escalate": n_actual_escalate,
            "escalate_predicted_execute": escalate_to_execute,
            "dangerous_error_rate_within_escalate": (
                escalate_to_execute / n_actual_escalate
                if n_actual_escalate
                else np.nan
            ),
            "correct_escalate": correct_escalate,
            "escalate_recall": (
                correct_escalate / n_actual_escalate
                if n_actual_escalate
                else np.nan
            ),
        }

    return overall, per_class, cal_rows, dangerous_summary


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "ablation_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE STEP 08b - SOLVER CONVERGENCE VALIDATION")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")
    print("Feature config : NO_MISMATCH")
    print(f"Solvers        : {', '.join(args.solvers)}")
    print(f"Regimes        : {', '.join(args.regimes)}")
    print(f"max_iter       : {args.max_iter}")
    print(f"tol            : {args.tol}")

    df = pd.read_parquet(input_path)

    required = [
        "expected_action_reference",
        "benchmark_episode_id",
        "source_case_id",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    ]
    for regime in args.regimes:
        required.append(SPLIT_CONFIG[regime]["column"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )

    print(f"\nRows: {len(df):,}")

    overall_rows = []
    per_class_rows = []
    calibration_rows = []
    dangerous_rows = []

    for solver in args.solvers:
        for regime in args.regimes:
            overall, pc, cal, dangerous = evaluate_one(
                df=df,
                solver=solver,
                regime=regime,
                C=args.C,
                max_iter=args.max_iter,
                ece_bins=args.ece_bins,
                tol=args.tol,
                out_dir=out_dir,
            )

            overall_rows.append(overall)
            per_class_rows.extend(pc)
            calibration_rows.extend(cal)

            if dangerous is not None:
                dangerous_rows.append(dangerous)

    overall_df = pd.DataFrame(overall_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    dangerous_df = pd.DataFrame(dangerous_rows)

    overall_df.to_csv(
        out_dir / "solver_validation_metrics.csv",
        index=False,
    )
    per_class_df.to_csv(
        out_dir / "solver_validation_per_class.csv",
        index=False,
    )
    calibration_df.to_csv(
        out_dir / "solver_validation_calibration.csv",
        index=False,
    )
    dangerous_df.to_csv(
        out_dir / "solver_validation_r4_dangerous_errors.csv",
        index=False,
    )

    # Feature manifest
    manifest = {}
    num, cat = get_feature_config(df, "NO_MISMATCH")
    manifest["NO_MISMATCH"] = {
        "numeric_features": num,
        "categorical_features": cat,
        "numeric_count": len(num),
        "categorical_count": len(cat),
    }

    manifest["excluded_consequence_features"] = [
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    ]

    with open(
        out_dir / "solver_validation_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    metadata: dict[str, Any] = {
        "input_file": str(input_path),
        "experiment": "solver_convergence_validation",
        "model": "multinomial logistic regression",
        "max_iter": int(args.max_iter),
        "convergence_rule": "converged iff max(model.n_iter_) < max_iter",
        "convergence_fields": [
            "max_iter",
            "iterations_required",
            "converged"
        ],
        "calibration": "one-vs-rest isotonic",
        "feature_config": "NO_MISMATCH",
        "solvers": args.solvers,
        "tol": float(args.tol),
        "regimes": args.regimes,
        "primary_question": (
            "Which logistic-regression solver converges across all three "
            "evaluation regimes for the selected NO_MISMATCH feature set?"
        ),
        "selection_rule": (
            "A solver is eligible to freeze only if converged=True in every "
            "requested regime. If both converge, compare discrimination and "
            "calibration rather than selecting solely on accuracy."
        ),
    }

    with open(
        out_dir / "solver_validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("ABLATION COMPLETE")
    print("=" * 80)

    print("\nOverall metrics:")
    print(
        overall_df[
            [
                "solver",
                "feature_config",
                "regime",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "brier_calibrated",
            ]
        ].to_string(index=False)
    )

    print("\nR4 dangerous error summary:")
    print(dangerous_df.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "solver_validation_metrics.csv",
        "solver_validation_per_class.csv",
        "solver_validation_calibration.csv",
        "solver_validation_r4_dangerous_errors.csv",
        "solver_validation_feature_manifest.json",
        "solver_validation_metadata.json",
    ]:
        print(f"  - {out_dir / name}")

    for solver in args.solvers:
        for regime in args.regimes:
            print(
                "  - "
                + str(
                    out_dir
                    / f"solver_validation_confusion_{solver}_{regime}.csv"
                )
            )



if __name__ == "__main__":
    main()
