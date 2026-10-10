"""
07_train_calibrated_baseline.py

Train and evaluate a calibrated multinomial logistic-regression baseline
for the RCSE experimental benchmark.

Input
-----
rcse_experimental_with_splits.parquet

Outputs
-------
baseline_metrics_overall.csv
baseline_metrics_per_class.csv
baseline_calibration_summary.csv
baseline_confusion_matrix_<regime>.csv
baseline_predictions_<regime>.parquet
baseline_predictions_<regime>.csv
baseline_feature_manifest.json
baseline_metadata.json

Regimes
-------
- grouped_iid
- temporal
- ood_exposure

Method
------
1. Fit multinomial logistic regression on train partition.
2. Use calibration partition to calibrate probabilities with isotonic
   one-vs-rest calibration.
3. Evaluate on the designated test partition.
4. Report:
   - accuracy
   - balanced accuracy
   - macro F1
   - weighted F1
   - log loss
   - multiclass Brier score
   - one-vs-rest ECE per class
   - confusion matrix
   - per-class precision/recall/F1/support

Important
---------
This script is a predictive baseline only. It does NOT implement RCSE.
Transaction exposure is excluded from the primary predictive feature set
so that later RCSE can consume consequence separately from predictive risk.
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
from sklearn.isotonic import IsotonicRegression


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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train calibrated logistic-regression RCSE baseline."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <input parent>/baseline_results",
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["grouped_iid", "temporal", "ood_exposure"],
        choices=list(SPLIT_CONFIG.keys()),
        help="Split regimes to evaluate.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum logistic-regression iterations.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength.",
    )
    parser.add_argument(
        "--ece-bins",
        type=int,
        default=15,
        help="Number of bins for expected calibration error.",
    )
    parser.add_argument(
        "--no-csv-predictions",
        action="store_true",
        help="Skip CSV prediction files; Parquet predictions are always written.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Feature definition
# ---------------------------------------------------------------------

def get_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Primary baseline features.

    Design choice:
    transaction_exposure_eur and transaction_risk_stratum are intentionally
    excluded from the predictive model so later RCSE can use consequence
    separately from predictive uncertainty.

    Future/outcome fields and intervention metadata are also excluded.
    """

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


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def multiclass_brier_score(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[str],
) -> float:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_onehot = np.zeros_like(proba, dtype=float)
    for i, y in enumerate(y_true):
        y_onehot[i, class_to_idx[y]] = 1.0
    return float(np.mean(np.sum((proba - y_onehot) ** 2, axis=1)))


def binary_ece(
    y_true_binary: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(prob)

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


# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

def fit_isotonic_calibrators(
    y_cal: np.ndarray,
    proba_cal: np.ndarray,
    classes: list[str],
) -> dict[str, IsotonicRegression | None]:
    calibrators: dict[str, IsotonicRegression | None] = {}

    for j, cls in enumerate(classes):
        y_bin = (y_cal == cls).astype(int)

        # Isotonic requires both classes to be present.
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


def apply_isotonic_calibration(
    proba: np.ndarray,
    calibrators: dict[str, IsotonicRegression | None],
    classes: list[str],
) -> np.ndarray:
    calibrated = np.zeros_like(proba, dtype=float)

    for j, cls in enumerate(classes):
        cal = calibrators[cls]
        if cal is None:
            calibrated[:, j] = proba[:, j]
        else:
            calibrated[:, j] = cal.predict(proba[:, j])

    # One-vs-rest isotonic outputs do not necessarily sum to 1.
    row_sums = calibrated.sum(axis=1, keepdims=True)

    # If all calibrated probabilities collapse to zero for a row,
    # fall back to the original probabilities.
    zero_rows = row_sums.squeeze() <= 1e-15
    if np.any(zero_rows):
        calibrated[zero_rows] = proba[zero_rows]
        row_sums = calibrated.sum(axis=1, keepdims=True)

    calibrated = calibrated / row_sums
    return calibrated


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    C: float,
    max_iter: int,
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

    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_regime(
    df: pd.DataFrame,
    regime: str,
    numeric_features: list[str],
    categorical_features: list[str],
    C: float,
    max_iter: int,
    ece_bins: int,
    out_dir: Path,
    write_csv_predictions: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:

    cfg = SPLIT_CONFIG[regime]
    split_col = cfg["column"]

    train = df.loc[df[split_col] == cfg["train"]].copy()
    cal = df.loc[df[split_col] == cfg["calibration"]].copy()
    test = df.loc[df[split_col] == cfg["test"]].copy()

    if train.empty or cal.empty or test.empty:
        raise RuntimeError(
            f"{regime}: train/calibration/test partition is empty."
        )

    feature_cols = numeric_features + categorical_features

    X_train = train[feature_cols]
    y_train = train["expected_action_reference"].astype(str).to_numpy()

    X_cal = cal[feature_cols]
    y_cal = cal["expected_action_reference"].astype(str).to_numpy()

    X_test = test[feature_cols]
    y_test = test["expected_action_reference"].astype(str).to_numpy()

    pipeline = build_pipeline(
        numeric_features,
        categorical_features,
        C=C,
        max_iter=max_iter,
    )

    print(f"   [{regime}] fitting logistic regression...")
    pipeline.fit(X_train, y_train)

    fitted_lr = pipeline.named_steps["model"]
    n_iter_values = np.asarray(fitted_lr.n_iter_, dtype=int).ravel()
    iterations_required = int(n_iter_values.max())
    converged = bool(iterations_required < max_iter)

    print(
        f"   [{regime}] optimizer iterations: {iterations_required:,} / "
        f"{max_iter:,} | converged={converged}"
    )

    model_classes = list(fitted_lr.classes_)

    # Ensure all expected classes are represented.
    missing_classes = [c for c in CLASSES if c not in model_classes]
    if missing_classes:
        raise RuntimeError(
            f"{regime}: model training is missing classes {missing_classes}"
        )

    # Reorder probabilities to canonical CLASSES order.
    class_idx = [model_classes.index(c) for c in CLASSES]

    raw_cal = pipeline.predict_proba(X_cal)[:, class_idx]
    raw_test = pipeline.predict_proba(X_test)[:, class_idx]

    calibrators = fit_isotonic_calibrators(
        y_cal,
        raw_cal,
        CLASSES,
    )

    cal_test = apply_isotonic_calibration(
        raw_test,
        calibrators,
        CLASSES,
    )

    pred_raw = np.array(CLASSES)[np.argmax(raw_test, axis=1)]
    pred_cal = np.array(CLASSES)[np.argmax(cal_test, axis=1)]

    # Overall metrics using calibrated predictions/probabilities.
    overall = {
        "regime": regime,
        "max_iter": int(max_iter),
        "iterations_required": iterations_required,
        "converged": converged,
        "train_episodes": len(train),
        "calibration_episodes": len(cal),
        "test_episodes": len(test),
        "accuracy": accuracy_score(y_test, pred_cal),
        "balanced_accuracy": balanced_accuracy_score(y_test, pred_cal),
        "macro_f1": f1_score(
            y_test,
            pred_cal,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_test,
            pred_cal,
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
            cal_test,
            labels=CLASSES,
        ),
        "brier_raw": multiclass_brier_score(
            y_test,
            raw_test,
            CLASSES,
        ),
        "brier_calibrated": multiclass_brier_score(
            y_test,
            cal_test,
            CLASSES,
        ),
    }

    # Per-class metrics
    report = classification_report(
        y_test,
        pred_cal,
        labels=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for cls in CLASSES:
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

    # Calibration metrics
    calibration_rows = []
    for j, cls in enumerate(CLASSES):
        y_bin = (y_test == cls).astype(int)

        calibration_rows.append(
            {
                "regime": regime,
                "class": cls,
                "ece_raw": binary_ece(
                    y_bin,
                    raw_test[:, j],
                    n_bins=ece_bins,
                ),
                "ece_calibrated": binary_ece(
                    y_bin,
                    cal_test[:, j],
                    n_bins=ece_bins,
                ),
                "mean_raw_probability": float(raw_test[:, j].mean()),
                "mean_calibrated_probability": float(
                    cal_test[:, j].mean()
                ),
                "empirical_prevalence": float(y_bin.mean()),
            }
        )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        pred_cal,
        labels=CLASSES,
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    )
    cm_df.to_csv(
        out_dir / f"baseline_confusion_matrix_{regime}.csv"
    )

    # Predictions
    pred_df = test[
        [
            "benchmark_episode_id",
            "source_case_id",
            "expected_action_reference",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
            "transaction_exposure_eur",
            "transaction_risk_stratum",
            split_col,
        ]
    ].copy()

    pred_df["predicted_action_raw"] = pred_raw
    pred_df["predicted_action_calibrated"] = pred_cal

    for j, cls in enumerate(CLASSES):
        pred_df[f"p_raw_{cls.lower()}"] = raw_test[:, j]
        pred_df[f"p_cal_{cls.lower()}"] = cal_test[:, j]

    # Predictive uncertainty summaries useful for later RCSE.
    pred_df["max_calibrated_probability"] = cal_test.max(axis=1)

    # Normalized entropy in [0,1]
    eps = 1e-15
    entropy = -np.sum(
        cal_test * np.log(np.clip(cal_test, eps, 1.0)),
        axis=1,
    )
    pred_df["predictive_entropy"] = entropy / np.log(len(CLASSES))

    pred_df.to_parquet(
        out_dir / f"baseline_predictions_{regime}.parquet",
        index=False,
    )
    if write_csv_predictions:
        pred_df.to_csv(
            out_dir / f"baseline_predictions_{regime}.csv",
            index=False,
        )

    return overall, per_class_rows, calibration_rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "baseline_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE CALIBRATED BASELINE TRAINER")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")
    print(f"Regimes: {', '.join(args.regimes)}")

    print("\n1/4 Loading split benchmark...")
    df = pd.read_parquet(input_path)

    required = [
        "expected_action_reference",
        "benchmark_episode_id",
        "source_case_id",
    ]
    for regime in args.regimes:
        required.append(SPLIT_CONFIG[regime]["column"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )

    numeric_features, categorical_features = get_feature_sets(df)

    if not numeric_features and not categorical_features:
        raise RuntimeError("No usable features were found.")

    print(f"   Rows: {len(df):,}")
    print(f"   Numeric features    : {len(numeric_features)}")
    print(f"   Categorical features: {len(categorical_features)}")

    # Explicitly ensure consequence is not in predictive feature set.
    forbidden = {
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    }
    used = set(numeric_features + categorical_features)
    leaked_consequence = forbidden.intersection(used)
    if leaked_consequence:
        raise AssertionError(
            f"Consequence variables leaked into baseline features: {leaked_consequence}"
        )

    print("\n2/4 Training/evaluating regimes...")

    overall_rows = []
    per_class_rows = []
    calibration_rows = []

    for regime in args.regimes:
        overall, pc, cal = evaluate_regime(
            df=df,
            regime=regime,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            C=args.C,
            max_iter=args.max_iter,
            ece_bins=args.ece_bins,
            out_dir=out_dir,
            write_csv_predictions=not args.no_csv_predictions,
        )

        overall_rows.append(overall)
        per_class_rows.extend(pc)
        calibration_rows.extend(cal)

    print("\n3/4 Writing metric summaries...")

    overall_df = pd.DataFrame(overall_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    overall_df.to_csv(
        out_dir / "baseline_metrics_overall.csv",
        index=False,
    )
    per_class_df.to_csv(
        out_dir / "baseline_metrics_per_class.csv",
        index=False,
    )
    calibration_df.to_csv(
        out_dir / "baseline_calibration_summary.csv",
        index=False,
    )

    feature_manifest = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_consequence_features": sorted(forbidden),
        "excluded_by_design": [
            "future/outcome fields",
            "benchmark/intervention metadata",
            "split columns",
            "source identifiers",
            "transaction consequence/exposure",
        ],
    }

    with open(
        out_dir / "baseline_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(feature_manifest, f, indent=2)

    metadata: dict[str, Any] = {
        "input_file": str(input_path),
        "model": "multinomial logistic regression",
        "solver": "saga",
        "regularization": "L2",
        "C": float(args.C),
        "max_iter": int(args.max_iter),
        "convergence_rule": "converged iff max(model.n_iter_) < max_iter",
        "convergence_fields": [
            "max_iter",
            "iterations_required",
            "converged"
        ],
        "calibration_method": "one-vs-rest isotonic regression",
        "calibration_partition": "dedicated split calibration partition",
        "classes": CLASSES,
        "regimes": args.regimes,
        "ece_bins": int(args.ece_bins),
        "primary_design_rule": (
            "Transaction exposure and risk stratum are excluded from the predictive "
            "model so later RCSE can combine calibrated predictive uncertainty with "
            "business consequence as a separate decision layer."
        ),
    }

    with open(
        out_dir / "baseline_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n4/4 Complete.")
    print("\nOverall metrics:")
    print(overall_df.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "baseline_metrics_overall.csv",
        "baseline_metrics_per_class.csv",
        "baseline_calibration_summary.csv",
        "baseline_feature_manifest.json",
        "baseline_metadata.json",
    ]:
        print(f"  - {out_dir / name}")

    for regime in args.regimes:
        print(f"  - {out_dir / f'baseline_confusion_matrix_{regime}.csv'}")
        print(f"  - {out_dir / f'baseline_predictions_{regime}.parquet'}")
        if not args.no_csv_predictions:
            print(f"  - {out_dir / f'baseline_predictions_{regime}.csv'}")


if __name__ == "__main__":
    main()
