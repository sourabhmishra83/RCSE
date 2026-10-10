"""
09a4_train_post_gather_safety_estimator.py

Train and calibrate the post-GATHER binary safety estimator for RCSE.

Target
------
Primary target: D1_CORE on the exact t1 transition benchmark.

SAFE_TO_EXECUTE:
    eventually clears after t1 AND none of the D1 core adverse/control
    signals occur after t1.

NOT_SAFE_TO_EXECUTE:
    at least one D1 core adverse/control signal occurs after t1:
        - invoice cancellation
        - payment block set
        - GR cancellation
        - price change
        - quantity change
        - subsequent invoice

AMBIGUOUS:
    excluded from model fitting/evaluation, but counted and reported.

Inputs
------
--input
    t1_transition_exact.parquet from Step 09a.2

--splits
    rcse_experimental_with_splits.parquet from Step 06.
    Used only to recover the already-frozen source-case split assignments.

Model
-----
Binary logistic regression
Solver      : L-BFGS
Penalty     : L2
C           : 1.0
max_iter    : 5000
tol         : 1e-4
Calibration : isotonic regression on dedicated calibration partition

Critical guardrails
-------------------
The estimator MUST NOT use:
    transaction_exposure_eur
    transaction_risk_stratum
    any after_t1_* outcome field
    case/source identifiers
    absolute t0/t1 timestamps
    candidate target labels
    mismatch-derived diagnostic fields

The model estimates:

    q_t1 = P(SAFE_TO_EXECUTE | O_t1)

Business consequence is added only later by RCSE.

Outputs
-------
post_gather_metrics.csv
post_gather_per_class.csv
post_gather_calibration.csv
post_gather_convergence.csv
post_gather_class_balance.csv
post_gather_feature_manifest.json
post_gather_metadata.json

post_gather_predictions_grouped_iid.parquet
post_gather_predictions_temporal.parquet
post_gather_predictions_ood_exposure.parquet

and matching CSVs unless --no-csv-predictions is supplied.
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
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"

D1_CORE_SIGNALS = [
    "after_t1_invoice_cancelled",
    "after_t1_payment_block_set",
    "after_t1_gr_cancel",
    "after_t1_price_change",
    "after_t1_quantity_change",
    "after_t1_subsequent_invoice",
]

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
        description="Step 09a.4: train calibrated post-GATHER safety estimator."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to t1_transition_exact.parquet",
    )
    p.add_argument(
        "--splits",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <input parent>/post_gather_safety_results",
    )
    p.add_argument(
        "--regimes",
        nargs="+",
        default=["grouped_iid", "temporal", "ood_exposure"],
        choices=list(SPLIT_CONFIG.keys()),
    )
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument(
        "--no-csv-predictions",
        action="store_true",
        help="Skip CSV prediction exports; Parquet is always written.",
    )
    return p.parse_args()


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def build_d1_target(df: pd.DataFrame) -> pd.Series:
    unsafe = pd.Series(False, index=df.index)

    for c in D1_CORE_SIGNALS:
        unsafe = unsafe | as_bool(df[c])

    cleared = as_bool(df["after_t1_eventually_cleared"])

    target = pd.Series(
        "AMBIGUOUS",
        index=df.index,
        dtype="object",
    )

    target.loc[cleared & ~unsafe] = SAFE_LABEL
    target.loc[unsafe] = UNSAFE_LABEL

    return target


def build_split_map(
    split_df: pd.DataFrame,
    regimes: list[str],
) -> pd.DataFrame:
    required = ["source_case_id"] + [
        SPLIT_CONFIG[r]["column"] for r in regimes
    ]
    require_columns(split_df, required)

    work = split_df[required].copy()
    work["source_case_id"] = work["source_case_id"].astype(str)

    # Every source case must have exactly one partition per regime.
    for regime in regimes:
        col = SPLIT_CONFIG[regime]["column"]
        nunique = work.groupby("source_case_id")[col].nunique(dropna=False)
        bad = nunique[nunique > 1]
        if len(bad):
            raise AssertionError(
                f"{len(bad):,} source cases have conflicting '{col}' assignments."
            )

    # Collapse experimental variants to one row per source case.
    return (
        work.groupby("source_case_id", as_index=False)
        .first()
    )


def get_feature_sets(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Exact t1 feature set.

    Only state available at or before t1 is eligible.
    No exposure/risk, post-t1 outcomes, IDs, absolute timestamps, or
    mismatch-derived diagnostics.
    """

    numeric_candidates = [
        # monetary evidence available at t1
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_at_t1_eur",
        "first_gr_value_at_t1_eur",

        # exact state/event counts through t1
        "events_total_case",  # removed below: this includes future events
        "events_available_at_t0_exact",
        "events_between_t0_t1_inclusive_t1",
        "events_available_at_t1_exact",

        "n_goods_receipts_by_t1",
        "n_cancel_gr_by_t1",
        "n_invoice_receipts_by_t1",
        "n_cancel_invoice_by_t1",
        "n_price_change_by_t1",
        "n_quantity_change_by_t1",
        "n_set_payment_block_by_t1",
        "n_remove_payment_block_by_t1",
        "n_service_entry_by_t1",
        "n_delete_po_by_t1",
        "n_block_po_by_t1",
        "n_reactivate_po_by_t1",
        "n_change_approval_by_t1",
        "n_release_po_by_t1",

        # exact transition-window counts
        "between_n_goods_receipts",
        "between_n_cancel_gr",
        "between_n_invoice_receipts",
        "between_n_cancel_invoice",
        "between_n_price_change",
        "between_n_quantity_change",
        "between_n_set_payment_block",
        "between_n_remove_payment_block",
        "between_n_service_entry",
        "between_n_delete_po",
        "between_n_block_po",
        "between_n_reactivate_po",
        "between_n_change_approval",
        "between_n_release_po",

        # timing available at t1
        "elapsed_t0_to_t1_days",
        "po_to_t1_days",
        "vendor_invoice_to_t1_days",
        "last_gr_to_t1_days",
    ]

    categorical_candidates = [
        "company",
        "source_system",
        "document_type",
        "document_category",
        "item_type",
        "item_category",
        "spend_classification",
        "spend_area",
        "sub_spend_area",
        "gr_based_invoice_verification",
        "goods_receipt_required",
        "gr_available_at_t1",
        "vendor_invoice_seen_at_t1",
        "po_created_at_t1",
        "had_gr_cancellation_by_t1",
        "had_invoice_cancellation_by_t1",
        "had_price_change_by_t1",
        "had_quantity_change_by_t1",
        "had_payment_block_by_t1",
        "between_any_non_gr_event",
    ]

    # Explicitly reject any feature that includes future information.
    forbidden_exact = {
        "events_total_case",  # full-case total includes events after t1
        "events_after_t1",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "diag_inv_po_abs_rel_diff_t1",
        "diag_inv_gr_abs_rel_diff_t1",
        "time_t1_to_clear_days",
    }

    numeric = [
        c for c in numeric_candidates
        if c in df.columns and c not in forbidden_exact
    ]

    categorical = [
        c for c in categorical_candidates
        if c in df.columns and c not in forbidden_exact
    ]

    used = numeric + categorical

    # Hard-pattern guardrails.
    pattern_forbidden = [
        c for c in used
        if (
            c.startswith("after_t1_")
            or c.startswith("target_")
            or c.startswith("unsafe_any_")
            or c in {
                "case_id",
                "source_case_id",
                "purchasing_document",
                "item_id",
                "vendor_id",
                "vendor_name_anon",
                "t0_time",
                "t1_time",
                "candidate_t1_time",
                "t0_event_order",
                "t1_event_order",
            }
            or "risk_stratum" in c
            or "exposure" in c
            or "diag_inv_" in c
        )
    ]

    if pattern_forbidden:
        raise AssertionError(
            "Forbidden t1 predictive features detected:\n  - "
            + "\n  - ".join(sorted(pattern_forbidden))
        )

    return numeric, categorical, sorted(forbidden_exact)


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

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    lr = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=C,
        max_iter=max_iter,
        tol=tol,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", lr),
        ]
    )


def fit_isotonic(
    y_cal_binary: np.ndarray,
    p_cal_raw: np.ndarray,
) -> IsotonicRegression:
    if len(np.unique(y_cal_binary)) < 2:
        raise RuntimeError(
            "Calibration partition does not contain both binary classes."
        )

    iso = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )
    iso.fit(p_cal_raw, y_cal_binary)
    return iso


def binary_ece(
    y_true: np.ndarray,
    prob: np.ndarray,
    n_bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(prob)

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]

        mask = (
            (prob >= lo) & (prob <= hi)
            if i == n_bins - 1
            else (prob >= lo) & (prob < hi)
        )

        if not np.any(mask):
            continue

        total += (
            mask.sum() / n
            * abs(
                float(y_true[mask].mean())
                - float(prob[mask].mean())
            )
        )

    return float(total)


def evaluate_regime(
    resolved: pd.DataFrame,
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
    dict[str, Any],
]:
    cfg = SPLIT_CONFIG[regime]
    split_col = cfg["column"]

    train = resolved[resolved[split_col] == cfg["train"]].copy()
    cal = resolved[resolved[split_col] == cfg["calibration"]].copy()
    test = resolved[resolved[split_col] == cfg["test"]].copy()

    if train.empty or cal.empty or test.empty:
        raise RuntimeError(
            f"{regime}: empty train/calibration/test partition."
        )

    features = numeric + categorical

    y_train_label = train["y_t1_d1"].astype(str).to_numpy()
    y_cal_label = cal["y_t1_d1"].astype(str).to_numpy()
    y_test_label = test["y_t1_d1"].astype(str).to_numpy()

    # Positive class = SAFE_TO_EXECUTE.
    y_train = (y_train_label == SAFE_LABEL).astype(int)
    y_cal = (y_cal_label == SAFE_LABEL).astype(int)
    y_test = (y_test_label == SAFE_LABEL).astype(int)

    pipe = build_pipeline(
        numeric=numeric,
        categorical=categorical,
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
        pipe.fit(train[features], y_train)

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
            f"Post-GATHER model did not converge for {regime}."
        )

    # Find probability column for positive SAFE class.
    classes = list(lr.classes_)
    safe_idx = classes.index(1)

    p_cal_raw = pipe.predict_proba(cal[features])[:, safe_idx]
    p_test_raw = pipe.predict_proba(test[features])[:, safe_idx]

    iso = fit_isotonic(
        y_cal_binary=y_cal,
        p_cal_raw=p_cal_raw,
    )
    p_test_cal = iso.predict(p_test_raw)

    pred_binary = (p_test_cal >= 0.5).astype(int)
    pred_label = np.where(
        pred_binary == 1,
        SAFE_LABEL,
        UNSAFE_LABEL,
    )

    # -------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------
    overall = {
        "regime": regime,
        "solver": "lbfgs",
        "feature_config": "EXACT_T1_D1",
        "target_definition": "D1_CORE_RESOLVED",
        "C": float(args.C),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "iterations_required": iterations_required,
        "convergence_warning": convergence_warning,
        "converged": converged,
        "train_cases": int(len(train)),
        "calibration_cases": int(len(cal)),
        "test_cases": int(len(test)),
        "test_safe_prevalence": float(y_test.mean()),
        "test_unsafe_prevalence": float((1 - y_test).mean()),
        "accuracy": accuracy_score(y_test, pred_binary),
        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            pred_binary,
        ),
        "macro_f1": f1_score(
            y_test,
            pred_binary,
            average="macro",
            zero_division=0,
        ),
        "safe_f1": f1_score(
            y_test,
            pred_binary,
            pos_label=1,
            zero_division=0,
        ),
        "unsafe_f1": f1_score(
            y_test,
            pred_binary,
            pos_label=0,
            zero_division=0,
        ),
        "roc_auc_raw": roc_auc_score(
            y_test,
            p_test_raw,
        ),
        "roc_auc_calibrated": roc_auc_score(
            y_test,
            p_test_cal,
        ),
        # PR-AUC for SAFE positive class.
        "average_precision_safe_raw": average_precision_score(
            y_test,
            p_test_raw,
        ),
        "average_precision_safe_calibrated": average_precision_score(
            y_test,
            p_test_cal,
        ),
        # PR-AUC for UNSAFE positive class.
        "average_precision_unsafe_raw": average_precision_score(
            1 - y_test,
            1 - p_test_raw,
        ),
        "average_precision_unsafe_calibrated": average_precision_score(
            1 - y_test,
            1 - p_test_cal,
        ),
        "log_loss_raw": log_loss(
            y_test,
            np.column_stack(
                [1 - p_test_raw, p_test_raw]
            ),
            labels=[0, 1],
        ),
        "log_loss_calibrated": log_loss(
            y_test,
            np.column_stack(
                [1 - p_test_cal, p_test_cal]
            ),
            labels=[0, 1],
        ),
        "brier_raw": brier_score_loss(
            y_test,
            p_test_raw,
        ),
        "brier_calibrated": brier_score_loss(
            y_test,
            p_test_cal,
        ),
        "ece_raw": binary_ece(
            y_test,
            p_test_raw,
            args.ece_bins,
        ),
        "ece_calibrated": binary_ece(
            y_test,
            p_test_cal,
            args.ece_bins,
        ),
    }

    # Per-class metrics.
    report = classification_report(
        y_test_label,
        pred_label,
        labels=[SAFE_LABEL, UNSAFE_LABEL],
        output_dict=True,
        zero_division=0,
    )

    per_class = []
    for label in [SAFE_LABEL, UNSAFE_LABEL]:
        row = report[label]
        per_class.append(
            {
                "regime": regime,
                "class": label,
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1-score"],
                "support": int(row["support"]),
            }
        )

    calibration_rows = [
        {
            "regime": regime,
            "class": SAFE_LABEL,
            "empirical_prevalence": float(y_test.mean()),
            "mean_raw_probability": float(p_test_raw.mean()),
            "mean_calibrated_probability": float(
                p_test_cal.mean()
            ),
            "ece_raw": overall["ece_raw"],
            "ece_calibrated": overall["ece_calibrated"],
            "brier_raw": overall["brier_raw"],
            "brier_calibrated": overall["brier_calibrated"],
        },
        {
            "regime": regime,
            "class": UNSAFE_LABEL,
            "empirical_prevalence": float((1 - y_test).mean()),
            "mean_raw_probability": float(
                (1 - p_test_raw).mean()
            ),
            "mean_calibrated_probability": float(
                (1 - p_test_cal).mean()
            ),
            # Binary ECE is symmetric if expressed as complement.
            "ece_raw": binary_ece(
                1 - y_test,
                1 - p_test_raw,
                args.ece_bins,
            ),
            "ece_calibrated": binary_ece(
                1 - y_test,
                1 - p_test_cal,
                args.ece_bins,
            ),
            "brier_raw": overall["brier_raw"],
            "brier_calibrated": overall["brier_calibrated"],
        },
    ]

    # Confusion matrix.
    cm = confusion_matrix(
        y_test_label,
        pred_label,
        labels=[SAFE_LABEL, UNSAFE_LABEL],
    )

    pd.DataFrame(
        cm,
        index=[
            f"actual_{SAFE_LABEL}",
            f"actual_{UNSAFE_LABEL}",
        ],
        columns=[
            f"pred_{SAFE_LABEL}",
            f"pred_{UNSAFE_LABEL}",
        ],
    ).to_csv(
        out_dir / f"post_gather_confusion_{regime}.csv"
    )

    # -------------------------------------------------------------
    # Prediction artifacts
    # -------------------------------------------------------------
    artifact_cols = [
        c for c in [
            "case_id",
            "source_case_id",
            "item_category",
            "t0_time",
            "t1_time",
            "elapsed_t0_to_t1_days",
            "transaction_exposure_eur",
            "transaction_risk_stratum",
            split_col,
            "y_t1_d1",
        ]
        if c in test.columns
    ]

    pred = test[artifact_cols].copy()

    pred["evaluation_regime"] = regime
    pred["p_safe_t1_raw"] = p_test_raw
    pred["p_safe_t1_cal"] = p_test_cal
    pred["p_not_safe_t1_raw"] = 1.0 - p_test_raw
    pred["p_not_safe_t1_cal"] = 1.0 - p_test_cal
    pred["predicted_t1_safety"] = pred_label

    # Entropy for binary safety probability.
    eps = 1e-15
    q = np.clip(p_test_cal, eps, 1.0 - eps)
    entropy = -(
        q * np.log(q)
        + (1.0 - q) * np.log(1.0 - q)
    )
    pred["predictive_entropy_t1"] = entropy / np.log(2.0)

    pred.to_parquet(
        out_dir / f"post_gather_predictions_{regime}.parquet",
        index=False,
    )

    if not args.no_csv_predictions:
        pred.to_csv(
            out_dir / f"post_gather_predictions_{regime}.csv",
            index=False,
        )

    convergence_row = {
        "regime": regime,
        "iterations_required": iterations_required,
        "max_iter": int(args.max_iter),
        "convergence_warning": convergence_warning,
        "converged": converged,
    }

    # Partition class balance.
    balance_row = {
        "regime": regime,
        "train_cases": len(train),
        "train_safe": int(
            (train["y_t1_d1"] == SAFE_LABEL).sum()
        ),
        "train_unsafe": int(
            (train["y_t1_d1"] == UNSAFE_LABEL).sum()
        ),
        "calibration_cases": len(cal),
        "calibration_safe": int(
            (cal["y_t1_d1"] == SAFE_LABEL).sum()
        ),
        "calibration_unsafe": int(
            (cal["y_t1_d1"] == UNSAFE_LABEL).sum()
        ),
        "test_cases": len(test),
        "test_safe": int(
            (test["y_t1_d1"] == SAFE_LABEL).sum()
        ),
        "test_unsafe": int(
            (test["y_t1_d1"] == UNSAFE_LABEL).sum()
        ),
    }

    return (
        overall,
        per_class,
        calibration_rows,
        convergence_row,
        balance_row,
    )


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    split_path = Path(args.splits).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not split_path.exists():
        raise FileNotFoundError(split_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "post_gather_safety_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE STEP 09a.4 - POST-GATHER SAFETY ESTIMATOR")
    print("=" * 80)
    print(f"Input transitions: {input_path}")
    print(f"Frozen splits    : {split_path}")
    print(f"Output           : {out_dir}")
    print("Target           : D1_CORE resolved subset")
    print("Model            : L-BFGS logistic + isotonic calibration")

    print("\n1/6 Loading exact t1 transitions...")
    df = pd.read_parquet(input_path)

    require_columns(
        df,
        [
            "case_id",
            "after_t1_eventually_cleared",
            *D1_CORE_SIGNALS,
        ],
    )

    df["case_id"] = df["case_id"].astype(str)
    df["source_case_id"] = df["case_id"]

    df["y_t1_d1"] = build_d1_target(df)

    n_all = len(df)
    n_ambiguous = int((df["y_t1_d1"] == "AMBIGUOUS").sum())

    resolved = df[
        df["y_t1_d1"] != "AMBIGUOUS"
    ].copy()

    print(f"   All exact t1 cases : {n_all:,}")
    print(f"   Resolved D1 cases  : {len(resolved):,}")
    print(f"   Ambiguous excluded : {n_ambiguous:,}")
    print(
        f"   SAFE               : "
        f"{int((resolved['y_t1_d1'] == SAFE_LABEL).sum()):,}"
    )
    print(
        f"   NOT_SAFE           : "
        f"{int((resolved['y_t1_d1'] == UNSAFE_LABEL).sum()):,}"
    )

    print("\n2/6 Loading frozen source-case split assignments...")
    split_df = pd.read_parquet(split_path)
    split_map = build_split_map(
        split_df,
        args.regimes,
    )

    resolved = resolved.merge(
        split_map,
        on="source_case_id",
        how="left",
        validate="one_to_one",
    )

    # Require complete mapping for each requested regime.
    for regime in args.regimes:
        col = SPLIT_CONFIG[regime]["column"]
        missing = int(resolved[col].isna().sum())
        if missing:
            raise RuntimeError(
                f"{missing:,} resolved t1 cases are missing {col} mapping."
            )

    print(
        f"   Split mappings joined: {len(resolved):,}/{len(resolved):,}"
    )

    print("\n3/6 Building exact t1 feature manifest...")
    numeric, categorical, forbidden_exact = get_feature_sets(
        resolved
    )

    print(f"   Numeric features    : {len(numeric)}")
    print(f"   Categorical features: {len(categorical)}")

    # Strong global guardrail over the actual used feature set.
    used = numeric + categorical
    forbidden_used = [
        c for c in used
        if (
            c.startswith("after_t1_")
            or "exposure" in c
            or "risk_stratum" in c
            or "diag_inv_" in c
            or c in {
                "case_id",
                "source_case_id",
                "t0_time",
                "t1_time",
                "candidate_t1_time",
                "events_total_case",
                "events_after_t1",
                "time_t1_to_clear_days",
            }
        )
    ]
    if forbidden_used:
        raise AssertionError(
            "Leakage/consequence features in predictor:\n  - "
            + "\n  - ".join(sorted(forbidden_used))
        )

    print("\n4/6 Training/evaluating frozen regimes...")
    overall_rows = []
    per_class_rows = []
    calibration_rows = []
    convergence_rows = []
    balance_rows = []

    for regime in args.regimes:
        (
            overall,
            per_class,
            calibration,
            convergence,
            balance,
        ) = evaluate_regime(
            resolved=resolved,
            regime=regime,
            numeric=numeric,
            categorical=categorical,
            args=args,
            out_dir=out_dir,
        )

        overall_rows.append(overall)
        per_class_rows.extend(per_class)
        calibration_rows.extend(calibration)
        convergence_rows.append(convergence)
        balance_rows.append(balance)

    print("\n5/6 Writing results...")

    metrics_df = pd.DataFrame(overall_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    convergence_df = pd.DataFrame(convergence_rows)
    balance_df = pd.DataFrame(balance_rows)

    metrics_df.to_csv(
        out_dir / "post_gather_metrics.csv",
        index=False,
    )
    per_class_df.to_csv(
        out_dir / "post_gather_per_class.csv",
        index=False,
    )
    calibration_df.to_csv(
        out_dir / "post_gather_calibration.csv",
        index=False,
    )
    convergence_df.to_csv(
        out_dir / "post_gather_convergence.csv",
        index=False,
    )
    balance_df.to_csv(
        out_dir / "post_gather_class_balance.csv",
        index=False,
    )

    manifest = {
        "target": "D1_CORE_RESOLVED",
        "positive_class": SAFE_LABEL,
        "negative_class": UNSAFE_LABEL,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "explicitly_excluded": [
            "transaction_exposure_eur",
            "transaction_risk_stratum",
            "all after_t1_* outcomes",
            "case/source identifiers",
            "absolute t0/t1 timestamps",
            "events_total_case",
            "events_after_t1",
            "time_t1_to_clear_days",
            "diag_inv_po_abs_rel_diff_t1",
            "diag_inv_gr_abs_rel_diff_t1",
        ],
        "forbidden_exact_columns_present_but_not_used": forbidden_exact,
    }

    with open(
        out_dir / "post_gather_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(manifest, f, indent=2)

    metadata: dict[str, Any] = {
        "step": "09a.4",
        "input_transition_file": str(input_path),
        "split_source_file": str(split_path),
        "all_t1_cases": int(n_all),
        "resolved_d1_cases": int(len(resolved)),
        "ambiguous_excluded": int(n_ambiguous),
        "target_definition": {
            "SAFE_TO_EXECUTE": (
                "eventually clears after t1 and no D1 core adverse/control "
                "signal occurs after t1"
            ),
            "NOT_SAFE_TO_EXECUTE": (
                "at least one D1 core adverse/control signal occurs after t1"
            ),
            "AMBIGUOUS": (
                "excluded from model fitting/evaluation"
            ),
            "D1_core_signals": D1_CORE_SIGNALS,
        },
        "model": {
            "type": "binary logistic regression",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": float(args.C),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "calibration": "isotonic on dedicated calibration partition",
        },
        "split_policy": (
            "Reuse the already-frozen source-case split assignments from "
            "rcse_experimental_with_splits.parquet."
        ),
        "architectural_boundary": (
            "The post-GATHER estimator predicts P(SAFE_TO_EXECUTE | O_t1) "
            "without transaction exposure/risk stratum or post-t1 outcome "
            "features. Consequence is added only later in the RCSE decision layer."
        ),
        "next_step": (
            "If discrimination and calibration are adequate, freeze q_t1 "
            "prediction artifacts and implement the RCSE/VoI decision policy."
        ),
    }

    with open(
        out_dir / "post_gather_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n6/6 Complete.")
    print("\nPost-GATHER metrics:")
    print(
        metrics_df[
            [
                "regime",
                "iterations_required",
                "converged",
                "test_cases",
                "test_unsafe_prevalence",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "average_precision_unsafe_calibrated",
                "brier_calibrated",
                "ece_calibrated",
            ]
        ].to_string(index=False)
    )

    print("\nOutputs:")
    for name in [
        "post_gather_metrics.csv",
        "post_gather_per_class.csv",
        "post_gather_calibration.csv",
        "post_gather_convergence.csv",
        "post_gather_class_balance.csv",
        "post_gather_feature_manifest.json",
        "post_gather_metadata.json",
    ]:
        print(f"  - {out_dir / name}")

    for regime in args.regimes:
        print(
            "  - "
            + str(
                out_dir
                / f"post_gather_predictions_{regime}.parquet"
            )
        )
        if not args.no_csv_predictions:
            print(
                "  - "
                + str(
                    out_dir
                    / f"post_gather_predictions_{regime}.csv"
                )
            )

    print(
        "\nIf these metrics are acceptable, the calibrated p_safe_t1 values "
        "become the post-GATHER continuation-risk input for RCSE."
    )


if __name__ == "__main__":
    main()
