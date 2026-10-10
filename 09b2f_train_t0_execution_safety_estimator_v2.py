#!/usr/bin/env python
"""
09b2f_train_t0_execution_safety_estimator.py

RCSE Step 09b.2f
Train and calibrate a dedicated natural-only t0 execution-safety estimator.

Primary target
--------------
D1_CORE_NATURAL

SAFE_TO_EXECUTE:
    required GR evidence is complete at t0
    AND the case eventually clears
    AND no D1 core adverse/control signal occurs after t0.

NOT_SAFE_TO_EXECUTE:
    required GR evidence is missing at t0
    OR at least one D1 core adverse/control signal occurs after t0.

AMBIGUOUS:
    neither rule resolves the case.
    Excluded from fitting and evaluation.

Important
---------
This estimator is SEPARATE from the frozen multiclass action model.
The original multiclass baseline remains unchanged.

Inputs
------
--base-v2
    rcse_base_v2.parquet

--splits
    rcse_experimental_with_splits.parquet
    Used only to recover already-frozen source-case split assignments.

Outputs
-------
t0_safety_metrics.csv
t0_safety_per_class.csv
t0_safety_calibration.csv
t0_safety_convergence.csv
t0_safety_class_balance.csv
t0_safety_feature_manifest.json
t0_safety_metadata.json

t0_safety_predictions_grouped_iid.parquet
t0_safety_predictions_temporal.parquet
t0_safety_predictions_ood_exposure.parquet

and matching CSVs unless --no-csv-predictions is supplied.

Model
-----
Binary logistic regression
Solver      : L-BFGS
Penalty     : L2
C           : 1.0
max_iter    : 5000
tol         : 1e-4
Calibration : isotonic regression on dedicated calibration partition

Critical feature exclusions
---------------------------
- all future outcome fields
- transaction_exposure_eur
- transaction_risk_stratum
- decision_year / decision_month
- IDs / vendor identifiers
- synthetic intervention metadata
- rejected mismatch variables:
    inv_po_abs_rel_diff
    inv_gr_abs_rel_diff
- gather/post-gather outcome variables
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
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"

CORE_SIGNALS = [
    "outcome_invoice_cancelled_after_t0",
    "outcome_payment_block_set_after_t0",
    "outcome_future_gr_cancel",
    "outcome_future_price_change",
    "outcome_future_quantity_change",
    "outcome_subsequent_invoice",
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
        description="Step 09b.2f: train natural-only t0 execution-safety estimator."
    )
    p.add_argument(
        "--base-v2",
        required=True,
        help="Path to rcse_base_v2.parquet",
    )
    p.add_argument(
        "--splits",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <splits parent>/t0_execution_safety_results",
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


def require_columns(
    df: pd.DataFrame,
    cols: list[str],
    label: str,
) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


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


def build_d1_target(df: pd.DataFrame) -> pd.Series:
    gr_required = as_bool(df["goods_receipt_required"])
    gr_available = as_bool(df["gr_available_at_t0"])

    evidence_complete = (~gr_required) | gr_available

    adverse = pd.Series(False, index=df.index)

    for c in CORE_SIGNALS:
        adverse = adverse | as_bool(df[c])

    cleared = as_bool(df["outcome_eventually_cleared"])

    target = pd.Series(
        "AMBIGUOUS",
        index=df.index,
        dtype="object",
    )

    # UNSAFE takes precedence.
    target.loc[(~evidence_complete) | adverse] = UNSAFE_LABEL

    target.loc[
        evidence_complete
        & cleared
        & (~adverse)
    ] = SAFE_LABEL

    return target


def build_split_map(
    split_df: pd.DataFrame,
    regimes: list[str],
) -> pd.DataFrame:
    required = ["source_case_id"] + [
        SPLIT_CONFIG[r]["column"]
        for r in regimes
    ]

    require_columns(
        split_df,
        required,
        "split benchmark",
    )

    work = split_df[required].copy()
    work["source_case_id"] = work["source_case_id"].astype(str)

    for regime in regimes:
        col = SPLIT_CONFIG[regime]["column"]

        conflicts = (
            work.groupby("source_case_id")[col]
            .nunique(dropna=False)
        )

        bad = conflicts[conflicts > 1]

        if len(bad):
            raise AssertionError(
                f"{len(bad):,} source cases have conflicting "
                f"'{col}' assignments."
            )

    return (
        work.groupby(
            "source_case_id",
            as_index=False,
        )
        .first()
    )


def get_feature_sets(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """
    Use the same general t0 evidence family as the frozen NO_MISMATCH action
    model, but never use future outcomes, consequence variables, IDs, or
    mismatch-derived fields.
    """

    numeric_candidates = [
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",

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
        "gr_available_at_t0",
        "vendor_invoice_seen_at_t0",
        "po_created_at_t0",
        "had_gr_cancellation_before_t0",
        "had_invoice_cancellation_before_t0",
        "had_price_change_before_t0",
        "had_quantity_change_before_t0",
        "had_payment_block_before_t0",
    ]

    numeric = [
        c for c in numeric_candidates
        if c in df.columns
    ]

    categorical = [
        c for c in categorical_candidates
        if c in df.columns
    ]

    explicit_exclusions = [
        "case_id",
        "source_case_id",
        "purchasing_document",
        "item_id",
        "vendor_id",
        "vendor_name_anon",
        "decision_time",
        "case_start_time",
        "case_end_time",
        "events_total_case",
        "events_future_after_t0",

        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",

        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "decision_year",
        "decision_month",

        "outcome_eventually_cleared",
        "outcome_invoice_cancelled_after_t0",
        "outcome_payment_block_removed_after_t0",
        "outcome_payment_block_set_after_t0",
        "outcome_future_gr",
        "outcome_future_gr_cancel",
        "outcome_future_price_change",
        "outcome_future_quantity_change",
        "outcome_subsequent_invoice",
        "outcome_additional_invoice_receipt",
        "outcome_future_service_entry",
        "outcome_any_post_t0_exception",

        "time_to_clear_days",
        "time_to_first_future_gr_days",
        "first_future_gr_time",
        "last_future_gr_time",
        "first_future_gr_value_eur",
        "latest_future_gr_value_eur",
        "n_future_goods_receipts",
        "first_clear_time",
        "first_payment_block_removal_time",
        "first_payment_block_set_time",
        "first_invoice_cancel_time",
        "first_cancel_invoice_value_eur",
        "first_additional_invoice_time",
        "first_additional_invoice_value_eur",
        "first_subsequent_invoice_time",
        "first_subsequent_invoice_value_eur",
        "first_future_price_change_time",
        "first_future_quantity_change_time",
        "future_sequence_preview",
        "gather_gr_possible",
        "gather_gr_has_value",
        "post_gather_inv_first_gr_abs_rel_diff",
        "post_gather_inv_latest_gr_abs_rel_diff",
    ]

    used = numeric + categorical

    # Pattern-based leakage guardrail.
    forbidden_used = [
        c for c in used
        if (
            c.startswith("outcome_")
            or c.startswith("first_future_")
            or c.startswith("latest_future_")
            or c.startswith("post_gather_")
            or "future_" in c
            or "exposure" in c
            or "risk_stratum" in c
            or c in {
                "inv_po_abs_rel_diff",
                "inv_gr_abs_rel_diff",
                "events_total_case",
                "events_future_after_t0",
                "case_id",
                "source_case_id",
            }
        )
    ]

    if forbidden_used:
        raise AssertionError(
            "Forbidden features detected:\n  - "
            + "\n  - ".join(sorted(forbidden_used))
        )

    return numeric, categorical, explicit_exclusions


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    C: float,
    max_iter: int,
    tol: float,
) -> Pipeline:

    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
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
            (
                "num",
                numeric_pipe,
                numeric,
            ),
            (
                "cat",
                categorical_pipe,
                categorical,
            ),
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
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def fit_isotonic(
    y_cal: np.ndarray,
    p_cal_raw: np.ndarray,
) -> IsotonicRegression:

    if len(np.unique(y_cal)) < 2:
        raise RuntimeError(
            "Calibration partition does not contain both safety classes."
        )

    iso = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    iso.fit(
        p_cal_raw,
        y_cal,
    )

    return iso


def binary_ece(
    y_true: np.ndarray,
    p: np.ndarray,
    n_bins: int,
) -> float:

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    total = 0.0
    n = len(p)

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]

        mask = (
            (p >= lo)
            & (p <= hi)
            if i == n_bins - 1
            else (
                (p >= lo)
                & (p < hi)
            )
        )

        if not np.any(mask):
            continue

        total += (
            mask.sum()
            / n
            * abs(
                float(
                    y_true[mask].mean()
                )
                - float(
                    p[mask].mean()
                )
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
    dict[str, Any],
]:

    cfg = SPLIT_CONFIG[regime]
    split_col = cfg["column"]

    train = df[
        df[split_col] == cfg["train"]
    ].copy()

    cal = df[
        df[split_col] == cfg["calibration"]
    ].copy()

    test = df[
        df[split_col] == cfg["test"]
    ].copy()

    if (
        train.empty
        or cal.empty
        or test.empty
    ):
        raise RuntimeError(
            f"{regime}: empty train/calibration/test partition."
        )

    features = numeric + categorical

    y_train_label = (
        train["y_t0_safety_d1"]
        .astype(str)
        .to_numpy()
    )

    y_cal_label = (
        cal["y_t0_safety_d1"]
        .astype(str)
        .to_numpy()
    )

    y_test_label = (
        test["y_t0_safety_d1"]
        .astype(str)
        .to_numpy()
    )

    # Positive class = SAFE_TO_EXECUTE.
    y_train = (
        y_train_label == SAFE_LABEL
    ).astype(int)

    y_cal = (
        y_cal_label == SAFE_LABEL
    ).astype(int)

    y_test = (
        y_test_label == SAFE_LABEL
    ).astype(int)

    pipe = build_pipeline(
        numeric=numeric,
        categorical=categorical,
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    print(
        f"   [{regime}] "
        f"train={len(train):,} "
        f"cal={len(cal):,} "
        f"test={len(test):,}"
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        pipe.fit(
            train[features],
            y_train,
        )

    convergence_warning = any(
        issubclass(
            w.category,
            ConvergenceWarning,
        )
        for w in caught
    )

    lr = pipe.named_steps[
        "model"
    ]

    n_iter_values = np.asarray(
        lr.n_iter_,
        dtype=int,
    ).ravel()

    iterations_required = int(
        n_iter_values.max()
    )

    converged = bool(
        iterations_required
        < args.max_iter
        and not convergence_warning
    )

    print(
        f"      iterations="
        f"{iterations_required:,}/"
        f"{args.max_iter:,} "
        f"warning={convergence_warning} "
        f"converged={converged}"
    )

    if not converged:
        raise RuntimeError(
            f"{regime}: t0 safety estimator did not converge."
        )

    classes = list(
        lr.classes_
    )

    safe_idx = classes.index(1)

    p_cal_raw = pipe.predict_proba(
        cal[features]
    )[:, safe_idx]

    p_test_raw = pipe.predict_proba(
        test[features]
    )[:, safe_idx]

    iso = fit_isotonic(
        y_cal=y_cal,
        p_cal_raw=p_cal_raw,
    )

    p_test_cal = iso.predict(
        p_test_raw
    )

    pred_binary = (
        p_test_cal >= 0.5
    ).astype(int)

    pred_label = np.where(
        pred_binary == 1,
        SAFE_LABEL,
        UNSAFE_LABEL,
    )

    overall = {
        "regime": regime,
        "target_definition": "D1_CORE_NATURAL_RESOLVED",
        "solver": "lbfgs",
        "C": float(args.C),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "iterations_required": iterations_required,
        "convergence_warning": convergence_warning,
        "converged": converged,

        "train_cases": int(len(train)),
        "calibration_cases": int(len(cal)),
        "test_cases": int(len(test)),

        "test_safe_prevalence": float(
            y_test.mean()
        ),
        "test_unsafe_prevalence": float(
            (1 - y_test).mean()
        ),

        "accuracy": accuracy_score(
            y_test,
            pred_binary,
        ),

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

        "average_precision_safe_raw": average_precision_score(
            y_test,
            p_test_raw,
        ),

        "average_precision_safe_calibrated": average_precision_score(
            y_test,
            p_test_cal,
        ),

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
                [
                    1 - p_test_raw,
                    p_test_raw,
                ]
            ),
            labels=[0, 1],
        ),

        "log_loss_calibrated": log_loss(
            y_test,
            np.column_stack(
                [
                    1 - p_test_cal,
                    p_test_cal,
                ]
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

    report = classification_report(
        y_test_label,
        pred_label,
        labels=[
            SAFE_LABEL,
            UNSAFE_LABEL,
        ],
        output_dict=True,
        zero_division=0,
    )

    per_class = []

    for label in [
        SAFE_LABEL,
        UNSAFE_LABEL,
    ]:
        row = report[label]

        per_class.append(
            {
                "regime": regime,
                "class": label,
                "precision": row[
                    "precision"
                ],
                "recall": row[
                    "recall"
                ],
                "f1": row[
                    "f1-score"
                ],
                "support": int(
                    row["support"]
                ),
            }
        )

    calibration = [
        {
            "regime": regime,
            "class": SAFE_LABEL,
            "empirical_prevalence": float(
                y_test.mean()
            ),
            "mean_raw_probability": float(
                p_test_raw.mean()
            ),
            "mean_calibrated_probability": float(
                p_test_cal.mean()
            ),
            "ece_raw": overall[
                "ece_raw"
            ],
            "ece_calibrated": overall[
                "ece_calibrated"
            ],
            "brier_raw": overall[
                "brier_raw"
            ],
            "brier_calibrated": overall[
                "brier_calibrated"
            ],
        },
        {
            "regime": regime,
            "class": UNSAFE_LABEL,
            "empirical_prevalence": float(
                (1 - y_test).mean()
            ),
            "mean_raw_probability": float(
                (1 - p_test_raw).mean()
            ),
            "mean_calibrated_probability": float(
                (1 - p_test_cal).mean()
            ),
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
            "brier_raw": overall[
                "brier_raw"
            ],
            "brier_calibrated": overall[
                "brier_calibrated"
            ],
        },
    ]

    cm = confusion_matrix(
        y_test_label,
        pred_label,
        labels=[
            SAFE_LABEL,
            UNSAFE_LABEL,
        ],
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
        out_dir
        / f"t0_safety_confusion_{regime}.csv"
    )

    artifact_cols = [
        c
        for c in [
            "source_case_id",
            "item_category",
            "decision_time",
            "transaction_exposure_eur",
            "transaction_risk_stratum",
            split_col,
            "y_t0_safety_d1",
        ]
        if c in test.columns
    ]

    pred = test[
        artifact_cols
    ].copy()

    pred[
        "evaluation_regime"
    ] = regime

    pred[
        "p_safe_t0_raw"
    ] = p_test_raw

    pred[
        "p_safe_t0_cal"
    ] = p_test_cal

    pred[
        "p_not_safe_t0_raw"
    ] = (
        1.0
        - p_test_raw
    )

    pred[
        "p_not_safe_t0_cal"
    ] = (
        1.0
        - p_test_cal
    )

    pred[
        "predicted_t0_safety"
    ] = pred_label

    eps = 1e-15

    q = np.clip(
        p_test_cal,
        eps,
        1.0 - eps,
    )

    entropy = -(
        q * np.log(q)
        + (1.0 - q)
        * np.log(
            1.0 - q
        )
    )

    pred[
        "predictive_entropy_t0_safety"
    ] = (
        entropy
        / np.log(2.0)
    )

    pred.to_parquet(
        out_dir
        / f"t0_safety_predictions_{regime}.parquet",
        index=False,
    )

    if not args.no_csv_predictions:
        pred.to_csv(
            out_dir
            / f"t0_safety_predictions_{regime}.csv",
            index=False,
        )

    convergence = {
        "regime": regime,
        "iterations_required": iterations_required,
        "max_iter": int(
            args.max_iter
        ),
        "convergence_warning": convergence_warning,
        "converged": converged,
    }

    balance = {
        "regime": regime,

        "train_cases": len(
            train
        ),
        "train_safe": int(
            (
                train[
                    "y_t0_safety_d1"
                ]
                == SAFE_LABEL
            ).sum()
        ),
        "train_unsafe": int(
            (
                train[
                    "y_t0_safety_d1"
                ]
                == UNSAFE_LABEL
            ).sum()
        ),

        "calibration_cases": len(
            cal
        ),
        "calibration_safe": int(
            (
                cal[
                    "y_t0_safety_d1"
                ]
                == SAFE_LABEL
            ).sum()
        ),
        "calibration_unsafe": int(
            (
                cal[
                    "y_t0_safety_d1"
                ]
                == UNSAFE_LABEL
            ).sum()
        ),

        "test_cases": len(
            test
        ),
        "test_safe": int(
            (
                test[
                    "y_t0_safety_d1"
                ]
                == SAFE_LABEL
            ).sum()
        ),
        "test_unsafe": int(
            (
                test[
                    "y_t0_safety_d1"
                ]
                == UNSAFE_LABEL
            ).sum()
        ),
    }

    return (
        overall,
        per_class,
        calibration,
        convergence,
        balance,
    )


def main() -> None:
    args = parse_args()

    base_path = Path(
        args.base_v2
    ).expanduser().resolve()

    split_path = Path(
        args.splits
    ).expanduser().resolve()

    if not base_path.exists():
        raise FileNotFoundError(
            base_path
        )

    if not split_path.exists():
        raise FileNotFoundError(
            split_path
        )

    out_dir = (
        Path(
            args.out
        ).expanduser().resolve()
        if args.out
        else split_path.parent
        / "t0_execution_safety_results"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2f - NATURAL-ONLY t0 EXECUTION-SAFETY ESTIMATOR"
    )
    print(
        "=" * 80
    )
    print(
        f"Base v2 : {base_path}"
    )
    print(
        f"Splits  : {split_path}"
    )
    print(
        f"Output  : {out_dir}"
    )
    print(
        "Target  : D1_CORE_NATURAL resolved subset"
    )
    print(
        "Model   : L-BFGS logistic + isotonic calibration"
    )
    print(
        "No synthetic episodes. No future-outcome features. "
        "No exposure/risk features."
    )

    print(
        "\n1/6 Loading natural source-case benchmark..."
    )

    base = pd.read_parquet(
        base_path
    )

    require_columns(
        base,
        [
            "case_id",
            "goods_receipt_required",
            "gr_available_at_t0",
            "outcome_eventually_cleared",
            *CORE_SIGNALS,
        ],
        "base v2",
    )

    if base[
        "case_id"
    ].duplicated().any():
        raise AssertionError(
            "base v2 must contain exactly one row per natural source case."
        )

    base = base.copy()

    base[
        "source_case_id"
    ] = base[
        "case_id"
    ].astype(str)

    print(
        "\n2/6 Loading frozen source-case split assignments..."
    )

    split_df = pd.read_parquet(
        split_path
    )

    split_map = build_split_map(
        split_df,
        args.regimes,
    )

    # IMPORTANT:
    # rcse_base_v2 contains the full natural v2 universe (210,692 cases),
    # whereas the frozen experimental benchmark/splits contain a smaller
    # source-case universe. The dedicated safety estimator must stay inside
    # the already-frozen experimental source-case universe.
    n_all_v2 = len(base)
    frozen_source_ids = set(
        split_map["source_case_id"].astype(str)
    )

    base_in_frozen = base[
        base["source_case_id"].isin(
            frozen_source_ids
        )
    ].copy()

    n_in_frozen = len(base_in_frozen)
    n_outside_frozen = (
        n_all_v2 - n_in_frozen
    )

    if n_in_frozen == 0:
        raise RuntimeError(
            "No rcse_base_v2 source cases overlap the frozen split universe."
        )

    base_in_frozen[
        "y_t0_safety_d1"
    ] = build_d1_target(
        base_in_frozen
    )

    n_ambiguous = int(
        (
            base_in_frozen[
                "y_t0_safety_d1"
            ]
            == "AMBIGUOUS"
        ).sum()
    )

    resolved = base_in_frozen[
        base_in_frozen[
            "y_t0_safety_d1"
        ]
        != "AMBIGUOUS"
    ].copy()

    print(
        f"   All v2 natural source cases        : {n_all_v2:,}"
    )
    print(
        f"   In frozen experimental universe    : {n_in_frozen:,}"
    )
    print(
        f"   Outside frozen universe (excluded) : {n_outside_frozen:,}"
    )
    print(
        f"   Resolved D1 cases in frozen set    : {len(resolved):,}"
    )
    print(
        f"   Ambiguous excluded                 : {n_ambiguous:,}"
    )
    print(
        f"   SAFE                               : "
        f"{int((resolved['y_t0_safety_d1'] == SAFE_LABEL).sum()):,}"
    )
    print(
        f"   NOT_SAFE                           : "
        f"{int((resolved['y_t0_safety_d1'] == UNSAFE_LABEL).sum()):,}"
    )

    resolved = resolved.merge(
        split_map,
        on="source_case_id",
        how="inner",
        validate="one_to_one",
    )

    for regime in args.regimes:
        col = SPLIT_CONFIG[
            regime
        ][
            "column"
        ]

        missing = int(
            resolved[
                col
            ].isna().sum()
        )

        if missing:
            raise RuntimeError(
                f"{missing:,} in-universe resolved cases are missing {col}."
            )

    print(
        f"   Split mappings joined             : "
        f"{len(resolved):,}/{len(resolved):,}"
    )

    print(
        "\n3/6 Building leakage-safe t0 feature manifest..."
    )

    (
        numeric,
        categorical,
        explicit_exclusions,
    ) = get_feature_sets(
        resolved
    )

    print(
        f"   Numeric features    : {len(numeric)}"
    )
    print(
        f"   Categorical features: {len(categorical)}"
    )

    print(
        "\n4/6 Training/evaluating regimes..."
    )

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
            df=resolved,
            regime=regime,
            numeric=numeric,
            categorical=categorical,
            args=args,
            out_dir=out_dir,
        )

        overall_rows.append(
            overall
        )

        per_class_rows.extend(
            per_class
        )

        calibration_rows.extend(
            calibration
        )

        convergence_rows.append(
            convergence
        )

        balance_rows.append(
            balance
        )

    print(
        "\n5/6 Writing outputs..."
    )

    metrics_df = pd.DataFrame(
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

    balance_df = pd.DataFrame(
        balance_rows
    )

    metrics_df.to_csv(
        out_dir
        / "t0_safety_metrics.csv",
        index=False,
    )

    per_class_df.to_csv(
        out_dir
        / "t0_safety_per_class.csv",
        index=False,
    )

    calibration_df.to_csv(
        out_dir
        / "t0_safety_calibration.csv",
        index=False,
    )

    convergence_df.to_csv(
        out_dir
        / "t0_safety_convergence.csv",
        index=False,
    )

    balance_df.to_csv(
        out_dir
        / "t0_safety_class_balance.csv",
        index=False,
    )

    manifest = {
        "step": "09b.2f",
        "target": "D1_CORE_NATURAL_RESOLVED",
        "positive_class": SAFE_LABEL,
        "negative_class": UNSAFE_LABEL,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "explicit_exclusions": explicit_exclusions,
        "core_target_signals": CORE_SIGNALS,
        "architectural_boundary": (
            "Future outcomes are used only to construct the D1 target. "
            "No future outcome, exposure/risk, intervention, identifier, "
            "or rejected mismatch field is used as a predictor."
        ),
    }

    with open(
        out_dir
        / "t0_safety_feature_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    metadata: dict[
        str,
        Any,
    ] = {
        "step": "09b.2f",
        "base_v2": str(
            base_path
        ),
        "split_source": str(
            split_path
        ),
        "all_v2_natural_source_cases": int(
            n_all_v2
        ),
        "frozen_experimental_source_cases_present_in_v2": int(
            n_in_frozen
        ),
        "v2_cases_outside_frozen_experimental_universe_excluded": int(
            n_outside_frozen
        ),
        "resolved_d1_cases": int(
            len(resolved)
        ),
        "ambiguous_excluded": int(
            n_ambiguous
        ),
        "target_definition": {
            "SAFE_TO_EXECUTE": (
                "required GR evidence complete at t0 AND "
                "eventually clears AND no D1 core adverse/control signal"
            ),
            "NOT_SAFE_TO_EXECUTE": (
                "required GR evidence missing at t0 OR "
                "at least one D1 core adverse/control signal"
            ),
            "AMBIGUOUS": (
                "excluded from fitting/evaluation"
            ),
            "D1_core_signals": CORE_SIGNALS,
        },
        "model": {
            "type": (
                "binary logistic regression"
            ),
            "solver": "lbfgs",
            "penalty": "l2",
            "C": float(
                args.C
            ),
            "max_iter": int(
                args.max_iter
            ),
            "tol": float(
                args.tol
            ),
            "calibration": (
                "isotonic on dedicated calibration partition"
            ),
        },
        "split_policy": (
            "Restrict rcse_base_v2 to the already-frozen experimental "
            "source-case universe, then reuse the frozen source-case split "
            "assignments from rcse_experimental_with_splits.parquet."
        ),
        "synthetic_episodes_used": False,
        "future_outcomes_used_as_features": False,
        "transaction_exposure_used_as_feature": False,
        "transaction_risk_stratum_used_as_feature": False,
        "rejected_mismatch_fields_used_as_features": False,
        "frozen_multiclass_model_modified": False,
        "next_step": (
            "Evaluate discrimination, calibration, and selective "
            "risk-coverage of p_safe_t0 before integrating the dedicated "
            "safety estimator into a revised RCSE policy."
        ),
    }

    with open(
        out_dir
        / "t0_safety_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n6/6 Complete."
    )

    print(
        "\nDedicated t0 execution-safety metrics:"
    )

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
                "roc_auc_calibrated",
                "average_precision_unsafe_calibrated",
                "brier_calibrated",
                "ece_calibrated",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for name in [
        "t0_safety_metrics.csv",
        "t0_safety_per_class.csv",
        "t0_safety_calibration.csv",
        "t0_safety_convergence.csv",
        "t0_safety_class_balance.csv",
        "t0_safety_feature_manifest.json",
        "t0_safety_metadata.json",
    ]:
        print(
            f"  - {out_dir / name}"
        )

    for regime in args.regimes:
        print(
            "  - "
            + str(
                out_dir
                / f"t0_safety_predictions_{regime}.parquet"
            )
        )

        if not args.no_csv_predictions:
            print(
                "  - "
                + str(
                    out_dir
                    / f"t0_safety_predictions_{regime}.csv"
                )
            )

    print(
        "\nNext: inspect whether the dedicated safety score materially "
        "improves EXECUTE-vs-NOT_SAFE discrimination and selective "
        "risk-coverage before modifying RCSE."
    )


if __name__ == "__main__":
    main()
