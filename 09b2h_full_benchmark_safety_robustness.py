#!/usr/bin/env python
"""
09b2h_full_benchmark_safety_robustness.py

RCSE Step 09b.2h
Full-benchmark robustness evaluation of the dedicated t0 execution-safety model.

Purpose
-------
Train exactly the same dedicated natural-only D1 execution-safety estimator
used in Step 09b.2f, then score the FULL frozen experimental test benchmark,
including synthetic uncertainty interventions.

Training remains:
    natural source cases only
    D1_CORE_NATURAL resolved only
    no future outcome features
    no exposure/risk features
    no synthetic intervention metadata
    no rejected mismatch features

Robustness evaluation:
    natural episodes:
        use D1 natural safety target when resolved
    synthetic intervention episodes:
        evaluation-only robustness target = NOT_SAFE_TO_EXECUTE

Important
---------
Synthetic NOT_SAFE labels are used for ROBUSTNESS EVALUATION ONLY.
They are never used for fitting or calibration.

Inputs
------
--base-v2
    rcse_base_v2.parquet

--experimental
    rcse_experimental_with_splits.parquet

Outputs
-------
full_benchmark_safety_metrics.csv
full_benchmark_safety_by_intervention.csv
full_benchmark_safety_thresholds.csv
full_benchmark_safety_score_quantiles.csv
full_benchmark_safety_false_safe_summary.csv
full_benchmark_safety_class_balance.csv
full_benchmark_safety_convergence.csv
full_benchmark_safety_feature_manifest.json
full_benchmark_safety_metadata.json

full_benchmark_safety_predictions_<regime>.parquet

Core question
-------------
Does a model trained only on natural resolved safety cases assign spuriously
high SAFE scores to controlled uncertainty states such as:
    missingness
    contradiction
    severe evidence loss
?
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
    f1_score,
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

THRESHOLDS = [
    0.50, 0.70, 0.80, 0.90, 0.95, 0.97, 0.98, 0.99, 0.995
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2h: full-benchmark robustness of t0 safety estimator."
    )
    p.add_argument(
        "--base-v2",
        required=True,
        help="Path to rcse_base_v2.parquet",
    )
    p.add_argument(
        "--experimental",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <experimental parent>/t0_safety_full_benchmark_robustness",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=5000,
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
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


def normalize_action(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def build_d1_target(df: pd.DataFrame) -> pd.Series:
    gr_required = as_bool(
        df["goods_receipt_required"]
    )

    gr_available = as_bool(
        df["gr_available_at_t0"]
    )

    evidence_complete = (
        (~gr_required)
        | gr_available
    )

    adverse = pd.Series(
        False,
        index=df.index,
    )

    for c in CORE_SIGNALS:
        adverse = (
            adverse
            | as_bool(df[c])
        )

    cleared = as_bool(
        df["outcome_eventually_cleared"]
    )

    target = pd.Series(
        "AMBIGUOUS",
        index=df.index,
        dtype="object",
    )

    target.loc[
        (~evidence_complete)
        | adverse
    ] = UNSAFE_LABEL

    target.loc[
        evidence_complete
        & cleared
        & (~adverse)
    ] = SAFE_LABEL

    return target


def build_split_map(
    exp: pd.DataFrame,
) -> pd.DataFrame:
    required = [
        "source_case_id",
        *[
            cfg["column"]
            for cfg in SPLIT_CONFIG.values()
        ],
    ]

    require_columns(
        exp,
        required,
        "experimental benchmark",
    )

    work = exp[
        required
    ].copy()

    work[
        "source_case_id"
    ] = work[
        "source_case_id"
    ].astype(str)

    for regime, cfg in SPLIT_CONFIG.items():
        col = cfg["column"]

        nuniq = (
            work
            .groupby(
                "source_case_id"
            )[col]
            .nunique(
                dropna=False
            )
        )

        bad = nuniq[
            nuniq > 1
        ]

        if len(bad):
            raise AssertionError(
                f"{regime}: conflicting split assignments "
                f"for {len(bad):,} source cases."
            )

    return (
        work
        .groupby(
            "source_case_id",
            as_index=False,
        )
        .first()
    )


def synthetic_flag(df: pd.DataFrame) -> pd.Series:
    if "synthetic_intervention" in df.columns:
        return as_bool(
            df[
                "synthetic_intervention"
            ]
        )

    if "is_synthetic_intervention" in df.columns:
        return as_bool(
            df[
                "is_synthetic_intervention"
            ]
        )

    if "intervention_type" in df.columns:
        z = (
            df[
                "intervention_type"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        return ~z.isin(
            [
                "",
                "NONE",
                "NATURAL",
                "NO_INTERVENTION",
            ]
        )

    return pd.Series(
        False,
        index=df.index,
    )


def intervention_family(df: pd.DataFrame) -> pd.Series:
    if "intervention_type" not in df.columns:
        return pd.Series(
            "UNKNOWN",
            index=df.index,
            dtype="object",
        )

    z = (
        df[
            "intervention_type"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series(
        "OTHER",
        index=df.index,
        dtype="object",
    )

    out.loc[
        z.isin(
            [
                "",
                "NONE",
                "NATURAL",
                "NO_INTERVENTION",
            ]
        )
    ] = "NATURAL"

    out.loc[
        z.str.contains(
            "MISSING",
            na=False,
        )
        | z.str.contains(
            "HIDE",
            na=False,
        )
    ] = "MISSINGNESS"

    out.loc[
        z.str.contains(
            "CONTRAD",
            na=False,
        )
        | z.str.contains(
            "PERTURB",
            na=False,
        )
    ] = "CONTRADICTION"

    out.loc[
        z.str.contains(
            "SEVERE",
            na=False,
        )
        | z.str.contains(
            "MULTI",
            na=False,
        )
    ] = "SEVERE_EVIDENCE_LOSS"

    return out


def get_feature_sets(
    base: pd.DataFrame,
    exp: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
    list[str],
]:
    """
    Same feature family as Step 09b.2f.
    Only columns present in BOTH natural base and experimental benchmark
    are eligible, ensuring identical representation at train and score time.
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
        c
        for c in numeric_candidates
        if (
            c in base.columns
            and c in exp.columns
        )
    ]

    categorical = [
        c
        for c in categorical_candidates
        if (
            c in base.columns
            and c in exp.columns
        )
    ]

    forbidden = [
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "synthetic_intervention",
        "is_synthetic_intervention",
        "benchmark_arm",
        "intervention_type",
        "intervention_strength",
        "expected_action_reference",
        *CORE_SIGNALS,
        "outcome_eventually_cleared",
    ]

    used = numeric + categorical

    bad = [
        c
        for c in used
        if (
            c.startswith("outcome_")
            or "future_" in c
            or "exposure" in c
            or "risk_stratum" in c
            or "intervention" in c
            or c in {
                "inv_po_abs_rel_diff",
                "inv_gr_abs_rel_diff",
            }
        )
    ]

    if bad:
        raise AssertionError(
            "Forbidden robustness-model features detected:\n  - "
            + "\n  - ".join(
                sorted(bad)
            )
        )

    return (
        numeric,
        categorical,
        forbidden,
    )


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    C: float,
    max_iter: int,
    tol: float,
) -> Pipeline:

    num = Pipeline(
        [
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

    cat = Pipeline(
        [
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
        [
            (
                "num",
                num,
                numeric,
            ),
            (
                "cat",
                cat,
                categorical,
            ),
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
        [
            (
                "preprocessor",
                pre,
            ),
            (
                "model",
                lr,
            ),
        ]
    )


def fit_model(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    args: argparse.Namespace,
) -> tuple[
    Pipeline,
    IsotonicRegression,
    dict[str, Any],
]:

    features = (
        numeric
        + categorical
    )

    y_train = (
        train[
            "y_t0_safety_d1"
        ]
        == SAFE_LABEL
    ).astype(int).to_numpy()

    y_cal = (
        cal[
            "y_t0_safety_d1"
        ]
        == SAFE_LABEL
    ).astype(int).to_numpy()

    pipe = build_pipeline(
        numeric=numeric,
        categorical=categorical,
        C=args.C,
        max_iter=args.max_iter,
        tol=args.tol,
    )

    with warnings.catch_warnings(
        record=True
    ) as caught:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        pipe.fit(
            train[
                features
            ],
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

    iterations = int(
        np.asarray(
            lr.n_iter_,
            dtype=int,
        ).ravel().max()
    )

    converged = bool(
        iterations
        < args.max_iter
        and not convergence_warning
    )

    if not converged:
        raise RuntimeError(
            "Natural-only safety estimator did not converge."
        )

    safe_idx = list(
        lr.classes_
    ).index(1)

    p_cal_raw = pipe.predict_proba(
        cal[
            features
        ]
    )[:, safe_idx]

    iso = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    iso.fit(
        p_cal_raw,
        y_cal,
    )

    return (
        pipe,
        iso,
        {
            "iterations_required": iterations,
            "convergence_warning": convergence_warning,
            "converged": converged,
        },
    )


def score(
    pipe: Pipeline,
    iso: IsotonicRegression,
    df: pd.DataFrame,
    features: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    lr = pipe.named_steps[
        "model"
    ]

    safe_idx = list(
        lr.classes_
    ).index(1)

    raw = pipe.predict_proba(
        df[
            features
        ]
    )[:, safe_idx]

    cal = iso.predict(
        raw
    )

    return (
        raw,
        cal,
    )


def build_robustness_target(
    test: pd.DataFrame,
    base_target_map: pd.Series,
) -> pd.Series:
    """
    Synthetic interventions are evaluation-only NOT_SAFE.

    Natural episodes inherit D1 natural target from source case.
    Ambiguous natural cases remain AMBIGUOUS.
    """

    syn = synthetic_flag(
        test
    )

    target = (
        test[
            "source_case_id"
        ]
        .astype(str)
        .map(
            base_target_map
        )
        .fillna(
            "AMBIGUOUS"
        )
        .astype("object")
    )

    target.loc[
        syn
    ] = UNSAFE_LABEL

    return target


def binary_metrics(
    y_label: pd.Series,
    p_safe: np.ndarray,
) -> dict[str, Any]:

    resolved = y_label.isin(
        [
            SAFE_LABEL,
            UNSAFE_LABEL,
        ]
    )

    y_lab = y_label.loc[
        resolved
    ]

    p = p_safe[
        resolved.to_numpy()
    ]

    if len(
        y_lab
    ) == 0:
        return {}

    y = (
        y_lab
        == SAFE_LABEL
    ).astype(int).to_numpy()

    pred = (
        p >= 0.5
    ).astype(int)

    out = {
        "resolved_n": int(
            len(y)
        ),
        "safe_n": int(
            y.sum()
        ),
        "unsafe_n": int(
            (1 - y).sum()
        ),
        "unsafe_prevalence": float(
            (1 - y).mean()
        ),
        "accuracy": accuracy_score(
            y,
            pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y,
            pred,
        ),
        "macro_f1": f1_score(
            y,
            pred,
            average="macro",
            zero_division=0,
        ),
        "brier": brier_score_loss(
            y,
            p,
        ),
    }

    if len(
        np.unique(y)
    ) >= 2:
        out[
            "roc_auc"
        ] = roc_auc_score(
            y,
            p,
        )

        out[
            "average_precision_unsafe"
        ] = average_precision_score(
            1 - y,
            1 - p,
        )

    else:
        out[
            "roc_auc"
        ] = np.nan

        out[
            "average_precision_unsafe"
        ] = np.nan

    return out


def group_rows(
    df: pd.DataFrame,
    score_col: str,
    score_type: str,
    regime: str,
) -> list[
    dict[str, Any]
]:

    rows = []

    for family, g in df.groupby(
        "_intervention_family",
        dropna=False,
    ):

        target = g[
            "_robustness_target"
        ]

        unsafe = (
            target
            == UNSAFE_LABEL
        )

        safe = (
            target
            == SAFE_LABEL
        )

        score = pd.to_numeric(
            g[
                score_col
            ],
            errors="coerce",
        )

        rows.append(
            {
                "regime": regime,
                "score_type": score_type,
                "intervention_family": family,
                "episodes": int(
                    len(g)
                ),
                "resolved_episodes": int(
                    target.isin(
                        [
                            SAFE_LABEL,
                            UNSAFE_LABEL,
                        ]
                    ).sum()
                ),
                "safe_episodes": int(
                    safe.sum()
                ),
                "unsafe_episodes": int(
                    unsafe.sum()
                ),
                "mean_p_safe": float(
                    score.mean()
                ),
                "median_p_safe": float(
                    score.median()
                ),
                "p90_p_safe": float(
                    score.quantile(
                        0.90
                    )
                ),
                "p95_p_safe": float(
                    score.quantile(
                        0.95
                    )
                ),
                "p99_p_safe": float(
                    score.quantile(
                        0.99
                    )
                ),
                "fraction_p_safe_ge_0_90": float(
                    (
                        score
                        >= 0.90
                    ).mean()
                ),
                "fraction_p_safe_ge_0_95": float(
                    (
                        score
                        >= 0.95
                    ).mean()
                ),
                "fraction_p_safe_ge_0_99": float(
                    (
                        score
                        >= 0.99
                    ).mean()
                ),
            }
        )

    return rows


def threshold_rows(
    df: pd.DataFrame,
    score_col: str,
    score_type: str,
    regime: str,
) -> list[
    dict[str, Any]
]:

    rows = []

    y = df[
        "_robustness_target"
    ]

    resolved = y.isin(
        [
            SAFE_LABEL,
            UNSAFE_LABEL,
        ]
    )

    d = df.loc[
        resolved
    ].copy()

    y = d[
        "_robustness_target"
    ]

    for tau in THRESHOLDS:
        selected = (
            d[
                score_col
            ]
            >= tau
        )

        n_sel = int(
            selected.sum()
        )

        selected_unsafe = int(
            (
                selected
                & (
                    y
                    == UNSAFE_LABEL
                )
            ).sum()
        )

        syn_selected = int(
            (
                selected
                & d[
                    "_is_synthetic"
                ]
            ).sum()
        )

        rows.append(
            {
                "regime": regime,
                "score_type": score_type,
                "threshold": tau,
                "resolved_test_n": int(
                    len(d)
                ),
                "selected_n": n_sel,
                "coverage": (
                    n_sel
                    / len(d)
                    if len(d)
                    else np.nan
                ),
                "unsafe_selected": selected_unsafe,
                "unsafe_execution_rate": (
                    selected_unsafe
                    / n_sel
                    if n_sel
                    else np.nan
                ),
                "synthetic_selected": syn_selected,
                "synthetic_fraction_of_selected": (
                    syn_selected
                    / n_sel
                    if n_sel
                    else np.nan
                ),
            }
        )

    return rows


def quantile_rows(
    df: pd.DataFrame,
    score_col: str,
    score_type: str,
    regime: str,
) -> list[
    dict[str, Any]
]:

    rows = []

    grouping = [
        (
            "robustness_target",
            "_robustness_target",
        ),
        (
            "intervention_family",
            "_intervention_family",
        ),
    ]

    for grouping_name, col in grouping:
        for value, g in df.groupby(
            col,
            dropna=False,
        ):
            x = pd.to_numeric(
                g[
                    score_col
                ],
                errors="coerce",
            ).dropna()

            rows.append(
                {
                    "regime": regime,
                    "score_type": score_type,
                    "grouping": grouping_name,
                    "group_value": str(
                        value
                    ),
                    "n": int(
                        len(x)
                    ),
                    "mean": float(
                        x.mean()
                    )
                    if len(x)
                    else np.nan,
                    "p10": float(
                        x.quantile(
                            0.10
                        )
                    )
                    if len(x)
                    else np.nan,
                    "p25": float(
                        x.quantile(
                            0.25
                        )
                    )
                    if len(x)
                    else np.nan,
                    "p50": float(
                        x.quantile(
                            0.50
                        )
                    )
                    if len(x)
                    else np.nan,
                    "p75": float(
                        x.quantile(
                            0.75
                        )
                    )
                    if len(x)
                    else np.nan,
                    "p90": float(
                        x.quantile(
                            0.90
                        )
                    )
                    if len(x)
                    else np.nan,
                    "p95": float(
                        x.quantile(
                            0.95
                        )
                    )
                    if len(x)
                    else np.nan,
                    "p99": float(
                        x.quantile(
                            0.99
                        )
                    )
                    if len(x)
                    else np.nan,
                    "max": float(
                        x.max()
                    )
                    if len(x)
                    else np.nan,
                }
            )

    return rows


def main() -> None:

    args = parse_args()

    base_path = Path(
        args.base_v2
    ).expanduser().resolve()

    exp_path = Path(
        args.experimental
    ).expanduser().resolve()

    if not base_path.exists():
        raise FileNotFoundError(
            base_path
        )

    if not exp_path.exists():
        raise FileNotFoundError(
            exp_path
        )

    out_dir = (
        Path(
            args.out
        ).expanduser().resolve()
        if args.out
        else exp_path.parent
        / "t0_safety_full_benchmark_robustness"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2h - FULL-BENCHMARK SAFETY ROBUSTNESS"
    )
    print(
        "=" * 80
    )
    print(
        f"Base v2      : {base_path}"
    )
    print(
        f"Experimental : {exp_path}"
    )
    print(
        f"Output       : {out_dir}"
    )
    print(
        "Train on natural D1-resolved only; score full experimental test benchmark."
    )
    print(
        "Synthetic NOT_SAFE labels are evaluation-only."
    )

    base = pd.read_parquet(
        base_path
    )

    exp = pd.read_parquet(
        exp_path
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

    require_columns(
        exp,
        [
            "benchmark_episode_id",
            "source_case_id",
            "expected_action_reference",
            *[
                cfg[
                    "column"
                ]
                for cfg in SPLIT_CONFIG.values()
            ],
        ],
        "experimental benchmark",
    )

    base = base.copy()
    exp = exp.copy()

    base[
        "source_case_id"
    ] = base[
        "case_id"
    ].astype(str)

    exp[
        "source_case_id"
    ] = exp[
        "source_case_id"
    ].astype(str)

    split_map = build_split_map(
        exp
    )

    frozen_ids = set(
        split_map[
            "source_case_id"
        ]
    )

    base = base[
        base[
            "source_case_id"
        ].isin(
            frozen_ids
        )
    ].copy()

    base[
        "y_t0_safety_d1"
    ] = build_d1_target(
        base
    )

    base_target_map = (
        base[
            [
                "source_case_id",
                "y_t0_safety_d1",
            ]
        ]
        .drop_duplicates(
            "source_case_id"
        )
        .set_index(
            "source_case_id"
        )[
            "y_t0_safety_d1"
        ]
    )

    resolved_base = base[
        base[
            "y_t0_safety_d1"
        ]
        != "AMBIGUOUS"
    ].copy()

    resolved_base = resolved_base.merge(
        split_map,
        on="source_case_id",
        how="inner",
        validate="one_to_one",
    )

    numeric, categorical, forbidden = get_feature_sets(
        base=resolved_base,
        exp=exp,
    )

    features = (
        numeric
        + categorical
    )

    print(
        f"\nFrozen natural source cases: {len(base):,}"
    )
    print(
        f"Resolved natural training universe: {len(resolved_base):,}"
    )
    print(
        f"Features: {len(numeric)} numeric + {len(categorical)} categorical"
    )

    metrics_rows = []
    family_rows = []
    threshold_out = []
    quantile_out = []
    convergence_rows = []
    balance_rows = []

    for regime, cfg in SPLIT_CONFIG.items():

        split_col = cfg[
            "column"
        ]

        train = resolved_base[
            resolved_base[
                split_col
            ]
            == cfg[
                "train"
            ]
        ].copy()

        cal = resolved_base[
            resolved_base[
                split_col
            ]
            == cfg[
                "calibration"
            ]
        ].copy()

        test = exp[
            exp[
                split_col
            ]
            == cfg[
                "test"
            ]
        ].copy()

        print(
            f"\n[{regime}] train={len(train):,} cal={len(cal):,} full-test={len(test):,}"
        )

        pipe, iso, convergence = fit_model(
            train=train,
            cal=cal,
            numeric=numeric,
            categorical=categorical,
            args=args,
        )

        print(
            f"   iterations={convergence['iterations_required']:,}/"
            f"{args.max_iter:,} converged={convergence['converged']}"
        )

        raw, cal_score = score(
            pipe=pipe,
            iso=iso,
            df=test,
            features=features,
        )

        test = test.copy()

        test[
            "_is_synthetic"
        ] = synthetic_flag(
            test
        )

        test[
            "_intervention_family"
        ] = intervention_family(
            test
        )

        test[
            "_robustness_target"
        ] = build_robustness_target(
            test=test,
            base_target_map=base_target_map,
        )

        test[
            "p_safe_t0_raw"
        ] = raw

        test[
            "p_safe_t0_cal"
        ] = cal_score

        test[
            "p_not_safe_t0_raw"
        ] = 1.0 - raw

        test[
            "p_not_safe_t0_cal"
        ] = 1.0 - cal_score

        # Main metrics for raw and calibrated scores.
        for score_type, score_col in [
            (
                "RAW",
                "p_safe_t0_raw",
            ),
            (
                "CALIBRATED",
                "p_safe_t0_cal",
            ),
        ]:

            m = binary_metrics(
                y_label=test[
                    "_robustness_target"
                ],
                p_safe=test[
                    score_col
                ].to_numpy(),
            )

            metrics_rows.append(
                {
                    "regime": regime,
                    "score_type": score_type,
                    "full_test_episodes": int(
                        len(test)
                    ),
                    "synthetic_test_episodes": int(
                        test[
                            "_is_synthetic"
                        ].sum()
                    ),
                    "natural_test_episodes": int(
                        (
                            ~test[
                                "_is_synthetic"
                            ]
                        ).sum()
                    ),
                    **m,
                }
            )

            family_rows.extend(
                group_rows(
                    df=test,
                    score_col=score_col,
                    score_type=score_type,
                    regime=regime,
                )
            )

            threshold_out.extend(
                threshold_rows(
                    df=test,
                    score_col=score_col,
                    score_type=score_type,
                    regime=regime,
                )
            )

            quantile_out.extend(
                quantile_rows(
                    df=test,
                    score_col=score_col,
                    score_type=score_type,
                    regime=regime,
                )
            )

        convergence_rows.append(
            {
                "regime": regime,
                **convergence,
            }
        )

        balance_rows.append(
            {
                "regime": regime,
                "train_n": int(
                    len(train)
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
                "calibration_n": int(
                    len(cal)
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
                "full_test_n": int(
                    len(test)
                ),
                "full_test_synthetic": int(
                    test[
                        "_is_synthetic"
                    ].sum()
                ),
                "full_test_natural": int(
                    (
                        ~test[
                            "_is_synthetic"
                        ]
                    ).sum()
                ),
                "full_test_robustness_resolved": int(
                    test[
                        "_robustness_target"
                    ].isin(
                        [
                            SAFE_LABEL,
                            UNSAFE_LABEL,
                        ]
                    ).sum()
                ),
            }
        )

        export_cols = [
            c
            for c in [
                "benchmark_episode_id",
                "source_case_id",
                "benchmark_arm",
                "intervention_type",
                "intervention_strength",
                "synthetic_intervention",
                "transaction_exposure_eur",
                "transaction_risk_stratum",
                "expected_action_reference",
                split_col,
                "_is_synthetic",
                "_intervention_family",
                "_robustness_target",
                "p_safe_t0_raw",
                "p_safe_t0_cal",
                "p_not_safe_t0_raw",
                "p_not_safe_t0_cal",
            ]
            if c in test.columns
        ]

        test[
            export_cols
        ].to_parquet(
            out_dir
            / f"full_benchmark_safety_predictions_{regime}.parquet",
            index=False,
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    family_df = pd.DataFrame(
        family_rows
    )

    threshold_df = pd.DataFrame(
        threshold_out
    )

    quantile_df = pd.DataFrame(
        quantile_out
    )

    convergence_df = pd.DataFrame(
        convergence_rows
    )

    balance_df = pd.DataFrame(
        balance_rows
    )

    metrics_df.to_csv(
        out_dir
        / "full_benchmark_safety_metrics.csv",
        index=False,
    )

    family_df.to_csv(
        out_dir
        / "full_benchmark_safety_by_intervention.csv",
        index=False,
    )

    threshold_df.to_csv(
        out_dir
        / "full_benchmark_safety_thresholds.csv",
        index=False,
    )

    quantile_df.to_csv(
        out_dir
        / "full_benchmark_safety_score_quantiles.csv",
        index=False,
    )

    convergence_df.to_csv(
        out_dir
        / "full_benchmark_safety_convergence.csv",
        index=False,
    )

    balance_df.to_csv(
        out_dir
        / "full_benchmark_safety_class_balance.csv",
        index=False,
    )

    # False-safe summary at high thresholds.
    fs = threshold_df[
        threshold_df[
            "threshold"
        ].isin(
            [
                0.90,
                0.95,
                0.99,
            ]
        )
    ].copy()

    fs.to_csv(
        out_dir
        / "full_benchmark_safety_false_safe_summary.csv",
        index=False,
    )

    manifest = {
        "step": "09b.2h",
        "training_target": "D1_CORE_NATURAL_RESOLVED",
        "training_scope": "natural source cases only",
        "synthetic_labels_used_for_training": False,
        "synthetic_labels_used_for_calibration": False,
        "synthetic_labels_used_for_evaluation_only": True,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "forbidden_feature_families": forbidden,
        "core_target_signals": CORE_SIGNALS,
    }

    with open(
        out_dir
        / "full_benchmark_safety_feature_manifest.json",
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
        "step": "09b.2h",
        "base_v2": str(
            base_path
        ),
        "experimental_benchmark": str(
            exp_path
        ),
        "training": {
            "scope": "natural D1-resolved source cases only",
            "model": "L-BFGS logistic regression + isotonic calibration",
            "C": float(
                args.C
            ),
            "max_iter": int(
                args.max_iter
            ),
            "tol": float(
                args.tol
            ),
        },
        "robustness_target": {
            "natural": (
                "D1 natural safety target when resolved; ambiguous remains ambiguous"
            ),
            "synthetic": (
                "NOT_SAFE_TO_EXECUTE for robustness evaluation only"
            ),
        },
        "guardrails": [
            "No synthetic episode is used in training.",
            "No synthetic episode is used in calibration.",
            "No future outcome field is used as a predictor.",
            "No exposure/risk field is used as a predictor.",
            "No intervention metadata is used as a predictor.",
            "No rejected mismatch feature is used as a predictor.",
        ],
        "decision_rule": (
            "If high SAFE scores remain rare on missingness/contradiction/"
            "severe-evidence-loss episodes, the safety estimator is robust "
            "enough to consider for RCSE v2. If synthetic uncertainty receives "
            "high SAFE scores frequently, add a conservative evidence-consistency "
            "gate rather than silently treating the safety probability as universal."
        ),
    }

    with open(
        out_dir
        / "full_benchmark_safety_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 09b.2h COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nFull-benchmark robustness metrics:"
    )

    print(
        metrics_df[
            [
                "regime",
                "score_type",
                "full_test_episodes",
                "synthetic_test_episodes",
                "resolved_n",
                "unsafe_prevalence",
                "balanced_accuracy",
                "macro_f1",
                "roc_auc",
                "average_precision_unsafe",
                "brier",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nHigh-safety-score rates by intervention family (calibrated):"
    )

    display = family_df[
        family_df[
            "score_type"
        ]
        == "CALIBRATED"
    ][
        [
            "regime",
            "intervention_family",
            "episodes",
            "mean_p_safe",
            "fraction_p_safe_ge_0_90",
            "fraction_p_safe_ge_0_95",
            "fraction_p_safe_ge_0_99",
        ]
    ]

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for p in sorted(
        out_dir.iterdir()
    ):
        if p.is_file():
            print(
                f"  - {p}"
            )

    print(
        "\nNext: decide whether the dedicated safety belief is robust enough "
        "for a revised RCSE policy, or whether an explicit evidence-consistency "
        "gate is required before autonomous execution."
    )


if __name__ == "__main__":
    main()
