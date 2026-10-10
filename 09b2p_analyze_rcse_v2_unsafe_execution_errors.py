#!/usr/bin/env python
"""
09b2p_analyze_rcse_v2_unsafe_execution_errors.py

RCSE Step 09b.2p
Unsafe-execution error decomposition for the frozen RCSE v2 reference policy.

Purpose
-------
Explain WHY the frozen RCSE v2 reference operating point still has a material
unsafe-execution rate.

This is DIAGNOSTIC ONLY.

NO MODEL TRAINING.
NO RECALIBRATION.
NO GATE CHANGES.
NO POLICY RETUNING.
NO SELECTION FROM THE TEST FRONTIER.

Primary focus
-------------
Frozen policy:
    RCSE_V2_FULL

Primary consequence:
    LOG_TRAIN_P90

Frozen reference parameters are read from:
    09b2l_rcse_v2_policy_spec.json

The analysis decomposes autonomous execution errors by:
- regime
- NATURAL / MISSINGNESS / CONTRADICTION / SEVERE_EVIDENCE_LOSS
- intervention type / strength
- Evidence Gate v2 state
- t0 safety-score bands
- transaction-exposure quantiles
- reference action
- source safety target
- high-confidence false-safe status

It also reconstructs the POST-GATHER continuation decision for top-level GATHER
episodes so we can inspect:
- post-GATHER EXECUTE / ESCALATE / ABSTAIN
- post-GATHER safety outcomes
- whether GATHER converts t0 uncertainty into a safe continuation

Important accounting audit
--------------------------
Step 09b.2n records top-level GATHER as realized loss = c_G only.

This script does NOT overwrite Step 09b.2n results. It additionally computes a
diagnostic end-to-end GATHER realized loss:

    c_G + realized post-GATHER continuation loss

so the difference is visible explicitly.

Inputs
------
--dataset-dir
    Step 09b.2m policy-ready datasets.

--results-dir
    Step 09b.2n rcse_v2_policy_results directory.

--spec
    Frozen 09b2l_rcse_v2_policy_spec.json

Outputs
-------
rcse_v2p_direct_execution_summary.csv
rcse_v2p_unsafe_by_intervention_family.csv
rcse_v2p_unsafe_by_intervention_type_strength.csv
rcse_v2p_unsafe_by_gate_state.csv
rcse_v2p_unsafe_by_safety_band.csv
rcse_v2p_unsafe_by_exposure_quantile.csv
rcse_v2p_unsafe_by_reference_action.csv
rcse_v2p_high_confidence_false_safe.csv
rcse_v2p_gather_continuation_summary.csv
rcse_v2p_gather_continuation_by_intervention.csv
rcse_v2p_gather_safety_conversion.csv
rcse_v2p_loss_accounting_audit.csv
rcse_v2p_episode_diagnostics_<regime>.parquet
rcse_v2p_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = ("grouped_iid", "temporal", "ood_exposure")

DATASET_FILES = {
    "grouped_iid": "rcse_v2_policy_dataset_grouped_iid.parquet",
    "temporal": "rcse_v2_policy_dataset_temporal.parquet",
    "ood_exposure": "rcse_v2_policy_dataset_ood_exposure.parquet",
}

DECISION_FILES = {
    "grouped_iid": "rcse_v2_policy_episode_decisions_grouped_iid.parquet",
    "temporal": "rcse_v2_policy_episode_decisions_temporal.parquet",
    "ood_exposure": "rcse_v2_policy_episode_decisions_ood_exposure.parquet",
}

PRIMARY_POLICY = "RCSE_V2_FULL"
PRIMARY_CONSEQUENCE = "LOG_TRAIN_P90"

SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"

SAFETY_BANDS = [
    (-np.inf, 0.50, "<0.50"),
    (0.50, 0.80, "0.50-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 0.95, "0.90-0.95"),
    (0.95, 0.97, "0.95-0.97"),
    (0.97, 0.99, "0.97-0.99"),
    (0.99, np.inf, ">=0.99"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2p: RCSE v2 unsafe-execution error decomposition."
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Step 09b.2m rcse_v2_policy_dataset directory.",
    )
    p.add_argument(
        "--results-dir",
        required=True,
        help="Step 09b.2n rcse_v2_policy_results directory.",
    )
    p.add_argument(
        "--spec",
        required=True,
        help="Frozen 09b2l_rcse_v2_policy_spec.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <results-dir>/unsafe_error_decomposition",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_action(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_safety_label(s: pd.Series) -> pd.Series:
    z = (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series("AMBIGUOUS", index=s.index, dtype="object")

    out.loc[
        z.isin(
            [
                SAFE_LABEL,
                "SAFE",
                "TRUE",
                "1",
            ]
        )
    ] = SAFE_LABEL

    out.loc[
        z.isin(
            [
                UNSAFE_LABEL,
                "UNSAFE",
                "NOT_SAFE",
                "FALSE",
                "0",
            ]
        )
    ] = UNSAFE_LABEL

    return out


def validate_spec(spec: dict[str, Any]) -> dict[str, float]:
    if spec.get("status") != "FROZEN_BEFORE_RCSE_V2_SIMULATION":
        raise RuntimeError(
            "Spec is not frozen RCSE v2 specification."
        )

    costs = spec["costs"]["reference"]
    risk = spec["risk_and_selectivity"]["reference"]

    return {
        "c_G": float(costs["c_G"]),
        "c_H": float(costs["c_H"]),
        "c_A": float(costs["c_A"]),
        "rho_H": float(costs["rho_H"]),
        "tau_safe_t0": float(risk["tau_safe_t0"]),
        "beta0": float(risk["beta0"]),
        "tau_safe_t1": float(risk["tau_safe_t1"]),
        "beta1": float(risk["beta1"]),
    }


def consequence_log_p90(df: pd.DataFrame) -> pd.Series:
    require_columns(
        df,
        [
            "transaction_exposure_eur",
            "train_exposure_p90_eur",
        ],
        "policy dataset",
    )

    ref_values = (
        pd.to_numeric(
            df["train_exposure_p90_eur"],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    if len(ref_values) != 1:
        raise RuntimeError(
            f"Expected exactly one train_exposure_p90_eur value; got {len(ref_values)}"
        )

    ref = float(ref_values[0])

    if ref <= 0:
        raise RuntimeError(
            f"Invalid train P90 exposure reference: {ref}"
        )

    exposure = (
        pd.to_numeric(
            df["transaction_exposure_eur"],
            errors="coerce",
        )
        .abs()
        .fillna(0.0)
    )

    return np.log1p(exposure) / np.log1p(ref)


def derive_t0_safety_target(df: pd.DataFrame) -> pd.Series:
    if "t0_safety_robustness_target" in df.columns:
        target = normalize_safety_label(
            df["t0_safety_robustness_target"]
        )

        unresolved = target == "AMBIGUOUS"

        if unresolved.any():
            ref = normalize_action(
                df.loc[
                    unresolved,
                    "expected_action_reference",
                ]
            )

            target.loc[
                unresolved
            ] = np.where(
                ref == "EXECUTE",
                SAFE_LABEL,
                UNSAFE_LABEL,
            )

        return target

    return pd.Series(
        np.where(
            normalize_action(
                df["expected_action_reference"]
            )
            == "EXECUTE",
            SAFE_LABEL,
            UNSAFE_LABEL,
        ),
        index=df.index,
    )


def safety_band(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(
        s,
        errors="coerce",
    )

    out = pd.Series(
        "MISSING",
        index=s.index,
        dtype="object",
    )

    for lo, hi, label in SAFETY_BANDS:
        if np.isneginf(lo):
            mask = x < hi
        elif np.isposinf(hi):
            mask = x >= lo
        else:
            mask = (x >= lo) & (x < hi)

        out.loc[
            mask
        ] = label

    return out


def exposure_quantile_band(df: pd.DataFrame) -> pd.Series:
    exposure = pd.to_numeric(
        df["transaction_exposure_eur"],
        errors="coerce",
    )

    valid = exposure.notna()

    out = pd.Series(
        "MISSING",
        index=df.index,
        dtype="object",
    )

    if valid.sum() == 0:
        return out

    # rank-based qcut avoids duplicate-edge failures.
    ranks = exposure.loc[
        valid
    ].rank(
        method="first"
    )

    labels = [
        "Q1_LOWEST",
        "Q2",
        "Q3",
        "Q4_HIGHEST",
    ]

    out.loc[
        valid
    ] = pd.qcut(
        ranks,
        q=4,
        labels=labels,
    ).astype(str)

    return out


def reconstruct_post_gather_continuation(
    df: pd.DataFrame,
    consequence: pd.Series,
    params: dict[str, float],
) -> tuple[pd.Series, pd.Series]:
    """
    Reconstruct continuation action exactly from the frozen v2 rule:
      min(EXECUTE, ESCALATE, ABSTAIN)
    after applying post-GATHER EXECUTE feasibility.
    """
    p1 = pd.to_numeric(
        df["p_safe_t1_for_primary_rcse"],
        errors="coerce",
    )

    l_H = (
        params["c_H"]
        + params["rho_H"] * consequence
    )

    l_A = pd.Series(
        params["c_A"],
        index=df.index,
        dtype=float,
    )

    l_E = (
        (1.0 - p1)
        * consequence
    )

    exec_feasible = (
        p1.notna()
        & (
            p1
            >= params["tau_safe_t1"]
        )
        & (
            l_E
            <= params["beta1"]
        )
    )

    l_E = l_E.where(
        exec_feasible,
        np.inf,
    )

    losses = pd.concat(
        [
            l_E.rename("EXECUTE"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    )

    action = losses.idxmin(
        axis=1
    )

    selected_expected_loss = losses.min(
        axis=1
    )

    return action, selected_expected_loss


def realized_continuation_loss(
    continuation_action: pd.Series,
    t1_target: pd.Series,
    consequence: pd.Series,
    params: dict[str, float],
) -> pd.Series:
    out = pd.Series(
        np.nan,
        index=continuation_action.index,
        dtype=float,
    )

    exec_mask = continuation_action == "EXECUTE"
    esc_mask = continuation_action == "ESCALATE"
    abstain_mask = continuation_action == "ABSTAIN"

    unsafe_t1 = (
        t1_target
        == UNSAFE_LABEL
    )

    # Ambiguous t1 executed continuations remain NaN because realized safety
    # is not supported by the available target.
    known_exec = (
        exec_mask
        & t1_target.isin(
            [
                SAFE_LABEL,
                UNSAFE_LABEL,
            ]
        )
    )

    out.loc[
        known_exec
    ] = (
        unsafe_t1.loc[
            known_exec
        ].astype(float)
        * consequence.loc[
            known_exec
        ]
    )

    out.loc[
        esc_mask
    ] = (
        params["c_H"]
        + params["rho_H"]
        * consequence.loc[
            esc_mask
        ]
    )

    out.loc[
        abstain_mask
    ] = params[
        "c_A"
    ]

    return out


def direct_execution_metrics(
    df: pd.DataFrame,
) -> dict[str, Any]:
    execute = (
        df["rcse_v2_full_action"]
        == "EXECUTE"
    )

    unsafe = (
        df["t0_safety_target_eval"]
        == UNSAFE_LABEL
    )

    n_exec = int(
        execute.sum()
    )

    unsafe_exec = int(
        (
            execute
            & unsafe
        ).sum()
    )

    exposure = pd.to_numeric(
        df["transaction_exposure_eur"],
        errors="coerce",
    ).abs().fillna(0.0)

    cons = pd.to_numeric(
        df["consequence_log_train_p90"],
        errors="coerce",
    ).fillna(0.0)

    selected_exposure = float(
        exposure.loc[
            execute
        ].sum()
    )

    unsafe_exposure = float(
        exposure.loc[
            execute & unsafe
        ].sum()
    )

    selected_cons = float(
        cons.loc[
            execute
        ].sum()
    )

    unsafe_cons = float(
        cons.loc[
            execute & unsafe
        ].sum()
    )

    return {
        "episodes": int(
            len(df)
        ),
        "execute_count": n_exec,
        "execution_coverage": (
            n_exec / len(df)
            if len(df)
            else np.nan
        ),
        "unsafe_execute_count": unsafe_exec,
        "unsafe_execution_rate": (
            unsafe_exec / n_exec
            if n_exec
            else np.nan
        ),
        "raw_exposure_weighted_unsafe_rate": (
            unsafe_exposure / selected_exposure
            if selected_exposure > 0
            else np.nan
        ),
        "consequence_weighted_unsafe_rate": (
            unsafe_cons / selected_cons
            if selected_cons > 0
            else np.nan
        ),
    }


def group_direct_errors(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    rows = []

    for keys, g in df.groupby(
        group_cols,
        dropna=False,
        observed=True,
    ):
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        row = {
            c: v
            for c, v in zip(
                group_cols,
                keys,
            )
        }

        row.update(
            direct_execution_metrics(
                g
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def high_confidence_false_safe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    thresholds = [
        0.95,
        0.97,
        0.99,
    ]

    p = pd.to_numeric(
        df["p_safe_t0_cal"],
        errors="coerce",
    )

    execute = (
        df["rcse_v2_full_action"]
        == "EXECUTE"
    )

    unsafe = (
        df["t0_safety_target_eval"]
        == UNSAFE_LABEL
    )

    for tau in thresholds:
        high = p >= tau

        for regime, g in df.groupby(
            "evaluation_regime",
            observed=True,
        ):
            idx = g.index

            high_g = high.loc[
                idx
            ]

            exec_g = execute.loc[
                idx
            ]

            unsafe_g = unsafe.loc[
                idx
            ]

            high_unsafe = (
                high_g
                & unsafe_g
            )

            high_unsafe_exec = (
                high_g
                & unsafe_g
                & exec_g
            )

            rows.append(
                {
                    "regime": regime,
                    "p_safe_threshold": tau,
                    "episodes": int(
                        len(g)
                    ),
                    "high_confidence_count": int(
                        high_g.sum()
                    ),
                    "high_confidence_unsafe_count": int(
                        high_unsafe.sum()
                    ),
                    "high_confidence_unsafe_rate_among_high_confidence": (
                        float(
                            high_unsafe.sum()
                            / high_g.sum()
                        )
                        if high_g.sum()
                        else np.nan
                    ),
                    "high_confidence_unsafe_execute_count": int(
                        high_unsafe_exec.sum()
                    ),
                    "fraction_high_confidence_unsafe_that_execute": (
                        float(
                            high_unsafe_exec.sum()
                            / high_unsafe.sum()
                        )
                        if high_unsafe.sum()
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def gather_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    gather = df[
        df[
            "rcse_v2_full_action"
        ]
        == "GATHER"
    ].copy()

    if gather.empty:
        return pd.DataFrame()

    for regime, g in gather.groupby(
        "evaluation_regime",
        observed=True,
    ):
        t1_known = g[
            "t1_safety_target_eval"
        ].isin(
            [
                SAFE_LABEL,
                UNSAFE_LABEL,
            ]
        )

        post_exec = (
            g[
                "post_gather_continuation_action"
            ]
            == "EXECUTE"
        )

        post_exec_known = (
            post_exec
            & t1_known
        )

        post_exec_unsafe = (
            post_exec_known
            & (
                g[
                    "t1_safety_target_eval"
                ]
                == UNSAFE_LABEL
            )
        )

        rows.append(
            {
                "regime": regime,
                "gather_count": int(
                    len(g)
                ),
                "post_gather_execute_count": int(
                    post_exec.sum()
                ),
                "post_gather_escalate_count": int(
                    (
                        g[
                            "post_gather_continuation_action"
                        ]
                        == "ESCALATE"
                    ).sum()
                ),
                "post_gather_abstain_count": int(
                    (
                        g[
                            "post_gather_continuation_action"
                        ]
                        == "ABSTAIN"
                    ).sum()
                ),
                "post_gather_execute_rate": float(
                    post_exec.mean()
                ),
                "post_gather_execute_known_safety_count": int(
                    post_exec_known.sum()
                ),
                "post_gather_unsafe_execute_count": int(
                    post_exec_unsafe.sum()
                ),
                "post_gather_unsafe_execution_rate_known": (
                    float(
                        post_exec_unsafe.sum()
                        / post_exec_known.sum()
                    )
                    if post_exec_known.sum()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def gather_conversion_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare t0 evaluation safety state to t1 safety state among GATHER cases.
    """
    gather = df[
        df[
            "rcse_v2_full_action"
        ]
        == "GATHER"
    ].copy()

    if gather.empty:
        return pd.DataFrame()

    rows = []

    for (
        regime,
        t0,
        t1,
    ), g in gather.groupby(
        [
            "evaluation_regime",
            "t0_safety_target_eval",
            "t1_safety_target_eval",
        ],
        dropna=False,
        observed=True,
    ):
        rows.append(
            {
                "regime": regime,
                "t0_safety_target": t0,
                "t1_safety_target": t1,
                "episodes": int(
                    len(g)
                ),
                "pct_within_regime_gather": float(
                    len(g)
                    / len(
                        gather[
                            gather[
                                "evaluation_regime"
                            ]
                            == regime
                        ]
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def loss_accounting_audit(
    df: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    """
    Compare Step 09b.2n top-level GATHER accounting (c_G only)
    with diagnostic end-to-end GATHER + continuation realized loss.
    """
    rows = []

    for regime, g in df.groupby(
        "evaluation_regime",
        observed=True,
    ):
        action = g[
            "rcse_v2_full_action"
        ]

        t0_unsafe = (
            g[
                "t0_safety_target_eval"
            ]
            == UNSAFE_LABEL
        )

        consequence = g[
            "consequence_log_train_p90"
        ]

        top_level = pd.Series(
            np.nan,
            index=g.index,
            dtype=float,
        )

        top_level.loc[
            action
            == "EXECUTE"
        ] = (
            t0_unsafe.loc[
                action
                == "EXECUTE"
            ].astype(float)
            * consequence.loc[
                action
                == "EXECUTE"
            ]
        )

        top_level.loc[
            action
            == "GATHER"
        ] = params[
            "c_G"
        ]

        top_level.loc[
            action
            == "ESCALATE"
        ] = (
            params[
                "c_H"
            ]
            + params[
                "rho_H"
            ]
            * consequence.loc[
                action
                == "ESCALATE"
            ]
        )

        top_level.loc[
            action
            == "ABSTAIN"
        ] = params[
            "c_A"
        ]

        end_to_end = top_level.copy()

        gather_mask = (
            action
            == "GATHER"
        )

        end_to_end.loc[
            gather_mask
        ] = (
            params[
                "c_G"
            ]
            + g.loc[
                gather_mask,
                "post_gather_realized_continuation_loss",
            ]
        )

        rows.append(
            {
                "regime": regime,
                "episodes": int(
                    len(g)
                ),
                "gather_count": int(
                    gather_mask.sum()
                ),
                "mean_top_level_realized_loss_09b2n_semantics": float(
                    top_level.mean()
                ),
                "mean_end_to_end_realized_loss_diagnostic": float(
                    end_to_end.mean()
                ),
                "delta_end_to_end_minus_top_level": float(
                    end_to_end.mean()
                    - top_level.mean()
                ),
                "gather_continuation_loss_missing_count": int(
                    g.loc[
                        gather_mask,
                        "post_gather_realized_continuation_loss",
                    ].isna().sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    dataset_dir = Path(
        args.dataset_dir
    ).expanduser().resolve()

    results_dir = Path(
        args.results_dir
    ).expanduser().resolve()

    spec_path = Path(
        args.spec
    ).expanduser().resolve()

    for p in [
        dataset_dir,
        results_dir,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    if not spec_path.exists():
        raise FileNotFoundError(
            spec_path
        )

    out_dir = (
        Path(
            args.out
        ).expanduser().resolve()
        if args.out
        else results_dir
        / "unsafe_error_decomposition"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spec = json.loads(
        spec_path.read_text(
            encoding="utf-8"
        )
    )

    params = validate_spec(
        spec
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2p - UNSAFE EXECUTION ERROR DECOMPOSITION"
    )
    print(
        "=" * 80
    )
    print(
        f"Dataset dir : {dataset_dir}"
    )
    print(
        f"Results dir : {results_dir}"
    )
    print(
        f"Frozen spec : {spec_path}"
    )
    print(
        f"Output      : {out_dir}"
    )
    print(
        f"Policy      : {PRIMARY_POLICY}"
    )
    print(
        f"Consequence : {PRIMARY_CONSEQUENCE}"
    )
    print(
        "NO TRAINING / NO RETUNING / FROZEN REFERENCE POLICY ONLY"
    )

    all_rows = []

    for regime in REGIMES:
        data_path = (
            dataset_dir
            / DATASET_FILES[
                regime
            ]
        )

        decision_path = (
            results_dir
            / DECISION_FILES[
                regime
            ]
        )

        if not data_path.exists():
            raise FileNotFoundError(
                data_path
            )

        if not decision_path.exists():
            raise FileNotFoundError(
                decision_path
            )

        df = pd.read_parquet(
            data_path
        )

        decisions = pd.read_parquet(
            decision_path
        )

        require_columns(
            df,
            [
                "benchmark_episode_id",
                "source_case_id",
                "gate_state_v2",
                "expected_action_reference",
                "transaction_exposure_eur",
                "train_exposure_p90_eur",
                "p_safe_t0_cal",
                "p_safe_t1_for_primary_rcse",
                "gather_policy_eligible",
            ],
            f"dataset/{regime}",
        )

        require_columns(
            decisions,
            [
                "benchmark_episode_id",
                "consequence_mode",
                f"action__{PRIMARY_POLICY}",
            ],
            f"decisions/{regime}",
        )

        d = decisions[
            decisions[
                "consequence_mode"
            ]
            == PRIMARY_CONSEQUENCE
        ][
            [
                "benchmark_episode_id",
                f"action__{PRIMARY_POLICY}",
            ]
        ].copy()

        if len(
            d
        ) != len(
            df
        ):
            raise RuntimeError(
                f"{regime}: expected one {PRIMARY_CONSEQUENCE} decision row "
                f"per dataset row; got decisions={len(d):,}, dataset={len(df):,}"
            )

        if d[
            "benchmark_episode_id"
        ].duplicated().any():
            raise AssertionError(
                f"{regime}: duplicate primary consequence decision rows."
            )

        merged = df.merge(
            d,
            on="benchmark_episode_id",
            how="left",
            validate="one_to_one",
        )

        action_col = f"action__{PRIMARY_POLICY}"

        merged[
            "rcse_v2_full_action"
        ] = normalize_action(
            merged[
                action_col
            ]
        )

        if merged[
            "rcse_v2_full_action"
        ].eq("").any():
            raise RuntimeError(
                f"{regime}: missing RCSE_V2_FULL actions after join."
            )

        merged[
            "evaluation_regime"
        ] = regime

        merged[
            "consequence_log_train_p90"
        ] = consequence_log_p90(
            merged
        )

        merged[
            "t0_safety_target_eval"
        ] = derive_t0_safety_target(
            merged
        )

        if "y_t1_d1" in merged.columns:
            merged[
                "t1_safety_target_eval"
            ] = normalize_safety_label(
                merged[
                    "y_t1_d1"
                ]
            )
        else:
            merged[
                "t1_safety_target_eval"
            ] = "AMBIGUOUS"

        merged[
            "p_safe_t0_band"
        ] = safety_band(
            merged[
                "p_safe_t0_cal"
            ]
        )

        merged[
            "exposure_quantile_within_regime"
        ] = exposure_quantile_band(
            merged
        )

        cont_action, cont_expected_loss = reconstruct_post_gather_continuation(
            merged,
            consequence=merged[
                "consequence_log_train_p90"
            ],
            params=params,
        )

        merged[
            "post_gather_continuation_action"
        ] = cont_action

        merged[
            "post_gather_expected_continuation_loss"
        ] = cont_expected_loss

        merged[
            "post_gather_realized_continuation_loss"
        ] = realized_continuation_loss(
            continuation_action=merged[
                "post_gather_continuation_action"
            ],
            t1_target=merged[
                "t1_safety_target_eval"
            ],
            consequence=merged[
                "consequence_log_train_p90"
            ],
            params=params,
        )

        merged[
            "is_direct_execute"
        ] = (
            merged[
                "rcse_v2_full_action"
            ]
            == "EXECUTE"
        )

        merged[
            "is_direct_unsafe_execute"
        ] = (
            merged[
                "is_direct_execute"
            ]
            & (
                merged[
                    "t0_safety_target_eval"
                ]
                == UNSAFE_LABEL
            )
        )

        merged[
            "is_high_confidence_false_safe_095"
        ] = (
            (
                pd.to_numeric(
                    merged[
                        "p_safe_t0_cal"
                    ],
                    errors="coerce",
                )
                >= 0.95
            )
            & (
                merged[
                    "t0_safety_target_eval"
                ]
                == UNSAFE_LABEL
            )
        )

        all_rows.append(
            merged
        )

        # Save regime diagnostics.
        export_cols = [
            c for c in [
                "benchmark_episode_id",
                "source_case_id",
                "evaluation_regime",
                "expected_action_reference",
                "gate_state_v2",
                "intervention_family_eval",
                "intervention_type",
                "intervention_strength",
                "transaction_risk_stratum",
                "transaction_exposure_eur",
                "exposure_quantile_within_regime",
                "p_safe_t0_raw",
                "p_safe_t0_cal",
                "p_safe_t0_band",
                "p_safe_t1_for_primary_rcse",
                "t0_safety_target_eval",
                "t1_safety_target_eval",
                "rcse_v2_full_action",
                "is_direct_execute",
                "is_direct_unsafe_execute",
                "is_high_confidence_false_safe_095",
                "gather_policy_eligible",
                "post_gather_continuation_action",
                "post_gather_expected_continuation_loss",
                "post_gather_realized_continuation_loss",
                "consequence_log_train_p90",
            ]
            if c in merged.columns
        ]

        merged[
            export_cols
        ].to_parquet(
            out_dir
            / f"rcse_v2p_episode_diagnostics_{regime}.parquet",
            index=False,
        )

        direct = direct_execution_metrics(
            merged
        )

        print(
            f"\n[{regime}] "
            f"execute={direct['execute_count']:,} "
            f"| coverage={direct['execution_coverage']:.4f} "
            f"| unsafe={direct['unsafe_execution_rate']:.4f}"
        )

    full = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------------
    # Direct execution summaries
    # --------------------------------------------------------------
    direct_summary_rows = []

    for regime, g in full.groupby(
        "evaluation_regime",
        observed=True,
    ):
        direct_summary_rows.append(
            {
                "regime": regime,
                **direct_execution_metrics(
                    g
                ),
            }
        )

    direct_summary = pd.DataFrame(
        direct_summary_rows
    )

    # Required decompositions
    if "intervention_family_eval" in full.columns:
        by_intervention = group_direct_errors(
            full,
            [
                "evaluation_regime",
                "intervention_family_eval",
            ],
        )
    else:
        by_intervention = pd.DataFrame()

    if {
        "intervention_type",
        "intervention_strength",
    }.issubset(
        full.columns
    ):
        by_type_strength = group_direct_errors(
            full,
            [
                "evaluation_regime",
                "intervention_type",
                "intervention_strength",
            ],
        )
    else:
        by_type_strength = pd.DataFrame()

    by_gate = group_direct_errors(
        full,
        [
            "evaluation_regime",
            "gate_state_v2",
        ],
    )

    by_safety_band = group_direct_errors(
        full,
        [
            "evaluation_regime",
            "p_safe_t0_band",
        ],
    )

    by_exposure = group_direct_errors(
        full,
        [
            "evaluation_regime",
            "exposure_quantile_within_regime",
        ],
    )

    by_reference = group_direct_errors(
        full,
        [
            "evaluation_regime",
            "expected_action_reference",
        ],
    )

    high_conf = high_confidence_false_safe(
        full
    )

    gather = gather_summary(
        full
    )

    if "intervention_family_eval" in full.columns:
        gather_by_intervention_rows = []

        gather_only = full[
            full[
                "rcse_v2_full_action"
            ]
            == "GATHER"
        ].copy()

        for (
            regime,
            fam,
        ), g in gather_only.groupby(
            [
                "evaluation_regime",
                "intervention_family_eval",
            ],
            dropna=False,
            observed=True,
        ):
            post_exec = (
                g[
                    "post_gather_continuation_action"
                ]
                == "EXECUTE"
            )

            known = (
                post_exec
                & g[
                    "t1_safety_target_eval"
                ].isin(
                    [
                        SAFE_LABEL,
                        UNSAFE_LABEL,
                    ]
                )
            )

            unsafe = (
                known
                & (
                    g[
                        "t1_safety_target_eval"
                    ]
                    == UNSAFE_LABEL
                )
            )

            gather_by_intervention_rows.append(
                {
                    "regime": regime,
                    "intervention_family": fam,
                    "gather_count": int(
                        len(g)
                    ),
                    "post_gather_execute_count": int(
                        post_exec.sum()
                    ),
                    "post_gather_execute_rate": float(
                        post_exec.mean()
                    ),
                    "known_post_gather_execute_safety_count": int(
                        known.sum()
                    ),
                    "unsafe_post_gather_execute_count": int(
                        unsafe.sum()
                    ),
                    "unsafe_post_gather_execution_rate_known": (
                        float(
                            unsafe.sum()
                            / known.sum()
                        )
                        if known.sum()
                        else np.nan
                    ),
                }
            )

        gather_by_intervention = pd.DataFrame(
            gather_by_intervention_rows
        )
    else:
        gather_by_intervention = pd.DataFrame()

    gather_conversion = gather_conversion_summary(
        full
    )

    loss_audit = loss_accounting_audit(
        full,
        params=params,
    )

    # --------------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------------
    direct_summary.to_csv(
        out_dir
        / "rcse_v2p_direct_execution_summary.csv",
        index=False,
    )

    if not by_intervention.empty:
        by_intervention.to_csv(
            out_dir
            / "rcse_v2p_unsafe_by_intervention_family.csv",
            index=False,
        )

    if not by_type_strength.empty:
        by_type_strength.to_csv(
            out_dir
            / "rcse_v2p_unsafe_by_intervention_type_strength.csv",
            index=False,
        )

    by_gate.to_csv(
        out_dir
        / "rcse_v2p_unsafe_by_gate_state.csv",
        index=False,
    )

    by_safety_band.to_csv(
        out_dir
        / "rcse_v2p_unsafe_by_safety_band.csv",
        index=False,
    )

    by_exposure.to_csv(
        out_dir
        / "rcse_v2p_unsafe_by_exposure_quantile.csv",
        index=False,
    )

    by_reference.to_csv(
        out_dir
        / "rcse_v2p_unsafe_by_reference_action.csv",
        index=False,
    )

    high_conf.to_csv(
        out_dir
        / "rcse_v2p_high_confidence_false_safe.csv",
        index=False,
    )

    gather.to_csv(
        out_dir
        / "rcse_v2p_gather_continuation_summary.csv",
        index=False,
    )

    if not gather_by_intervention.empty:
        gather_by_intervention.to_csv(
            out_dir
            / "rcse_v2p_gather_continuation_by_intervention.csv",
            index=False,
        )

    gather_conversion.to_csv(
        out_dir
        / "rcse_v2p_gather_safety_conversion.csv",
        index=False,
    )

    loss_audit.to_csv(
        out_dir
        / "rcse_v2p_loss_accounting_audit.csv",
        index=False,
    )

    metadata: dict[
        str,
        Any,
    ] = {
        "step": "09b.2p",
        "policy": PRIMARY_POLICY,
        "consequence_mode": PRIMARY_CONSEQUENCE,
        "frozen_reference_parameters": params,
        "diagnostic_only": True,
        "training_performed": False,
        "recalibration_performed": False,
        "gate_modified": False,
        "policy_retuned": False,
        "frontier_selected_point_used": False,
        "direct_execution_safety_semantics": (
            "t0_safety_robustness_target when available; unresolved cases "
            "fall back to expected_action_reference semantics."
        ),
        "post_gather_safety_semantics": (
            "y_t1_d1 when available; ambiguous t1 outcomes remain unresolved."
        ),
        "loss_accounting_note": (
            "Step 09b.2n top-level realized loss assigns GATHER=c_G only. "
            "This diagnostic additionally reports c_G plus realized continuation "
            "loss. It does not overwrite or silently modify Step 09b.2n outputs."
        ),
        "next_decision": (
            "Use the error decomposition to determine whether the residual "
            "unsafe-execution floor is driven by natural false-safe cases, "
            "residual synthetic contradictions, OOD safety-score failure, "
            "post-GATHER continuation, or a mixture. Only then decide whether "
            "method changes or statistical uncertainty analysis are warranted."
        ),
    }

    with open(
        out_dir
        / "rcse_v2p_metadata.json",
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
        "STEP 09b.2p COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nDirect unsafe-execution summary:"
    )

    print(
        direct_summary.to_string(
            index=False
        )
    )

    if not by_intervention.empty:
        print(
            "\nUnsafe execution by intervention family:"
        )

        print(
            by_intervention.to_string(
                index=False
            )
        )

    print(
        "\nUnsafe execution by t0 safety-score band:"
    )

    print(
        by_safety_band.to_string(
            index=False
        )
    )

    print(
        "\nHigh-confidence false-safe summary:"
    )

    print(
        high_conf.to_string(
            index=False
        )
    )

    print(
        "\nPost-GATHER continuation summary:"
    )

    print(
        gather.to_string(
            index=False
        )
    )

    print(
        "\nLoss-accounting audit:"
    )

    print(
        loss_audit.to_string(
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
        "\nNext: interpret the residual unsafe-execution floor before any "
        "RCSE method change or statistical bootstrap."
    )


if __name__ == "__main__":
    main()
