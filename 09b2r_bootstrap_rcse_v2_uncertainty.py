#!/usr/bin/env python
"""
09b2r_bootstrap_rcse_v2_uncertainty.py

RCSE Step 09b.2r
Source-case-level bootstrap uncertainty analysis for the frozen RCSE v2 results.

Purpose
-------
Quantify statistical uncertainty for the already-frozen RCSE v2 primary
reference analysis WITHOUT modifying the model, gate, policy, thresholds,
or consequence specification.

Bootstrap unit
--------------
source_case_id

All benchmark episodes derived from a sampled source case move together.
This preserves dependence among NATURAL and synthetic intervention episodes
derived from the same underlying source transaction.

Primary policy
--------------
RCSE_V2_FULL

Primary consequence
-------------------
LOG_TRAIN_P90

Predeclared paired ablations
----------------------------
RCSE_V2_FULL vs:
    RCSE_V2_NO_GATHER
    V2_SAFETY_NO_GATE
    V2_GATE_MULTICLASS_EXECUTE
    VOI_BLIND_V1
    RCSE_V1_FULL

Primary metrics
---------------
- autonomous_execution_coverage
- unsafe_execution_rate
- raw_exposure_weighted_unsafe_rate
- consequence_weighted_unsafe_rate
- mean_realized_normalized_loss_09b2n
- mean_end_to_end_realized_loss_diagnostic
- human_escalation_rate
- gather_rate
- abstention_rate

Additional RCSE_V2_FULL diagnostics
-----------------------------------
- natural-only execution coverage
- natural-only unsafe execution rate
- contradiction gate block rate
- residual-valid contradiction execution rate
- missingness block rate
- severe-evidence-loss block rate

Important
---------
NO MODEL TRAINING.
NO RECALIBRATION.
NO GATE MODIFICATION.
NO POLICY RETUNING.
NO NEW OPERATING-POINT SELECTION.

Default bootstrap:
    2000 replicates
    seed = 20260823

Outputs
-------
rcse_v2r_bootstrap_policy_ci.csv
rcse_v2r_bootstrap_ablation_ci.csv
rcse_v2r_bootstrap_diagnostic_ci.csv
rcse_v2r_bootstrap_point_estimates.csv
rcse_v2r_bootstrap_replicates_policy.parquet
rcse_v2r_bootstrap_replicates_ablation.parquet
rcse_v2r_bootstrap_replicates_diagnostics.parquet
rcse_v2r_bootstrap_metadata.json
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

PRIMARY_CONSEQUENCE = "LOG_TRAIN_P90"
PRIMARY_POLICY = "RCSE_V2_FULL"

POLICIES = (
    "ARGMAX_V1",
    "CONFIDENCE_THRESHOLD_V1",
    "VOI_BLIND_V1",
    "RCSE_V1_FULL",
    "V2_SAFETY_NO_GATE",
    "V2_GATE_MULTICLASS_EXECUTE",
    "RCSE_V2_NO_GATHER",
    "RCSE_V2_FULL",
)

ABLATIONS = (
    ("RCSE_V2_FULL", "RCSE_V2_NO_GATHER", "GATHER_CONTRIBUTION"),
    ("RCSE_V2_FULL", "V2_SAFETY_NO_GATE", "EVIDENCE_GATE_CONTRIBUTION"),
    (
        "RCSE_V2_FULL",
        "V2_GATE_MULTICLASS_EXECUTE",
        "DEDICATED_SAFETY_BELIEF_CONTRIBUTION",
    ),
    ("RCSE_V2_FULL", "VOI_BLIND_V1", "CONSEQUENCE_AWARE_VS_VOI_BLIND"),
    ("RCSE_V2_FULL", "RCSE_V1_FULL", "V2_VS_V1"),
)

SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2r: source-case bootstrap uncertainty for frozen RCSE v2."
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
        help="Default: <results-dir>/bootstrap_uncertainty",
    )

    p.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
        help="Number of source-case bootstrap replicates.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=20260823,
        help="Random seed.",
    )

    p.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Percentile CI level.",
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
        z.isin([SAFE_LABEL, "SAFE", "TRUE", "1"])
    ] = SAFE_LABEL

    out.loc[
        z.isin([UNSAFE_LABEL, "UNSAFE", "NOT_SAFE", "FALSE", "0"])
    ] = UNSAFE_LABEL

    return out


def validate_spec(spec: dict[str, Any]) -> dict[str, float]:
    if spec.get("status") != "FROZEN_BEFORE_RCSE_V2_SIMULATION":
        raise RuntimeError(
            "Spec is not the frozen RCSE v2 specification."
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
            f"Expected one train_exposure_p90_eur value; got {len(ref_values)}"
        )

    ref = float(ref_values[0])

    if ref <= 0:
        raise RuntimeError(
            f"Invalid train exposure P90 reference: {ref}"
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


def infer_intervention_family(df: pd.DataFrame) -> pd.Series:
    if "intervention_family_eval" in df.columns:
        return (
            df["intervention_family_eval"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "intervention_type" not in df.columns:
        return pd.Series(
            "UNKNOWN",
            index=df.index,
            dtype="object",
        )

    z = (
        df["intervention_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series("OTHER", index=df.index, dtype="object")

    out.loc[
        z.isin(["", "NONE", "NATURAL", "NO_INTERVENTION"])
    ] = "NATURAL"

    out.loc[
        z.str.contains("MISSING", na=False)
        | z.str.contains("HIDE", na=False)
    ] = "MISSINGNESS"

    out.loc[
        z.str.contains("CONTRAD", na=False)
        | z.str.contains("PERTURB", na=False)
    ] = "CONTRADICTION"

    out.loc[
        z.str.contains("SEVERE", na=False)
        | z.str.contains("MULTI", na=False)
    ] = "SEVERE_EVIDENCE_LOSS"

    return out


def reconstruct_post_gather_action(
    df: pd.DataFrame,
    consequence: pd.Series,
    params: dict[str, float],
) -> pd.Series:
    p1 = pd.to_numeric(
        df["p_safe_t1_for_primary_rcse"],
        errors="coerce",
    )

    l_e = (
        (1.0 - p1)
        * consequence
    )

    exec_feasible = (
        p1.notna()
        & (p1 >= params["tau_safe_t1"])
        & (l_e <= params["beta1"])
    )

    l_e = l_e.where(
        exec_feasible,
        np.inf,
    )

    l_h = (
        params["c_H"]
        + params["rho_H"] * consequence
    )

    l_a = pd.Series(
        params["c_A"],
        index=df.index,
        dtype=float,
    )

    losses = pd.concat(
        [
            l_e.rename("EXECUTE"),
            l_h.rename("ESCALATE"),
            l_a.rename("ABSTAIN"),
        ],
        axis=1,
    )

    return losses.idxmin(axis=1)


def derive_t1_safety_target(df: pd.DataFrame) -> pd.Series:
    if "y_t1_d1" not in df.columns:
        return pd.Series(
            "AMBIGUOUS",
            index=df.index,
            dtype="object",
        )

    return normalize_safety_label(
        df["y_t1_d1"]
    )


def realized_loss_09b2n(
    action: pd.Series,
    unsafe: pd.Series,
    consequence: pd.Series,
    params: dict[str, float],
) -> pd.Series:
    out = pd.Series(
        np.nan,
        index=action.index,
        dtype=float,
    )

    execute = action == "EXECUTE"
    gather = action == "GATHER"
    escalate = action == "ESCALATE"
    abstain = action == "ABSTAIN"

    out.loc[
        execute
    ] = (
        unsafe.loc[
            execute
        ].astype(float)
        * consequence.loc[
            execute
        ]
    )

    out.loc[
        gather
    ] = params["c_G"]

    out.loc[
        escalate
    ] = (
        params["c_H"]
        + params["rho_H"]
        * consequence.loc[
            escalate
        ]
    )

    out.loc[
        abstain
    ] = params["c_A"]

    return out


def realized_post_gather_continuation_loss(
    df: pd.DataFrame,
    post_action: pd.Series,
    consequence: pd.Series,
    params: dict[str, float],
) -> pd.Series:
    target = derive_t1_safety_target(df)

    out = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    execute = post_action == "EXECUTE"
    escalate = post_action == "ESCALATE"
    abstain = post_action == "ABSTAIN"

    known_exec = (
        execute
        & target.isin(
            [
                SAFE_LABEL,
                UNSAFE_LABEL,
            ]
        )
    )

    unsafe_exec = (
        known_exec
        & (target == UNSAFE_LABEL)
    )

    out.loc[
        known_exec
    ] = 0.0

    out.loc[
        unsafe_exec
    ] = consequence.loc[
        unsafe_exec
    ]

    out.loc[
        escalate
    ] = (
        params["c_H"]
        + params["rho_H"]
        * consequence.loc[
            escalate
        ]
    )

    out.loc[
        abstain
    ] = params["c_A"]

    return out


def prepare_regime_data(
    dataset_dir: Path,
    results_dir: Path,
    regime: str,
    params: dict[str, float],
) -> pd.DataFrame:
    dataset_path = (
        dataset_dir
        / DATASET_FILES[
            regime
        ]
    )

    decisions_path = (
        results_dir
        / DECISION_FILES[
            regime
        ]
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            dataset_path
        )

    if not decisions_path.exists():
        raise FileNotFoundError(
            decisions_path
        )

    df = pd.read_parquet(
        dataset_path
    )

    decisions = pd.read_parquet(
        decisions_path
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
            "p_safe_t1_for_primary_rcse",
        ],
        f"dataset/{regime}",
    )

    required_decision_cols = [
        "benchmark_episode_id",
        "consequence_mode",
        *[
            f"action__{p}"
            for p in POLICIES
        ],
    ]

    require_columns(
        decisions,
        required_decision_cols,
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
            *[
                f"action__{p}"
                for p in POLICIES
            ],
        ]
    ].copy()

    if d[
        "benchmark_episode_id"
    ].duplicated().any():
        raise AssertionError(
            f"{regime}: duplicate primary-consequence decision rows."
        )

    df = df.merge(
        d,
        on="benchmark_episode_id",
        how="left",
        validate="one_to_one",
    )

    df[
        "source_case_id"
    ] = df[
        "source_case_id"
    ].astype(str)

    df[
        "evaluation_regime"
    ] = regime

    df[
        "_consequence"
    ] = consequence_log_p90(
        df
    )

    df[
        "_t0_safety_target"
    ] = derive_t0_safety_target(
        df
    )

    df[
        "_unsafe_t0"
    ] = (
        df[
            "_t0_safety_target"
        ]
        == UNSAFE_LABEL
    )

    df[
        "_intervention_family_r"
    ] = infer_intervention_family(
        df
    )

    post_action = reconstruct_post_gather_action(
        df,
        consequence=df[
            "_consequence"
        ],
        params=params,
    )

    df[
        "_post_gather_action"
    ] = post_action

    post_realized = realized_post_gather_continuation_loss(
        df,
        post_action=post_action,
        consequence=df[
            "_consequence"
        ],
        params=params,
    )

    df[
        "_post_gather_realized_loss"
    ] = post_realized

    for policy in POLICIES:
        action_col = f"action__{policy}"

        df[
            action_col
        ] = normalize_action(
            df[
                action_col
            ]
        )

        top = realized_loss_09b2n(
            action=df[
                action_col
            ],
            unsafe=df[
                "_unsafe_t0"
            ],
            consequence=df[
                "_consequence"
            ],
            params=params,
        )

        df[
            f"_loss09__{policy}"
        ] = top

        end = top.copy()

        gather = (
            df[
                action_col
            ]
            == "GATHER"
        )

        end.loc[
            gather
        ] = (
            params["c_G"]
            + df.loc[
                gather,
                "_post_gather_realized_loss",
            ]
        )

        df[
            f"_loss_e2e__{policy}"
        ] = end

    return df


def weighted_sum(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    return float(
        np.sum(
            values
            * weights
        )
    )


def weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    mask = np.isfinite(
        values
    )

    if not np.any(
        mask
    ):
        return np.nan

    denom = np.sum(
        weights[
            mask
        ]
    )

    if denom <= 0:
        return np.nan

    return float(
        np.sum(
            values[
                mask
            ]
            * weights[
                mask
            ]
        )
        / denom
    )


def metric_bundle(
    df: pd.DataFrame,
    weights: np.ndarray,
    policy: str,
) -> dict[str, float]:
    action = df[
        f"action__{policy}"
    ].to_numpy()

    unsafe = df[
        "_unsafe_t0"
    ].to_numpy(
        dtype=bool
    )

    consequence = df[
        "_consequence"
    ].to_numpy(
        dtype=float
    )

    exposure = (
        pd.to_numeric(
            df[
                "transaction_exposure_eur"
            ],
            errors="coerce",
        )
        .abs()
        .fillna(0.0)
        .to_numpy(
            dtype=float
        )
    )

    total_w = float(
        np.sum(
            weights
        )
    )

    execute = action == "EXECUTE"
    gather = action == "GATHER"
    escalate = action == "ESCALATE"
    abstain = action == "ABSTAIN"

    exec_w = weighted_sum(
        execute.astype(float),
        weights,
    )

    unsafe_exec = (
        execute
        & unsafe
    )

    unsafe_exec_w = weighted_sum(
        unsafe_exec.astype(float),
        weights,
    )

    selected_exposure = weighted_sum(
        exposure
        * execute.astype(float),
        weights,
    )

    unsafe_selected_exposure = weighted_sum(
        exposure
        * unsafe_exec.astype(float),
        weights,
    )

    selected_consequence = weighted_sum(
        consequence
        * execute.astype(float),
        weights,
    )

    unsafe_selected_consequence = weighted_sum(
        consequence
        * unsafe_exec.astype(float),
        weights,
    )

    loss09 = df[
        f"_loss09__{policy}"
    ].to_numpy(
        dtype=float
    )

    losse2e = df[
        f"_loss_e2e__{policy}"
    ].to_numpy(
        dtype=float
    )

    return {
        "autonomous_execution_coverage": (
            exec_w / total_w
            if total_w > 0
            else np.nan
        ),
        "unsafe_execution_rate": (
            unsafe_exec_w / exec_w
            if exec_w > 0
            else np.nan
        ),
        "raw_exposure_weighted_unsafe_rate": (
            unsafe_selected_exposure
            / selected_exposure
            if selected_exposure > 0
            else np.nan
        ),
        "consequence_weighted_unsafe_rate": (
            unsafe_selected_consequence
            / selected_consequence
            if selected_consequence > 0
            else np.nan
        ),
        "mean_realized_normalized_loss_09b2n": weighted_mean(
            loss09,
            weights,
        ),
        "mean_end_to_end_realized_loss_diagnostic": weighted_mean(
            losse2e,
            weights,
        ),
        "human_escalation_rate": (
            weighted_sum(
                escalate.astype(float),
                weights,
            )
            / total_w
            if total_w > 0
            else np.nan
        ),
        "gather_rate": (
            weighted_sum(
                gather.astype(float),
                weights,
            )
            / total_w
            if total_w > 0
            else np.nan
        ),
        "abstention_rate": (
            weighted_sum(
                abstain.astype(float),
                weights,
            )
            / total_w
            if total_w > 0
            else np.nan
        ),
    }


def diagnostic_bundle(
    df: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float]:
    action = df[
        f"action__{PRIMARY_POLICY}"
    ].to_numpy()

    family = df[
        "_intervention_family_r"
    ].to_numpy()

    gate = (
        df[
            "gate_state_v2"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .to_numpy()
    )

    unsafe = df[
        "_unsafe_t0"
    ].to_numpy(
        dtype=bool
    )

    ref_execute = (
        normalize_action(
            df[
                "expected_action_reference"
            ]
        ).to_numpy()
        == "EXECUTE"
    )

    natural = family == "NATURAL"
    contradiction = family == "CONTRADICTION"
    missingness = family == "MISSINGNESS"
    severe = family == "SEVERE_EVIDENCE_LOSS"

    execute = action == "EXECUTE"

    natural_ref_exec = (
        natural
        & ref_execute
    )

    natural_exec = (
        natural
        & execute
    )

    natural_unsafe_exec = (
        natural_exec
        & unsafe
    )

    contradiction_blocked = (
        contradiction
        & (gate != "VALID")
    )

    contradiction_valid = (
        contradiction
        & (gate == "VALID")
    )

    contradiction_valid_executed = (
        contradiction_valid
        & execute
    )

    missingness_blocked = (
        missingness
        & (gate != "VALID")
    )

    severe_blocked = (
        severe
        & (gate != "VALID")
    )

    def ratio(num_mask: np.ndarray, den_mask: np.ndarray) -> float:
        num = weighted_sum(
            num_mask.astype(float),
            weights,
        )

        den = weighted_sum(
            den_mask.astype(float),
            weights,
        )

        return (
            num / den
            if den > 0
            else np.nan
        )

    return {
        "natural_reference_execute_retention_rate": ratio(
            natural_ref_exec
            & (gate == "VALID"),
            natural_ref_exec,
        ),
        "natural_execution_coverage": ratio(
            natural_exec,
            natural,
        ),
        "natural_unsafe_execution_rate": ratio(
            natural_unsafe_exec,
            natural_exec,
        ),
        "contradiction_gate_block_rate": ratio(
            contradiction_blocked,
            contradiction,
        ),
        "residual_valid_contradiction_execution_rate": ratio(
            contradiction_valid_executed,
            contradiction_valid,
        ),
        "missingness_gate_block_rate": ratio(
            missingness_blocked,
            missingness,
        ),
        "severe_evidence_loss_gate_block_rate": ratio(
            severe_blocked,
            severe,
        ),
    }


def source_case_bootstrap_weights(
    source_codes: np.ndarray,
    n_sources: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample n_sources source cases with replacement.
    Return episode-level multiplicity weights.
    """
    sampled = rng.integers(
        low=0,
        high=n_sources,
        size=n_sources,
    )

    source_multiplicity = np.bincount(
        sampled,
        minlength=n_sources,
    ).astype(float)

    return source_multiplicity[
        source_codes
    ]


def ci_from_values(
    values: pd.Series,
    ci_level: float,
) -> dict[str, float]:
    x = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(x) == 0:
        return {
            "bootstrap_valid_reps": 0,
            "bootstrap_mean": np.nan,
            "bootstrap_se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }

    alpha = (
        1.0
        - ci_level
    )

    return {
        "bootstrap_valid_reps": int(
            len(x)
        ),
        "bootstrap_mean": float(
            x.mean()
        ),
        "bootstrap_se": float(
            x.std(
                ddof=1
            )
        )
        if len(x) > 1
        else np.nan,
        "ci_lower": float(
            x.quantile(
                alpha / 2.0
            )
        ),
        "ci_upper": float(
            x.quantile(
                1.0
                - alpha / 2.0
            )
        ),
    }


def summarize_policy_ci(
    point_df: pd.DataFrame,
    reps_df: pd.DataFrame,
    ci_level: float,
) -> pd.DataFrame:
    metrics = [
        "autonomous_execution_coverage",
        "unsafe_execution_rate",
        "raw_exposure_weighted_unsafe_rate",
        "consequence_weighted_unsafe_rate",
        "mean_realized_normalized_loss_09b2n",
        "mean_end_to_end_realized_loss_diagnostic",
        "human_escalation_rate",
        "gather_rate",
        "abstention_rate",
    ]

    rows = []

    for (
        regime,
        policy,
    ), g in reps_df.groupby(
        [
            "regime",
            "policy",
        ],
        observed=True,
    ):
        point = point_df[
            (
                point_df[
                    "regime"
                ]
                == regime
            )
            & (
                point_df[
                    "policy"
                ]
                == policy
            )
        ].iloc[0]

        for metric in metrics:
            row = {
                "regime": regime,
                "policy": policy,
                "metric": metric,
                "point_estimate": float(
                    point[
                        metric
                    ]
                )
                if pd.notna(
                    point[
                        metric
                    ]
                )
                else np.nan,
            }

            row.update(
                ci_from_values(
                    g[
                        metric
                    ],
                    ci_level,
                )
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def summarize_ablation_ci(
    reps_df: pd.DataFrame,
    ci_level: float,
) -> pd.DataFrame:
    metrics = [
        "autonomous_execution_coverage",
        "unsafe_execution_rate",
        "raw_exposure_weighted_unsafe_rate",
        "consequence_weighted_unsafe_rate",
        "mean_realized_normalized_loss_09b2n",
        "mean_end_to_end_realized_loss_diagnostic",
        "human_escalation_rate",
        "gather_rate",
        "abstention_rate",
    ]

    rows = []

    for regime in REGIMES:
        rg = reps_df[
            reps_df[
                "regime"
            ]
            == regime
        ]

        pivot = rg.set_index(
            [
                "bootstrap_rep",
                "policy",
            ]
        )

        for left, right, label in ABLATIONS:
            for metric in metrics:
                left_s = pivot.xs(
                    left,
                    level="policy",
                )[
                    metric
                ]

                right_s = pivot.xs(
                    right,
                    level="policy",
                )[
                    metric
                ]

                common = left_s.index.intersection(
                    right_s.index
                )

                delta = (
                    left_s.loc[
                        common
                    ]
                    - right_s.loc[
                        common
                    ]
                )

                ci = ci_from_values(
                    delta,
                    ci_level,
                )

                rows.append(
                    {
                        "regime": regime,
                        "comparison": label,
                        "left_policy": left,
                        "right_policy": right,
                        "metric": metric,
                        "delta_definition": "left_minus_right",
                        **ci,
                        "probability_delta_below_zero": (
                            float(
                                (
                                    delta.dropna()
                                    < 0
                                ).mean()
                            )
                            if delta.notna().any()
                            else np.nan
                        ),
                        "probability_delta_above_zero": (
                            float(
                                (
                                    delta.dropna()
                                    > 0
                                ).mean()
                            )
                            if delta.notna().any()
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def summarize_diagnostic_ci(
    point_df: pd.DataFrame,
    reps_df: pd.DataFrame,
    ci_level: float,
) -> pd.DataFrame:
    metric_cols = [
        c
        for c in reps_df.columns
        if c not in {
            "regime",
            "bootstrap_rep",
        }
    ]

    rows = []

    for regime in REGIMES:
        g = reps_df[
            reps_df[
                "regime"
            ]
            == regime
        ]

        point = point_df[
            point_df[
                "regime"
            ]
            == regime
        ].iloc[0]

        for metric in metric_cols:
            row = {
                "regime": regime,
                "metric": metric,
                "point_estimate": (
                    float(
                        point[
                            metric
                        ]
                    )
                    if pd.notna(
                        point[
                            metric
                        ]
                    )
                    else np.nan
                ),
            }

            row.update(
                ci_from_values(
                    g[
                        metric
                    ],
                    ci_level,
                )
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    if args.bootstrap_reps < 100:
        raise ValueError(
            "--bootstrap-reps should be at least 100."
        )

    if not (
        0.50
        < args.ci_level
        < 1.0
    ):
        raise ValueError(
            "--ci-level must be between 0.50 and 1."
        )

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
        / "bootstrap_uncertainty"
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
        "RCSE STEP 09b.2r - SOURCE-CASE BOOTSTRAP UNCERTAINTY"
    )
    print(
        "=" * 80
    )
    print(
        f"Dataset dir    : {dataset_dir}"
    )
    print(
        f"Results dir    : {results_dir}"
    )
    print(
        f"Frozen spec    : {spec_path}"
    )
    print(
        f"Output         : {out_dir}"
    )
    print(
        f"Bootstrap reps : {args.bootstrap_reps:,}"
    )
    print(
        f"Seed           : {args.seed}"
    )
    print(
        f"CI level       : {args.ci_level:.3f}"
    )
    print(
        f"Consequence    : {PRIMARY_CONSEQUENCE}"
    )
    print(
        "Bootstrap unit : source_case_id"
    )
    print(
        "NO TRAINING / NO GATE MODIFICATION / NO RETUNING"
    )

    rng = np.random.default_rng(
        args.seed
    )

    all_policy_points = []
    all_diag_points = []
    all_policy_reps = []
    all_diag_reps = []

    regime_meta = {}

    for regime in REGIMES:
        print(
            f"\n[{regime}] loading..."
        )

        df = prepare_regime_data(
            dataset_dir=dataset_dir,
            results_dir=results_dir,
            regime=regime,
            params=params,
        )

        source_cat = pd.Categorical(
            df[
                "source_case_id"
            ]
        )

        source_codes = source_cat.codes

        if (
            source_codes
            < 0
        ).any():
            raise RuntimeError(
                f"{regime}: missing source_case_id."
            )

        n_sources = len(
            source_cat.categories
        )

        n_episodes = len(
            df
        )

        base_weights = np.ones(
            n_episodes,
            dtype=float,
        )

        regime_meta[
            regime
        ] = {
            "episodes": int(
                n_episodes
            ),
            "unique_source_cases": int(
                n_sources
            ),
        }

        print(
            f"   episodes={n_episodes:,} "
            f"| source cases={n_sources:,}"
        )

        # Point estimates.
        for policy in POLICIES:
            m = metric_bundle(
                df,
                weights=base_weights,
                policy=policy,
            )

            all_policy_points.append(
                {
                    "regime": regime,
                    "policy": policy,
                    **m,
                }
            )

        d0 = diagnostic_bundle(
            df,
            weights=base_weights,
        )

        all_diag_points.append(
            {
                "regime": regime,
                **d0,
            }
        )

        # Bootstrap.
        for b in range(
            args.bootstrap_reps
        ):
            weights = source_case_bootstrap_weights(
                source_codes=source_codes,
                n_sources=n_sources,
                rng=rng,
            )

            for policy in POLICIES:
                m = metric_bundle(
                    df,
                    weights=weights,
                    policy=policy,
                )

                all_policy_reps.append(
                    {
                        "regime": regime,
                        "bootstrap_rep": b + 1,
                        "policy": policy,
                        **m,
                    }
                )

            d = diagnostic_bundle(
                df,
                weights=weights,
            )

            all_diag_reps.append(
                {
                    "regime": regime,
                    "bootstrap_rep": b + 1,
                    **d,
                }
            )

            if (
                (b + 1)
                % 250
                == 0
            ):
                print(
                    f"   bootstrap {b + 1:,}/{args.bootstrap_reps:,}"
                )

    policy_points_df = pd.DataFrame(
        all_policy_points
    )

    diag_points_df = pd.DataFrame(
        all_diag_points
    )

    policy_reps_df = pd.DataFrame(
        all_policy_reps
    )

    diag_reps_df = pd.DataFrame(
        all_diag_reps
    )

    policy_ci = summarize_policy_ci(
        point_df=policy_points_df,
        reps_df=policy_reps_df,
        ci_level=args.ci_level,
    )

    ablation_ci = summarize_ablation_ci(
        reps_df=policy_reps_df,
        ci_level=args.ci_level,
    )

    diag_ci = summarize_diagnostic_ci(
        point_df=diag_points_df,
        reps_df=diag_reps_df,
        ci_level=args.ci_level,
    )

    # Combined point estimate file.
    point_out = policy_points_df.copy()

    policy_points_df.to_csv(
        out_dir
        / "rcse_v2r_bootstrap_point_estimates.csv",
        index=False,
    )

    diag_points_df.to_csv(
        out_dir
        / "rcse_v2r_bootstrap_diagnostic_point_estimates.csv",
        index=False,
    )

    policy_ci.to_csv(
        out_dir
        / "rcse_v2r_bootstrap_policy_ci.csv",
        index=False,
    )

    ablation_ci.to_csv(
        out_dir
        / "rcse_v2r_bootstrap_ablation_ci.csv",
        index=False,
    )

    diag_ci.to_csv(
        out_dir
        / "rcse_v2r_bootstrap_diagnostic_ci.csv",
        index=False,
    )

    policy_reps_df.to_parquet(
        out_dir
        / "rcse_v2r_bootstrap_replicates_policy.parquet",
        index=False,
    )

    diag_reps_df.to_parquet(
        out_dir
        / "rcse_v2r_bootstrap_replicates_diagnostics.parquet",
        index=False,
    )

    # Paired ablation replicate artifact, wide enough for later plotting.
    ablation_rep_rows = []

    for regime in REGIMES:
        rg = policy_reps_df[
            policy_reps_df[
                "regime"
            ]
            == regime
        ]

        for left, right, label in ABLATIONS:
            left_df = rg[
                rg[
                    "policy"
                ]
                == left
            ].set_index(
                "bootstrap_rep"
            )

            right_df = rg[
                rg[
                    "policy"
                ]
                == right
            ].set_index(
                "bootstrap_rep"
            )

            common = left_df.index.intersection(
                right_df.index
            )

            metric_cols = [
                "autonomous_execution_coverage",
                "unsafe_execution_rate",
                "raw_exposure_weighted_unsafe_rate",
                "consequence_weighted_unsafe_rate",
                "mean_realized_normalized_loss_09b2n",
                "mean_end_to_end_realized_loss_diagnostic",
                "human_escalation_rate",
                "gather_rate",
                "abstention_rate",
            ]

            temp = pd.DataFrame(
                {
                    "regime": regime,
                    "bootstrap_rep": common,
                    "comparison": label,
                    "left_policy": left,
                    "right_policy": right,
                }
            )

            for metric in metric_cols:
                temp[
                    f"delta__{metric}"
                ] = (
                    left_df.loc[
                        common,
                        metric,
                    ].to_numpy()
                    - right_df.loc[
                        common,
                        metric,
                    ].to_numpy()
                )

            ablation_rep_rows.append(
                temp
            )

    ablation_reps_df = pd.concat(
        ablation_rep_rows,
        ignore_index=True,
    )

    ablation_reps_df.to_parquet(
        out_dir
        / "rcse_v2r_bootstrap_replicates_ablation.parquet",
        index=False,
    )

    metadata: dict[
        str,
        Any,
    ] = {
        "step": "09b.2r",
        "analysis": "source_case_level_cluster_bootstrap",
        "bootstrap_reps": int(
            args.bootstrap_reps
        ),
        "seed": int(
            args.seed
        ),
        "ci_level": float(
            args.ci_level
        ),
        "bootstrap_unit": "source_case_id",
        "primary_consequence": PRIMARY_CONSEQUENCE,
        "primary_policy": PRIMARY_POLICY,
        "policies": list(
            POLICIES
        ),
        "paired_ablations": [
            {
                "left": left,
                "right": right,
                "label": label,
            }
            for left, right, label in ABLATIONS
        ],
        "regime_populations": regime_meta,
        "frozen_reference_parameters": params,
        "training_performed": False,
        "recalibration_performed": False,
        "gate_modified": False,
        "policy_retuned": False,
        "new_operating_point_selected": False,
        "loss_metrics": {
            "mean_realized_normalized_loss_09b2n": (
                "Preserves Step 09b.2n top-level GATHER=c_G accounting."
            ),
            "mean_end_to_end_realized_loss_diagnostic": (
                "Adds realized post-GATHER continuation loss to c_G. "
                "Diagnostic only; does not overwrite frozen Step 09b.2n results."
            ),
        },
        "ci_method": (
            "Percentile bootstrap confidence intervals over source-case cluster "
            "resamples."
        ),
        "important_interpretation": (
            "Confidence intervals quantify sampling uncertainty of the frozen "
            "analysis. They do not justify changing thresholds, gate rules, or "
            "selecting new policy operating points."
        ),
        "next_decision": (
            "Use paired bootstrap intervals to determine which RCSE v2 effects "
            "are statistically stable, then lock Step 09 results and move to "
            "publication tables/figures unless a genuine implementation defect "
            "is identified."
        ),
    }

    with open(
        out_dir
        / "rcse_v2r_bootstrap_metadata.json",
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
        "STEP 09b.2r COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nRCSE_V2_FULL primary metric confidence intervals:"
    )

    focus_metrics = [
        "autonomous_execution_coverage",
        "unsafe_execution_rate",
        "raw_exposure_weighted_unsafe_rate",
        "mean_realized_normalized_loss_09b2n",
        "mean_end_to_end_realized_loss_diagnostic",
        "human_escalation_rate",
        "gather_rate",
    ]

    focus = policy_ci[
        (
            policy_ci[
                "policy"
            ]
            == PRIMARY_POLICY
        )
        & (
            policy_ci[
                "metric"
            ].isin(
                focus_metrics
            )
        )
    ]

    print(
        focus[
            [
                "regime",
                "metric",
                "point_estimate",
                "bootstrap_mean",
                "bootstrap_se",
                "ci_lower",
                "ci_upper",
                "bootstrap_valid_reps",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRCSE_V2_FULL diagnostic confidence intervals:"
    )

    print(
        diag_ci[
            [
                "regime",
                "metric",
                "point_estimate",
                "bootstrap_mean",
                "bootstrap_se",
                "ci_lower",
                "ci_upper",
                "bootstrap_valid_reps",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nKey paired ablation confidence intervals:"
    )

    key_metrics = [
        "unsafe_execution_rate",
        "autonomous_execution_coverage",
        "mean_realized_normalized_loss_09b2n",
        "mean_end_to_end_realized_loss_diagnostic",
    ]

    ab_focus = ablation_ci[
        ablation_ci[
            "metric"
        ].isin(
            key_metrics
        )
    ]

    print(
        ab_focus[
            [
                "regime",
                "comparison",
                "metric",
                "bootstrap_mean",
                "bootstrap_se",
                "ci_lower",
                "ci_upper",
                "probability_delta_below_zero",
                "probability_delta_above_zero",
            ]
        ].to_string(
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
        "\nNext: interpret statistical stability of the frozen RCSE v2 effects. "
        "Do not modify the policy based on these confidence intervals."
    )


if __name__ == "__main__":
    main()
