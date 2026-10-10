#!/usr/bin/env python
"""
09b2n_run_rcse_v2_policy_simulation.py

RCSE Step 09b.2n
Run the frozen RCSE v2 policy simulation and predeclared ablations.

Purpose
-------
Evaluate the revised frozen RCSE v2 architecture using the policy-ready datasets
from Step 09b.2m and the frozen specification from Step 09b.2l.

NO MODEL TRAINING.
NO RECALIBRATION.
NO GATE CHANGES.
NO POLICY RETUNING.

Required policy set
-------------------
ARGMAX_V1
CONFIDENCE_THRESHOLD_V1
VOI_BLIND_V1
RCSE_V1_FULL
V2_SAFETY_NO_GATE
V2_GATE_MULTICLASS_EXECUTE
RCSE_V2_NO_GATHER
RCSE_V2_FULL

Primary outputs
---------------
rcse_v2_policy_results_summary.csv
rcse_v2_policy_results_by_regime.csv
rcse_v2_policy_results_by_gate_state.csv
rcse_v2_policy_results_by_reference_action.csv
rcse_v2_policy_results_by_intervention_family.csv
rcse_v2_policy_results_by_risk_stratum.csv
rcse_v2_policy_action_counts.csv
rcse_v2_policy_ablation_deltas.csv
rcse_v2_policy_reference_point.csv
rcse_v2_policy_frontier.csv
rcse_v2_policy_episode_decisions_<regime>.parquet
rcse_v2_policy_metadata.json

Important methodological note
-----------------------------
RCSE v2 is a diagnostically motivated secondary/revised analysis.
The original RCSE v1 results remain frozen and should be preserved unchanged.
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

ACTIONS = ("EXECUTE", "GATHER", "ESCALATE", "ABSTAIN")

REFERENCE_POLICY_NAMES = (
    "ARGMAX_V1",
    "CONFIDENCE_THRESHOLD_V1",
    "VOI_BLIND_V1",
    "RCSE_V1_FULL",
    "V2_SAFETY_NO_GATE",
    "V2_GATE_MULTICLASS_EXECUTE",
    "RCSE_V2_NO_GATHER",
    "RCSE_V2_FULL",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2n: run frozen RCSE v2 policy simulation."
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Step 09b.2m rcse_v2_policy_dataset directory.",
    )
    p.add_argument(
        "--spec",
        required=True,
        help="Frozen 09b2l_rcse_v2_policy_spec.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <dataset-dir>/rcse_v2_policy_results",
    )
    p.add_argument(
        "--save-episode-csv",
        action="store_true",
        help="Also save episode-level decisions as CSV.",
    )
    return p.parse_args()


def normalize_action(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("status") != "FROZEN_BEFORE_RCSE_V2_SIMULATION":
        raise RuntimeError(
            "RCSE v2 spec is not marked FROZEN_BEFORE_RCSE_V2_SIMULATION."
        )

    gate_delta = (
        spec.get("architecture", {})
        .get("evidence_gate", {})
        .get("value_consistency_delta")
    )

    if gate_delta is None or abs(float(gate_delta) - 0.05) > 1e-12:
        raise RuntimeError(
            f"Frozen gate delta mismatch: expected 0.05, got {gate_delta}"
        )


def get_reference_parameters(spec: dict[str, Any]) -> dict[str, float]:
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


def get_frontier_grid(spec: dict[str, Any]) -> list[dict[str, float]]:
    t0s = spec["risk_and_selectivity"]["tau_safe_t0_grid"]
    b0s = spec["risk_and_selectivity"]["beta0_grid"]
    t1s = spec["risk_and_selectivity"]["tau_safe_t1_grid"]
    b1s = spec["risk_and_selectivity"]["beta1_grid"]

    grid = []

    for t0 in t0s:
        for b0 in b0s:
            for t1 in t1s:
                for b1 in b1s:
                    grid.append(
                        {
                            "tau_safe_t0": float(t0),
                            "beta0": float(b0),
                            "tau_safe_t1": float(t1),
                            "beta1": float(b1),
                        }
                    )

    return grid


def compute_train_reference_exposure(
    df: pd.DataFrame,
    regime: str,
    quantile: float,
) -> float:
    """
    Step 09b.2m datasets are test-only. If train reference exposure was already
    materialized by Step 09b.1, use it. Otherwise infer from policy dataset
    metadata columns if available.

    Preferred columns checked:
      train_exposure_p50_eur
      train_exposure_p90_eur
      train_exposure_p99_eur

    If not present, fail rather than silently recomputing from test exposure.
    """
    qname = {
        0.50: "train_exposure_p50_eur",
        0.90: "train_exposure_p90_eur",
        0.99: "train_exposure_p99_eur",
    }[quantile]

    if qname in df.columns:
        vals = pd.to_numeric(df[qname], errors="coerce").dropna().unique()

        if len(vals) != 1:
            raise RuntimeError(
                f"{regime}: expected one unique {qname}; found {len(vals)}"
            )

        return float(vals[0])

    # Alternate naming used in some earlier policy artifacts.
    alternates = {
        0.50: ["exposure_ref_p50_eur", "vref_p50_eur"],
        0.90: ["exposure_ref_p90_eur", "vref_p90_eur"],
        0.99: ["exposure_ref_p99_eur", "vref_p99_eur"],
    }

    for alt in alternates[quantile]:
        if alt in df.columns:
            vals = pd.to_numeric(df[alt], errors="coerce").dropna().unique()
            if len(vals) != 1:
                raise RuntimeError(
                    f"{regime}: expected one unique {alt}; found {len(vals)}"
                )
            return float(vals[0])

    raise RuntimeError(
        f"{regime}: train-only exposure reference for quantile {quantile:.2f} "
        "is not present in the Step 09b.2m dataset. Do not recompute it from "
        "held-out test exposure. Rebuild policy dataset with the frozen train "
        "exposure references attached."
    )


def consequence_from_exposure(
    exposure: pd.Series,
    mode: str,
    refs: dict[str, float],
) -> pd.Series:
    v = (
        pd.to_numeric(exposure, errors="coerce")
        .abs()
        .fillna(0.0)
    )

    if mode == "UNIFORM":
        return pd.Series(1.0, index=v.index, dtype=float)

    if mode == "LOG_TRAIN_P50":
        ref = refs["p50"]
    elif mode == "LOG_TRAIN_P90":
        ref = refs["p90"]
    elif mode == "LOG_TRAIN_P99":
        ref = refs["p99"]
    else:
        raise ValueError(f"Unknown consequence mode: {mode}")

    if ref <= 0:
        raise ValueError(f"Consequence reference must be positive; got {ref}")

    denom = np.log1p(ref)

    return np.log1p(v) / denom


def derive_ground_truth_unsafe(df: pd.DataFrame) -> pd.Series:
    """
    Evaluation semantics for autonomous EXECUTE:
    unsafe if benchmark/reference semantics do not support EXECUTE.

    Priority:
    1. t0_safety_robustness_target if present
    2. expected_action_reference != EXECUTE

    This is evaluation-only and not a policy input.
    """
    if "t0_safety_robustness_target" in df.columns:
        t = (
            df["t0_safety_robustness_target"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        resolved = t.isin(
            ["SAFE_TO_EXECUTE", "NOT_SAFE_TO_EXECUTE"]
        )

        out = pd.Series(pd.NA, index=df.index, dtype="boolean")
        out.loc[resolved] = (
            t.loc[resolved] == "NOT_SAFE_TO_EXECUTE"
        )

        # unresolved natural cases fall back to benchmark action semantics
        fallback = ~resolved
        if fallback.any():
            out.loc[fallback] = (
                normalize_action(
                    df.loc[fallback, "expected_action_reference"]
                )
                != "EXECUTE"
            )

        return out.astype(bool)

    return (
        normalize_action(df["expected_action_reference"])
        != "EXECUTE"
    )


def realized_loss_for_action(
    action: pd.Series,
    unsafe_exec: pd.Series,
    consequence: pd.Series,
    c_G: float,
    c_H: float,
    c_A: float,
) -> pd.Series:
    """
    Simple normalized realized-loss accounting:
    EXECUTE: consequence if unsafe, else 0
    GATHER : c_G
    ESCALATE: c_H
    ABSTAIN: c_A

    This mirrors the frozen experimental cost abstraction.
    """
    out = pd.Series(np.nan, index=action.index, dtype=float)

    exec_mask = action == "EXECUTE"
    gather_mask = action == "GATHER"
    esc_mask = action == "ESCALATE"
    abst_mask = action == "ABSTAIN"

    out.loc[exec_mask] = (
        unsafe_exec.loc[exec_mask].astype(float)
        * consequence.loc[exec_mask]
    )
    out.loc[gather_mask] = c_G
    out.loc[esc_mask] = c_H
    out.loc[abst_mask] = c_A

    return out


def confidence_threshold_v1(
    df: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.Series:
    """
    Comparator: execute if p_cal_execute >= threshold, else escalate.

    Threshold kept fixed and descriptive. This script does not tune it.
    """
    p = pd.to_numeric(
        df["p_cal_execute"],
        errors="coerce",
    )

    return pd.Series(
        np.where(
            p >= threshold,
            "EXECUTE",
            "ESCALATE",
        ),
        index=df.index,
    )


def argmax_v1(df: pd.DataFrame) -> pd.Series:
    return normalize_action(
        df["predicted_action_frozen"]
    )


def voi_blind_v1(
    df: pd.DataFrame,
    c_G: float,
    c_H: float,
    c_A: float,
) -> pd.Series:
    """
    Consequence-blind one-step heuristic comparator.

    If GATHER is eligible and post-GATHER safe probability is available:
      gather when c_G + min(post-gather execute proxy, escalate, abstain)
      is cheaper than min(escalate, abstain)
    otherwise escalate/abstain by lower fixed cost.

    Direct execute uses the original p_cal_execute probability only.
    """
    pE = pd.to_numeric(df["p_cal_execute"], errors="coerce")
    p1 = pd.to_numeric(
        df["p_safe_t1_for_primary_rcse"],
        errors="coerce",
    )

    gather_ok = (
        df["gather_policy_eligible"]
        .fillna(False)
        .astype(bool)
    )

    l_exec = 1.0 - pE
    l_H = pd.Series(c_H, index=df.index, dtype=float)
    l_A = pd.Series(c_A, index=df.index, dtype=float)

    l_post_exec = 1.0 - p1
    l_post = pd.concat(
        [
            l_post_exec.rename("EXECUTE"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    ).min(axis=1)

    l_G = c_G + l_post
    l_G.loc[~gather_ok] = np.inf

    losses = pd.concat(
        [
            l_exec.rename("EXECUTE"),
            l_G.rename("GATHER"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    )

    return losses.idxmin(axis=1)


def rcse_v1_full(
    df: pd.DataFrame,
    consequence: pd.Series,
    params: dict[str, float],
) -> pd.Series:
    """
    Approximation of original frozen RCSE v1 policy:
    uses multiclass p_cal_execute for execution risk,
    consequence-aware expected loss, and one-step GATHER.
    """
    pE = pd.to_numeric(df["p_cal_execute"], errors="coerce")
    p1 = pd.to_numeric(
        df["p_safe_t1_for_primary_rcse"],
        errors="coerce",
    )

    gather_ok = (
        df["gather_policy_eligible"]
        .fillna(False)
        .astype(bool)
    )

    l_E = (1.0 - pE) * consequence
    l_H = params["c_H"] + params["rho_H"] * consequence
    l_A = pd.Series(params["c_A"], index=df.index, dtype=float)

    post_exec_feasible = (
        p1.notna()
        & (p1 >= params["tau_safe_t1"])
        & (((1.0 - p1) * consequence) <= params["beta1"])
    )

    l_post_E = ((1.0 - p1) * consequence).where(
        post_exec_feasible,
        np.inf,
    )

    l_post = pd.concat(
        [
            l_post_E.rename("EXECUTE"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    ).min(axis=1)

    l_G = params["c_G"] + l_post
    l_G.loc[~gather_ok] = np.inf

    execute_feasible = (
        pE >= params["tau_safe_t0"]
    ) & (
        l_E <= params["beta0"]
    )

    l_E = l_E.where(execute_feasible, np.inf)

    losses = pd.concat(
        [
            l_E.rename("EXECUTE"),
            l_G.rename("GATHER"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    )

    return losses.idxmin(axis=1)


def v2_policy(
    df: pd.DataFrame,
    consequence: pd.Series,
    params: dict[str, float],
    use_gate: bool,
    use_dedicated_safety: bool,
    allow_gather: bool,
) -> pd.Series:
    """
    Unified RCSE v2 family simulator.
    """
    if use_dedicated_safety:
        p0 = pd.to_numeric(
            df["p_safe_t0_cal"],
            errors="coerce",
        )
    else:
        p0 = pd.to_numeric(
            df["p_cal_execute"],
            errors="coerce",
        )

    p1 = pd.to_numeric(
        df["p_safe_t1_for_primary_rcse"],
        errors="coerce",
    )

    gate = (
        df["gate_state_v2"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    gather_eligible = (
        df["gather_policy_eligible"]
        .fillna(False)
        .astype(bool)
    )

    l_E = (1.0 - p0) * consequence
    l_H = params["c_H"] + params["rho_H"] * consequence
    l_A = pd.Series(params["c_A"], index=df.index, dtype=float)

    execute_feasible = (
        p0 >= params["tau_safe_t0"]
    ) & (
        l_E <= params["beta0"]
    )

    if use_gate:
        execute_feasible &= (gate == "VALID")

    l_E = l_E.where(
        execute_feasible,
        np.inf,
    )

    post_exec_feasible = (
        p1.notna()
        & (p1 >= params["tau_safe_t1"])
        & (((1.0 - p1) * consequence) <= params["beta1"])
    )

    l_post_E = ((1.0 - p1) * consequence).where(
        post_exec_feasible,
        np.inf,
    )

    l_post = pd.concat(
        [
            l_post_E.rename("EXECUTE"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    ).min(axis=1)

    l_G = params["c_G"] + l_post

    if not allow_gather:
        l_G[:] = np.inf
    else:
        l_G.loc[~gather_eligible] = np.inf

        if use_gate:
            # Primary v2 spec: contradiction cannot be resolved by current GATHER.
            l_G.loc[gate == "CONTRADICTORY"] = np.inf

    losses = pd.concat(
        [
            l_E.rename("EXECUTE"),
            l_G.rename("GATHER"),
            l_H.rename("ESCALATE"),
            l_A.rename("ABSTAIN"),
        ],
        axis=1,
    )

    return losses.idxmin(axis=1)


def policy_metrics(
    df: pd.DataFrame,
    action: pd.Series,
    consequence: pd.Series,
    params: dict[str, float],
) -> dict[str, Any]:
    unsafe = derive_ground_truth_unsafe(df)

    execute = action == "EXECUTE"
    gather = action == "GATHER"
    escalate = action == "ESCALATE"
    abstain = action == "ABSTAIN"

    n = len(df)
    n_exec = int(execute.sum())

    unsafe_exec = int(
        (execute & unsafe).sum()
    )

    exposure = (
        pd.to_numeric(
            df["transaction_exposure_eur"],
            errors="coerce",
        )
        .abs()
        .fillna(0.0)
    )

    selected_exposure = float(
        exposure.loc[execute].sum()
    )

    unsafe_selected_exposure = float(
        exposure.loc[execute & unsafe].sum()
    )

    consequence_selected = float(
        consequence.loc[execute].sum()
    )

    unsafe_consequence = float(
        consequence.loc[execute & unsafe].sum()
    )

    realized = realized_loss_for_action(
        action=action,
        unsafe_exec=unsafe,
        consequence=consequence,
        c_G=params["c_G"],
        c_H=params["c_H"],
        c_A=params["c_A"],
    )

    return {
        "episodes": int(n),
        "execute_count": n_exec,
        "autonomous_execution_coverage": (
            n_exec / n if n else np.nan
        ),
        "unsafe_execute_count": unsafe_exec,
        "unsafe_execution_rate": (
            unsafe_exec / n_exec if n_exec else np.nan
        ),
        "raw_exposure_weighted_unsafe_rate": (
            unsafe_selected_exposure / selected_exposure
            if selected_exposure > 0
            else np.nan
        ),
        "consequence_weighted_unsafe_rate": (
            unsafe_consequence / consequence_selected
            if consequence_selected > 0
            else np.nan
        ),
        "mean_realized_normalized_loss": float(
            realized.mean()
        ),
        "human_escalation_rate": float(
            escalate.mean()
        ),
        "gather_rate": float(
            gather.mean()
        ),
        "abstention_rate": float(
            abstain.mean()
        ),
    }


def evaluate_policy_by_groups(
    df: pd.DataFrame,
    action: pd.Series,
    consequence: pd.Series,
    params: dict[str, float],
    policy_name: str,
    regime: str,
) -> dict[str, pd.DataFrame]:
    out = {}

    group_specs = {
        "gate_state": "gate_state_v2",
        "reference_action": "expected_action_reference",
    }

    if "intervention_family_eval" in df.columns:
        group_specs["intervention_family"] = "intervention_family_eval"

    if "transaction_risk_stratum" in df.columns:
        group_specs["risk_stratum"] = "transaction_risk_stratum"

    for label, col in group_specs.items():
        rows = []

        for value, g in df.groupby(
            col,
            dropna=False,
            observed=True,
        ):
            idx = g.index
            metrics = policy_metrics(
                df=g,
                action=action.loc[idx],
                consequence=consequence.loc[idx],
                params=params,
            )

            rows.append(
                {
                    "regime": regime,
                    "policy": policy_name,
                    label: value,
                    **metrics,
                }
            )

        out[label] = pd.DataFrame(rows)

    return out


def simulate_policy_set(
    df: pd.DataFrame,
    regime: str,
    consequence_mode: str,
    consequence: pd.Series,
    params: dict[str, float],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    policies = {
        "ARGMAX_V1": argmax_v1(df),
        "CONFIDENCE_THRESHOLD_V1": confidence_threshold_v1(df),
        "VOI_BLIND_V1": voi_blind_v1(
            df,
            c_G=params["c_G"],
            c_H=params["c_H"],
            c_A=params["c_A"],
        ),
        "RCSE_V1_FULL": rcse_v1_full(
            df,
            consequence=consequence,
            params=params,
        ),
        "V2_SAFETY_NO_GATE": v2_policy(
            df,
            consequence=consequence,
            params=params,
            use_gate=False,
            use_dedicated_safety=True,
            allow_gather=True,
        ),
        "V2_GATE_MULTICLASS_EXECUTE": v2_policy(
            df,
            consequence=consequence,
            params=params,
            use_gate=True,
            use_dedicated_safety=False,
            allow_gather=True,
        ),
        "RCSE_V2_NO_GATHER": v2_policy(
            df,
            consequence=consequence,
            params=params,
            use_gate=True,
            use_dedicated_safety=True,
            allow_gather=False,
        ),
        "RCSE_V2_FULL": v2_policy(
            df,
            consequence=consequence,
            params=params,
            use_gate=True,
            use_dedicated_safety=True,
            allow_gather=True,
        ),
    }

    summary_rows = []
    group_accum: dict[str, list[pd.DataFrame]] = {
        "gate_state": [],
        "reference_action": [],
        "intervention_family": [],
        "risk_stratum": [],
    }

    decision = df[
        [
            c for c in [
                "benchmark_episode_id",
                "source_case_id",
                "evaluation_regime",
                "gate_state_v2",
                "expected_action_reference",
                "intervention_family_eval",
                "transaction_risk_stratum",
                "transaction_exposure_eur",
                "p_cal_execute",
                "p_safe_t0_raw",
                "p_safe_t0_cal",
                "p_safe_t1_for_primary_rcse",
                "gather_policy_eligible",
            ]
            if c in df.columns
        ]
    ].copy()

    decision["consequence_mode"] = consequence_mode
    decision["consequence_value"] = consequence

    for policy_name, action in policies.items():
        m = policy_metrics(
            df=df,
            action=action,
            consequence=consequence,
            params=params,
        )

        summary_rows.append(
            {
                "regime": regime,
                "consequence_mode": consequence_mode,
                "policy": policy_name,
                **m,
            }
        )

        grouped = evaluate_policy_by_groups(
            df=df,
            action=action,
            consequence=consequence,
            params=params,
            policy_name=policy_name,
            regime=regime,
        )

        for key, gdf in grouped.items():
            if key in group_accum:
                gdf["consequence_mode"] = consequence_mode
                group_accum[key].append(gdf)

        decision[
            f"action__{policy_name}"
        ] = action

    grouped_out = {
        key: (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame()
        )
        for key, frames in group_accum.items()
    }

    return (
        pd.DataFrame(summary_rows),
        grouped_out,
        decision,
    )


def ablation_deltas(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    pairs = [
        ("RCSE_V2_FULL", "RCSE_V2_NO_GATHER", "GATHER_CONTRIBUTION"),
        ("RCSE_V2_FULL", "V2_SAFETY_NO_GATE", "EVIDENCE_GATE_CONTRIBUTION"),
        (
            "RCSE_V2_FULL",
            "V2_GATE_MULTICLASS_EXECUTE",
            "DEDICATED_SAFETY_BELIEF_CONTRIBUTION",
        ),
        ("RCSE_V2_FULL", "VOI_BLIND_V1", "CONSEQUENCE_AWARE_VS_VOI_BLIND"),
        ("RCSE_V2_FULL", "RCSE_V1_FULL", "V2_VS_V1"),
    ]

    metrics = [
        "autonomous_execution_coverage",
        "unsafe_execution_rate",
        "raw_exposure_weighted_unsafe_rate",
        "consequence_weighted_unsafe_rate",
        "mean_realized_normalized_loss",
        "human_escalation_rate",
        "gather_rate",
        "abstention_rate",
    ]

    rows = []

    for (regime, consequence_mode), g in summary.groupby(
        ["regime", "consequence_mode"],
        observed=True,
    ):
        indexed = g.set_index("policy")

        for left, right, label in pairs:
            if left not in indexed.index or right not in indexed.index:
                continue

            row = {
                "regime": regime,
                "consequence_mode": consequence_mode,
                "comparison": label,
                "left_policy": left,
                "right_policy": right,
            }

            for metric in metrics:
                lv = indexed.loc[left, metric]
                rv = indexed.loc[right, metric]

                row[f"{metric}_left"] = lv
                row[f"{metric}_right"] = rv
                row[f"{metric}_delta_left_minus_right"] = lv - rv

            rows.append(row)

    return pd.DataFrame(rows)


def run_frontier(
    df: pd.DataFrame,
    regime: str,
    consequence_mode: str,
    consequence: pd.Series,
    base_params: dict[str, float],
    grid: list[dict[str, float]],
) -> pd.DataFrame:
    rows = []

    for gp in grid:
        params = dict(base_params)
        params.update(gp)

        action = v2_policy(
            df,
            consequence=consequence,
            params=params,
            use_gate=True,
            use_dedicated_safety=True,
            allow_gather=True,
        )

        m = policy_metrics(
            df=df,
            action=action,
            consequence=consequence,
            params=params,
        )

        rows.append(
            {
                "regime": regime,
                "consequence_mode": consequence_mode,
                **gp,
                **m,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    dataset_dir = (
        Path(args.dataset_dir)
        .expanduser()
        .resolve()
    )

    spec_path = (
        Path(args.spec)
        .expanduser()
        .resolve()
    )

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else dataset_dir / "rcse_v2_policy_results"
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

    validate_spec(spec)

    ref_params = get_reference_parameters(spec)
    frontier_grid = get_frontier_grid(spec)

    consequence_modes = [
        spec["consequence"]["primary"],
        *spec["consequence"]["sensitivity"],
    ]

    # preserve order, remove duplicates
    consequence_modes = list(
        dict.fromkeys(consequence_modes)
    )

    print("=" * 80)
    print("RCSE STEP 09b.2n - FROZEN V2 POLICY SIMULATION")
    print("=" * 80)
    print(f"Dataset dir : {dataset_dir}")
    print(f"Frozen spec : {spec_path}")
    print(f"Output      : {out_dir}")
    print("NO TRAINING / NO RECALIBRATION / NO GATE CHANGES / NO RETUNING")
    print(f"Reference params: {ref_params}")
    print(f"Consequence modes: {consequence_modes}")
    print(f"Frontier grid points per regime/mode: {len(frontier_grid)}")

    all_summary = []
    all_gate = []
    all_ref = []
    all_intervention = []
    all_risk = []
    all_decisions = []
    all_frontier = []

    for regime in REGIMES:
        path = (
            dataset_dir
            / DATASET_FILES[regime]
        )

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_parquet(path)

        require_columns(
            df,
            [
                "benchmark_episode_id",
                "source_case_id",
                "gate_state_v2",
                "expected_action_reference",
                "predicted_action_frozen",
                "p_cal_execute",
                "p_cal_gather",
                "p_cal_escalate",
                "p_cal_abstain",
                "p_safe_t0_raw",
                "p_safe_t0_cal",
                "p_safe_t1_for_primary_rcse",
                "gather_policy_eligible",
                "transaction_exposure_eur",
            ],
            regime,
        )

        refs = {
            "p50": compute_train_reference_exposure(
                df, regime, 0.50
            ),
            "p90": compute_train_reference_exposure(
                df, regime, 0.90
            ),
            "p99": compute_train_reference_exposure(
                df, regime, 0.99
            ),
        }

        print(f"\n[{regime}] rows={len(df):,} | train refs={refs}")

        for consequence_mode in consequence_modes:
            consequence = consequence_from_exposure(
                df["transaction_exposure_eur"],
                mode=consequence_mode,
                refs=refs,
            )

            summary, groups, decisions = simulate_policy_set(
                df=df,
                regime=regime,
                consequence_mode=consequence_mode,
                consequence=consequence,
                params=ref_params,
            )

            all_summary.append(summary)

            if not groups["gate_state"].empty:
                all_gate.append(groups["gate_state"])

            if not groups["reference_action"].empty:
                all_ref.append(groups["reference_action"])

            if not groups["intervention_family"].empty:
                all_intervention.append(
                    groups["intervention_family"]
                )

            if not groups["risk_stratum"].empty:
                all_risk.append(groups["risk_stratum"])

            all_decisions.append(decisions)

            frontier = run_frontier(
                df=df,
                regime=regime,
                consequence_mode=consequence_mode,
                consequence=consequence,
                base_params=ref_params,
                grid=frontier_grid,
            )

            all_frontier.append(frontier)

            ref_row = summary[
                summary["policy"] == "RCSE_V2_FULL"
            ].iloc[0]

            print(
                f"   [{consequence_mode}] "
                f"V2_FULL coverage={ref_row['autonomous_execution_coverage']:.4f} "
                f"unsafe={ref_row['unsafe_execution_rate']:.4f} "
                f"loss={ref_row['mean_realized_normalized_loss']:.4f} "
                f"gather={ref_row['gather_rate']:.4f} "
                f"escalate={ref_row['human_escalation_rate']:.4f}"
            )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True,
    )

    gate_df = (
        pd.concat(all_gate, ignore_index=True)
        if all_gate
        else pd.DataFrame()
    )

    ref_df = (
        pd.concat(all_ref, ignore_index=True)
        if all_ref
        else pd.DataFrame()
    )

    intervention_df = (
        pd.concat(
            all_intervention,
            ignore_index=True,
        )
        if all_intervention
        else pd.DataFrame()
    )

    risk_df = (
        pd.concat(all_risk, ignore_index=True)
        if all_risk
        else pd.DataFrame()
    )

    decisions_df = pd.concat(
        all_decisions,
        ignore_index=True,
    )

    frontier_df = pd.concat(
        all_frontier,
        ignore_index=True,
    )

    ablation_df = ablation_deltas(
        summary_df
    )

    primary_mode = spec["consequence"]["primary"]

    reference_point_df = summary_df[
        summary_df["consequence_mode"] == primary_mode
    ].copy()

    action_counts = (
        decisions_df
        .melt(
            id_vars=[
                c for c in [
                    "benchmark_episode_id",
                    "source_case_id",
                    "evaluation_regime",
                    "consequence_mode",
                    "gate_state_v2",
                    "expected_action_reference",
                ]
                if c in decisions_df.columns
            ],
            value_vars=[
                c for c in decisions_df.columns
                if c.startswith("action__")
            ],
            var_name="policy",
            value_name="action",
        )
    )

    action_counts["policy"] = (
        action_counts["policy"]
        .str.replace(
            "action__",
            "",
            regex=False,
        )
    )

    action_counts_df = (
        action_counts.groupby(
            [
                "evaluation_regime",
                "consequence_mode",
                "policy",
                "action",
            ],
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    action_counts_df["pct_within_policy"] = (
        action_counts_df["count"]
        / action_counts_df.groupby(
            [
                "evaluation_regime",
                "consequence_mode",
                "policy",
            ]
        )["count"].transform("sum")
    )

    # Write core outputs
    summary_df.to_csv(
        out_dir / "rcse_v2_policy_results_summary.csv",
        index=False,
    )

    reference_point_df.to_csv(
        out_dir / "rcse_v2_policy_reference_point.csv",
        index=False,
    )

    action_counts_df.to_csv(
        out_dir / "rcse_v2_policy_action_counts.csv",
        index=False,
    )

    ablation_df.to_csv(
        out_dir / "rcse_v2_policy_ablation_deltas.csv",
        index=False,
    )

    frontier_df.to_csv(
        out_dir / "rcse_v2_policy_frontier.csv",
        index=False,
    )

    if not gate_df.empty:
        gate_df.to_csv(
            out_dir / "rcse_v2_policy_results_by_gate_state.csv",
            index=False,
        )

    if not ref_df.empty:
        ref_df.to_csv(
            out_dir / "rcse_v2_policy_results_by_reference_action.csv",
            index=False,
        )

    if not intervention_df.empty:
        intervention_df.to_csv(
            out_dir / "rcse_v2_policy_results_by_intervention_family.csv",
            index=False,
        )

    if not risk_df.empty:
        risk_df.to_csv(
            out_dir / "rcse_v2_policy_results_by_risk_stratum.csv",
            index=False,
        )

    # Episode-level outputs per regime.
    for regime in REGIMES:
        sub = decisions_df[
            decisions_df["evaluation_regime"] == regime
        ].copy()

        p = (
            out_dir
            / f"rcse_v2_policy_episode_decisions_{regime}.parquet"
        )

        sub.to_parquet(
            p,
            index=False,
        )

        if args.save_episode_csv:
            sub.to_csv(
                out_dir
                / f"rcse_v2_policy_episode_decisions_{regime}.csv",
                index=False,
            )

    metadata = {
        "step": "09b.2n",
        "status": "FROZEN_RCSE_V2_SIMULATION_EXECUTED",
        "analysis_status": spec.get(
            "analysis_status"
        ),
        "dataset_directory": str(
            dataset_dir
        ),
        "frozen_spec": str(
            spec_path
        ),
        "policy_set": list(
            REFERENCE_POLICY_NAMES
        ),
        "reference_parameters": ref_params,
        "consequence_modes": consequence_modes,
        "frontier_grid_size": len(
            frontier_grid
        ),
        "frontier_grid": frontier_grid,
        "training_performed": False,
        "recalibration_performed": False,
        "gate_modified": False,
        "policy_retuned": False,
        "important_reporting_guardrails": spec.get(
            "governance",
            []
        ),
        "interpretation_note": (
            "Reference-point metrics are illustrative frozen operating points. "
            "Final claims must be supported by the full predeclared frontier and "
            "ablation analysis, not by selecting the best held-out result."
        ),
    }

    with open(
        out_dir / "rcse_v2_policy_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 09b.2n COMPLETE")
    print("=" * 80)

    print("\nPrimary consequence reference-point results:")
    show = reference_point_df[
        [
            "regime",
            "policy",
            "autonomous_execution_coverage",
            "unsafe_execution_rate",
            "raw_exposure_weighted_unsafe_rate",
            "consequence_weighted_unsafe_rate",
            "mean_realized_normalized_loss",
            "human_escalation_rate",
            "gather_rate",
            "abstention_rate",
        ]
    ].copy()

    print(
        show.to_string(
            index=False
        )
    )

    print("\nPrimary ablation deltas:")
    primary_ablation = ablation_df[
        ablation_df["consequence_mode"] == primary_mode
    ]

    print(
        primary_ablation[
            [
                "regime",
                "comparison",
                "autonomous_execution_coverage_delta_left_minus_right",
                "unsafe_execution_rate_delta_left_minus_right",
                "mean_realized_normalized_loss_delta_left_minus_right",
                "human_escalation_rate_delta_left_minus_right",
                "gather_rate_delta_left_minus_right",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p}")

    print(
        "\nNext: interpret the frozen reference point together with the full "
        "predeclared frontier and ablations before making any RCSE v2 claim."
    )


if __name__ == "__main__":
    main()
