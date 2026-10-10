"""
09b2b_run_rcse_policy_simulation.py

Run the frozen comparator + RCSE policy simulation defined in Step 09b.2a.

Inputs
------
--policy-dir
    Step 09b.1 rcse_policy_dataset directory containing:
        rcse_policy_dataset_grouped_iid.parquet
        rcse_policy_dataset_temporal.parquet
        rcse_policy_dataset_ood_exposure.parquet

--split-benchmark
    rcse_experimental_with_splits.parquet
    Used ONLY to derive TRAIN-only exposure normalization statistics.

--spec
    09b2a_rcse_policy_spec.json
    The frozen policy specification. The simulator reads this file rather than
    silently hard-coding alternative thresholds/costs.

Outputs
-------
rcse_policy_reference_results.csv
rcse_confidence_threshold_sweep.csv
rcse_risk_budget_frontier.csv
rcse_post_gather_gate_sensitivity.csv
rcse_cost_sensitivity.csv
rcse_consequence_sensitivity.csv
rcse_all_policy_results.csv
rcse_action_distribution.csv
rcse_train_exposure_reference.csv
rcse_policy_simulation_metadata.json

Reference action-detail Parquets:
rcse_reference_actions_<regime>.parquet

Evaluation logic
----------------
Primary episode-level final outcomes:

- Direct t0 EXECUTE:
    unsafe iff expected_action_reference != EXECUTE

- GATHER -> post-t1 EXECUTE:
    unsafe iff y_t1_d1 != SAFE_TO_EXECUTE

- ESCALATE:
    incurs c_H + rho_H * C(v)

- ABSTAIN:
    incurs c_A

- GATHER:
    incurs c_G plus downstream realized loss

Autonomous execution coverage includes both direct t0 EXECUTE and
post-GATHER EXECUTE.

Important
---------
The post-GATHER q_t1 score is an auxiliary selective score. It is NOT assumed
to be universally calibrated under OOD.

The simulator reports both:
1. model-based expected policy loss used by the decision rule, and
2. realized normalized benchmark loss computed from held-out reference/outcome
   labels.

No test-set result is used to alter the frozen grids.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = ["grouped_iid", "temporal", "ood_exposure"]

POLICY_FILES = {
    "grouped_iid": "rcse_policy_dataset_grouped_iid.parquet",
    "temporal": "rcse_policy_dataset_temporal.parquet",
    "ood_exposure": "rcse_policy_dataset_ood_exposure.parquet",
}

SPLIT_COLUMNS = {
    "grouped_iid": "split_grouped_iid",
    "temporal": "split_temporal",
    "ood_exposure": "split_ood_exposure",
}

SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"

ACTIONS = ["EXECUTE", "GATHER", "ESCALATE", "ABSTAIN"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2b: run frozen comparator + RCSE policy simulation."
    )

    p.add_argument(
        "--policy-dir",
        required=True,
        help="Step 09b.1 rcse_policy_dataset directory.",
    )
    p.add_argument(
        "--split-benchmark",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet.",
    )
    p.add_argument(
        "--spec",
        required=True,
        help="Path to frozen 09b2a_rcse_policy_spec.json.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <policy-dir>/policy_simulation_results",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def get_train_exposure_refs(
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute TRAIN-only exposure percentiles per regime at source-case level.

    Source-case-level deduplication prevents experimental variants of the same
    transaction from overweighting percentile estimation.
    """
    require_columns(
        split_df,
        [
            "source_case_id",
            "transaction_exposure_eur",
            *SPLIT_COLUMNS.values(),
        ],
        "split benchmark",
    )

    work = split_df[
        [
            "source_case_id",
            "transaction_exposure_eur",
            *SPLIT_COLUMNS.values(),
        ]
    ].copy()

    work["source_case_id"] = work["source_case_id"].astype(str)
    work["transaction_exposure_eur"] = pd.to_numeric(
        work["transaction_exposure_eur"],
        errors="coerce",
    ).abs()

    rows = []

    for regime in REGIMES:
        split_col = SPLIT_COLUMNS[regime]

        train = work.loc[
            work[split_col] == "train",
            ["source_case_id", "transaction_exposure_eur"],
        ].copy()

        # Collapse experimental variants.
        train = (
            train.groupby("source_case_id", as_index=False)
            ["transaction_exposure_eur"]
            .first()
        )

        x = train["transaction_exposure_eur"].dropna()

        if x.empty:
            raise RuntimeError(
                f"{regime}: no TRAIN exposure values available."
            )

        rows.append(
            {
                "regime": regime,
                "train_source_cases": int(len(train)),
                "non_null_train_exposure": int(len(x)),
                "p50_exposure_eur": float(x.quantile(0.50)),
                "p90_exposure_eur": float(x.quantile(0.90)),
                "p99_exposure_eur": float(x.quantile(0.99)),
            }
        )

    return pd.DataFrame(rows)


def consequence_values(
    exposure: pd.Series,
    regime: str,
    consequence_name: str,
    exposure_refs: pd.DataFrame,
) -> np.ndarray:
    v = pd.to_numeric(
        exposure,
        errors="coerce",
    ).abs().fillna(0.0).to_numpy(dtype=float)

    if consequence_name == "UNIFORM":
        return np.ones(len(v), dtype=float)

    ref_row = exposure_refs.loc[
        exposure_refs["regime"] == regime
    ].iloc[0]

    ref_map = {
        "LOG_TRAIN_P50": float(ref_row["p50_exposure_eur"]),
        "LOG_TRAIN_P90": float(ref_row["p90_exposure_eur"]),
        "LOG_TRAIN_P99": float(ref_row["p99_exposure_eur"]),
    }

    if consequence_name not in ref_map:
        raise ValueError(
            f"Unknown consequence function: {consequence_name}"
        )

    v_ref = ref_map[consequence_name]

    if not np.isfinite(v_ref) or v_ref <= 0:
        raise ValueError(
            f"{regime}/{consequence_name}: invalid v_ref={v_ref}"
        )

    return np.log1p(v) / np.log1p(v_ref)


def post_gather_continuation(
    df: pd.DataFrame,
    C: np.ndarray,
    c_H: float,
    c_A: float,
    rho_H: float,
    tau1: float,
    beta1: float,
) -> dict[str, np.ndarray]:
    """
    Compute post-GATHER continuation expected losses/actions.

    Post-t1 EXECUTE is feasible only if:
        q_t1 >= tau1
        (1-q_t1)*C <= beta1
    """
    q = pd.to_numeric(
        df["p_safe_t1_for_primary_rcse"],
        errors="coerce",
    ).to_numpy(dtype=float)

    post_execute_loss = (1.0 - q) * C

    post_execute_feasible = (
        np.isfinite(q)
        & (q >= tau1)
        & (post_execute_loss <= beta1)
    )

    post_escalate_loss = c_H + rho_H * C
    post_abstain_loss = np.full(
        len(df),
        c_A,
        dtype=float,
    )

    # Choose minimum feasible continuation.
    stacked = np.column_stack(
        [
            np.where(
                post_execute_feasible,
                post_execute_loss,
                np.inf,
            ),
            post_escalate_loss,
            post_abstain_loss,
        ]
    )

    idx = np.argmin(stacked, axis=1)

    labels = np.array(
        ["EXECUTE", "ESCALATE", "ABSTAIN"],
        dtype=object,
    )[idx]

    min_loss = stacked[
        np.arange(len(df)),
        idx,
    ]

    return {
        "post_action": labels,
        "post_expected_loss": min_loss,
        "post_execute_loss": post_execute_loss,
        "post_execute_feasible": post_execute_feasible,
    }


def choose_min_loss_action(
    losses: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    labels = list(losses.keys())
    mat = np.column_stack(
        [losses[a] for a in labels]
    )

    idx = np.argmin(mat, axis=1)

    action = np.array(
        labels,
        dtype=object,
    )[idx]

    selected_loss = mat[
        np.arange(len(mat)),
        idx,
    ]

    return action, selected_loss


def realized_metrics(
    df: pd.DataFrame,
    primary_action: np.ndarray,
    final_action: np.ndarray,
    expected_selected_loss: np.ndarray,
    C: np.ndarray,
    c_G: float,
    c_H: float,
    c_A: float,
    rho_H: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    n = len(df)

    expected_ref = normalize_text(
        df["expected_action_reference"]
    ).to_numpy(dtype=object)

    y_t1 = (
        df["y_t1_d1"]
        .fillna("")
        .astype(str)
        .to_numpy(dtype=object)
        if "y_t1_d1" in df.columns
        else np.array([""] * n, dtype=object)
    )

    direct_execute = (
        (primary_action == "EXECUTE")
        & (final_action == "EXECUTE")
    )

    gathered_then_execute = (
        (primary_action == "GATHER")
        & (final_action == "EXECUTE")
    )

    autonomous_execute = final_action == "EXECUTE"

    unsafe = np.zeros(
        n,
        dtype=bool,
    )

    # Direct t0 execution is unsafe if benchmark reference is not EXECUTE.
    unsafe[direct_execute] = (
        expected_ref[direct_execute] != "EXECUTE"
    )

    # Post-GATHER execution uses D1 safety outcome.
    unsafe[gathered_then_execute] = (
        y_t1[gathered_then_execute] != SAFE_LABEL
    )

    # If a GATHER->EXECUTE somehow lacks t1 target, conservatively unsafe.
    missing_t1_target = (
        gathered_then_execute
        & ~np.isin(
            y_t1,
            [SAFE_LABEL, UNSAFE_LABEL],
        )
    )
    unsafe[missing_t1_target] = True

    realized_loss = np.zeros(
        n,
        dtype=float,
    )

    # Direct EXECUTE.
    realized_loss[
        primary_action == "EXECUTE"
    ] = (
        unsafe[primary_action == "EXECUTE"]
        * C[primary_action == "EXECUTE"]
    )

    # Direct ESCALATE.
    mask = primary_action == "ESCALATE"
    realized_loss[mask] = (
        c_H + rho_H * C[mask]
    )

    # Direct ABSTAIN.
    mask = primary_action == "ABSTAIN"
    realized_loss[mask] = c_A

    # Sequential GATHER.
    gmask = primary_action == "GATHER"
    if gmask.any():
        # Gather acquisition cost.
        realized_loss[gmask] = c_G

        # Add continuation realized cost.
        ge = gmask & (final_action == "EXECUTE")
        realized_loss[ge] += (
            unsafe[ge] * C[ge]
        )

        gh = gmask & (final_action == "ESCALATE")
        realized_loss[gh] += (
            c_H + rho_H * C[gh]
        )

        ga = gmask & (final_action == "ABSTAIN")
        realized_loss[ga] += c_A

    autonomous_count = int(
        autonomous_execute.sum()
    )

    unsafe_exec_count = int(
        (unsafe & autonomous_execute).sum()
    )

    safe_exec_count = int(
        autonomous_count - unsafe_exec_count
    )

    coverage = (
        autonomous_count / n
        if n else np.nan
    )

    unsafe_rate = (
        unsafe_exec_count / autonomous_count
        if autonomous_count
        else np.nan
    )

    # Raw euro exposure-weighted unsafe execution rate.
    raw_exp = pd.to_numeric(
        df["transaction_exposure_eur"],
        errors="coerce",
    ).abs().fillna(0.0).to_numpy(dtype=float)

    exec_exp = float(
        raw_exp[autonomous_execute].sum()
    )
    unsafe_exec_exp = float(
        raw_exp[
            autonomous_execute & unsafe
        ].sum()
    )

    raw_exposure_weighted_unsafe = (
        unsafe_exec_exp / exec_exp
        if exec_exp > 0
        else np.nan
    )

    # Consequence-function-weighted unsafe execution rate.
    exec_c = float(
        C[autonomous_execute].sum()
    )
    unsafe_exec_c = float(
        C[
            autonomous_execute & unsafe
        ].sum()
    )

    consequence_weighted_unsafe = (
        unsafe_exec_c / exec_c
        if exec_c > 0
        else np.nan
    )

    result = {
        "episodes": int(n),
        "autonomous_execute_count": autonomous_count,
        "autonomous_execution_coverage": float(coverage),
        "safe_execute_count": safe_exec_count,
        "unsafe_execute_count": unsafe_exec_count,
        "unsafe_execution_rate": float(unsafe_rate)
        if np.isfinite(unsafe_rate)
        else np.nan,
        "raw_exposure_weighted_unsafe_rate": float(
            raw_exposure_weighted_unsafe
        )
        if np.isfinite(raw_exposure_weighted_unsafe)
        else np.nan,
        "consequence_weighted_unsafe_rate": float(
            consequence_weighted_unsafe
        )
        if np.isfinite(consequence_weighted_unsafe)
        else np.nan,
        "mean_expected_policy_loss": float(
            np.mean(expected_selected_loss)
        ),
        "mean_realized_normalized_loss": float(
            np.mean(realized_loss)
        ),
        "human_escalation_rate": float(
            np.mean(final_action == "ESCALATE")
        ),
        "gather_rate": float(
            np.mean(primary_action == "GATHER")
        ),
        "abstention_rate": float(
            np.mean(final_action == "ABSTAIN")
        ),
        "direct_execute_rate": float(
            np.mean(primary_action == "EXECUTE")
        ),
        "post_gather_execute_rate": float(
            np.mean(gathered_then_execute)
        ),
        "missing_t1_target_on_post_gather_execute": int(
            missing_t1_target.sum()
        ),
    }

    detail = pd.DataFrame(
        {
            "benchmark_episode_id": df[
                "benchmark_episode_id"
            ].astype(str).to_numpy(),
            "source_case_id": df[
                "source_case_id"
            ].astype(str).to_numpy(),
            "expected_action_reference": expected_ref,
            "primary_action": primary_action,
            "final_action": final_action,
            "autonomous_execute": autonomous_execute,
            "unsafe_execute": unsafe,
            "consequence_value": C,
            "expected_selected_loss": expected_selected_loss,
            "realized_normalized_loss": realized_loss,
        }
    )

    return result, detail


def simulate_policy(
    df: pd.DataFrame,
    regime: str,
    policy: str,
    consequence_name: str,
    exposure_refs: pd.DataFrame,
    c_G: float,
    c_H: float,
    c_A: float,
    rho_H: float,
    beta0: float | None,
    tau1: float | None,
    beta1: float | None,
    tau0: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:

    C = consequence_values(
        df["transaction_exposure_eur"],
        regime,
        consequence_name,
        exposure_refs,
    )

    pE = pd.to_numeric(
        df["p_cal_execute"],
        errors="coerce",
    ).to_numpy(dtype=float)

    n = len(df)

    L_E = (1.0 - pE) * C
    L_H = c_H + rho_H * C
    L_A = np.full(
        n,
        c_A,
        dtype=float,
    )

    if policy == "ARGMAX":
        primary_action = (
            df["predicted_action_frozen"]
            .astype(str)
            .str.upper()
            .to_numpy(dtype=object)
        )

        # Predictive ARGMAX does not use t1 continuation.
        # If it outputs GATHER, conservatively gather then fall back to the
        # cheaper non-autonomous disposition.
        fallback = (
            "ESCALATE"
            if np.mean(L_H) <= c_A
            else "ABSTAIN"
        )

        final_action = primary_action.copy()
        final_action[
            primary_action == "GATHER"
        ] = fallback

        # Expected selected loss under common loss accounting.
        expected_selected_loss = np.zeros(
            n,
            dtype=float,
        )

        m = primary_action == "EXECUTE"
        expected_selected_loss[m] = L_E[m]

        m = primary_action == "ESCALATE"
        expected_selected_loss[m] = L_H[m]

        m = primary_action == "ABSTAIN"
        expected_selected_loss[m] = L_A[m]

        m = primary_action == "GATHER"
        if fallback == "ESCALATE":
            expected_selected_loss[m] = (
                c_G + L_H[m]
            )
        else:
            expected_selected_loss[m] = (
                c_G + L_A[m]
            )

    elif policy == "CONFIDENCE_THRESHOLD":
        if tau0 is None:
            raise ValueError(
                "CONFIDENCE_THRESHOLD requires tau0."
            )

        primary_action = np.where(
            pE >= tau0,
            "EXECUTE",
            "ESCALATE",
        ).astype(object)

        final_action = primary_action.copy()

        expected_selected_loss = np.where(
            primary_action == "EXECUTE",
            L_E,
            L_H,
        )

    elif policy == "COST_AWARE_BLIND":
        # consequence_name should be UNIFORM by design.
        primary_action, expected_selected_loss = choose_min_loss_action(
            {
                "EXECUTE": L_E,
                "ESCALATE": L_H,
                "ABSTAIN": L_A,
            }
        )
        final_action = primary_action.copy()

    elif policy in {
        "VOI_BLIND",
        "RCSE_NO_GATHER",
        "RCSE_FULL",
    }:
        if beta0 is None:
            raise ValueError(
                f"{policy} requires beta0."
            )

        execute_feasible = (
            L_E <= beta0
        )

        direct_losses = {
            "EXECUTE": np.where(
                execute_feasible,
                L_E,
                np.inf,
            ),
            "ESCALATE": L_H,
            "ABSTAIN": L_A,
        }

        if policy == "RCSE_NO_GATHER":
            primary_action, expected_selected_loss = choose_min_loss_action(
                direct_losses
            )
            final_action = primary_action.copy()

        else:
            if tau1 is None or beta1 is None:
                raise ValueError(
                    f"{policy} requires tau1 and beta1."
                )

            post = post_gather_continuation(
                df=df,
                C=C,
                c_H=c_H,
                c_A=c_A,
                rho_H=rho_H,
                tau1=tau1,
                beta1=beta1,
            )

            gather_eligible = (
                df["gather_policy_eligible"]
                .fillna(False)
                .astype(bool)
                .to_numpy()
            )

            L_G = np.where(
                gather_eligible,
                c_G + post["post_expected_loss"],
                np.inf,
            )

            primary_action, expected_selected_loss = choose_min_loss_action(
                {
                    **direct_losses,
                    "GATHER": L_G,
                }
            )

            final_action = primary_action.copy()

            gather_mask = primary_action == "GATHER"

            final_action[gather_mask] = (
                post["post_action"][gather_mask]
            )

    else:
        raise ValueError(
            f"Unknown policy: {policy}"
        )

    metrics, detail = realized_metrics(
        df=df,
        primary_action=primary_action,
        final_action=final_action,
        expected_selected_loss=expected_selected_loss,
        C=C,
        c_G=c_G,
        c_H=c_H,
        c_A=c_A,
        rho_H=rho_H,
    )

    metrics.update(
        {
            "regime": regime,
            "policy": policy,
            "consequence_function": consequence_name,
            "c_G": float(c_G),
            "c_H": float(c_H),
            "c_A": float(c_A),
            "rho_H": float(rho_H),
            "beta0": beta0,
            "tau1": tau1,
            "beta1": beta1,
            "tau0": tau0,
        }
    )

    return metrics, detail


def valid_cost_combinations(spec: dict[str, Any]) -> list[tuple[float, float, float]]:
    grid = spec["cost_sensitivity_grid"]

    combos = []

    for c_G, c_H, c_A in itertools.product(
        grid["c_G"],
        grid["c_H"],
        grid["c_A"],
    ):
        if c_G < c_H < c_A:
            combos.append(
                (
                    float(c_G),
                    float(c_H),
                    float(c_A),
                )
            )

    return combos


def config_key(row: dict[str, Any]) -> tuple:
    fields = [
        "regime",
        "policy",
        "consequence_function",
        "c_G",
        "c_H",
        "c_A",
        "rho_H",
        "beta0",
        "tau1",
        "beta1",
        "tau0",
        "sweep_family",
    ]
    return tuple(
        row.get(k)
        for k in fields
    )


def main() -> None:
    args = parse_args()

    policy_dir = Path(
        args.policy_dir
    ).expanduser().resolve()

    split_path = Path(
        args.split_benchmark
    ).expanduser().resolve()

    spec_path = Path(
        args.spec
    ).expanduser().resolve()

    if not policy_dir.exists():
        raise FileNotFoundError(policy_dir)
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not spec_path.exists():
        raise FileNotFoundError(spec_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else policy_dir / "policy_simulation_results"
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

    if spec.get("status") != "FROZEN_BEFORE_POLICY_SIMULATION":
        raise RuntimeError(
            "Policy spec is not marked FROZEN_BEFORE_POLICY_SIMULATION."
        )

    print("=" * 80)
    print("RCSE STEP 09b.2b - FROZEN POLICY SIMULATION")
    print("=" * 80)
    print(f"Policy data : {policy_dir}")
    print(f"Split data  : {split_path}")
    print(f"Frozen spec : {spec_path}")
    print(f"Output      : {out_dir}")

    split_df = pd.read_parquet(
        split_path
    )

    exposure_refs = get_train_exposure_refs(
        split_df
    )

    exposure_refs.to_csv(
        out_dir / "rcse_train_exposure_reference.csv",
        index=False,
    )

    ref = spec[
        "primary_reference_configuration"
    ]

    ref_consequence = ref[
        "consequence_function"
    ]

    ref_cG = float(ref["c_G"])
    ref_cH = float(ref["c_H"])
    ref_cA = float(ref["c_A"])
    ref_rho = float(ref["rho_H"])
    ref_beta0 = float(ref["beta0"])
    ref_tau1 = float(ref["tau1"])
    ref_beta1 = float(ref["beta1"])

    tau0_grid = spec[
        "policies"
    ][
        "CONFIDENCE_THRESHOLD"
    ][
        "tau0_grid"
    ]

    beta0_grid = spec[
        "risk_budget"
    ][
        "beta0_grid"
    ]

    tau1_grid = spec[
        "post_gather_gate"
    ][
        "tau1_grid"
    ]

    beta1_grid = spec[
        "post_gather_gate"
    ][
        "beta1_grid"
    ]

    cost_combos = valid_cost_combinations(
        spec
    )

    consequence_names = [
        spec[
            "consequence_functions"
        ][
            "primary"
        ][
            "name"
        ],
        *[
            x["name"]
            for x in spec[
                "consequence_functions"
            ][
                "sensitivity"
            ]
        ],
    ]

    all_rows: list[dict[str, Any]] = []
    action_distribution_rows: list[dict[str, Any]] = []
    reference_action_details = {}

    # Keep output families separately for convenience.
    family_rows = {
        "reference": [],
        "confidence_threshold": [],
        "risk_budget_frontier": [],
        "post_gather_gate": [],
        "cost_sensitivity": [],
        "consequence_sensitivity": [],
    }

    for regime in REGIMES:
        data_path = (
            policy_dir
            / POLICY_FILES[regime]
        )

        if not data_path.exists():
            raise FileNotFoundError(
                data_path
            )

        df = pd.read_parquet(
            data_path
        )

        require_columns(
            df,
            [
                "benchmark_episode_id",
                "source_case_id",
                "expected_action_reference",
                "p_cal_execute",
                "p_cal_gather",
                "p_cal_escalate",
                "p_cal_abstain",
                "predicted_action_frozen",
                "transaction_exposure_eur",
                "transaction_risk_stratum",
                "gather_policy_eligible",
                "p_safe_t1_for_primary_rcse",
                "y_t1_d1",
            ],
            f"policy dataset/{regime}",
        )

        print(
            f"\n[{regime}] episodes={len(df):,}"
        )

        # -------------------------------------------------------------
        # A. Reference configurations
        # -------------------------------------------------------------
        reference_policies = [
            ("ARGMAX", ref_consequence),
            ("COST_AWARE_BLIND", "UNIFORM"),
            ("VOI_BLIND", "UNIFORM"),
            ("RCSE_NO_GATHER", ref_consequence),
            ("RCSE_FULL", ref_consequence),
        ]

        regime_reference_details = []

        for policy, consequence_name in reference_policies:
            metrics, detail = simulate_policy(
                df=df,
                regime=regime,
                policy=policy,
                consequence_name=consequence_name,
                exposure_refs=exposure_refs,
                c_G=ref_cG,
                c_H=ref_cH,
                c_A=ref_cA,
                rho_H=ref_rho,
                beta0=ref_beta0
                if policy in {
                    "VOI_BLIND",
                    "RCSE_NO_GATHER",
                    "RCSE_FULL",
                }
                else None,
                tau1=ref_tau1
                if policy in {
                    "VOI_BLIND",
                    "RCSE_FULL",
                }
                else None,
                beta1=ref_beta1
                if policy in {
                    "VOI_BLIND",
                    "RCSE_FULL",
                }
                else None,
            )

            metrics["sweep_family"] = "reference"
            metrics["config_label"] = "FROZEN_REFERENCE"

            all_rows.append(metrics)
            family_rows["reference"].append(metrics)

            # Action distributions.
            dist = (
                detail.groupby(
                    [
                        "primary_action",
                        "final_action",
                    ]
                )
                .size()
                .reset_index(name="count")
            )
            dist["regime"] = regime
            dist["policy"] = policy
            dist["config_label"] = "FROZEN_REFERENCE"
            dist["pct"] = (
                dist["count"]
                / len(detail)
            )

            action_distribution_rows.extend(
                dist.to_dict(
                    orient="records"
                )
            )

            detail["policy"] = policy
            detail["config_label"] = "FROZEN_REFERENCE"
            regime_reference_details.append(
                detail
            )

        reference_action_details[regime] = pd.concat(
            regime_reference_details,
            ignore_index=True,
        )

        # -------------------------------------------------------------
        # B. Confidence-threshold sweep
        # -------------------------------------------------------------
        for tau0 in tau0_grid:
            metrics, _ = simulate_policy(
                df=df,
                regime=regime,
                policy="CONFIDENCE_THRESHOLD",
                consequence_name=ref_consequence,
                exposure_refs=exposure_refs,
                c_G=ref_cG,
                c_H=ref_cH,
                c_A=ref_cA,
                rho_H=ref_rho,
                beta0=None,
                tau1=None,
                beta1=None,
                tau0=float(tau0),
            )

            metrics["sweep_family"] = "confidence_threshold"
            metrics["config_label"] = f"tau0={tau0}"

            all_rows.append(metrics)
            family_rows["confidence_threshold"].append(metrics)

        # -------------------------------------------------------------
        # C. Risk-budget frontier (all else frozen)
        # -------------------------------------------------------------
        for policy in [
            "VOI_BLIND",
            "RCSE_NO_GATHER",
            "RCSE_FULL",
        ]:
            consequence_name = (
                "UNIFORM"
                if policy == "VOI_BLIND"
                else ref_consequence
            )

            for beta0 in beta0_grid:
                metrics, _ = simulate_policy(
                    df=df,
                    regime=regime,
                    policy=policy,
                    consequence_name=consequence_name,
                    exposure_refs=exposure_refs,
                    c_G=ref_cG,
                    c_H=ref_cH,
                    c_A=ref_cA,
                    rho_H=ref_rho,
                    beta0=float(beta0),
                    tau1=ref_tau1
                    if policy in {
                        "VOI_BLIND",
                        "RCSE_FULL",
                    }
                    else None,
                    beta1=ref_beta1
                    if policy in {
                        "VOI_BLIND",
                        "RCSE_FULL",
                    }
                    else None,
                )

                metrics["sweep_family"] = "risk_budget_frontier"
                metrics["config_label"] = f"beta0={beta0}"

                all_rows.append(metrics)
                family_rows["risk_budget_frontier"].append(metrics)

        # -------------------------------------------------------------
        # D. Post-GATHER gate sensitivity
        # -------------------------------------------------------------
        for policy in [
            "VOI_BLIND",
            "RCSE_FULL",
        ]:
            consequence_name = (
                "UNIFORM"
                if policy == "VOI_BLIND"
                else ref_consequence
            )

            for tau1, beta1 in itertools.product(
                tau1_grid,
                beta1_grid,
            ):
                metrics, _ = simulate_policy(
                    df=df,
                    regime=regime,
                    policy=policy,
                    consequence_name=consequence_name,
                    exposure_refs=exposure_refs,
                    c_G=ref_cG,
                    c_H=ref_cH,
                    c_A=ref_cA,
                    rho_H=ref_rho,
                    beta0=ref_beta0,
                    tau1=float(tau1),
                    beta1=float(beta1),
                )

                metrics["sweep_family"] = "post_gather_gate"
                metrics["config_label"] = (
                    f"tau1={tau1};beta1={beta1}"
                )

                all_rows.append(metrics)
                family_rows["post_gather_gate"].append(metrics)

        # -------------------------------------------------------------
        # E. Cost sensitivity at frozen risk settings
        # -------------------------------------------------------------
        for policy in [
            "COST_AWARE_BLIND",
            "VOI_BLIND",
            "RCSE_NO_GATHER",
            "RCSE_FULL",
        ]:
            consequence_name = (
                "UNIFORM"
                if policy in {
                    "COST_AWARE_BLIND",
                    "VOI_BLIND",
                }
                else ref_consequence
            )

            for c_G, c_H, c_A in cost_combos:
                metrics, _ = simulate_policy(
                    df=df,
                    regime=regime,
                    policy=policy,
                    consequence_name=consequence_name,
                    exposure_refs=exposure_refs,
                    c_G=c_G,
                    c_H=c_H,
                    c_A=c_A,
                    rho_H=ref_rho,
                    beta0=ref_beta0
                    if policy in {
                        "VOI_BLIND",
                        "RCSE_NO_GATHER",
                        "RCSE_FULL",
                    }
                    else None,
                    tau1=ref_tau1
                    if policy in {
                        "VOI_BLIND",
                        "RCSE_FULL",
                    }
                    else None,
                    beta1=ref_beta1
                    if policy in {
                        "VOI_BLIND",
                        "RCSE_FULL",
                    }
                    else None,
                )

                metrics["sweep_family"] = "cost_sensitivity"
                metrics["config_label"] = (
                    f"cG={c_G};cH={c_H};cA={c_A}"
                )

                all_rows.append(metrics)
                family_rows["cost_sensitivity"].append(metrics)

        # -------------------------------------------------------------
        # F. Consequence-function sensitivity
        # -------------------------------------------------------------
        for policy in [
            "RCSE_NO_GATHER",
            "RCSE_FULL",
        ]:
            for consequence_name in consequence_names:
                metrics, _ = simulate_policy(
                    df=df,
                    regime=regime,
                    policy=policy,
                    consequence_name=consequence_name,
                    exposure_refs=exposure_refs,
                    c_G=ref_cG,
                    c_H=ref_cH,
                    c_A=ref_cA,
                    rho_H=ref_rho,
                    beta0=ref_beta0,
                    tau1=ref_tau1
                    if policy == "RCSE_FULL"
                    else None,
                    beta1=ref_beta1
                    if policy == "RCSE_FULL"
                    else None,
                )

                metrics["sweep_family"] = "consequence_sensitivity"
                metrics["config_label"] = consequence_name

                all_rows.append(metrics)
                family_rows["consequence_sensitivity"].append(metrics)

    # -----------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------
    all_df = pd.DataFrame(
        all_rows
    )

    # Deduplicate accidental overlaps across sweep families only within each
    # exact family/config definition; preserve same parameter point if it
    # appears in different declared sweep families.
    all_df.to_csv(
        out_dir / "rcse_all_policy_results.csv",
        index=False,
    )

    pd.DataFrame(
        family_rows["reference"]
    ).to_csv(
        out_dir / "rcse_policy_reference_results.csv",
        index=False,
    )

    pd.DataFrame(
        family_rows["confidence_threshold"]
    ).to_csv(
        out_dir / "rcse_confidence_threshold_sweep.csv",
        index=False,
    )

    pd.DataFrame(
        family_rows["risk_budget_frontier"]
    ).to_csv(
        out_dir / "rcse_risk_budget_frontier.csv",
        index=False,
    )

    pd.DataFrame(
        family_rows["post_gather_gate"]
    ).to_csv(
        out_dir / "rcse_post_gather_gate_sensitivity.csv",
        index=False,
    )

    pd.DataFrame(
        family_rows["cost_sensitivity"]
    ).to_csv(
        out_dir / "rcse_cost_sensitivity.csv",
        index=False,
    )

    pd.DataFrame(
        family_rows["consequence_sensitivity"]
    ).to_csv(
        out_dir / "rcse_consequence_sensitivity.csv",
        index=False,
    )

    pd.DataFrame(
        action_distribution_rows
    ).to_csv(
        out_dir / "rcse_action_distribution.csv",
        index=False,
    )

    for regime, detail in reference_action_details.items():
        detail.to_parquet(
            out_dir
            / f"rcse_reference_actions_{regime}.parquet",
            index=False,
        )

    metadata: dict[str, Any] = {
        "step": "09b.2b",
        "policy_directory": str(policy_dir),
        "split_benchmark": str(split_path),
        "frozen_spec": str(spec_path),
        "spec_status": spec.get("status"),
        "regimes": REGIMES,
        "train_exposure_reference_rule": (
            "source-case-level TRAIN-only exposure percentiles"
        ),
        "reference_configuration": ref,
        "confidence_threshold_grid": tau0_grid,
        "beta0_grid": beta0_grid,
        "tau1_grid": tau1_grid,
        "beta1_grid": beta1_grid,
        "valid_cost_combinations": len(cost_combos),
        "consequence_functions": consequence_names,
        "realized_loss_rule": {
            "direct_execute": (
                "C(v) if expected_action_reference != EXECUTE, else 0"
            ),
            "post_gather_execute": (
                "c_G + C(v) if y_t1_d1 != SAFE_TO_EXECUTE, else c_G"
            ),
            "escalate": "c_H + rho_H*C(v)",
            "abstain": "c_A",
            "gather_then_escalate": "c_G + c_H + rho_H*C(v)",
            "gather_then_abstain": "c_G + c_A",
        },
        "argmax_gather_convention": (
            "ARGMAX does not use t1 continuation; if it predicts GATHER, "
            "the simulation applies GATHER cost then the cheaper non-autonomous "
            "fallback (ESCALATE or ABSTAIN) for end-to-end loss accounting."
        ),
        "governance": spec.get(
            "test_set_governance",
            [],
        ),
        "interpretation": (
            "Do not select a preferred parameter point from test performance. "
            "Use predeclared frontiers/sensitivity ranges plus the frozen "
            "reference configuration."
        ),
    }

    with open(
        out_dir / "rcse_policy_simulation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 09b.2b COMPLETE")
    print("=" * 80)

    ref_df = pd.DataFrame(
        family_rows["reference"]
    )

    print("\nFrozen reference results:")
    cols = [
        "regime",
        "policy",
        "autonomous_execution_coverage",
        "unsafe_execution_rate",
        "raw_exposure_weighted_unsafe_rate",
        "mean_realized_normalized_loss",
        "human_escalation_rate",
        "gather_rate",
        "abstention_rate",
    ]

    print(
        ref_df[cols].to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for name in [
        "rcse_policy_reference_results.csv",
        "rcse_confidence_threshold_sweep.csv",
        "rcse_risk_budget_frontier.csv",
        "rcse_post_gather_gate_sensitivity.csv",
        "rcse_cost_sensitivity.csv",
        "rcse_consequence_sensitivity.csv",
        "rcse_all_policy_results.csv",
        "rcse_action_distribution.csv",
        "rcse_train_exposure_reference.csv",
        "rcse_policy_simulation_metadata.json",
    ]:
        print(f"  - {out_dir / name}")

    for regime in REGIMES:
        print(
            "  - "
            + str(
                out_dir
                / f"rcse_reference_actions_{regime}.parquet"
            )
        )

    print(
        "\nNext: analyze frontiers and sensitivity; do not retune the frozen "
        "policy specification based on test-set winners."
    )


if __name__ == "__main__":
    main()
