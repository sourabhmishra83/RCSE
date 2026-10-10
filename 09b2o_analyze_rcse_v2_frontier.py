#!/usr/bin/env python
"""
09b2o_analyze_rcse_v2_frontier.py

RCSE Step 09b.2o
Analyze the already-generated frozen RCSE v2 selective-autonomy frontier.

Purpose
-------
Use the predeclared frontier produced by Step 09b.2n to answer:

    How much autonomous execution coverage is achievable under fixed
    unsafe-execution risk budgets?

This is FRONTIER ANALYSIS ONLY.

NO MODEL TRAINING.
NO RECALIBRATION.
NO GATE CHANGES.
NO POLICY RETUNING.
NO NEW THRESHOLD SEARCH OUTSIDE THE PREDECLARED GRID.

Input
-----
--frontier
    rcse_v2_policy_frontier.csv

Optional:
--reference-point
    rcse_v2_policy_reference_point.csv

Outputs
-------
rcse_v2_frontier_risk_cap_summary.csv
rcse_v2_frontier_risk_cap_candidates.csv
rcse_v2_frontier_pareto.csv
rcse_v2_frontier_reference_comparison.csv
rcse_v2_frontier_consequence_summary.csv
rcse_v2_frontier_metadata.json
rcse_v2_frontier_coverage_vs_unsafe.png
rcse_v2_frontier_coverage_vs_loss.png
rcse_v2_frontier_coverage_vs_exposure_weighted_risk.png

Default risk caps
-----------------
1%, 2%, 5%, 10%, 20%

Selection semantics
-------------------
For each regime / consequence mode / risk cap:

1. Keep frontier points with:
       unsafe_execution_rate <= risk_cap
2. Among feasible points, choose maximum autonomous execution coverage.
3. Ties are resolved deterministically by:
       lower mean realized normalized loss
       lower consequence-weighted unsafe rate
       lower raw exposure-weighted unsafe rate
       lower human escalation rate
       lower gather rate
       higher tau_safe_t0
       lower beta0
       higher tau_safe_t1
       lower beta1

The selected rows are DESCRIPTIVE operating points from the already-frozen grid.
They MUST NOT be treated as newly tuned production thresholds.

Pareto frontier
---------------
A point is retained if no other point has:
    >= autonomous execution coverage
AND <= unsafe execution rate
AND <= mean realized normalized loss
with at least one strict improvement.

Important
---------
If no point satisfies a requested risk cap, the output records that explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RISK_CAPS = [0.01, 0.02, 0.05, 0.10, 0.20]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2o: analyze frozen RCSE v2 frontier."
    )

    p.add_argument(
        "--frontier",
        required=True,
        help="Path to rcse_v2_policy_frontier.csv from Step 09b.2n",
    )

    p.add_argument(
        "--reference-point",
        default=None,
        help="Optional path to rcse_v2_policy_reference_point.csv",
    )

    p.add_argument(
        "--out",
        default=None,
        help="Default: <frontier parent>/frontier_analysis",
    )

    p.add_argument(
        "--risk-caps",
        nargs="*",
        type=float,
        default=DEFAULT_RISK_CAPS,
        help="Fixed unsafe-execution risk caps.",
    )

    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plots.",
    )

    return p.parse_args()


def require_columns(
    df: pd.DataFrame,
    cols: list[str],
    label: str,
) -> None:
    missing = [
        c for c in cols
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_frontier(
    df: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "regime",
        "consequence_mode",
        "tau_safe_t0",
        "beta0",
        "tau_safe_t1",
        "beta1",
        "episodes",
        "execute_count",
        "autonomous_execution_coverage",
        "unsafe_execute_count",
        "unsafe_execution_rate",
        "raw_exposure_weighted_unsafe_rate",
        "consequence_weighted_unsafe_rate",
        "mean_realized_normalized_loss",
        "human_escalation_rate",
        "gather_rate",
        "abstention_rate",
    ]

    require_columns(
        df,
        required,
        "frontier",
    )

    out = df.copy()

    numeric_cols = [
        c for c in required
        if c not in {
            "regime",
            "consequence_mode",
        }
    ]

    for c in numeric_cols:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce",
        )

    if out[
        [
            "regime",
            "consequence_mode",
            "tau_safe_t0",
            "beta0",
            "tau_safe_t1",
            "beta1",
        ]
    ].duplicated().any():
        raise AssertionError(
            "Frontier contains duplicate parameter points "
            "within a regime/consequence mode."
        )

    return out


def feasible_for_cap(
    df: pd.DataFrame,
    risk_cap: float,
) -> pd.DataFrame:

    # Zero-execution points have NaN unsafe rate.
    # They are technically safe but have zero autonomy.
    risk = df[
        "unsafe_execution_rate"
    ]

    feasible = (
        (df["execute_count"] == 0)
        | (
            risk.notna()
            & (risk <= risk_cap)
        )
    )

    return df.loc[
        feasible
    ].copy()


def choose_best_at_cap(
    df: pd.DataFrame,
    regime: str,
    consequence_mode: str,
    risk_cap: float,
) -> dict[str, Any]:

    feasible = feasible_for_cap(
        df,
        risk_cap,
    )

    if feasible.empty:
        return {
            "regime": regime,
            "consequence_mode": consequence_mode,
            "risk_cap": risk_cap,
            "feasible": False,
            "feasible_point_count": 0,
            "selected_execute_count": 0,
            "selected_coverage": 0.0,
            "selected_unsafe_execution_rate": np.nan,
            "selected_raw_exposure_weighted_unsafe_rate": np.nan,
            "selected_consequence_weighted_unsafe_rate": np.nan,
            "selected_mean_realized_normalized_loss": np.nan,
            "selected_human_escalation_rate": np.nan,
            "selected_gather_rate": np.nan,
            "selected_abstention_rate": np.nan,
            "tau_safe_t0": np.nan,
            "beta0": np.nan,
            "tau_safe_t1": np.nan,
            "beta1": np.nan,
        }

    ranked = feasible.sort_values(
        by=[
            "autonomous_execution_coverage",
            "mean_realized_normalized_loss",
            "consequence_weighted_unsafe_rate",
            "raw_exposure_weighted_unsafe_rate",
            "human_escalation_rate",
            "gather_rate",
            "tau_safe_t0",
            "beta0",
            "tau_safe_t1",
            "beta1",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            True,
        ],
        na_position="last",
        kind="stable",
    )

    row = ranked.iloc[0]

    return {
        "regime": regime,
        "consequence_mode": consequence_mode,
        "risk_cap": risk_cap,
        "feasible": True,
        "feasible_point_count": int(
            len(feasible)
        ),
        "selected_execute_count": int(
            row["execute_count"]
        ),
        "selected_coverage": float(
            row["autonomous_execution_coverage"]
        ),
        "selected_unsafe_execution_rate": (
            float(row["unsafe_execution_rate"])
            if pd.notna(row["unsafe_execution_rate"])
            else np.nan
        ),
        "selected_raw_exposure_weighted_unsafe_rate": (
            float(
                row[
                    "raw_exposure_weighted_unsafe_rate"
                ]
            )
            if pd.notna(
                row[
                    "raw_exposure_weighted_unsafe_rate"
                ]
            )
            else np.nan
        ),
        "selected_consequence_weighted_unsafe_rate": (
            float(
                row[
                    "consequence_weighted_unsafe_rate"
                ]
            )
            if pd.notna(
                row[
                    "consequence_weighted_unsafe_rate"
                ]
            )
            else np.nan
        ),
        "selected_mean_realized_normalized_loss": float(
            row[
                "mean_realized_normalized_loss"
            ]
        ),
        "selected_human_escalation_rate": float(
            row[
                "human_escalation_rate"
            ]
        ),
        "selected_gather_rate": float(
            row[
                "gather_rate"
            ]
        ),
        "selected_abstention_rate": float(
            row[
                "abstention_rate"
            ]
        ),
        "tau_safe_t0": float(
            row["tau_safe_t0"]
        ),
        "beta0": float(
            row["beta0"]
        ),
        "tau_safe_t1": float(
            row["tau_safe_t1"]
        ),
        "beta1": float(
            row["beta1"]
        ),
    }


def build_cap_candidate_table(
    frontier: pd.DataFrame,
    risk_caps: list[float],
) -> pd.DataFrame:

    rows = []

    for (
        regime,
        consequence_mode,
    ), g in frontier.groupby(
        [
            "regime",
            "consequence_mode",
        ],
        observed=True,
    ):

        for cap in risk_caps:
            f = feasible_for_cap(
                g,
                cap,
            )

            if f.empty:
                continue

            temp = f.copy()
            temp["risk_cap"] = cap

            rows.append(
                temp
            )

    if not rows:
        return pd.DataFrame()

    out = pd.concat(
        rows,
        ignore_index=True,
    )

    return out


def pareto_frontier(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Three-objective Pareto set:
      maximize coverage
      minimize unsafe execution rate
      minimize realized normalized loss

    Zero-execution NaN unsafe risk is treated as 0 risk for dominance only.
    """

    work = df.copy()

    risk_for_dom = (
        work[
            "unsafe_execution_rate"
        ]
        .fillna(0.0)
        .to_numpy()
    )

    coverage = (
        work[
            "autonomous_execution_coverage"
        ]
        .to_numpy()
    )

    loss = (
        work[
            "mean_realized_normalized_loss"
        ]
        .to_numpy()
    )

    n = len(work)

    dominated = np.zeros(
        n,
        dtype=bool,
    )

    for i in range(n):
        if dominated[i]:
            continue

        better_or_equal = (
            (coverage >= coverage[i])
            & (
                risk_for_dom
                <= risk_for_dom[i]
            )
            & (
                loss
                <= loss[i]
            )
        )

        strictly_better = (
            (coverage > coverage[i])
            | (
                risk_for_dom
                < risk_for_dom[i]
            )
            | (
                loss
                < loss[i]
            )
        )

        dominators = (
            better_or_equal
            & strictly_better
        )

        dominators[i] = False

        if np.any(
            dominators
        ):
            dominated[i] = True

    out = work.loc[
        ~dominated
    ].copy()

    out[
        "pareto_rank_order"
    ] = np.arange(
        1,
        len(out) + 1,
    )

    return out.sort_values(
        [
            "autonomous_execution_coverage",
            "unsafe_execution_rate",
            "mean_realized_normalized_loss",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        na_position="first",
    )


def build_pareto_table(
    frontier: pd.DataFrame,
) -> pd.DataFrame:

    parts = []

    for (
        regime,
        consequence_mode,
    ), g in frontier.groupby(
        [
            "regime",
            "consequence_mode",
        ],
        observed=True,
    ):

        p = pareto_frontier(
            g
        )

        p.insert(
            0,
            "pareto_group",
            f"{regime}|{consequence_mode}",
        )

        parts.append(
            p
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def reference_comparison(
    cap_summary: pd.DataFrame,
    reference_path: Path | None,
) -> pd.DataFrame:

    if reference_path is None:
        return pd.DataFrame()

    if not reference_path.exists():
        raise FileNotFoundError(
            reference_path
        )

    ref = pd.read_csv(
        reference_path
    )

    require_columns(
        ref,
        [
            "regime",
            "consequence_mode",
            "policy",
            "autonomous_execution_coverage",
            "unsafe_execution_rate",
            "mean_realized_normalized_loss",
        ],
        "reference point",
    )

    ref = ref[
        ref[
            "policy"
        ]
        == "RCSE_V2_FULL"
    ].copy()

    ref = ref.rename(
        columns={
            "autonomous_execution_coverage": "reference_coverage",
            "unsafe_execution_rate": "reference_unsafe_execution_rate",
            "mean_realized_normalized_loss": "reference_mean_realized_normalized_loss",
            "raw_exposure_weighted_unsafe_rate": "reference_raw_exposure_weighted_unsafe_rate",
            "consequence_weighted_unsafe_rate": "reference_consequence_weighted_unsafe_rate",
            "human_escalation_rate": "reference_human_escalation_rate",
            "gather_rate": "reference_gather_rate",
            "abstention_rate": "reference_abstention_rate",
        }
    )

    cols = [
        c for c in [
            "regime",
            "consequence_mode",
            "reference_coverage",
            "reference_unsafe_execution_rate",
            "reference_mean_realized_normalized_loss",
            "reference_raw_exposure_weighted_unsafe_rate",
            "reference_consequence_weighted_unsafe_rate",
            "reference_human_escalation_rate",
            "reference_gather_rate",
            "reference_abstention_rate",
        ]
        if c in ref.columns
    ]

    ref = ref[
        cols
    ]

    merged = cap_summary.merge(
        ref,
        on=[
            "regime",
            "consequence_mode",
        ],
        how="left",
        validate="many_to_one",
    )

    merged[
        "coverage_delta_vs_reference"
    ] = (
        merged[
            "selected_coverage"
        ]
        - merged[
            "reference_coverage"
        ]
    )

    merged[
        "unsafe_rate_delta_vs_reference"
    ] = (
        merged[
            "selected_unsafe_execution_rate"
        ]
        - merged[
            "reference_unsafe_execution_rate"
        ]
    )

    merged[
        "loss_delta_vs_reference"
    ] = (
        merged[
            "selected_mean_realized_normalized_loss"
        ]
        - merged[
            "reference_mean_realized_normalized_loss"
        ]
    )

    return merged


def consequence_summary(
    cap_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        regime,
        risk_cap,
    ), g in cap_summary.groupby(
        [
            "regime",
            "risk_cap",
        ],
        observed=True,
    ):

        feasible = g[
            g[
                "feasible"
            ]
        ].copy()

        if feasible.empty:
            rows.append(
                {
                    "regime": regime,
                    "risk_cap": risk_cap,
                    "best_consequence_mode": None,
                    "max_coverage_across_modes": 0.0,
                    "unsafe_execution_rate": np.nan,
                    "mean_realized_normalized_loss": np.nan,
                }
            )

            continue

        best = feasible.sort_values(
            [
                "selected_coverage",
                "selected_mean_realized_normalized_loss",
                "selected_unsafe_execution_rate",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).iloc[0]

        rows.append(
            {
                "regime": regime,
                "risk_cap": risk_cap,
                "best_consequence_mode": best[
                    "consequence_mode"
                ],
                "max_coverage_across_modes": best[
                    "selected_coverage"
                ],
                "unsafe_execution_rate": best[
                    "selected_unsafe_execution_rate"
                ],
                "mean_realized_normalized_loss": best[
                    "selected_mean_realized_normalized_loss"
                ],
            }
        )

    return pd.DataFrame(
        rows
    )


def make_plots(
    frontier: pd.DataFrame,
    out_dir: Path,
) -> None:

    import matplotlib.pyplot as plt

    # Plot 1: coverage vs unsafe risk
    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for (
        regime,
        mode,
    ), g in frontier.groupby(
        [
            "regime",
            "consequence_mode",
        ]
    ):
        ax.scatter(
            g[
                "autonomous_execution_coverage"
            ],
            g[
                "unsafe_execution_rate"
            ],
            s=14,
            alpha=0.5,
            label=f"{regime} | {mode}",
        )

    ax.set_xlabel(
        "Autonomous execution coverage"
    )

    ax.set_ylabel(
        "Unsafe execution rate"
    )

    ax.set_title(
        "RCSE v2 frozen frontier: coverage vs unsafe risk"
    )

    ax.legend(
        fontsize=7
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "rcse_v2_frontier_coverage_vs_unsafe.png",
        dpi=200,
    )

    plt.close(
        fig
    )

    # Plot 2: coverage vs normalized loss
    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for (
        regime,
        mode,
    ), g in frontier.groupby(
        [
            "regime",
            "consequence_mode",
        ]
    ):
        ax.scatter(
            g[
                "autonomous_execution_coverage"
            ],
            g[
                "mean_realized_normalized_loss"
            ],
            s=14,
            alpha=0.5,
            label=f"{regime} | {mode}",
        )

    ax.set_xlabel(
        "Autonomous execution coverage"
    )

    ax.set_ylabel(
        "Mean realized normalized loss"
    )

    ax.set_title(
        "RCSE v2 frozen frontier: coverage vs realized loss"
    )

    ax.legend(
        fontsize=7
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "rcse_v2_frontier_coverage_vs_loss.png",
        dpi=200,
    )

    plt.close(
        fig
    )

    # Plot 3: coverage vs exposure-weighted unsafe risk
    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for (
        regime,
        mode,
    ), g in frontier.groupby(
        [
            "regime",
            "consequence_mode",
        ]
    ):
        ax.scatter(
            g[
                "autonomous_execution_coverage"
            ],
            g[
                "raw_exposure_weighted_unsafe_rate"
            ],
            s=14,
            alpha=0.5,
            label=f"{regime} | {mode}",
        )

    ax.set_xlabel(
        "Autonomous execution coverage"
    )

    ax.set_ylabel(
        "Raw exposure-weighted unsafe rate"
    )

    ax.set_title(
        "RCSE v2 frozen frontier: coverage vs exposure-weighted risk"
    )

    ax.legend(
        fontsize=7
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        out_dir
        / "rcse_v2_frontier_coverage_vs_exposure_weighted_risk.png",
        dpi=200,
    )

    plt.close(
        fig
    )


def main() -> None:

    args = parse_args()

    frontier_path = (
        Path(
            args.frontier
        )
        .expanduser()
        .resolve()
    )

    if not frontier_path.exists():
        raise FileNotFoundError(
            frontier_path
        )

    reference_path = (
        Path(
            args.reference_point
        )
        .expanduser()
        .resolve()
        if args.reference_point
        else None
    )

    out_dir = (
        Path(
            args.out
        )
        .expanduser()
        .resolve()
        if args.out
        else frontier_path.parent
        / "frontier_analysis"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_caps = sorted(
        set(
            float(x)
            for x in args.risk_caps
        )
    )

    frontier = normalize_frontier(
        pd.read_csv(
            frontier_path
        )
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2o - FROZEN FRONTIER ANALYSIS"
    )
    print(
        "=" * 80
    )

    print(
        f"Frontier       : {frontier_path}"
    )

    print(
        f"Reference point: {reference_path if reference_path else 'not supplied'}"
    )

    print(
        f"Output         : {out_dir}"
    )

    print(
        f"Risk caps      : {risk_caps}"
    )

    print(
        "NO TRAINING / NO RETUNING / ANALYSIS OF PREDECLARED GRID ONLY"
    )

    print(
        f"Frontier rows  : {len(frontier):,}"
    )

    cap_rows = []

    for (
        regime,
        consequence_mode,
    ), g in frontier.groupby(
        [
            "regime",
            "consequence_mode",
        ],
        observed=True,
    ):

        for cap in risk_caps:
            cap_rows.append(
                choose_best_at_cap(
                    df=g,
                    regime=regime,
                    consequence_mode=consequence_mode,
                    risk_cap=cap,
                )
            )

    cap_summary = pd.DataFrame(
        cap_rows
    )

    candidates = build_cap_candidate_table(
        frontier,
        risk_caps,
    )

    pareto = build_pareto_table(
        frontier
    )

    ref_compare = reference_comparison(
        cap_summary=cap_summary,
        reference_path=reference_path,
    )

    consequence = consequence_summary(
        cap_summary
    )

    cap_summary.to_csv(
        out_dir
        / "rcse_v2_frontier_risk_cap_summary.csv",
        index=False,
    )

    candidates.to_csv(
        out_dir
        / "rcse_v2_frontier_risk_cap_candidates.csv",
        index=False,
    )

    pareto.to_csv(
        out_dir
        / "rcse_v2_frontier_pareto.csv",
        index=False,
    )

    consequence.to_csv(
        out_dir
        / "rcse_v2_frontier_consequence_summary.csv",
        index=False,
    )

    if not ref_compare.empty:
        ref_compare.to_csv(
            out_dir
            / "rcse_v2_frontier_reference_comparison.csv",
            index=False,
        )

    metadata: dict[
        str,
        Any,
    ] = {
        "step": "09b.2o",
        "frontier_file": str(
            frontier_path
        ),
        "reference_point_file": (
            str(reference_path)
            if reference_path
            else None
        ),
        "risk_caps": risk_caps,
        "selection_rule": (
            "Maximum coverage among predeclared frontier points satisfying "
            "unsafe_execution_rate <= fixed cap; deterministic tie-breaking "
            "by lower realized loss/risk/escalation/gather burden."
        ),
        "pareto_objectives": {
            "maximize": [
                "autonomous_execution_coverage"
            ],
            "minimize": [
                "unsafe_execution_rate",
                "mean_realized_normalized_loss",
            ],
        },
        "training_performed": False,
        "recalibration_performed": False,
        "gate_modified": False,
        "policy_retuned": False,
        "new_threshold_grid_introduced": False,
        "important_interpretation": (
            "Selected cap points are descriptive operating points from the "
            "already-frozen parameter grid. They must not be presented as "
            "newly tuned production thresholds."
        ),
        "next_decision": (
            "Determine whether RCSE v2 provides meaningful autonomous coverage "
            "at low unsafe-execution caps in grouped IID, temporal, and R4 OOD, "
            "then decide whether statistical uncertainty/bootstrap analysis is "
            "needed before paper claims."
        ),
    }

    with open(
        out_dir
        / "rcse_v2_frontier_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    if not args.no_plots:
        make_plots(
            frontier=frontier,
            out_dir=out_dir,
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 09b.2o COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nRisk-cap operating points:"
    )

    show_cols = [
        "regime",
        "consequence_mode",
        "risk_cap",
        "feasible",
        "feasible_point_count",
        "selected_coverage",
        "selected_unsafe_execution_rate",
        "selected_raw_exposure_weighted_unsafe_rate",
        "selected_consequence_weighted_unsafe_rate",
        "selected_mean_realized_normalized_loss",
        "selected_human_escalation_rate",
        "selected_gather_rate",
        "tau_safe_t0",
        "beta0",
        "tau_safe_t1",
        "beta1",
    ]

    print(
        cap_summary[
            show_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nBest coverage across consequence modes by risk cap:"
    )

    print(
        consequence.to_string(
            index=False
        )
    )

    print(
        "\nPareto points per regime/consequence mode:"
    )

    pareto_counts = (
        pareto.groupby(
            [
                "regime",
                "consequence_mode",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="pareto_points"
        )
    )

    print(
        pareto_counts.to_string(
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
        "\nNext: interpret whether the frozen RCSE v2 grid achieves useful "
        "coverage under low fixed unsafe-execution caps, then quantify "
        "uncertainty statistically before making final claims."
    )


if __name__ == "__main__":
    main()
