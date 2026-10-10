#!/usr/bin/env python
"""
09b2s_build_frozen_publication_results.py

RCSE Step 09b.2s
Build canonical frozen publication results for Step 09.

Purpose
-------
Convert the already-completed RCSE Step 09 outputs into publication-ready
tables, figures, and a single machine-readable frozen-results manifest.

This is a REPORTING / CONSOLIDATION STEP ONLY.

NO MODEL TRAINING.
NO RECALIBRATION.
NO GATE MODIFICATION.
NO POLICY RETUNING.
NO NEW THRESHOLD SELECTION.
NO NEW STATISTICAL TESTS.

Inputs
------
--results-dir
    Step 09b.2n rcse_v2_policy_results directory.

--bootstrap-dir
    Step 09b.2r bootstrap_uncertainty directory.

--frontier-dir
    Step 09b.2o frontier_analysis directory.

--error-dir
    Step 09b.2p unsafe_error_decomposition directory.

--contradiction-dir
    Step 09b.2q residual_contradiction_audit directory.

--spec
    Frozen 09b2l_rcse_v2_policy_spec.json

Outputs
-------
TABLE_01_primary_rcse_v2_performance.csv
TABLE_02_natural_and_intervention_robustness.csv
TABLE_03_component_ablation_bootstrap.csv
TABLE_04_ood_generalization.csv
TABLE_05_frontier_risk_caps.csv

FIGURE_01_frontier_coverage_vs_risk.png
FIGURE_02_gate_robustness.png
FIGURE_03_natural_vs_full_benchmark_risk.png
FIGURE_04_ablation_effects.png

rcse_step09_frozen_results_manifest.json
rcse_step09_publication_summary.md
rcse_step09_publication_metadata.json

Publication rules
-----------------
1. RCSE v2 is a diagnostically motivated revised/secondary analysis.
2. Original RCSE v1 results remain preserved.
3. OOD loss superiority is not claimed when paired bootstrap CI includes zero.
4. Unsafe-execution-rate differences are reported as N/A when one comparator
   has no autonomous executions.
5. Frontier operating points are descriptive points from the frozen grid,
   not newly tuned thresholds.
6. Natural-only risk is reported separately from full synthetic robustness risk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_POLICY = "RCSE_V2_FULL"
PRIMARY_CONSEQUENCE = "LOG_TRAIN_P90"

REGIME_ORDER = ["grouped_iid", "temporal", "ood_exposure"]

DISPLAY_REGIME = {
    "grouped_iid": "Grouped IID",
    "temporal": "Temporal",
    "ood_exposure": "OOD exposure",
}

KEY_PRIMARY_METRICS = [
    "autonomous_execution_coverage",
    "unsafe_execution_rate",
    "raw_exposure_weighted_unsafe_rate",
    "mean_realized_normalized_loss_09b2n",
    "mean_end_to_end_realized_loss_diagnostic",
    "human_escalation_rate",
    "gather_rate",
]

KEY_DIAGNOSTICS = [
    "natural_reference_execute_retention_rate",
    "natural_execution_coverage",
    "natural_unsafe_execution_rate",
    "contradiction_gate_block_rate",
    "residual_valid_contradiction_execution_rate",
    "missingness_gate_block_rate",
    "severe_evidence_loss_gate_block_rate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2s: build frozen publication results."
    )

    p.add_argument(
        "--results-dir",
        required=True,
        help="Step 09b.2n rcse_v2_policy_results directory.",
    )

    p.add_argument(
        "--bootstrap-dir",
        required=True,
        help="Step 09b.2r bootstrap_uncertainty directory.",
    )

    p.add_argument(
        "--frontier-dir",
        required=True,
        help="Step 09b.2o frontier_analysis directory.",
    )

    p.add_argument(
        "--error-dir",
        required=True,
        help="Step 09b.2p unsafe_error_decomposition directory.",
    )

    p.add_argument(
        "--contradiction-dir",
        required=True,
        help="Step 09b.2q residual_contradiction_audit directory.",
    )

    p.add_argument(
        "--spec",
        required=True,
        help="Frozen 09b2l_rcse_v2_policy_spec.json",
    )

    p.add_argument(
        "--out",
        default=None,
        help="Default: <results-dir>/frozen_publication_results",
    )

    return p.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def fmt_pct(x: float | int | None, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{100.0 * float(x):.{digits}f}%"


def fmt_num(x: float | int | None, digits: int = 4) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):.{digits}f}"


def ci_text(point: float, lo: float, hi: float, pct: bool = False) -> str:
    if pd.isna(point):
        return "N/A"

    if pct:
        if pd.isna(lo) or pd.isna(hi):
            return fmt_pct(point)
        return f"{fmt_pct(point)} [{fmt_pct(lo)}, {fmt_pct(hi)}]"

    if pd.isna(lo) or pd.isna(hi):
        return fmt_num(point)

    return f"{fmt_num(point)} [{fmt_num(lo)}, {fmt_num(hi)}]"


def load_inputs(
    results_dir: Path,
    bootstrap_dir: Path,
    frontier_dir: Path,
    error_dir: Path,
    contradiction_dir: Path,
) -> dict[str, pd.DataFrame]:
    files = {
        "reference": results_dir / "rcse_v2_policy_reference_point.csv",
        "summary": results_dir / "rcse_v2_policy_results_summary.csv",
        "policy_ci": bootstrap_dir / "rcse_v2r_bootstrap_policy_ci.csv",
        "ablation_ci": bootstrap_dir / "rcse_v2r_bootstrap_ablation_ci.csv",
        "diagnostic_ci": bootstrap_dir / "rcse_v2r_bootstrap_diagnostic_ci.csv",
        "diag_points": bootstrap_dir / "rcse_v2r_bootstrap_diagnostic_point_estimates.csv",
        "frontier_caps": frontier_dir / "rcse_v2_frontier_risk_cap_summary.csv",
        "frontier_pareto": frontier_dir / "rcse_v2_frontier_pareto.csv",
        "direct_errors": error_dir / "rcse_v2p_direct_execution_summary.csv",
        "intervention_errors": error_dir / "rcse_v2p_unsafe_by_intervention_family.csv",
        "contradiction_flow": contradiction_dir / "rcse_v2q_contradiction_flow.csv",
        "contradiction_type_strength": contradiction_dir / "rcse_v2q_contradiction_by_type_strength.csv",
    }

    out = {}
    for key, path in files.items():
        require_file(path)
        out[key] = pd.read_csv(path)

    return out


def build_primary_table(
    policy_ci: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        policy_ci,
        [
            "regime",
            "policy",
            "metric",
            "point_estimate",
            "ci_lower",
            "ci_upper",
        ],
        "bootstrap policy CI",
    )

    f = policy_ci[
        (policy_ci["policy"] == PRIMARY_POLICY)
        & (policy_ci["metric"].isin(KEY_PRIMARY_METRICS))
    ].copy()

    rows = []

    for regime in REGIME_ORDER:
        rg = f[f["regime"] == regime].set_index("metric")

        def get(metric: str):
            if metric not in rg.index:
                return (np.nan, np.nan, np.nan)
            r = rg.loc[metric]
            return (
                float(r["point_estimate"]),
                float(r["ci_lower"]) if pd.notna(r["ci_lower"]) else np.nan,
                float(r["ci_upper"]) if pd.notna(r["ci_upper"]) else np.nan,
            )

        cov = get("autonomous_execution_coverage")
        unsafe = get("unsafe_execution_rate")
        exrisk = get("raw_exposure_weighted_unsafe_rate")
        loss09 = get("mean_realized_normalized_loss_09b2n")
        losse2e = get("mean_end_to_end_realized_loss_diagnostic")
        esc = get("human_escalation_rate")
        gather = get("gather_rate")

        rows.append(
            {
                "regime": regime,
                "regime_display": DISPLAY_REGIME[regime],
                "autonomous_execution_coverage": cov[0],
                "autonomous_execution_coverage_ci_lower": cov[1],
                "autonomous_execution_coverage_ci_upper": cov[2],
                "autonomous_execution_coverage_display": ci_text(*cov, pct=True),
                "unsafe_execution_rate": unsafe[0],
                "unsafe_execution_rate_ci_lower": unsafe[1],
                "unsafe_execution_rate_ci_upper": unsafe[2],
                "unsafe_execution_rate_display": ci_text(*unsafe, pct=True),
                "raw_exposure_weighted_unsafe_rate": exrisk[0],
                "raw_exposure_weighted_unsafe_rate_ci_lower": exrisk[1],
                "raw_exposure_weighted_unsafe_rate_ci_upper": exrisk[2],
                "raw_exposure_weighted_unsafe_rate_display": ci_text(*exrisk, pct=True),
                "mean_realized_normalized_loss_09b2n": loss09[0],
                "mean_realized_normalized_loss_09b2n_ci_lower": loss09[1],
                "mean_realized_normalized_loss_09b2n_ci_upper": loss09[2],
                "mean_realized_normalized_loss_09b2n_display": ci_text(*loss09),
                "mean_end_to_end_realized_loss_diagnostic": losse2e[0],
                "mean_end_to_end_realized_loss_diagnostic_ci_lower": losse2e[1],
                "mean_end_to_end_realized_loss_diagnostic_ci_upper": losse2e[2],
                "mean_end_to_end_realized_loss_diagnostic_display": ci_text(*losse2e),
                "human_escalation_rate": esc[0],
                "human_escalation_rate_ci_lower": esc[1],
                "human_escalation_rate_ci_upper": esc[2],
                "human_escalation_rate_display": ci_text(*esc, pct=True),
                "gather_rate": gather[0],
                "gather_rate_ci_lower": gather[1],
                "gather_rate_ci_upper": gather[2],
                "gather_rate_display": ci_text(*gather, pct=True),
            }
        )

    return pd.DataFrame(rows)


def build_robustness_table(
    diagnostic_ci: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        diagnostic_ci,
        [
            "regime",
            "metric",
            "point_estimate",
            "ci_lower",
            "ci_upper",
        ],
        "bootstrap diagnostic CI",
    )

    f = diagnostic_ci[
        diagnostic_ci["metric"].isin(KEY_DIAGNOSTICS)
    ].copy()

    rows = []

    for regime in REGIME_ORDER:
        rg = f[f["regime"] == regime].set_index("metric")

        def get(metric: str):
            if metric not in rg.index:
                return (np.nan, np.nan, np.nan)
            r = rg.loc[metric]
            return (
                float(r["point_estimate"]),
                float(r["ci_lower"]) if pd.notna(r["ci_lower"]) else np.nan,
                float(r["ci_upper"]) if pd.notna(r["ci_upper"]) else np.nan,
            )

        vals = {m: get(m) for m in KEY_DIAGNOSTICS}

        rows.append(
            {
                "regime": regime,
                "regime_display": DISPLAY_REGIME[regime],
                **{
                    m: vals[m][0]
                    for m in KEY_DIAGNOSTICS
                },
                **{
                    f"{m}_ci_lower": vals[m][1]
                    for m in KEY_DIAGNOSTICS
                },
                **{
                    f"{m}_ci_upper": vals[m][2]
                    for m in KEY_DIAGNOSTICS
                },
                **{
                    f"{m}_display": ci_text(*vals[m], pct=True)
                    for m in KEY_DIAGNOSTICS
                },
            }
        )

    return pd.DataFrame(rows)


def build_ablation_table(
    ablation_ci: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        ablation_ci,
        [
            "regime",
            "comparison",
            "left_policy",
            "right_policy",
            "metric",
            "bootstrap_mean",
            "ci_lower",
            "ci_upper",
        ],
        "bootstrap ablation CI",
    )

    keep_metrics = [
        "autonomous_execution_coverage",
        "unsafe_execution_rate",
        "mean_realized_normalized_loss_09b2n",
        "mean_end_to_end_realized_loss_diagnostic",
    ]

    out = ablation_ci[
        ablation_ci["metric"].isin(keep_metrics)
    ].copy()

    out["ci_excludes_zero"] = (
        (
            out["ci_lower"].notna()
            & out["ci_upper"].notna()
            & (
                (out["ci_lower"] > 0)
                | (out["ci_upper"] < 0)
            )
        )
    )

    out["metric_defined"] = out[
        ["bootstrap_mean", "ci_lower", "ci_upper"]
    ].notna().all(axis=1)

    out["reporting_status"] = np.where(
        out["metric_defined"],
        np.where(
            out["ci_excludes_zero"],
            "CI excludes zero",
            "CI includes zero",
        ),
        "N/A (undefined metric for at least one comparator)",
    )

    out["effect_display"] = out.apply(
        lambda r: (
            "N/A"
            if not r["metric_defined"]
            else f"{r['bootstrap_mean']:.4f} [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]"
        ),
        axis=1,
    )

    return out


def build_ood_table(
    primary: pd.DataFrame,
    robustness: pd.DataFrame,
    ablation: pd.DataFrame,
) -> pd.DataFrame:
    p = primary[
        primary["regime"] == "ood_exposure"
    ].iloc[0]

    r = robustness[
        robustness["regime"] == "ood_exposure"
    ].iloc[0]

    rows = [
        {
            "category": "Primary",
            "metric": "Autonomous execution coverage",
            "estimate": p["autonomous_execution_coverage"],
            "ci_lower": p["autonomous_execution_coverage_ci_lower"],
            "ci_upper": p["autonomous_execution_coverage_ci_upper"],
            "interpretation": "Very low autonomous coverage under OOD exposure shift.",
        },
        {
            "category": "Primary",
            "metric": "Unsafe execution rate",
            "estimate": p["unsafe_execution_rate"],
            "ci_lower": p["unsafe_execution_rate_ci_lower"],
            "ci_upper": p["unsafe_execution_rate_ci_upper"],
            "interpretation": "Wide CI because very few OOD cases are autonomously executed.",
        },
        {
            "category": "Primary",
            "metric": "Human escalation rate",
            "estimate": p["human_escalation_rate"],
            "ci_lower": p["human_escalation_rate_ci_lower"],
            "ci_upper": p["human_escalation_rate_ci_upper"],
            "interpretation": "RCSE behaves conservatively under OOD shift.",
        },
        {
            "category": "Natural-only",
            "metric": "Natural unsafe execution rate",
            "estimate": r["natural_unsafe_execution_rate"],
            "ci_lower": r["natural_unsafe_execution_rate_ci_lower"],
            "ci_upper": r["natural_unsafe_execution_rate_ci_upper"],
            "interpretation": "Natural OOD execution remains uncertain and materially weaker than IID/temporal.",
        },
        {
            "category": "Gate",
            "metric": "Contradiction block rate",
            "estimate": r["contradiction_gate_block_rate"],
            "ci_lower": r["contradiction_gate_block_rate_ci_lower"],
            "ci_upper": r["contradiction_gate_block_rate_ci_upper"],
            "interpretation": "Evidence gate remains effective under OOD.",
        },
    ]

    for comparison in [
        "CONSEQUENCE_AWARE_VS_VOI_BLIND",
        "V2_VS_V1",
        "GATHER_CONTRIBUTION",
    ]:
        sub = ablation[
            (ablation["regime"] == "ood_exposure")
            & (ablation["comparison"] == comparison)
            & (
                ablation["metric"]
                == "mean_end_to_end_realized_loss_diagnostic"
            )
        ]

        if len(sub):
            rr = sub.iloc[0]
            rows.append(
                {
                    "category": "Paired ablation",
                    "metric": comparison,
                    "estimate": rr["bootstrap_mean"],
                    "ci_lower": rr["ci_lower"],
                    "ci_upper": rr["ci_upper"],
                    "interpretation": (
                        "Statistically stable difference."
                        if rr["ci_excludes_zero"]
                        else "CI includes zero; do not claim OOD loss superiority."
                    ),
                }
            )

    out = pd.DataFrame(rows)

    out["display"] = out.apply(
        lambda r: ci_text(
            r["estimate"],
            r["ci_lower"],
            r["ci_upper"],
            pct=("rate" in r["metric"].lower() or "coverage" in r["metric"].lower()),
        ),
        axis=1,
    )

    return out


def build_frontier_table(
    frontier_caps: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        frontier_caps,
        [
            "regime",
            "consequence_mode",
            "risk_cap",
            "feasible",
            "selected_coverage",
            "selected_unsafe_execution_rate",
            "tau_safe_t0",
            "beta0",
            "tau_safe_t1",
            "beta1",
        ],
        "frontier risk cap summary",
    )

    f = frontier_caps[
        frontier_caps["consequence_mode"]
        == PRIMARY_CONSEQUENCE
    ].copy()

    f["regime_display"] = f["regime"].map(DISPLAY_REGIME)
    f["risk_cap_display"] = f["risk_cap"].map(fmt_pct)
    f["selected_coverage_display"] = f["selected_coverage"].map(fmt_pct)
    f["selected_unsafe_execution_rate_display"] = (
        f["selected_unsafe_execution_rate"].map(fmt_pct)
    )

    f["reporting_note"] = np.where(
        f["feasible"],
        "Descriptive point from frozen predeclared grid",
        "No feasible frozen-grid point at this risk cap",
    )

    return f


def make_figures(
    out_dir: Path,
    frontier_caps: pd.DataFrame,
    robustness: pd.DataFrame,
    primary: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    # Figure 1: coverage vs risk cap (primary consequence only)
    f = frontier_caps[
        frontier_caps["consequence_mode"] == PRIMARY_CONSEQUENCE
    ].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    for regime in REGIME_ORDER:
        g = f[f["regime"] == regime].sort_values("risk_cap")
        ax.plot(
            g["risk_cap"],
            g["selected_coverage"],
            marker="o",
            label=DISPLAY_REGIME[regime],
        )

    ax.set_xlabel("Unsafe-execution risk cap")
    ax.set_ylabel("Maximum descriptive autonomous coverage")
    ax.set_title("Frozen RCSE v2 frontier under fixed unsafe-risk caps")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out_dir / "FIGURE_01_frontier_coverage_vs_risk.png",
        dpi=200,
    )
    plt.close(fig)

    # Figure 2: gate robustness
    x = np.arange(len(REGIME_ORDER))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8, 6))

    contradiction = [
        robustness.loc[
            robustness["regime"] == r,
            "contradiction_gate_block_rate",
        ].iloc[0]
        for r in REGIME_ORDER
    ]

    missingness = [
        robustness.loc[
            robustness["regime"] == r,
            "missingness_gate_block_rate",
        ].iloc[0]
        for r in REGIME_ORDER
    ]

    severe = [
        robustness.loc[
            robustness["regime"] == r,
            "severe_evidence_loss_gate_block_rate",
        ].iloc[0]
        for r in REGIME_ORDER
    ]

    ax.bar(x - width, contradiction, width, label="Contradiction")
    ax.bar(x, missingness, width, label="Missingness")
    ax.bar(x + width, severe, width, label="Severe evidence loss")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_REGIME[r] for r in REGIME_ORDER])
    ax.set_ylabel("Gate block rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Evidence Gate v2 robustness")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out_dir / "FIGURE_02_gate_robustness.png",
        dpi=200,
    )
    plt.close(fig)

    # Figure 3: natural vs full-benchmark unsafe rate
    fig, ax = plt.subplots(figsize=(8, 6))

    full_risk = [
        primary.loc[
            primary["regime"] == r,
            "unsafe_execution_rate",
        ].iloc[0]
        for r in REGIME_ORDER
    ]

    natural_risk = [
        robustness.loc[
            robustness["regime"] == r,
            "natural_unsafe_execution_rate",
        ].iloc[0]
        for r in REGIME_ORDER
    ]

    ax.bar(x - width / 2, full_risk, width, label="Full benchmark")
    ax.bar(x + width / 2, natural_risk, width, label="Natural only")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_REGIME[r] for r in REGIME_ORDER])
    ax.set_ylabel("Unsafe execution rate")
    ax.set_title("RCSE v2 risk: full benchmark vs natural episodes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        out_dir / "FIGURE_03_natural_vs_full_benchmark_risk.png",
        dpi=200,
    )
    plt.close(fig)

    # Figure 4: key ablation unsafe-rate effects
    key = ablation[
        (ablation["metric"] == "unsafe_execution_rate")
        & (
            ablation["comparison"].isin(
                [
                    "EVIDENCE_GATE_CONTRIBUTION",
                    "CONSEQUENCE_AWARE_VS_VOI_BLIND",
                    "V2_VS_V1",
                ]
            )
        )
    ].copy()

    key = key[key["metric_defined"]]

    if len(key):
        fig, ax = plt.subplots(figsize=(9, 6))

        labels = (
            key["regime"].map(DISPLAY_REGIME)
            + " | "
            + key["comparison"]
        )

        xpos = np.arange(len(key))

        ax.errorbar(
            xpos,
            key["bootstrap_mean"],
            yerr=np.vstack(
                [
                    key["bootstrap_mean"] - key["ci_lower"],
                    key["ci_upper"] - key["bootstrap_mean"],
                ]
            ),
            fmt="o",
            capsize=3,
        )

        ax.axhline(0.0, linewidth=1)
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=60, ha="right")
        ax.set_ylabel("Paired Δ unsafe execution rate")
        ax.set_title("Frozen RCSE v2 ablation effects")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            out_dir / "FIGURE_04_ablation_effects.png",
            dpi=200,
        )
        plt.close(fig)


def build_manifest(
    spec: dict[str, Any],
    primary: pd.DataFrame,
    robustness: pd.DataFrame,
    ablation: pd.DataFrame,
    ood: pd.DataFrame,
    frontier: pd.DataFrame,
) -> dict[str, Any]:
    manifest = {
        "step": "09b.2s",
        "status": "STEP_09_FROZEN_PUBLICATION_RESULTS",
        "analysis_status": spec.get("analysis_status"),
        "primary_policy": PRIMARY_POLICY,
        "primary_consequence": PRIMARY_CONSEQUENCE,
        "frozen_spec_status": spec.get("status"),
        "publication_rules": [
            "RCSE v2 is a revised/secondary analysis motivated by documented diagnostics.",
            "Original RCSE v1 results remain preserved.",
            "No OOD loss-superiority claim when paired bootstrap CI includes zero.",
            "Unsafe-rate ablations are N/A when one comparator has zero autonomous executions.",
            "Frontier points are descriptive points from the frozen predeclared grid, not retuned thresholds.",
            "Natural-only risk must be distinguished from full synthetic robustness risk.",
        ],
        "primary_results": {},
        "robustness_results": {},
        "ablation_results": [],
        "ood_results": [],
        "frontier_results": [],
    }

    for _, r in primary.iterrows():
        regime = r["regime"]
        manifest["primary_results"][regime] = {
            "autonomous_execution_coverage": {
                "estimate": r["autonomous_execution_coverage"],
                "ci_lower": r["autonomous_execution_coverage_ci_lower"],
                "ci_upper": r["autonomous_execution_coverage_ci_upper"],
            },
            "unsafe_execution_rate": {
                "estimate": r["unsafe_execution_rate"],
                "ci_lower": r["unsafe_execution_rate_ci_lower"],
                "ci_upper": r["unsafe_execution_rate_ci_upper"],
            },
            "raw_exposure_weighted_unsafe_rate": {
                "estimate": r["raw_exposure_weighted_unsafe_rate"],
                "ci_lower": r["raw_exposure_weighted_unsafe_rate_ci_lower"],
                "ci_upper": r["raw_exposure_weighted_unsafe_rate_ci_upper"],
            },
            "mean_realized_normalized_loss_09b2n": {
                "estimate": r["mean_realized_normalized_loss_09b2n"],
                "ci_lower": r["mean_realized_normalized_loss_09b2n_ci_lower"],
                "ci_upper": r["mean_realized_normalized_loss_09b2n_ci_upper"],
            },
            "mean_end_to_end_realized_loss_diagnostic": {
                "estimate": r["mean_end_to_end_realized_loss_diagnostic"],
                "ci_lower": r["mean_end_to_end_realized_loss_diagnostic_ci_lower"],
                "ci_upper": r["mean_end_to_end_realized_loss_diagnostic_ci_upper"],
            },
        }

    for _, r in robustness.iterrows():
        regime = r["regime"]
        manifest["robustness_results"][regime] = {
            m: {
                "estimate": r[m],
                "ci_lower": r[f"{m}_ci_lower"],
                "ci_upper": r[f"{m}_ci_upper"],
            }
            for m in KEY_DIAGNOSTICS
        }

    for _, r in ablation.iterrows():
        manifest["ablation_results"].append(
            {
                "regime": r["regime"],
                "comparison": r["comparison"],
                "metric": r["metric"],
                "bootstrap_mean": (
                    None
                    if pd.isna(r["bootstrap_mean"])
                    else float(r["bootstrap_mean"])
                ),
                "ci_lower": (
                    None
                    if pd.isna(r["ci_lower"])
                    else float(r["ci_lower"])
                ),
                "ci_upper": (
                    None
                    if pd.isna(r["ci_upper"])
                    else float(r["ci_upper"])
                ),
                "reporting_status": r["reporting_status"],
            }
        )

    for _, r in ood.iterrows():
        manifest["ood_results"].append(
            {
                "category": r["category"],
                "metric": r["metric"],
                "estimate": (
                    None
                    if pd.isna(r["estimate"])
                    else float(r["estimate"])
                ),
                "ci_lower": (
                    None
                    if pd.isna(r["ci_lower"])
                    else float(r["ci_lower"])
                ),
                "ci_upper": (
                    None
                    if pd.isna(r["ci_upper"])
                    else float(r["ci_upper"])
                ),
                "interpretation": r["interpretation"],
            }
        )

    for _, r in frontier.iterrows():
        manifest["frontier_results"].append(
            {
                "regime": r["regime"],
                "risk_cap": float(r["risk_cap"]),
                "feasible": bool(r["feasible"]),
                "selected_coverage": (
                    None
                    if pd.isna(r["selected_coverage"])
                    else float(r["selected_coverage"])
                ),
                "selected_unsafe_execution_rate": (
                    None
                    if pd.isna(r["selected_unsafe_execution_rate"])
                    else float(r["selected_unsafe_execution_rate"])
                ),
                "tau_safe_t0": (
                    None
                    if pd.isna(r["tau_safe_t0"])
                    else float(r["tau_safe_t0"])
                ),
                "beta0": (
                    None
                    if pd.isna(r["beta0"])
                    else float(r["beta0"])
                ),
                "tau_safe_t1": (
                    None
                    if pd.isna(r["tau_safe_t1"])
                    else float(r["tau_safe_t1"])
                ),
                "beta1": (
                    None
                    if pd.isna(r["beta1"])
                    else float(r["beta1"])
                ),
            }
        )

    return manifest


def build_summary_md(
    primary: pd.DataFrame,
    robustness: pd.DataFrame,
    ood: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# RCSE Step 09 Frozen Publication Summary")
    lines.append("")
    lines.append("## Primary RCSE v2 reference results")
    lines.append("")
    lines.append("| Regime | Coverage | Unsafe execution | Exposure-weighted unsafe | End-to-end loss | Escalation | GATHER |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in primary.iterrows():
        lines.append(
            f"| {r['regime_display']} | "
            f"{r['autonomous_execution_coverage_display']} | "
            f"{r['unsafe_execution_rate_display']} | "
            f"{r['raw_exposure_weighted_unsafe_rate_display']} | "
            f"{r['mean_end_to_end_realized_loss_diagnostic_display']} | "
            f"{r['human_escalation_rate_display']} | "
            f"{r['gather_rate_display']} |"
        )

    lines.append("")
    lines.append("## Natural-only robustness")
    lines.append("")
    lines.append("| Regime | Natural execution coverage | Natural unsafe execution | Contradiction block | Missingness block | Severe evidence-loss block |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for _, r in robustness.iterrows():
        lines.append(
            f"| {r['regime_display']} | "
            f"{r['natural_execution_coverage_display']} | "
            f"{r['natural_unsafe_execution_rate_display']} | "
            f"{r['contradiction_gate_block_rate_display']} | "
            f"{r['missingness_gate_block_rate_display']} | "
            f"{r['severe_evidence_loss_gate_block_rate_display']} |"
        )

    lines.append("")
    lines.append("## Publication-safe interpretation")
    lines.append("")
    lines.append(
        "- The evidence-validity gate is a necessary architectural component: "
        "paired ablation shows large, statistically stable reductions in unsafe execution and realized loss."
    )
    lines.append(
        "- Natural-case execution risk is substantially lower than full-benchmark risk; "
        "the full benchmark intentionally includes controlled synthetic interventions."
    )
    lines.append(
        "- Residual unsafe execution is dominated by near-boundary contradiction cases at the frozen 5% consistency threshold."
    )
    lines.append(
        "- RCSE behaves conservatively under OOD exposure shift, primarily through escalation rather than maintaining autonomous coverage."
    )
    lines.append(
        "- GATHER improves end-to-end loss in grouped-IID and temporal settings but does not show robust benefit under OOD exposure shift."
    )
    lines.append(
        "- Do not claim universal OOD superiority, universal low-risk autonomy, or production readiness from the current reference point."
    )

    lines.append("")
    lines.append("## OOD reporting notes")
    lines.append("")

    for _, r in ood.iterrows():
        lines.append(
            f"- **{r['metric']}**: {r['display']} — {r['interpretation']}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    bootstrap_dir = Path(args.bootstrap_dir).expanduser().resolve()
    frontier_dir = Path(args.frontier_dir).expanduser().resolve()
    error_dir = Path(args.error_dir).expanduser().resolve()
    contradiction_dir = Path(args.contradiction_dir).expanduser().resolve()
    spec_path = Path(args.spec).expanduser().resolve()

    for p in [
        results_dir,
        bootstrap_dir,
        frontier_dir,
        error_dir,
        contradiction_dir,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    require_file(spec_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else results_dir / "frozen_publication_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    if spec.get("status") != "FROZEN_BEFORE_RCSE_V2_SIMULATION":
        raise RuntimeError(
            "Provided RCSE v2 spec is not marked frozen before simulation."
        )

    print("=" * 80)
    print("RCSE STEP 09b.2s - BUILD FROZEN PUBLICATION RESULTS")
    print("=" * 80)
    print(f"Results dir       : {results_dir}")
    print(f"Bootstrap dir     : {bootstrap_dir}")
    print(f"Frontier dir      : {frontier_dir}")
    print(f"Error dir         : {error_dir}")
    print(f"Contradiction dir : {contradiction_dir}")
    print(f"Frozen spec       : {spec_path}")
    print(f"Output            : {out_dir}")
    print("REPORTING / CONSOLIDATION ONLY")
    print("NO TRAINING / NO RETUNING / NO NEW ANALYSIS")

    data = load_inputs(
        results_dir=results_dir,
        bootstrap_dir=bootstrap_dir,
        frontier_dir=frontier_dir,
        error_dir=error_dir,
        contradiction_dir=contradiction_dir,
    )

    primary = build_primary_table(
        data["policy_ci"]
    )

    robustness = build_robustness_table(
        data["diagnostic_ci"]
    )

    ablation = build_ablation_table(
        data["ablation_ci"]
    )

    ood = build_ood_table(
        primary=primary,
        robustness=robustness,
        ablation=ablation,
    )

    frontier = build_frontier_table(
        data["frontier_caps"]
    )

    # Write tables
    primary.to_csv(
        out_dir / "TABLE_01_primary_rcse_v2_performance.csv",
        index=False,
    )

    robustness.to_csv(
        out_dir / "TABLE_02_natural_and_intervention_robustness.csv",
        index=False,
    )

    ablation.to_csv(
        out_dir / "TABLE_03_component_ablation_bootstrap.csv",
        index=False,
    )

    ood.to_csv(
        out_dir / "TABLE_04_ood_generalization.csv",
        index=False,
    )

    frontier.to_csv(
        out_dir / "TABLE_05_frontier_risk_caps.csv",
        index=False,
    )

    # Figures
    make_figures(
        out_dir=out_dir,
        frontier_caps=data["frontier_caps"],
        robustness=robustness,
        primary=primary,
        ablation=ablation,
    )

    # Manifest
    manifest = build_manifest(
        spec=spec,
        primary=primary,
        robustness=robustness,
        ablation=ablation,
        ood=ood,
        frontier=frontier,
    )

    manifest_path = (
        out_dir / "rcse_step09_frozen_results_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Markdown summary
    summary_md = build_summary_md(
        primary=primary,
        robustness=robustness,
        ood=ood,
    )

    summary_path = (
        out_dir / "rcse_step09_publication_summary.md"
    )

    summary_path.write_text(
        summary_md,
        encoding="utf-8",
    )

    metadata = {
        "step": "09b.2s",
        "status": "STEP_09_PUBLICATION_RESULTS_FROZEN",
        "analysis_type": "reporting_consolidation_only",
        "primary_policy": PRIMARY_POLICY,
        "primary_consequence": PRIMARY_CONSEQUENCE,
        "source_artifacts": {
            "results_dir": str(results_dir),
            "bootstrap_dir": str(bootstrap_dir),
            "frontier_dir": str(frontier_dir),
            "error_dir": str(error_dir),
            "contradiction_dir": str(contradiction_dir),
            "frozen_spec": str(spec_path),
        },
        "new_training_performed": False,
        "new_recalibration_performed": False,
        "gate_modified": False,
        "policy_retuned": False,
        "new_threshold_selection": False,
        "new_statistical_tests": False,
        "publication_outputs": [
            "TABLE_01_primary_rcse_v2_performance.csv",
            "TABLE_02_natural_and_intervention_robustness.csv",
            "TABLE_03_component_ablation_bootstrap.csv",
            "TABLE_04_ood_generalization.csv",
            "TABLE_05_frontier_risk_caps.csv",
            "FIGURE_01_frontier_coverage_vs_risk.png",
            "FIGURE_02_gate_robustness.png",
            "FIGURE_03_natural_vs_full_benchmark_risk.png",
            "FIGURE_04_ablation_effects.png",
            "rcse_step09_frozen_results_manifest.json",
            "rcse_step09_publication_summary.md",
        ],
        "next_step": (
            "Use the frozen manifest/tables/figures as the only authoritative "
            "Step 09 source for manuscript Results writing. Do not re-open model "
            "or policy tuning unless a genuine implementation defect is discovered."
        ),
    }

    metadata_path = (
        out_dir / "rcse_step09_publication_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("STEP 09b.2s COMPLETE")
    print("=" * 80)

    print("\nPrimary RCSE v2 performance:")
    print(
        primary[
            [
                "regime_display",
                "autonomous_execution_coverage_display",
                "unsafe_execution_rate_display",
                "raw_exposure_weighted_unsafe_rate_display",
                "mean_end_to_end_realized_loss_diagnostic_display",
                "human_escalation_rate_display",
                "gather_rate_display",
            ]
        ].to_string(index=False)
    )

    print("\nNatural/intervention robustness:")
    print(
        robustness[
            [
                "regime_display",
                "natural_execution_coverage_display",
                "natural_unsafe_execution_rate_display",
                "contradiction_gate_block_rate_display",
                "missingness_gate_block_rate_display",
                "severe_evidence_loss_gate_block_rate_display",
            ]
        ].to_string(index=False)
    )

    print("\nOOD reporting summary:")
    print(
        ood[
            [
                "category",
                "metric",
                "display",
                "interpretation",
            ]
        ].to_string(index=False)
    )

    print("\nOutputs:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p}")

    print(
        "\nStep 09 is now publication-frozen. Use "
        "rcse_step09_frozen_results_manifest.json as the authoritative source "
        "for all Step 09 manuscript numbers."
    )


if __name__ == "__main__":
    main()
