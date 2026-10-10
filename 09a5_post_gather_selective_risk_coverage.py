"""
09a5_post_gather_selective_risk_coverage.py

Evaluate the calibrated post-GATHER safety probability as a selective
execution score.

Goal
----
Determine whether:

    q_t1 = P(SAFE_TO_EXECUTE | O_t1)

supports a useful autonomy-risk frontier even though default 0.5-threshold
classification has weak NOT_SAFE recall.

Inputs
------
Prediction artifacts produced by Step 09a.4:

    post_gather_predictions_grouped_iid.parquet
    post_gather_predictions_temporal.parquet
    post_gather_predictions_ood_exposure.parquet

Each file must contain at least:

    y_t1_d1
    p_safe_t1_cal
    transaction_exposure_eur

Outputs
-------
post_gather_fixed_threshold_results.csv
post_gather_exact_risk_coverage.csv
post_gather_risk_cap_operating_points.csv
post_gather_coverage_target_operating_points.csv
post_gather_regime_summary.csv
post_gather_selective_risk_metadata.json

Optional plots:
post_gather_risk_coverage.png
post_gather_weighted_risk_coverage.png

Interpretation
--------------
For threshold tau, execute only when:

    p_safe_t1_cal >= tau

Coverage:
    selected executions / all resolved test cases

Unsafe execution rate:
    NOT_SAFE cases among selected executions

Exposure-weighted unsafe execution rate:
    exposure-weighted NOT_SAFE mass among selected executions

This is a DIAGNOSTIC analysis. Thresholds selected from test-set operating
points must not later be presented as prospectively tuned enterprise policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"

REGIME_FILES = {
    "grouped_iid": "post_gather_predictions_grouped_iid.parquet",
    "temporal": "post_gather_predictions_temporal.parquet",
    "ood_exposure": "post_gather_predictions_ood_exposure.parquet",
}

DEFAULT_THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999,
]

RISK_CAPS = [0.01, 0.02, 0.05, 0.10]
COVERAGE_TARGETS = [0.10, 0.25, 0.50, 0.75, 0.90]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09a.5: post-GATHER selective risk-coverage analysis."
    )

    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Step 09a.4 post_gather_predictions_*.parquet files.",
    )

    p.add_argument(
        "--out",
        default=None,
        help="Default: <input-dir>/selective_risk_coverage",
    )

    p.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Fixed probability thresholds for diagnostic operating points.",
    )

    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG risk-coverage plots.",
    )

    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], regime: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{regime}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_exposure(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").abs()

    # Use 0 only where exposure is genuinely absent for weighting purposes.
    # We report missing exposure separately.
    return x.fillna(0.0)


def evaluate_selection(
    df: pd.DataFrame,
    selected_mask: pd.Series,
) -> dict[str, Any]:
    n = len(df)
    selected = df.loc[selected_mask].copy()

    n_selected = len(selected)

    if n_selected == 0:
        return {
            "selected_count": 0,
            "coverage": 0.0,
            "safe_selected": 0,
            "unsafe_selected": 0,
            "unsafe_execution_rate": np.nan,
            "safe_execution_rate": np.nan,
            "selected_exposure_sum_eur": 0.0,
            "unsafe_selected_exposure_eur": 0.0,
            "exposure_weighted_unsafe_rate": np.nan,
            "mean_p_safe_selected": np.nan,
            "median_p_safe_selected": np.nan,
        }

    unsafe = selected["y_t1_d1"] == UNSAFE_LABEL
    safe = selected["y_t1_d1"] == SAFE_LABEL

    exposure = normalize_exposure(
        selected["transaction_exposure_eur"]
    )

    unsafe_exposure = float(
        exposure.loc[unsafe].sum()
    )
    total_exposure = float(
        exposure.sum()
    )

    return {
        "selected_count": int(n_selected),
        "coverage": float(n_selected / n),
        "safe_selected": int(safe.sum()),
        "unsafe_selected": int(unsafe.sum()),
        "unsafe_execution_rate": float(
            unsafe.mean()
        ),
        "safe_execution_rate": float(
            safe.mean()
        ),
        "selected_exposure_sum_eur": total_exposure,
        "unsafe_selected_exposure_eur": unsafe_exposure,
        "exposure_weighted_unsafe_rate": (
            unsafe_exposure / total_exposure
            if total_exposure > 0
            else np.nan
        ),
        "mean_p_safe_selected": float(
            selected["p_safe_t1_cal"].mean()
        ),
        "median_p_safe_selected": float(
            selected["p_safe_t1_cal"].median()
        ),
    }


def fixed_threshold_table(
    df: pd.DataFrame,
    regime: str,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []

    for tau in sorted(set(thresholds)):
        selected_mask = df["p_safe_t1_cal"] >= tau
        result = evaluate_selection(
            df,
            selected_mask,
        )

        rows.append(
            {
                "regime": regime,
                "threshold": float(tau),
                **result,
            }
        )

    return pd.DataFrame(rows)


def exact_risk_coverage_curve(
    df: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    """
    Exact selective curve by sorting highest q_t1 first.

    Row k corresponds to autonomously executing the top k cases ranked by
    calibrated safety probability.
    """
    work = df.copy()

    work["is_unsafe"] = (
        work["y_t1_d1"] == UNSAFE_LABEL
    ).astype(int)

    work["exposure_weight"] = normalize_exposure(
        work["transaction_exposure_eur"]
    )

    work["unsafe_exposure"] = (
        work["is_unsafe"]
        * work["exposure_weight"]
    )

    work = work.sort_values(
        ["p_safe_t1_cal", "case_id"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    n = len(work)

    work["rank"] = np.arange(
        1,
        n + 1,
        dtype=np.int64,
    )

    work["coverage"] = work["rank"] / n

    work["cum_unsafe"] = (
        work["is_unsafe"].cumsum()
    )

    work["unsafe_execution_rate"] = (
        work["cum_unsafe"]
        / work["rank"]
    )

    work["cum_selected_exposure_eur"] = (
        work["exposure_weight"].cumsum()
    )

    work["cum_unsafe_exposure_eur"] = (
        work["unsafe_exposure"].cumsum()
    )

    denom = work[
        "cum_selected_exposure_eur"
    ].replace(0.0, np.nan)

    work["exposure_weighted_unsafe_rate"] = (
        work["cum_unsafe_exposure_eur"]
        / denom
    )

    work["selection_threshold"] = (
        work["p_safe_t1_cal"]
    )

    out = work[
        [
            "rank",
            "coverage",
            "selection_threshold",
            "cum_unsafe",
            "unsafe_execution_rate",
            "cum_selected_exposure_eur",
            "cum_unsafe_exposure_eur",
            "exposure_weighted_unsafe_rate",
        ]
    ].copy()

    out.insert(
        0,
        "regime",
        regime,
    )

    return out


def area_under_risk_coverage(
    curve: pd.DataFrame,
    risk_col: str,
) -> float:
    temp = curve[
        ["coverage", risk_col]
    ].dropna()

    if len(temp) < 2:
        return np.nan

    x = temp["coverage"].to_numpy()
    y = temp[risk_col].to_numpy()

    return float(
        np.trapz(y, x)
    )


def risk_cap_points(
    curve: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    """
    Maximum observed coverage whose cumulative unsafe risk is <= cap.

    Diagnostic only; this uses test outcomes and therefore is not a
    prospectively selected operational threshold.
    """
    rows = []

    for cap in RISK_CAPS:
        eligible = curve[
            curve["unsafe_execution_rate"] <= cap
        ]

        if eligible.empty:
            rows.append(
                {
                    "regime": regime,
                    "risk_cap": cap,
                    "max_coverage": 0.0,
                    "threshold_at_max_coverage": np.nan,
                    "unsafe_execution_rate": np.nan,
                    "exposure_weighted_unsafe_rate": np.nan,
                    "selected_count": 0,
                }
            )
            continue

        row = eligible.iloc[
            eligible["coverage"].argmax()
        ]

        rows.append(
            {
                "regime": regime,
                "risk_cap": cap,
                "max_coverage": float(
                    row["coverage"]
                ),
                "threshold_at_max_coverage": float(
                    row["selection_threshold"]
                ),
                "unsafe_execution_rate": float(
                    row["unsafe_execution_rate"]
                ),
                "exposure_weighted_unsafe_rate": float(
                    row["exposure_weighted_unsafe_rate"]
                )
                if pd.notna(
                    row["exposure_weighted_unsafe_rate"]
                )
                else np.nan,
                "selected_count": int(
                    row["rank"]
                ),
            }
        )

    return pd.DataFrame(rows)


def coverage_target_points(
    curve: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    """
    Risk at approximately fixed target coverages.
    """
    rows = []

    for target in COVERAGE_TARGETS:
        idx = (
            curve["coverage"] - target
        ).abs().idxmin()

        row = curve.loc[idx]

        rows.append(
            {
                "regime": regime,
                "coverage_target": target,
                "actual_coverage": float(
                    row["coverage"]
                ),
                "threshold": float(
                    row["selection_threshold"]
                ),
                "unsafe_execution_rate": float(
                    row["unsafe_execution_rate"]
                ),
                "exposure_weighted_unsafe_rate": (
                    float(
                        row[
                            "exposure_weighted_unsafe_rate"
                        ]
                    )
                    if pd.notna(
                        row[
                            "exposure_weighted_unsafe_rate"
                        ]
                    )
                    else np.nan
                ),
                "selected_count": int(
                    row["rank"]
                ),
            }
        )

    return pd.DataFrame(rows)


def regime_summary(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    regime: str,
) -> dict[str, Any]:
    unsafe = (
        df["y_t1_d1"] == UNSAFE_LABEL
    )

    exposure = normalize_exposure(
        df["transaction_exposure_eur"]
    )

    unsafe_exposure = float(
        exposure.loc[unsafe].sum()
    )

    total_exposure = float(
        exposure.sum()
    )

    return {
        "regime": regime,
        "n_cases": int(len(df)),
        "safe_cases": int(
            (df["y_t1_d1"] == SAFE_LABEL).sum()
        ),
        "unsafe_cases": int(unsafe.sum()),
        "unsafe_prevalence": float(
            unsafe.mean()
        ),
        "missing_exposure_count": int(
            pd.to_numeric(
                df["transaction_exposure_eur"],
                errors="coerce",
            ).isna().sum()
        ),
        "total_exposure_eur": total_exposure,
        "unsafe_exposure_eur": unsafe_exposure,
        "full_coverage_exposure_weighted_unsafe_rate": (
            unsafe_exposure / total_exposure
            if total_exposure > 0
            else np.nan
        ),
        "mean_p_safe": float(
            df["p_safe_t1_cal"].mean()
        ),
        "median_p_safe": float(
            df["p_safe_t1_cal"].median()
        ),
        "min_p_safe": float(
            df["p_safe_t1_cal"].min()
        ),
        "max_p_safe": float(
            df["p_safe_t1_cal"].max()
        ),
        "aurc_unsafe": area_under_risk_coverage(
            curve,
            "unsafe_execution_rate",
        ),
        "aurc_exposure_weighted": area_under_risk_coverage(
            curve,
            "exposure_weighted_unsafe_rate",
        ),
    }


def make_plots(
    exact_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    # Unweighted risk-coverage
    fig, ax = plt.subplots(figsize=(8, 6))

    for regime, g in exact_df.groupby("regime"):
        ax.plot(
            g["coverage"],
            g["unsafe_execution_rate"],
            label=regime,
        )

    ax.set_xlabel("Autonomous execution coverage")
    ax.set_ylabel("Unsafe execution rate")
    ax.set_title(
        "Post-GATHER selective risk–coverage"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        out_dir
        / "post_gather_risk_coverage.png",
        dpi=200,
    )
    plt.close(fig)

    # Exposure-weighted risk-coverage
    fig, ax = plt.subplots(figsize=(8, 6))

    for regime, g in exact_df.groupby("regime"):
        ax.plot(
            g["coverage"],
            g["exposure_weighted_unsafe_rate"],
            label=regime,
        )

    ax.set_xlabel("Autonomous execution coverage")
    ax.set_ylabel(
        "Exposure-weighted unsafe execution rate"
    )
    ax.set_title(
        "Post-GATHER exposure-weighted risk–coverage"
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    fig.savefig(
        out_dir
        / "post_gather_weighted_risk_coverage.png",
        dpi=200,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = Path(
        args.input_dir
    ).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_dir / "selective_risk_coverage"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("RCSE STEP 09a.5 - POST-GATHER SELECTIVE RISK-COVERAGE")
    print("=" * 80)
    print(f"Input : {input_dir}")
    print(f"Output: {out_dir}")

    fixed_frames = []
    exact_frames = []
    risk_cap_frames = []
    coverage_frames = []
    summary_rows = []

    for regime, filename in REGIME_FILES.items():
        path = input_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {regime} prediction file: {path}"
            )

        print(f"\n[{regime}] loading {filename} ...")

        df = pd.read_parquet(path)

        require_columns(
            df,
            [
                "case_id",
                "y_t1_d1",
                "p_safe_t1_cal",
                "transaction_exposure_eur",
            ],
            regime,
        )

        # Only resolved binary cases belong in Step 09a.5.
        df = df[
            df["y_t1_d1"].isin(
                [SAFE_LABEL, UNSAFE_LABEL]
            )
        ].copy()

        if df.empty:
            raise RuntimeError(
                f"{regime}: no resolved post-GATHER cases."
            )

        if (
            (df["p_safe_t1_cal"] < 0)
            | (df["p_safe_t1_cal"] > 1)
        ).any():
            raise ValueError(
                f"{regime}: p_safe_t1_cal outside [0,1]."
            )

        fixed = fixed_threshold_table(
            df,
            regime,
            args.thresholds,
        )

        exact = exact_risk_coverage_curve(
            df,
            regime,
        )

        risk_caps = risk_cap_points(
            exact,
            regime,
        )

        coverage_points = coverage_target_points(
            exact,
            regime,
        )

        summary = regime_summary(
            df,
            exact,
            regime,
        )

        fixed_frames.append(fixed)
        exact_frames.append(exact)
        risk_cap_frames.append(risk_caps)
        coverage_frames.append(coverage_points)
        summary_rows.append(summary)

        print(
            f"   Cases={len(df):,} | "
            f"unsafe prevalence={summary['unsafe_prevalence']:.4f} | "
            f"AURC={summary['aurc_unsafe']:.4f}"
        )

    fixed_df = pd.concat(
        fixed_frames,
        ignore_index=True,
    )

    exact_df = pd.concat(
        exact_frames,
        ignore_index=True,
    )

    risk_cap_df = pd.concat(
        risk_cap_frames,
        ignore_index=True,
    )

    coverage_df = pd.concat(
        coverage_frames,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    fixed_df.to_csv(
        out_dir
        / "post_gather_fixed_threshold_results.csv",
        index=False,
    )

    exact_df.to_csv(
        out_dir
        / "post_gather_exact_risk_coverage.csv",
        index=False,
    )

    risk_cap_df.to_csv(
        out_dir
        / "post_gather_risk_cap_operating_points.csv",
        index=False,
    )

    coverage_df.to_csv(
        out_dir
        / "post_gather_coverage_target_operating_points.csv",
        index=False,
    )

    summary_df.to_csv(
        out_dir
        / "post_gather_regime_summary.csv",
        index=False,
    )

    metadata = {
        "step": "09a.5",
        "input_directory": str(input_dir),
        "probability_score": "p_safe_t1_cal",
        "selection_rule": "EXECUTE iff p_safe_t1_cal >= threshold",
        "target": {
            "safe": SAFE_LABEL,
            "unsafe": UNSAFE_LABEL,
        },
        "fixed_thresholds": sorted(
            set(float(x) for x in args.thresholds)
        ),
        "risk_caps": RISK_CAPS,
        "coverage_targets": COVERAGE_TARGETS,
        "primary_metrics": [
            "coverage",
            "unsafe_execution_rate",
            "exposure_weighted_unsafe_rate",
        ],
        "important_interpretation_rule": (
            "Risk-cap and coverage-target operating points are descriptive "
            "test-set diagnostics. They must not later be presented as "
            "prospectively tuned operational thresholds."
        ),
        "decision_criterion": (
            "Retain q_t1 as a VoI/selective-execution input only if higher "
            "p_safe_t1_cal yields materially lower unsafe risk at useful "
            "coverage. If no meaningful separation exists, do not treat "
            "q_t1 as a reliable quantitative post-GATHER safety estimate."
        ),
    }

    with open(
        out_dir
        / "post_gather_selective_risk_metadata.json",
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
            exact_df,
            out_dir,
        )

    print("\n" + "=" * 80)
    print("STEP 09a.5 COMPLETE")
    print("=" * 80)

    print("\nRegime summary:")
    print(
        summary_df[
            [
                "regime",
                "n_cases",
                "unsafe_prevalence",
                "aurc_unsafe",
                "aurc_exposure_weighted",
            ]
        ].to_string(index=False)
    )

    print("\nCoverage-target operating points:")
    print(
        coverage_df[
            [
                "regime",
                "coverage_target",
                "actual_coverage",
                "threshold",
                "unsafe_execution_rate",
                "exposure_weighted_unsafe_rate",
            ]
        ].to_string(index=False)
    )

    print("\nRisk-cap operating points:")
    print(
        risk_cap_df[
            [
                "regime",
                "risk_cap",
                "max_coverage",
                "threshold_at_max_coverage",
                "unsafe_execution_rate",
                "exposure_weighted_unsafe_rate",
            ]
        ].to_string(index=False)
    )

    print("\nOutputs:")
    for name in [
        "post_gather_fixed_threshold_results.csv",
        "post_gather_exact_risk_coverage.csv",
        "post_gather_risk_cap_operating_points.csv",
        "post_gather_coverage_target_operating_points.csv",
        "post_gather_regime_summary.csv",
        "post_gather_selective_risk_metadata.json",
        None if args.no_plots else "post_gather_risk_coverage.png",
        None if args.no_plots else "post_gather_weighted_risk_coverage.png",
    ]:
        if name:
            print(f"  - {out_dir / name}")

    print(
        "\nNext decision: determine whether q_t1 provides a useful selective "
        "risk-coverage frontier before integrating it into VoI/RCSE."
    )


if __name__ == "__main__":
    main()
