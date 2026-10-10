#!/usr/bin/env python
"""
09b2g_t0_safety_selective_risk_coverage.py

RCSE Step 09b.2g
Selective risk-coverage analysis for the dedicated t0 execution-safety model.

Purpose
-------
Evaluate whether the dedicated safety score:

    q_t0 = P(SAFE_TO_EXECUTE | O_t0)

supports useful autonomous execution under selective thresholds.

This script evaluates BOTH:
    - p_safe_t0_raw
    - p_safe_t0_cal

No retraining, recalibration, or policy retuning is performed.

Inputs
------
Directory from Step 09b.2f containing:
    t0_safety_predictions_grouped_iid.parquet
    t0_safety_predictions_temporal.parquet
    t0_safety_predictions_ood_exposure.parquet

Outputs
-------
t0_safety_fixed_threshold_results.csv
t0_safety_exact_risk_coverage.csv
t0_safety_coverage_target_operating_points.csv
t0_safety_risk_cap_operating_points.csv
t0_safety_regime_summary.csv
t0_safety_raw_vs_calibrated_frontier_summary.csv
t0_safety_selective_risk_metadata.json
t0_safety_risk_coverage_raw.png
t0_safety_risk_coverage_calibrated.png
t0_safety_weighted_risk_coverage_raw.png
t0_safety_weighted_risk_coverage_calibrated.png

Primary quantities
------------------
For a threshold tau:

    EXECUTE iff p_safe >= tau

Coverage:
    selected executions / all resolved t0 safety test cases

Unsafe execution rate:
    P(NOT_SAFE_TO_EXECUTE | selected for execution)

Exposure-weighted unsafe rate:
    sum(exposure on selected unsafe cases)
    /
    sum(exposure on all selected cases)

Important
---------
Risk-cap operating points use held-out test outcomes and are therefore
DESCRIPTIVE ONLY. They are not prospectively tuned enterprise thresholds.
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
    "grouped_iid": "t0_safety_predictions_grouped_iid.parquet",
    "temporal": "t0_safety_predictions_temporal.parquet",
    "ood_exposure": "t0_safety_predictions_ood_exposure.parquet",
}

SCORES = {
    "RAW": "p_safe_t0_raw",
    "CALIBRATED": "p_safe_t0_cal",
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

COVERAGE_TARGETS = [
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
]

RISK_CAPS = [
    0.01,
    0.02,
    0.05,
    0.10,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2g: t0 safety selective risk-coverage analysis."
    )

    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing Step 09b.2f t0_safety_predictions_*.parquet",
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
        help="Fixed descriptive safety-score thresholds.",
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


def normalize_exposure(
    s: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            s,
            errors="coerce",
        )
        .abs()
        .fillna(0.0)
    )


def evaluate_selection(
    df: pd.DataFrame,
    selected: pd.Series,
    score_col: str,
) -> dict[str, Any]:

    n = len(df)

    chosen = df.loc[
        selected
    ].copy()

    n_selected = len(
        chosen
    )

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
            "mean_safety_score_selected": np.nan,
            "median_safety_score_selected": np.nan,
        }

    unsafe = (
        chosen[
            "y_t0_safety_d1"
        ]
        == UNSAFE_LABEL
    )

    safe = (
        chosen[
            "y_t0_safety_d1"
        ]
        == SAFE_LABEL
    )

    exposure = normalize_exposure(
        chosen[
            "transaction_exposure_eur"
        ]
    )

    total_exposure = float(
        exposure.sum()
    )

    unsafe_exposure = float(
        exposure.loc[
            unsafe
        ].sum()
    )

    return {
        "selected_count": int(
            n_selected
        ),
        "coverage": float(
            n_selected / n
        ),
        "safe_selected": int(
            safe.sum()
        ),
        "unsafe_selected": int(
            unsafe.sum()
        ),
        "unsafe_execution_rate": float(
            unsafe.mean()
        ),
        "safe_execution_rate": float(
            safe.mean()
        ),
        "selected_exposure_sum_eur": total_exposure,
        "unsafe_selected_exposure_eur": unsafe_exposure,
        "exposure_weighted_unsafe_rate": (
            unsafe_exposure
            / total_exposure
            if total_exposure > 0
            else np.nan
        ),
        "mean_safety_score_selected": float(
            chosen[
                score_col
            ].mean()
        ),
        "median_safety_score_selected": float(
            chosen[
                score_col
            ].median()
        ),
    }


def fixed_threshold_table(
    df: pd.DataFrame,
    regime: str,
    score_type: str,
    score_col: str,
    thresholds: list[float],
) -> pd.DataFrame:

    rows = []

    for tau in sorted(
        set(
            float(x)
            for x in thresholds
        )
    ):

        selected = (
            df[
                score_col
            ]
            >= tau
        )

        result = evaluate_selection(
            df=df,
            selected=selected,
            score_col=score_col,
        )

        rows.append(
            {
                "regime": regime,
                "score_type": score_type,
                "threshold": tau,
                **result,
            }
        )

    return pd.DataFrame(
        rows
    )


def exact_curve(
    df: pd.DataFrame,
    regime: str,
    score_type: str,
    score_col: str,
) -> pd.DataFrame:
    """
    Sort cases from highest safety score to lowest.
    Each prefix is a selective autonomous-execution set.
    """

    work = df.copy()

    work[
        "is_unsafe"
    ] = (
        work[
            "y_t0_safety_d1"
        ]
        == UNSAFE_LABEL
    ).astype(int)

    work[
        "exposure_weight"
    ] = normalize_exposure(
        work[
            "transaction_exposure_eur"
        ]
    )

    work[
        "unsafe_exposure"
    ] = (
        work[
            "is_unsafe"
        ]
        * work[
            "exposure_weight"
        ]
    )

    sort_cols = [
        score_col,
    ]

    ascending = [
        False,
    ]

    if "source_case_id" in work.columns:
        sort_cols.append(
            "source_case_id"
        )
        ascending.append(
            True
        )

    work = work.sort_values(
        sort_cols,
        ascending=ascending,
        kind="stable",
    ).reset_index(
        drop=True
    )

    n = len(
        work
    )

    work[
        "rank"
    ] = np.arange(
        1,
        n + 1,
        dtype=np.int64,
    )

    work[
        "coverage"
    ] = (
        work[
            "rank"
        ]
        / n
    )

    work[
        "cum_unsafe"
    ] = (
        work[
            "is_unsafe"
        ]
        .cumsum()
    )

    work[
        "unsafe_execution_rate"
    ] = (
        work[
            "cum_unsafe"
        ]
        / work[
            "rank"
        ]
    )

    work[
        "cum_selected_exposure_eur"
    ] = (
        work[
            "exposure_weight"
        ]
        .cumsum()
    )

    work[
        "cum_unsafe_exposure_eur"
    ] = (
        work[
            "unsafe_exposure"
        ]
        .cumsum()
    )

    denom = (
        work[
            "cum_selected_exposure_eur"
        ]
        .replace(
            0.0,
            np.nan,
        )
    )

    work[
        "exposure_weighted_unsafe_rate"
    ] = (
        work[
            "cum_unsafe_exposure_eur"
        ]
        / denom
    )

    work[
        "selection_threshold"
    ] = (
        work[
            score_col
        ]
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
        "score_type",
        score_type,
    )

    out.insert(
        0,
        "regime",
        regime,
    )

    return out


def area_under_curve(
    curve: pd.DataFrame,
    risk_col: str,
) -> float:

    temp = curve[
        [
            "coverage",
            risk_col,
        ]
    ].dropna()

    if len(
        temp
    ) < 2:
        return np.nan

    return float(
        np.trapz(
            temp[
                risk_col
            ].to_numpy(),
            temp[
                "coverage"
            ].to_numpy(),
        )
    )


def coverage_target_points(
    curve: pd.DataFrame,
    regime: str,
    score_type: str,
) -> pd.DataFrame:

    rows = []

    for target in COVERAGE_TARGETS:
        idx = (
            curve[
                "coverage"
            ]
            - target
        ).abs().idxmin()

        row = curve.loc[
            idx
        ]

        rows.append(
            {
                "regime": regime,
                "score_type": score_type,
                "coverage_target": target,
                "actual_coverage": float(
                    row[
                        "coverage"
                    ]
                ),
                "threshold": float(
                    row[
                        "selection_threshold"
                    ]
                ),
                "unsafe_execution_rate": float(
                    row[
                        "unsafe_execution_rate"
                    ]
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
                    row[
                        "rank"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def risk_cap_points(
    curve: pd.DataFrame,
    regime: str,
    score_type: str,
) -> pd.DataFrame:
    """
    Maximum observed test coverage at or below each realized unsafe-risk cap.

    DESCRIPTIVE ONLY.
    """

    rows = []

    for cap in RISK_CAPS:
        eligible = curve[
            curve[
                "unsafe_execution_rate"
            ]
            <= cap
        ]

        if eligible.empty:
            rows.append(
                {
                    "regime": regime,
                    "score_type": score_type,
                    "risk_cap": cap,
                    "max_coverage": 0.0,
                    "threshold_at_max_coverage": np.nan,
                    "unsafe_execution_rate": np.nan,
                    "exposure_weighted_unsafe_rate": np.nan,
                    "selected_count": 0,
                }
            )

            continue

        row = eligible.loc[
            eligible[
                "coverage"
            ].idxmax()
        ]

        rows.append(
            {
                "regime": regime,
                "score_type": score_type,
                "risk_cap": cap,
                "max_coverage": float(
                    row[
                        "coverage"
                    ]
                ),
                "threshold_at_max_coverage": float(
                    row[
                        "selection_threshold"
                    ]
                ),
                "unsafe_execution_rate": float(
                    row[
                        "unsafe_execution_rate"
                    ]
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
                    row[
                        "rank"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def regime_summary(
    df: pd.DataFrame,
    curve: pd.DataFrame,
    regime: str,
    score_type: str,
    score_col: str,
) -> dict[str, Any]:

    unsafe = (
        df[
            "y_t0_safety_d1"
        ]
        == UNSAFE_LABEL
    )

    exposure = normalize_exposure(
        df[
            "transaction_exposure_eur"
        ]
    )

    total_exposure = float(
        exposure.sum()
    )

    unsafe_exposure = float(
        exposure.loc[
            unsafe
        ].sum()
    )

    return {
        "regime": regime,
        "score_type": score_type,
        "n_cases": int(
            len(df)
        ),
        "safe_cases": int(
            (
                df[
                    "y_t0_safety_d1"
                ]
                == SAFE_LABEL
            ).sum()
        ),
        "unsafe_cases": int(
            unsafe.sum()
        ),
        "unsafe_prevalence": float(
            unsafe.mean()
        ),
        "missing_exposure_count": int(
            pd.to_numeric(
                df[
                    "transaction_exposure_eur"
                ],
                errors="coerce",
            )
            .isna()
            .sum()
        ),
        "full_coverage_exposure_weighted_unsafe_rate": (
            unsafe_exposure
            / total_exposure
            if total_exposure > 0
            else np.nan
        ),
        "mean_safety_score": float(
            df[
                score_col
            ].mean()
        ),
        "median_safety_score": float(
            df[
                score_col
            ].median()
        ),
        "min_safety_score": float(
            df[
                score_col
            ].min()
        ),
        "max_safety_score": float(
            df[
                score_col
            ].max()
        ),
        "aurc_unsafe": area_under_curve(
            curve,
            "unsafe_execution_rate",
        ),
        "aurc_exposure_weighted": area_under_curve(
            curve,
            "exposure_weighted_unsafe_rate",
        ),
    }


def raw_vs_cal_summary(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for regime in REGIME_FILES:
        sub = summary_df[
            summary_df[
                "regime"
            ]
            == regime
        ].set_index(
            "score_type"
        )

        if not {
            "RAW",
            "CALIBRATED",
        }.issubset(
            sub.index
        ):
            continue

        rows.append(
            {
                "regime": regime,
                "unsafe_prevalence": float(
                    sub.loc[
                        "RAW",
                        "unsafe_prevalence",
                    ]
                ),
                "aurc_unsafe_raw": float(
                    sub.loc[
                        "RAW",
                        "aurc_unsafe",
                    ]
                ),
                "aurc_unsafe_calibrated": float(
                    sub.loc[
                        "CALIBRATED",
                        "aurc_unsafe",
                    ]
                ),
                "aurc_delta_cal_minus_raw": float(
                    sub.loc[
                        "CALIBRATED",
                        "aurc_unsafe",
                    ]
                    - sub.loc[
                        "RAW",
                        "aurc_unsafe",
                    ]
                ),
                "weighted_aurc_raw": float(
                    sub.loc[
                        "RAW",
                        "aurc_exposure_weighted",
                    ]
                ),
                "weighted_aurc_calibrated": float(
                    sub.loc[
                        "CALIBRATED",
                        "aurc_exposure_weighted",
                    ]
                ),
                "weighted_aurc_delta_cal_minus_raw": float(
                    sub.loc[
                        "CALIBRATED",
                        "aurc_exposure_weighted",
                    ]
                    - sub.loc[
                        "RAW",
                        "aurc_exposure_weighted",
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def make_plots(
    exact_df: pd.DataFrame,
    out_dir: Path,
) -> None:

    import matplotlib.pyplot as plt

    for score_type in [
        "RAW",
        "CALIBRATED",
    ]:

        sub = exact_df[
            exact_df[
                "score_type"
            ]
            == score_type
        ]

        fig, ax = plt.subplots(
            figsize=(
                8,
                6,
            )
        )

        for regime, g in sub.groupby(
            "regime"
        ):
            ax.plot(
                g[
                    "coverage"
                ],
                g[
                    "unsafe_execution_rate"
                ],
                label=regime,
            )

        ax.set_xlabel(
            "Autonomous execution coverage"
        )

        ax.set_ylabel(
            "Unsafe execution rate"
        )

        ax.set_title(
            f"t0 safety selective risk–coverage ({score_type.lower()})"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            out_dir
            / (
                "t0_safety_risk_coverage_"
                + score_type.lower()
                + ".png"
            ),
            dpi=200,
        )

        plt.close(
            fig
        )

        fig, ax = plt.subplots(
            figsize=(
                8,
                6,
            )
        )

        for regime, g in sub.groupby(
            "regime"
        ):
            ax.plot(
                g[
                    "coverage"
                ],
                g[
                    "exposure_weighted_unsafe_rate"
                ],
                label=regime,
            )

        ax.set_xlabel(
            "Autonomous execution coverage"
        )

        ax.set_ylabel(
            "Exposure-weighted unsafe execution rate"
        )

        ax.set_title(
            "t0 safety exposure-weighted risk–coverage "
            f"({score_type.lower()})"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            out_dir
            / (
                "t0_safety_weighted_risk_coverage_"
                + score_type.lower()
                + ".png"
            ),
            dpi=200,
        )

        plt.close(
            fig
        )


def main() -> None:

    args = parse_args()

    input_dir = Path(
        args.input_dir
    ).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(
            input_dir
        )

    out_dir = (
        Path(
            args.out
        ).expanduser().resolve()
        if args.out
        else input_dir
        / "selective_risk_coverage"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2g - t0 SAFETY SELECTIVE RISK-COVERAGE"
    )
    print(
        "=" * 80
    )
    print(
        f"Input : {input_dir}"
    )
    print(
        f"Output: {out_dir}"
    )
    print(
        "Evaluating RAW and CALIBRATED dedicated safety scores."
    )
    print(
        "NO RETRAINING / NO RECALIBRATION / NO POLICY RETUNING"
    )

    fixed_frames = []
    exact_frames = []
    coverage_frames = []
    risk_cap_frames = []
    summary_rows = []

    for regime, filename in REGIME_FILES.items():

        path = input_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {path}"
            )

        print(
            f"\n[{regime}] loading {filename} ..."
        )

        df = pd.read_parquet(
            path
        )

        require_columns(
            df,
            [
                "source_case_id",
                "y_t0_safety_d1",
                "transaction_exposure_eur",
                *SCORES.values(),
            ],
            regime,
        )

        df = df[
            df[
                "y_t0_safety_d1"
            ].isin(
                [
                    SAFE_LABEL,
                    UNSAFE_LABEL,
                ]
            )
        ].copy()

        if df.empty:
            raise RuntimeError(
                f"{regime}: no resolved safety cases."
            )

        for score_type, score_col in SCORES.items():

            score = pd.to_numeric(
                df[
                    score_col
                ],
                errors="coerce",
            )

            if score.isna().any():
                raise ValueError(
                    f"{regime}/{score_type}: missing safety scores."
                )

            if (
                (score < 0)
                | (score > 1)
            ).any():
                raise ValueError(
                    f"{regime}/{score_type}: safety score outside [0,1]."
                )

            fixed = fixed_threshold_table(
                df=df,
                regime=regime,
                score_type=score_type,
                score_col=score_col,
                thresholds=args.thresholds,
            )

            curve = exact_curve(
                df=df,
                regime=regime,
                score_type=score_type,
                score_col=score_col,
            )

            coverage = coverage_target_points(
                curve=curve,
                regime=regime,
                score_type=score_type,
            )

            risk_caps = risk_cap_points(
                curve=curve,
                regime=regime,
                score_type=score_type,
            )

            summary = regime_summary(
                df=df,
                curve=curve,
                regime=regime,
                score_type=score_type,
                score_col=score_col,
            )

            fixed_frames.append(
                fixed
            )

            exact_frames.append(
                curve
            )

            coverage_frames.append(
                coverage
            )

            risk_cap_frames.append(
                risk_caps
            )

            summary_rows.append(
                summary
            )

            print(
                f"   {score_type:<10} "
                f"n={len(df):,} | "
                f"unsafe={summary['unsafe_prevalence']:.4f} | "
                f"AURC={summary['aurc_unsafe']:.4f} | "
                f"weighted AURC={summary['aurc_exposure_weighted']:.4f}"
            )

    fixed_df = pd.concat(
        fixed_frames,
        ignore_index=True,
    )

    exact_df = pd.concat(
        exact_frames,
        ignore_index=True,
    )

    coverage_df = pd.concat(
        coverage_frames,
        ignore_index=True,
    )

    risk_cap_df = pd.concat(
        risk_cap_frames,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    compare_df = raw_vs_cal_summary(
        summary_df
    )

    fixed_df.to_csv(
        out_dir
        / "t0_safety_fixed_threshold_results.csv",
        index=False,
    )

    exact_df.to_csv(
        out_dir
        / "t0_safety_exact_risk_coverage.csv",
        index=False,
    )

    coverage_df.to_csv(
        out_dir
        / "t0_safety_coverage_target_operating_points.csv",
        index=False,
    )

    risk_cap_df.to_csv(
        out_dir
        / "t0_safety_risk_cap_operating_points.csv",
        index=False,
    )

    summary_df.to_csv(
        out_dir
        / "t0_safety_regime_summary.csv",
        index=False,
    )

    compare_df.to_csv(
        out_dir
        / "t0_safety_raw_vs_calibrated_frontier_summary.csv",
        index=False,
    )

    metadata = {
        "step": "09b.2g",
        "purpose": (
            "Evaluate the dedicated natural-only t0 execution-safety "
            "estimator as a selective autonomous-execution score."
        ),
        "scores": SCORES,
        "selection_rule": (
            "EXECUTE iff chosen p_safe_t0 score >= threshold"
        ),
        "thresholds": sorted(
            set(
                float(x)
                for x in args.thresholds
            )
        ),
        "coverage_targets": COVERAGE_TARGETS,
        "risk_caps": RISK_CAPS,
        "primary_metrics": [
            "coverage",
            "unsafe_execution_rate",
            "exposure_weighted_unsafe_rate",
            "AURC",
        ],
        "retraining_performed": False,
        "recalibration_performed": False,
        "policy_retuning_performed": False,
        "important_guardrail": (
            "Risk-cap and coverage-target operating points use held-out "
            "test outcomes and are descriptive diagnostics only. They must "
            "not be reused as prospectively tuned policy thresholds."
        ),
        "next_decision": (
            "If the dedicated safety score provides strong selective "
            "risk-coverage behavior, freeze it as the execution-safety "
            "belief input and define a revised RCSE policy specification "
            "before any rerun. If R4 ranking remains poor, use a conservative "
            "OOD fallback rather than claiming universal calibration."
        ),
    }

    with open(
        out_dir
        / "t0_safety_selective_risk_metadata.json",
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
            exact_df=exact_df,
            out_dir=out_dir,
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 09b.2g COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nRegime/score summary:"
    )

    print(
        summary_df[
            [
                "regime",
                "score_type",
                "n_cases",
                "unsafe_prevalence",
                "aurc_unsafe",
                "aurc_exposure_weighted",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nRaw vs calibrated frontier comparison:"
    )

    print(
        compare_df.to_string(
            index=False
        )
    )

    print(
        "\nCoverage-target operating points:"
    )

    print(
        coverage_df[
            [
                "regime",
                "score_type",
                "coverage_target",
                "actual_coverage",
                "threshold",
                "unsafe_execution_rate",
                "exposure_weighted_unsafe_rate",
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
        "\nNext: determine whether the dedicated t0 safety score is strong "
        "enough to become the execution-risk belief in a revised frozen "
        "RCSE policy specification."
    )


if __name__ == "__main__":
    main()
