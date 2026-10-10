"""
09b1_build_rcse_policy_dataset.py

Build the policy-ready RCSE evaluation dataset by joining:

1. Frozen t0 action probabilities from Step 08c
2. Post-GATHER t1 safety probabilities from Step 09a.4
3. Natural GATHER eligibility / transition availability

This script DOES NOT yet choose RCSE actions or tune costs.

Why this step exists
--------------------
The t1 post-GATHER estimator is valid only for real delayed-GR transitions.
It should not be silently applied to every synthetic benchmark episode.

Therefore this script creates an explicit policy-ready dataset with:

    gather_transition_available
    gather_policy_eligible
    p_safe_t1_cal
    p_not_safe_t1_cal

and leaves ineligible episodes with missing t1 continuation scores.

Inputs
------
--t0-dir
    Step 08c frozen_baseline_results directory

--t1-dir
    Step 09a.4 post_gather_safety_results directory

Expected files
--------------
t0:
    frozen_predictions_grouped_iid.parquet
    frozen_predictions_temporal.parquet
    frozen_predictions_ood_exposure.parquet

t1:
    post_gather_predictions_grouped_iid.parquet
    post_gather_predictions_temporal.parquet
    post_gather_predictions_ood_exposure.parquet

Outputs
-------
rcse_policy_dataset_grouped_iid.parquet
rcse_policy_dataset_temporal.parquet
rcse_policy_dataset_ood_exposure.parquet

rcse_policy_dataset_summary.csv
rcse_gather_eligibility_summary.csv
rcse_policy_dataset_metadata.json

Methodological rule
-------------------
GATHER is eligible only when ALL of the following are true:

1. A real post-GATHER transition exists for the source case.
2. The t0 experimental reference response is GATHER.
3. The episode is not a synthetic contradiction / synthetic evidence-loss
   condition requiring different semantics.

This conservative rule makes the first RCSE implementation use natural
delayed-evidence GATHER episodes only.

The rule can later be expanded in a separately documented sensitivity test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = [
    "grouped_iid",
    "temporal",
    "ood_exposure",
]

T0_FILE = {
    "grouped_iid": "frozen_predictions_grouped_iid.parquet",
    "temporal": "frozen_predictions_temporal.parquet",
    "ood_exposure": "frozen_predictions_ood_exposure.parquet",
}

T1_FILE = {
    "grouped_iid": "post_gather_predictions_grouped_iid.parquet",
    "temporal": "post_gather_predictions_temporal.parquet",
    "ood_exposure": "post_gather_predictions_ood_exposure.parquet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.1: build RCSE policy-ready evaluation datasets."
    )

    p.add_argument(
        "--t0-dir",
        required=True,
        help="Directory containing Step 08c frozen_predictions_*.parquet",
    )

    p.add_argument(
        "--t1-dir",
        required=True,
        help="Directory containing Step 09a.4 post_gather_predictions_*.parquet",
    )

    p.add_argument(
        "--out",
        default=None,
        help="Default: <t0-dir>/rcse_policy_dataset",
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


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def infer_synthetic_flag(df: pd.DataFrame) -> pd.Series:
    """
    Prefer explicit synthetic_intervention if present.
    Otherwise infer from intervention_type.
    """
    if "synthetic_intervention" in df.columns:
        s = df["synthetic_intervention"]

        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)

        return (
            s.fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes", "y", "t"})
        )

    if "intervention_type" in df.columns:
        txt = normalize_text(df["intervention_type"])
        return ~txt.isin({"", "NONE", "NATURAL", "NO_INTERVENTION"})

    return pd.Series(False, index=df.index)


def classify_intervention_family(df: pd.DataFrame) -> pd.Series:
    """
    Coarse diagnostic grouping only.
    Does not alter benchmark labels.
    """
    if "intervention_type" not in df.columns:
        return pd.Series("UNKNOWN", index=df.index)

    txt = normalize_text(df["intervention_type"])

    family = pd.Series(
        "OTHER",
        index=df.index,
        dtype="object",
    )

    family.loc[
        txt.isin({"", "NONE", "NATURAL", "NO_INTERVENTION"})
    ] = "NATURAL"

    family.loc[
        txt.str.contains("MISSING", na=False)
        | txt.str.contains("HIDE", na=False)
    ] = "MISSINGNESS"

    family.loc[
        txt.str.contains("CONTRAD", na=False)
        | txt.str.contains("PERTURB", na=False)
    ] = "CONTRADICTION"

    family.loc[
        txt.str.contains("SEVERE", na=False)
        | txt.str.contains("MULTI", na=False)
    ] = "SEVERE_EVIDENCE_LOSS"

    return family


def build_regime(
    t0: pd.DataFrame,
    t1: pd.DataFrame,
    regime: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:

    require_columns(
        t0,
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
        ],
        f"t0/{regime}",
    )

    require_columns(
        t1,
        [
            "case_id",
            "y_t1_d1",
            "p_safe_t1_cal",
            "p_not_safe_t1_cal",
        ],
        f"t1/{regime}",
    )

    work = t0.copy()

    work["source_case_id"] = work["source_case_id"].astype(str)

    t1_small = t1.copy()
    t1_small["case_id"] = t1_small["case_id"].astype(str)

    # One post-GATHER row per case/regime should exist.
    dup = t1_small["case_id"].duplicated(keep=False)
    if dup.any():
        raise AssertionError(
            f"{regime}: duplicated t1 case_id rows detected: "
            f"{int(dup.sum()):,}"
        )

    t1_keep = [
        c for c in [
            "case_id",
            "y_t1_d1",
            "p_safe_t1_raw",
            "p_safe_t1_cal",
            "p_not_safe_t1_raw",
            "p_not_safe_t1_cal",
            "predicted_t1_safety",
            "predictive_entropy_t1",
            "elapsed_t0_to_t1_days",
        ]
        if c in t1_small.columns
    ]

    t1_small = t1_small[t1_keep].rename(
        columns={"case_id": "source_case_id"}
    )

    work = work.merge(
        t1_small,
        on="source_case_id",
        how="left",
        validate="many_to_one",
        indicator="_t1_join",
    )

    work["gather_transition_available"] = (
        work["_t1_join"] == "both"
    )

    work.drop(columns=["_t1_join"], inplace=True)

    work["is_synthetic_intervention"] = infer_synthetic_flag(work)
    work["intervention_family"] = classify_intervention_family(work)

    expected = normalize_text(
        work["expected_action_reference"]
    )

    work["reference_is_gather"] = (
        expected == "GATHER"
    )

    # Conservative primary policy eligibility:
    # only naturally observed delayed-evidence GATHER episodes.
    work["gather_policy_eligible"] = (
        work["gather_transition_available"]
        & work["reference_is_gather"]
        & (~work["is_synthetic_intervention"])
    )

    # Why a GATHER episode is not eligible.
    reason = pd.Series(
        "ELIGIBLE",
        index=work.index,
        dtype="object",
    )

    reason.loc[
        ~work["gather_transition_available"]
    ] = "NO_REAL_T1_TRANSITION"

    reason.loc[
        work["gather_transition_available"]
        & ~work["reference_is_gather"]
    ] = "REFERENCE_NOT_GATHER"

    reason.loc[
        work["gather_transition_available"]
        & work["reference_is_gather"]
        & work["is_synthetic_intervention"]
    ] = "SYNTHETIC_EPISODE_EXCLUDED_PRIMARY"

    work["gather_ineligibility_reason"] = reason

    # Sanity checks on probabilities.
    prob_cols = [
        "p_cal_execute",
        "p_cal_gather",
        "p_cal_escalate",
        "p_cal_abstain",
    ]

    p_sum = work[prob_cols].sum(axis=1)

    if not np.allclose(
        p_sum.to_numpy(),
        np.ones(len(work)),
        atol=1e-6,
        rtol=1e-6,
    ):
        bad = int(
            (~np.isclose(
                p_sum.to_numpy(),
                1.0,
                atol=1e-6,
                rtol=1e-6,
            )).sum()
        )
        raise AssertionError(
            f"{regime}: {bad:,} rows have t0 calibrated probabilities "
            "that do not sum to 1."
        )

    # t1 scores must exist for eligible rows.
    eligible_missing_t1 = (
        work["gather_policy_eligible"]
        & work["p_safe_t1_cal"].isna()
    )

    if eligible_missing_t1.any():
        raise AssertionError(
            f"{regime}: {int(eligible_missing_t1.sum()):,} "
            "eligible GATHER rows have missing t1 scores."
        )

    # Derived quantities for the policy engine.
    work["p_nonexecute_t0"] = (
        1.0 - work["p_cal_execute"]
    )

    work["p_nonexecute_t0_check"] = (
        work["p_cal_gather"]
        + work["p_cal_escalate"]
        + work["p_cal_abstain"]
    )

    if not np.allclose(
        work["p_nonexecute_t0"],
        work["p_nonexecute_t0_check"],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise AssertionError(
            f"{regime}: p_nonexecute identity failed."
        )

    # Null out t1 score on rows that are not eligible for the PRIMARY
    # policy, while preserving observed diagnostic value separately.
    work["p_safe_t1_cal_observed"] = work["p_safe_t1_cal"]
    work["p_not_safe_t1_cal_observed"] = work["p_not_safe_t1_cal"]

    work["p_safe_t1_for_primary_rcse"] = np.where(
        work["gather_policy_eligible"],
        work["p_safe_t1_cal"],
        np.nan,
    )

    work["p_not_safe_t1_for_primary_rcse"] = np.where(
        work["gather_policy_eligible"],
        work["p_not_safe_t1_cal"],
        np.nan,
    )

    # Summary.
    summary = {
        "regime": regime,
        "episodes": int(len(work)),
        "unique_source_cases": int(
            work["source_case_id"].nunique()
        ),
        "real_t1_transition_available": int(
            work["gather_transition_available"].sum()
        ),
        "reference_gather_episodes": int(
            work["reference_is_gather"].sum()
        ),
        "primary_gather_eligible": int(
            work["gather_policy_eligible"].sum()
        ),
        "primary_gather_eligible_pct": float(
            work["gather_policy_eligible"].mean()
        ),
        "eligible_unique_source_cases": int(
            work.loc[
                work["gather_policy_eligible"],
                "source_case_id",
            ].nunique()
        ),
        "synthetic_episodes": int(
            work["is_synthetic_intervention"].sum()
        ),
    }

    elig_summary = (
        work.groupby(
            [
                "gather_policy_eligible",
                "gather_ineligibility_reason",
                "intervention_family",
            ],
            dropna=False,
        )
        .agg(
            episodes=("benchmark_episode_id", "count"),
            unique_source_cases=(
                "source_case_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    elig_summary.insert(
        0,
        "regime",
        regime,
    )

    return work, summary, elig_summary


def main() -> None:
    args = parse_args()

    t0_dir = Path(
        args.t0_dir
    ).expanduser().resolve()

    t1_dir = Path(
        args.t1_dir
    ).expanduser().resolve()

    if not t0_dir.exists():
        raise FileNotFoundError(
            f"t0 directory not found: {t0_dir}"
        )

    if not t1_dir.exists():
        raise FileNotFoundError(
            f"t1 directory not found: {t1_dir}"
        )

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else t0_dir / "rcse_policy_dataset"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("RCSE STEP 09b.1 - BUILD POLICY-READY DATASET")
    print("=" * 80)
    print(f"t0 input: {t0_dir}")
    print(f"t1 input: {t1_dir}")
    print(f"Output  : {out_dir}")

    summary_rows = []
    eligibility_frames = []

    for regime in REGIMES:
        t0_path = t0_dir / T0_FILE[regime]
        t1_path = t1_dir / T1_FILE[regime]

        if not t0_path.exists():
            raise FileNotFoundError(
                f"Missing t0 file: {t0_path}"
            )

        if not t1_path.exists():
            raise FileNotFoundError(
                f"Missing t1 file: {t1_path}"
            )

        print(f"\n[{regime}] loading frozen artifacts...")

        t0 = pd.read_parquet(t0_path)
        t1 = pd.read_parquet(t1_path)

        policy_df, summary, elig = build_regime(
            t0=t0,
            t1=t1,
            regime=regime,
        )

        out_path = (
            out_dir
            / f"rcse_policy_dataset_{regime}.parquet"
        )

        policy_df.to_parquet(
            out_path,
            index=False,
        )

        summary_rows.append(summary)
        eligibility_frames.append(elig)

        print(
            f"   episodes={summary['episodes']:,} | "
            f"t1 available={summary['real_t1_transition_available']:,} | "
            f"primary GATHER eligible={summary['primary_gather_eligible']:,}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    eligibility_df = pd.concat(
        eligibility_frames,
        ignore_index=True,
    )

    summary_df.to_csv(
        out_dir / "rcse_policy_dataset_summary.csv",
        index=False,
    )

    eligibility_df.to_csv(
        out_dir / "rcse_gather_eligibility_summary.csv",
        index=False,
    )

    metadata: dict[str, Any] = {
        "step": "09b.1",
        "t0_input_directory": str(t0_dir),
        "t1_input_directory": str(t1_dir),
        "regimes": REGIMES,
        "join_key": "t0.source_case_id == t1.case_id",
        "primary_gather_eligibility_rule": [
            "real post-GATHER t1 transition exists for source case",
            "expected_action_reference == GATHER",
            "episode is not a synthetic intervention",
        ],
        "primary_policy_t1_fields": [
            "p_safe_t1_for_primary_rcse",
            "p_not_safe_t1_for_primary_rcse",
        ],
        "diagnostic_t1_fields": [
            "p_safe_t1_cal_observed",
            "p_not_safe_t1_cal_observed",
        ],
        "important_boundary": (
            "The primary RCSE GATHER branch is restricted to naturally "
            "observed delayed-evidence episodes. Synthetic uncertainty "
            "episodes are not assigned a real t1 continuation score in the "
            "primary policy evaluation."
        ),
        "next_step": (
            "Step 09b.2: implement parameterized comparator and RCSE policy "
            "simulation over this policy-ready dataset."
        ),
    }

    with open(
        out_dir / "rcse_policy_dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 09b.1 COMPLETE")
    print("=" * 80)

    print("\nPolicy-ready dataset summary:")
    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for regime in REGIMES:
        print(
            "  - "
            + str(
                out_dir
                / f"rcse_policy_dataset_{regime}.parquet"
            )
        )

    print(
        "  - "
        + str(
            out_dir
            / "rcse_policy_dataset_summary.csv"
        )
    )

    print(
        "  - "
        + str(
            out_dir
            / "rcse_gather_eligibility_summary.csv"
        )
    )

    print(
        "  - "
        + str(
            out_dir
            / "rcse_policy_dataset_metadata.json"
        )
    )

    print(
        "\nNext: Step 09b.2 will apply comparator policies and RCSE "
        "cost/risk-budget grids to these frozen policy-ready datasets."
    )


if __name__ == "__main__":
    main()
