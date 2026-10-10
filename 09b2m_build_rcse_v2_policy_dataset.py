#!/usr/bin/env python
"""
09b2m_build_rcse_v2_policy_dataset.py

RCSE Step 09b.2m
Build policy-ready RCSE v2 evaluation datasets by joining all frozen components.

Purpose
-------
Create one validated held-out policy dataset per regime containing:

1. Frozen multiclass action/planning probabilities
2. Dedicated t0 execution-safety probabilities
3. Evidence Gate v2 state at delta=0.05
4. Existing post-GATHER safety / VoI fields
5. Exposure / consequence inputs
6. Reference and post-GATHER outcome labels
7. Regime identifiers and split metadata

NO MODEL TRAINING.
NO RECALIBRATION.
NO POLICY RETUNING.
NO CHANGE TO THE FROZEN RCSE v2 SPECIFICATION.

Inputs
------
--policy-dir
    Step 09b.1 policy-ready dataset directory:
        rcse_policy_dataset_grouped_iid.parquet
        rcse_policy_dataset_temporal.parquet
        rcse_policy_dataset_ood_exposure.parquet

--safety-robustness-dir
    Step 09b.2h directory:
        full_benchmark_safety_predictions_grouped_iid.parquet
        full_benchmark_safety_predictions_temporal.parquet
        full_benchmark_safety_predictions_ood_exposure.parquet

--gate-dir
    Step 09b.2k directory containing:
        evidence_gate_v2_row_diagnostics.parquet

--spec
    09b2l_rcse_v2_policy_spec.json

Outputs
-------
rcse_v2_policy_dataset_grouped_iid.parquet
rcse_v2_policy_dataset_temporal.parquet
rcse_v2_policy_dataset_ood_exposure.parquet

rcse_v2_dataset_validation_summary.csv
rcse_v2_join_validation.csv
rcse_v2_gate_state_summary.csv
rcse_v2_safety_score_summary.csv
rcse_v2_policy_manifest.json
rcse_v2_policy_metadata.json

Critical invariants
-------------------
Expected test counts:
    grouped_iid : 36,919
    temporal    : 37,580
    ood_exposure: 2,536

Every output row must have:
    benchmark_episode_id
    frozen action probabilities
    p_safe_t0_raw / p_safe_t0_cal
    gate_state_v2
    transaction_exposure_eur
    expected_action_reference

Evidence gate:
    delta = 0.05
    gate state comes from Step 09b.2k only

This script does NOT recompute or alter the frozen gate specification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = ("grouped_iid", "temporal", "ood_exposure")

EXPECTED_TEST_COUNTS = {
    "grouped_iid": 36919,
    "temporal": 37580,
    "ood_exposure": 2536,
}

POLICY_FILES = {
    "grouped_iid": "rcse_policy_dataset_grouped_iid.parquet",
    "temporal": "rcse_policy_dataset_temporal.parquet",
    "ood_exposure": "rcse_policy_dataset_ood_exposure.parquet",
}

SAFETY_FILES = {
    "grouped_iid": "full_benchmark_safety_predictions_grouped_iid.parquet",
    "temporal": "full_benchmark_safety_predictions_temporal.parquet",
    "ood_exposure": "full_benchmark_safety_predictions_ood_exposure.parquet",
}

# Step 09b.2k stores the 0.05 state using this generated column name.
GATE_STATE_COLUMN_CANDIDATES = (
    "_gate_state_delta_0_05",
    "gate_state_delta_0_05",
    "gate_state_v2",
)

REQUIRED_POLICY_COLUMNS = [
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
]

REQUIRED_SAFETY_COLUMNS = [
    "benchmark_episode_id",
    "p_safe_t0_raw",
    "p_safe_t0_cal",
]

SAFE_LABEL = "SAFE_TO_EXECUTE"
UNSAFE_LABEL = "NOT_SAFE_TO_EXECUTE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2m: build validated RCSE v2 policy-ready datasets."
    )
    p.add_argument(
        "--policy-dir",
        required=True,
        help="Step 09b.1 rcse_policy_dataset directory.",
    )
    p.add_argument(
        "--safety-robustness-dir",
        required=True,
        help="Step 09b.2h t0_safety_full_benchmark_robustness directory.",
    )
    p.add_argument(
        "--gate-dir",
        required=True,
        help="Step 09b.2k evidence_gate_v2_audit directory.",
    )
    p.add_argument(
        "--spec",
        required=True,
        help="Frozen 09b2l_rcse_v2_policy_spec.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <policy-dir>/rcse_v2_policy_dataset",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def assert_unique(df: pd.DataFrame, col: str, label: str) -> None:
    dup = df[col].duplicated(keep=False)
    if dup.any():
        examples = (
            df.loc[dup, col]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise AssertionError(
            f"{label}: {int(dup.sum()):,} duplicate rows for {col}. "
            f"Example IDs: {examples}"
        )


def normalize_action(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )


def resolve_gate_column(gate_df: pd.DataFrame) -> str:
    for c in GATE_STATE_COLUMN_CANDIDATES:
        if c in gate_df.columns:
            return c

    gate_like = [
        c for c in gate_df.columns
        if "gate_state" in c.lower()
    ]

    raise KeyError(
        "Could not locate frozen delta=0.05 gate-state column. "
        f"Gate-like columns found: {gate_like}"
    )


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("status") != "FROZEN_BEFORE_RCSE_V2_SIMULATION":
        raise RuntimeError(
            "RCSE v2 spec is not marked FROZEN_BEFORE_RCSE_V2_SIMULATION."
        )

    gate = (
        spec.get("architecture", {})
        .get("evidence_gate", {})
    )

    delta = gate.get("value_consistency_delta")

    if delta is None or abs(float(delta) - 0.05) > 1e-12:
        raise RuntimeError(
            f"Frozen RCSE v2 spec does not use gate delta=0.05; got {delta}"
        )

    safety = (
        spec.get("architecture", {})
        .get("t0_execution_safety_belief", {})
    )

    if safety.get("primary_score") != "p_safe_t0_cal":
        raise RuntimeError(
            "Frozen RCSE v2 spec primary safety score is not p_safe_t0_cal."
        )


def load_gate_table(gate_dir: Path) -> tuple[pd.DataFrame, str]:
    path = gate_dir / "evidence_gate_v2_row_diagnostics.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Step 09b.2k gate artifact: {path}"
        )

    gate = pd.read_parquet(path)

    require_columns(
        gate,
        ["benchmark_episode_id"],
        "evidence gate v2",
    )

    gate_col = resolve_gate_column(gate)

    gate = gate[
        [
            "benchmark_episode_id",
            gate_col,
        ]
    ].copy()

    gate = gate.rename(
        columns={gate_col: "gate_state_v2"}
    )

    assert_unique(
        gate,
        "benchmark_episode_id",
        "gate v2 diagnostics",
    )

    valid_states = {"VALID", "INCOMPLETE", "CONTRADICTORY"}

    bad = ~gate["gate_state_v2"].isin(valid_states)

    if bad.any():
        examples = (
            gate.loc[bad, "gate_state_v2"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            f"Unexpected gate states: {examples}"
        )

    return gate, path.name


def select_safety_columns(
    df: pd.DataFrame,
    regime: str,
) -> pd.DataFrame:
    require_columns(
        df,
        REQUIRED_SAFETY_COLUMNS,
        f"Step 09b.2h/{regime}",
    )

    cols = [
        "benchmark_episode_id",
        "p_safe_t0_raw",
        "p_safe_t0_cal",
    ]

    # Keep robustness/evaluation semantics if present.
    for c in [
        "_robustness_target",
        "_is_synthetic",
        "_intervention_family",
        "evaluation_regime",
    ]:
        if c in df.columns:
            cols.append(c)

    out = df[cols].copy()

    # Rename private diagnostic columns into stable policy-dataset names.
    rename = {
        "_robustness_target": "t0_safety_robustness_target",
        "_is_synthetic": "is_synthetic_intervention_eval",
        "_intervention_family": "intervention_family_eval",
    }

    out = out.rename(columns=rename)

    assert_unique(
        out,
        "benchmark_episode_id",
        f"Step 09b.2h/{regime}",
    )

    return out


def validate_probability(
    df: pd.DataFrame,
    col: str,
    label: str,
) -> dict[str, Any]:
    x = pd.to_numeric(
        df[col],
        errors="coerce",
    )

    missing = int(
        x.isna().sum()
    )

    outside = int(
        (
            (x < 0)
            | (x > 1)
        ).sum()
    )

    if missing:
        raise ValueError(
            f"{label}: {missing:,} missing values in {col}"
        )

    if outside:
        raise ValueError(
            f"{label}: {outside:,} values outside [0,1] in {col}"
        )

    return {
        "column": col,
        "missing": missing,
        "outside_0_1": outside,
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def check_policy_probability_sum(
    df: pd.DataFrame,
) -> dict[str, float]:
    cols = [
        "p_cal_execute",
        "p_cal_gather",
        "p_cal_escalate",
        "p_cal_abstain",
    ]

    mat = df[cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    sums = mat.sum(axis=1)

    return {
        "action_prob_sum_mean": float(
            sums.mean()
        ),
        "action_prob_sum_min": float(
            sums.min()
        ),
        "action_prob_sum_max": float(
            sums.max()
        ),
        "action_prob_sum_max_abs_error_from_1": float(
            (sums - 1.0).abs().max()
        ),
    }


def create_regime_dataset(
    regime: str,
    policy_dir: Path,
    safety_dir: Path,
    gate: pd.DataFrame,
    out_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    list[dict[str, Any]],
]:
    policy_path = (
        policy_dir
        / POLICY_FILES[regime]
    )

    safety_path = (
        safety_dir
        / SAFETY_FILES[regime]
    )

    if not policy_path.exists():
        raise FileNotFoundError(
            policy_path
        )

    if not safety_path.exists():
        raise FileNotFoundError(
            safety_path
        )

    policy = pd.read_parquet(
        policy_path
    )

    safety_raw = pd.read_parquet(
        safety_path
    )

    require_columns(
        policy,
        REQUIRED_POLICY_COLUMNS,
        f"Step 09b.1/{regime}",
    )

    assert_unique(
        policy,
        "benchmark_episode_id",
        f"Step 09b.1/{regime}",
    )

    safety = select_safety_columns(
        safety_raw,
        regime,
    )

    expected_n = EXPECTED_TEST_COUNTS[
        regime
    ]

    if len(policy) != expected_n:
        raise AssertionError(
            f"{regime}: Step 09b.1 policy rows={len(policy):,}; "
            f"expected {expected_n:,}"
        )

    # --------------------------------------------------------------
    # Join dedicated t0 safety scores.
    # --------------------------------------------------------------
    merged = policy.merge(
        safety,
        on="benchmark_episode_id",
        how="left",
        validate="one_to_one",
        indicator="_safety_join",
    )

    missing_safety = int(
        (
            merged["_safety_join"]
            != "both"
        ).sum()
    )

    if missing_safety:
        examples = (
            merged.loc[
                merged["_safety_join"]
                != "both",
                "benchmark_episode_id",
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{regime}: {missing_safety:,} policy episodes missing dedicated "
            f"t0 safety scores. Example IDs: {examples}"
        )

    merged.drop(
        columns=["_safety_join"],
        inplace=True,
    )

    # --------------------------------------------------------------
    # Join frozen Evidence Gate v2 state.
    # --------------------------------------------------------------
    merged = merged.merge(
        gate,
        on="benchmark_episode_id",
        how="left",
        validate="one_to_one",
        indicator="_gate_join",
    )

    missing_gate = int(
        (
            merged["_gate_join"]
            != "both"
        ).sum()
    )

    if missing_gate:
        examples = (
            merged.loc[
                merged["_gate_join"]
                != "both",
                "benchmark_episode_id",
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise RuntimeError(
            f"{regime}: {missing_gate:,} episodes missing Evidence Gate v2 state. "
            f"Example IDs: {examples}"
        )

    merged.drop(
        columns=["_gate_join"],
        inplace=True,
    )

    # --------------------------------------------------------------
    # Stable, explicit v2 convenience fields.
    # --------------------------------------------------------------
    merged[
        "evaluation_regime"
    ] = regime

    merged[
        "rcse_v2_gate_valid"
    ] = (
        merged[
            "gate_state_v2"
        ]
        == "VALID"
    )

    merged[
        "rcse_v2_gate_incomplete"
    ] = (
        merged[
            "gate_state_v2"
        ]
        == "INCOMPLETE"
    )

    merged[
        "rcse_v2_gate_contradictory"
    ] = (
        merged[
            "gate_state_v2"
        ]
        == "CONTRADICTORY"
    )

    merged[
        "rcse_v2_execute_probability_belief"
    ] = pd.to_numeric(
        merged[
            "p_safe_t0_cal"
        ],
        errors="coerce",
    )

    merged[
        "rcse_v2_execute_risk_probability"
    ] = (
        1.0
        - merged[
            "rcse_v2_execute_probability_belief"
        ]
    )

    merged[
        "reference_is_execute"
    ] = (
        normalize_action(
            merged[
                "expected_action_reference"
            ]
        )
        == "EXECUTE"
    )

    # --------------------------------------------------------------
    # Validation.
    # --------------------------------------------------------------
    assert_unique(
        merged,
        "benchmark_episode_id",
        f"RCSE v2/{regime}",
    )

    if len(merged) != expected_n:
        raise AssertionError(
            f"{regime}: merged RCSE v2 rows={len(merged):,}; expected {expected_n:,}"
        )

    prob_validation = []

    for c in [
        "p_cal_execute",
        "p_cal_gather",
        "p_cal_escalate",
        "p_cal_abstain",
        "p_safe_t0_raw",
        "p_safe_t0_cal",
    ]:
        row = validate_probability(
            merged,
            c,
            f"{regime}",
        )

        row[
            "regime"
        ] = regime

        prob_validation.append(
            row
        )

    # Post-GATHER safety may legitimately be unavailable for ineligible cases.
    p_t1 = pd.to_numeric(
        merged[
            "p_safe_t1_for_primary_rcse"
        ],
        errors="coerce",
    )

    gather_eligible = (
        merged[
            "gather_policy_eligible"
        ]
        .fillna(False)
        .astype(bool)
    )

    missing_t1_when_eligible = int(
        (
            gather_eligible
            & p_t1.isna()
        ).sum()
    )

    if missing_t1_when_eligible:
        raise RuntimeError(
            f"{regime}: {missing_t1_when_eligible:,} gather-eligible cases "
            "lack p_safe_t1_for_primary_rcse."
        )

    if (
        (
            p_t1.dropna()
            < 0
        )
        | (
            p_t1.dropna()
            > 1
        )
    ).any():
        raise ValueError(
            f"{regime}: p_safe_t1_for_primary_rcse contains values outside [0,1]."
        )

    exposure = pd.to_numeric(
        merged[
            "transaction_exposure_eur"
        ],
        errors="coerce",
    )

    missing_exposure = int(
        exposure.isna().sum()
    )

    # Missing exposure is recorded rather than fatal because previous runs
    # already identified a small exposure-denominator edge case.
    gate_counts = (
        merged[
            "gate_state_v2"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    validation = {
        "regime": regime,
        "expected_rows": expected_n,
        "actual_rows": int(
            len(merged)
        ),
        "unique_benchmark_episode_ids": int(
            merged[
                "benchmark_episode_id"
            ]
            .nunique()
        ),
        "missing_safety_join": missing_safety,
        "missing_gate_join": missing_gate,
        "missing_transaction_exposure": missing_exposure,
        "gather_eligible_count": int(
            gather_eligible.sum()
        ),
        "missing_t1_score_when_gather_eligible": missing_t1_when_eligible,
        "gate_valid_count": int(
            gate_counts.get(
                "VALID",
                0,
            )
        ),
        "gate_incomplete_count": int(
            gate_counts.get(
                "INCOMPLETE",
                0,
            )
        ),
        "gate_contradictory_count": int(
            gate_counts.get(
                "CONTRADICTORY",
                0,
            )
        ),
        **check_policy_probability_sum(
            merged
        ),
    }

    output_path = (
        out_dir
        / f"rcse_v2_policy_dataset_{regime}.parquet"
    )

    merged.to_parquet(
        output_path,
        index=False,
    )

    return (
        merged,
        validation,
        prob_validation,
    )


def main() -> None:
    args = parse_args()

    policy_dir = (
        Path(
            args.policy_dir
        )
        .expanduser()
        .resolve()
    )

    safety_dir = (
        Path(
            args.safety_robustness_dir
        )
        .expanduser()
        .resolve()
    )

    gate_dir = (
        Path(
            args.gate_dir
        )
        .expanduser()
        .resolve()
    )

    spec_path = (
        Path(
            args.spec
        )
        .expanduser()
        .resolve()
    )

    for p in [
        policy_dir,
        safety_dir,
        gate_dir,
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
        )
        .expanduser()
        .resolve()
        if args.out
        else policy_dir
        / "rcse_v2_policy_dataset"
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

    validate_spec(
        spec
    )

    gate, gate_filename = load_gate_table(
        gate_dir
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2m - BUILD V2 POLICY-READY DATASET"
    )
    print(
        "=" * 80
    )
    print(
        f"Policy source      : {policy_dir}"
    )
    print(
        f"Safety source      : {safety_dir}"
    )
    print(
        f"Gate source        : {gate_dir}"
    )
    print(
        f"Frozen v2 spec     : {spec_path}"
    )
    print(
        f"Output             : {out_dir}"
    )
    print(
        "NO TRAINING / NO RECALIBRATION / NO POLICY RETUNING"
    )
    print(
        "Frozen evidence-gate delta = 0.05"
    )

    validation_rows = []
    probability_rows = []
    gate_summary_rows = []
    safety_summary_rows = []

    output_files = {}

    for regime in REGIMES:
        print(
            f"\n[{regime}] assembling..."
        )

        (
            df,
            validation,
            prob_validation,
        ) = create_regime_dataset(
            regime=regime,
            policy_dir=policy_dir,
            safety_dir=safety_dir,
            gate=gate,
            out_dir=out_dir,
        )

        validation_rows.append(
            validation
        )

        probability_rows.extend(
            prob_validation
        )

        gate_summary = (
            df[
                "gate_state_v2"
            ]
            .value_counts(
                normalize=False
            )
            .rename_axis(
                "gate_state_v2"
            )
            .reset_index(
                name="count"
            )
        )

        gate_summary[
            "pct"
        ] = (
            gate_summary[
                "count"
            ]
            / len(df)
        )

        gate_summary[
            "regime"
        ] = regime

        gate_summary_rows.extend(
            gate_summary.to_dict(
                orient="records"
            )
        )

        for score_col in [
            "p_safe_t0_raw",
            "p_safe_t0_cal",
        ]:
            s = pd.to_numeric(
                df[
                    score_col
                ],
                errors="coerce",
            )

            safety_summary_rows.append(
                {
                    "regime": regime,
                    "score": score_col,
                    "n": int(
                        len(s)
                    ),
                    "mean": float(
                        s.mean()
                    ),
                    "p10": float(
                        s.quantile(
                            0.10
                        )
                    ),
                    "p25": float(
                        s.quantile(
                            0.25
                        )
                    ),
                    "p50": float(
                        s.quantile(
                            0.50
                        )
                    ),
                    "p75": float(
                        s.quantile(
                            0.75
                        )
                    ),
                    "p90": float(
                        s.quantile(
                            0.90
                        )
                    ),
                    "p95": float(
                        s.quantile(
                            0.95
                        )
                    ),
                    "p99": float(
                        s.quantile(
                            0.99
                        )
                    ),
                    "max": float(
                        s.max()
                    ),
                }
            )

        output_files[
            regime
        ] = str(
            out_dir
            / f"rcse_v2_policy_dataset_{regime}.parquet"
        )

        print(
            f"   rows={len(df):,} "
            f"| VALID={(df['gate_state_v2'] == 'VALID').sum():,} "
            f"| INCOMPLETE={(df['gate_state_v2'] == 'INCOMPLETE').sum():,} "
            f"| CONTRADICTORY={(df['gate_state_v2'] == 'CONTRADICTORY').sum():,}"
        )

    validation_df = pd.DataFrame(
        validation_rows
    )

    probability_df = pd.DataFrame(
        probability_rows
    )

    gate_summary_df = pd.DataFrame(
        gate_summary_rows
    )

    safety_summary_df = pd.DataFrame(
        safety_summary_rows
    )

    validation_df.to_csv(
        out_dir
        / "rcse_v2_dataset_validation_summary.csv",
        index=False,
    )

    probability_df.to_csv(
        out_dir
        / "rcse_v2_join_validation.csv",
        index=False,
    )

    gate_summary_df.to_csv(
        out_dir
        / "rcse_v2_gate_state_summary.csv",
        index=False,
    )

    safety_summary_df.to_csv(
        out_dir
        / "rcse_v2_safety_score_summary.csv",
        index=False,
    )

    manifest = {
        "step": "09b.2m",
        "status": "POLICY_READY_DATASET_BUILT_FROM_FROZEN_COMPONENTS",
        "regimes": list(
            REGIMES
        ),
        "expected_test_counts": EXPECTED_TEST_COUNTS,
        "policy_source_files": POLICY_FILES,
        "safety_source_files": SAFETY_FILES,
        "gate_source_file": gate_filename,
        "gate_delta": 0.05,
        "frozen_spec": str(
            spec_path
        ),
        "required_action_columns": REQUIRED_POLICY_COLUMNS,
        "required_safety_columns": REQUIRED_SAFETY_COLUMNS,
        "stable_v2_columns_added": [
            "gate_state_v2",
            "rcse_v2_gate_valid",
            "rcse_v2_gate_incomplete",
            "rcse_v2_gate_contradictory",
            "rcse_v2_execute_probability_belief",
            "rcse_v2_execute_risk_probability",
            "reference_is_execute",
            "evaluation_regime",
        ],
        "decision_feature_boundary": {
            "execution_safety_belief": "p_safe_t0_cal",
            "planning_beliefs": [
                "p_cal_execute",
                "p_cal_gather",
                "p_cal_escalate",
                "p_cal_abstain",
            ],
            "evidence_gate": "gate_state_v2",
            "consequence_input": "transaction_exposure_eur",
            "post_gather_belief": "p_safe_t1_for_primary_rcse",
        },
        "evaluation_only_fields_may_include": [
            "expected_action_reference",
            "y_t1_d1",
            "t0_safety_robustness_target",
            "is_synthetic_intervention_eval",
            "intervention_family_eval",
        ],
    }

    with open(
        out_dir
        / "rcse_v2_policy_manifest.json",
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
        "step": "09b.2m",
        "policy_directory": str(
            policy_dir
        ),
        "safety_robustness_directory": str(
            safety_dir
        ),
        "gate_directory": str(
            gate_dir
        ),
        "frozen_spec": str(
            spec_path
        ),
        "output_directory": str(
            out_dir
        ),
        "output_files": output_files,
        "training_performed": False,
        "recalibration_performed": False,
        "policy_retuning_performed": False,
        "gate_recomputed": False,
        "gate_delta": 0.05,
        "primary_t0_safety_belief": "p_safe_t0_cal",
        "important_guardrail": (
            "This step only assembles frozen artifacts. It does not alter the "
            "RCSE v2 specification or choose operating thresholds."
        ),
        "next_step": (
            "Step 09b.2n: run the frozen RCSE v2 policy simulation and required "
            "ablations using these validated policy-ready datasets."
        ),
    }

    with open(
        out_dir
        / "rcse_v2_policy_metadata.json",
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
        "STEP 09b.2m COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nDataset validation:"
    )

    print(
        validation_df[
            [
                "regime",
                "expected_rows",
                "actual_rows",
                "unique_benchmark_episode_ids",
                "missing_safety_join",
                "missing_gate_join",
                "missing_transaction_exposure",
                "gather_eligible_count",
                "missing_t1_score_when_gather_eligible",
                "gate_valid_count",
                "gate_incomplete_count",
                "gate_contradictory_count",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nAction-probability simplex validation:"
    )

    print(
        validation_df[
            [
                "regime",
                "action_prob_sum_mean",
                "action_prob_sum_min",
                "action_prob_sum_max",
                "action_prob_sum_max_abs_error_from_1",
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
        "\nNext: Step 09b.2n — run the frozen RCSE v2 policy simulation "
        "and predeclared ablations. Do not change the v2 spec first."
    )


if __name__ == "__main__":
    main()
