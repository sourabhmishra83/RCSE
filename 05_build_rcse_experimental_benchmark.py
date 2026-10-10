"""
05_build_rcse_experimental_benchmark.py

Construct the first RCSE experimental benchmark from the BPI-derived v2 table.

Design
------
This benchmark combines:
1) Natural complete-evidence cases
2) Natural delayed-evidence cases
3) Controlled missingness interventions
4) Controlled contradiction interventions
5) Transaction exposure strata retained separately from intervention type

Input
-----
rcse_base_v2.parquet

Outputs
-------
rcse_experimental_benchmark.parquet
rcse_experimental_benchmark.csv
rcse_intervention_summary.csv
rcse_intervention_by_risk_stratum.csv
rcse_experimental_metadata.json

Important methodological choices
--------------------------------
- Do NOT use ambiguous BPI monetary mismatch arithmetic as a ground-truth REVIEW label.
- Natural labels are based on evidence availability:
    READY              -> GR available at t0
    WAIT_FOR_EVIDENCE  -> GR unavailable at t0
- REVIEW / ABSTAIN-like conditions are introduced through controlled interventions.
- Intervention metadata is explicit and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NATURAL_READY = "NATURAL_READY"
NATURAL_WAIT = "NATURAL_WAIT_FOR_EVIDENCE"

INT_NONE = "NONE"
INT_MISSING_GR = "MISSING_GR"
INT_MISSING_PO_VALUE = "MISSING_PO_VALUE"
INT_MISSING_INVOICE_VALUE = "MISSING_INVOICE_VALUE"
INT_PO_VALUE_CONTRADICTION = "PO_VALUE_CONTRADICTION"
INT_INVOICE_VALUE_CONTRADICTION = "INVOICE_VALUE_CONTRADICTION"
INT_MULTI_FIELD_MISSING = "MULTI_FIELD_MISSING"

EXPECTED_EXECUTE = "EXECUTE"
EXPECTED_GATHER = "GATHER"
EXPECTED_ESCALATE = "ESCALATE"
EXPECTED_ABSTAIN = "ABSTAIN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RCSE experimental benchmark with controlled interventions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to rcse_base_v2.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <input parent>/experimental_benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--max-natural-per-class",
        type=int,
        default=50000,
        help="Cap natural samples per natural class. Default: 50000",
    )
    parser.add_argument(
        "--n-per-intervention",
        type=int,
        default=15000,
        help="Number of base cases sampled for each controlled intervention. Default: 15000",
    )
    parser.add_argument(
        "--contradiction-strengths",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50],
        help="Relative value perturbations for contradiction experiments.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the full CSV.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def sample_cases(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].copy()


def add_common_metadata(
    part: pd.DataFrame,
    benchmark_arm: str,
    intervention_type: str,
    intervention_strength: float | None,
    expected_action: str,
    synthetic_intervention: bool,
) -> pd.DataFrame:
    part = part.copy()
    part["benchmark_arm"] = benchmark_arm
    part["intervention_type"] = intervention_type
    part["intervention_strength"] = intervention_strength
    part["expected_action_reference"] = expected_action
    part["synthetic_intervention"] = synthetic_intervention
    return part


def make_natural_ready(
    base: pd.DataFrame,
    max_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = base.loc[base["gr_available_at_t0"] == True].copy()  # noqa: E712
    part = sample_cases(candidates, max_n, rng)
    return add_common_metadata(
        part,
        benchmark_arm="NATURAL_COMPLETE_EVIDENCE",
        intervention_type=INT_NONE,
        intervention_strength=None,
        expected_action=EXPECTED_EXECUTE,
        synthetic_intervention=False,
    )


def make_natural_wait(
    base: pd.DataFrame,
    max_n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = base.loc[base["gr_available_at_t0"] == False].copy()  # noqa: E712
    part = sample_cases(candidates, max_n, rng)
    return add_common_metadata(
        part,
        benchmark_arm="NATURAL_DELAYED_EVIDENCE",
        intervention_type=INT_NONE,
        intervention_strength=None,
        expected_action=EXPECTED_GATHER,
        synthetic_intervention=False,
    )


def make_missing_gr(
    base_ready: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    part = sample_cases(base_ready, n, rng)

    part["original_gr_available_at_t0"] = part["gr_available_at_t0"]
    part["original_latest_gr_value_before_t0_eur"] = part[
        "latest_gr_value_before_t0_eur"
    ]

    part["gr_available_at_t0"] = False
    part["latest_gr_value_before_t0_eur"] = np.nan
    part["inv_gr_abs_rel_diff"] = np.nan

    return add_common_metadata(
        part,
        benchmark_arm="CONTROLLED_MISSINGNESS",
        intervention_type=INT_MISSING_GR,
        intervention_strength=1.0,
        expected_action=EXPECTED_GATHER,
        synthetic_intervention=True,
    )


def make_missing_po(
    base_ready: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    part = sample_cases(base_ready, n, rng)

    part["original_po_value_initial_eur"] = part["po_value_initial_eur"]
    part["po_value_initial_eur"] = np.nan
    part["inv_po_abs_rel_diff"] = np.nan

    return add_common_metadata(
        part,
        benchmark_arm="CONTROLLED_MISSINGNESS",
        intervention_type=INT_MISSING_PO_VALUE,
        intervention_strength=1.0,
        expected_action=EXPECTED_ESCALATE,
        synthetic_intervention=True,
    )


def make_missing_invoice(
    base_ready: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    part = sample_cases(base_ready, n, rng)

    part["original_invoice_value_t0_eur"] = part["invoice_value_t0_eur"]
    part["invoice_value_t0_eur"] = np.nan
    part["inv_po_abs_rel_diff"] = np.nan
    part["inv_gr_abs_rel_diff"] = np.nan

    return add_common_metadata(
        part,
        benchmark_arm="CONTROLLED_MISSINGNESS",
        intervention_type=INT_MISSING_INVOICE_VALUE,
        intervention_strength=1.0,
        expected_action=EXPECTED_ESCALATE,
        synthetic_intervention=True,
    )


def make_multi_field_missing(
    base_ready: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    part = sample_cases(base_ready, n, rng)

    part["original_po_value_initial_eur"] = part["po_value_initial_eur"]
    part["original_invoice_value_t0_eur"] = part["invoice_value_t0_eur"]
    part["original_gr_available_at_t0"] = part["gr_available_at_t0"]
    part["original_latest_gr_value_before_t0_eur"] = part[
        "latest_gr_value_before_t0_eur"
    ]

    part["po_value_initial_eur"] = np.nan
    part["invoice_value_t0_eur"] = np.nan
    part["gr_available_at_t0"] = False
    part["latest_gr_value_before_t0_eur"] = np.nan
    part["inv_po_abs_rel_diff"] = np.nan
    part["inv_gr_abs_rel_diff"] = np.nan

    return add_common_metadata(
        part,
        benchmark_arm="CONTROLLED_SEVERE_MISSINGNESS",
        intervention_type=INT_MULTI_FIELD_MISSING,
        intervention_strength=1.0,
        expected_action=EXPECTED_ABSTAIN,
        synthetic_intervention=True,
    )


def recompute_rel_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.abs()
    out = (a - b).abs() / denom.replace(0, np.nan)
    return out


def make_po_contradiction(
    base_ready: pd.DataFrame,
    n: int,
    strength: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = base_ready.loc[
        base_ready["po_value_initial_eur"].notna()
        & base_ready["invoice_value_t0_eur"].notna()
    ].copy()

    part = sample_cases(candidates, n, rng)

    part["original_po_value_initial_eur"] = part["po_value_initial_eur"]

    # Alternate positive and negative perturbations for balance.
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(part))
    factor = 1.0 + (signs * strength)

    # Prevent nonsensical sign flips at high perturbations.
    factor = np.maximum(factor, 0.01)

    part["po_value_initial_eur"] = (
        part["original_po_value_initial_eur"] * factor
    )

    part["inv_po_abs_rel_diff"] = recompute_rel_diff(
        part["invoice_value_t0_eur"],
        part["po_value_initial_eur"],
    )

    return add_common_metadata(
        part,
        benchmark_arm="CONTROLLED_CONTRADICTION",
        intervention_type=INT_PO_VALUE_CONTRADICTION,
        intervention_strength=float(strength),
        expected_action=EXPECTED_ESCALATE,
        synthetic_intervention=True,
    )


def make_invoice_contradiction(
    base_ready: pd.DataFrame,
    n: int,
    strength: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    candidates = base_ready.loc[
        base_ready["invoice_value_t0_eur"].notna()
    ].copy()

    part = sample_cases(candidates, n, rng)

    part["original_invoice_value_t0_eur"] = part["invoice_value_t0_eur"]

    signs = rng.choice(np.array([-1.0, 1.0]), size=len(part))
    factor = 1.0 + (signs * strength)
    factor = np.maximum(factor, 0.01)

    part["invoice_value_t0_eur"] = (
        part["original_invoice_value_t0_eur"] * factor
    )

    if "po_value_initial_eur" in part.columns:
        part["inv_po_abs_rel_diff"] = recompute_rel_diff(
            part["invoice_value_t0_eur"],
            part["po_value_initial_eur"],
        )

    if "latest_gr_value_before_t0_eur" in part.columns:
        part["inv_gr_abs_rel_diff"] = recompute_rel_diff(
            part["invoice_value_t0_eur"],
            part["latest_gr_value_before_t0_eur"],
        )

    return add_common_metadata(
        part,
        benchmark_arm="CONTROLLED_CONTRADICTION",
        intervention_type=INT_INVOICE_VALUE_CONTRADICTION,
        intervention_strength=float(strength),
        expected_action=EXPECTED_ESCALATE,
        synthetic_intervention=True,
    )


def ensure_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every benchmark row has original-value audit columns even when
    no intervention touched that field.
    """
    df = df.copy()

    audit_map = {
        "original_po_value_initial_eur": "po_value_initial_eur",
        "original_invoice_value_t0_eur": "invoice_value_t0_eur",
        "original_gr_available_at_t0": "gr_available_at_t0",
        "original_latest_gr_value_before_t0_eur": "latest_gr_value_before_t0_eur",
    }

    for audit_col, live_col in audit_map.items():
        if audit_col not in df.columns:
            df[audit_col] = df[live_col]
        else:
            df[audit_col] = df[audit_col].where(
                df[audit_col].notna(),
                df[live_col],
            )

    return df


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "experimental_benchmark"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print("RCSE EXPERIMENTAL BENCHMARK BUILDER")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")
    print(f"Seed  : {args.seed}")

    print("\n1/6 Loading base benchmark...")
    base = pd.read_parquet(input_path)

    required = [
        "case_id",
        "item_category",
        "gr_available_at_t0",
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "decision_time",
    ]
    require_columns(base, required)

    print(f"   Rows: {len(base):,}")

    # Base pools
    base_ready = base.loc[
        base["gr_available_at_t0"] == True
    ].copy()  # noqa: E712

    print(f"   Natural complete-evidence pool: {len(base_ready):,}")
    print(
        f"   Natural delayed-evidence pool : "
        f"{int((base['gr_available_at_t0'] == False).sum()):,}"
    )

    print("\n2/6 Creating natural benchmark arms...")
    parts: list[pd.DataFrame] = []

    parts.append(
        make_natural_ready(
            base,
            args.max_natural_per_class,
            rng,
        )
    )

    parts.append(
        make_natural_wait(
            base,
            args.max_natural_per_class,
            rng,
        )
    )

    print("\n3/6 Creating controlled missingness arms...")
    parts.append(
        make_missing_gr(
            base_ready,
            args.n_per_intervention,
            rng,
        )
    )
    parts.append(
        make_missing_po(
            base_ready,
            args.n_per_intervention,
            rng,
        )
    )
    parts.append(
        make_missing_invoice(
            base_ready,
            args.n_per_intervention,
            rng,
        )
    )
    parts.append(
        make_multi_field_missing(
            base_ready,
            args.n_per_intervention,
            rng,
        )
    )

    print("\n4/6 Creating controlled contradiction arms...")
    for strength in args.contradiction_strengths:
        parts.append(
            make_po_contradiction(
                base_ready,
                args.n_per_intervention,
                strength,
                rng,
            )
        )
        parts.append(
            make_invoice_contradiction(
                base_ready,
                args.n_per_intervention,
                strength,
                rng,
            )
        )

    print("\n5/6 Combining and validating benchmark...")
    benchmark = pd.concat(
        parts,
        ignore_index=True,
        sort=False,
    )

    benchmark = ensure_original_columns(benchmark)

    # Unique benchmark episode ID: same source case can appear in multiple arms.
    benchmark["benchmark_episode_id"] = [
        f"RCSE_{i:09d}" for i in range(len(benchmark))
    ]

    # Preserve source case identity separately.
    benchmark["source_case_id"] = benchmark["case_id"]

    # Flags comparing manipulated vs original observation.
    benchmark["po_value_was_modified"] = (
        benchmark["po_value_initial_eur"]
        != benchmark["original_po_value_initial_eur"]
    ) & ~(
        benchmark["po_value_initial_eur"].isna()
        & benchmark["original_po_value_initial_eur"].isna()
    )

    benchmark["invoice_value_was_modified"] = (
        benchmark["invoice_value_t0_eur"]
        != benchmark["original_invoice_value_t0_eur"]
    ) & ~(
        benchmark["invoice_value_t0_eur"].isna()
        & benchmark["original_invoice_value_t0_eur"].isna()
    )

    benchmark["gr_observation_was_modified"] = (
        benchmark["gr_available_at_t0"]
        != benchmark["original_gr_available_at_t0"]
    ) | (
        benchmark["latest_gr_value_before_t0_eur"]
        != benchmark["original_latest_gr_value_before_t0_eur"]
    )

    # Simple intervention severity rank for stratified reporting.
    severity_map = {
        INT_NONE: 0,
        INT_MISSING_GR: 1,
        INT_MISSING_PO_VALUE: 2,
        INT_MISSING_INVOICE_VALUE: 2,
        INT_PO_VALUE_CONTRADICTION: 2,
        INT_INVOICE_VALUE_CONTRADICTION: 2,
        INT_MULTI_FIELD_MISSING: 3,
    }
    benchmark["intervention_severity_rank"] = benchmark[
        "intervention_type"
    ].map(severity_map).astype("Int64")

    # Stable ordering
    benchmark = benchmark.sort_values(
        [
            "synthetic_intervention",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
            "decision_time",
            "source_case_id",
        ],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)

    # Assertions
    assert benchmark["benchmark_episode_id"].is_unique
    assert benchmark["expected_action_reference"].notna().all()
    assert benchmark["intervention_type"].notna().all()

    # Natural arms must not be marked synthetic
    assert not benchmark.loc[
        benchmark["benchmark_arm"].str.startswith("NATURAL"),
        "synthetic_intervention",
    ].any()

    # Controlled arms must be synthetic
    assert benchmark.loc[
        benchmark["benchmark_arm"].str.startswith("CONTROLLED"),
        "synthetic_intervention",
    ].all()

    print(f"   Final benchmark episodes: {len(benchmark):,}")

    print("\n6/6 Writing benchmark and summaries...")

    parquet_path = out_dir / "rcse_experimental_benchmark.parquet"
    benchmark.to_parquet(parquet_path, index=False)

    if not args.no_csv:
        benchmark.to_csv(
            out_dir / "rcse_experimental_benchmark.csv",
            index=False,
        )

    summary = (
        benchmark.groupby(
            [
                "benchmark_arm",
                "intervention_type",
                "intervention_strength",
                "expected_action_reference",
                "synthetic_intervention",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    summary["pct_of_benchmark"] = (
        summary["count"] / len(benchmark) * 100
    ).round(4)
    summary.to_csv(
        out_dir / "rcse_intervention_summary.csv",
        index=False,
    )

    by_risk = (
        benchmark.groupby(
            [
                "intervention_type",
                "intervention_strength",
                "transaction_risk_stratum",
                "expected_action_reference",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_risk.to_csv(
        out_dir / "rcse_intervention_by_risk_stratum.csv",
        index=False,
    )

    action_summary = (
        benchmark["expected_action_reference"]
        .value_counts()
        .rename_axis("expected_action_reference")
        .reset_index(name="count")
    )
    action_summary["pct"] = (
        action_summary["count"] / len(benchmark) * 100
    ).round(4)
    action_summary.to_csv(
        out_dir / "rcse_expected_action_summary.csv",
        index=False,
    )

    metadata: dict[str, Any] = {
        "input_file": str(input_path),
        "benchmark_rows": int(len(benchmark)),
        "seed": int(args.seed),
        "benchmark_version": "experimental_v1",
        "natural_sampling_cap_per_class": int(
            args.max_natural_per_class
        ),
        "controlled_samples_per_intervention": int(
            args.n_per_intervention
        ),
        "contradiction_strengths": [
            float(x) for x in args.contradiction_strengths
        ],
        "natural_reference_policy": {
            "GR available at t0": EXPECTED_EXECUTE,
            "GR unavailable at t0": EXPECTED_GATHER,
        },
        "controlled_reference_policy": {
            INT_MISSING_GR: EXPECTED_GATHER,
            INT_MISSING_PO_VALUE: EXPECTED_ESCALATE,
            INT_MISSING_INVOICE_VALUE: EXPECTED_ESCALATE,
            INT_PO_VALUE_CONTRADICTION: EXPECTED_ESCALATE,
            INT_INVOICE_VALUE_CONTRADICTION: EXPECTED_ESCALATE,
            INT_MULTI_FIELD_MISSING: EXPECTED_ABSTAIN,
        },
        "methodological_notes": [
            (
                "The benchmark does not use BPI monetary mismatch arithmetic "
                "as a historical REVIEW ground-truth label."
            ),
            (
                "Natural delayed-evidence episodes come directly from the "
                "BPI process state at t0."
            ),
            (
                "REVIEW/ABSTAIN-like conditions are introduced through "
                "controlled, fully logged interventions."
            ),
            (
                "Transaction exposure remains an independent variable and "
                "does not determine the intervention class."
            ),
            (
                "The same source case can appear in multiple experimental "
                "arms; benchmark_episode_id uniquely identifies each episode."
            ),
        ],
    }

    with open(
        out_dir / "rcse_experimental_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK BUILD COMPLETE")
    print("=" * 80)

    print("\nExpected action distribution:")
    print(action_summary.to_string(index=False))

    print("\nIntervention summary:")
    print(summary.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "rcse_experimental_benchmark.parquet",
        None if args.no_csv else "rcse_experimental_benchmark.csv",
        "rcse_intervention_summary.csv",
        "rcse_intervention_by_risk_stratum.csv",
        "rcse_expected_action_summary.csv",
        "rcse_experimental_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
