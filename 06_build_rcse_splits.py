"""
06_build_rcse_splits.py

Create leakage-safe train/calibration/test splits for the RCSE experimental benchmark.

Input
-----
rcse_experimental_benchmark.parquet

Outputs
-------
rcse_experimental_with_splits.parquet
rcse_experimental_with_splits.csv
rcse_grouped_iid_split_summary.csv
rcse_temporal_split_summary.csv
rcse_ood_exposure_split_summary.csv
rcse_split_metadata.json

Split regimes
-------------
1. Grouped IID
   - 70% train
   - 15% calibration
   - 15% test
   - grouped by source_case_id so all variants remain together

2. Temporal
   - earliest 70% of source cases by decision_time -> train
   - next 15% -> calibration
   - latest 15% -> test
   - source-case grouped

3. OOD exposure
   - R1/R2/R3 source cases split into train/calibration/in-domain test
   - R4 source cases reserved exclusively for OOD test
   - source-case grouped

Important
---------
The same BPI source case can appear in multiple intervention arms.
All variants of a source case MUST stay in the same partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe RCSE data splits."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to rcse_experimental_benchmark.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <input parent>/splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for grouped IID and in-domain OOD splits.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Training fraction. Default: 0.70",
    )
    parser.add_argument(
        "--cal-frac",
        type=float,
        default=0.15,
        help="Calibration fraction. Default: 0.15",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Test fraction. Default: 0.15",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing full CSV output.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def validate_fracs(train: float, cal: float, test: float) -> None:
    total = train + cal + test
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Split fractions must sum to 1.0; got {total:.6f}"
        )
    for name, value in [("train", train), ("cal", cal), ("test", test)]:
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} fraction must be between 0 and 1.")


def assign_grouped_random_split(
    case_ids: pd.Series,
    train_frac: float,
    cal_frac: float,
    seed: int,
) -> dict[str, str]:
    unique_cases = pd.Index(case_ids.dropna().unique())
    rng = np.random.default_rng(seed)
    shuffled = unique_cases.to_numpy(copy=True)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(np.floor(n * train_frac))
    n_cal = int(np.floor(n * cal_frac))

    train_cases = shuffled[:n_train]
    cal_cases = shuffled[n_train:n_train + n_cal]
    test_cases = shuffled[n_train + n_cal:]

    mapping = {}
    mapping.update({str(x): "train" for x in train_cases})
    mapping.update({str(x): "calibration" for x in cal_cases})
    mapping.update({str(x): "test" for x in test_cases})
    return mapping


def build_case_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per source case.

    Because the same source case appears in multiple experimental arms,
    decision_time and risk stratum should be invariant. We validate that.
    """
    grouped = df.groupby("source_case_id", dropna=False)

    rows = []
    for case_id, g in grouped:
        times = pd.to_datetime(g["decision_time"], utc=True, errors="coerce").dropna()
        risk = g["transaction_risk_stratum"].dropna().astype(str).unique()

        if len(times) == 0:
            decision_time = pd.NaT
        else:
            # Same source case should have same decision time across variants.
            decision_time = times.min()

        if len(risk) == 0:
            risk_stratum = np.nan
        elif len(risk) == 1:
            risk_stratum = risk[0]
        else:
            raise ValueError(
                f"source_case_id {case_id} has inconsistent risk strata: {risk}"
            )

        rows.append(
            {
                "source_case_id": case_id,
                "decision_time": decision_time,
                "transaction_risk_stratum": risk_stratum,
                "episode_count": len(g),
            }
        )

    case_table = pd.DataFrame(rows)
    return case_table


def summarize_split(
    df: pd.DataFrame,
    split_col: str,
    regime: str,
) -> pd.DataFrame:
    summary = (
        df.groupby(split_col, dropna=False)
        .agg(
            episodes=("benchmark_episode_id", "count"),
            source_cases=("source_case_id", "nunique"),
        )
        .reset_index()
        .rename(columns={split_col: "partition"})
    )

    summary["episode_pct"] = (
        summary["episodes"] / len(df) * 100
    ).round(4)

    total_cases = df["source_case_id"].nunique()
    summary["source_case_pct"] = (
        summary["source_cases"] / total_cases * 100
    ).round(4)

    summary.insert(0, "regime", regime)
    return summary


def assert_no_case_leakage(df: pd.DataFrame, split_col: str) -> None:
    leaks = (
        df.groupby("source_case_id")[split_col]
        .nunique(dropna=False)
    )
    bad = leaks[leaks > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad):,} source cases appear in multiple {split_col} partitions."
        )


def main() -> None:
    args = parse_args()
    validate_fracs(args.train_frac, args.cal_frac, args.test_frac)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "splits"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE SPLIT BUILDER")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")
    print(f"Seed  : {args.seed}")

    print("\n1/6 Loading experimental benchmark...")
    df = pd.read_parquet(input_path)

    required = [
        "benchmark_episode_id",
        "source_case_id",
        "decision_time",
        "transaction_risk_stratum",
        "expected_action_reference",
        "benchmark_arm",
        "intervention_type",
    ]
    require_columns(df, required)

    df["decision_time"] = pd.to_datetime(
        df["decision_time"], utc=True, errors="coerce"
    )

    print(f"   Episodes    : {len(df):,}")
    print(f"   Source cases: {df['source_case_id'].nunique():,}")

    case_table = build_case_table(df)
    print(f"   Case table  : {len(case_table):,}")

    # -----------------------------------------------------------------
    # 2. Grouped IID split
    # -----------------------------------------------------------------
    print("\n2/6 Building grouped IID split...")

    iid_map = assign_grouped_random_split(
        case_table["source_case_id"],
        args.train_frac,
        args.cal_frac,
        args.seed,
    )

    case_table["split_grouped_iid"] = (
        case_table["source_case_id"].astype(str).map(iid_map)
    )

    df = df.merge(
        case_table[["source_case_id", "split_grouped_iid"]],
        on="source_case_id",
        how="left",
        validate="many_to_one",
    )

    assert_no_case_leakage(df, "split_grouped_iid")

    iid_summary = summarize_split(
        df,
        "split_grouped_iid",
        "grouped_iid",
    )
    iid_summary.to_csv(
        out_dir / "rcse_grouped_iid_split_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 3. Temporal split
    # -----------------------------------------------------------------
    print("\n3/6 Building temporal split...")

    temporal_cases = case_table.sort_values(
        ["decision_time", "source_case_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    n = len(temporal_cases)
    n_train = int(np.floor(n * args.train_frac))
    n_cal = int(np.floor(n * args.cal_frac))

    temporal_cases["split_temporal"] = "test"
    temporal_cases.loc[:n_train - 1, "split_temporal"] = "train"
    temporal_cases.loc[
        n_train:n_train + n_cal - 1,
        "split_temporal",
    ] = "calibration"

    df = df.merge(
        temporal_cases[["source_case_id", "split_temporal"]],
        on="source_case_id",
        how="left",
        validate="many_to_one",
    )

    assert_no_case_leakage(df, "split_temporal")

    temporal_summary = summarize_split(
        df,
        "split_temporal",
        "temporal",
    )
    temporal_summary.to_csv(
        out_dir / "rcse_temporal_split_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 4. OOD exposure split
    # -----------------------------------------------------------------
    print("\n4/6 Building OOD exposure split...")

    ood_cases = case_table.copy()
    ood_cases["split_ood_exposure"] = pd.NA

    is_r4 = (
        ood_cases["transaction_risk_stratum"].astype(str)
        == "R4_99_100"
    )

    # All R4 cases are held out as OOD test.
    ood_cases.loc[is_r4, "split_ood_exposure"] = "ood_test_r4"

    in_domain = ood_cases.loc[~is_r4].copy()

    # Re-normalize train/cal/test fractions within non-R4 cases.
    # Here we retain the requested ratios.
    in_map = assign_grouped_random_split(
        in_domain["source_case_id"],
        args.train_frac,
        args.cal_frac,
        args.seed + 1000,
    )

    ood_cases.loc[~is_r4, "split_ood_exposure"] = (
        ood_cases.loc[~is_r4, "source_case_id"]
        .astype(str)
        .map(in_map)
        .map(
            {
                "train": "train",
                "calibration": "calibration",
                "test": "id_test",
            }
        )
    )

    df = df.merge(
        ood_cases[["source_case_id", "split_ood_exposure"]],
        on="source_case_id",
        how="left",
        validate="many_to_one",
    )

    assert_no_case_leakage(df, "split_ood_exposure")

    ood_summary = summarize_split(
        df,
        "split_ood_exposure",
        "ood_exposure",
    )
    ood_summary.to_csv(
        out_dir / "rcse_ood_exposure_split_summary.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 5. Additional diagnostics
    # -----------------------------------------------------------------
    print("\n5/6 Writing split diagnostics...")

    # Distribution of reference actions by split/regime
    action_tables = []
    for split_col, regime in [
        ("split_grouped_iid", "grouped_iid"),
        ("split_temporal", "temporal"),
        ("split_ood_exposure", "ood_exposure"),
    ]:
        tab = (
            df.groupby(
                [split_col, "expected_action_reference"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .rename(columns={split_col: "partition"})
        )
        tab.insert(0, "regime", regime)
        action_tables.append(tab)

    pd.concat(
        action_tables,
        ignore_index=True,
    ).to_csv(
        out_dir / "rcse_split_action_distribution.csv",
        index=False,
    )

    # Distribution of intervention types by split/regime
    intervention_tables = []
    for split_col, regime in [
        ("split_grouped_iid", "grouped_iid"),
        ("split_temporal", "temporal"),
        ("split_ood_exposure", "ood_exposure"),
    ]:
        tab = (
            df.groupby(
                [split_col, "intervention_type"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
            .rename(columns={split_col: "partition"})
        )
        tab.insert(0, "regime", regime)
        intervention_tables.append(tab)

    pd.concat(
        intervention_tables,
        ignore_index=True,
    ).to_csv(
        out_dir / "rcse_split_intervention_distribution.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 6. Save enriched dataset and metadata
    # -----------------------------------------------------------------
    print("\n6/6 Saving enriched benchmark...")

    parquet_path = out_dir / "rcse_experimental_with_splits.parquet"
    df.to_parquet(parquet_path, index=False)

    if not args.no_csv:
        df.to_csv(
            out_dir / "rcse_experimental_with_splits.csv",
            index=False,
        )

    # Temporal cut points for reproducibility
    temporal_train_max = temporal_cases.loc[
        temporal_cases["split_temporal"] == "train",
        "decision_time",
    ].max()
    temporal_cal_min = temporal_cases.loc[
        temporal_cases["split_temporal"] == "calibration",
        "decision_time",
    ].min()
    temporal_cal_max = temporal_cases.loc[
        temporal_cases["split_temporal"] == "calibration",
        "decision_time",
    ].max()
    temporal_test_min = temporal_cases.loc[
        temporal_cases["split_temporal"] == "test",
        "decision_time",
    ].min()

    metadata = {
        "input_file": str(input_path),
        "episodes": int(len(df)),
        "source_cases": int(df["source_case_id"].nunique()),
        "split_version": "v1",
        "seed": int(args.seed),
        "fractions": {
            "train": float(args.train_frac),
            "calibration": float(args.cal_frac),
            "test": float(args.test_frac),
        },
        "grouping_key": "source_case_id",
        "group_leakage_rule": (
            "All intervention variants derived from the same source_case_id "
            "must remain in the same partition for every split regime."
        ),
        "regimes": {
            "grouped_iid": {
                "description": (
                    "Random grouped split by source_case_id."
                ),
            },
            "temporal": {
                "description": (
                    "Source cases ordered by decision_time; earliest cases train, "
                    "middle cases calibration, latest cases test."
                ),
                "train_max_time": (
                    None
                    if pd.isna(temporal_train_max)
                    else str(temporal_train_max)
                ),
                "calibration_min_time": (
                    None
                    if pd.isna(temporal_cal_min)
                    else str(temporal_cal_min)
                ),
                "calibration_max_time": (
                    None
                    if pd.isna(temporal_cal_max)
                    else str(temporal_cal_max)
                ),
                "test_min_time": (
                    None
                    if pd.isna(temporal_test_min)
                    else str(temporal_test_min)
                ),
            },
            "ood_exposure": {
                "description": (
                    "All R4_99_100 source cases held out exclusively as OOD test. "
                    "R1-R3 source cases form train/calibration/in-domain test."
                ),
                "ood_risk_stratum": "R4_99_100",
            },
        },
    }

    with open(
        out_dir / "rcse_split_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("SPLIT BUILD COMPLETE")
    print("=" * 80)

    print("\nGrouped IID:")
    print(iid_summary.to_string(index=False))

    print("\nTemporal:")
    print(temporal_summary.to_string(index=False))

    print("\nOOD exposure:")
    print(ood_summary.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "rcse_experimental_with_splits.parquet",
        None if args.no_csv else "rcse_experimental_with_splits.csv",
        "rcse_grouped_iid_split_summary.csv",
        "rcse_temporal_split_summary.csv",
        "rcse_ood_exposure_split_summary.csv",
        "rcse_split_action_distribution.csv",
        "rcse_split_intervention_distribution.csv",
        "rcse_split_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
