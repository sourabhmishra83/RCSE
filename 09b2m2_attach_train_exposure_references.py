#!/usr/bin/env python
"""
09b2m2_attach_train_exposure_references.py

RCSE Step 09b.2m.2
Attach TRAIN-only source-case exposure reference percentiles to the already
validated RCSE v2 policy-ready test datasets.

Purpose
-------
Step 09b.2n correctly refuses to derive consequence normalization references
from held-out test exposure.

This patch computes, separately for each frozen regime:

    P50_train_exposure
    P90_train_exposure
    P99_train_exposure

using UNIQUE TRAIN source cases only from:
    rcse_experimental_with_splits.parquet

and attaches the three constant reference values to each corresponding
RCSE v2 policy-ready test dataset.

NO MODEL TRAINING.
NO RECALIBRATION.
NO POLICY RETUNING.
NO TEST-SET EXPOSURE USED TO DEFINE CONSEQUENCE REFERENCES.

Inputs
------
--experimental
    rcse_experimental_with_splits.parquet

--dataset-dir
    Step 09b.2m rcse_v2_policy_dataset directory.

Outputs
-------
The existing three RCSE v2 policy datasets are safely replaced after a
validation pass, with these added columns:

    train_exposure_p50_eur
    train_exposure_p90_eur
    train_exposure_p99_eur

Also writes:
    rcse_v2_train_exposure_references.csv
    rcse_v2_train_exposure_reference_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_CONFIG = {
    "grouped_iid": {
        "split_col": "split_grouped_iid",
        "train_label": "train",
        "dataset": "rcse_v2_policy_dataset_grouped_iid.parquet",
    },
    "temporal": {
        "split_col": "split_temporal",
        "train_label": "train",
        "dataset": "rcse_v2_policy_dataset_temporal.parquet",
    },
    "ood_exposure": {
        "split_col": "split_ood_exposure",
        "train_label": "train",
        "dataset": "rcse_v2_policy_dataset_ood_exposure.parquet",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Attach TRAIN-only exposure reference percentiles to RCSE v2 datasets."
    )
    p.add_argument(
        "--experimental",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Step 09b.2m rcse_v2_policy_dataset directory",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def build_source_case_exposure(
    exp: pd.DataFrame,
    split_col: str,
    train_label: str,
    regime: str,
) -> pd.DataFrame:
    """
    Return exactly one TRAIN exposure value per source case.

    Experimental intervention variants must not multiply-weight a source case.
    """
    train = exp[
        exp[split_col].astype(str).str.lower()
        == train_label.lower()
    ][
        [
            "source_case_id",
            "transaction_exposure_eur",
        ]
    ].copy()

    if train.empty:
        raise RuntimeError(
            f"{regime}: no TRAIN rows found using {split_col}={train_label!r}"
        )

    train["source_case_id"] = train["source_case_id"].astype(str)

    train["transaction_exposure_eur"] = pd.to_numeric(
        train["transaction_exposure_eur"],
        errors="coerce",
    ).abs()

    missing = int(
        train["transaction_exposure_eur"].isna().sum()
    )

    if missing:
        raise RuntimeError(
            f"{regime}: {missing:,} TRAIN episode rows have missing exposure."
        )

    # Verify that all intervention variants of a source case carry the same
    # attached transaction exposure. If not, consequence normalization is not
    # well-defined at source-case level.
    exposure_nunique = (
        train.groupby("source_case_id")[
            "transaction_exposure_eur"
        ]
        .nunique(dropna=False)
    )

    conflicts = exposure_nunique[
        exposure_nunique > 1
    ]

    if len(conflicts):
        examples = conflicts.head(10).index.tolist()

        raise AssertionError(
            f"{regime}: {len(conflicts):,} TRAIN source cases have conflicting "
            f"transaction_exposure_eur across experimental variants. "
            f"Examples: {examples}"
        )

    source = (
        train.groupby(
            "source_case_id",
            as_index=False,
        )[
            "transaction_exposure_eur"
        ]
        .first()
    )

    if source["source_case_id"].duplicated().any():
        raise AssertionError(
            f"{regime}: source-case exposure table is unexpectedly non-unique."
        )

    return source


def main() -> None:
    args = parse_args()

    exp_path = Path(
        args.experimental
    ).expanduser().resolve()

    dataset_dir = Path(
        args.dataset_dir
    ).expanduser().resolve()

    if not exp_path.exists():
        raise FileNotFoundError(exp_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    print("=" * 80)
    print("RCSE STEP 09b.2m.2 - ATTACH TRAIN EXPOSURE REFERENCES")
    print("=" * 80)
    print(f"Experimental : {exp_path}")
    print(f"Dataset dir  : {dataset_dir}")
    print("References   : UNIQUE TRAIN source cases only")
    print("NO TEST EXPOSURE USED TO DEFINE P50/P90/P99")

    exp = pd.read_parquet(exp_path)

    require_columns(
        exp,
        [
            "source_case_id",
            "transaction_exposure_eur",
            *[
                cfg["split_col"]
                for cfg in REGIME_CONFIG.values()
            ],
        ],
        "experimental split benchmark",
    )

    summary_rows = []

    for regime, cfg in REGIME_CONFIG.items():
        dataset_path = (
            dataset_dir
            / cfg["dataset"]
        )

        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        source = build_source_case_exposure(
            exp=exp,
            split_col=cfg["split_col"],
            train_label=cfg["train_label"],
            regime=regime,
        )

        exposure = source[
            "transaction_exposure_eur"
        ]

        p50 = float(
            exposure.quantile(
                0.50
            )
        )

        p90 = float(
            exposure.quantile(
                0.90
            )
        )

        p99 = float(
            exposure.quantile(
                0.99
            )
        )

        if not (
            np.isfinite(p50)
            and np.isfinite(p90)
            and np.isfinite(p99)
        ):
            raise RuntimeError(
                f"{regime}: non-finite TRAIN exposure reference."
            )

        if not (
            0 <= p50 <= p90 <= p99
        ):
            raise AssertionError(
                f"{regime}: invalid exposure percentile ordering: "
                f"P50={p50}, P90={p90}, P99={p99}"
            )

        df = pd.read_parquet(
            dataset_path
        )

        require_columns(
            df,
            [
                "benchmark_episode_id",
                "transaction_exposure_eur",
            ],
            f"v2 policy dataset/{regime}",
        )

        if df["benchmark_episode_id"].duplicated().any():
            raise AssertionError(
                f"{regime}: duplicate benchmark_episode_id in policy dataset."
            )

        # If columns already exist, verify rather than silently change them.
        refs = {
            "train_exposure_p50_eur": p50,
            "train_exposure_p90_eur": p90,
            "train_exposure_p99_eur": p99,
        }

        for col, val in refs.items():
            if col in df.columns:
                existing = pd.to_numeric(
                    df[col],
                    errors="coerce",
                ).dropna().unique()

                if len(existing) > 1:
                    raise RuntimeError(
                        f"{regime}: existing {col} has multiple values."
                    )

                if len(existing) == 1 and not np.isclose(
                    existing[0],
                    val,
                    rtol=1e-12,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"{regime}: existing {col}={existing[0]} conflicts "
                        f"with freshly computed TRAIN-only value={val}."
                    )

            df[col] = val

        # Safe write: temporary file then replace.
        temp_path = dataset_path.with_suffix(
            ".tmp.parquet"
        )

        df.to_parquet(
            temp_path,
            index=False,
        )

        # Verify the materialized file before replacement.
        check = pd.read_parquet(
            temp_path,
            columns=[
                "benchmark_episode_id",
                "train_exposure_p50_eur",
                "train_exposure_p90_eur",
                "train_exposure_p99_eur",
            ],
        )

        if len(check) != len(df):
            raise RuntimeError(
                f"{regime}: temporary output row-count mismatch."
            )

        dataset_path.unlink()
        temp_path.rename(
            dataset_path
        )

        summary_rows.append(
            {
                "regime": regime,
                "train_episode_rows_before_source_dedup": int(
                    (
                        exp[cfg["split_col"]]
                        .astype(str)
                        .str.lower()
                        == cfg["train_label"].lower()
                    ).sum()
                ),
                "train_unique_source_cases": int(
                    len(source)
                ),
                "train_exposure_p50_eur": p50,
                "train_exposure_p90_eur": p90,
                "train_exposure_p99_eur": p99,
                "policy_test_rows_preserved": int(
                    len(df)
                ),
                "dataset_file": str(
                    dataset_path
                ),
            }
        )

        print(
            f"\n[{regime}] "
            f"TRAIN source cases={len(source):,} "
            f"| P50={p50:,.6f} "
            f"| P90={p90:,.6f} "
            f"| P99={p99:,.6f} "
            f"| test rows preserved={len(df):,}"
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        dataset_dir
        / "rcse_v2_train_exposure_references.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    metadata = {
        "step": "09b.2m.2",
        "purpose": (
            "Attach frozen TRAIN-only source-case exposure percentiles required "
            "for RCSE v2 consequence normalization."
        ),
        "experimental_split_benchmark": str(
            exp_path
        ),
        "dataset_directory": str(
            dataset_dir
        ),
        "reference_population": (
            "Unique source_case_id values in the TRAIN partition of each frozen regime."
        ),
        "percentiles": [
            0.50,
            0.90,
            0.99,
        ],
        "test_exposure_used_to_define_reference": False,
        "training_performed": False,
        "recalibration_performed": False,
        "policy_retuned": False,
        "gate_modified": False,
        "columns_added": [
            "train_exposure_p50_eur",
            "train_exposure_p90_eur",
            "train_exposure_p99_eur",
        ],
        "next_step": (
            "Rerun the unchanged Step 09b.2n frozen RCSE v2 simulation."
        ),
    }

    metadata_path = (
        dataset_dir
        / "rcse_v2_train_exposure_reference_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("STEP 09b.2m.2 COMPLETE")
    print("=" * 80)

    print("\nTRAIN-only exposure references:")
    print(
        summary[
            [
                "regime",
                "train_unique_source_cases",
                "train_exposure_p50_eur",
                "train_exposure_p90_eur",
                "train_exposure_p99_eur",
                "policy_test_rows_preserved",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(f"  - {summary_path}")
    print(f"  - {metadata_path}")

    print(
        "\nNext: rerun 09b2n_run_rcse_v2_policy_simulation.py unchanged."
    )


if __name__ == "__main__":
    main()
