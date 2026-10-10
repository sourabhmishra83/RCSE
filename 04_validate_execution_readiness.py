"""
04_validate_execution_readiness.py

Diagnostic validation for RCSE execution-readiness labels.

Purpose
-------
Validate whether REVIEW cases in execution_readiness_v1 reflect genuine
pre-t0 evidence contradictions or could be artifacts of repeated / partial
goods receipts and cumulative-value semantics.

Input
-----
rcse_labeled.parquet

Outputs
-------
review_reason_combinations.csv
review_receipt_multiplicity.csv
review_mismatch_magnitude_summary.csv
review_value_pattern_sample.csv
review_by_item_category.csv
review_by_gr_count.csv
review_by_invoice_count.csv
review_downstream_outcomes.csv
review_validation_metadata.json

Important
---------
This script is diagnostic only. It does NOT modify labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_REVIEW = "REVIEW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RCSE execution-readiness REVIEW labels."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to rcse_labeled.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <input parent>/review_validation",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of REVIEW rows to export for manual inspection.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def pct(n: int, d: int) -> float:
    return round((n / d * 100), 4) if d else np.nan


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "review_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE REVIEW-LABEL VALIDATION")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")

    print("\n1/7 Loading labeled benchmark...")
    df = pd.read_parquet(input_path)

    required = [
        "case_id",
        "execution_readiness_label",
        "execution_readiness_reason_codes",
        "item_category",
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "n_goods_receipts_before_t0",
        "n_invoice_receipts_at_t0_history",
        "n_cancel_gr_before_t0",
        "n_cancel_invoice_before_t0",
        "n_price_change_before_t0",
        "n_quantity_change_before_t0",
        "outcome_eventually_cleared",
        "outcome_invoice_cancelled_after_t0",
        "outcome_payment_block_removed_after_t0",
        "outcome_future_gr",
        "outcome_future_gr_cancel",
        "outcome_future_price_change",
        "outcome_future_quantity_change",
        "outcome_subsequent_invoice",
        "outcome_additional_invoice_receipt",
        "outcome_any_post_t0_exception",
    ]
    require_columns(df, required)

    review = df.loc[
        df["execution_readiness_label"] == LABEL_REVIEW
    ].copy()

    print(f"   Total rows : {len(df):,}")
    print(f"   REVIEW rows: {len(review):,}")

    if review.empty:
        raise RuntimeError("No REVIEW cases found.")

    # -----------------------------------------------------------------
    # 2. Reason combinations
    # -----------------------------------------------------------------
    print("\n2/7 Decomposing REVIEW reason combinations...")

    combo = (
        review["execution_readiness_reason_codes"]
        .fillna("NONE")
        .value_counts()
        .rename_axis("reason_combination")
        .reset_index(name="count")
    )
    combo["pct_of_review"] = (
        combo["count"] / len(review) * 100
    ).round(4)
    combo.to_csv(
        out_dir / "review_reason_combinations.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 3. Receipt / invoice multiplicity
    # -----------------------------------------------------------------
    print("\n3/7 Profiling receipt and invoice multiplicity...")

    review["gr_count_bucket"] = pd.cut(
        review["n_goods_receipts_before_t0"],
        bins=[-1, 0, 1, 2, 3, 5, 10, np.inf],
        labels=[
            "0",
            "1",
            "2",
            "3",
            "4-5",
            "6-10",
            "11+",
        ],
    )

    review["invoice_count_bucket"] = pd.cut(
        review["n_invoice_receipts_at_t0_history"],
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=[
            "1",
            "2",
            "3",
            "4-5",
            "6-10",
            "11+",
        ],
        include_lowest=True,
    )

    gr_count = (
        review.groupby("gr_count_bucket", observed=False)
        .size()
        .reset_index(name="count")
    )
    gr_count["pct_of_review"] = (
        gr_count["count"] / len(review) * 100
    ).round(4)
    gr_count.to_csv(
        out_dir / "review_by_gr_count.csv",
        index=False,
    )

    inv_count = (
        review.groupby("invoice_count_bucket", observed=False)
        .size()
        .reset_index(name="count")
    )
    inv_count["pct_of_review"] = (
        inv_count["count"] / len(review) * 100
    ).round(4)
    inv_count.to_csv(
        out_dir / "review_by_invoice_count.csv",
        index=False,
    )

    multiplicity_rows = [
        {
            "metric": "review_cases",
            "count": len(review),
            "pct_of_review": 100.0,
        },
        {
            "metric": "more_than_one_gr_before_t0",
            "count": int((review["n_goods_receipts_before_t0"] > 1).sum()),
            "pct_of_review": pct(
                int((review["n_goods_receipts_before_t0"] > 1).sum()),
                len(review),
            ),
        },
        {
            "metric": "more_than_one_invoice_receipt_by_t0",
            "count": int(
                (review["n_invoice_receipts_at_t0_history"] > 1).sum()
            ),
            "pct_of_review": pct(
                int(
                    (review["n_invoice_receipts_at_t0_history"] > 1).sum()
                ),
                len(review),
            ),
        },
        {
            "metric": "prior_gr_cancellation",
            "count": int((review["n_cancel_gr_before_t0"] > 0).sum()),
            "pct_of_review": pct(
                int((review["n_cancel_gr_before_t0"] > 0).sum()),
                len(review),
            ),
        },
        {
            "metric": "prior_invoice_cancellation",
            "count": int(
                (review["n_cancel_invoice_before_t0"] > 0).sum()
            ),
            "pct_of_review": pct(
                int(
                    (review["n_cancel_invoice_before_t0"] > 0).sum()
                ),
                len(review),
            ),
        },
        {
            "metric": "prior_price_change",
            "count": int((review["n_price_change_before_t0"] > 0).sum()),
            "pct_of_review": pct(
                int((review["n_price_change_before_t0"] > 0).sum()),
                len(review),
            ),
        },
        {
            "metric": "prior_quantity_change",
            "count": int(
                (review["n_quantity_change_before_t0"] > 0).sum()
            ),
            "pct_of_review": pct(
                int(
                    (review["n_quantity_change_before_t0"] > 0).sum()
                ),
                len(review),
            ),
        },
    ]

    pd.DataFrame(multiplicity_rows).to_csv(
        out_dir / "review_receipt_multiplicity.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 4. Mismatch magnitude
    # -----------------------------------------------------------------
    print("\n4/7 Summarizing mismatch magnitudes...")

    mismatch_vars = [
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
    ]

    mismatch_summary = (
        review[mismatch_vars]
        .describe(
            percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        .T
        .reset_index()
        .rename(columns={"index": "metric"})
    )
    mismatch_summary.to_csv(
        out_dir / "review_mismatch_magnitude_summary.csv",
        index=False,
    )

    # Threshold bands help determine whether cases barely cross 1%
    # or are materially different.
    for col in mismatch_vars:
        band_col = f"{col}_band"
        review[band_col] = pd.cut(
            review[col],
            bins=[
                -np.inf,
                0.01,
                0.02,
                0.05,
                0.10,
                0.25,
                0.50,
                1.00,
                np.inf,
            ],
            labels=[
                "<=1%",
                "1-2%",
                "2-5%",
                "5-10%",
                "10-25%",
                "25-50%",
                "50-100%",
                ">100%",
            ],
        )

    mismatch_band_rows = []
    for col in mismatch_vars:
        band_col = f"{col}_band"
        counts = (
            review[band_col]
            .value_counts(dropna=False, sort=False)
            .rename_axis("band")
            .reset_index(name="count")
        )
        counts["metric"] = col
        counts["pct_of_review"] = (
            counts["count"] / len(review) * 100
        ).round(4)
        mismatch_band_rows.append(counts)

    pd.concat(
        mismatch_band_rows,
        ignore_index=True,
    ).to_csv(
        out_dir / "review_mismatch_bands.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 5. Item category and structural profile
    # -----------------------------------------------------------------
    print("\n5/7 Profiling REVIEW cases by process type...")

    by_item = (
        review.groupby("item_category", dropna=False)
        .agg(
            review_cases=("case_id", "count"),
            median_gr_count=("n_goods_receipts_before_t0", "median"),
            mean_gr_count=("n_goods_receipts_before_t0", "mean"),
            pct_multi_gr=(
                "n_goods_receipts_before_t0",
                lambda s: round(float((s > 1).mean() * 100), 4),
            ),
            median_inv_po_diff=("inv_po_abs_rel_diff", "median"),
            median_inv_gr_diff=("inv_gr_abs_rel_diff", "median"),
        )
        .reset_index()
    )

    by_item["pct_of_review"] = (
        by_item["review_cases"] / len(review) * 100
    ).round(4)

    by_item.to_csv(
        out_dir / "review_by_item_category.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 6. Downstream outcomes (diagnostic only)
    # -----------------------------------------------------------------
    print("\n6/7 Summarizing downstream outcomes...")

    outcome_cols = [
        "outcome_eventually_cleared",
        "outcome_invoice_cancelled_after_t0",
        "outcome_payment_block_removed_after_t0",
        "outcome_future_gr",
        "outcome_future_gr_cancel",
        "outcome_future_price_change",
        "outcome_future_quantity_change",
        "outcome_subsequent_invoice",
        "outcome_additional_invoice_receipt",
        "outcome_any_post_t0_exception",
    ]

    outcome_rows = []
    for col in outcome_cols:
        s = review[col].fillna(False).astype(bool)
        outcome_rows.append(
            {
                "outcome": col,
                "true_count": int(s.sum()),
                "pct_of_review": round(float(s.mean() * 100), 4),
            }
        )

    pd.DataFrame(outcome_rows).to_csv(
        out_dir / "review_downstream_outcomes.csv",
        index=False,
    )

    # -----------------------------------------------------------------
    # 7. Manual inspection sample
    # -----------------------------------------------------------------
    print("\n7/7 Writing manual-inspection sample and metadata...")

    sample_cols = [
        "case_id",
        "item_category",
        "execution_readiness_reason_codes",
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "n_goods_receipts_before_t0",
        "n_invoice_receipts_at_t0_history",
        "n_cancel_gr_before_t0",
        "n_cancel_invoice_before_t0",
        "n_price_change_before_t0",
        "n_quantity_change_before_t0",
        "gr_available_at_t0",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "outcome_eventually_cleared",
        "outcome_invoice_cancelled_after_t0",
        "outcome_payment_block_removed_after_t0",
        "outcome_future_gr",
        "outcome_future_gr_cancel",
        "outcome_future_price_change",
        "outcome_future_quantity_change",
        "outcome_additional_invoice_receipt",
        "outcome_any_post_t0_exception",
    ]

    sample_cols = [c for c in sample_cols if c in review.columns]

    # Deterministic stratified sample by reason combination where possible.
    n_sample = min(args.sample_size, len(review))
    sample = (
        review[sample_cols]
        .sort_values(
            [
                "execution_readiness_reason_codes",
                "inv_gr_abs_rel_diff",
                "inv_po_abs_rel_diff",
            ],
            ascending=[True, False, False],
            kind="stable",
        )
        .head(n_sample)
    )

    sample.to_csv(
        out_dir / "review_value_pattern_sample.csv",
        index=False,
    )

    # Useful topline diagnostics
    multi_gr = int((review["n_goods_receipts_before_t0"] > 1).sum())
    exact_one_gr = int((review["n_goods_receipts_before_t0"] == 1).sum())
    zero_gr = int((review["n_goods_receipts_before_t0"] == 0).sum())

    metadata = {
        "input_file": str(input_path),
        "total_rows": int(len(df)),
        "review_rows": int(len(review)),
        "review_pct_of_all": round(float(len(review) / len(df) * 100), 4),
        "diagnostic_only": True,
        "label_modification": False,
        "review_gr_structure": {
            "zero_gr_before_t0": zero_gr,
            "exactly_one_gr_before_t0": exact_one_gr,
            "more_than_one_gr_before_t0": multi_gr,
        },
        "interpretation_goal": (
            "Determine whether REVIEW cases, especially INV_GR_MISMATCH cases, "
            "reflect genuine evidence contradiction or artifacts from repeated/"
            "partial goods receipts and cumulative-value semantics."
        ),
        "future_outcome_usage": (
            "Post-t0 outcomes are summarized for diagnostic validation only and "
            "are not used to construct execution-readiness labels."
        ),
    }

    with open(
        out_dir / "review_validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("VALIDATION OUTPUT COMPLETE")
    print("=" * 80)

    print("\nTop reason combinations:")
    print(combo.head(15).to_string(index=False))

    print("\nGR multiplicity among REVIEW cases:")
    print(gr_count.to_string(index=False))

    print("\nOutputs:")
    for name in [
        "review_reason_combinations.csv",
        "review_receipt_multiplicity.csv",
        "review_mismatch_magnitude_summary.csv",
        "review_mismatch_bands.csv",
        "review_value_pattern_sample.csv",
        "review_by_item_category.csv",
        "review_by_gr_count.csv",
        "review_by_invoice_count.csv",
        "review_downstream_outcomes.csv",
        "review_validation_metadata.json",
    ]:
        print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
