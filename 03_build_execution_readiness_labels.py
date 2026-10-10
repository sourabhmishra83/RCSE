"""
03_build_execution_readiness_labels.py

Build transparent execution-readiness labels for the RCSE benchmark.

Input
-----
rcse_base_v2.parquet

Output
------
rcse_labeled.parquet
rcse_labeled.csv
execution_readiness_summary.csv
execution_readiness_reason_summary.csv
execution_readiness_by_item_category.csv
execution_readiness_by_risk_stratum.csv
execution_readiness_metadata.json

Label classes
-------------
READY
WAIT_FOR_EVIDENCE
REVIEW

Important methodological rule
-----------------------------
The label is derived only from information available at t0 plus explicit,
documented process-semantics rules. Future events are NOT used to assign
READY / WAIT_FOR_EVIDENCE / REVIEW.

Future outcome columns remain available only for downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABEL_READY = "READY"
LABEL_WAIT = "WAIT_FOR_EVIDENCE"
LABEL_REVIEW = "REVIEW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create transparent execution-readiness labels for RCSE."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to rcse_base_v2.parquet",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: input parent.",
    )
    parser.add_argument(
        "--po-inv-tolerance",
        type=float,
        default=0.01,
        help="Relative PO-vs-invoice mismatch threshold. Default: 0.01 (1%%).",
    )
    parser.add_argument(
        "--inv-gr-tolerance",
        type=float,
        default=0.01,
        help="Relative invoice-vs-GR mismatch threshold. Default: 0.01 (1%%).",
    )
    parser.add_argument(
        "--review-prior-price-change",
        action="store_true",
        help="If set, any price change before t0 forces REVIEW.",
    )
    parser.add_argument(
        "--review-prior-quantity-change",
        action="store_true",
        help="If set, any quantity change before t0 forces REVIEW.",
    )
    parser.add_argument(
        "--review-prior-gr-cancel",
        action="store_true",
        help="If set, any GR cancellation before t0 forces REVIEW.",
    )
    parser.add_argument(
        "--review-prior-invoice-cancel",
        action="store_true",
        help="If set, any invoice cancellation before t0 forces REVIEW.",
    )
    parser.add_argument(
        "--review-prior-payment-block",
        action="store_true",
        help="If set, any payment block history before t0 forces REVIEW.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing rcse_labeled.csv.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE EXECUTION READINESS LABEL BUILDER")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")
    print()

    print("1/5 Loading benchmark...")
    df = pd.read_parquet(input_path)

    required = [
        "case_id",
        "item_category",
        "gr_available_at_t0",
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "had_gr_cancellation_before_t0",
        "had_invoice_cancellation_before_t0",
        "had_price_change_before_t0",
        "had_quantity_change_before_t0",
        "had_payment_block_before_t0",
    ]
    require_columns(df, required)

    print(f"   Rows: {len(df):,}")

    print("\n2/5 Building reason flags...")

    # -------------------------------------------------------------
    # Evidence completeness flags
    # -------------------------------------------------------------
    df["reason_missing_po_value"] = df["po_value_initial_eur"].isna()
    df["reason_missing_invoice_value"] = df["invoice_value_t0_eur"].isna()

    # Goods receipt evidence is required for 3-way execution readiness.
    df["reason_missing_gr"] = ~bool_series(df, "gr_available_at_t0")

    df["reason_missing_gr_value"] = (
        bool_series(df, "gr_available_at_t0")
        & df["latest_gr_value_before_t0_eur"].isna()
    )

    # -------------------------------------------------------------
    # Mismatch flags
    # -------------------------------------------------------------
    df["reason_po_inv_mismatch"] = (
        df["inv_po_abs_rel_diff"].notna()
        & (df["inv_po_abs_rel_diff"] > args.po_inv_tolerance)
    )

    df["reason_inv_gr_mismatch"] = (
        bool_series(df, "gr_available_at_t0")
        & df["inv_gr_abs_rel_diff"].notna()
        & (df["inv_gr_abs_rel_diff"] > args.inv_gr_tolerance)
    )

    # If GR exists but relative comparison could not be computed,
    # make that explicit rather than silently calling the case READY.
    df["reason_gr_match_unknown"] = (
        bool_series(df, "gr_available_at_t0")
        & df["latest_gr_value_before_t0_eur"].notna()
        & df["invoice_value_t0_eur"].notna()
        & df["inv_gr_abs_rel_diff"].isna()
    )

    df["reason_po_inv_match_unknown"] = (
        df["po_value_initial_eur"].notna()
        & df["invoice_value_t0_eur"].notna()
        & df["inv_po_abs_rel_diff"].isna()
    )

    # -------------------------------------------------------------
    # Historical-control / rework flags
    # These are preserved separately even if not configured to force REVIEW.
    # -------------------------------------------------------------
    df["reason_prior_gr_cancellation"] = bool_series(
        df, "had_gr_cancellation_before_t0"
    )
    df["reason_prior_invoice_cancellation"] = bool_series(
        df, "had_invoice_cancellation_before_t0"
    )
    df["reason_prior_price_change"] = bool_series(
        df, "had_price_change_before_t0"
    )
    df["reason_prior_quantity_change"] = bool_series(
        df, "had_quantity_change_before_t0"
    )
    df["reason_prior_payment_block"] = bool_series(
        df, "had_payment_block_before_t0"
    )

    # -------------------------------------------------------------
    # Mandatory REVIEW rules
    # -------------------------------------------------------------
    mandatory_review_reasons = [
        "reason_missing_po_value",
        "reason_missing_invoice_value",
        "reason_missing_gr_value",
        "reason_po_inv_mismatch",
        "reason_inv_gr_mismatch",
        "reason_gr_match_unknown",
        "reason_po_inv_match_unknown",
    ]

    optional_review_reasons = []

    if args.review_prior_price_change:
        optional_review_reasons.append("reason_prior_price_change")
    if args.review_prior_quantity_change:
        optional_review_reasons.append("reason_prior_quantity_change")
    if args.review_prior_gr_cancel:
        optional_review_reasons.append("reason_prior_gr_cancellation")
    if args.review_prior_invoice_cancel:
        optional_review_reasons.append("reason_prior_invoice_cancellation")
    if args.review_prior_payment_block:
        optional_review_reasons.append("reason_prior_payment_block")

    review_reasons = mandatory_review_reasons + optional_review_reasons

    df["review_required_by_evidence"] = (
        df[review_reasons].any(axis=1)
        if review_reasons
        else False
    )

    # -------------------------------------------------------------
    # Label precedence
    #
    # 1. REVIEW if explicit contradiction/insufficiency exists
    # 2. WAIT_FOR_EVIDENCE if required GR is absent but no contradiction exists
    # 3. READY otherwise
    #
    # This is deliberate: a case with both missing GR and an existing PO/INV
    # mismatch should not be classified merely as WAIT.
    # -------------------------------------------------------------
    df["execution_readiness_label"] = LABEL_READY

    df.loc[
        df["reason_missing_gr"] & ~df["review_required_by_evidence"],
        "execution_readiness_label",
    ] = LABEL_WAIT

    df.loc[
        df["review_required_by_evidence"],
        "execution_readiness_label",
    ] = LABEL_REVIEW

    # -------------------------------------------------------------
    # Compact reason code string for auditing
    # -------------------------------------------------------------
    reason_cols = [
        "reason_missing_po_value",
        "reason_missing_invoice_value",
        "reason_missing_gr",
        "reason_missing_gr_value",
        "reason_po_inv_mismatch",
        "reason_inv_gr_mismatch",
        "reason_gr_match_unknown",
        "reason_po_inv_match_unknown",
        "reason_prior_gr_cancellation",
        "reason_prior_invoice_cancellation",
        "reason_prior_price_change",
        "reason_prior_quantity_change",
        "reason_prior_payment_block",
    ]

    reason_names = {
        "reason_missing_po_value": "MISSING_PO_VALUE",
        "reason_missing_invoice_value": "MISSING_INVOICE_VALUE",
        "reason_missing_gr": "MISSING_GR",
        "reason_missing_gr_value": "MISSING_GR_VALUE",
        "reason_po_inv_mismatch": "PO_INV_MISMATCH",
        "reason_inv_gr_mismatch": "INV_GR_MISMATCH",
        "reason_gr_match_unknown": "GR_MATCH_UNKNOWN",
        "reason_po_inv_match_unknown": "PO_INV_MATCH_UNKNOWN",
        "reason_prior_gr_cancellation": "PRIOR_GR_CANCELLATION",
        "reason_prior_invoice_cancellation": "PRIOR_INVOICE_CANCELLATION",
        "reason_prior_price_change": "PRIOR_PRICE_CHANGE",
        "reason_prior_quantity_change": "PRIOR_QUANTITY_CHANGE",
        "reason_prior_payment_block": "PRIOR_PAYMENT_BLOCK",
    }

    def build_reason_codes(row: pd.Series) -> str:
        vals = [
            reason_names[c]
            for c in reason_cols
            if bool(row[c])
        ]
        return "|".join(vals) if vals else "NONE"

    df["execution_readiness_reason_codes"] = df.apply(
        build_reason_codes,
        axis=1,
    )

    # Binary helper targets for later modeling.
    df["target_ready"] = (
        df["execution_readiness_label"] == LABEL_READY
    ).astype(int)

    df["target_wait_for_evidence"] = (
        df["execution_readiness_label"] == LABEL_WAIT
    ).astype(int)

    df["target_review"] = (
        df["execution_readiness_label"] == LABEL_REVIEW
    ).astype(int)

    # Safe-to-execute reference label for binary selective-execution experiments.
    df["target_execution_ready_binary"] = df["target_ready"]

    print("\n3/5 Writing labeled dataset...")

    parquet_path = out_dir / "rcse_labeled.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"   Wrote {parquet_path.name}")

    if not args.no_csv:
        csv_path = out_dir / "rcse_labeled.csv"
        df.to_csv(csv_path, index=False)
        print(f"   Wrote {csv_path.name}")

    print("\n4/5 Writing summaries...")

    # Main class summary
    summary = (
        df["execution_readiness_label"]
        .value_counts(dropna=False)
        .rename_axis("execution_readiness_label")
        .reset_index(name="count")
    )
    summary["pct"] = (summary["count"] / len(df) * 100).round(4)
    summary.to_csv(
        out_dir / "execution_readiness_summary.csv",
        index=False,
    )

    # Reason summary
    reason_rows = []
    for c in reason_cols:
        reason_rows.append(
            {
                "reason_flag": c,
                "reason_code": reason_names[c],
                "count": int(df[c].sum()),
                "pct": round(float(df[c].mean() * 100), 4),
                "forces_review_in_this_run": c in review_reasons,
            }
        )
    reason_summary = pd.DataFrame(reason_rows).sort_values(
        "count",
        ascending=False,
    )
    reason_summary.to_csv(
        out_dir / "execution_readiness_reason_summary.csv",
        index=False,
    )

    # By item category
    by_item = (
        df.groupby(
            ["item_category", "execution_readiness_label"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_item["pct_within_item_category"] = (
        by_item["count"]
        / by_item.groupby("item_category")["count"].transform("sum")
        * 100
    ).round(4)
    by_item.to_csv(
        out_dir / "execution_readiness_by_item_category.csv",
        index=False,
    )

    # By transaction risk stratum when available
    if "transaction_risk_stratum" in df.columns:
        by_risk = (
            df.groupby(
                [
                    "transaction_risk_stratum",
                    "execution_readiness_label",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )
        by_risk["pct_within_risk_stratum"] = (
            by_risk["count"]
            / by_risk.groupby("transaction_risk_stratum")["count"].transform("sum")
            * 100
        ).round(4)
        by_risk.to_csv(
            out_dir / "execution_readiness_by_risk_stratum.csv",
            index=False,
        )

    # Metadata
    metadata = {
        "input_file": str(input_path),
        "rows": int(len(df)),
        "label_version": "execution_readiness_v1",
        "labels": [
            LABEL_READY,
            LABEL_WAIT,
            LABEL_REVIEW,
        ],
        "label_precedence": [
            "REVIEW",
            "WAIT_FOR_EVIDENCE",
            "READY",
        ],
        "decision_point": "first Record Invoice Receipt",
        "po_invoice_relative_tolerance": float(args.po_inv_tolerance),
        "invoice_gr_relative_tolerance": float(args.inv_gr_tolerance),
        "mandatory_review_reasons": mandatory_review_reasons,
        "optional_review_reasons_enabled": optional_review_reasons,
        "methodological_rule": (
            "Execution-readiness labels are derived only from t0-visible evidence "
            "and documented process semantics. Future outcome columns are not used "
            "to assign READY, WAIT_FOR_EVIDENCE, or REVIEW."
        ),
        "important_interpretation": {
            "READY": (
                "Required t0 evidence is present and no configured contradiction "
                "or review condition is triggered."
            ),
            "WAIT_FOR_EVIDENCE": (
                "Required GR evidence is absent at t0 but no other configured "
                "contradiction is currently present."
            ),
            "REVIEW": (
                "Available evidence is missing/contradictory in a way that should "
                "prevent autonomous execution under the configured reference policy."
            ),
        },
    }

    with open(
        out_dir / "execution_readiness_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n5/5 Validation...")
    assert df["case_id"].is_unique, "Expected one row per case."
    assert df["execution_readiness_label"].notna().all()
    assert set(df["execution_readiness_label"].unique()).issubset(
        {LABEL_READY, LABEL_WAIT, LABEL_REVIEW}
    )

    # A READY case cannot be missing required GR under this reference policy.
    assert not (
        (df["execution_readiness_label"] == LABEL_READY)
        & df["reason_missing_gr"]
    ).any()

    # A REVIEW case must have at least one configured review trigger.
    assert (
        df.loc[
            df["execution_readiness_label"] == LABEL_REVIEW,
            "review_required_by_evidence",
        ]
        .all()
    )

    print("   PASS: one row per case")
    print("   PASS: valid labels")
    print("   PASS: READY never has missing GR")
    print("   PASS: REVIEW always has configured review trigger")

    print("\n" + "=" * 80)
    print("LABEL BUILD COMPLETE")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nTop readiness reasons:")
    print(
        reason_summary[
            ["reason_code", "count", "pct", "forces_review_in_this_run"]
        ]
        .head(15)
        .to_string(index=False)
    )

    print("\nOutputs:")
    for name in [
        "rcse_labeled.parquet",
        None if args.no_csv else "rcse_labeled.csv",
        "execution_readiness_summary.csv",
        "execution_readiness_reason_summary.csv",
        "execution_readiness_by_item_category.csv",
        (
            "execution_readiness_by_risk_stratum.csv"
            if "transaction_risk_stratum" in df.columns
            else None
        ),
        "execution_readiness_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
