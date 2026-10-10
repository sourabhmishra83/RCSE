"""
01_build_rcse_base.py

Builds the first RCSE benchmark table from the BPI Challenge 2019 XES log.

Design choices
--------------
- Domain: Procure-to-Pay, 3-way matching only.
- Decision point t0: FIRST "Record Invoice Receipt" event in each case.
- Features: only information available at or before t0.
- Outcomes: events strictly after t0.
- No LLM-generated labels.
- "Cumulative net worth (EUR)" is NOT blindly summed. Values are captured
  at specific events / latest pre-decision observations.

Outputs
-------
rcse_base.parquet
rcse_base.csv                      (optional; enabled by default)
rcse_filter_report.csv
rcse_item_category_counts.csv
rcse_outcome_summary.csv
rcse_numeric_summary.csv
rcse_missingness.csv
rcse_build_metadata.json

Usage
-----
python 01_build_rcse_base.py --xes "C:\\path\\to\\BPI_Challenge_2019.xes"

Optional:
python 01_build_rcse_base.py --xes "C:\\path\\to\\BPI_Challenge_2019.xes" --out "C:\\path\\to\\rcse_output"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pm4py


# ---------------------------------------------------------------------
# Constants from the actual BPI Challenge 2019 schema
# ---------------------------------------------------------------------

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"
VALUE_COL = "Cumulative net worth (EUR)"

ITEM_CATEGORY_COL = "case:Item Category"
GR_BASED_IV_COL = "case:GR-Based Inv. Verif."
GOODS_RECEIPT_FLAG_COL = "case:Goods Receipt"

THREE_WAY_CATEGORIES = {
    "3-way match, invoice after GR",
    "3-way match, invoice before GR",
}

ACT_CREATE_PO = "Create Purchase Order Item"
ACT_VENDOR_INVOICE = "Vendor creates invoice"
ACT_INVOICE_RECEIPT = "Record Invoice Receipt"
ACT_GOODS_RECEIPT = "Record Goods Receipt"
ACT_CANCEL_GR = "Cancel Goods Receipt"
ACT_CANCEL_INVOICE = "Cancel Invoice Receipt"
ACT_CLEAR_INVOICE = "Clear Invoice"
ACT_PRICE_CHANGE = "Change Price"
ACT_QTY_CHANGE = "Change Quantity"
ACT_SET_PAYMENT_BLOCK = "Set Payment Block"
ACT_REMOVE_PAYMENT_BLOCK = "Remove Payment Block"
ACT_SUBSEQUENT_INVOICE = "Record Subsequent Invoice"
ACT_SERVICE_ENTRY = "Record Service Entry Sheet"
ACT_DELETE_PO = "Delete Purchase Order Item"
ACT_BLOCK_PO = "Block Purchase Order Item"
ACT_REACTIVATE_PO = "Reactivate Purchase Order Item"
ACT_CHANGE_APPROVAL = "Change Approval for Purchase Order"
ACT_RELEASE_PO = "Release Purchase Order"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RCSE benchmark from BPIC 2019 XES.")
    parser.add_argument(
        "--xes",
        required=True,
        help="Path to BPI Challenge 2019 .xes or .xes.gz file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Default: <XES parent>/rcse_output",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the full rcse_base.csv (Parquet is always written).",
    )
    return parser.parse_args()


def normalize_bool(value: Any) -> Any:
    """Convert common truthy/falsy representations while preserving NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return value


def first_non_null(series: pd.Series) -> Any:
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def last_non_null(series: pd.Series) -> Any:
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan


def safe_days(delta: pd.Timedelta | Any) -> float:
    if pd.isna(delta):
        return np.nan
    return delta.total_seconds() / 86400.0


def event_count(df: pd.DataFrame, activity: str) -> int:
    return int((df[ACTIVITY_COL] == activity).sum())


def has_event(df: pd.DataFrame, activity: str) -> bool:
    return bool((df[ACTIVITY_COL] == activity).any())


def event_value_first(df: pd.DataFrame, activity: str) -> float:
    s = df.loc[df[ACTIVITY_COL] == activity, VALUE_COL].dropna()
    return float(s.iloc[0]) if not s.empty else np.nan


def event_value_last(df: pd.DataFrame, activity: str) -> float:
    s = df.loc[df[ACTIVITY_COL] == activity, VALUE_COL].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def event_time_first(df: pd.DataFrame, activity: str) -> pd.Timestamp | pd.NaT:
    s = df.loc[df[ACTIVITY_COL] == activity, TIME_COL].dropna()
    return s.iloc[0] if not s.empty else pd.NaT


def event_time_last(df: pd.DataFrame, activity: str) -> pd.Timestamp | pd.NaT:
    s = df.loc[df[ACTIVITY_COL] == activity, TIME_COL].dropna()
    return s.iloc[-1] if not s.empty else pd.NaT


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            "Required columns missing from XES conversion:\n  - "
            + "\n  - ".join(missing)
        )


def case_static_value(g: pd.DataFrame, col: str) -> Any:
    return first_non_null(g[col]) if col in g.columns else np.nan


# ---------------------------------------------------------------------
# Build one RCSE row
# ---------------------------------------------------------------------

def build_case_row(g: pd.DataFrame) -> dict[str, Any] | None:
    """
    Build one row per PO-item case at the first Record Invoice Receipt.

    Pre-decision features use events <= t0.
    Outcome variables use events > t0.
    """
    g = g.sort_values([TIME_COL, "_event_order"], kind="stable").copy()

    invoice_rows = g[g[ACTIVITY_COL] == ACT_INVOICE_RECEIPT]
    if invoice_rows.empty:
        return None

    # First recorded invoice receipt is the decision point.
    t0_idx = invoice_rows.index[0]
    t0 = g.loc[t0_idx, TIME_COL]

    # Because equal-timestamp ordering can matter, use original event order too.
    t0_order = g.loc[t0_idx, "_event_order"]

    past_mask = (g[TIME_COL] < t0) | (
        (g[TIME_COL] == t0) & (g["_event_order"] <= t0_order)
    )
    future_mask = (g[TIME_COL] > t0) | (
        (g[TIME_COL] == t0) & (g["_event_order"] > t0_order)
    )

    past = g.loc[past_mask].copy()
    future = g.loc[future_mask].copy()

    case_id = case_static_value(g, CASE_COL)
    item_category = case_static_value(g, ITEM_CATEGORY_COL)

    po_time = event_time_first(past, ACT_CREATE_PO)
    vendor_invoice_time = event_time_first(past, ACT_VENDOR_INVOICE)
    last_gr_time = event_time_last(past, ACT_GOODS_RECEIPT)

    clear_time = event_time_first(future, ACT_CLEAR_INVOICE)
    first_future_gr_time = event_time_first(future, ACT_GOODS_RECEIPT)

    po_value_initial = event_value_first(past, ACT_CREATE_PO)
    invoice_value_t0 = float(g.loc[t0_idx, VALUE_COL]) if pd.notna(g.loc[t0_idx, VALUE_COL]) else np.nan
    latest_gr_value_before_t0 = event_value_last(past, ACT_GOODS_RECEIPT)

    # Relative deviations are descriptive evidence variables only.
    # They do NOT imply business correctness by themselves.
    if pd.notna(po_value_initial) and abs(po_value_initial) > 1e-12 and pd.notna(invoice_value_t0):
        inv_po_abs_rel_diff = abs(invoice_value_t0 - po_value_initial) / abs(po_value_initial)
    else:
        inv_po_abs_rel_diff = np.nan

    if (
        pd.notna(latest_gr_value_before_t0)
        and abs(latest_gr_value_before_t0) > 1e-12
        and pd.notna(invoice_value_t0)
    ):
        inv_gr_abs_rel_diff = abs(invoice_value_t0 - latest_gr_value_before_t0) / abs(latest_gr_value_before_t0)
    else:
        inv_gr_abs_rel_diff = np.nan

    row = {
        # -------------------------------------------------------------
        # Identity / static case attributes
        # -------------------------------------------------------------
        "case_id": case_id,
        "purchasing_document": case_static_value(g, "case:Purchasing Document"),
        "item_id": case_static_value(g, "case:Item"),
        "vendor_id": case_static_value(g, "case:Vendor"),
        "vendor_name_anon": case_static_value(g, "case:Name"),
        "company": case_static_value(g, "case:Company"),
        "source_system": case_static_value(g, "case:Source"),
        "document_type": case_static_value(g, "case:Document Type"),
        "document_category": case_static_value(g, "case:Purch. Doc. Category name"),
        "item_type": case_static_value(g, "case:Item Type"),
        "item_category": item_category,
        "spend_classification": case_static_value(g, "case:Spend classification text"),
        "spend_area": case_static_value(g, "case:Spend area text"),
        "sub_spend_area": case_static_value(g, "case:Sub spend area text"),
        "gr_based_invoice_verification": normalize_bool(
            case_static_value(g, GR_BASED_IV_COL)
        ),
        "goods_receipt_required": normalize_bool(
            case_static_value(g, GOODS_RECEIPT_FLAG_COL)
        ),

        # -------------------------------------------------------------
        # Decision point
        # -------------------------------------------------------------
        "decision_time": t0,
        "decision_activity": ACT_INVOICE_RECEIPT,
        "case_start_time": g[TIME_COL].min(),
        "case_end_time": g[TIME_COL].max(),
        "events_total_case": int(len(g)),
        "events_available_at_t0": int(len(past)),
        "events_future_after_t0": int(len(future)),

        # -------------------------------------------------------------
        # Monetary observations at/before t0
        # IMPORTANT: not blindly summed.
        # -------------------------------------------------------------
        "po_value_initial_eur": po_value_initial,
        "invoice_value_t0_eur": invoice_value_t0,
        "latest_gr_value_before_t0_eur": latest_gr_value_before_t0,
        "inv_po_abs_rel_diff": inv_po_abs_rel_diff,
        "inv_gr_abs_rel_diff": inv_gr_abs_rel_diff,

        # -------------------------------------------------------------
        # Evidence availability at t0
        # -------------------------------------------------------------
        "gr_available_at_t0": has_event(past, ACT_GOODS_RECEIPT),
        "vendor_invoice_seen_at_t0": has_event(past, ACT_VENDOR_INVOICE),
        "po_created_at_t0": has_event(past, ACT_CREATE_PO),

        # -------------------------------------------------------------
        # Pre-decision process history
        # -------------------------------------------------------------
        "n_goods_receipts_before_t0": event_count(past, ACT_GOODS_RECEIPT),
        "n_cancel_gr_before_t0": event_count(past, ACT_CANCEL_GR),
        "n_invoice_receipts_at_t0_history": event_count(past, ACT_INVOICE_RECEIPT),
        "n_cancel_invoice_before_t0": event_count(past, ACT_CANCEL_INVOICE),
        "n_price_change_before_t0": event_count(past, ACT_PRICE_CHANGE),
        "n_quantity_change_before_t0": event_count(past, ACT_QTY_CHANGE),
        "n_set_payment_block_before_t0": event_count(past, ACT_SET_PAYMENT_BLOCK),
        "n_remove_payment_block_before_t0": event_count(past, ACT_REMOVE_PAYMENT_BLOCK),
        "n_service_entry_before_t0": event_count(past, ACT_SERVICE_ENTRY),
        "n_delete_po_before_t0": event_count(past, ACT_DELETE_PO),
        "n_block_po_before_t0": event_count(past, ACT_BLOCK_PO),
        "n_reactivate_po_before_t0": event_count(past, ACT_REACTIVATE_PO),
        "n_change_approval_before_t0": event_count(past, ACT_CHANGE_APPROVAL),
        "n_release_po_before_t0": event_count(past, ACT_RELEASE_PO),

        # Convenient boolean history indicators
        "had_gr_cancellation_before_t0": has_event(past, ACT_CANCEL_GR),
        "had_invoice_cancellation_before_t0": has_event(past, ACT_CANCEL_INVOICE),
        "had_price_change_before_t0": has_event(past, ACT_PRICE_CHANGE),
        "had_quantity_change_before_t0": has_event(past, ACT_QTY_CHANGE),
        "had_payment_block_before_t0": (
            has_event(past, ACT_SET_PAYMENT_BLOCK)
            or has_event(past, ACT_REMOVE_PAYMENT_BLOCK)
        ),

        # -------------------------------------------------------------
        # Timing features available at t0
        # -------------------------------------------------------------
        "po_to_invoice_days": safe_days(t0 - po_time) if pd.notna(po_time) else np.nan,
        "vendor_invoice_to_recorded_invoice_days": (
            safe_days(t0 - vendor_invoice_time)
            if pd.notna(vendor_invoice_time)
            else np.nan
        ),
        "last_gr_to_invoice_days": (
            safe_days(t0 - last_gr_time)
            if pd.notna(last_gr_time)
            else np.nan
        ),

        # -------------------------------------------------------------
        # FUTURE OUTCOMES -- never expose these as model inputs
        # -------------------------------------------------------------
        "outcome_eventually_cleared": has_event(future, ACT_CLEAR_INVOICE),
        "outcome_invoice_cancelled_after_t0": has_event(future, ACT_CANCEL_INVOICE),
        "outcome_payment_block_removed_after_t0": has_event(future, ACT_REMOVE_PAYMENT_BLOCK),
        "outcome_payment_block_set_after_t0": has_event(future, ACT_SET_PAYMENT_BLOCK),
        "outcome_future_gr": has_event(future, ACT_GOODS_RECEIPT),
        "outcome_future_gr_cancel": has_event(future, ACT_CANCEL_GR),
        "outcome_future_price_change": has_event(future, ACT_PRICE_CHANGE),
        "outcome_future_quantity_change": has_event(future, ACT_QTY_CHANGE),
        "outcome_subsequent_invoice": has_event(future, ACT_SUBSEQUENT_INVOICE),
        "outcome_additional_invoice_receipt": has_event(future, ACT_INVOICE_RECEIPT),
        "outcome_future_service_entry": has_event(future, ACT_SERVICE_ENTRY),
        "time_to_clear_days": (
            safe_days(clear_time - t0)
            if pd.notna(clear_time)
            else np.nan
        ),
        "time_to_first_future_gr_days": (
            safe_days(first_future_gr_time - t0)
            if pd.notna(first_future_gr_time)
            else np.nan
        ),
    }

    # Conservative descriptive future-exception composite.
    # This is NOT yet the RCSE "unsafe" ground-truth label.
    row["outcome_any_post_t0_exception"] = bool(
        row["outcome_invoice_cancelled_after_t0"]
        or row["outcome_payment_block_removed_after_t0"]
        or row["outcome_payment_block_set_after_t0"]
        or row["outcome_future_gr_cancel"]
        or row["outcome_future_price_change"]
        or row["outcome_future_quantity_change"]
        or row["outcome_subsequent_invoice"]
    )

    return row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    xes_path = Path(args.xes).expanduser().resolve()
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file not found: {xes_path}")

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else xes_path.parent / "rcse_output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE BENCHMARK BUILDER - BPI Challenge 2019")
    print("=" * 80)
    print(f"Input : {xes_path}")
    print(f"Output: {out_dir}")
    print()

    print("1/6 Reading XES...")
    df = pm4py.read_xes(str(xes_path))
    if not isinstance(df, pd.DataFrame):
        df = pm4py.convert_to_dataframe(df)

    required = [
        CASE_COL,
        ACTIVITY_COL,
        TIME_COL,
        VALUE_COL,
        ITEM_CATEGORY_COL,
        GR_BASED_IV_COL,
        GOODS_RECEIPT_FLAG_COL,
    ]
    require_columns(df, required)

    # Preserve original XES order to resolve same-timestamp event sequencing.
    df = df.reset_index(drop=True)
    df["_event_order"] = np.arange(len(df), dtype=np.int64)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")

    if df[TIME_COL].isna().any():
        print(
            f"WARNING: {int(df[TIME_COL].isna().sum()):,} events have invalid/missing timestamps."
        )

    total_events = len(df)
    total_cases = df[CASE_COL].nunique(dropna=True)

    print(f"   Events: {total_events:,}")
    print(f"   Cases : {total_cases:,}")

    print("\n2/6 Filtering to 3-way matching cases...")
    target_mask = df[ITEM_CATEGORY_COL].isin(THREE_WAY_CATEGORIES)
    three_way_df = df.loc[target_mask].copy()

    three_way_cases = three_way_df[CASE_COL].nunique(dropna=True)
    print(f"   3-way cases: {three_way_cases:,}")

    # Cases with a recorded invoice receipt -- required to define t0.
    cases_with_invoice = set(
        three_way_df.loc[
            three_way_df[ACTIVITY_COL] == ACT_INVOICE_RECEIPT,
            CASE_COL,
        ].dropna().unique()
    )
    eligible_df = three_way_df[
        three_way_df[CASE_COL].isin(cases_with_invoice)
    ].copy()

    eligible_cases = eligible_df[CASE_COL].nunique(dropna=True)
    print(f"   3-way cases with Record Invoice Receipt: {eligible_cases:,}")

    # Filtering report
    filter_report = pd.DataFrame(
        [
            {
                "stage": "All BPIC 2019 cases",
                "cases": total_cases,
                "events": total_events,
            },
            {
                "stage": "3-way matching cases",
                "cases": three_way_cases,
                "events": len(three_way_df),
            },
            {
                "stage": "3-way cases with Record Invoice Receipt",
                "cases": eligible_cases,
                "events": len(eligible_df),
            },
        ]
    )
    filter_report["case_retention_vs_all_pct"] = (
        filter_report["cases"] / total_cases * 100
    ).round(4)
    filter_report.to_csv(out_dir / "rcse_filter_report.csv", index=False)

    item_counts = (
        df[[CASE_COL, ITEM_CATEGORY_COL]]
        .drop_duplicates(CASE_COL)
        .groupby(ITEM_CATEGORY_COL, dropna=False)[CASE_COL]
        .nunique()
        .reset_index(name="cases")
        .sort_values("cases", ascending=False)
    )
    item_counts.to_csv(out_dir / "rcse_item_category_counts.csv", index=False)

    print("\n3/6 Building one RCSE row per eligible case...")
    rows: list[dict[str, Any]] = []

    grouped = eligible_df.groupby(CASE_COL, sort=False)
    n_groups = grouped.ngroups

    for i, (_, g) in enumerate(grouped, start=1):
        row = build_case_row(g)
        if row is not None:
            rows.append(row)

        if i % 10000 == 0 or i == n_groups:
            print(f"   Processed {i:,}/{n_groups:,} cases")

    base = pd.DataFrame(rows)
    print(f"   RCSE rows created: {len(base):,}")

    if base.empty:
        raise RuntimeError("No benchmark rows were created.")

    # Ensure temporal types
    for c in [
        "decision_time",
        "case_start_time",
        "case_end_time",
    ]:
        if c in base.columns:
            base[c] = pd.to_datetime(base[c], utc=True, errors="coerce")

    # Transaction-value percentile / risk stratum.
    # Use absolute initial PO value as the exposure proxy for v1.
    exposure = base["po_value_initial_eur"].abs()
    base["transaction_exposure_eur"] = exposure

    valid_exposure = exposure.dropna()
    if not valid_exposure.empty:
        p50 = valid_exposure.quantile(0.50)
        p90 = valid_exposure.quantile(0.90)
        p99 = valid_exposure.quantile(0.99)

        def stratum(x: float) -> str | float:
            if pd.isna(x):
                return np.nan
            if x <= p50:
                return "R1_0_50"
            if x <= p90:
                return "R2_50_90"
            if x <= p99:
                return "R3_90_99"
            return "R4_99_100"

        base["transaction_risk_stratum"] = exposure.map(stratum)
    else:
        p50 = p90 = p99 = np.nan
        base["transaction_risk_stratum"] = np.nan

    # Temporal helper columns for later split construction.
    base["decision_year"] = base["decision_time"].dt.year
    base["decision_month"] = base["decision_time"].dt.to_period("M").astype(str)

    # Stable sort for reproducibility
    base = base.sort_values(
        ["decision_time", "case_id"],
        kind="stable",
    ).reset_index(drop=True)

    print("\n4/6 Writing benchmark...")
    parquet_path = out_dir / "rcse_base.parquet"
    try:
        base.to_parquet(parquet_path, index=False)
        print(f"   Wrote {parquet_path.name}")
    except ImportError as exc:
        raise ImportError(
            "Parquet support requires pyarrow. Run: pip install pyarrow"
        ) from exc

    if not args.no_csv:
        csv_path = out_dir / "rcse_base.csv"
        base.to_csv(csv_path, index=False)
        print(f"   Wrote {csv_path.name}")

    print("\n5/6 Writing data-quality summaries...")

    # Missingness
    missingness = pd.DataFrame(
        {
            "column": base.columns,
            "dtype": [str(base[c].dtype) for c in base.columns],
            "non_null": [int(base[c].notna().sum()) for c in base.columns],
            "missing": [int(base[c].isna().sum()) for c in base.columns],
            "missing_pct": [
                round(float(base[c].isna().mean() * 100), 4)
                for c in base.columns
            ],
            "unique_values": [
                int(base[c].nunique(dropna=True))
                for c in base.columns
            ],
        }
    )
    missingness.to_csv(out_dir / "rcse_missingness.csv", index=False)

    # Numeric summaries
    numeric_cols = base.select_dtypes(include=[np.number]).columns
    numeric_summary = (
        base[numeric_cols]
        .describe(
            percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        )
        .T
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    numeric_summary.to_csv(out_dir / "rcse_numeric_summary.csv", index=False)

    # Outcome summary
    outcome_cols = [c for c in base.columns if c.startswith("outcome_")]
    outcome_rows = []
    for c in outcome_cols:
        s = base[c]
        if pd.api.types.is_bool_dtype(s) or set(s.dropna().unique()).issubset({True, False}):
            outcome_rows.append(
                {
                    "outcome": c,
                    "true_count": int((s == True).sum()),  # noqa: E712
                    "true_pct": round(float((s == True).mean() * 100), 4),  # noqa: E712
                }
            )
    pd.DataFrame(outcome_rows).to_csv(
        out_dir / "rcse_outcome_summary.csv",
        index=False,
    )

    # Metadata / reproducibility
    metadata = {
        "input_file": str(xes_path),
        "total_events_input": int(total_events),
        "total_cases_input": int(total_cases),
        "three_way_cases": int(three_way_cases),
        "eligible_cases_with_record_invoice_receipt": int(eligible_cases),
        "rcse_rows": int(len(base)),
        "decision_point": ACT_INVOICE_RECEIPT,
        "three_way_categories": sorted(THREE_WAY_CATEGORIES),
        "transaction_exposure_proxy": "abs(po_value_initial_eur)",
        "risk_percentiles": {
            "p50": None if pd.isna(p50) else float(p50),
            "p90": None if pd.isna(p90) else float(p90),
            "p99": None if pd.isna(p99) else float(p99),
        },
        "temporal_leakage_rule": (
            "Features use events before or including the first Record Invoice Receipt "
            "according to timestamp and original XES event order. Outcomes use later events."
        ),
        "important_note": (
            "'Cumulative net worth (EUR)' is captured at specific events and is not "
            "blindly summed across event rows."
        ),
    }

    with open(
        out_dir / "rcse_build_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n6/6 Validation checks...")

    # Basic assertions
    assert base["case_id"].is_unique, "Expected exactly one row per case."
    assert base["decision_time"].notna().all(), "Decision time should never be missing."
    assert set(base["item_category"].dropna().unique()).issubset(
        THREE_WAY_CATEGORIES
    ), "Unexpected item category after filter."

    # The first invoice receipt itself must be included in the history.
    assert (
        base["n_invoice_receipts_at_t0_history"] >= 1
    ).all(), "Every row must contain its t0 invoice receipt."

    print("   PASS: one row per case")
    print("   PASS: decision time present")
    print("   PASS: only target 3-way categories")
    print("   PASS: t0 invoice included in pre-decision history")

    print("\n" + "=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(base):,}")
    print()
    print("Item-category distribution:")
    print(base["item_category"].value_counts(dropna=False).to_string())
    print()
    print("Risk-stratum distribution:")
    print(base["transaction_risk_stratum"].value_counts(dropna=False).to_string())
    print()
    print("Key outputs:")
    for name in [
        "rcse_base.parquet",
        "rcse_base.csv" if not args.no_csv else None,
        "rcse_filter_report.csv",
        "rcse_item_category_counts.csv",
        "rcse_outcome_summary.csv",
        "rcse_numeric_summary.csv",
        "rcse_missingness.csv",
        "rcse_build_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
