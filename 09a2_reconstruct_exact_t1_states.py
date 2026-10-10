"""
09a2_reconstruct_exact_t1_states.py

Reconstruct exact post-GATHER t1 states from the raw BPI Challenge 2019 XES.

Purpose
-------
For the natural delayed-GR population identified in Step 09a.1, replay the
original event sequence and reconstruct the complete process state at:

    t0 = first "Record Invoice Receipt"
    t1 = first "Record Goods Receipt" strictly after t0 in exact XES order

This preserves same-timestamp event ordering using original XES event order.

IMPORTANT
---------
This script does NOT create a post-GATHER action target Y_t1 and does NOT train
a t1 model. It creates an exact transition benchmark so that candidate t1
targets can be defined and validated separately.

Inputs
------
--xes
    Raw BPI Challenge 2019 XES file.

--candidates
    gather_transition_candidates.parquet from Step 09a.1.

Outputs
-------
t1_transition_exact.parquet
t1_transition_exact.csv
t1_transition_summary.csv
t1_transition_event_counts.csv
t1_transition_validation.csv
t1_transition_outcomes_after_t1.csv
t1_transition_metadata.json
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
# BPI Challenge 2019 schema / activities
# ---------------------------------------------------------------------

CASE_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIME_COL = "time:timestamp"
VALUE_COL = "Cumulative net worth (EUR)"

ITEM_CATEGORY_COL = "case:Item Category"
GR_BASED_IV_COL = "case:GR-Based Inv. Verif."
GOODS_RECEIPT_FLAG_COL = "case:Goods Receipt"

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09a.2: reconstruct exact t1 states from raw XES."
    )
    p.add_argument(
        "--xes",
        required=True,
        help="Path to BPI Challenge 2019 XES file.",
    )
    p.add_argument(
        "--candidates",
        required=True,
        help="Path to gather_transition_candidates.parquet.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <candidates parent>/t1_exact_reconstruction",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip full transition CSV; Parquet is always written.",
    )
    return p.parse_args()


def normalize_bool(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    x = str(value).strip().lower()
    if x in {"true", "1", "yes", "y"}:
        return True
    if x in {"false", "0", "no", "n"}:
        return False
    return value


def first_non_null(s: pd.Series) -> Any:
    s = s.dropna()
    return s.iloc[0] if not s.empty else np.nan


def case_static_value(g: pd.DataFrame, col: str) -> Any:
    return first_non_null(g[col]) if col in g.columns else np.nan


def event_count(df: pd.DataFrame, activity: str) -> int:
    return int((df[ACTIVITY_COL] == activity).sum())


def has_event(df: pd.DataFrame, activity: str) -> bool:
    return bool((df[ACTIVITY_COL] == activity).any())


def event_time_first(df: pd.DataFrame, activity: str) -> pd.Timestamp | pd.NaT:
    s = df.loc[df[ACTIVITY_COL] == activity, TIME_COL].dropna()
    return s.iloc[0] if not s.empty else pd.NaT


def event_time_last(df: pd.DataFrame, activity: str) -> pd.Timestamp | pd.NaT:
    s = df.loc[df[ACTIVITY_COL] == activity, TIME_COL].dropna()
    return s.iloc[-1] if not s.empty else pd.NaT


def event_value_first(df: pd.DataFrame, activity: str) -> float:
    s = df.loc[df[ACTIVITY_COL] == activity, VALUE_COL].dropna()
    return float(s.iloc[0]) if not s.empty else np.nan


def event_value_last(df: pd.DataFrame, activity: str) -> float:
    s = df.loc[df[ACTIVITY_COL] == activity, VALUE_COL].dropna()
    return float(s.iloc[-1]) if not s.empty else np.nan


def safe_days(delta: Any) -> float:
    if pd.isna(delta):
        return np.nan
    return float(delta.total_seconds() / 86400.0)


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def split_at_event(
    g: pd.DataFrame,
    timestamp: pd.Timestamp,
    event_order: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inclusive state through event, plus strictly later events."""
    through = (
        (g[TIME_COL] < timestamp)
        | (
            (g[TIME_COL] == timestamp)
            & (g["_event_order"] <= event_order)
        )
    )
    after = (
        (g[TIME_COL] > timestamp)
        | (
            (g[TIME_COL] == timestamp)
            & (g["_event_order"] > event_order)
        )
    )
    return g.loc[through].copy(), g.loc[after].copy()


def build_exact_transition(
    g: pd.DataFrame,
    candidate_t1_time: pd.Timestamp | pd.NaT,
) -> dict[str, Any] | None:
    g = g.sort_values(
        [TIME_COL, "_event_order"],
        kind="stable",
    ).copy()

    invoice_rows = g[g[ACTIVITY_COL] == ACT_INVOICE_RECEIPT]
    if invoice_rows.empty:
        return None

    # Exact t0 = first invoice receipt.
    t0_row = invoice_rows.iloc[0]
    t0 = t0_row[TIME_COL]
    t0_order = int(t0_row["_event_order"])

    at_t0, after_t0 = split_at_event(g, t0, t0_order)

    # Exact t1 = first GR strictly after t0 in timestamp + original order.
    future_gr = after_t0[
        after_t0[ACTIVITY_COL] == ACT_GOODS_RECEIPT
    ].sort_values(
        [TIME_COL, "_event_order"],
        kind="stable",
    )

    if future_gr.empty:
        return None

    t1_row = future_gr.iloc[0]
    t1 = t1_row[TIME_COL]
    t1_order = int(t1_row["_event_order"])

    at_t1, after_t1 = split_at_event(g, t1, t1_order)

    # Events between t0 and t1, including t1 GR but excluding t0.
    between_mask = (
        (
            (g[TIME_COL] > t0)
            | (
                (g[TIME_COL] == t0)
                & (g["_event_order"] > t0_order)
            )
        )
        & (
            (g[TIME_COL] < t1)
            | (
                (g[TIME_COL] == t1)
                & (g["_event_order"] <= t1_order)
            )
        )
    )
    between = g.loc[between_mask].copy()

    po_time_t1 = event_time_first(at_t1, ACT_CREATE_PO)
    vendor_invoice_time_t1 = event_time_first(
        at_t1,
        ACT_VENDOR_INVOICE,
    )
    last_gr_time_t1 = event_time_last(
        at_t1,
        ACT_GOODS_RECEIPT,
    )

    invoice_value_t0 = (
        float(t0_row[VALUE_COL])
        if pd.notna(t0_row[VALUE_COL])
        else np.nan
    )

    first_gr_value_t1 = (
        float(t1_row[VALUE_COL])
        if pd.notna(t1_row[VALUE_COL])
        else np.nan
    )

    row = {
        # Identity / static context
        "case_id": case_static_value(g, CASE_COL),
        "purchasing_document": case_static_value(
            g, "case:Purchasing Document"
        ),
        "item_id": case_static_value(g, "case:Item"),
        "vendor_id": case_static_value(g, "case:Vendor"),
        "vendor_name_anon": case_static_value(g, "case:Name"),
        "company": case_static_value(g, "case:Company"),
        "source_system": case_static_value(g, "case:Source"),
        "document_type": case_static_value(
            g, "case:Document Type"
        ),
        "document_category": case_static_value(
            g, "case:Purch. Doc. Category name"
        ),
        "item_type": case_static_value(g, "case:Item Type"),
        "item_category": case_static_value(
            g, ITEM_CATEGORY_COL
        ),
        "spend_classification": case_static_value(
            g, "case:Spend classification text"
        ),
        "spend_area": case_static_value(
            g, "case:Spend area text"
        ),
        "sub_spend_area": case_static_value(
            g, "case:Sub spend area text"
        ),
        "gr_based_invoice_verification": normalize_bool(
            case_static_value(g, GR_BASED_IV_COL)
        ),
        "goods_receipt_required": normalize_bool(
            case_static_value(g, GOODS_RECEIPT_FLAG_COL)
        ),

        # Exact boundaries
        "t0_time": t0,
        "t0_event_order": t0_order,
        "t1_time": t1,
        "t1_event_order": t1_order,
        "candidate_t1_time": candidate_t1_time,
        "candidate_t1_time_matches_exact": (
            bool(t1 == candidate_t1_time)
            if pd.notna(candidate_t1_time)
            else False
        ),
        "t1_same_timestamp_as_t0": bool(t1 == t0),
        "events_total_case": int(len(g)),
        "events_available_at_t0_exact": int(len(at_t0)),
        "events_between_t0_t1_inclusive_t1": int(len(between)),
        "events_available_at_t1_exact": int(len(at_t1)),
        "events_after_t1": int(len(after_t1)),
        "elapsed_t0_to_t1_days": safe_days(t1 - t0),

        # Monetary state
        "po_value_initial_eur": event_value_first(
            at_t1,
            ACT_CREATE_PO,
        ),
        "invoice_value_t0_eur": invoice_value_t0,
        "latest_gr_value_at_t1_eur": event_value_last(
            at_t1,
            ACT_GOODS_RECEIPT,
        ),
        "first_gr_value_at_t1_eur": first_gr_value_t1,

        # Availability at exact t1
        "gr_available_at_t1": has_event(
            at_t1,
            ACT_GOODS_RECEIPT,
        ),
        "vendor_invoice_seen_at_t1": has_event(
            at_t1,
            ACT_VENDOR_INVOICE,
        ),
        "po_created_at_t1": has_event(
            at_t1,
            ACT_CREATE_PO,
        ),

        # Exact process-history counts through t1
        "n_goods_receipts_by_t1": event_count(
            at_t1, ACT_GOODS_RECEIPT
        ),
        "n_cancel_gr_by_t1": event_count(
            at_t1, ACT_CANCEL_GR
        ),
        "n_invoice_receipts_by_t1": event_count(
            at_t1, ACT_INVOICE_RECEIPT
        ),
        "n_cancel_invoice_by_t1": event_count(
            at_t1, ACT_CANCEL_INVOICE
        ),
        "n_price_change_by_t1": event_count(
            at_t1, ACT_PRICE_CHANGE
        ),
        "n_quantity_change_by_t1": event_count(
            at_t1, ACT_QTY_CHANGE
        ),
        "n_set_payment_block_by_t1": event_count(
            at_t1, ACT_SET_PAYMENT_BLOCK
        ),
        "n_remove_payment_block_by_t1": event_count(
            at_t1, ACT_REMOVE_PAYMENT_BLOCK
        ),
        "n_service_entry_by_t1": event_count(
            at_t1, ACT_SERVICE_ENTRY
        ),
        "n_delete_po_by_t1": event_count(
            at_t1, ACT_DELETE_PO
        ),
        "n_block_po_by_t1": event_count(
            at_t1, ACT_BLOCK_PO
        ),
        "n_reactivate_po_by_t1": event_count(
            at_t1, ACT_REACTIVATE_PO
        ),
        "n_change_approval_by_t1": event_count(
            at_t1, ACT_CHANGE_APPROVAL
        ),
        "n_release_po_by_t1": event_count(
            at_t1, ACT_RELEASE_PO
        ),

        # Exact booleans through t1
        "had_gr_cancellation_by_t1": has_event(
            at_t1, ACT_CANCEL_GR
        ),
        "had_invoice_cancellation_by_t1": has_event(
            at_t1, ACT_CANCEL_INVOICE
        ),
        "had_price_change_by_t1": has_event(
            at_t1, ACT_PRICE_CHANGE
        ),
        "had_quantity_change_by_t1": has_event(
            at_t1, ACT_QTY_CHANGE
        ),
        "had_payment_block_by_t1": (
            has_event(at_t1, ACT_SET_PAYMENT_BLOCK)
            or has_event(at_t1, ACT_REMOVE_PAYMENT_BLOCK)
        ),

        # Timing features at t1
        "po_to_t1_days": (
            safe_days(t1 - po_time_t1)
            if pd.notna(po_time_t1)
            else np.nan
        ),
        "vendor_invoice_to_t1_days": (
            safe_days(t1 - vendor_invoice_time_t1)
            if pd.notna(vendor_invoice_time_t1)
            else np.nan
        ),
        "last_gr_to_t1_days": (
            safe_days(t1 - last_gr_time_t1)
            if pd.notna(last_gr_time_t1)
            else np.nan
        ),

        # Exact activity counts BETWEEN t0 and t1
        "between_n_goods_receipts": event_count(
            between, ACT_GOODS_RECEIPT
        ),
        "between_n_cancel_gr": event_count(
            between, ACT_CANCEL_GR
        ),
        "between_n_invoice_receipts": event_count(
            between, ACT_INVOICE_RECEIPT
        ),
        "between_n_cancel_invoice": event_count(
            between, ACT_CANCEL_INVOICE
        ),
        "between_n_price_change": event_count(
            between, ACT_PRICE_CHANGE
        ),
        "between_n_quantity_change": event_count(
            between, ACT_QTY_CHANGE
        ),
        "between_n_set_payment_block": event_count(
            between, ACT_SET_PAYMENT_BLOCK
        ),
        "between_n_remove_payment_block": event_count(
            between, ACT_REMOVE_PAYMENT_BLOCK
        ),
        "between_n_service_entry": event_count(
            between, ACT_SERVICE_ENTRY
        ),
        "between_n_delete_po": event_count(
            between, ACT_DELETE_PO
        ),
        "between_n_block_po": event_count(
            between, ACT_BLOCK_PO
        ),
        "between_n_reactivate_po": event_count(
            between, ACT_REACTIVATE_PO
        ),
        "between_n_change_approval": event_count(
            between, ACT_CHANGE_APPROVAL
        ),
        "between_n_release_po": event_count(
            between, ACT_RELEASE_PO
        ),

        # Post-t1 observations for later TARGET VALIDATION ONLY.
        # Never use these as t1 model features.
        "after_t1_eventually_cleared": has_event(
            after_t1, ACT_CLEAR_INVOICE
        ),
        "after_t1_invoice_cancelled": has_event(
            after_t1, ACT_CANCEL_INVOICE
        ),
        "after_t1_payment_block_removed": has_event(
            after_t1, ACT_REMOVE_PAYMENT_BLOCK
        ),
        "after_t1_payment_block_set": has_event(
            after_t1, ACT_SET_PAYMENT_BLOCK
        ),
        "after_t1_future_gr": has_event(
            after_t1, ACT_GOODS_RECEIPT
        ),
        "after_t1_gr_cancel": has_event(
            after_t1, ACT_CANCEL_GR
        ),
        "after_t1_price_change": has_event(
            after_t1, ACT_PRICE_CHANGE
        ),
        "after_t1_quantity_change": has_event(
            after_t1, ACT_QTY_CHANGE
        ),
        "after_t1_subsequent_invoice": has_event(
            after_t1, ACT_SUBSEQUENT_INVOICE
        ),
        "after_t1_additional_invoice": has_event(
            after_t1, ACT_INVOICE_RECEIPT
        ),
        "after_t1_service_entry": has_event(
            after_t1, ACT_SERVICE_ENTRY
        ),
        "time_t1_to_clear_days": (
            safe_days(
                event_time_first(after_t1, ACT_CLEAR_INVOICE) - t1
            )
            if pd.notna(
                event_time_first(after_t1, ACT_CLEAR_INVOICE)
            )
            else np.nan
        ),
    }

    row["between_any_non_gr_event"] = bool(
        len(
            between[
                between[ACTIVITY_COL] != ACT_GOODS_RECEIPT
            ]
        )
        > 0
    )

    row["after_t1_any_selected_exception"] = bool(
        row["after_t1_invoice_cancelled"]
        or row["after_t1_payment_block_set"]
        or row["after_t1_gr_cancel"]
        or row["after_t1_price_change"]
        or row["after_t1_quantity_change"]
        or row["after_t1_subsequent_invoice"]
    )

    # Diagnostic mismatch fields only; do not use as validated correctness.
    po_value = row["po_value_initial_eur"]
    inv_value = row["invoice_value_t0_eur"]
    gr_value = row["latest_gr_value_at_t1_eur"]

    row["diag_inv_po_abs_rel_diff_t1"] = (
        abs(inv_value - po_value) / abs(po_value)
        if pd.notna(inv_value)
        and pd.notna(po_value)
        and abs(po_value) > 1e-12
        else np.nan
    )

    row["diag_inv_gr_abs_rel_diff_t1"] = (
        abs(inv_value - gr_value) / abs(gr_value)
        if pd.notna(inv_value)
        and pd.notna(gr_value)
        and abs(gr_value) > 1e-12
        else np.nan
    )

    return row


def main() -> None:
    args = parse_args()

    xes_path = Path(args.xes).expanduser().resolve()
    cand_path = Path(args.candidates).expanduser().resolve()

    if not xes_path.exists():
        raise FileNotFoundError(f"XES not found: {xes_path}")
    if not cand_path.exists():
        raise FileNotFoundError(
            f"Candidate file not found: {cand_path}"
        )

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else cand_path.parent / "t1_exact_reconstruction"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE STEP 09a.2 - EXACT t1 RECONSTRUCTION")
    print("=" * 80)
    print(f"XES       : {xes_path}")
    print(f"Candidates: {cand_path}")
    print(f"Output    : {out_dir}")

    print("\n1/6 Reading candidate population...")
    candidates = pd.read_parquet(cand_path)
    require_columns(
        candidates,
        ["case_id", "t1_time"],
    )
    candidates["case_id"] = candidates["case_id"].astype(str)
    candidates["t1_time"] = pd.to_datetime(
        candidates["t1_time"],
        utc=True,
        errors="coerce",
    )

    candidate_ids = set(candidates["case_id"])
    candidate_t1 = (
        candidates[
            ["case_id", "t1_time"]
        ]
        .drop_duplicates("case_id")
        .set_index("case_id")["t1_time"]
        .to_dict()
    )

    print(f"   Candidate cases: {len(candidate_ids):,}")

    print("\n2/6 Reading raw XES...")
    df = pm4py.read_xes(str(xes_path))
    if not isinstance(df, pd.DataFrame):
        df = pm4py.convert_to_dataframe(df)

    require_columns(
        df,
        [
            CASE_COL,
            ACTIVITY_COL,
            TIME_COL,
            VALUE_COL,
        ],
    )

    # Original import order is the tie-breaker for same timestamps.
    df = df.reset_index(drop=True)
    df["_event_order"] = np.arange(
        len(df),
        dtype=np.int64,
    )
    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        utc=True,
        errors="coerce",
    )
    df[CASE_COL] = df[CASE_COL].astype(str)

    print(f"   Raw events: {len(df):,}")

    print("\n3/6 Filtering to Step-09a.1 candidate cases...")
    work = df[
        df[CASE_COL].isin(candidate_ids)
    ].copy()

    found_ids = set(work[CASE_COL].unique())
    missing_ids = sorted(candidate_ids - found_ids)

    print(f"   Candidate events: {len(work):,}")
    print(f"   Cases found      : {len(found_ids):,}")
    print(f"   Missing cases    : {len(missing_ids):,}")

    print("\n4/6 Reconstructing exact t0 and t1 states...")
    rows: list[dict[str, Any]] = []

    grouped = work.groupby(
        CASE_COL,
        sort=False,
    )
    n_groups = grouped.ngroups

    for i, (case_id, g) in enumerate(
        grouped,
        start=1,
    ):
        row = build_exact_transition(
            g,
            candidate_t1.get(str(case_id), pd.NaT),
        )
        if row is not None:
            rows.append(row)

        if i % 1000 == 0 or i == n_groups:
            print(
                f"   Processed {i:,}/{n_groups:,} cases"
            )

    exact = pd.DataFrame(rows)

    if exact.empty:
        raise RuntimeError(
            "No exact t1 transition rows were reconstructed."
        )

    exact["case_id"] = exact["case_id"].astype(str)

    # Attach exposure / risk stratum from Step 09a.1 candidates.
    attach_cols = [
        c for c in [
            "case_id",
            "transaction_exposure_eur",
            "transaction_risk_stratum",
        ]
        if c in candidates.columns
    ]

    if len(attach_cols) > 1:
        attach = candidates[
            attach_cols
        ].drop_duplicates("case_id")

        exact = exact.merge(
            attach,
            on="case_id",
            how="left",
            validate="one_to_one",
        )

    exact = exact.sort_values(
        ["t0_time", "case_id"],
        kind="stable",
    ).reset_index(drop=True)

    print(f"   Exact transition rows: {len(exact):,}")

    print("\n5/6 Validating reconstruction...")

    validation_rows = []

    validation_rows.append(
        {
            "check": "candidate_cases",
            "value": len(candidate_ids),
        }
    )
    validation_rows.append(
        {
            "check": "cases_found_in_xes",
            "value": len(found_ids),
        }
    )
    validation_rows.append(
        {
            "check": "exact_transition_rows",
            "value": len(exact),
        }
    )
    validation_rows.append(
        {
            "check": "candidate_t1_time_match",
            "value": int(
                exact[
                    "candidate_t1_time_matches_exact"
                ].sum()
            ),
        }
    )
    validation_rows.append(
        {
            "check": "candidate_t1_time_mismatch",
            "value": int(
                (
                    ~exact[
                        "candidate_t1_time_matches_exact"
                    ]
                ).sum()
            ),
        }
    )
    validation_rows.append(
        {
            "check": "same_timestamp_t0_t1",
            "value": int(
                exact[
                    "t1_same_timestamp_as_t0"
                ].sum()
            ),
        }
    )
    validation_rows.append(
        {
            "check": "gr_available_at_t1_false",
            "value": int(
                (
                    ~exact["gr_available_at_t1"]
                ).sum()
            ),
        }
    )

    validation_df = pd.DataFrame(
        validation_rows
    )
    validation_df.to_csv(
        out_dir / "t1_transition_validation.csv",
        index=False,
    )

    # Event-count profile between t0 and t1.
    between_cols = [
        c for c in exact.columns
        if c.startswith("between_n_")
    ]

    event_count_rows = []
    for c in between_cols:
        s = exact[c]
        event_count_rows.append(
            {
                "variable": c,
                "cases_with_event": int((s > 0).sum()),
                "pct_cases_with_event": round(
                    float((s > 0).mean() * 100),
                    4,
                ),
                "total_events": int(s.sum()),
                "mean_events": float(s.mean()),
                "max_events": int(s.max()),
            }
        )

    pd.DataFrame(
        event_count_rows
    ).to_csv(
        out_dir / "t1_transition_event_counts.csv",
        index=False,
    )

    # Post-t1 descriptive outcomes for later target design.
    outcome_cols = [
        c for c in exact.columns
        if c.startswith("after_t1_")
        and exact[c].dtype == bool
    ]

    outcome_rows = []
    for c in outcome_cols:
        outcome_rows.append(
            {
                "outcome": c,
                "count": int(exact[c].sum()),
                "pct": round(
                    float(exact[c].mean() * 100),
                    4,
                ),
            }
        )

    pd.DataFrame(
        outcome_rows
    ).to_csv(
        out_dir / "t1_transition_outcomes_after_t1.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "candidate_cases",
                "value": len(candidate_ids),
            },
            {
                "metric": "exact_transition_rows",
                "value": len(exact),
            },
            {
                "metric": "t1_time_exact_match",
                "value": int(
                    exact[
                        "candidate_t1_time_matches_exact"
                    ].sum()
                ),
            },
            {
                "metric": "same_timestamp_t0_t1",
                "value": int(
                    exact[
                        "t1_same_timestamp_as_t0"
                    ].sum()
                ),
            },
            {
                "metric": "cases_with_non_gr_event_between_t0_t1",
                "value": int(
                    exact[
                        "between_any_non_gr_event"
                    ].sum()
                ),
            },
            {
                "metric": "median_wait_days",
                "value": float(
                    exact[
                        "elapsed_t0_to_t1_days"
                    ].median()
                ),
            },
            {
                "metric": "p90_wait_days",
                "value": float(
                    exact[
                        "elapsed_t0_to_t1_days"
                    ].quantile(0.90)
                ),
            },
        ]
    )

    summary.to_csv(
        out_dir / "t1_transition_summary.csv",
        index=False,
    )

    print("\n6/6 Writing exact transition benchmark...")

    exact.to_parquet(
        out_dir / "t1_transition_exact.parquet",
        index=False,
    )

    if not args.no_csv:
        exact.to_csv(
            out_dir / "t1_transition_exact.csv",
            index=False,
        )

    metadata = {
        "step": "09a.2",
        "xes_file": str(xes_path),
        "candidate_file": str(cand_path),
        "candidate_cases": int(len(candidate_ids)),
        "reconstructed_cases": int(len(exact)),
        "t0_definition": (
            "first Record Invoice Receipt in timestamp + original XES order"
        ),
        "t1_definition": (
            "first Record Goods Receipt strictly after t0 in timestamp + "
            "original XES event order"
        ),
        "same_timestamp_order_preserved": True,
        "post_t1_outcomes_are_features": False,
        "post_t1_outcomes_purpose": (
            "target-design diagnostics only; never expose them to a t1 model"
        ),
        "post_gather_target_created": False,
        "next_methodological_step": (
            "Validate candidate Y_t1 definitions using exact t1 state and "
            "post-t1 outcomes before training any post-GATHER estimator."
        ),
    }

    with open(
        out_dir / "t1_transition_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 09a.2 COMPLETE")
    print("=" * 80)
    print(
        f"Candidate cases       : {len(candidate_ids):,}"
    )
    print(
        f"Reconstructed cases   : {len(exact):,}"
    )
    print(
        "Exact t1 time matches : "
        f"{int(exact['candidate_t1_time_matches_exact'].sum()):,}"
    )
    print(
        "Same timestamp t0/t1  : "
        f"{int(exact['t1_same_timestamp_as_t0'].sum()):,}"
    )
    print(
        "Non-GR event between  : "
        f"{int(exact['between_any_non_gr_event'].sum()):,}"
    )

    print(
        "\nNo Y_t1 label has been created. "
        "The next step is target-definition validation."
    )

    print("\nOutputs:")
    for name in [
        "t1_transition_exact.parquet",
        None if args.no_csv else "t1_transition_exact.csv",
        "t1_transition_summary.csv",
        "t1_transition_event_counts.csv",
        "t1_transition_validation.csv",
        "t1_transition_outcomes_after_t1.csv",
        "t1_transition_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
