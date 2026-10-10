"""
09a1_diagnose_gather_transition.py

Diagnostic for the natural GATHER transition in the RCSE benchmark.

Purpose
-------
Before training any post-GATHER estimator, determine whether the v2 benchmark
contains enough information to construct a defensible state transition:

    O_t0 -> GATHER -> O_t1

where t1 is the first future goods-receipt event after the original invoice
decision point t0.

Input
-----
rcse_base_v2.parquet

Outputs
-------
gather_transition_summary.csv
gather_transition_wait_time_summary.csv
gather_transition_by_item_category.csv
gather_transition_by_risk_stratum.csv
gather_transition_event_timing.csv
gather_transition_feature_reconstructability.csv
gather_transition_candidates.parquet
gather_transition_candidates.csv
gather_transition_metadata.json

Important methodological rule
-----------------------------
This script is DIAGNOSTIC ONLY. It does not train a post-GATHER estimator and
does not create post-GATHER action labels.

The v2 table contains selected post-t0 timestamps and future-GR observations,
but it does not necessarily contain every event/state update needed to recreate
the complete feature vector at t1. This script makes that limitation explicit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09a.1: diagnose natural GATHER transitions."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to rcse_base_v2.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <input parent>/gather_transition_diagnostic",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip full candidate CSV; Parquet is always written.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def pct(count: int, total: int) -> float:
    return round((count / total * 100.0), 4) if total else np.nan


def percentile_summary(series: pd.Series, metric: str) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "metric": metric,
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "metric": metric,
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "gather_transition_diagnostic"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE STEP 09a.1 - GATHER TRANSITION DIAGNOSTIC")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")

    df = pd.read_parquet(input_path)

    required = [
        "case_id",
        "item_category",
        "decision_time",
        "gr_available_at_t0",
        "gather_gr_possible",
        "gather_gr_has_value",
        "first_future_gr_time",
        "first_future_gr_value_eur",
        "latest_future_gr_value_eur",
        "n_future_goods_receipts",
        "time_to_first_future_gr_days",
        "n_goods_receipts_before_t0",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
    ]
    require_columns(df, required)

    # Normalize timestamps.
    timestamp_cols = [
        "decision_time",
        "first_future_gr_time",
        "last_future_gr_time",
        "first_clear_time",
        "first_payment_block_removal_time",
        "first_payment_block_set_time",
        "first_invoice_cancel_time",
        "first_additional_invoice_time",
        "first_subsequent_invoice_time",
        "first_future_price_change_time",
        "first_future_quantity_change_time",
    ]
    for c in timestamp_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    n_all = len(df)

    no_gr_t0 = ~df["gr_available_at_t0"].fillna(False).astype(bool)
    future_gr = df["gather_gr_possible"].fillna(False).astype(bool)
    future_gr_value = df["gather_gr_has_value"].fillna(False).astype(bool)
    valid_t1 = df["first_future_gr_time"].notna()

    eligible_mask = (
        no_gr_t0
        & future_gr
        & future_gr_value
        & valid_t1
    )

    candidates = df.loc[eligible_mask].copy()

    if candidates.empty:
        raise RuntimeError(
            "No natural delayed-GR GATHER candidates were found."
        )

    candidates["t1_time"] = candidates["first_future_gr_time"]
    candidates["t1_after_t0"] = (
        candidates["t1_time"] > candidates["decision_time"]
    )
    candidates["t1_same_timestamp_as_t0"] = (
        candidates["t1_time"] == candidates["decision_time"]
    )
    candidates["t1_before_t0_invalid"] = (
        candidates["t1_time"] < candidates["decision_time"]
    )

    # Minimal state variables that are exactly derivable at the first future GR.
    candidates["gr_available_at_t1"] = True
    candidates["latest_gr_value_at_t1_eur"] = (
        candidates["first_future_gr_value_eur"]
    )
    candidates["n_goods_receipts_by_t1"] = (
        candidates["n_goods_receipts_before_t0"] + 1
    )
    candidates["elapsed_t0_to_t1_days"] = (
        (
            candidates["t1_time"]
            - candidates["decision_time"]
        ).dt.total_seconds()
        / 86400.0
    )

    # Selected post-t0 events for which v2 stores FIRST timestamps.
    # These flags are diagnostic. Events exactly at t1 may have unresolved
    # intra-timestamp order because v2 does not preserve original event order.
    event_timestamp_map = {
        "clear": "first_clear_time",
        "payment_block_removal": "first_payment_block_removal_time",
        "payment_block_set": "first_payment_block_set_time",
        "invoice_cancel": "first_invoice_cancel_time",
        "additional_invoice": "first_additional_invoice_time",
        "subsequent_invoice": "first_subsequent_invoice_time",
        "future_price_change": "first_future_price_change_time",
        "future_quantity_change": "first_future_quantity_change_time",
    }

    event_timing_rows = []

    for event_name, col in event_timestamp_map.items():
        if col not in candidates.columns:
            continue

        ts = candidates[col]

        before_t1 = ts.notna() & (ts < candidates["t1_time"])
        same_t1 = ts.notna() & (ts == candidates["t1_time"])
        after_t1 = ts.notna() & (ts > candidates["t1_time"])

        candidates[f"{event_name}_before_t1"] = before_t1
        candidates[f"{event_name}_same_timestamp_t1"] = same_t1

        event_timing_rows.append(
            {
                "event": event_name,
                "timestamp_column": col,
                "observed_anywhere_after_t0": int(ts.notna().sum()),
                "before_t1": int(before_t1.sum()),
                "same_timestamp_as_t1": int(same_t1.sum()),
                "after_t1": int(after_t1.sum()),
                "pct_candidates_before_t1": pct(
                    int(before_t1.sum()),
                    len(candidates),
                ),
                "pct_candidates_same_timestamp_t1": pct(
                    int(same_t1.sum()),
                    len(candidates),
                ),
            }
        )

    # Any selected event strictly before t1 means the state evolved between
    # invoice receipt and first future GR.
    before_cols = [
        c for c in candidates.columns
        if c.endswith("_before_t1")
    ]
    same_cols = [
        c for c in candidates.columns
        if c.endswith("_same_timestamp_t1")
    ]

    candidates["selected_intervening_event_before_t1"] = (
        candidates[before_cols].any(axis=1)
        if before_cols else False
    )
    candidates["selected_same_timestamp_order_ambiguity"] = (
        candidates[same_cols].any(axis=1)
        if same_cols else False
    )

    # ------------------------------------------------------------------
    # Population summary
    # ------------------------------------------------------------------
    summary_rows = [
        {
            "stage": "all_v2_cases",
            "count": n_all,
            "pct_of_all": 100.0,
        },
        {
            "stage": "no_gr_available_at_t0",
            "count": int(no_gr_t0.sum()),
            "pct_of_all": pct(int(no_gr_t0.sum()), n_all),
        },
        {
            "stage": "future_gr_possible",
            "count": int(future_gr.sum()),
            "pct_of_all": pct(int(future_gr.sum()), n_all),
        },
        {
            "stage": "no_gr_at_t0_and_future_gr",
            "count": int((no_gr_t0 & future_gr).sum()),
            "pct_of_all": pct(int((no_gr_t0 & future_gr).sum()), n_all),
        },
        {
            "stage": "eligible_natural_gather_transition",
            "count": int(len(candidates)),
            "pct_of_all": pct(len(candidates), n_all),
        },
        {
            "stage": "eligible_t1_strictly_after_t0",
            "count": int(candidates["t1_after_t0"].sum()),
            "pct_of_all": pct(
                int(candidates["t1_after_t0"].sum()),
                n_all,
            ),
        },
        {
            "stage": "eligible_t1_same_timestamp_as_t0",
            "count": int(candidates["t1_same_timestamp_as_t0"].sum()),
            "pct_of_all": pct(
                int(candidates["t1_same_timestamp_as_t0"].sum()),
                n_all,
            ),
        },
        {
            "stage": "invalid_t1_before_t0",
            "count": int(candidates["t1_before_t0_invalid"].sum()),
            "pct_of_all": pct(
                int(candidates["t1_before_t0_invalid"].sum()),
                n_all,
            ),
        },
        {
            "stage": "selected_intervening_event_before_t1",
            "count": int(
                candidates["selected_intervening_event_before_t1"].sum()
            ),
            "pct_of_all": pct(
                int(
                    candidates[
                        "selected_intervening_event_before_t1"
                    ].sum()
                ),
                n_all,
            ),
        },
        {
            "stage": "selected_same_timestamp_order_ambiguity",
            "count": int(
                candidates[
                    "selected_same_timestamp_order_ambiguity"
                ].sum()
            ),
            "pct_of_all": pct(
                int(
                    candidates[
                        "selected_same_timestamp_order_ambiguity"
                    ].sum()
                ),
                n_all,
            ),
        },
    ]

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "gather_transition_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Wait-time and receipt-count summaries
    # ------------------------------------------------------------------
    wait_summary = pd.DataFrame(
        [
            percentile_summary(
                candidates["elapsed_t0_to_t1_days"],
                "elapsed_t0_to_t1_days",
            ),
            percentile_summary(
                candidates["time_to_first_future_gr_days"],
                "stored_time_to_first_future_gr_days",
            ),
            percentile_summary(
                candidates["n_future_goods_receipts"],
                "n_future_goods_receipts",
            ),
            percentile_summary(
                candidates["transaction_exposure_eur"],
                "transaction_exposure_eur",
            ),
        ]
    )
    wait_summary.to_csv(
        out_dir / "gather_transition_wait_time_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Subgroup summaries
    # ------------------------------------------------------------------
    by_item = (
        candidates.groupby("item_category", dropna=False)
        .agg(
            cases=("case_id", "count"),
            median_wait_days=("elapsed_t0_to_t1_days", "median"),
            p90_wait_days=(
                "elapsed_t0_to_t1_days",
                lambda s: s.quantile(0.90),
            ),
            median_exposure_eur=("transaction_exposure_eur", "median"),
            pct_intervening_event=(
                "selected_intervening_event_before_t1",
                lambda s: float(s.mean() * 100.0),
            ),
        )
        .reset_index()
    )
    by_item["pct_of_candidates"] = (
        by_item["cases"] / len(candidates) * 100.0
    ).round(4)
    by_item.to_csv(
        out_dir / "gather_transition_by_item_category.csv",
        index=False,
    )

    by_risk = (
        candidates.groupby(
            "transaction_risk_stratum",
            dropna=False,
        )
        .agg(
            cases=("case_id", "count"),
            median_wait_days=("elapsed_t0_to_t1_days", "median"),
            p90_wait_days=(
                "elapsed_t0_to_t1_days",
                lambda s: s.quantile(0.90),
            ),
            median_exposure_eur=("transaction_exposure_eur", "median"),
        )
        .reset_index()
    )
    by_risk["pct_of_candidates"] = (
        by_risk["cases"] / len(candidates) * 100.0
    ).round(4)
    by_risk.to_csv(
        out_dir / "gather_transition_by_risk_stratum.csv",
        index=False,
    )

    pd.DataFrame(event_timing_rows).to_csv(
        out_dir / "gather_transition_event_timing.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Feature reconstructability audit
    # ------------------------------------------------------------------
    feature_rows = [
        # Static fields
        {
            "feature": "item_category / company / source_system / document metadata",
            "status": "EXACT",
            "reason": "Static case context does not change between t0 and t1.",
        },
        {
            "feature": "po_value_initial_eur",
            "status": "EXACT",
            "reason": "Initial PO observation is carried forward unchanged.",
        },
        {
            "feature": "invoice_value_t0_eur",
            "status": "EXACT",
            "reason": "Original invoice observation is carried forward unchanged.",
        },
        {
            "feature": "gr_available_at_t1",
            "status": "EXACT",
            "reason": "By definition t1 is the first future GR; therefore GR is available.",
        },
        {
            "feature": "latest_gr_value_at_t1_eur",
            "status": "EXACT",
            "reason": "Equals first_future_gr_value_eur at the first future GR transition.",
        },
        {
            "feature": "n_goods_receipts_by_t1",
            "status": "EXACT",
            "reason": "Equals n_goods_receipts_before_t0 + 1 for first future GR.",
        },
        {
            "feature": "elapsed_t0_to_t1_days",
            "status": "EXACT",
            "reason": "Derived from decision_time and first_future_gr_time.",
        },
        {
            "feature": "selected history flags before t1",
            "status": "PARTIAL",
            "reason": (
                "Some first post-t0 timestamps are available, allowing boolean "
                "presence-before-t1 flags, but exact counts/order are not fully stored."
            ),
        },
        {
            "feature": "price/quantity/payment-block event counts by t1",
            "status": "NOT_EXACT_FROM_V2",
            "reason": (
                "v2 stores selected first future timestamps, not all event counts "
                "between t0 and t1."
            ),
        },
        {
            "feature": "events_available_at_t1",
            "status": "NOT_EXACT_FROM_V2",
            "reason": (
                "v2 stores total future-event count, not the complete count of events "
                "strictly up to first future GR."
            ),
        },
        {
            "feature": "same-timestamp event ordering at t1",
            "status": "NOT_EXACT_FROM_V2",
            "reason": (
                "Original XES event order is not preserved in the v2 case-level table "
                "for multiple events sharing the t1 timestamp."
            ),
        },
        {
            "feature": "post-GATHER action target Y_t1",
            "status": "NOT_PRESENT",
            "reason": (
                "The v2 benchmark contains future process observations but no validated "
                "ground-truth action label at t1. A target must be defined separately."
            ),
        },
    ]

    pd.DataFrame(feature_rows).to_csv(
        out_dir / "gather_transition_feature_reconstructability.csv",
        index=False,
    )

    # Candidate export: keep important fields plus diagnostic updates.
    export_cols = [
        "case_id",
        "item_category",
        "decision_time",
        "t1_time",
        "elapsed_t0_to_t1_days",
        "gr_available_at_t0",
        "gr_available_at_t1",
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "latest_gr_value_before_t0_eur",
        "first_future_gr_value_eur",
        "latest_gr_value_at_t1_eur",
        "latest_future_gr_value_eur",
        "n_goods_receipts_before_t0",
        "n_goods_receipts_by_t1",
        "n_future_goods_receipts",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "t1_after_t0",
        "t1_same_timestamp_as_t0",
        "t1_before_t0_invalid",
        "selected_intervening_event_before_t1",
        "selected_same_timestamp_order_ambiguity",
    ]

    # Include dynamically created event flags.
    export_cols.extend(before_cols)
    export_cols.extend(same_cols)
    export_cols = [
        c for c in export_cols
        if c in candidates.columns
    ]

    candidate_export = candidates[export_cols].copy()

    candidate_export.to_parquet(
        out_dir / "gather_transition_candidates.parquet",
        index=False,
    )
    if not args.no_csv:
        candidate_export.to_csv(
            out_dir / "gather_transition_candidates.csv",
            index=False,
        )

    invalid_before = int(
        candidates["t1_before_t0_invalid"].sum()
    )
    same_ts = int(
        candidates["t1_same_timestamp_as_t0"].sum()
    )
    intervening = int(
        candidates[
            "selected_intervening_event_before_t1"
        ].sum()
    )

    metadata = {
        "step": "09a.1",
        "input_file": str(input_path),
        "total_v2_cases": int(n_all),
        "eligible_natural_gather_cases": int(len(candidates)),
        "t1_definition": "first_future_gr_time",
        "eligibility_rule": (
            "gr_available_at_t0=False AND gather_gr_possible=True AND "
            "gather_gr_has_value=True AND first_future_gr_time is present"
        ),
        "invalid_t1_before_t0_cases": invalid_before,
        "same_timestamp_t0_t1_cases": same_ts,
        "selected_intervening_event_before_t1_cases": intervening,
        "minimal_t1_state_reconstructible": True,
        "full_original_feature_vector_reconstructible_from_v2": False,
        "post_gather_action_target_present_in_v2": False,
        "methodological_conclusion": (
            "The v2 table supports an exact minimal first-GR transition state, "
            "but not a complete reconstruction of every original predictive "
            "feature or a validated post-GATHER action target. A dedicated t1 "
            "estimator should not be trained until the target and state extraction "
            "semantics are explicitly defined. If complete event-state reconstruction "
            "is required, return to the raw XES log."
        ),
    }

    with open(
        out_dir / "gather_transition_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 09a.1 DIAGNOSTIC COMPLETE")
    print("=" * 80)

    print(f"All v2 cases                     : {n_all:,}")
    print(
        "No GR at t0                     : "
        f"{int(no_gr_t0.sum()):,}"
    )
    print(
        "No GR at t0 + future GR         : "
        f"{int((no_gr_t0 & future_gr).sum()):,}"
    )
    print(
        "Eligible natural GATHER cases   : "
        f"{len(candidates):,}"
    )
    print(
        "t1 strictly after t0            : "
        f"{int(candidates['t1_after_t0'].sum()):,}"
    )
    print(
        "t1 same timestamp as t0         : "
        f"{same_ts:,}"
    )
    print(
        "Invalid t1 before t0            : "
        f"{invalid_before:,}"
    )
    print(
        "Selected intervening event < t1 : "
        f"{intervening:,}"
    )

    print("\nKey conclusion:")
    print(
        "  v2 can reconstruct a minimal first-GR t1 state, but it cannot "
        "reconstruct every original feature exactly and does not contain a "
        "validated Y_t1 action label."
    )

    print("\nOutputs:")
    for name in [
        "gather_transition_summary.csv",
        "gather_transition_wait_time_summary.csv",
        "gather_transition_by_item_category.csv",
        "gather_transition_by_risk_stratum.csv",
        "gather_transition_event_timing.csv",
        "gather_transition_feature_reconstructability.csv",
        "gather_transition_candidates.parquet",
        None if args.no_csv else "gather_transition_candidates.csv",
        "gather_transition_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
