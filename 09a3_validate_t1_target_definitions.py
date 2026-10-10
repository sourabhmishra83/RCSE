"""
09a3_validate_t1_target_definitions.py

Validate candidate post-GATHER target definitions for RCSE.

Purpose
-------
Use the exact t1 transition benchmark from Step 09a.2 to determine whether a
defensible post-GATHER target Y_t1 can be constructed before training any
post-GATHER estimator.

This script does NOT train a model and does NOT permanently freeze a target.
It compares multiple candidate target definitions and quantifies ambiguity,
signal overlap, class balance, timing, and subgroup behavior.

Input
-----
t1_transition_exact.parquet
(or t1_transition_exact.csv)

Candidate target families
-------------------------
D1_CORE:
    NOT_SAFE if any core adverse/control signal occurs after t1:
        invoice cancellation
        payment block set
        GR cancellation
        price change
        quantity change
        subsequent invoice
    SAFE if eventually cleared AND no core adverse/control signal
    AMBIGUOUS otherwise

D2_EXPANDED:
    D1 plus additional invoice receipt as a NOT_SAFE signal.

D3_CONSERVATIVE:
    D2 plus future GR and future service entry as NOT_SAFE signals.
    This is intentionally conservative and is included as sensitivity only.

The script also evaluates a binary-on-resolved subset:
    SAFE_TO_EXECUTE
    NOT_SAFE_TO_EXECUTE
with AMBIGUOUS cases excluded.

Important
---------
Post-t1 future behavior is used only to construct/evaluate candidate target
definitions. Those future fields MUST NOT be used as t1 predictive features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CORE_UNSAFE_SIGNALS = [
    "after_t1_invoice_cancelled",
    "after_t1_payment_block_set",
    "after_t1_gr_cancel",
    "after_t1_price_change",
    "after_t1_quantity_change",
    "after_t1_subsequent_invoice",
]

EXPANDED_UNSAFE_SIGNALS = CORE_UNSAFE_SIGNALS + [
    "after_t1_additional_invoice",
]

CONSERVATIVE_UNSAFE_SIGNALS = EXPANDED_UNSAFE_SIGNALS + [
    "after_t1_future_gr",
    "after_t1_service_entry",
]

TARGET_DEFINITIONS = {
    "D1_CORE": CORE_UNSAFE_SIGNALS,
    "D2_EXPANDED": EXPANDED_UNSAFE_SIGNALS,
    "D3_CONSERVATIVE": CONSERVATIVE_UNSAFE_SIGNALS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09a.3: validate candidate post-GATHER Y_t1 targets."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to t1_transition_exact.parquet or CSV.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <input parent>/t1_target_validation",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip row-level diagnostic CSV.",
    )
    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported input format: {suffix}. Use .parquet or .csv."
    )


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns:\n  - " + "\n  - ".join(missing)
        )


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    true_values = {"true", "1", "yes", "y", "t"}
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(true_values)
    )


def pct(n: int, d: int) -> float:
    return round(float(n / d * 100.0), 4) if d else np.nan


def make_target(
    df: pd.DataFrame,
    unsafe_signals: list[str],
) -> tuple[pd.Series, pd.Series]:
    """
    Three-state target:
      NOT_SAFE_TO_EXECUTE: any configured unsafe signal after t1
      SAFE_TO_EXECUTE: eventually cleared and no configured unsafe signal
      AMBIGUOUS: neither of the above

    Returns target and unsafe-any flag.
    """
    unsafe_any = pd.Series(False, index=df.index)

    for c in unsafe_signals:
        unsafe_any = unsafe_any | as_bool(df[c])

    cleared = as_bool(df["after_t1_eventually_cleared"])

    target = pd.Series(
        "AMBIGUOUS",
        index=df.index,
        dtype="object",
    )

    target.loc[cleared & ~unsafe_any] = "SAFE_TO_EXECUTE"
    target.loc[unsafe_any] = "NOT_SAFE_TO_EXECUTE"

    return target, unsafe_any


def numeric_summary(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
) -> pd.DataFrame:
    rows = []

    for group, g in df.groupby(group_col, dropna=False):
        s = pd.to_numeric(
            g[metric_col],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                group_col: group,
                "metric": metric_col,
                "n": int(len(g)),
                "non_null": int(len(s)),
                "mean": float(s.mean()) if len(s) else np.nan,
                "p50": float(s.quantile(0.50)) if len(s) else np.nan,
                "p90": float(s.quantile(0.90)) if len(s) else np.nan,
                "p95": float(s.quantile(0.95)) if len(s) else np.nan,
                "p99": float(s.quantile(0.99)) if len(s) else np.nan,
                "max": float(s.max()) if len(s) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else input_path.parent / "t1_target_validation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RCSE STEP 09a.3 - t1 TARGET VALIDITY DIAGNOSTIC")
    print("=" * 80)
    print(f"Input : {input_path}")
    print(f"Output: {out_dir}")

    df = load_table(input_path)

    required = [
        "case_id",
        "item_category",
        "elapsed_t0_to_t1_days",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "after_t1_eventually_cleared",
        "time_t1_to_clear_days",
        *CONSERVATIVE_UNSAFE_SIGNALS,
    ]
    require_columns(df, required)

    # Normalize outcome booleans.
    outcome_cols = [
        "after_t1_eventually_cleared",
        "after_t1_payment_block_removed",
        *CONSERVATIVE_UNSAFE_SIGNALS,
    ]
    outcome_cols = [c for c in outcome_cols if c in df.columns]

    for c in outcome_cols:
        df[c] = as_bool(df[c])

    n = len(df)
    print(f"\nRows: {n:,}")

    # ------------------------------------------------------------------
    # 1. Raw post-t1 outcome signals
    # ------------------------------------------------------------------
    signal_rows = []
    for c in outcome_cols:
        count = int(df[c].sum())
        signal_rows.append(
            {
                "signal": c,
                "count": count,
                "pct": pct(count, n),
            }
        )

    signal_df = pd.DataFrame(signal_rows).sort_values(
        "count",
        ascending=False,
    )
    signal_df.to_csv(
        out_dir / "t1_target_signal_prevalence.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 2. Signal overlap / co-occurrence matrix
    # ------------------------------------------------------------------
    signal_matrix = df[outcome_cols].astype(int)
    overlap = signal_matrix.T.dot(signal_matrix)
    overlap.index.name = "signal"
    overlap.to_csv(
        out_dir / "t1_target_signal_overlap_matrix.csv"
    )

    # Conflict table: clearing AND each adverse signal
    conflict_rows = []
    cleared = df["after_t1_eventually_cleared"]

    for c in CONSERVATIVE_UNSAFE_SIGNALS:
        both = cleared & df[c]
        conflict_rows.append(
            {
                "unsafe_signal": c,
                "signal_count": int(df[c].sum()),
                "cleared_and_signal_count": int(both.sum()),
                "pct_signal_that_also_clears": (
                    round(
                        float(
                            both.sum()
                            / max(int(df[c].sum()), 1)
                            * 100.0
                        ),
                        4,
                    )
                ),
                "pct_all_cases": pct(int(both.sum()), n),
            }
        )

    pd.DataFrame(conflict_rows).to_csv(
        out_dir / "t1_target_clearance_conflicts.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Candidate target definitions
    # ------------------------------------------------------------------
    target_summary_rows = []
    resolved_rows = []
    row_target_cols = []

    for name, unsafe_signals in TARGET_DEFINITIONS.items():
        target_col = f"target_{name.lower()}"
        unsafe_col = f"unsafe_any_{name.lower()}"

        target, unsafe_any = make_target(
            df,
            unsafe_signals,
        )

        df[target_col] = target
        df[unsafe_col] = unsafe_any
        row_target_cols.extend([target_col, unsafe_col])

        counts = target.value_counts()

        for label in [
            "SAFE_TO_EXECUTE",
            "NOT_SAFE_TO_EXECUTE",
            "AMBIGUOUS",
        ]:
            count = int(counts.get(label, 0))
            target_summary_rows.append(
                {
                    "definition": name,
                    "label": label,
                    "count": count,
                    "pct_all": pct(count, n),
                }
            )

        resolved_mask = target != "AMBIGUOUS"
        n_resolved = int(resolved_mask.sum())
        n_safe = int(
            (target[resolved_mask] == "SAFE_TO_EXECUTE").sum()
        )
        n_unsafe = int(
            (
                target[resolved_mask]
                == "NOT_SAFE_TO_EXECUTE"
            ).sum()
        )

        resolved_rows.append(
            {
                "definition": name,
                "unsafe_signals": "|".join(unsafe_signals),
                "resolved_cases": n_resolved,
                "resolved_pct_all": pct(n_resolved, n),
                "ambiguous_cases": int((~resolved_mask).sum()),
                "ambiguous_pct_all": pct(
                    int((~resolved_mask).sum()),
                    n,
                ),
                "safe_resolved": n_safe,
                "unsafe_resolved": n_unsafe,
                "unsafe_pct_resolved": pct(
                    n_unsafe,
                    n_resolved,
                ),
                "safe_to_unsafe_ratio": (
                    round(n_safe / n_unsafe, 6)
                    if n_unsafe
                    else np.inf
                ),
            }
        )

    target_summary_df = pd.DataFrame(
        target_summary_rows
    )
    target_summary_df.to_csv(
        out_dir / "t1_candidate_target_class_balance.csv",
        index=False,
    )

    resolved_df = pd.DataFrame(resolved_rows)
    resolved_df.to_csv(
        out_dir / "t1_candidate_target_resolvability.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4. Cross-definition stability
    # ------------------------------------------------------------------
    target_cols = [
        f"target_{name.lower()}"
        for name in TARGET_DEFINITIONS
    ]

    stability = (
        df.groupby(target_cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    stability["pct"] = (
        stability["count"] / n * 100.0
    ).round(4)
    stability.to_csv(
        out_dir / "t1_target_definition_stability.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. D1 profile by item category and risk stratum
    # ------------------------------------------------------------------
    primary_candidate = "target_d1_core"

    by_item = (
        df.groupby(
            ["item_category", primary_candidate],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_item["pct_within_item_category"] = (
        by_item["count"]
        / by_item.groupby("item_category")["count"].transform("sum")
        * 100.0
    ).round(4)
    by_item.to_csv(
        out_dir / "t1_d1_target_by_item_category.csv",
        index=False,
    )

    by_risk = (
        df.groupby(
            [
                "transaction_risk_stratum",
                primary_candidate,
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    by_risk["pct_within_risk_stratum"] = (
        by_risk["count"]
        / by_risk.groupby(
            "transaction_risk_stratum"
        )["count"].transform("sum")
        * 100.0
    ).round(4)
    by_risk.to_csv(
        out_dir / "t1_d1_target_by_risk_stratum.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 6. Timing/exposure profile for candidate classes
    # ------------------------------------------------------------------
    timing_frames = []

    for metric in [
        "time_t1_to_clear_days",
        "elapsed_t0_to_t1_days",
        "transaction_exposure_eur",
    ]:
        timing_frames.append(
            numeric_summary(
                df,
                primary_candidate,
                metric,
            )
        )

    pd.concat(
        timing_frames,
        ignore_index=True,
    ).to_csv(
        out_dir / "t1_d1_target_numeric_profile.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 7. Investigate ambiguous D1 cases
    # ------------------------------------------------------------------
    ambiguous = df[
        df[primary_candidate] == "AMBIGUOUS"
    ].copy()

    ambiguous_rows = []

    if len(ambiguous):
        for c in outcome_cols:
            ambiguous_rows.append(
                {
                    "signal": c,
                    "count": int(ambiguous[c].sum()),
                    "pct_ambiguous": pct(
                        int(ambiguous[c].sum()),
                        len(ambiguous),
                    ),
                }
            )

    pd.DataFrame(ambiguous_rows).to_csv(
        out_dir / "t1_d1_ambiguous_signal_profile.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8. Binary resolved subset diagnostics
    # ------------------------------------------------------------------
    resolved = df[
        df[primary_candidate] != "AMBIGUOUS"
    ].copy()

    resolved["binary_y_t1"] = np.where(
        resolved[primary_candidate] == "SAFE_TO_EXECUTE",
        1,
        0,
    )

    binary_summary = pd.DataFrame(
        [
            {
                "metric": "all_cases",
                "value": n,
            },
            {
                "metric": "resolved_d1_cases",
                "value": len(resolved),
            },
            {
                "metric": "resolved_d1_pct_all",
                "value": (
                    len(resolved) / n * 100.0
                    if n else np.nan
                ),
            },
            {
                "metric": "safe_cases",
                "value": int(
                    (resolved["binary_y_t1"] == 1).sum()
                ),
            },
            {
                "metric": "not_safe_cases",
                "value": int(
                    (resolved["binary_y_t1"] == 0).sum()
                ),
            },
            {
                "metric": "not_safe_pct_resolved",
                "value": float(
                    (
                        resolved["binary_y_t1"] == 0
                    ).mean()
                    * 100.0
                ),
            },
        ]
    )

    binary_summary.to_csv(
        out_dir / "t1_d1_binary_resolved_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9. Machine-readable recommendation
    # ------------------------------------------------------------------
    d1_row = resolved_df[
        resolved_df["definition"] == "D1_CORE"
    ].iloc[0]

    d1_ambiguous_pct = float(
        d1_row["ambiguous_pct_all"]
    )
    d1_unsafe_pct_resolved = float(
        d1_row["unsafe_pct_resolved"]
    )

    # This is only a diagnostic recommendation, not an automatic scientific
    # conclusion.
    if d1_ambiguous_pct <= 10.0 and 1.0 <= d1_unsafe_pct_resolved <= 40.0:
        recommendation = (
            "D1_CORE binary-on-resolved subset is empirically viable for "
            "post-GATHER SAFE_TO_EXECUTE vs NOT_SAFE_TO_EXECUTE modeling, "
            "subject to substantive review of the adverse-signal semantics."
        )
    elif d1_ambiguous_pct > 10.0:
        recommendation = (
            "D1_CORE leaves more than 10% ambiguous; do not train a binary "
            "post-GATHER estimator until ambiguity is resolved or explicitly "
            "modeled."
        )
    else:
        recommendation = (
            "D1_CORE is highly imbalanced on the resolved subset; consider "
            "whether the post-GATHER target has enough unsafe examples for "
            "reliable estimation and evaluation."
        )

    metadata: dict[str, Any] = {
        "step": "09a.3",
        "input_file": str(input_path),
        "rows": int(n),
        "candidate_target_definitions": {
            k: v
            for k, v in TARGET_DEFINITIONS.items()
        },
        "primary_candidate_for_diagnostics": "D1_CORE",
        "D1_CORE_semantics": {
            "NOT_SAFE_TO_EXECUTE": (
                "At least one core adverse/control signal occurs after t1."
            ),
            "SAFE_TO_EXECUTE": (
                "Case later clears and no core adverse/control signal occurs "
                "after t1."
            ),
            "AMBIGUOUS": (
                "Neither a core adverse/control signal nor later clearing "
                "provides sufficient evidence for the candidate target."
            ),
        },
        "future_data_rule": (
            "Post-t1 outcomes are candidate target/evaluation information only "
            "and must never be supplied as t1 model features."
        ),
        "four_class_target_status": (
            "NOT JUSTIFIED by this script. GATHER and ABSTAIN are not inferred "
            "from post-t1 event traces merely to force a four-action label."
        ),
        "binary_target_status": "DIAGNOSTIC_ONLY_NOT_FROZEN",
        "diagnostic_recommendation": recommendation,
    }

    with open(
        out_dir / "t1_target_validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)

    # Row-level diagnostic export.
    row_export_cols = [
        "case_id",
        "item_category",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        "elapsed_t0_to_t1_days",
        "time_t1_to_clear_days",
        *outcome_cols,
        *target_cols,
    ]

    row_export_cols = list(
        dict.fromkeys(
            c for c in row_export_cols
            if c in df.columns
        )
    )

    if not args.no_csv:
        df[row_export_cols].to_csv(
            out_dir / "t1_target_row_diagnostics.csv",
            index=False,
        )

    print("\n" + "=" * 80)
    print("STEP 09a.3 COMPLETE")
    print("=" * 80)

    print("\nCandidate target resolvability:")
    print(
        resolved_df[
            [
                "definition",
                "resolved_cases",
                "resolved_pct_all",
                "ambiguous_cases",
                "ambiguous_pct_all",
                "safe_resolved",
                "unsafe_resolved",
                "unsafe_pct_resolved",
            ]
        ].to_string(index=False)
    )

    print("\nD1_CORE recommendation:")
    print(f"  {recommendation}")

    print("\nOutputs:")
    for name in [
        "t1_target_signal_prevalence.csv",
        "t1_target_signal_overlap_matrix.csv",
        "t1_target_clearance_conflicts.csv",
        "t1_candidate_target_class_balance.csv",
        "t1_candidate_target_resolvability.csv",
        "t1_target_definition_stability.csv",
        "t1_d1_target_by_item_category.csv",
        "t1_d1_target_by_risk_stratum.csv",
        "t1_d1_target_numeric_profile.csv",
        "t1_d1_ambiguous_signal_profile.csv",
        "t1_d1_binary_resolved_summary.csv",
        None if args.no_csv else "t1_target_row_diagnostics.csv",
        "t1_target_validation_metadata.json",
    ]:
        if name:
            print(f"  - {out_dir / name}")


if __name__ == "__main__":
    main()
