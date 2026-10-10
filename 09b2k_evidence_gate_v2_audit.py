#!/usr/bin/env python
"""
09b2k_evidence_gate_v2_audit.py

RCSE Step 09b.2k
Evidence-gate v2 specification and audit.

Purpose
-------
Audit a semantically motivated deterministic evidence-validity gate that combines:

1. Core evidence presence/completeness
2. Process-history consistency
3. Value consistency using existing invoice-vs-PO / invoice-vs-GR relative-difference fields

NO MODEL TRAINING.
NO RCSE POLICY CHANGE.
NO THRESHOLD SELECTION FROM TEST PERFORMANCE.

The value thresholds are predeclared sensitivity points aligned to the already
frozen contradiction intervention strengths.

Inputs
------
--experimental
    rcse_experimental_with_splits.parquet

--safety-robustness-dir
    Directory from Step 09b.2h containing:
        full_benchmark_safety_predictions_grouped_iid.parquet
        full_benchmark_safety_predictions_temporal.parquet
        full_benchmark_safety_predictions_ood_exposure.parquet

Outputs
-------
evidence_gate_v2_summary.csv
evidence_gate_v2_by_intervention.csv
evidence_gate_v2_by_contradiction_strength.csv
evidence_gate_v2_natural_false_blocks.csv
evidence_gate_v2_rule_ablation.csv
evidence_gate_v2_false_safe_overlap.csv
evidence_gate_v2_threshold_sensitivity.csv
evidence_gate_v2_row_diagnostics.parquet
evidence_gate_v2_manifest.json
evidence_gate_v2_metadata.json

Gate state
----------
VALID
INCOMPLETE
CONTRADICTORY

INCOMPLETE rules
----------------
- missing PO value
- missing invoice value
- required GR evidence unavailable
- vendor invoice evidence unavailable
- PO creation evidence unavailable

CONTRADICTORY process rules
---------------------------
- GR unavailable but GR count > 0
- GR available but GR count == 0
- cancellation/change/block flags without corresponding history counts

CONTRADICTORY value rule for threshold delta
--------------------------------------------
- inv_po_abs_rel_diff >= delta, when finite
OR
- GR available AND inv_gr_abs_rel_diff >= delta, when finite

IMPORTANT
---------
The mismatch fields are NOT used as historical REVIEW ground truth.
Here they are audited only as deterministic evidence-consistency relationships.

Evaluation-only fields
----------------------
intervention_type
intervention_strength
synthetic_intervention
expected_action_reference
p_safe_t0_raw / p_safe_t0_cal

These are never gate inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIME_FILES = {
    "grouped_iid": "full_benchmark_safety_predictions_grouped_iid.parquet",
    "temporal": "full_benchmark_safety_predictions_temporal.parquet",
    "ood_exposure": "full_benchmark_safety_predictions_ood_exposure.parquet",
}

SPLIT_CONFIG = {
    "grouped_iid": ("split_grouped_iid", "test"),
    "temporal": ("split_temporal", "test"),
    "ood_exposure": ("split_ood_exposure", "ood_test_r4"),
}

DEFAULT_DELTAS = [0.05, 0.10, 0.25, 0.50]
HIGH_SAFE_THRESHOLDS = [0.90, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2k: evidence-gate v2 audit."
    )
    p.add_argument(
        "--experimental",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--safety-robustness-dir",
        required=True,
        help="Step 09b.2h output directory.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <experimental parent>/evidence_gate_v2_audit",
    )
    p.add_argument(
        "--deltas",
        nargs="*",
        type=float,
        default=DEFAULT_DELTAS,
        help="Predeclared value-consistency thresholds.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def intervention_family(df: pd.DataFrame) -> pd.Series:
    z = (
        df["intervention_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series("OTHER", index=df.index, dtype="object")

    out.loc[
        z.isin(["", "NONE", "NATURAL", "NO_INTERVENTION"])
    ] = "NATURAL"

    out.loc[
        z.str.contains("MISSING", na=False)
        | z.str.contains("HIDE", na=False)
    ] = "MISSINGNESS"

    out.loc[
        z.str.contains("CONTRAD", na=False)
        | z.str.contains("PERTURB", na=False)
    ] = "CONTRADICTION"

    out.loc[
        z.str.contains("SEVERE", na=False)
        | z.str.contains("MULTI", na=False)
    ] = "SEVERE_EVIDENCE_LOSS"

    return out


def build_fixed_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Rules independent of value-consistency threshold delta.
    """
    gr_required = as_bool(df["goods_receipt_required"])
    gr_available = as_bool(df["gr_available_at_t0"])
    vendor_seen = as_bool(df["vendor_invoice_seen_at_t0"])
    po_created = as_bool(df["po_created_at_t0"])

    po_value = pd.to_numeric(df["po_value_initial_eur"], errors="coerce")
    inv_value = pd.to_numeric(df["invoice_value_t0_eur"], errors="coerce")

    n_gr = pd.to_numeric(
        df["n_goods_receipts_before_t0"], errors="coerce"
    ).fillna(0)

    n_cancel_gr = pd.to_numeric(
        df["n_cancel_gr_before_t0"], errors="coerce"
    ).fillna(0)

    n_cancel_inv = pd.to_numeric(
        df["n_cancel_invoice_before_t0"], errors="coerce"
    ).fillna(0)

    n_price = pd.to_numeric(
        df["n_price_change_before_t0"], errors="coerce"
    ).fillna(0)

    n_qty = pd.to_numeric(
        df["n_quantity_change_before_t0"], errors="coerce"
    ).fillna(0)

    n_set_block = pd.to_numeric(
        df["n_set_payment_block_before_t0"], errors="coerce"
    ).fillna(0)

    n_remove_block = pd.to_numeric(
        df["n_remove_payment_block_before_t0"], errors="coerce"
    ).fillna(0)

    rules = {
        # Explicit value/evidence presence.
        "missing_po_value": po_value.isna(),
        "missing_invoice_value": inv_value.isna(),
        "missing_required_gr": gr_required & (~gr_available),
        "missing_vendor_invoice_evidence": ~vendor_seen,
        "missing_po_creation_evidence": ~po_created,

        # Process-history consistency.
        "gr_flag_false_but_gr_count_positive": (
            (~gr_available) & (n_gr > 0)
        ),
        "gr_flag_true_but_gr_count_zero": (
            gr_available & (n_gr == 0)
        ),
        "gr_cancel_flag_without_cancel_count": (
            as_bool(df["had_gr_cancellation_before_t0"])
            & (n_cancel_gr == 0)
        ),
        "invoice_cancel_flag_without_cancel_count": (
            as_bool(df["had_invoice_cancellation_before_t0"])
            & (n_cancel_inv == 0)
        ),
        "price_change_flag_without_count": (
            as_bool(df["had_price_change_before_t0"])
            & (n_price == 0)
        ),
        "quantity_change_flag_without_count": (
            as_bool(df["had_quantity_change_before_t0"])
            & (n_qty == 0)
        ),
        "payment_block_flag_without_history": (
            as_bool(df["had_payment_block_before_t0"])
            & ((n_set_block + n_remove_block) == 0)
        ),
    }

    return rules


def build_value_rule(df: pd.DataFrame, delta: float) -> pd.Series:
    """
    Value-consistency rule. No intervention metadata is referenced here.
    """
    gr_available = as_bool(df["gr_available_at_t0"])

    inv_po = pd.to_numeric(
        df["inv_po_abs_rel_diff"],
        errors="coerce",
    )

    inv_gr = pd.to_numeric(
        df["inv_gr_abs_rel_diff"],
        errors="coerce",
    )

    po_bad = inv_po.notna() & (inv_po >= delta)

    gr_bad = (
        gr_available
        & inv_gr.notna()
        & (inv_gr >= delta)
    )

    return po_bad | gr_bad


def classify_gate(
    df: pd.DataFrame,
    fixed_rules: dict[str, pd.Series],
    value_rule: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    incomplete_names = [
        "missing_po_value",
        "missing_invoice_value",
        "missing_required_gr",
        "missing_vendor_invoice_evidence",
        "missing_po_creation_evidence",
    ]

    contradictory_names = [
        c for c in fixed_rules
        if c not in incomplete_names
    ]

    incomplete = pd.Series(False, index=df.index)
    contradictory = value_rule.copy()

    for c in incomplete_names:
        incomplete = incomplete | fixed_rules[c]

    for c in contradictory_names:
        contradictory = contradictory | fixed_rules[c]

    state = pd.Series("VALID", index=df.index, dtype="object")
    state.loc[incomplete] = "INCOMPLETE"
    state.loc[contradictory] = "CONTRADICTORY"

    return state, incomplete, contradictory


def summarize_by_intervention(
    df: pd.DataFrame,
    state_col: str,
    delta: float,
) -> pd.DataFrame:
    rows = []

    for fam, g in df.groupby("_intervention_family", dropna=False):
        blocked = g[state_col] != "VALID"

        rows.append(
            {
                "delta": delta,
                "intervention_family": fam,
                "episodes": int(len(g)),
                "blocked_count": int(blocked.sum()),
                "block_rate": float(blocked.mean()),
                "valid_rate": float((~blocked).mean()),
                "incomplete_rate": float(
                    (g[state_col] == "INCOMPLETE").mean()
                ),
                "contradictory_rate": float(
                    (g[state_col] == "CONTRADICTORY").mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def natural_false_block(
    df: pd.DataFrame,
    state_col: str,
    delta: float,
) -> pd.DataFrame:
    natural = df[
        df["_intervention_family"] == "NATURAL"
    ].copy()

    rows = []

    populations = {
        "ALL_NATURAL": pd.Series(True, index=natural.index),
        "NATURAL_REFERENCE_EXECUTE": (
            natural["expected_action_reference"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            == "EXECUTE"
        ),
    }

    for name, mask in populations.items():
        sub = natural.loc[mask]
        blocked = sub[state_col] != "VALID"

        rows.append(
            {
                "delta": delta,
                "population": name,
                "episodes": int(len(sub)),
                "blocked_count": int(blocked.sum()),
                "false_block_rate": (
                    float(blocked.mean()) if len(sub) else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def contradiction_strength_summary(
    df: pd.DataFrame,
    state_col: str,
    delta: float,
) -> pd.DataFrame:
    g = df[
        df["_intervention_family"] == "CONTRADICTION"
    ].copy()

    if g.empty:
        return pd.DataFrame()

    rows = []

    group_cols = ["intervention_type", "intervention_strength"]

    for keys, sub in g.groupby(
        group_cols,
        dropna=False,
        observed=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        blocked = sub[state_col] != "VALID"

        rows.append(
            {
                "delta": delta,
                "intervention_type": keys[0],
                "intervention_strength": keys[1],
                "episodes": int(len(sub)),
                "blocked_count": int(blocked.sum()),
                "detection_rate": float(blocked.mean()),
                "valid_rate": float((~blocked).mean()),
            }
        )

    return pd.DataFrame(rows)


def rule_ablation(
    df: pd.DataFrame,
    fixed_rules: dict[str, pd.Series],
    value_rule: pd.Series,
    delta: float,
) -> pd.DataFrame:
    rules = {
        **fixed_rules,
        f"value_inconsistency_delta_{delta:g}": value_rule,
    }

    rows = []

    families = [
        "NATURAL",
        "MISSINGNESS",
        "CONTRADICTION",
        "SEVERE_EVIDENCE_LOSS",
    ]

    for name, mask in rules.items():
        row = {
            "delta": delta,
            "rule": name,
            "overall_trigger_count": int(mask.sum()),
            "overall_trigger_rate": float(mask.mean()),
        }

        for fam in families:
            fm = df["_intervention_family"] == fam
            denom = int(fm.sum())
            numer = int((mask & fm).sum())

            row[f"{fam.lower()}_trigger_count"] = numer
            row[f"{fam.lower()}_trigger_rate"] = (
                numer / denom if denom else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def load_safety_predictions(
    robustness_dir: Path,
) -> pd.DataFrame:
    parts = []

    for regime, filename in REGIME_FILES.items():
        path = robustness_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing Step 09b.2h artifact: {path}"
            )

        x = pd.read_parquet(path)

        require_columns(
            x,
            [
                "benchmark_episode_id",
                "p_safe_t0_raw",
                "p_safe_t0_cal",
            ],
            filename,
        )

        x = x[
            [
                "benchmark_episode_id",
                "p_safe_t0_raw",
                "p_safe_t0_cal",
            ]
        ].copy()

        x["evaluation_regime"] = regime
        parts.append(x)

    return pd.concat(parts, ignore_index=True)


def false_safe_overlap(
    eval_df: pd.DataFrame,
    state_col: str,
    delta: float,
) -> pd.DataFrame:
    rows = []

    contradiction = eval_df[
        eval_df["_intervention_family"] == "CONTRADICTION"
    ].copy()

    for regime, rg in contradiction.groupby(
        "evaluation_regime",
        dropna=False,
    ):
        for score_type, col in [
            ("RAW", "p_safe_t0_raw"),
            ("CALIBRATED", "p_safe_t0_cal"),
        ]:
            for tau in HIGH_SAFE_THRESHOLDS:
                high = pd.to_numeric(
                    rg[col], errors="coerce"
                ) >= tau

                n_high = int(high.sum())

                if n_high == 0:
                    blocked_high = 0
                    residual = 0
                else:
                    blocked = rg[state_col] != "VALID"
                    blocked_high = int((high & blocked).sum())
                    residual = int((high & (~blocked)).sum())

                rows.append(
                    {
                        "delta": delta,
                        "regime": regime,
                        "score_type": score_type,
                        "high_safe_threshold": tau,
                        "contradiction_episodes": int(len(rg)),
                        "high_safe_contradiction_count": n_high,
                        "high_safe_contradiction_rate": (
                            n_high / len(rg) if len(rg) else np.nan
                        ),
                        "blocked_high_safe_contradictions": blocked_high,
                        "block_rate_within_high_safe_contradictions": (
                            blocked_high / n_high if n_high else np.nan
                        ),
                        "residual_high_safe_valid_contradictions": residual,
                        "residual_high_safe_valid_rate": (
                            residual / n_high if n_high else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    exp_path = Path(args.experimental).expanduser().resolve()
    robustness_dir = (
        Path(args.safety_robustness_dir)
        .expanduser()
        .resolve()
    )

    if not exp_path.exists():
        raise FileNotFoundError(exp_path)

    if not robustness_dir.exists():
        raise FileNotFoundError(robustness_dir)

    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else exp_path.parent / "evidence_gate_v2_audit"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    deltas = sorted(set(float(x) for x in args.deltas))

    print("=" * 80)
    print("RCSE STEP 09b.2k - EVIDENCE-GATE v2 AUDIT")
    print("=" * 80)
    print(f"Experimental       : {exp_path}")
    print(f"Safety robustness  : {robustness_dir}")
    print(f"Output             : {out_dir}")
    print(f"Value deltas       : {deltas}")
    print("NO TRAINING / NO POLICY CHANGE / NO TEST-TUNED THRESHOLD SELECTION")

    df = pd.read_parquet(exp_path)

    required = [
        "benchmark_episode_id",
        "source_case_id",
        "intervention_type",
        "intervention_strength",
        "expected_action_reference",

        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",

        "goods_receipt_required",
        "gr_available_at_t0",
        "vendor_invoice_seen_at_t0",
        "po_created_at_t0",

        "n_goods_receipts_before_t0",
        "n_cancel_gr_before_t0",
        "n_cancel_invoice_before_t0",
        "n_price_change_before_t0",
        "n_quantity_change_before_t0",
        "n_set_payment_block_before_t0",
        "n_remove_payment_block_before_t0",

        "had_gr_cancellation_before_t0",
        "had_invoice_cancellation_before_t0",
        "had_price_change_before_t0",
        "had_quantity_change_before_t0",
        "had_payment_block_before_t0",
    ]

    require_columns(df, required, "experimental benchmark")

    df = df.copy()
    df["_intervention_family"] = intervention_family(df)

    fixed_rules = build_fixed_rules(df)

    safety = load_safety_predictions(
        robustness_dir
    )

    # One experimental episode appears in only one test regime artifact.
    eval_df = df.merge(
        safety,
        on="benchmark_episode_id",
        how="left",
        validate="one_to_many",
    )

    summary_parts = []
    by_intervention_parts = []
    strength_parts = []
    natural_parts = []
    ablation_parts = []
    overlap_parts = []
    threshold_summary_rows = []

    # Keep row-level gate states for all deltas.
    row_diag_cols = [
        c for c in [
            "benchmark_episode_id",
            "source_case_id",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
            "expected_action_reference",
            "_intervention_family",
            "po_value_initial_eur",
            "invoice_value_t0_eur",
            "latest_gr_value_before_t0_eur",
            "inv_po_abs_rel_diff",
            "inv_gr_abs_rel_diff",
            "goods_receipt_required",
            "gr_available_at_t0",
        ]
        if c in df.columns
    ]

    row_diag = df[row_diag_cols].copy()

    for delta in deltas:
        value_rule = build_value_rule(
            df,
            delta,
        )

        state, incomplete, contradictory = classify_gate(
            df=df,
            fixed_rules=fixed_rules,
            value_rule=value_rule,
        )

        state_col = f"_gate_state_delta_{str(delta).replace('.', '_')}"

        df[state_col] = state
        row_diag[state_col] = state

        by_int = summarize_by_intervention(
            df=df,
            state_col=state_col,
            delta=delta,
        )

        nat = natural_false_block(
            df=df,
            state_col=state_col,
            delta=delta,
        )

        strength = contradiction_strength_summary(
            df=df,
            state_col=state_col,
            delta=delta,
        )

        abl = rule_ablation(
            df=df,
            fixed_rules=fixed_rules,
            value_rule=value_rule,
            delta=delta,
        )

        # Attach gate state to test safety-score rows.
        gate_lookup = df[
            [
                "benchmark_episode_id",
                "_intervention_family",
                state_col,
            ]
        ].copy()

        safety_eval = safety.merge(
            gate_lookup,
            on="benchmark_episode_id",
            how="left",
            validate="many_to_one",
        )

        overlap = false_safe_overlap(
            eval_df=safety_eval,
            state_col=state_col,
            delta=delta,
        )

        by_intervention_parts.append(by_int)
        natural_parts.append(nat)
        strength_parts.append(strength)
        ablation_parts.append(abl)
        overlap_parts.append(overlap)

        contradiction_row = by_int[
            by_int["intervention_family"] == "CONTRADICTION"
        ].iloc[0]

        missing_row = by_int[
            by_int["intervention_family"] == "MISSINGNESS"
        ].iloc[0]

        severe_row = by_int[
            by_int["intervention_family"] == "SEVERE_EVIDENCE_LOSS"
        ].iloc[0]

        nat_exec_row = nat[
            nat["population"] == "NATURAL_REFERENCE_EXECUTE"
        ].iloc[0]

        nat_all_row = nat[
            nat["population"] == "ALL_NATURAL"
        ].iloc[0]

        threshold_summary_rows.append(
            {
                "delta": delta,
                "contradiction_detection_rate": float(
                    contradiction_row["block_rate"]
                ),
                "missingness_detection_rate": float(
                    missing_row["block_rate"]
                ),
                "severe_detection_rate": float(
                    severe_row["block_rate"]
                ),
                "natural_all_block_rate": float(
                    nat_all_row["false_block_rate"]
                ),
                "natural_reference_execute_false_block_rate": float(
                    nat_exec_row["false_block_rate"]
                ),
            }
        )

    by_intervention_df = pd.concat(
        by_intervention_parts,
        ignore_index=True,
    )

    natural_df = pd.concat(
        natural_parts,
        ignore_index=True,
    )

    strength_df = pd.concat(
        strength_parts,
        ignore_index=True,
    )

    ablation_df = pd.concat(
        ablation_parts,
        ignore_index=True,
    )

    overlap_df = pd.concat(
        overlap_parts,
        ignore_index=True,
    )

    threshold_df = pd.DataFrame(
        threshold_summary_rows
    )

    summary_df = threshold_df.copy()

    summary_df.to_csv(
        out_dir / "evidence_gate_v2_summary.csv",
        index=False,
    )

    by_intervention_df.to_csv(
        out_dir / "evidence_gate_v2_by_intervention.csv",
        index=False,
    )

    strength_df.to_csv(
        out_dir / "evidence_gate_v2_by_contradiction_strength.csv",
        index=False,
    )

    natural_df.to_csv(
        out_dir / "evidence_gate_v2_natural_false_blocks.csv",
        index=False,
    )

    ablation_df.to_csv(
        out_dir / "evidence_gate_v2_rule_ablation.csv",
        index=False,
    )

    overlap_df.to_csv(
        out_dir / "evidence_gate_v2_false_safe_overlap.csv",
        index=False,
    )

    threshold_df.to_csv(
        out_dir / "evidence_gate_v2_threshold_sensitivity.csv",
        index=False,
    )

    row_diag.to_parquet(
        out_dir / "evidence_gate_v2_row_diagnostics.parquet",
        index=False,
    )

    manifest = {
        "step": "09b.2k",
        "gate_states": [
            "VALID",
            "INCOMPLETE",
            "CONTRADICTORY",
        ],
        "precedence": (
            "CONTRADICTORY overrides INCOMPLETE; otherwise VALID"
        ),
        "fixed_incomplete_rules": [
            "po_value_initial_eur is missing",
            "invoice_value_t0_eur is missing",
            "goods_receipt_required=True AND gr_available_at_t0=False",
            "vendor_invoice_seen_at_t0=False",
            "po_created_at_t0=False",
        ],
        "fixed_process_consistency_rules": [
            "gr_available_at_t0=False AND n_goods_receipts_before_t0>0",
            "gr_available_at_t0=True AND n_goods_receipts_before_t0==0",
            "had_gr_cancellation_before_t0=True AND n_cancel_gr_before_t0==0",
            "had_invoice_cancellation_before_t0=True AND n_cancel_invoice_before_t0==0",
            "had_price_change_before_t0=True AND n_price_change_before_t0==0",
            "had_quantity_change_before_t0=True AND n_quantity_change_before_t0==0",
            (
                "had_payment_block_before_t0=True AND "
                "(n_set_payment_block_before_t0+n_remove_payment_block_before_t0)==0"
            ),
        ],
        "value_consistency_rule": (
            "inv_po_abs_rel_diff >= delta OR "
            "(gr_available_at_t0=True AND inv_gr_abs_rel_diff >= delta)"
        ),
        "predeclared_deltas": deltas,
        "important_semantics": (
            "Mismatch fields are audited here only as deterministic consistency "
            "relationships; they remain rejected as universal historical REVIEW ground truth."
        ),
    }

    with open(
        out_dir / "evidence_gate_v2_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    metadata: dict[str, Any] = {
        "step": "09b.2k",
        "experimental_benchmark": str(exp_path),
        "safety_robustness_directory": str(robustness_dir),
        "model_training_performed": False,
        "policy_modified": False,
        "value_thresholds_selected_from_test": False,
        "threshold_origin": (
            "Sensitivity points aligned to the already-frozen controlled "
            "contradiction strengths; no post-hoc winner selection."
        ),
        "evaluation_only_fields": [
            "intervention_type",
            "intervention_strength",
            "expected_action_reference",
            "p_safe_t0_raw",
            "p_safe_t0_cal",
        ],
        "next_decision": (
            "Review contradiction detection, missingness detection, natural "
            "reference-EXECUTE false blocking, and overlap with high-confidence "
            "safety false positives. If a threshold is scientifically frozen, "
            "freeze the gate before RCSE v2 simulation."
        ),
    }

    with open(
        out_dir / "evidence_gate_v2_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 09b.2k COMPLETE")
    print("=" * 80)

    print("\nThreshold sensitivity summary:")
    print(
        threshold_df.to_string(
            index=False
        )
    )

    print("\nContradiction detection by intervention strength:")
    print(
        strength_df.to_string(
            index=False
        )
    )

    print("\nNatural false-block summary:")
    print(
        natural_df.to_string(
            index=False
        )
    )

    print("\nHigh-confidence contradiction overlap (calibrated p_safe >= 0.95):")
    focus = overlap_df[
        (overlap_df["score_type"] == "CALIBRATED")
        & (overlap_df["high_safe_threshold"] == 0.95)
    ]

    print(
        focus[
            [
                "delta",
                "regime",
                "high_safe_contradiction_count",
                "block_rate_within_high_safe_contradictions",
                "residual_high_safe_valid_contradictions",
                "residual_high_safe_valid_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nOutputs:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p}")

    print(
        "\nNext: do not choose a delta merely because it wins on the test set. "
        "Use these results to decide whether one semantically justified value-"
        "consistency threshold can be frozen before RCSE v2."
    )


if __name__ == "__main__":
    main()
