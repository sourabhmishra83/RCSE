#!/usr/bin/env python
"""
09b2i_evidence_consistency_gate_audit.py

RCSE Step 09b.2i
Design and audit an observable evidence-consistency gate WITHOUT retraining.

Purpose
-------
Step 09b.2h showed that a natural-only t0 safety estimator can be highly
confident on synthetic contradiction and missingness episodes.

This step asks a different question:

    Can simple, observable t0 evidence-consistency rules identify those
    unsafe uncertainty states before the learned safety score is even
    considered?

No model is trained.

Inputs
------
--experimental
    rcse_experimental_with_splits.parquet

Outputs
-------
evidence_gate_rule_manifest.json
evidence_gate_detection_by_intervention.csv
evidence_gate_false_block_natural.csv
evidence_gate_confusion_summary.csv
evidence_gate_rule_ablation.csv
evidence_gate_row_diagnostics.parquet
evidence_gate_metadata.json

Gate states
-----------
VALID
INCOMPLETE
CONTRADICTORY

Primary observable rules
------------------------
INCOMPLETE if any of:
    goods_receipt_required == True AND gr_available_at_t0 == False
    vendor_invoice_seen_at_t0 == False
    po_created_at_t0 == False

CONTRADICTORY if any of:
    gr_available_at_t0 == False AND n_goods_receipts_before_t0 > 0
    gr_available_at_t0 == True  AND n_goods_receipts_before_t0 == 0
    had_gr_cancellation_before_t0 == True AND n_cancel_gr_before_t0 == 0
    had_invoice_cancellation_before_t0 == True AND n_cancel_invoice_before_t0 == 0
    had_price_change_before_t0 == True AND n_price_change_before_t0 == 0
    had_quantity_change_before_t0 == True AND n_quantity_change_before_t0 == 0
    had_payment_block_before_t0 == True AND
        (n_set_payment_block_before_t0 + n_remove_payment_block_before_t0 == 0)

VALID otherwise.

Important
---------
The gate uses ONLY t0-observable state and process-history consistency.
It does NOT use:
- intervention_type as a feature
- synthetic_intervention as a feature
- future outcomes
- exposure/risk
- expected_action_reference
- learned safety probabilities

Those fields may be used ONLY for evaluation/reporting.
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
        description="Step 09b.2i: observable evidence-consistency gate audit."
    )
    p.add_argument(
        "--experimental",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <experimental parent>/evidence_consistency_gate_audit",
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
    """
    EVALUATION ONLY.
    """
    if "intervention_type" not in df.columns:
        return pd.Series(
            "UNKNOWN",
            index=df.index,
            dtype="object",
        )

    z = (
        df["intervention_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series(
        "OTHER",
        index=df.index,
        dtype="object",
    )

    out.loc[
        z.isin(
            ["", "NONE", "NATURAL", "NO_INTERVENTION"]
        )
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


def synthetic_flag(df: pd.DataFrame) -> pd.Series:
    """
    EVALUATION ONLY.
    """
    if "synthetic_intervention" in df.columns:
        return as_bool(df["synthetic_intervention"])

    if "is_synthetic_intervention" in df.columns:
        return as_bool(df["is_synthetic_intervention"])

    fam = intervention_family(df)
    return fam != "NATURAL"


def build_gate_rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    gr_required = as_bool(
        df["goods_receipt_required"]
    )

    gr_available = as_bool(
        df["gr_available_at_t0"]
    )

    vendor_invoice_seen = as_bool(
        df["vendor_invoice_seen_at_t0"]
    )

    po_created = as_bool(
        df["po_created_at_t0"]
    )

    had_gr_cancel = as_bool(
        df["had_gr_cancellation_before_t0"]
    )

    had_inv_cancel = as_bool(
        df["had_invoice_cancellation_before_t0"]
    )

    had_price_change = as_bool(
        df["had_price_change_before_t0"]
    )

    had_qty_change = as_bool(
        df["had_quantity_change_before_t0"]
    )

    had_payment_block = as_bool(
        df["had_payment_block_before_t0"]
    )

    n_gr = pd.to_numeric(
        df["n_goods_receipts_before_t0"],
        errors="coerce",
    ).fillna(0)

    n_cancel_gr = pd.to_numeric(
        df["n_cancel_gr_before_t0"],
        errors="coerce",
    ).fillna(0)

    n_cancel_inv = pd.to_numeric(
        df["n_cancel_invoice_before_t0"],
        errors="coerce",
    ).fillna(0)

    n_price = pd.to_numeric(
        df["n_price_change_before_t0"],
        errors="coerce",
    ).fillna(0)

    n_qty = pd.to_numeric(
        df["n_quantity_change_before_t0"],
        errors="coerce",
    ).fillna(0)

    n_set_block = pd.to_numeric(
        df["n_set_payment_block_before_t0"],
        errors="coerce",
    ).fillna(0)

    n_remove_block = pd.to_numeric(
        df["n_remove_payment_block_before_t0"],
        errors="coerce",
    ).fillna(0)

    rules = {
        # Incompleteness
        "missing_required_gr": (
            gr_required
            & (~gr_available)
        ),
        "missing_vendor_invoice_evidence": (
            ~vendor_invoice_seen
        ),
        "missing_po_creation_evidence": (
            ~po_created
        ),

        # Contradictions
        "gr_flag_false_but_gr_count_positive": (
            (~gr_available)
            & (n_gr > 0)
        ),
        "gr_flag_true_but_gr_count_zero": (
            gr_available
            & (n_gr == 0)
        ),
        "gr_cancel_flag_without_cancel_count": (
            had_gr_cancel
            & (n_cancel_gr == 0)
        ),
        "invoice_cancel_flag_without_cancel_count": (
            had_inv_cancel
            & (n_cancel_inv == 0)
        ),
        "price_change_flag_without_count": (
            had_price_change
            & (n_price == 0)
        ),
        "quantity_change_flag_without_count": (
            had_qty_change
            & (n_qty == 0)
        ),
        "payment_block_flag_without_history": (
            had_payment_block
            & (
                (n_set_block + n_remove_block)
                == 0
            )
        ),
    }

    return rules


def classify_gate(
    df: pd.DataFrame,
    rules: dict[str, pd.Series],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    incomplete_rule_names = [
        "missing_required_gr",
        "missing_vendor_invoice_evidence",
        "missing_po_creation_evidence",
    ]

    contradiction_rule_names = [
        c
        for c in rules
        if c not in incomplete_rule_names
    ]

    incomplete = pd.Series(
        False,
        index=df.index,
    )

    contradictory = pd.Series(
        False,
        index=df.index,
    )

    for c in incomplete_rule_names:
        incomplete = (
            incomplete
            | rules[c]
        )

    for c in contradiction_rule_names:
        contradictory = (
            contradictory
            | rules[c]
        )

    state = pd.Series(
        "VALID",
        index=df.index,
        dtype="object",
    )

    # Contradiction has precedence over incompleteness.
    state.loc[
        incomplete
    ] = "INCOMPLETE"

    state.loc[
        contradictory
    ] = "CONTRADICTORY"

    return state, incomplete, contradictory


def summarize_detection(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for family, g in df.groupby(
        "_intervention_family",
        dropna=False,
    ):
        blocked = (
            g["_gate_state"]
            != "VALID"
        )

        rows.append(
            {
                "intervention_family": family,
                "episodes": int(len(g)),
                "blocked_count": int(
                    blocked.sum()
                ),
                "block_rate": float(
                    blocked.mean()
                ),
                "incomplete_rate": float(
                    (
                        g["_gate_state"]
                        == "INCOMPLETE"
                    ).mean()
                ),
                "contradictory_rate": float(
                    (
                        g["_gate_state"]
                        == "CONTRADICTORY"
                    ).mean()
                ),
                "valid_rate": float(
                    (
                        g["_gate_state"]
                        == "VALID"
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def natural_false_block_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Uses natural episodes only.
    If expected_action_reference exists, report false block specifically
    among natural reference-EXECUTE episodes, but do not use that field in gate.
    """
    natural = df[
        df["_intervention_family"]
        == "NATURAL"
    ].copy()

    rows = []

    blocked = (
        natural["_gate_state"]
        != "VALID"
    )

    rows.append(
        {
            "population": "ALL_NATURAL",
            "episodes": int(
                len(natural)
            ),
            "blocked_count": int(
                blocked.sum()
            ),
            "false_block_rate": float(
                blocked.mean()
            )
            if len(natural)
            else np.nan,
        }
    )

    if "expected_action_reference" in natural.columns:
        ref = (
            natural[
                "expected_action_reference"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        nat_exec = natural[
            ref == "EXECUTE"
        ].copy()

        blocked_exec = (
            nat_exec["_gate_state"]
            != "VALID"
        )

        rows.append(
            {
                "population": "NATURAL_REFERENCE_EXECUTE",
                "episodes": int(
                    len(nat_exec)
                ),
                "blocked_count": int(
                    blocked_exec.sum()
                ),
                "false_block_rate": float(
                    blocked_exec.mean()
                )
                if len(nat_exec)
                else np.nan,
            }
        )

    return pd.DataFrame(
        rows
    )


def rule_ablation(
    df: pd.DataFrame,
    rules: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []

    families = [
        "NATURAL",
        "MISSINGNESS",
        "CONTRADICTION",
        "SEVERE_EVIDENCE_LOSS",
    ]

    for rule_name, mask in rules.items():
        row = {
            "rule": rule_name,
            "overall_trigger_count": int(
                mask.sum()
            ),
            "overall_trigger_rate": float(
                mask.mean()
            ),
        }

        for fam in families:
            fam_mask = (
                df["_intervention_family"]
                == fam
            )

            denom = int(
                fam_mask.sum()
            )

            numer = int(
                (
                    mask
                    & fam_mask
                ).sum()
            )

            row[
                f"{fam.lower()}_trigger_count"
            ] = numer

            row[
                f"{fam.lower()}_trigger_rate"
            ] = (
                numer / denom
                if denom
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    exp_path = Path(
        args.experimental
    ).expanduser().resolve()

    if not exp_path.exists():
        raise FileNotFoundError(
            exp_path
        )

    out_dir = (
        Path(
            args.out
        ).expanduser().resolve()
        if args.out
        else exp_path.parent
        / "evidence_consistency_gate_audit"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2i - EVIDENCE-CONSISTENCY GATE AUDIT"
    )
    print(
        "=" * 80
    )
    print(
        f"Experimental : {exp_path}"
    )
    print(
        f"Output       : {out_dir}"
    )
    print(
        "NO MODEL TRAINING. Gate uses only t0-observable evidence consistency."
    )

    df = pd.read_parquet(
        exp_path
    )

    required = [
        "benchmark_episode_id",
        "source_case_id",

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

    require_columns(
        df,
        required,
        "experimental benchmark",
    )

    df = df.copy()

    df[
        "_intervention_family"
    ] = intervention_family(
        df
    )

    df[
        "_is_synthetic"
    ] = synthetic_flag(
        df
    )

    rules = build_gate_rules(
        df
    )

    gate_state, incomplete, contradictory = classify_gate(
        df=df,
        rules=rules,
    )

    df[
        "_gate_state"
    ] = gate_state

    df[
        "_gate_incomplete"
    ] = incomplete

    df[
        "_gate_contradictory"
    ] = contradictory

    for name, mask in rules.items():
        df[
            f"_rule_{name}"
        ] = mask

    detection = summarize_detection(
        df
    )

    natural_false_block = natural_false_block_summary(
        df
    )

    ablation = rule_ablation(
        df=df,
        rules=rules,
    )

    # Compact confusion-style summary:
    # synthetic should ideally be BLOCKED; natural ideally VALID.
    natural = (
        df[
            "_intervention_family"
        ]
        == "NATURAL"
    )

    synthetic = ~natural

    blocked = (
        df[
            "_gate_state"
        ]
        != "VALID"
    )

    confusion = pd.DataFrame(
        [
            {
                "population": "NATURAL",
                "episodes": int(
                    natural.sum()
                ),
                "valid_count": int(
                    (
                        natural
                        & (~blocked)
                    ).sum()
                ),
                "blocked_count": int(
                    (
                        natural
                        & blocked
                    ).sum()
                ),
                "block_rate": float(
                    blocked.loc[
                        natural
                    ].mean()
                )
                if natural.any()
                else np.nan,
            },
            {
                "population": "SYNTHETIC",
                "episodes": int(
                    synthetic.sum()
                ),
                "valid_count": int(
                    (
                        synthetic
                        & (~blocked)
                    ).sum()
                ),
                "blocked_count": int(
                    (
                        synthetic
                        & blocked
                    ).sum()
                ),
                "block_rate": float(
                    blocked.loc[
                        synthetic
                    ].mean()
                )
                if synthetic.any()
                else np.nan,
            },
        ]
    )

    detection.to_csv(
        out_dir
        / "evidence_gate_detection_by_intervention.csv",
        index=False,
    )

    natural_false_block.to_csv(
        out_dir
        / "evidence_gate_false_block_natural.csv",
        index=False,
    )

    confusion.to_csv(
        out_dir
        / "evidence_gate_confusion_summary.csv",
        index=False,
    )

    ablation.to_csv(
        out_dir
        / "evidence_gate_rule_ablation.csv",
        index=False,
    )

    export_cols = [
        c
        for c in [
            "benchmark_episode_id",
            "source_case_id",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
            "synthetic_intervention",
            "expected_action_reference",

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

            "_intervention_family",
            "_is_synthetic",
            "_gate_state",
            "_gate_incomplete",
            "_gate_contradictory",
            *[
                f"_rule_{name}"
                for name in rules
            ],
        ]
        if c in df.columns
    ]

    df[
        export_cols
    ].to_parquet(
        out_dir
        / "evidence_gate_row_diagnostics.parquet",
        index=False,
    )

    rule_manifest = {
        "step": "09b.2i",
        "gate_states": [
            "VALID",
            "INCOMPLETE",
            "CONTRADICTORY",
        ],
        "precedence": (
            "CONTRADICTORY overrides INCOMPLETE; otherwise VALID"
        ),
        "rules": {
            "INCOMPLETE": [
                "goods_receipt_required=True AND gr_available_at_t0=False",
                "vendor_invoice_seen_at_t0=False",
                "po_created_at_t0=False",
            ],
            "CONTRADICTORY": [
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
        },
        "feature_boundary": (
            "Only t0-observable evidence-state and history-consistency fields "
            "are used by the gate."
        ),
    }

    with open(
        out_dir
        / "evidence_gate_rule_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rule_manifest,
            f,
            indent=2,
        )

    metadata: dict[str, Any] = {
        "step": "09b.2i",
        "experimental_benchmark": str(
            exp_path
        ),
        "model_training_performed": False,
        "gate_uses_intervention_metadata": False,
        "gate_uses_future_outcomes": False,
        "gate_uses_exposure_or_risk": False,
        "gate_uses_expected_action_reference": False,
        "evaluation_uses_intervention_family": True,
        "important_interpretation": (
            "A useful gate should block a high fraction of synthetic uncertainty "
            "episodes while keeping the block rate on natural/reference-EXECUTE "
            "episodes acceptably low."
        ),
        "next_decision": (
            "If contradiction/missingness detection is inadequate or natural "
            "false-blocking is excessive, refine ONLY semantically justified "
            "observable consistency rules. Do not optimize on test labels."
        ),
    }

    with open(
        out_dir
        / "evidence_gate_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 80
    )
    print(
        "STEP 09b.2i COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nGate detection by intervention family:"
    )

    print(
        detection.to_string(
            index=False
        )
    )

    print(
        "\nNatural false-block summary:"
    )

    print(
        natural_false_block.to_string(
            index=False
        )
    )

    print(
        "\nRule-level ablation:"
    )

    show_cols = [
        "rule",
        "natural_trigger_rate",
        "missingness_trigger_rate",
        "contradiction_trigger_rate",
        "severe_evidence_loss_trigger_rate",
    ]

    print(
        ablation[
            show_cols
        ].to_string(
            index=False
        )
    )

    print(
        "\nOutputs:"
    )

    for p in sorted(
        out_dir.iterdir()
    ):
        if p.is_file():
            print(
                f"  - {p}"
            )

    print(
        "\nNext: decide whether these semantically fixed observable rules "
        "provide a sufficient evidence-validity gate for RCSE v2."
    )


if __name__ == "__main__":
    main()
