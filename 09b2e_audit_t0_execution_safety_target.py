#!/usr/bin/env python
"""
09b2e_audit_t0_execution_safety_target.py

RCSE Step 09b.2e
Audit candidate t0 execution-safety target definitions BEFORE training any
dedicated binary safety estimator.

Purpose
-------
The frozen multiclass action model's p(EXECUTE) is not a strong execution-safety
signal. Before creating a binary safety model, determine what "safe autonomous
execution at t0" should mean operationally.

This script DOES NOT:
- train any model,
- recalibrate any model,
- alter the frozen multiclass labels,
- alter the frozen RCSE policy,
- choose a final safety target automatically.

Inputs
------
--base-v2
    rcse_base_v2.parquet
    Contains one natural source-case row with t0 evidence and post-t0 outcomes.

--experimental
    rcse_experimental_with_splits.parquet
    Contains experimental episodes, source_case_id, interventions, and
    expected_action_reference.

Outputs
-------
t0_safety_signal_prevalence.csv
t0_safety_signal_overlap_matrix.csv
t0_safety_candidate_class_balance.csv
t0_safety_candidate_resolvability.csv
t0_safety_vs_reference_action.csv
t0_safety_by_intervention_family.csv
t0_safety_by_risk_stratum.csv
t0_safety_candidate_stability.csv
t0_safety_ambiguous_profile.csv
t0_safety_row_diagnostics.parquet
t0_safety_target_audit_metadata.json

Candidate definitions
---------------------
D1_CORE_NATURAL
    SAFE:
        required evidence is complete at t0,
        case eventually clears,
        and no CORE post-t0 adverse/control signal occurs.

    UNSAFE:
        required GR evidence is missing at t0,
        OR a CORE post-t0 adverse/control signal occurs.

    AMBIGUOUS:
        everything else.

D2_EXPANDED_NATURAL
    D1 plus additional invoice receipt as an adverse/control signal.

D3_HYBRID_EXPERIMENTAL
    For synthetic intervention episodes:
        UNSAFE (because the controlled intervention explicitly creates a state
        whose frozen benchmark reference action is GATHER/ESCALATE/ABSTAIN).

    For natural episodes:
        use D1_CORE_NATURAL.

D4_REFERENCE_BINARY
    SAFE iff expected_action_reference == EXECUTE.
    This is included only as a comparison baseline and MUST NOT automatically
    become the safety target.

CORE future signals
-------------------
- outcome_invoice_cancelled_after_t0
- outcome_payment_block_set_after_t0
- outcome_future_gr_cancel
- outcome_future_price_change
- outcome_future_quantity_change
- outcome_subsequent_invoice

Expanded adds:
- outcome_additional_invoice_receipt

Important interpretation
------------------------
Future outcomes are used ONLY for target-definition audit / possible target
construction. They are never t0 model features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CORE_SIGNALS = [
    "outcome_invoice_cancelled_after_t0",
    "outcome_payment_block_set_after_t0",
    "outcome_future_gr_cancel",
    "outcome_future_price_change",
    "outcome_future_quantity_change",
    "outcome_subsequent_invoice",
]

EXPANDED_SIGNALS = CORE_SIGNALS + [
    "outcome_additional_invoice_receipt",
]

REFERENCE_ACTIONS = ["EXECUTE", "GATHER", "ESCALATE", "ABSTAIN"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 09b.2e: audit t0 execution-safety target definitions."
    )
    p.add_argument(
        "--base-v2",
        type=Path,
        required=True,
        help="Path to rcse_base_v2.parquet",
    )
    p.add_argument(
        "--experimental",
        type=Path,
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <experimental parent>/t0_safety_target_audit",
    )
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def require(df: pd.DataFrame, cols: Iterable[str], label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} missing required columns:\n  - " + "\n  - ".join(missing)
        )


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    z = s.astype("string").str.strip().str.lower()
    return z.isin(["true", "1", "yes", "y", "t"])


def normalize_action(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def synthetic_flag(df: pd.DataFrame) -> pd.Series:
    if "synthetic_intervention" in df.columns:
        return as_bool(df["synthetic_intervention"])

    if "is_synthetic_intervention" in df.columns:
        return as_bool(df["is_synthetic_intervention"])

    if "intervention_type" in df.columns:
        z = (
            df["intervention_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        return ~z.isin(["", "NONE", "NATURAL", "NO_INTERVENTION"])

    return pd.Series(False, index=df.index)


def intervention_family(df: pd.DataFrame) -> pd.Series:
    if "intervention_type" not in df.columns:
        return pd.Series("UNKNOWN", index=df.index, dtype="string")

    z = (
        df["intervention_type"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series("OTHER", index=df.index, dtype="string")
    out.loc[z.isin(["", "NONE", "NATURAL", "NO_INTERVENTION"])] = "NATURAL"
    out.loc[z.str.contains("MISSING", na=False) | z.str.contains("HIDE", na=False)] = "MISSINGNESS"
    out.loc[z.str.contains("CONTRAD", na=False) | z.str.contains("PERTURB", na=False)] = "CONTRADICTION"
    out.loc[z.str.contains("SEVERE", na=False) | z.str.contains("MULTI", na=False)] = "SEVERE_EVIDENCE_LOSS"
    return out


def any_signal(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    x = pd.Series(False, index=df.index)
    for c in cols:
        x = x | as_bool(df[c])
    return x


def evidence_complete_at_t0(df: pd.DataFrame) -> pd.Series:
    """
    Required GR evidence is considered complete if:
      - GR is not required, OR
      - GR is available at t0.

    This is intentionally narrow and transparent.
    """
    gr_required = as_bool(df["goods_receipt_required"])
    gr_available = as_bool(df["gr_available_at_t0"])
    return (~gr_required) | gr_available


def make_natural_target(
    df: pd.DataFrame,
    adverse_cols: list[str],
) -> pd.Series:
    complete = evidence_complete_at_t0(df)
    adverse = any_signal(df, adverse_cols)
    cleared = as_bool(df["outcome_eventually_cleared"])

    target = pd.Series("AMBIGUOUS", index=df.index, dtype="string")

    # Unsafe if evidence required for execution is missing OR a core future
    # corrective/control signal is observed.
    target.loc[(~complete) | adverse] = "UNSAFE"

    # Safe only if evidence is complete, process later clears, and no core
    # adverse/control signal occurs.
    target.loc[complete & cleared & (~adverse)] = "SAFE"

    return target


def candidate_balance(df: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)

    for c in candidate_cols:
        counts = df[c].value_counts(dropna=False)
        for label in ["SAFE", "UNSAFE", "AMBIGUOUS"]:
            count = int(counts.get(label, 0))
            rows.append({
                "candidate": c,
                "label": label,
                "count": count,
                "pct_all": count / n if n else np.nan,
            })

    return pd.DataFrame(rows)


def candidate_resolvability(df: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows = []

    for c in candidate_cols:
        resolved = df[c].isin(["SAFE", "UNSAFE"])
        n_resolved = int(resolved.sum())
        n_safe = int((df.loc[resolved, c] == "SAFE").sum())
        n_unsafe = int((df.loc[resolved, c] == "UNSAFE").sum())
        n_amb = int((df[c] == "AMBIGUOUS").sum())

        rows.append({
            "candidate": c,
            "resolved_cases": n_resolved,
            "resolved_pct_all": n_resolved / len(df),
            "safe_resolved": n_safe,
            "unsafe_resolved": n_unsafe,
            "unsafe_pct_resolved": n_unsafe / n_resolved if n_resolved else np.nan,
            "ambiguous_cases": n_amb,
            "ambiguous_pct_all": n_amb / len(df),
        })

    return pd.DataFrame(rows)


def agreement_with_reference(df: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    rows = []

    for c in candidate_cols:
        tmp = (
            df.groupby(
                [c, "_reference_action"],
                dropna=False,
                observed=True,
            )
            .size()
            .reset_index(name="count")
        )
        tmp.insert(0, "candidate", c)
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def subgroup_table(df: pd.DataFrame, candidate: str, group_col: str) -> pd.DataFrame:
    tmp = (
        df.groupby(
            [group_col, candidate],
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )
    tmp["pct_within_group"] = (
        tmp["count"]
        / tmp.groupby(group_col)["count"].transform("sum")
    )
    return tmp


def main():
    args = parse_args()

    base_path = args.base_v2.resolve()
    exp_path = args.experimental.resolve()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (exp_path.parent / "t0_safety_target_audit").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 79)
    print("RCSE STEP 09b.2e - t0 EXECUTION-SAFETY TARGET AUDIT")
    print("=" * 79)
    print(f"Base v2      : {base_path}")
    print(f"Experimental : {exp_path}")
    print(f"Output       : {out_dir}")
    print("NO TRAINING / NO RETUNING")
    print()

    base = read_table(base_path)
    exp = read_table(exp_path)

    require(
        base,
        [
            "case_id",
            "goods_receipt_required",
            "gr_available_at_t0",
            "outcome_eventually_cleared",
            *EXPANDED_SIGNALS,
        ],
        "base v2",
    )

    require(
        exp,
        [
            "benchmark_episode_id",
            "source_case_id",
            "expected_action_reference",
        ],
        "experimental benchmark",
    )

    base = base.copy()
    exp = exp.copy()

    base["case_id"] = base["case_id"].astype(str)
    exp["source_case_id"] = exp["source_case_id"].astype(str)

    # Deduplicate base rows defensively.
    if base["case_id"].duplicated().any():
        raise AssertionError(
            f"base v2 contains duplicated case_id values: "
            f"{int(base['case_id'].duplicated().sum()):,}"
        )

    keep_base = [
        "case_id",
        "goods_receipt_required",
        "gr_available_at_t0",
        "outcome_eventually_cleared",
        "outcome_payment_block_removed_after_t0",
        "outcome_future_gr",
        "outcome_future_service_entry",
        "outcome_any_post_t0_exception",
        "gather_gr_possible",
        "gather_gr_has_value",
        "transaction_exposure_eur",
        "transaction_risk_stratum",
        *EXPANDED_SIGNALS,
    ]
    keep_base = [c for c in keep_base if c in base.columns]

    merged = exp.merge(
        base[keep_base].rename(columns={"case_id": "source_case_id"}),
        on="source_case_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_base"),
        indicator="_base_join",
    )

    missing_source = int((merged["_base_join"] != "both").sum())
    if missing_source:
        raise RuntimeError(
            f"{missing_source:,} experimental rows could not be joined to base v2."
        )
    merged.drop(columns=["_base_join"], inplace=True)

    merged["_reference_action"] = normalize_action(
        merged["expected_action_reference"]
    )
    merged["_is_synthetic"] = synthetic_flag(merged)
    merged["_intervention_family"] = intervention_family(merged)

    merged["_evidence_complete_t0"] = evidence_complete_at_t0(merged)
    merged["_core_adverse_any"] = any_signal(merged, CORE_SIGNALS)
    merged["_expanded_adverse_any"] = any_signal(merged, EXPANDED_SIGNALS)

    # ------------------------------------------------------------------
    # Candidate target definitions
    # ------------------------------------------------------------------
    merged["target_d1_core_natural"] = make_natural_target(
        merged,
        CORE_SIGNALS,
    )

    merged["target_d2_expanded_natural"] = make_natural_target(
        merged,
        EXPANDED_SIGNALS,
    )

    # Hybrid experimental target:
    # synthetic intervention => UNSAFE;
    # natural episode => D1 semantics.
    merged["target_d3_hybrid_experimental"] = merged[
        "target_d1_core_natural"
    ].copy()
    merged.loc[
        merged["_is_synthetic"],
        "target_d3_hybrid_experimental",
    ] = "UNSAFE"

    # Existing reference binary as comparison only.
    merged["target_d4_reference_binary"] = np.where(
        merged["_reference_action"] == "EXECUTE",
        "SAFE",
        "UNSAFE",
    )

    candidate_cols = [
        "target_d1_core_natural",
        "target_d2_expanded_natural",
        "target_d3_hybrid_experimental",
        "target_d4_reference_binary",
    ]

    # ------------------------------------------------------------------
    # Raw signal prevalence
    # ------------------------------------------------------------------
    signal_cols = [
        "goods_receipt_required",
        "gr_available_at_t0",
        "outcome_eventually_cleared",
        *EXPANDED_SIGNALS,
    ]
    signal_rows = []

    for c in signal_cols:
        b = as_bool(merged[c])
        signal_rows.append({
            "signal": c,
            "count_true": int(b.sum()),
            "pct_true": float(b.mean()),
        })

    pd.DataFrame(signal_rows).to_csv(
        out_dir / "t0_safety_signal_prevalence.csv",
        index=False,
    )

    # Overlap matrix for adverse/control signals.
    signal_matrix = pd.DataFrame({
        c: as_bool(merged[c]).astype(int)
        for c in EXPANDED_SIGNALS
    })
    overlap = signal_matrix.T.dot(signal_matrix)
    overlap.index.name = "signal"
    overlap.to_csv(
        out_dir / "t0_safety_signal_overlap_matrix.csv"
    )

    # Candidate balance/resolvability.
    balance = candidate_balance(merged, candidate_cols)
    balance.to_csv(
        out_dir / "t0_safety_candidate_class_balance.csv",
        index=False,
    )

    resolvability = candidate_resolvability(merged, candidate_cols)
    resolvability.to_csv(
        out_dir / "t0_safety_candidate_resolvability.csv",
        index=False,
    )

    # Candidate vs current multiclass reference.
    agreement = agreement_with_reference(merged, candidate_cols)
    agreement.to_csv(
        out_dir / "t0_safety_vs_reference_action.csv",
        index=False,
    )

    # Stability among candidate definitions.
    stability = (
        merged.groupby(
            candidate_cols,
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    stability["pct"] = stability["count"] / len(merged)
    stability.to_csv(
        out_dir / "t0_safety_candidate_stability.csv",
        index=False,
    )

    # Subgroups for the key hybrid candidate.
    primary_candidate = "target_d3_hybrid_experimental"

    by_intervention = subgroup_table(
        merged,
        primary_candidate,
        "_intervention_family",
    )
    by_intervention.to_csv(
        out_dir / "t0_safety_by_intervention_family.csv",
        index=False,
    )

    if "transaction_risk_stratum" in merged.columns:
        by_risk = subgroup_table(
            merged,
            primary_candidate,
            "transaction_risk_stratum",
        )
        by_risk.to_csv(
            out_dir / "t0_safety_by_risk_stratum.csv",
            index=False,
        )

    # Natural-only audit is important because future outcomes belong to source
    # cases, not synthetic intervention semantics.
    natural = merged.loc[~merged["_is_synthetic"]].copy()

    natural_agreement = agreement_with_reference(
        natural,
        [
            "target_d1_core_natural",
            "target_d2_expanded_natural",
            "target_d4_reference_binary",
        ],
    )
    natural_agreement.to_csv(
        out_dir / "t0_safety_natural_only_vs_reference.csv",
        index=False,
    )

    # Ambiguous profile for D1.
    amb = merged[
        merged["target_d1_core_natural"] == "AMBIGUOUS"
    ].copy()

    ambiguous_rows = []
    for c in [
        "outcome_eventually_cleared",
        *EXPANDED_SIGNALS,
        "outcome_future_gr",
        "outcome_payment_block_removed_after_t0",
        "gather_gr_possible",
    ]:
        if c in amb.columns:
            b = as_bool(amb[c])
            ambiguous_rows.append({
                "signal": c,
                "count_true": int(b.sum()),
                "pct_ambiguous": float(b.mean()) if len(amb) else np.nan,
            })

    pd.DataFrame(ambiguous_rows).to_csv(
        out_dir / "t0_safety_ambiguous_profile.csv",
        index=False,
    )

    # Row-level artifact for review / later target freezing.
    row_cols = [
        c for c in [
            "benchmark_episode_id",
            "source_case_id",
            "expected_action_reference",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
            "synthetic_intervention",
            "transaction_exposure_eur",
            "transaction_risk_stratum",
            "goods_receipt_required",
            "gr_available_at_t0",
            "outcome_eventually_cleared",
            "outcome_future_gr",
            "outcome_payment_block_removed_after_t0",
            "gather_gr_possible",
            *EXPANDED_SIGNALS,
            "_is_synthetic",
            "_intervention_family",
            "_evidence_complete_t0",
            "_core_adverse_any",
            "_expanded_adverse_any",
            *candidate_cols,
        ]
        if c in merged.columns
    ]

    merged[row_cols].to_parquet(
        out_dir / "t0_safety_row_diagnostics.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Compact comparison metrics
    # ------------------------------------------------------------------
    comparison_rows = []

    for c in candidate_cols:
        safe = merged[c] == "SAFE"
        unsafe = merged[c] == "UNSAFE"
        resolved = safe | unsafe

        ref_exec = merged["_reference_action"] == "EXECUTE"

        comparison_rows.append({
            "candidate": c,
            "resolved_n": int(resolved.sum()),
            "resolved_pct": float(resolved.mean()),
            "safe_n": int(safe.sum()),
            "unsafe_n": int(unsafe.sum()),
            "ambiguous_n": int((merged[c] == "AMBIGUOUS").sum()),
            "safe_and_reference_execute": int((safe & ref_exec).sum()),
            "safe_but_reference_not_execute": int((safe & ~ref_exec).sum()),
            "unsafe_but_reference_execute": int((unsafe & ref_exec).sum()),
            "unsafe_and_reference_not_execute": int((unsafe & ~ref_exec).sum()),
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        out_dir / "t0_safety_candidate_reference_conflicts.csv",
        index=False,
    )

    metadata = {
        "step": "09b.2e",
        "purpose": (
            "Audit operationally defensible t0 execution-safety targets before "
            "training any dedicated safety estimator."
        ),
        "training_performed": False,
        "recalibration_performed": False,
        "policy_retuning_performed": False,
        "base_v2": str(base_path),
        "experimental_benchmark": str(exp_path),
        "rows": int(len(merged)),
        "natural_rows": int((~merged["_is_synthetic"]).sum()),
        "synthetic_rows": int(merged["_is_synthetic"].sum()),
        "core_signals": CORE_SIGNALS,
        "expanded_signals": EXPANDED_SIGNALS,
        "candidate_definitions": {
            "D1_CORE_NATURAL": {
                "SAFE": (
                    "required GR evidence complete at t0 AND eventually clears "
                    "AND no core adverse/control signal"
                ),
                "UNSAFE": (
                    "required GR evidence missing at t0 OR any core adverse/control signal"
                ),
                "AMBIGUOUS": "neither SAFE nor UNSAFE rule resolves the case",
            },
            "D2_EXPANDED_NATURAL": (
                "D1 plus additional invoice receipt as an adverse/control signal"
            ),
            "D3_HYBRID_EXPERIMENTAL": (
                "synthetic intervention episodes are UNSAFE; natural episodes use D1"
            ),
            "D4_REFERENCE_BINARY": (
                "SAFE iff expected_action_reference == EXECUTE; comparison only"
            ),
        },
        "important_guardrail": (
            "Future outcomes are used only for target-definition audit / possible "
            "target construction and must never be t0 predictive features."
        ),
        "decision_rule_for_next_step": (
            "Do not automatically choose the candidate with highest agreement "
            "with expected_action_reference. Freeze a target only if its process "
            "semantics, ambiguity, and conflicts are defensible."
        ),
    }

    (out_dir / "t0_safety_target_audit_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 79)
    print("STEP 09b.2e COMPLETE")
    print("=" * 79)

    print("\nCandidate resolvability:")
    print(resolvability.to_string(index=False))

    print("\nCandidate vs reference-action conflict summary:")
    print(comparison.to_string(index=False))

    print("\nNatural/synthetic episode counts:")
    print(
        f"  Natural   : {int((~merged['_is_synthetic']).sum()):,}\n"
        f"  Synthetic : {int(merged['_is_synthetic'].sum()):,}"
    )

    print("\nOutputs:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p}")

    print(
        "\nNext decision: review D1/D2/D3 semantics and conflicts. "
        "Do NOT train a binary safety model until a target is explicitly frozen."
    )


if __name__ == "__main__":
    main()
