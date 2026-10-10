#!/usr/bin/env python
"""
09b2j_intervention_footprint_audit.py

RCSE Step 09b.2j
Paired natural-vs-intervention observable-footprint audit.

Purpose
-------
For every synthetic experimental episode, pair it with the corresponding
NATURAL episode from the same source_case_id and quantify exactly which
t0-observable fields changed.

This step is DIAGNOSTIC ONLY.

It does NOT:
- train a model,
- modify the evidence gate,
- modify the RCSE policy,
- use future outcomes as gate inputs,
- use test performance to optimize a rule.

Inputs
------
--experimental
    rcse_experimental_with_splits.parquet

Outputs
-------
intervention_footprint_field_change_rates.csv
intervention_footprint_numeric_change_summary.csv
intervention_footprint_categorical_transitions.csv
intervention_footprint_boolean_transitions.csv
intervention_footprint_episode_change_counts.csv
intervention_footprint_change_combinations.csv
intervention_footprint_by_strength.csv
intervention_footprint_pair_diagnostics.parquet
intervention_footprint_manifest.json
intervention_footprint_metadata.json

Primary questions
-----------------
1. Which observable fields are actually altered by each intervention family?
2. How often does each field change?
3. For numeric fields, what is the direction/magnitude of change?
4. For booleans/categories, which state transitions are created?
5. Do contradiction interventions primarily alter value relationships rather
   than flag/count consistency?
6. Which observable relationships could support a semantically defensible
   deterministic evidence-consistency rule?

Important
---------
Intervention metadata is used only to GROUP and DESCRIBE the synthetic
perturbations. It is not treated as a deployable gate input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Candidate observable t0 fields.
NUMERIC_OBSERVABLES = [
    "po_value_initial_eur",
    "invoice_value_t0_eur",
    "latest_gr_value_before_t0_eur",

    "events_available_at_t0",

    "n_goods_receipts_before_t0",
    "n_cancel_gr_before_t0",
    "n_invoice_receipts_at_t0_history",
    "n_cancel_invoice_before_t0",
    "n_price_change_before_t0",
    "n_quantity_change_before_t0",
    "n_set_payment_block_before_t0",
    "n_remove_payment_block_before_t0",
    "n_service_entry_before_t0",
    "n_delete_po_before_t0",
    "n_block_po_before_t0",
    "n_reactivate_po_before_t0",
    "n_change_approval_before_t0",
    "n_release_po_before_t0",

    "po_to_invoice_days",
    "vendor_invoice_to_recorded_invoice_days",
    "last_gr_to_invoice_days",

    # These were rejected as learned REVIEW evidence but remain useful
    # DIAGNOSTIC relationships for footprint analysis.
    "inv_po_abs_rel_diff",
    "inv_gr_abs_rel_diff",
]

BOOLEAN_OBSERVABLES = [
    "gr_based_invoice_verification",
    "goods_receipt_required",
    "gr_available_at_t0",
    "vendor_invoice_seen_at_t0",
    "po_created_at_t0",

    "had_gr_cancellation_before_t0",
    "had_invoice_cancellation_before_t0",
    "had_price_change_before_t0",
    "had_quantity_change_before_t0",
    "had_payment_block_before_t0",
]

CATEGORICAL_OBSERVABLES = [
    "company",
    "source_system",
    "document_type",
    "document_category",
    "item_type",
    "item_category",
    "spend_classification",
    "spend_area",
    "sub_spend_area",
]

# Never include future outcomes or consequence fields as footprint inputs.
FORBIDDEN_PREFIXES = [
    "outcome_",
    "first_future_",
    "latest_future_",
    "post_gather_",
]

FORBIDDEN_EXACT = {
    "transaction_exposure_eur",
    "transaction_risk_stratum",
    "decision_year",
    "decision_month",
    "future_sequence_preview",
    "case_end_time",
    "events_future_after_t0",
    "time_to_clear_days",
    "time_to_first_future_gr_days",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2j: paired natural-vs-intervention footprint audit."
    )
    p.add_argument(
        "--experimental",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <experimental parent>/intervention_footprint_audit",
    )
    p.add_argument(
        "--numeric-atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for numeric equality.",
    )
    p.add_argument(
        "--numeric-rtol",
        type=float,
        default=1e-9,
        help="Relative tolerance for numeric equality.",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype("boolean")

    z = (
        s.astype("string")
        .str.strip()
        .str.lower()
    )

    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out.loc[z.isin(["true", "1", "yes", "y", "t"])] = True
    out.loc[z.isin(["false", "0", "no", "n", "f"])] = False
    return out


def normalize_cat(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .fillna("__NA__")
        .str.strip()
    )


def intervention_family(df: pd.DataFrame) -> pd.Series:
    if "intervention_type" not in df.columns:
        raise KeyError("intervention_type is required for footprint grouping.")

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


def synthetic_flag(df: pd.DataFrame) -> pd.Series:
    if "synthetic_intervention" in df.columns:
        x = df["synthetic_intervention"]
        if pd.api.types.is_bool_dtype(x):
            return x.fillna(False)

        z = (
            x.astype("string")
            .str.strip()
            .str.lower()
        )
        return z.isin(["true", "1", "yes", "y", "t"])

    fam = intervention_family(df)
    return fam != "NATURAL"


def validate_observable_columns(cols: list[str]) -> None:
    bad = []
    for c in cols:
        if c in FORBIDDEN_EXACT:
            bad.append(c)
        if any(c.startswith(p) for p in FORBIDDEN_PREFIXES):
            bad.append(c)

    if bad:
        raise AssertionError(
            "Forbidden future/consequence columns accidentally selected:\n  - "
            + "\n  - ".join(sorted(set(bad)))
        )


def numeric_changed(
    natural: pd.Series,
    synthetic: pd.Series,
    atol: float,
    rtol: float,
) -> pd.Series:
    a = pd.to_numeric(natural, errors="coerce")
    b = pd.to_numeric(synthetic, errors="coerce")

    both_na = a.isna() & b.isna()
    one_na = a.isna() ^ b.isna()

    equal_numeric = np.isclose(
        a.fillna(0).to_numpy(dtype=float),
        b.fillna(0).to_numpy(dtype=float),
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )

    changed = one_na | (~both_na & ~pd.Series(equal_numeric, index=a.index))
    return changed


def build_natural_map(
    df: pd.DataFrame,
) -> pd.DataFrame:
    natural = df[
        df["_intervention_family"] == "NATURAL"
    ].copy()

    if natural.empty:
        raise RuntimeError("No NATURAL episodes found.")

    dup = natural["source_case_id"].duplicated(keep=False)

    if dup.any():
        # Try to resolve exact duplicates by source case.
        compare_cols = [
            c for c in (
                NUMERIC_OBSERVABLES
                + BOOLEAN_OBSERVABLES
                + CATEGORICAL_OBSERVABLES
            )
            if c in natural.columns
        ]

        nunique = (
            natural.groupby("source_case_id")[compare_cols]
            .nunique(dropna=False)
        )

        conflicting = nunique.max(axis=1) > 1

        if conflicting.any():
            bad_ids = conflicting[conflicting].index[:10].tolist()
            raise AssertionError(
                "Multiple non-equivalent NATURAL episodes exist for some source_case_id values. "
                f"Examples: {bad_ids}"
            )

        natural = natural.drop_duplicates(
            "source_case_id",
            keep="first",
        )

    return natural


def pair_synthetic_to_natural(
    df: pd.DataFrame,
    natural: pd.DataFrame,
    observable_cols: list[str],
) -> pd.DataFrame:
    synth = df[
        df["_is_synthetic"]
    ].copy()

    nat_cols = [
        "source_case_id",
        "benchmark_episode_id",
        *observable_cols,
    ]

    nat = natural[nat_cols].copy()

    rename = {
        "benchmark_episode_id": "natural_benchmark_episode_id"
    }

    for c in observable_cols:
        rename[c] = f"natural__{c}"

    nat = nat.rename(columns=rename)

    paired = synth.merge(
        nat,
        on="source_case_id",
        how="left",
        validate="many_to_one",
        indicator="_natural_join",
    )

    missing = int((paired["_natural_join"] != "both").sum())

    if missing:
        raise RuntimeError(
            f"{missing:,} synthetic episodes could not be paired to a NATURAL episode."
        )

    paired.drop(columns=["_natural_join"], inplace=True)
    return paired


def summarize_field_changes(
    paired: pd.DataFrame,
    numeric_cols: list[str],
    bool_cols: list[str],
    cat_cols: list[str],
    atol: float,
    rtol: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    field_rows = []
    numeric_rows = []
    bool_rows = []
    cat_rows = []

    families = sorted(
        paired["_intervention_family"]
        .dropna()
        .unique()
    )

    for family in families:
        g = paired[
            paired["_intervention_family"] == family
        ].copy()

        if g.empty:
            continue

        # Numeric
        for c in numeric_cols:
            nat = pd.to_numeric(
                g[f"natural__{c}"],
                errors="coerce",
            )
            syn = pd.to_numeric(
                g[c],
                errors="coerce",
            )

            changed = numeric_changed(
                nat,
                syn,
                atol=atol,
                rtol=rtol,
            )

            n_changed = int(changed.sum())

            delta = syn - nat

            rel = pd.Series(
                np.nan,
                index=g.index,
                dtype=float,
            )

            denom = nat.abs()

            valid_rel = (
                changed
                & nat.notna()
                & syn.notna()
                & (denom > atol)
            )

            rel.loc[valid_rel] = (
                delta.loc[valid_rel]
                / denom.loc[valid_rel]
            )

            field_rows.append(
                {
                    "intervention_family": family,
                    "field": c,
                    "field_type": "NUMERIC",
                    "episodes": int(len(g)),
                    "changed_count": n_changed,
                    "changed_rate": n_changed / len(g),
                }
            )

            ch_delta = delta.loc[changed].dropna()
            ch_rel = rel.loc[changed].dropna()

            numeric_rows.append(
                {
                    "intervention_family": family,
                    "field": c,
                    "changed_count": n_changed,
                    "changed_rate": n_changed / len(g),
                    "missing_natural_to_value": int(
                        (nat.isna() & syn.notna()).sum()
                    ),
                    "value_to_missing": int(
                        (nat.notna() & syn.isna()).sum()
                    ),
                    "mean_delta": float(ch_delta.mean()) if len(ch_delta) else np.nan,
                    "median_delta": float(ch_delta.median()) if len(ch_delta) else np.nan,
                    "mean_abs_delta": float(ch_delta.abs().mean()) if len(ch_delta) else np.nan,
                    "median_abs_delta": float(ch_delta.abs().median()) if len(ch_delta) else np.nan,
                    "mean_relative_change": float(ch_rel.mean()) if len(ch_rel) else np.nan,
                    "median_relative_change": float(ch_rel.median()) if len(ch_rel) else np.nan,
                    "positive_delta_count": int((ch_delta > 0).sum()) if len(ch_delta) else 0,
                    "negative_delta_count": int((ch_delta < 0).sum()) if len(ch_delta) else 0,
                    "zero_delta_count_among_changed": int((ch_delta == 0).sum()) if len(ch_delta) else 0,
                }
            )

        # Boolean
        for c in bool_cols:
            nat = as_bool_series(
                g[f"natural__{c}"]
            )
            syn = as_bool_series(
                g[c]
            )

            nat_s = nat.astype("string").fillna("__NA__")
            syn_s = syn.astype("string").fillna("__NA__")

            changed = nat_s != syn_s

            field_rows.append(
                {
                    "intervention_family": family,
                    "field": c,
                    "field_type": "BOOLEAN",
                    "episodes": int(len(g)),
                    "changed_count": int(changed.sum()),
                    "changed_rate": float(changed.mean()),
                }
            )

            trans = (
                pd.DataFrame(
                    {
                        "natural_value": nat_s,
                        "synthetic_value": syn_s,
                        "changed": changed,
                    }
                )
                .groupby(
                    ["natural_value", "synthetic_value"],
                    dropna=False,
                )
                .size()
                .reset_index(name="count")
            )

            trans["intervention_family"] = family
            trans["field"] = c
            trans["changed"] = (
                trans["natural_value"]
                != trans["synthetic_value"]
            )

            bool_rows.extend(
                trans[
                    [
                        "intervention_family",
                        "field",
                        "natural_value",
                        "synthetic_value",
                        "changed",
                        "count",
                    ]
                ].to_dict(
                    orient="records"
                )
            )

        # Categorical
        for c in cat_cols:
            nat = normalize_cat(
                g[f"natural__{c}"]
            )
            syn = normalize_cat(
                g[c]
            )

            changed = nat != syn

            field_rows.append(
                {
                    "intervention_family": family,
                    "field": c,
                    "field_type": "CATEGORICAL",
                    "episodes": int(len(g)),
                    "changed_count": int(changed.sum()),
                    "changed_rate": float(changed.mean()),
                }
            )

            if changed.any():
                trans = (
                    pd.DataFrame(
                        {
                            "natural_value": nat.loc[changed],
                            "synthetic_value": syn.loc[changed],
                        }
                    )
                    .groupby(
                        ["natural_value", "synthetic_value"],
                        dropna=False,
                    )
                    .size()
                    .reset_index(name="count")
                    .sort_values(
                        "count",
                        ascending=False,
                    )
                )

                trans["intervention_family"] = family
                trans["field"] = c

                cat_rows.extend(
                    trans[
                        [
                            "intervention_family",
                            "field",
                            "natural_value",
                            "synthetic_value",
                            "count",
                        ]
                    ].to_dict(
                        orient="records"
                    )
                )

    return (
        pd.DataFrame(field_rows),
        pd.DataFrame(numeric_rows),
        pd.DataFrame(bool_rows),
        pd.DataFrame(cat_rows),
    )


def build_episode_change_matrix(
    paired: pd.DataFrame,
    numeric_cols: list[str],
    bool_cols: list[str],
    cat_cols: list[str],
    atol: float,
    rtol: float,
) -> pd.DataFrame:
    out = paired[
        [
            c for c in [
                "benchmark_episode_id",
                "natural_benchmark_episode_id",
                "source_case_id",
                "benchmark_arm",
                "intervention_type",
                "intervention_strength",
                "_intervention_family",
            ]
            if c in paired.columns
        ]
    ].copy()

    change_cols = []

    for c in numeric_cols:
        col = f"changed__{c}"

        out[col] = numeric_changed(
            paired[f"natural__{c}"],
            paired[c],
            atol=atol,
            rtol=rtol,
        ).to_numpy()

        change_cols.append(col)

    for c in bool_cols:
        nat = (
            as_bool_series(
                paired[f"natural__{c}"]
            )
            .astype("string")
            .fillna("__NA__")
        )

        syn = (
            as_bool_series(
                paired[c]
            )
            .astype("string")
            .fillna("__NA__")
        )

        col = f"changed__{c}"
        out[col] = (nat != syn).to_numpy()
        change_cols.append(col)

    for c in cat_cols:
        nat = normalize_cat(
            paired[f"natural__{c}"]
        )

        syn = normalize_cat(
            paired[c]
        )

        col = f"changed__{c}"
        out[col] = (nat != syn).to_numpy()
        change_cols.append(col)

    out["changed_field_count"] = (
        out[change_cols]
        .sum(axis=1)
        .astype(int)
    )

    out["changed_fields"] = out[
        change_cols
    ].apply(
        lambda row: "|".join(
            c.replace("changed__", "")
            for c, v in row.items()
            if bool(v)
        ),
        axis=1,
    )

    return out


def summarize_by_strength(
    episode_matrix: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "_intervention_family",
    ]

    if "intervention_type" in episode_matrix.columns:
        group_cols.append(
            "intervention_type"
        )

    if "intervention_strength" in episode_matrix.columns:
        group_cols.append(
            "intervention_strength"
        )

    rows = []

    for keys, g in episode_matrix.groupby(
        group_cols,
        dropna=False,
        observed=True,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {
            col: val
            for col, val in zip(group_cols, keys)
        }

        row.update(
            {
                "episodes": int(len(g)),
                "mean_changed_fields": float(
                    g["changed_field_count"].mean()
                ),
                "median_changed_fields": float(
                    g["changed_field_count"].median()
                ),
                "p90_changed_fields": float(
                    g["changed_field_count"].quantile(0.90)
                ),
                "zero_change_episodes": int(
                    (g["changed_field_count"] == 0).sum()
                ),
                "zero_change_rate": float(
                    (g["changed_field_count"] == 0).mean()
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


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
        Path(args.out).expanduser().resolve()
        if args.out
        else exp_path.parent / "intervention_footprint_audit"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("RCSE STEP 09b.2j - PAIRED INTERVENTION FOOTPRINT AUDIT")
    print("=" * 80)
    print(f"Experimental : {exp_path}")
    print(f"Output       : {out_dir}")
    print("NO MODEL TRAINING. NO GATE CHANGES. NO POLICY CHANGES.")

    df = pd.read_parquet(
        exp_path
    )

    require_columns(
        df,
        [
            "benchmark_episode_id",
            "source_case_id",
            "intervention_type",
        ],
        "experimental benchmark",
    )

    df = df.copy()

    df["source_case_id"] = (
        df["source_case_id"]
        .astype(str)
    )

    df["_intervention_family"] = intervention_family(
        df
    )

    df["_is_synthetic"] = synthetic_flag(
        df
    )

    numeric_cols = [
        c for c in NUMERIC_OBSERVABLES
        if c in df.columns
    ]

    bool_cols = [
        c for c in BOOLEAN_OBSERVABLES
        if c in df.columns
    ]

    cat_cols = [
        c for c in CATEGORICAL_OBSERVABLES
        if c in df.columns
    ]

    observable_cols = (
        numeric_cols
        + bool_cols
        + cat_cols
    )

    validate_observable_columns(
        observable_cols
    )

    print(
        f"\nObservable fields: "
        f"{len(numeric_cols)} numeric + "
        f"{len(bool_cols)} boolean + "
        f"{len(cat_cols)} categorical = "
        f"{len(observable_cols)} total"
    )

    natural = build_natural_map(
        df
    )

    paired = pair_synthetic_to_natural(
        df=df,
        natural=natural,
        observable_cols=observable_cols,
    )

    print(
        f"Natural source-case episodes : {len(natural):,}"
    )
    print(
        f"Synthetic paired episodes    : {len(paired):,}"
    )

    field_rates, numeric_summary, bool_transitions, cat_transitions = (
        summarize_field_changes(
            paired=paired,
            numeric_cols=numeric_cols,
            bool_cols=bool_cols,
            cat_cols=cat_cols,
            atol=args.numeric_atol,
            rtol=args.numeric_rtol,
        )
    )

    episode_matrix = build_episode_change_matrix(
        paired=paired,
        numeric_cols=numeric_cols,
        bool_cols=bool_cols,
        cat_cols=cat_cols,
        atol=args.numeric_atol,
        rtol=args.numeric_rtol,
    )

    # Most common changed-field combinations.
    combinations = (
        episode_matrix.groupby(
            [
                "_intervention_family",
                "changed_fields",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )

    combinations["pct_within_family"] = (
        combinations["count"]
        / combinations.groupby(
            "_intervention_family"
        )["count"].transform("sum")
    )

    combinations = combinations.sort_values(
        [
            "_intervention_family",
            "count",
        ],
        ascending=[
            True,
            False,
        ],
    )

    by_strength = summarize_by_strength(
        episode_matrix
    )

    field_rates.to_csv(
        out_dir
        / "intervention_footprint_field_change_rates.csv",
        index=False,
    )

    numeric_summary.to_csv(
        out_dir
        / "intervention_footprint_numeric_change_summary.csv",
        index=False,
    )

    bool_transitions.to_csv(
        out_dir
        / "intervention_footprint_boolean_transitions.csv",
        index=False,
    )

    cat_transitions.to_csv(
        out_dir
        / "intervention_footprint_categorical_transitions.csv",
        index=False,
    )

    episode_matrix[
        [
            c for c in episode_matrix.columns
            if not c.startswith("changed__")
        ]
    ].to_csv(
        out_dir
        / "intervention_footprint_episode_change_counts.csv",
        index=False,
    )

    combinations.to_csv(
        out_dir
        / "intervention_footprint_change_combinations.csv",
        index=False,
    )

    by_strength.to_csv(
        out_dir
        / "intervention_footprint_by_strength.csv",
        index=False,
    )

    # Row-level paired artifact includes natural and synthetic values for later
    # semantic inspection.
    export_cols = [
        c for c in [
            "benchmark_episode_id",
            "natural_benchmark_episode_id",
            "source_case_id",
            "benchmark_arm",
            "intervention_type",
            "intervention_strength",
            "_intervention_family",
            *observable_cols,
            *[
                f"natural__{c}"
                for c in observable_cols
            ],
        ]
        if c in paired.columns
    ]

    paired[
        export_cols
    ].to_parquet(
        out_dir
        / "intervention_footprint_pair_diagnostics.parquet",
        index=False,
    )

    manifest = {
        "step": "09b.2j",
        "numeric_observables": numeric_cols,
        "boolean_observables": bool_cols,
        "categorical_observables": cat_cols,
        "forbidden_future_or_consequence_columns": sorted(
            list(FORBIDDEN_EXACT)
            + FORBIDDEN_PREFIXES
        ),
        "pairing_key": "source_case_id",
        "natural_reference": (
            "NATURAL experimental episode from same source_case_id"
        ),
        "diagnostic_only": True,
    }

    with open(
        out_dir
        / "intervention_footprint_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    metadata: dict[str, Any] = {
        "step": "09b.2j",
        "experimental_benchmark": str(
            exp_path
        ),
        "natural_episode_count": int(
            len(natural)
        ),
        "paired_synthetic_episode_count": int(
            len(paired)
        ),
        "numeric_tolerance": {
            "atol": args.numeric_atol,
            "rtol": args.numeric_rtol,
        },
        "model_training_performed": False,
        "gate_modified": False,
        "policy_modified": False,
        "intervention_metadata_usage": (
            "diagnostic grouping only"
        ),
        "important_guardrail": (
            "Changed fields identified here are not automatically valid gate rules. "
            "A later rule may be proposed only if its ERP/process semantics are "
            "defensible independent of test-label optimization."
        ),
        "next_decision": (
            "Inspect which observable relationships encode contradiction/missingness, "
            "then determine whether semantically defensible deterministic consistency "
            "checks can be frozen before RCSE v2."
        ),
    }

    with open(
        out_dir
        / "intervention_footprint_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("STEP 09b.2j COMPLETE")
    print("=" * 80)

    print("\nTop changed fields by intervention family:")

    for family in sorted(
        field_rates["intervention_family"].unique()
    ):
        sub = (
            field_rates[
                field_rates["intervention_family"] == family
            ]
            .sort_values(
                "changed_rate",
                ascending=False,
            )
            .head(12)
        )

        print(f"\n[{family}]")
        print(
            sub[
                [
                    "field",
                    "field_type",
                    "changed_count",
                    "changed_rate",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nChange-count summary by intervention/strength:")
    print(
        by_strength.to_string(
            index=False
        )
    )

    print("\nMost common changed-field combinations:")
    for family in sorted(
        combinations["_intervention_family"].unique()
    ):
        sub = (
            combinations[
                combinations["_intervention_family"] == family
            ]
            .head(10)
        )

        print(f"\n[{family}]")
        print(
            sub[
                [
                    "changed_fields",
                    "count",
                    "pct_within_family",
                ]
            ].to_string(
                index=False
            )
        )

    print("\nOutputs:")
    for p in sorted(
        out_dir.iterdir()
    ):
        if p.is_file():
            print(f"  - {p}")

    print(
        "\nNext: use the footprint results to determine whether contradiction "
        "and missingness can be captured by semantically justified observable "
        "consistency checks before freezing an RCSE v2 evidence gate."
    )


if __name__ == "__main__":
    main()
