#!/usr/bin/env python
"""
09b2q_residual_contradiction_separability_audit.py

RCSE Step 09b.2q
Residual contradiction separability audit.

Purpose
-------
Diagnose the dominant remaining RCSE v2 failure mode identified in Step 09b.2p:

    residual synthetic CONTRADICTION episodes that remain VALID under
    Evidence Gate v2 and are subsequently executed.

This step asks:

    Are those residual contradictions separable from legitimate natural
    reference-EXECUTE cases using semantically defensible observable evidence?

DIAGNOSTIC ONLY.

NO MODEL TRAINING.
NO RECALIBRATION.
NO GATE MODIFICATION.
NO POLICY RETUNING.
NO TEST-SET THRESHOLD SELECTION.

Inputs
------
--dataset-dir
    Step 09b.2m rcse_v2_policy_dataset directory.

--results-dir
    Step 09b.2n rcse_v2_policy_results directory.

--experimental
    rcse_experimental_with_splits.parquet
    Used only to attach intervention metadata and any observable fields not
    retained in the policy-ready dataset.

--spec
    Frozen 09b2l_rcse_v2_policy_spec.json

Outputs
-------
rcse_v2q_contradiction_flow.csv
rcse_v2q_contradiction_by_type_strength.csv
rcse_v2q_gate_rule_activation.csv
rcse_v2q_residual_valid_rule_profile.csv
rcse_v2q_residual_executed_rule_profile.csv
rcse_v2q_observable_distribution_summary.csv
rcse_v2q_residual_vs_natural_separability.csv
rcse_v2q_residual_vs_natural_quantile_overlap.csv
rcse_v2q_episode_diagnostics_<regime>.parquet
rcse_v2q_metadata.json

Primary comparison groups
-------------------------
1. CONTRADICTION_BLOCKED
2. CONTRADICTION_VALID_NOT_EXECUTED
3. CONTRADICTION_VALID_EXECUTED
4. NATURAL_REFERENCE_EXECUTE

Key observable variables
------------------------
- po_value_initial_eur
- invoice_value_t0_eur
- latest_gr_value_before_t0_eur
- inv_po_abs_rel_diff
- inv_gr_abs_rel_diff
- gr_available_at_t0
- goods_receipt_required
- p_safe_t0_cal
- transaction_exposure_eur
- process/history counts and flags

Important interpretation
------------------------
If residual valid/executed contradictions occupy a region that is also common
among natural reference-EXECUTE cases, then adding a deterministic rule to
catch them may create substantial legitimate false blocking.

If they occupy a clearly distinct or internally impossible observable region,
a later semantically justified gate revision may be considered -- but not here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REGIMES = ("grouped_iid", "temporal", "ood_exposure")

DATASET_FILES = {
    "grouped_iid": "rcse_v2_policy_dataset_grouped_iid.parquet",
    "temporal": "rcse_v2_policy_dataset_temporal.parquet",
    "ood_exposure": "rcse_v2_policy_dataset_ood_exposure.parquet",
}

DECISION_FILES = {
    "grouped_iid": "rcse_v2_policy_episode_decisions_grouped_iid.parquet",
    "temporal": "rcse_v2_policy_episode_decisions_temporal.parquet",
    "ood_exposure": "rcse_v2_policy_episode_decisions_ood_exposure.parquet",
}

PRIMARY_POLICY = "RCSE_V2_FULL"
PRIMARY_CONSEQUENCE = "LOG_TRAIN_P90"

NUMERIC_OBSERVABLES = [
    "po_value_initial_eur",
    "invoice_value_t0_eur",
    "latest_gr_value_before_t0_eur",
    "inv_po_abs_rel_diff",
    "inv_gr_abs_rel_diff",
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
    "p_safe_t0_cal",
    "p_safe_t0_raw",
    "transaction_exposure_eur",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 09b.2q: residual contradiction separability audit."
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Step 09b.2m rcse_v2_policy_dataset directory.",
    )
    p.add_argument(
        "--results-dir",
        required=True,
        help="Step 09b.2n rcse_v2_policy_results directory.",
    )
    p.add_argument(
        "--experimental",
        required=True,
        help="Path to rcse_experimental_with_splits.parquet",
    )
    p.add_argument(
        "--spec",
        required=True,
        help="Frozen 09b2l_rcse_v2_policy_spec.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Default: <results-dir>/residual_contradiction_audit",
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label}: missing required columns:\n  - "
            + "\n  - ".join(missing)
        )


def normalize_action(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
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


def intervention_family(s: pd.Series) -> pd.Series:
    z = (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    out = pd.Series("OTHER", index=s.index, dtype="object")

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


def validate_spec(spec: dict[str, Any]) -> float:
    if spec.get("status") != "FROZEN_BEFORE_RCSE_V2_SIMULATION":
        raise RuntimeError(
            "Spec is not frozen RCSE v2 specification."
        )

    delta = (
        spec.get("architecture", {})
        .get("evidence_gate", {})
        .get("value_consistency_delta")
    )

    if delta is None:
        raise RuntimeError(
            "Frozen gate delta missing from specification."
        )

    return float(delta)


def build_gate_rules(
    df: pd.DataFrame,
    delta: float,
) -> dict[str, pd.Series]:
    require_columns(
        df,
        [
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
        ],
        "gate-rule input",
    )

    gr_required = as_bool(df["goods_receipt_required"])
    gr_available = as_bool(df["gr_available_at_t0"])
    vendor_seen = as_bool(df["vendor_invoice_seen_at_t0"])
    po_created = as_bool(df["po_created_at_t0"])

    po_value = pd.to_numeric(
        df["po_value_initial_eur"],
        errors="coerce",
    )

    inv_value = pd.to_numeric(
        df["invoice_value_t0_eur"],
        errors="coerce",
    )

    inv_po = pd.to_numeric(
        df["inv_po_abs_rel_diff"],
        errors="coerce",
    )

    inv_gr = pd.to_numeric(
        df["inv_gr_abs_rel_diff"],
        errors="coerce",
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
        "missing_po_value": po_value.isna(),
        "missing_invoice_value": inv_value.isna(),
        "missing_required_gr": gr_required & (~gr_available),
        "missing_vendor_invoice_evidence": ~vendor_seen,
        "missing_po_creation_evidence": ~po_created,
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
        f"value_inconsistency_delta_{delta:g}": (
            (inv_po.notna() & (inv_po >= delta))
            | (
                gr_available
                & inv_gr.notna()
                & (inv_gr >= delta)
            )
        ),
    }

    return rules


def attach_experimental_fields(
    df: pd.DataFrame,
    exp: pd.DataFrame,
) -> pd.DataFrame:
    exp_cols = [
        c for c in [
            "benchmark_episode_id",
            "intervention_type",
            "intervention_strength",
            "benchmark_arm",
            "synthetic_intervention",
            *NUMERIC_OBSERVABLES,
            *BOOLEAN_OBSERVABLES,
        ]
        if c in exp.columns
    ]

    exp_sub = exp[exp_cols].copy()

    if exp_sub["benchmark_episode_id"].duplicated().any():
        raise AssertionError(
            "Experimental benchmark_episode_id is not unique."
        )

    missing_from_df = [
        c for c in exp_cols
        if c != "benchmark_episode_id"
        and c not in df.columns
    ]

    if not missing_from_df:
        return df

    merge_cols = [
        "benchmark_episode_id",
        *missing_from_df,
    ]

    out = df.merge(
        exp_sub[merge_cols],
        on="benchmark_episode_id",
        how="left",
        validate="one_to_one",
    )

    return out


def classify_comparison_group(df: pd.DataFrame) -> pd.Series:
    fam = df["_intervention_family_q"]
    gate = (
        df["gate_state_v2"]
        .fillna("")
        .astype(str)
        .str.upper()
    )
    action = df["rcse_v2_full_action"]
    ref = normalize_action(
        df["expected_action_reference"]
    )

    out = pd.Series(
        "OTHER",
        index=df.index,
        dtype="object",
    )

    contradiction = fam == "CONTRADICTION"

    out.loc[
        contradiction
        & (gate != "VALID")
    ] = "CONTRADICTION_BLOCKED"

    out.loc[
        contradiction
        & (gate == "VALID")
        & (action != "EXECUTE")
    ] = "CONTRADICTION_VALID_NOT_EXECUTED"

    out.loc[
        contradiction
        & (gate == "VALID")
        & (action == "EXECUTE")
    ] = "CONTRADICTION_VALID_EXECUTED"

    out.loc[
        (fam == "NATURAL")
        & (ref == "EXECUTE")
    ] = "NATURAL_REFERENCE_EXECUTE"

    return out


def flow_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for regime, g in df.groupby(
        "evaluation_regime",
        observed=True,
    ):
        contradiction = g[
            g["_intervention_family_q"]
            == "CONTRADICTION"
        ]

        if contradiction.empty:
            continue

        blocked = (
            contradiction[
                "gate_state_v2"
            ]
            != "VALID"
        )

        valid = ~blocked

        executed = (
            contradiction[
                "rcse_v2_full_action"
            ]
            == "EXECUTE"
        )

        rows.append(
            {
                "regime": regime,
                "contradiction_episodes": int(
                    len(contradiction)
                ),
                "gate_blocked_count": int(
                    blocked.sum()
                ),
                "gate_block_rate": float(
                    blocked.mean()
                ),
                "residual_valid_count": int(
                    valid.sum()
                ),
                "residual_valid_rate": float(
                    valid.mean()
                ),
                "residual_valid_executed_count": int(
                    (valid & executed).sum()
                ),
                "residual_valid_execution_rate": (
                    float(
                        (valid & executed).sum()
                        / valid.sum()
                    )
                    if valid.sum()
                    else np.nan
                ),
                "contradiction_execution_rate_overall": float(
                    executed.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def type_strength_summary(df: pd.DataFrame) -> pd.DataFrame:
    c = df[
        df["_intervention_family_q"]
        == "CONTRADICTION"
    ].copy()

    rows = []

    for (
        regime,
        itype,
        strength,
    ), g in c.groupby(
        [
            "evaluation_regime",
            "intervention_type",
            "intervention_strength",
        ],
        dropna=False,
        observed=True,
    ):
        blocked = (
            g["gate_state_v2"]
            != "VALID"
        )

        valid = ~blocked

        executed = (
            g["rcse_v2_full_action"]
            == "EXECUTE"
        )

        rows.append(
            {
                "regime": regime,
                "intervention_type": itype,
                "intervention_strength": strength,
                "episodes": int(
                    len(g)
                ),
                "gate_blocked": int(
                    blocked.sum()
                ),
                "gate_block_rate": float(
                    blocked.mean()
                ),
                "residual_valid": int(
                    valid.sum()
                ),
                "residual_valid_rate": float(
                    valid.mean()
                ),
                "residual_valid_executed": int(
                    (valid & executed).sum()
                ),
                "execution_rate_within_residual_valid": (
                    float(
                        (valid & executed).sum()
                        / valid.sum()
                    )
                    if valid.sum()
                    else np.nan
                ),
                "mean_p_safe_t0_cal_residual_valid": (
                    float(
                        pd.to_numeric(
                            g.loc[
                                valid,
                                "p_safe_t0_cal",
                            ],
                            errors="coerce",
                        ).mean()
                    )
                    if valid.sum()
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def rule_activation_summary(
    df: pd.DataFrame,
    rule_cols: list[str],
) -> pd.DataFrame:
    rows = []

    groups = [
        "CONTRADICTION_BLOCKED",
        "CONTRADICTION_VALID_NOT_EXECUTED",
        "CONTRADICTION_VALID_EXECUTED",
        "NATURAL_REFERENCE_EXECUTE",
    ]

    for regime in REGIMES:
        rg = df[
            df["evaluation_regime"]
            == regime
        ]

        for group in groups:
            g = rg[
                rg["_comparison_group"]
                == group
            ]

            for rule in rule_cols:
                rows.append(
                    {
                        "regime": regime,
                        "comparison_group": group,
                        "rule": rule.replace(
                            "_rule__",
                            "",
                        ),
                        "episodes": int(
                            len(g)
                        ),
                        "trigger_count": int(
                            g[rule].sum()
                        )
                        if len(g)
                        else 0,
                        "trigger_rate": (
                            float(
                                g[rule].mean()
                            )
                            if len(g)
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def distribution_summary(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    rows = []

    groups = [
        "CONTRADICTION_BLOCKED",
        "CONTRADICTION_VALID_NOT_EXECUTED",
        "CONTRADICTION_VALID_EXECUTED",
        "NATURAL_REFERENCE_EXECUTE",
    ]

    for regime in REGIMES:
        rg = df[
            df["evaluation_regime"]
            == regime
        ]

        for group in groups:
            g = rg[
                rg["_comparison_group"]
                == group
            ]

            for var in variables:
                if var not in g.columns:
                    continue

                x = pd.to_numeric(
                    g[var],
                    errors="coerce",
                ).dropna()

                rows.append(
                    {
                        "regime": regime,
                        "comparison_group": group,
                        "variable": var,
                        "n": int(
                            len(x)
                        ),
                        "missing_n": int(
                            len(g) - len(x)
                        ),
                        "mean": (
                            float(x.mean())
                            if len(x)
                            else np.nan
                        ),
                        "std": (
                            float(x.std(ddof=1))
                            if len(x) > 1
                            else np.nan
                        ),
                        "p01": (
                            float(x.quantile(0.01))
                            if len(x)
                            else np.nan
                        ),
                        "p05": (
                            float(x.quantile(0.05))
                            if len(x)
                            else np.nan
                        ),
                        "p25": (
                            float(x.quantile(0.25))
                            if len(x)
                            else np.nan
                        ),
                        "p50": (
                            float(x.quantile(0.50))
                            if len(x)
                            else np.nan
                        ),
                        "p75": (
                            float(x.quantile(0.75))
                            if len(x)
                            else np.nan
                        ),
                        "p95": (
                            float(x.quantile(0.95))
                            if len(x)
                            else np.nan
                        ),
                        "p99": (
                            float(x.quantile(0.99))
                            if len(x)
                            else np.nan
                        ),
                        "min": (
                            float(x.min())
                            if len(x)
                            else np.nan
                        ),
                        "max": (
                            float(x.max())
                            if len(x)
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def standardized_mean_difference(
    a: pd.Series,
    b: pd.Series,
) -> float:
    a = pd.to_numeric(
        a,
        errors="coerce",
    ).dropna()

    b = pd.to_numeric(
        b,
        errors="coerce",
    ).dropna()

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = float(
        a.var(ddof=1)
    )

    vb = float(
        b.var(ddof=1)
    )

    pooled = np.sqrt(
        (va + vb)
        / 2.0
    )

    if pooled == 0:
        if float(a.mean()) == float(b.mean()):
            return 0.0
        return np.inf

    return float(
        (
            a.mean()
            - b.mean()
        )
        / pooled
    )


def empirical_ks_distance(
    a: pd.Series,
    b: pd.Series,
) -> float:
    """
    Numpy-only two-sample KS distance.
    """
    a = np.sort(
        pd.to_numeric(
            a,
            errors="coerce",
        ).dropna().to_numpy()
    )

    b = np.sort(
        pd.to_numeric(
            b,
            errors="coerce",
        ).dropna().to_numpy()
    )

    if len(a) == 0 or len(b) == 0:
        return np.nan

    values = np.sort(
        np.unique(
            np.concatenate(
                [
                    a,
                    b,
                ]
            )
        )
    )

    cdf_a = np.searchsorted(
        a,
        values,
        side="right",
    ) / len(a)

    cdf_b = np.searchsorted(
        b,
        values,
        side="right",
    ) / len(b)

    return float(
        np.max(
            np.abs(
                cdf_a
                - cdf_b
            )
        )
    )


def separability_summary(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    rows = []

    for regime in REGIMES:
        rg = df[
            df["evaluation_regime"]
            == regime
        ]

        residual = rg[
            rg["_comparison_group"]
            == "CONTRADICTION_VALID_EXECUTED"
        ]

        natural = rg[
            rg["_comparison_group"]
            == "NATURAL_REFERENCE_EXECUTE"
        ]

        for var in variables:
            if (
                var not in residual.columns
                or var not in natural.columns
            ):
                continue

            a = pd.to_numeric(
                residual[var],
                errors="coerce",
            )

            b = pd.to_numeric(
                natural[var],
                errors="coerce",
            )

            rows.append(
                {
                    "regime": regime,
                    "variable": var,
                    "residual_executed_n": int(
                        a.notna().sum()
                    ),
                    "natural_reference_execute_n": int(
                        b.notna().sum()
                    ),
                    "residual_mean": (
                        float(a.mean())
                        if a.notna().any()
                        else np.nan
                    ),
                    "natural_mean": (
                        float(b.mean())
                        if b.notna().any()
                        else np.nan
                    ),
                    "standardized_mean_difference": standardized_mean_difference(
                        a,
                        b,
                    ),
                    "absolute_standardized_mean_difference": (
                        abs(
                            standardized_mean_difference(
                                a,
                                b,
                            )
                        )
                        if pd.notna(
                            standardized_mean_difference(
                                a,
                                b,
                            )
                        )
                        else np.nan
                    ),
                    "empirical_ks_distance": empirical_ks_distance(
                        a,
                        b,
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def quantile_overlap_summary(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Diagnostic support overlap:
    fraction of residual executed contradictions whose value falls inside the
    natural reference-EXECUTE 5th-95th and 1st-99th percentile intervals.
    """
    rows = []

    for regime in REGIMES:
        rg = df[
            df["evaluation_regime"]
            == regime
        ]

        residual = rg[
            rg["_comparison_group"]
            == "CONTRADICTION_VALID_EXECUTED"
        ]

        natural = rg[
            rg["_comparison_group"]
            == "NATURAL_REFERENCE_EXECUTE"
        ]

        for var in variables:
            if (
                var not in residual.columns
                or var not in natural.columns
            ):
                continue

            r = pd.to_numeric(
                residual[var],
                errors="coerce",
            ).dropna()

            n = pd.to_numeric(
                natural[var],
                errors="coerce",
            ).dropna()

            if len(r) == 0 or len(n) == 0:
                rows.append(
                    {
                        "regime": regime,
                        "variable": var,
                        "residual_n": int(
                            len(r)
                        ),
                        "natural_n": int(
                            len(n)
                        ),
                        "natural_p01": np.nan,
                        "natural_p05": np.nan,
                        "natural_p95": np.nan,
                        "natural_p99": np.nan,
                        "residual_fraction_inside_natural_p05_p95": np.nan,
                        "residual_fraction_inside_natural_p01_p99": np.nan,
                    }
                )
                continue

            p01 = float(
                n.quantile(
                    0.01
                )
            )

            p05 = float(
                n.quantile(
                    0.05
                )
            )

            p95 = float(
                n.quantile(
                    0.95
                )
            )

            p99 = float(
                n.quantile(
                    0.99
                )
            )

            rows.append(
                {
                    "regime": regime,
                    "variable": var,
                    "residual_n": int(
                        len(r)
                    ),
                    "natural_n": int(
                        len(n)
                    ),
                    "natural_p01": p01,
                    "natural_p05": p05,
                    "natural_p95": p95,
                    "natural_p99": p99,
                    "residual_fraction_inside_natural_p05_p95": float(
                        (
                            (r >= p05)
                            & (r <= p95)
                        ).mean()
                    ),
                    "residual_fraction_inside_natural_p01_p99": float(
                        (
                            (r >= p01)
                            & (r <= p99)
                        ).mean()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def boolean_transition_summary(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    rows = []

    groups = [
        "CONTRADICTION_BLOCKED",
        "CONTRADICTION_VALID_NOT_EXECUTED",
        "CONTRADICTION_VALID_EXECUTED",
        "NATURAL_REFERENCE_EXECUTE",
    ]

    for regime in REGIMES:
        rg = df[
            df["evaluation_regime"]
            == regime
        ]

        for group in groups:
            g = rg[
                rg["_comparison_group"]
                == group
            ]

            for var in variables:
                if var not in g.columns:
                    continue

                x = as_bool(
                    g[var]
                )

                rows.append(
                    {
                        "regime": regime,
                        "comparison_group": group,
                        "variable": var,
                        "episodes": int(
                            len(g)
                        ),
                        "true_count": int(
                            x.sum()
                        )
                        if len(g)
                        else 0,
                        "true_rate": (
                            float(
                                x.mean()
                            )
                            if len(g)
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def main() -> None:
    args = parse_args()

    dataset_dir = Path(
        args.dataset_dir
    ).expanduser().resolve()

    results_dir = Path(
        args.results_dir
    ).expanduser().resolve()

    exp_path = Path(
        args.experimental
    ).expanduser().resolve()

    spec_path = Path(
        args.spec
    ).expanduser().resolve()

    for p in [
        dataset_dir,
        results_dir,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                p
            )

    if not exp_path.exists():
        raise FileNotFoundError(
            exp_path
        )

    if not spec_path.exists():
        raise FileNotFoundError(
            spec_path
        )

    out_dir = (
        Path(
            args.out
        ).expanduser().resolve()
        if args.out
        else results_dir
        / "residual_contradiction_audit"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spec = json.loads(
        spec_path.read_text(
            encoding="utf-8"
        )
    )

    delta = validate_spec(
        spec
    )

    exp = pd.read_parquet(
        exp_path
    )

    require_columns(
        exp,
        [
            "benchmark_episode_id",
            "intervention_type",
            "intervention_strength",
        ],
        "experimental benchmark",
    )

    print(
        "=" * 80
    )
    print(
        "RCSE STEP 09b.2q - RESIDUAL CONTRADICTION SEPARABILITY AUDIT"
    )
    print(
        "=" * 80
    )
    print(
        f"Dataset dir   : {dataset_dir}"
    )
    print(
        f"Results dir   : {results_dir}"
    )
    print(
        f"Experimental  : {exp_path}"
    )
    print(
        f"Frozen spec   : {spec_path}"
    )
    print(
        f"Gate delta    : {delta}"
    )
    print(
        f"Output        : {out_dir}"
    )
    print(
        "NO TRAINING / NO GATE MODIFICATION / NO RETUNING"
    )

    all_rows = []

    for regime in REGIMES:
        data_path = (
            dataset_dir
            / DATASET_FILES[
                regime
            ]
        )

        decision_path = (
            results_dir
            / DECISION_FILES[
                regime
            ]
        )

        if not data_path.exists():
            raise FileNotFoundError(
                data_path
            )

        if not decision_path.exists():
            raise FileNotFoundError(
                decision_path
            )

        df = pd.read_parquet(
            data_path
        )

        decisions = pd.read_parquet(
            decision_path
        )

        require_columns(
            df,
            [
                "benchmark_episode_id",
                "source_case_id",
                "gate_state_v2",
                "expected_action_reference",
                "p_safe_t0_cal",
                "transaction_exposure_eur",
            ],
            f"dataset/{regime}",
        )

        primary = decisions[
            decisions[
                "consequence_mode"
            ]
            == PRIMARY_CONSEQUENCE
        ][
            [
                "benchmark_episode_id",
                f"action__{PRIMARY_POLICY}",
            ]
        ].copy()

        if primary[
            "benchmark_episode_id"
        ].duplicated().any():
            raise AssertionError(
                f"{regime}: duplicate primary policy decision rows."
            )

        df = df.merge(
            primary,
            on="benchmark_episode_id",
            how="left",
            validate="one_to_one",
        )

        df[
            "rcse_v2_full_action"
        ] = normalize_action(
            df[
                f"action__{PRIMARY_POLICY}"
            ]
        )

        df = attach_experimental_fields(
            df,
            exp,
        )

        df[
            "_intervention_family_q"
        ] = intervention_family(
            df[
                "intervention_type"
            ]
        )

        df[
            "evaluation_regime"
        ] = regime

        rules = build_gate_rules(
            df,
            delta=delta,
        )

        for name, mask in rules.items():
            df[
                f"_rule__{name}"
            ] = mask

        df[
            "_rule_trigger_count"
        ] = pd.DataFrame(
            {
                name: mask
                for name, mask in rules.items()
            }
        ).sum(
            axis=1
        )

        df[
            "_triggered_rules"
        ] = pd.DataFrame(
            {
                name: mask
                for name, mask in rules.items()
            }
        ).apply(
            lambda row: "|".join(
                name
                for name, val in row.items()
                if bool(val)
            ),
            axis=1,
        )

        df[
            "_comparison_group"
        ] = classify_comparison_group(
            df
        )

        all_rows.append(
            df
        )

        diag_cols = [
            c for c in [
                "benchmark_episode_id",
                "source_case_id",
                "evaluation_regime",
                "intervention_type",
                "intervention_strength",
                "_intervention_family_q",
                "_comparison_group",
                "gate_state_v2",
                "rcse_v2_full_action",
                "expected_action_reference",
                "_rule_trigger_count",
                "_triggered_rules",
                *NUMERIC_OBSERVABLES,
                *BOOLEAN_OBSERVABLES,
                *[
                    f"_rule__{name}"
                    for name in rules
                ],
            ]
            if c in df.columns
        ]

        df[
            diag_cols
        ].to_parquet(
            out_dir
            / f"rcse_v2q_episode_diagnostics_{regime}.parquet",
            index=False,
        )

    full = pd.concat(
        all_rows,
        ignore_index=True,
    )

    rule_cols = [
        c for c in full.columns
        if c.startswith(
            "_rule__"
        )
    ]

    numeric_vars = [
        c for c in NUMERIC_OBSERVABLES
        if c in full.columns
    ]

    bool_vars = [
        c for c in BOOLEAN_OBSERVABLES
        if c in full.columns
    ]

    flow = flow_summary(
        full
    )

    type_strength = type_strength_summary(
        full
    )

    rule_activation = rule_activation_summary(
        full,
        rule_cols=rule_cols,
    )

    residual_valid_rule_profile = rule_activation[
        rule_activation[
            "comparison_group"
        ]
        == "CONTRADICTION_VALID_NOT_EXECUTED"
    ].copy()

    residual_executed_rule_profile = rule_activation[
        rule_activation[
            "comparison_group"
        ]
        == "CONTRADICTION_VALID_EXECUTED"
    ].copy()

    distribution = distribution_summary(
        full,
        variables=numeric_vars,
    )

    separability = separability_summary(
        full,
        variables=numeric_vars,
    )

    overlap = quantile_overlap_summary(
        full,
        variables=numeric_vars,
    )

    bool_summary = boolean_transition_summary(
        full,
        variables=bool_vars,
    )

    flow.to_csv(
        out_dir
        / "rcse_v2q_contradiction_flow.csv",
        index=False,
    )

    type_strength.to_csv(
        out_dir
        / "rcse_v2q_contradiction_by_type_strength.csv",
        index=False,
    )

    rule_activation.to_csv(
        out_dir
        / "rcse_v2q_gate_rule_activation.csv",
        index=False,
    )

    residual_valid_rule_profile.to_csv(
        out_dir
        / "rcse_v2q_residual_valid_rule_profile.csv",
        index=False,
    )

    residual_executed_rule_profile.to_csv(
        out_dir
        / "rcse_v2q_residual_executed_rule_profile.csv",
        index=False,
    )

    distribution.to_csv(
        out_dir
        / "rcse_v2q_observable_distribution_summary.csv",
        index=False,
    )

    separability.to_csv(
        out_dir
        / "rcse_v2q_residual_vs_natural_separability.csv",
        index=False,
    )

    overlap.to_csv(
        out_dir
        / "rcse_v2q_residual_vs_natural_quantile_overlap.csv",
        index=False,
    )

    bool_summary.to_csv(
        out_dir
        / "rcse_v2q_boolean_observable_summary.csv",
        index=False,
    )

    metadata: dict[str, Any] = {
        "step": "09b.2q",
        "policy": PRIMARY_POLICY,
        "consequence_mode": PRIMARY_CONSEQUENCE,
        "gate_delta": delta,
        "analysis_type": "diagnostic_only",
        "training_performed": False,
        "recalibration_performed": False,
        "gate_modified": False,
        "policy_retuned": False,
        "frontier_selected_point_used": False,
        "comparison_groups": [
            "CONTRADICTION_BLOCKED",
            "CONTRADICTION_VALID_NOT_EXECUTED",
            "CONTRADICTION_VALID_EXECUTED",
            "NATURAL_REFERENCE_EXECUTE",
        ],
        "separability_metrics": [
            "standardized_mean_difference",
            "absolute_standardized_mean_difference",
            "empirical_ks_distance",
            "residual fraction inside natural 5th-95th percentile interval",
            "residual fraction inside natural 1st-99th percentile interval",
        ],
        "important_guardrail": (
            "Large distributional separation does not automatically justify a "
            "new gate rule. Any later rule must have independent ERP/process "
            "semantics and must not be chosen solely to remove test errors."
        ),
        "next_decision": (
            "Determine whether residual executed contradictions are observably "
            "distinct from legitimate natural reference-EXECUTE cases. If yes, "
            "consider a semantically justified gate revision in a separately "
            "declared secondary analysis. If no, report an observability limit "
            "and proceed to statistical uncertainty analysis."
        ),
    }

    with open(
        out_dir
        / "rcse_v2q_metadata.json",
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
        "STEP 09b.2q COMPLETE"
    )
    print(
        "=" * 80
    )

    print(
        "\nContradiction flow:"
    )
    print(
        flow.to_string(
            index=False
        )
    )

    print(
        "\nContradiction by type / strength:"
    )
    print(
        type_strength.to_string(
            index=False
        )
    )

    print(
        "\nTop residual-vs-natural separability variables "
        "(sorted by |standardized mean difference|):"
    )

    for regime in REGIMES:
        sub = (
            separability[
                separability[
                    "regime"
                ]
                == regime
            ]
            .sort_values(
                "absolute_standardized_mean_difference",
                ascending=False,
                na_position="last",
            )
            .head(
                12
            )
        )

        print(
            f"\n[{regime}]"
        )

        print(
            sub[
                [
                    "variable",
                    "residual_executed_n",
                    "natural_reference_execute_n",
                    "residual_mean",
                    "natural_mean",
                    "standardized_mean_difference",
                    "empirical_ks_distance",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nResidual contradiction overlap with natural reference-EXECUTE "
        "support (key value relationships):"
    )

    key_vars = [
        "inv_po_abs_rel_diff",
        "inv_gr_abs_rel_diff",
        "po_value_initial_eur",
        "invoice_value_t0_eur",
        "p_safe_t0_cal",
        "transaction_exposure_eur",
    ]

    key_overlap = overlap[
        overlap[
            "variable"
        ].isin(
            key_vars
        )
    ]

    print(
        key_overlap.to_string(
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
        "\nNext: decide whether the dominant residual contradictions are "
        "semantically detectable from available t0 evidence or represent an "
        "observability boundary. Do not modify the gate from this script."
    )


if __name__ == "__main__":
    main()
