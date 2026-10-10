#!/usr/bin/env python
"""
09b2c_execute_probability_diagnostic.py

RCSE Step 09b.2c
Frozen t0 EXECUTE-probability diagnostic.

Purpose
-------
Diagnose whether the near-absence of direct autonomous execution in Step 09b.2b
is primarily caused by:
  A) useful EXECUTE-vs-NOT-EXECUTE ranking but low absolute p(EXECUTE), or
  B) weak discrimination between EXECUTE and NOT-EXECUTE.

IMPORTANT:
- This script DOES NOT retrain, recalibrate, or retune the frozen model.
- It only analyzes existing frozen Step-08c prediction artifacts.
- All thresholds are descriptive diagnostics, not policy tuning.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

REGIMES = ("grouped_iid", "temporal", "ood_exposure")
DEFAULT_THRESHOLDS = [
    0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 0.95, 0.99
]

P_EXECUTE_CANDIDATES = (
    "p_cal_execute",
    "p_execute_calibrated",
    "calibrated_probability_execute",
    "prob_execute_calibrated",
    "prob_execute",
    "p_execute",
)

Y_CANDIDATES = (
    "expected_action_reference",
    "reference_action",
    "y_true",
    "actual_action",
    "target_action",
)

PRED_CANDIDATES = (
    "predicted_argmax_action",
    "argmax_action",
    "predicted_action",
    "y_pred",
)

ID_CANDIDATES = ("benchmark_episode_id", "episode_id")
SOURCE_ID_CANDIDATES = ("source_case_id", "case_id")
SYNTH_CANDIDATES = (
    "is_synthetic_intervention",
    "synthetic_intervention",
    "is_synthetic",
)
ARM_CANDIDATES = ("benchmark_arm",)
INTERVENTION_CANDIDATES = ("intervention_type", "intervention_family")
RISK_CANDIDATES = ("transaction_risk_stratum", "risk_stratum")
EXPOSURE_CANDIDATES = ("transaction_exposure_eur", "exposure_eur")


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 09b.2c: diagnose frozen t0 calibrated EXECUTE probability."
    )
    p.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing frozen Step-08c prediction parquet/csv files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <prediction-dir>/execute_probability_diagnostic",
    )
    p.add_argument(
        "--split-benchmark",
        type=Path,
        default=None,
        help=(
            "Optional split benchmark parquet/csv. Used only to attach missing subgroup "
            "fields by benchmark_episode_id; no model fitting occurs."
        ),
    )
    return p.parse_args()


def first_existing(columns: Iterable[str], candidates: Iterable[str]):
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def find_prediction_file(folder: Path, regime: str) -> Path:
    exact = [
        folder / f"frozen_baseline_predictions_{regime}.parquet",
        folder / f"baseline_predictions_{regime}.parquet",
        folder / f"predictions_{regime}.parquet",
        folder / f"frozen_predictions_{regime}.parquet",
        folder / f"frozen_baseline_predictions_{regime}.csv",
        folder / f"baseline_predictions_{regime}.csv",
        folder / f"predictions_{regime}.csv",
    ]
    for f in exact:
        if f.exists():
            return f

    matches = []
    for ext in ("*.parquet", "*.csv"):
        for f in folder.rglob(ext):
            name = f.name.lower()
            if regime in name and "prediction" in name:
                # Exclude post-GATHER artifacts: Step 09b.2c is t0 only.
                if "post_gather" not in name and "t1" not in name:
                    matches.append(f)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No t0 prediction file found for regime '{regime}' under {folder}"
        )
    raise RuntimeError(
        f"Multiple candidate prediction files found for {regime}:\n  - "
        + "\n  - ".join(str(x) for x in matches)
        + "\nSpecify a directory containing only the frozen Step-08c prediction artifacts."
    )


def normalize_action(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def boolish_to_label(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.map({True: "SYNTHETIC", False: "NATURAL"}).astype("string")
    z = s.astype("string").str.strip().str.lower()
    return np.where(
        z.isin(["true", "1", "yes", "y", "synthetic"]),
        "SYNTHETIC",
        np.where(z.isin(["false", "0", "no", "n", "natural"]), "NATURAL", s.astype("string")),
    )


def quantile_row(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    q = x.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "std": float(x.std()) if len(x) > 1 else np.nan,
        "min": float(q.loc[0]) if len(x) else np.nan,
        "p01": float(q.loc[.01]) if len(x) else np.nan,
        "p05": float(q.loc[.05]) if len(x) else np.nan,
        "p10": float(q.loc[.10]) if len(x) else np.nan,
        "p25": float(q.loc[.25]) if len(x) else np.nan,
        "p50": float(q.loc[.50]) if len(x) else np.nan,
        "p75": float(q.loc[.75]) if len(x) else np.nan,
        "p90": float(q.loc[.90]) if len(x) else np.nan,
        "p95": float(q.loc[.95]) if len(x) else np.nan,
        "p99": float(q.loc[.99]) if len(x) else np.nan,
        "max": float(q.loc[1]) if len(x) else np.nan,
    }


def summarize_by(df: pd.DataFrame, regime: str, group_col: str) -> list[dict]:
    out = []
    if group_col not in df.columns:
        return out
    for key, g in df.groupby(group_col, dropna=False, observed=True):
        row = {"regime": regime, "grouping": group_col, "group_value": str(key)}
        row.update(quantile_row(g["_p_execute"]))
        row["execute_prevalence"] = float(g["_is_execute"].mean())
        out.append(row)
    return out


def threshold_diagnostics(df: pd.DataFrame, regime: str) -> list[dict]:
    y = df["_is_execute"].to_numpy(dtype=int)
    p = df["_p_execute"].to_numpy(dtype=float)
    rows = []
    for t in DEFAULT_THRESHOLDS:
        selected = p >= t
        n_sel = int(selected.sum())
        tp = int(((y == 1) & selected).sum())
        fp = int(((y == 0) & selected).sum())
        total_pos = int((y == 1).sum())
        precision = tp / n_sel if n_sel else np.nan
        recall = tp / total_pos if total_pos else np.nan
        rows.append({
            "regime": regime,
            "threshold": t,
            "selected_n": n_sel,
            "coverage_all_episodes": n_sel / len(df),
            "precision_execute_among_selected": precision,
            "unsafe_rate_if_selected_as_execute": (fp / n_sel) if n_sel else np.nan,
            "recall_of_true_execute": recall,
            "true_execute_selected": tp,
            "non_execute_selected": fp,
        })
    return rows


def discrimination_metrics(df: pd.DataFrame, regime: str) -> dict:
    y = df["_is_execute"].to_numpy(dtype=int)
    p = df["_p_execute"].to_numpy(dtype=float)
    prevalence = float(y.mean())

    if len(np.unique(y)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))

    return {
        "regime": regime,
        "n": int(len(df)),
        "true_execute_n": int(y.sum()),
        "true_not_execute_n": int((1-y).sum()),
        "execute_prevalence": prevalence,
        "auroc_execute_vs_not": auroc,
        "auprc_execute_vs_not": auprc,
        "auprc_prevalence_baseline": prevalence,
        "auprc_lift_over_prevalence": (auprc / prevalence) if prevalence > 0 and not math.isnan(auprc) else np.nan,
        "fraction_p_execute_ge_0_50": float((p >= .50).mean()),
        "fraction_p_execute_ge_0_20": float((p >= .20).mean()),
        "max_p_execute": float(np.max(p)),
        "mean_p_execute": float(np.mean(p)),
    }


def curve_tables(df: pd.DataFrame, regime: str):
    y = df["_is_execute"].to_numpy(dtype=int)
    p = df["_p_execute"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), pd.DataFrame()

    fpr, tpr, rt = roc_curve(y, p)
    roc_df = pd.DataFrame({
        "regime": regime, "threshold": rt, "fpr": fpr, "tpr": tpr
    })

    precision, recall, pt = precision_recall_curve(y, p)
    # sklearn returns one fewer threshold than precision/recall.
    pr_df = pd.DataFrame({
        "regime": regime,
        "threshold": np.r_[pt, np.nan],
        "precision": precision,
        "recall": recall,
    })
    return roc_df, pr_df


def attach_missing_subgroups(df: pd.DataFrame, benchmark: pd.DataFrame | None):
    if benchmark is None:
        return df

    left_id = first_existing(df.columns, ID_CANDIDATES)
    right_id = first_existing(benchmark.columns, ID_CANDIDATES)
    if left_id is None or right_id is None:
        return df

    desired = []
    candidate_sets = [
        SOURCE_ID_CANDIDATES, SYNTH_CANDIDATES, ARM_CANDIDATES,
        INTERVENTION_CANDIDATES, RISK_CANDIDATES, EXPOSURE_CANDIDATES,
    ]
    for cs in candidate_sets:
        c = first_existing(benchmark.columns, cs)
        if c and c not in df.columns:
            desired.append(c)

    if not desired:
        return df

    b = benchmark[[right_id] + desired].drop_duplicates(right_id)
    if right_id != left_id:
        b = b.rename(columns={right_id: left_id})
    return df.merge(b, on=left_id, how="left", validate="m:1")


def main():
    args = parse_args()
    pred_dir = args.prediction_dir.resolve()
    out_dir = (args.output_dir or (pred_dir / "execute_probability_diagnostic")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = None
    if args.split_benchmark:
        print(f"Loading optional split benchmark: {args.split_benchmark}")
        benchmark = read_table(args.split_benchmark.resolve())

    print("=" * 79)
    print("RCSE STEP 09b.2c - FROZEN EXECUTE PROBABILITY DIAGNOSTIC")
    print("=" * 79)
    print(f"Prediction dir : {pred_dir}")
    print(f"Output dir     : {out_dir}")
    print("NO RETRAINING / NO RECALIBRATION / NO POLICY RETUNING")
    print()

    metric_rows = []
    quantile_rows = []
    subgroup_rows = []
    threshold_rows = []
    roc_parts = []
    pr_parts = []
    file_manifest = {}

    for regime in REGIMES:
        path = find_prediction_file(pred_dir, regime)
        file_manifest[regime] = str(path)
        print(f"[{regime}] {path.name}")
        df = read_table(path)
        df = attach_missing_subgroups(df, benchmark)

        pcol = first_existing(df.columns, P_EXECUTE_CANDIDATES)
        ycol = first_existing(df.columns, Y_CANDIDATES)
        predcol = first_existing(df.columns, PRED_CANDIDATES)

        if pcol is None:
            possible = [c for c in df.columns if "execute" in c.lower() and ("prob" in c.lower() or c.lower().startswith("p_"))]
            raise KeyError(
                f"[{regime}] Could not identify calibrated EXECUTE probability column.\n"
                f"EXECUTE-like columns found: {possible}"
            )
        if ycol is None:
            raise KeyError(
                f"[{regime}] Could not identify reference-action column. "
                f"Expected one of: {Y_CANDIDATES}"
            )

        df = df.copy()
        df["_p_execute"] = pd.to_numeric(df[pcol], errors="coerce")
        df["_y_action"] = normalize_action(df[ycol])
        df["_is_execute"] = (df["_y_action"] == "EXECUTE").astype(int)

        bad = df["_p_execute"].isna()
        if bad.any():
            print(f"   WARNING: dropping {int(bad.sum()):,} rows with missing/non-numeric {pcol}")
            df = df.loc[~bad].copy()

        if ((df["_p_execute"] < -1e-9) | (df["_p_execute"] > 1 + 1e-9)).any():
            raise ValueError(f"[{regime}] {pcol} contains values outside [0,1].")

        metric_rows.append(discrimination_metrics(df, regime))

        # Core true EXECUTE vs NOT-EXECUTE probability distributions.
        for label, g in df.groupby("_is_execute"):
            row = {
                "regime": regime,
                "class": "EXECUTE" if label == 1 else "NOT_EXECUTE",
            }
            row.update(quantile_row(g["_p_execute"]))
            quantile_rows.append(row)

        # Requested subgroup diagnostics.
        group_cols = []

        synth = first_existing(df.columns, SYNTH_CANDIDATES)
        if synth:
            df["_natural_vs_synthetic"] = boolish_to_label(df[synth])
            group_cols.append("_natural_vs_synthetic")

        for cs in (ARM_CANDIDATES, INTERVENTION_CANDIDATES, RISK_CANDIDATES):
            c = first_existing(df.columns, cs)
            if c:
                group_cols.append(c)

        if predcol:
            df["_predicted_action"] = normalize_action(df[predcol])
            group_cols.append("_predicted_action")
            df["_prediction_correct"] = np.where(
                df["_predicted_action"] == df["_y_action"], "CORRECT", "INCORRECT"
            )
            group_cols.append("_prediction_correct")

        # Exposure: report both risk stratum (above) and TRAIN-independent descriptive
        # within-file quantile bins. These bins are diagnostic only and never fed to policy.
        expcol = first_existing(df.columns, EXPOSURE_CANDIDATES)
        if expcol:
            exp = pd.to_numeric(df[expcol], errors="coerce")
            try:
                df["_exposure_quantile_bin"] = pd.qcut(
                    exp, q=[0, .50, .90, .99, 1.0],
                    labels=["Q0-50", "Q50-90", "Q90-99", "Q99-100"],
                    duplicates="drop",
                )
                group_cols.append("_exposure_quantile_bin")
            except ValueError:
                pass

        # Always summarize by reference action.
        group_cols.insert(0, "_y_action")

        for gc in group_cols:
            subgroup_rows.extend(summarize_by(df, regime, gc))

        threshold_rows.extend(threshold_diagnostics(df, regime))
        roc_df, pr_df = curve_tables(df, regime)
        if not roc_df.empty:
            roc_parts.append(roc_df)
        if not pr_df.empty:
            pr_parts.append(pr_df)

        print(
            f"   n={len(df):,} | EXECUTE prevalence={df['_is_execute'].mean():.4f} | "
            f"max pE={df['_p_execute'].max():.4f} | "
            f"pE>=0.50={(df['_p_execute'] >= .50).mean():.6f}"
        )

    metrics = pd.DataFrame(metric_rows)
    quantiles = pd.DataFrame(quantile_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    thresholds = pd.DataFrame(threshold_rows)
    roc_all = pd.concat(roc_parts, ignore_index=True) if roc_parts else pd.DataFrame()
    pr_all = pd.concat(pr_parts, ignore_index=True) if pr_parts else pd.DataFrame()

    metrics.to_csv(out_dir / "execute_binary_discrimination.csv", index=False)
    quantiles.to_csv(out_dir / "execute_probability_quantiles.csv", index=False)
    subgroups.to_csv(out_dir / "execute_probability_subgroups.csv", index=False)
    thresholds.to_csv(out_dir / "execute_descriptive_thresholds.csv", index=False)
    if not roc_all.empty:
        roc_all.to_csv(out_dir / "execute_roc_curve.csv", index=False)
    if not pr_all.empty:
        pr_all.to_csv(out_dir / "execute_precision_recall_curve.csv", index=False)

    metadata = {
        "step": "09b.2c",
        "purpose": "Frozen t0 EXECUTE probability diagnostic",
        "retraining_performed": False,
        "recalibration_performed": False,
        "policy_retuning_performed": False,
        "regimes": list(REGIMES),
        "prediction_files": file_manifest,
        "descriptive_thresholds": DEFAULT_THRESHOLDS,
        "interpretation_rule": {
            "possibility_A": "Strong binary ranking but low absolute p(EXECUTE): investigate multiclass/calibration probability scale.",
            "possibility_B": "Weak binary AUROC/AUPRC and overlapping distributions: frozen predictor lacks a strong execution-safety signal.",
            "warning": "Do not choose a new policy threshold from these test diagnostics."
        }
    }
    (out_dir / "execute_probability_diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 79)
    print("STEP 09b.2c COMPLETE")
    print("=" * 79)
    print("\nBinary EXECUTE-vs-NOT-EXECUTE discrimination:")
    print(metrics.to_string(index=False))

    print("\nCore p(EXECUTE) quantiles:")
    show = quantiles[["regime", "class", "n", "mean", "p50", "p90", "p95", "p99", "max"]]
    print(show.to_string(index=False))

    print("\nOutputs:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            print(f"  - {f}")

    print("\nInterpretation:")
    print("  A) Good AUROC/AUPRC lift + low absolute pE -> ranking may be useful but probability scale is suppressed.")
    print("  B) Weak AUROC/AUPRC + overlapping EXECUTE/NOT_EXECUTE quantiles -> weak execution discrimination.")
    print("  Do not retune the policy from these test-set diagnostics.")


if __name__ == "__main__":
    main()
