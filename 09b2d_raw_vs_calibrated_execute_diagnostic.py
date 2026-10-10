#!/usr/bin/env python
"""
09b2d_raw_vs_calibrated_execute_diagnostic.py

RCSE Step 09b.2d
Compare raw vs isotonic-calibrated t0 EXECUTE scores without retraining.

Purpose
-------
Determine whether the weak direct-execution signal observed in Step 09b.2c
is primarily caused by:
  A) isotonic calibration compressing/damaging useful raw EXECUTE ranking, or
  B) weak EXECUTE-vs-NOT-EXECUTE discrimination already present in the raw model.

This script DOES NOT:
- retrain the model,
- recalibrate the model,
- alter the frozen RCSE policy,
- choose new thresholds from test performance.

Inputs
------
Frozen Step-08c prediction artifacts for:
  grouped_iid
  temporal
  ood_exposure

Expected probability columns:
  p_raw_execute
  p_cal_execute

Outputs
-------
execute_raw_vs_calibrated_metrics.csv
execute_raw_vs_calibrated_quantiles.csv
execute_probability_compression.csv
execute_rank_agreement.csv
execute_unique_value_summary.csv
execute_raw_vs_calibrated_thresholds.csv
execute_raw_vs_calibrated_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

REGIMES = ("grouped_iid", "temporal", "ood_exposure")

RAW_CANDIDATES = (
    "p_raw_execute",
    "p_execute_raw",
    "raw_probability_execute",
    "prob_execute_raw",
)

CAL_CANDIDATES = (
    "p_cal_execute",
    "p_execute_calibrated",
    "calibrated_probability_execute",
    "prob_execute_calibrated",
)

Y_CANDIDATES = (
    "expected_action_reference",
    "reference_action",
    "y_true",
    "actual_action",
    "target_action",
)

DEFAULT_THRESHOLDS = [
    0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 09b.2d: raw-vs-calibrated EXECUTE discrimination diagnostic."
    )
    p.add_argument(
        "--prediction-dir",
        type=Path,
        required=True,
        help="Directory containing frozen Step-08c prediction files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <prediction-dir>/execute_raw_vs_calibrated_diagnostic",
    )
    return p.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file: {path}")


def first_existing(columns: Iterable[str], candidates: Iterable[str]):
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def find_prediction_file(folder: Path, regime: str) -> Path:
    preferred = [
        folder / f"frozen_predictions_{regime}.parquet",
        folder / f"frozen_baseline_predictions_{regime}.parquet",
        folder / f"baseline_predictions_{regime}.parquet",
        folder / f"frozen_predictions_{regime}.csv",
        folder / f"frozen_baseline_predictions_{regime}.csv",
        folder / f"baseline_predictions_{regime}.csv",
    ]
    for p in preferred:
        if p.exists():
            return p

    matches = []
    for pat in ("*.parquet", "*.csv"):
        for p in folder.rglob(pat):
            n = p.name.lower()
            if regime in n and "prediction" in n and "post_gather" not in n and "t1" not in n:
                matches.append(p)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No frozen t0 prediction artifact found for regime={regime}"
        )
    raise RuntimeError(
        f"Multiple candidate prediction files for {regime}:\n  - "
        + "\n  - ".join(map(str, matches))
    )


def normalize_action(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def qsummary(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    qs = x.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1.0])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "std": float(x.std()) if len(x) > 1 else np.nan,
        "min": float(qs.loc[0]) if len(x) else np.nan,
        "p01": float(qs.loc[.01]) if len(x) else np.nan,
        "p05": float(qs.loc[.05]) if len(x) else np.nan,
        "p10": float(qs.loc[.10]) if len(x) else np.nan,
        "p25": float(qs.loc[.25]) if len(x) else np.nan,
        "p50": float(qs.loc[.50]) if len(x) else np.nan,
        "p75": float(qs.loc[.75]) if len(x) else np.nan,
        "p90": float(qs.loc[.90]) if len(x) else np.nan,
        "p95": float(qs.loc[.95]) if len(x) else np.nan,
        "p99": float(qs.loc[.99]) if len(x) else np.nan,
        "max": float(qs.loc[1.0]) if len(x) else np.nan,
    }


def metrics_for_score(y: np.ndarray, p: np.ndarray) -> dict:
    prevalence = float(y.mean())
    if len(np.unique(y)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))

    return {
        "execute_prevalence": prevalence,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_baseline": prevalence,
        "auprc_lift": (auprc / prevalence) if prevalence > 0 else np.nan,
        "mean_score": float(np.mean(p)),
        "max_score": float(np.max(p)),
        "fraction_ge_0_50": float(np.mean(p >= 0.50)),
        "fraction_ge_0_20": float(np.mean(p >= 0.20)),
    }


def threshold_rows(regime: str, y: np.ndarray, p: np.ndarray, score_type: str):
    rows = []
    positives = int(y.sum())

    for t in DEFAULT_THRESHOLDS:
        sel = p >= t
        n_sel = int(sel.sum())
        tp = int(((y == 1) & sel).sum())
        fp = int(((y == 0) & sel).sum())

        rows.append({
            "regime": regime,
            "score_type": score_type,
            "threshold": t,
            "selected_n": n_sel,
            "coverage": n_sel / len(y),
            "precision_execute": (tp / n_sel) if n_sel else np.nan,
            "unsafe_rate_if_execute": (fp / n_sel) if n_sel else np.nan,
            "recall_execute": (tp / positives) if positives else np.nan,
        })

    return rows


def main():
    args = parse_args()
    pred_dir = args.prediction_dir.resolve()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (pred_dir / "execute_raw_vs_calibrated_diagnostic").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 79)
    print("RCSE STEP 09b.2d - RAW VS CALIBRATED EXECUTE DIAGNOSTIC")
    print("=" * 79)
    print(f"Prediction dir : {pred_dir}")
    print(f"Output dir     : {out_dir}")
    print("NO RETRAINING / NO RECALIBRATION / NO POLICY RETUNING")
    print()

    metric_rows = []
    quantile_rows = []
    compression_rows = []
    rank_rows = []
    unique_rows = []
    threshold_out = []
    manifest = {}

    for regime in REGIMES:
        path = find_prediction_file(pred_dir, regime)
        manifest[regime] = str(path)
        print(f"[{regime}] {path.name}")

        df = read_table(path)

        raw_col = first_existing(df.columns, RAW_CANDIDATES)
        cal_col = first_existing(df.columns, CAL_CANDIDATES)
        y_col = first_existing(df.columns, Y_CANDIDATES)

        if raw_col is None:
            execute_like = [
                c for c in df.columns
                if "execute" in c.lower()
            ]
            raise KeyError(
                f"[{regime}] raw EXECUTE probability column not found.\n"
                f"EXECUTE-like columns: {execute_like}"
            )
        if cal_col is None:
            raise KeyError(f"[{regime}] calibrated EXECUTE probability column not found.")
        if y_col is None:
            raise KeyError(f"[{regime}] reference action column not found.")

        work = df[[raw_col, cal_col, y_col]].copy()
        work["_raw"] = pd.to_numeric(work[raw_col], errors="coerce")
        work["_cal"] = pd.to_numeric(work[cal_col], errors="coerce")
        work["_y"] = (normalize_action(work[y_col]) == "EXECUTE").astype(int)

        before = len(work)
        work = work.dropna(subset=["_raw", "_cal"])
        dropped = before - len(work)

        if dropped:
            print(f"   WARNING: dropped {dropped:,} rows with missing raw/cal score")

        for c in ("_raw", "_cal"):
            if ((work[c] < -1e-9) | (work[c] > 1 + 1e-9)).any():
                raise ValueError(f"[{regime}] {c} has values outside [0,1].")

        y = work["_y"].to_numpy(dtype=int)
        raw = work["_raw"].to_numpy(dtype=float)
        cal = work["_cal"].to_numpy(dtype=float)

        # Discrimination comparison.
        raw_m = metrics_for_score(y, raw)
        cal_m = metrics_for_score(y, cal)

        for score_type, m in [("RAW", raw_m), ("CALIBRATED", cal_m)]:
            metric_rows.append({
                "regime": regime,
                "score_type": score_type,
                "n": len(work),
                "true_execute_n": int(y.sum()),
                "true_not_execute_n": int((1-y).sum()),
                **m,
            })

        # Quantiles by class.
        for label, g in work.groupby("_y"):
            class_name = "EXECUTE" if label == 1 else "NOT_EXECUTE"
            for score_type, col in [("RAW", "_raw"), ("CALIBRATED", "_cal")]:
                row = {
                    "regime": regime,
                    "class": class_name,
                    "score_type": score_type,
                }
                row.update(qsummary(g[col]))
                quantile_rows.append(row)

        # Compression / unique value analysis.
        raw_unique = int(pd.Series(raw).nunique(dropna=True))
        cal_unique = int(pd.Series(cal).nunique(dropna=True))

        unique_rows.append({
            "regime": regime,
            "n": len(work),
            "raw_unique_values": raw_unique,
            "calibrated_unique_values": cal_unique,
            "raw_unique_fraction": raw_unique / len(work),
            "calibrated_unique_fraction": cal_unique / len(work),
            "compression_ratio_cal_to_raw_unique": (
                cal_unique / raw_unique if raw_unique else np.nan
            ),
        })

        # How many raw values map to same calibrated output.
        mapping = (
            work.groupby("_cal", dropna=False)
            .agg(
                n_rows=("_raw", "size"),
                raw_min=("_raw", "min"),
                raw_max=("_raw", "max"),
                raw_unique=("_raw", "nunique"),
                execute_rate=("_y", "mean"),
            )
            .reset_index()
            .sort_values("n_rows", ascending=False)
        )
        mapping.insert(0, "regime", regime)
        mapping.to_csv(
            out_dir / f"execute_isotonic_mapping_{regime}.csv",
            index=False,
        )

        compression_rows.append({
            "regime": regime,
            "largest_calibrated_plateau_rows": int(mapping["n_rows"].max()),
            "largest_calibrated_plateau_fraction": float(mapping["n_rows"].max() / len(work)),
            "median_rows_per_calibrated_value": float(mapping["n_rows"].median()),
            "max_raw_unique_mapped_to_one_cal_value": int(mapping["raw_unique"].max()),
        })

        # Rank agreement.
        pearson = float(pd.Series(raw).corr(pd.Series(cal), method="pearson"))
        spearman = float(pd.Series(raw).corr(pd.Series(cal), method="spearman"))

        raw_rank = pd.Series(raw).rank(method="average", pct=True)
        cal_rank = pd.Series(cal).rank(method="average", pct=True)
        rank_abs_diff = (raw_rank - cal_rank).abs()

        rank_rows.append({
            "regime": regime,
            "pearson_raw_vs_cal": pearson,
            "spearman_raw_vs_cal": spearman,
            "mean_abs_percentile_rank_change": float(rank_abs_diff.mean()),
            "p90_abs_percentile_rank_change": float(rank_abs_diff.quantile(.90)),
            "p99_abs_percentile_rank_change": float(rank_abs_diff.quantile(.99)),
        })

        threshold_out.extend(threshold_rows(regime, y, raw, "RAW"))
        threshold_out.extend(threshold_rows(regime, y, cal, "CALIBRATED"))

        print(
            f"   RAW: AUROC={raw_m['auroc']:.4f} AUPRC={raw_m['auprc']:.4f} "
            f"max={raw_m['max_score']:.4f} unique={raw_unique:,}"
        )
        print(
            f"   CAL: AUROC={cal_m['auroc']:.4f} AUPRC={cal_m['auprc']:.4f} "
            f"max={cal_m['max_score']:.4f} unique={cal_unique:,}"
        )
        print(
            f"   Spearman(raw,cal)={spearman:.4f} | "
            f"largest calibrated plateau={mapping['n_rows'].max():,} rows"
        )

    metrics_df = pd.DataFrame(metric_rows)
    quantiles_df = pd.DataFrame(quantile_rows)
    compression_df = pd.DataFrame(compression_rows)
    rank_df = pd.DataFrame(rank_rows)
    unique_df = pd.DataFrame(unique_rows)
    threshold_df = pd.DataFrame(threshold_out)

    metrics_df.to_csv(out_dir / "execute_raw_vs_calibrated_metrics.csv", index=False)
    quantiles_df.to_csv(out_dir / "execute_raw_vs_calibrated_quantiles.csv", index=False)
    compression_df.to_csv(out_dir / "execute_probability_compression.csv", index=False)
    rank_df.to_csv(out_dir / "execute_rank_agreement.csv", index=False)
    unique_df.to_csv(out_dir / "execute_unique_value_summary.csv", index=False)
    threshold_df.to_csv(out_dir / "execute_raw_vs_calibrated_thresholds.csv", index=False)

    # Regime-level delta table for immediate interpretation.
    piv = metrics_df.pivot(
        index="regime",
        columns="score_type",
        values=["auroc", "auprc", "max_score", "fraction_ge_0_50"],
    )
    delta_rows = []
    for regime in REGIMES:
        delta_rows.append({
            "regime": regime,
            "auroc_raw": float(piv.loc[regime, ("auroc", "RAW")]),
            "auroc_cal": float(piv.loc[regime, ("auroc", "CALIBRATED")]),
            "auroc_delta_raw_minus_cal": float(
                piv.loc[regime, ("auroc", "RAW")] - piv.loc[regime, ("auroc", "CALIBRATED")]
            ),
            "auprc_raw": float(piv.loc[regime, ("auprc", "RAW")]),
            "auprc_cal": float(piv.loc[regime, ("auprc", "CALIBRATED")]),
            "auprc_delta_raw_minus_cal": float(
                piv.loc[regime, ("auprc", "RAW")] - piv.loc[regime, ("auprc", "CALIBRATED")]
            ),
        })
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(out_dir / "execute_raw_vs_calibrated_deltas.csv", index=False)

    metadata = {
        "step": "09b.2d",
        "purpose": "Compare frozen raw and isotonic-calibrated EXECUTE discrimination.",
        "retraining_performed": False,
        "recalibration_performed": False,
        "policy_retuning_performed": False,
        "prediction_files": manifest,
        "raw_candidates": list(RAW_CANDIDATES),
        "calibrated_candidates": list(CAL_CANDIDATES),
        "interpretation": {
            "calibration_damage_signal": (
                "Raw AUROC/AUPRC materially exceed calibrated AUROC/AUPRC, "
                "with substantial unique-value compression or rank disruption."
            ),
            "weak_base_signal": (
                "Raw AUROC/AUPRC are also weak and close to calibrated performance."
            ),
            "important_note": (
                "Isotonic calibration can create probability plateaus while preserving "
                "much of the ranking; unique-value compression alone is not evidence "
                "that calibration caused the weak EXECUTE discrimination."
            ),
        },
    }

    (out_dir / "execute_raw_vs_calibrated_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 79)
    print("STEP 09b.2d COMPLETE")
    print("=" * 79)
    print("\nRaw vs calibrated discrimination deltas:")
    print(delta_df.to_string(index=False))

    print("\nUnique-value / compression summary:")
    print(unique_df.to_string(index=False))

    print("\nRank agreement:")
    print(rank_df.to_string(index=False))

    print("\nOutputs:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  - {p}")

    print("\nDecision rule for the next step:")
    print("  If RAW materially outperforms CAL -> investigate calibration design using calibration data only.")
    print("  If RAW is also weak -> dedicated t0 execution-safety estimator becomes methodologically justified.")


if __name__ == "__main__":
    main()
