# Reproducing the RCSE experiments

## 1. What this repository supplies

The repository includes analysis source, frozen derived datasets, saved models, predictions, split metadata, numerical results, figures, and additional validation experiments. It excludes manuscripts, reviewer comments/responses, document-generation programs, and the raw XES log. Large duplicate CSV dataset exports are omitted; their Parquet counterparts are included.

There are three levels of reproduction:

| Level | Inputs | Action |
| --- | --- | --- |
| Inspect and validate saved evidence | Included CSV/Parquet/JSON files | Run the verification utility and inspect results |
| Recompute additional experiments | Included original derived datasets and frozen artifacts | Run the numerical scripts below in order |
| Reconstruct everything from raw events | Downloaded XES plus historical build scripts/settings | Follow the historical stage map; this is a separate task from reproducing the saved additional experiments |

The historical scripts include alternative versions. Their numeric prefixes do not define one universal command sequence. Required command-line settings and selected variants are recorded in their help, policy specifications, split metadata, and existing result metadata. Do not run every script indiscriminately over the frozen outputs.

## 2. Clone and create the environment

The numerical runs used Python 3.12.3. Use Python 3.12 and the pinned numerical dependencies, especially scikit-learn 1.6.1 for the saved models.

```bash
git clone https://github.com/sourabhmishra83/RCSE.git
cd RCSE
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproduction.txt
```

Activate on Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
```

Run commands from the repository root. The additional experiment scripts resolve inputs relative to their file locations. Older metadata can contain absolute paths from the executed machine; those are provenance records, not paths that must exist on the new machine. Original historical scripts generally expose paths through command-line arguments. VS Code workspace files describe the original Windows setup and are optional.

## 3. Verify the included evidence first

```bash
python tools/verify_reproduction.py
```

This checks required inputs, action-count accounting, coverage/UER arithmetic, always-escalate loss, calibration-selection/test separation, feature exclusion, training convergence, stress-test results, and fixed-horizon output structure. It does not retrain models or prove that the safety labels are complete. Saved model/results directories remain available for inspection without rerunning the expensive experiments.

## 4. Recompute the additional experiments

Use a separate clone for reruns if you want to preserve the supplied outputs exactly. Each script writes into its own existing experiment directory and can replace saved outputs. Review differences with Git afterward. The source log is needed only for the horizon reconstruction in step 5.

| Step | Command | Main purpose and outputs |
| --- | --- | --- |
| 1 | `python rq2_revision/run_rq2_revision.py` | Retrain action model without full-trace event count; joined predictions, AUROC/AP, paired bootstrap intervals |
| 2 | `python safety_mismatch_revision/run_safety_mismatch.py` | Add only the two monetary-ratio features; models, natural/full predictions, AUROC differences, fixed-threshold gate diagnostics |
| 3 | `python revision_complete/run_revision_analyses.py` | All-policy absolute metrics, explicit escalation, loss intervals, cost grid, independent stress, ambiguous-label bounds, consequence-only ablation |
| 4 | `python revision_complete/run_calibration_baselines.py` | Independent isotonic/selection halves, LTT/CRC baselines, threshold sweeps, calibration-selected no-GATHER frontier |
| 5 | `python revision_complete/run_followup_windows.py` | Reconstruct 30/60/90-day targets from raw source events, evaluated with fixed scores |
| 6 | `python revision_complete/build_figures.py` | Rebuild five PNG/SVG figures from saved numerical results |
| 7 | `python tools/verify_reproduction.py` | Recheck the resulting evidence and split separation |

Steps 1 and 2 train logistic models and run 2,000-resample comparisons; step 3 includes 2,000 paired source-cluster loss resamples. They take substantially longer than the verification utility. Logs and training audits show progress and convergence. Do not interpret a partially written CSV during a running experiment as its final result.

Important included inputs:

```text
rcse_output/rcse_base_v2.parquet
rcse_output/experimental_benchmark/splits/rcse_experimental_with_splits.parquet
rcse_output/experimental_benchmark/splits/frozen_baseline_results/
rcse_output/experimental_benchmark/splits/t0_execution_safety_results/
rcse_output/experimental_benchmark/splits/frozen_baseline_results/rcse_policy_dataset/rcse_v2_policy_dataset/
09b2l_rcse_v2_policy_spec.json
revision_complete/analysis_protocol.json
```

Step 3 needs the mismatch models from step 2. Step 4 needs the corrected action models from step 1. Step 5 uses the separate-calibration models from step 4. Figure generation needs the corresponding saved result CSVs. The frozen original safety predictions remain a reference in the shared-target comparison.

## 5. Obtain the raw log for horizon reconstruction

Download [BPI Challenge 2019 from 4TU.ResearchData](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1), decompress it, and name it `BPI_Challenge_2019.xes` in the repository root. See [source-data details](../revision_complete/DATA_ACCESS.md). The executed run's file SHA256 is recorded in `followup_window_manifest.json`; a differently packaged or updated source file can differ and should be documented.

The parser streams traces. It evaluates events strictly after decision time through the inclusive horizon endpoint. It leaves unresolved cases ambiguous. Reaching the end of the available log is not proof of complete outcome capture.

## 6. Historical pipeline stage map

| Stage | Relevant scripts | Result family |
| --- | --- | --- |
| Source inspection and base states | `inspect_bpic2019.py`, `01_*`, `02_*` | Event-log summaries and t0 feature tables |
| Labels and intervention benchmark | `03_*`, `04_*`, `05_*` | Readiness labels, benchmark episodes |
| Source-grouped splits | `06_*` | IID/temporal/exposure-OOD assignments and metadata |
| Action models and score diagnostics | `07_*`, `08_*`, `08b_*`, `08c_*` | Training audits, calibrated predictions, ablations |
| Historical evidence acquisition | `09a1_*` through `09a5_*` | t1 reconstruction, auxiliary model, diagnostics |
| Safety target/model | `09b2e_*` through `09b2h_*` | Natural D1 labels, models, full-benchmark scoring |
| Gate specification | `09b2i_*` through `09b2k_*` | Invariant/footprint audits and fixed Gate v2 |
| Frozen decision policy | `09b2l_*`, `09b2m_*`, `09b2m2_*`, `09b2n_*` | Specification, policy datasets, decisions/results |
| Reporting and uncertainty | `09b2o_*` through `09b2s_*` | Frontier, unsafe-error decomposition, bootstrap results |

Use `python SCRIPT.py --help` for argument-based historical programs. Preserve source grouping and original settings when attempting a raw rebuild. The additional validation commands above are the supported direct rerun path using included frozen inputs; the stage map is not a claim of a tested one-command raw rebuild.

## 7. Compare outputs and interpret differences

Compare CSV case counts, selected thresholds, AUROC, confidence intervals, and mean loss against the saved files. Seeds and bootstrap counts are recorded in manifests. Randomized resamples are fixed within a script, but different BLAS/CPU/library environments may produce small floating-point differences. Bitwise equality of joblib files and every figure is not promised. Input hashes refer to bytes from the executed workspace; Git line-ending conversion can change source-file byte hashes without changing the program.

Use [methods](METHODS.md) to understand populations and estimands, and [results](RESULTS.md) to understand supported and unsupported conclusions. Historical fallback UER, resolved-only UER, CRC marginal risk, and LTT conditional risk are different quantities; they must not be merged into one comparison.


## Additional label-loss sensitivity

After the original policy outputs and revision analyses exist, run `python revision_complete/run_loss_label_sensitivity.py`. It reads saved scores and decisions, writes `natural_loss_label_sensitivity.csv` and its manifest, and checks the original-fallback losses against the previously saved results. Models and policies are held fixed.


The immutable research snapshot is [RCSE v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.23273149), DOI 10.5281/zenodo.23273149. It archives the versioned [GitHub release](https://github.com/sourabhmishra83/RCSE/releases/tag/v1.0.0).
