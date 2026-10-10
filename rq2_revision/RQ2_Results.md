# RQ2 post-review validation

Completed using the existing Miniconda environment. Original Paper2 files were read only.

## Method

The action model was retrained on the original full benchmark training episodes, with only events_total_case removed. The remaining 21 numeric and 19 categorical features, logistic-regression settings (L-BFGS, L2, C=1, max_iter=5000, tol=0.0001), original splits, and one-vs-rest isotonic calibration with probability renormalization were preserved.

The safety estimator was not retrained. Its existing frozen scores and D1 labels were joined to corrected action scores on every case in its original natural D1-resolved held-out population. No test outcomes were used for training, calibration, or threshold selection. Train, calibration, and test source-case sets were verified disjoint.

Evidence-complete cases satisfy: goods_receipt_required is false OR gr_available_at_t0 is true. D1-ambiguous cases remain excluded. Bootstrap CIs use 2,000 paired source-case resamples (seed 20261003), independently within each regime and subset; each natural source case occurs once. Models remain fixed during bootstrap. These intervals do not capture training variability.

## Calibrated AUROC results

| Regime | Population | Cases | NOT_SAFE | Action AUROC [95% CI] | Safety AUROC [95% CI] | Safety minus action [95% CI] |
|---|---|---:|---:|---|---|---|
| grouped_iid | natural_d1_resolved | 20,604 | 3,294 | 0.8782 [0.8696, 0.8876] | 0.9465 [0.9407, 0.9520] | 0.0682 [0.0605, 0.0765] |
| grouped_iid | evidence_complete | 18,064 | 754 | 0.4681 [0.4467, 0.4874] | 0.7666 [0.7453, 0.7859] | 0.2985 [0.2692, 0.3291] |
| temporal | natural_d1_resolved | 9,089 | 1,993 | 0.9224 [0.9139, 0.9308] | 0.9607 [0.9543, 0.9664] | 0.0383 [0.0318, 0.0452] |
| temporal | evidence_complete | 7,472 | 376 | 0.5892 [0.5650, 0.6126] | 0.7925 [0.7664, 0.8178] | 0.2034 [0.1755, 0.2331] |
| ood_exposure | natural_d1_resolved | 1,376 | 452 | 0.7312 [0.6999, 0.7606] | 0.7494 [0.7179, 0.7824] | 0.0182 [-0.0202, 0.0595] |
| ood_exposure | evidence_complete | 1,219 | 295 | 0.5910 [0.5533, 0.6302] | 0.6189 [0.5771, 0.6602] | 0.0278 [-0.0341, 0.0910] |

## Interpretation and limits

Interpret the paired differences separately for IID, temporal, and exposure OOD. Evidence-complete analysis removes the readily observable missing-GR component of D1, isolating discrimination among complete cases. It does not establish deployable safety guarantees. Average precision for both SAFE and NOT_SAFE is included in the CSVs because SAFE is highly prevalent in some subsets.

This is a post-review sensitivity analysis, not a replacement of original frozen results. It changes both the action feature set and the scored population relative to the original manuscript; it cannot attribute changes in AUROC to feature removal alone. Action and safety models retain different training objectives and populations. This experiment does not yet add mismatch features, assess unseen corruption, revise policies, or resolve temporal right censoring.

## Files

- rq2_point_estimates.csv: raw and calibrated AUROC and average precision, case counts and label prevalence.
- rq2_paired_bootstrap_ci.csv: calibrated-score intervals and paired safety-minus-action differences.
- paired_predictions_*.parquet: row-level joined predictions and evidence-completeness flags.
- training_audit.csv: convergence and training/calibration/test sizes.
- action_model_without_future_count_*.joblib: trained action pipelines and calibrators; load only in trusted environments.
- experiment_manifest.json: frozen configuration, library version, and SHA-256 hashes of source inputs.
- run_rq2_revision.py: reproducible experiment script.
- RCSE_Revision.code-workspace: VS Code workspace with Miniconda interpreter and run task.

## Open in VS Code

Open RCSE_Revision.code-workspace in VS Code. The configured build task (Ctrl+Shift+B) reruns the experiment and overwrites files in this separate revision output directory. It does not write to Paper2. No editor connection was used for the completed run.
