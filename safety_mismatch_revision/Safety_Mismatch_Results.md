# Mismatch-feature safety sensitivity

Post-review analysis. Added only inv_po_abs_rel_diff and inv_gr_abs_rel_diff to the original natural-only safety model. Original splits, logistic regression, and isotonic calibration are preserved. Original artifacts remain unchanged.

| regime | subset | model | cases | unsafe | auroc | ap_unsafe |
| --- | --- | --- | --- | --- | --- | --- |
| grouped_iid | natural_resolved | original | 20604 | 3294 | 0.946486 | 0.898544 |
| grouped_iid | natural_resolved | mismatch | 20604 | 3294 | 0.946469 | 0.899378 |
| grouped_iid | evidence_complete | original | 18064 | 754 | 0.766604 | 0.202062 |
| grouped_iid | evidence_complete | mismatch | 18064 | 754 | 0.766433 | 0.203172 |
| temporal | natural_resolved | original | 9089 | 1993 | 0.960731 | 0.928834 |
| temporal | natural_resolved | mismatch | 9089 | 1993 | 0.960184 | 0.928289 |
| temporal | evidence_complete | original | 7472 | 376 | 0.792546 | 0.237748 |
| temporal | evidence_complete | mismatch | 7472 | 376 | 0.789726 | 0.229788 |
| ood_exposure | natural_resolved | original | 1376 | 452 | 0.749364 | 0.698701 |
| ood_exposure | natural_resolved | mismatch | 1376 | 452 | 0.749579 | 0.694504 |
| ood_exposure | evidence_complete | original | 1219 | 295 | 0.618855 | 0.426084 |
| ood_exposure | evidence_complete | mismatch | 1219 | 295 | 0.619183 | 0.420515 |

Paired AUROC differences (mismatch minus original):

| regime | subset | metric | estimate | lower | upper | replicates |
| --- | --- | --- | --- | --- | --- | --- |
| grouped_iid | natural_resolved | mismatch_minus_original_auroc | -1.68452e-05 | -0.000331178 | 0.000300856 | 2000 |
| grouped_iid | evidence_complete | mismatch_minus_original_auroc | -0.000170131 | -0.00158099 | 0.00118115 | 2000 |
| temporal | natural_resolved | mismatch_minus_original_auroc | -0.000547116 | -0.00134966 | 0.000178763 | 2000 |
| temporal | evidence_complete | mismatch_minus_original_auroc | -0.00281961 | -0.00727285 | 0.00104479 | 2000 |
| ood_exposure | natural_resolved | mismatch_minus_original_auroc | 0.000214295 | -0.00290943 | 0.00319518 | 2000 |
| ood_exposure | evidence_complete | mismatch_minus_original_auroc | 0.000328344 | -0.00464316 | 0.00476783 | 2000 |

Gate increment at fixed score threshold 0.95 and delta 0.05:

| regime | family | threshold | gate | resolved_episodes | executed | unsafe | coverage | uer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grouped_iid | ALL_RESOLVED | 0.95 | False | 35861 | 21320 | 16462 | 0.594518 | 0.772139 |
| grouped_iid | ALL_RESOLVED | 0.95 | True | 35861 | 5904 | 1084 | 0.164636 | 0.183604 |
| temporal | ALL_RESOLVED | 0.95 | False | 32434 | 23877 | 21517 | 0.736172 | 0.90116 |
| temporal | ALL_RESOLVED | 0.95 | True | 32434 | 3750 | 1423 | 0.115619 | 0.379467 |
| ood_exposure | ALL_RESOLVED | 0.95 | False | 2465 | 166 | 140 | 0.0673428 | 0.843373 |
| ood_exposure | ALL_RESOLVED | 0.95 | True | 2465 | 33 | 8 | 0.0133874 | 0.242424 |

Limits: gate results are score-filter diagnostics, not a rerun of RCSE expected-loss/GATHER policies. Synthetic cases are evaluation-only NOT_SAFE under the existing robustness convention; natural ambiguous cases are excluded. Gate can lower unsafe counts by lowering coverage; this is not a matched-coverage comparison. Existing intervention families are not an independent gate stress test. Natural-only training cannot establish robustness to previously unseen corruption. Bootstrap captures held-out sampling, not training variability.
