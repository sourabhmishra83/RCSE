# Matched natural case comparison

This extension compares policies on the same natural D1-resolved source cases.
It reuses the saved safety model's raw and independently calibrated scores and
the corrected action comparator. No fitting or selection uses test labels.

From the RCSE repository root, using the pinned reproduction environment:

```bash
python revision_complete/run_matched_natural_comparison.py
```

The required input predictions are the three regimes' `independent_selection_cases_*.parquet`
and `separate_calibration_natural_test_*.parquet` in `revision_complete/`.
The script also reads the base data, frozen policy datasets, specification, and
bootstrap helpers already included in the RCSE repository. Raw XES is not needed.
If rebuilding the score files, first run `revision_complete/run_calibration_baselines.py`.

## Outputs

- `protocol.json`: fixed candidate grids, targets, populations, and inference scope.
- `policy_results.csv`: coverage, unsafe counts, conditional UER, exact binomial
  test-sample intervals, outcome-based mean loss, selected parameters, and selection counts.
- `ltt_selection_audit.csv`: all tested scalar thresholds and binomial decisions.
- `descriptive_sweeps.csv`: held-out score/budget sweeps for visualization only.
- `paired_loss_difference.csv`: paired source-case bootstrap mean-loss differences.
- `retention_audit.csv`: gate-pass retention and actual policy execution retention,
  with their numerators and denominators, for the original natural benchmark.
- `reference_loss_audit.csv`: original no-GATHER predicted cost, its budget-based
  bound, and outcome-based modeled loss. A bound on predicted cost does not bound
  realized outcome-based loss.
- `matched_risk_coverage.png` and `.svg`: matched-cohort descriptive comparison.
- `manifest.json`: input score-file hashes and limitations.

Mean loss charges `unsafe * consequence` on executed cases and 0.10 on escalated
cases, averaging over every case in the stated resolved cohort. The consequence
reference is the frozen source-case training P90 for each regime. No claim of
actual monetary savings or observed financial harm follows from these costs.

LTT targets conditional UER under its assumptions. CRC targets marginal unsafe
incidence per case, which is a different risk. Bonferroni correction is within each
fixed family and target, not simultaneous across every reported analysis.
No exchangeability guarantee applies to temporal shift or exposure OOD.
The 25 scalar thresholds and 20 RCSE pairs are different candidate families;
strict-target comparisons do not establish dominance over all possible thresholds.

The original bootstrap field `natural_reference_execute_retention_rate` measures
gate-pass retention, not execution retention. Original scientific outputs are
preserved; `retention_audit.csv` supplies both correctly named quantities.

The Zenodo v1.0.1 package (10.5281/zenodo.23283511) predates this extension.
It supplies the preceding repository inputs; this extension is additional material.
