# Full population robustness study

This extension studies all 210,692 eligible natural purchase-order item cases,
without conditioning inclusion on the intervention benchmark. One case contributes
one row. The primary policy comparisons are conditional on resolved outcomes;
`ambiguity_bounds.csv` also reports both bounds for unresolved executions.

## Reproduce

Use the repository's pinned Python environment and run from the repository root:

```bash
python revision_complete/run_source_followup_audit.py
python revision_complete/run_robustness_extension.py
python revision_complete/run_selection_and_stress_audit.py
python revision_complete/run_robustness_uncertainty.py
```

The first command needs the separately downloaded `BPI_Challenge_2019.xes` in the
repository root. It audits 1,595,923 raw events and creates the supplied compact
`all_eligible_followup.parquet`. If using that saved file, the remaining analyses do
not need the raw XES. Data access and source license are described in
`revision_complete/DATA_ACCESS.md`. Source code and outputs from older analyses are
retained; their benchmark populations must not be substituted for this population.

## Protocol

- IID: a fixed uniform source-case SHA256 hash splits all eligible cases 70/15/15.
- Later temporal sensitivity: 70/15/15 by decision time and source-case ID, with D1.
- Earlier temporal: Jan-Mar 2018 training, Apr-May calibration, September testing;
  every partition uses a 90-day outcome target. The latest calibration horizon ends
  before testing starts. Calendar opportunity does not establish complete capture.
- Exposure OOD: the highest 1% of absolute PO values is held out. Lower values have
  train, calibration, and ID-test partitions. The ID-test partition is explicitly
  reported, not silently discarded from the cohort accounting.
- Models: regularized logistic regression and sklearn histogram gradient boosting,
  using the same 40 decision-time fields. Neither includes future total event count.
- Twenty source-ID hash seeds divide resolved calibration cases into isotonic-fit
  and independent selection halves. Only the calibrator is refitted across seeds;
  the trained model and test cases stay fixed.
- Safety FST tests all distinct fitted isotonic values from highest to lowest and
  stops at the first nonrejection. Safety split FST learns a starting node on the
  first half, then uses the independent second half for fixed-sequence testing.
  The first-half binomial calculation is only a heuristic, not a certificate.
- RCSE graph uses eight fixed safety-floor chains with increasing loss budgets,
  error budget 0.05/8 per chain, and stopping at first failure. Safety floors extend
  to zero, avoiding the former 0.90 floor as an artificial coverage ceiling.
- Certification is relative to the selection distribution and assumptions. It
  does not imply shifted test risk control, simultaneous control across seeds, or
  outcome capture completeness. Repeated-seed summaries are descriptive stability
  summaries, not twenty independent external validations.

## Outputs and interpretation

`cohorts.csv` and `all_eligible_split_assignments.parquet` account for every partition,
including unresolved cases and the OOD ID-test set. `training_audit.csv` records
training-only consequence references and model iterations. `raw_scores_*.parquet`,
model files, and first-seed calibrators permit independent evaluation.

`repeated_policy_results.csv` contains all 20-seed results, including counts,
coverage, conditional UER, value-weighted UER, modeled loss, and selected parameters.
`repeated_summary.csv` includes certification frequency and median coverage across
all repetitions. Its median UER excludes undefined zero-execution repetitions;
it therefore need not describe the same repetition as median coverage.
`sequential_selection_audit.csv` saves each first-seed second-half test.

`break_even.csv` and `cost_curves.csv` distinguish exact fixed-decision repricing
from recomputation of RCSE decisions at each human cost. A fixed-decision
break-even is avoided unsafe consequence divided by additional escalations.
It is not a claim that the policy was reoptimized at that cost.
`consequence_sensitivity.csv` compares log, linear, and linear capped at 10 times
the training P90 consequence, evaluating all policies on a common scale within
each mode. Value-weighted UER is unsafe selected PO exposure divided by all
selected PO exposure. It is not actual observed monetary loss.

`source_timestamp_audit.json` reports unusual source years without automatically
declaring their timestamps invalid. `score_integrity_probes.csv` is a gate
specification-gap and confidence-sensitivity test, not a downstream harm experiment.
`legacy_selection_reanalysis.csv` isolates the earlier design-weighted 2% headline
and shows its dependence on selection procedure and calibration split.

`paired_loss_uncertainty.csv` uses 2,000 paired natural-case bootstrap repetitions;
`paired_capacity_auc.csv` uses 1,000 paired repetitions within complete-evidence
cases. Both condition on trained models. The two publication figures have SVG and
PNG copies, with underlying curve tables retained. All results are exploratory
reanalyses of an already inspected event log, not a fresh confirmatory holdout.
