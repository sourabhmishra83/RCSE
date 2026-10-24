# Grouped holdouts and component decision audit

This exploratory extension uses the same 210,692 eligible item cases, 40 predictors,
D1 target, model settings, and deterministic hash seed as the full-population study.
It adds purchasing-document and vendor holdouts, eight-chain scalar safety
selection, a plug-in cost comparator, clustered uncertainty, and absolute exposure
totals. Existing results in `../robustness_extension/` are preserved.

## Reproduce

From the repository root, in the environment specified by
`requirements-reproduction.txt`, run:

```bash
python revision_complete/run_grouped_component_extension.py
python revision_complete/plot_grouped_component_extension.py
python revision_complete/validate_grouped_component_extension.py
```

Inputs are the included `rcse_output/rcse_base_v2.parquet`, the existing feature
manifest and methods scripts, and the saved full-population models, calibrators,
scores and figure data under `revision_complete/robustness_extension/`.
No raw XES download is required for this extension. The grouped experiment caches
trained models; remove its `logistic_group_*.joblib` and `histboost_group_*.joblib`
files before rerunning if retraining rather than cached evaluation is required.
The released grouped score files are recalculated on every run.

The source base is partitioned before dropping ambiguous cases. Each group is
assigned by SHA256 of `all-eligible-20261023:` plus its original identifier:
less than 0.7 is training, less than 0.85 calibration, otherwise test.
All items of a group stay together. Group size means case fractions need not
equal exactly 70/15/15. Calibration halves hash `20261004` plus the group ID.
PO and vendor identifiers are not model predictors. The two grouped fits use
the same hyperparameters as the original item-hash study; no test tuning is used.

## Selection and inference

Scalar Safety graph and Gate safety graph each have eight predeclared starting
thresholds: 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97, 0.99. Within each chain,
the start is tested first, then every smaller distinct isotonic anchor value in
descending order. A chain stops at its first nonrejection; each receives delta/8
with delta=0.05. Maximum independent selection-half coverage chooses among passed
nodes. Twenty original calibration-half seeds and all four risk targets are used.
RCSE graph has the same number of chains and error allocation, but a different
candidate geometry and number of nodes. This is a procedure comparison, not proof
of an intrinsic family advantage. Isotonic predictions interpolate between fitted
anchors; the number of distinct test scores can exceed the number of anchors.

Binomial certification is nominal under item independence. Correlated items can
violate that assumption; grouped training and clustered bootstrap intervals do
not make the item-binomial test a cluster-valid certificate. Nothing here gives a
production, simultaneous-across-seeds, or distribution-shift guarantee.

The plug-in cost policy executes if the gate is VALID and `(1-score)*C <= 0.10`.
It is RCSE with safety floor zero and budget 0.10, and minimizes expected loss
only under the supplied probabilities and cost assumptions. It need not minimize
observed test loss. The stricter reference policy retains floor 0.95 and budget
0.05, explicitly pricing additional risk constraints.

Loss contrasts resample paired PO clusters, or vendors in the vendor holdout,
2,000 times. Case-bootstrap contrasts are also supplied. AUROC differences use
1,000 paired cluster replicates on complete-evidence resolved test cases. All
intervals condition on fitted models and use seed 20261023. They do not include
training variability or dependence beyond the resampling unit.

## Results and interpretation

* The item split has 25,260 of 31,561 test items sharing a purchasing document
  with training. PO-grouped complete-evidence AUROCs are 0.7406 logistic and
  0.8026 boosted, gap 0.0620 [0.0424, 0.0824]. Vendor-grouped AUROCs are 0.8052
  and 0.8057, gap 0.0005 [-0.0263, 0.0320]. Different test populations preclude
  attributing gap changes solely to memorization.
* At the 2% nominal target, eight-chain IID safety certifies in 6/20 logistic
  repetitions and 20/20 boosted repetitions; gated safety in 7/20 and 20/20.
  RCSE graph certifies in 19/20 and 20/20 under its existing candidate geometry.
  Learned-start safety results remain available separately.
* The fixed-horizon temporal gate-minus-completeness log-scale loss difference
  is -0.000761 [-0.001373, -0.000172] under PO-cluster resampling.
* The logistic linear temporal RCSE-minus-safety difference is -0.001724
  [-0.004095, 0.000263]. The largest five beneficial cases supply 47.9% of
  the total beneficial contribution; removing five largest absolute contributions
  reverses the point estimate. This is an uncertain, concentrated result.
* Boosted linear temporal RCSE-minus-safety is -0.003099
  [-0.005728, -0.000935], a conditional benefit under that consequence scale.
* IID plug-in cost losses are 0.036073 logistic and 0.031069 boosted; temporal
  losses are 0.037725 and 0.034960. Exposure-OOD losses are 0.141283 and
  0.167116, illustrating failure of an assumed-probability optimum under shift.

`fixed_policies.csv` includes unsafe exposure executed and total exposure
escalated, both in EUR per 1,000 cohort cases. These sum item-level absolute PO
amounts, not deduplicated PO totals, realized harm, or staffing cost. Restriction
cannot increase absolute unsafe exposure executed, although value-weighted UER
can rise when safe high-value executions are removed from its denominator.

`loss_concentration.csv` reports the largest beneficial contribution shares and
the mean paired contrast after removing one or five largest absolute case
contributions. These are diagnostic sensitivities, not policy-selection rules.

## Output map

* `grouped_split_assignments.parquet`, `grouped_cohorts.csv`: group isolation and
  full cohort accounting, including ambiguous cases.
* `raw_scores_group_*.parquet`, models and calibrators: complete grouped fits.
* `capacity_auc.csv`, `original_group_overlap.csv`: discrimination and overlap.
* `fixed_policies.csv`, `loss_contrasts.csv`, `loss_concentration.csv`: policy,
  exposure, uncertainty, and tail sensitivity evidence.
* `safety_graph_results.csv`, `safety_graph_summary.csv`,
  `safety_graph_audit.csv`, `isotonic_steps.csv`: all seeds, targets, stopping
  traces for the first seed, and isotonic resolution.
* `risk_coverage.*`, `break_even.*`: independent-axis publication figures.
* `protocol.json`, `completion.json`, `validation.json`: design and checks.

Manuscripts, reviewer correspondence, document builders, editorial logs and
credentials are excluded from the public archive.
