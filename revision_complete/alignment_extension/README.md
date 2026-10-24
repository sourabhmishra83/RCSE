# Consequence alignment and repeated vendor holdouts

This exploratory extension preserves all earlier outputs and adds three checks:
consequence permutation, PO-grouped policy/selection comparisons, and ten
cold-start vendor allocations. Hyperparameters and outcomes remain unchanged.
Group identifiers and vendor names do not enter the forty model predictors.

## Reproduce

Use the environment in `requirements-reproduction.txt` and run from the root:

```bash
python revision_complete/run_consequence_placebo.py
python revision_complete/run_vendor_split_sensitivity.py
python revision_complete/summarize_alignment_components.py
python revision_complete/plot_alignment_extension.py
python revision_complete/validate_alignment_extension.py
```

Inputs are included derived datasets, feature manifests, frozen models and raw
scores in `robustness_extension/` and `grouped_component_extension/`, plus the
existing source base. No raw XES download is required for these checks. Vendor
models are cached; remove only `vendor_*_logistic.joblib` and
`vendor_*_histboost.joblib` in this directory to force retraining. Other score,
calibration and result outputs are recomputed. One original vendor fit is reused
exactly; nine additional pairs of models are fitted.

## Permuted-consequence placebo

Twenty fixed permutation assignments use seed sequences `[20261024, regime
index, permutation ID]`. Each separately permutes the actual marginal log
consequence among all resolved calibration cases and among resolved test cases.
The assignment is held fixed across both models and twenty calibration-half
seeds; it never uses labels. The original model, isotonic-fit procedure, gate,
8-by-6 RCSE graph and error allocation remain unchanged. All four nominal risk
targets are reported, across item IID, PO-grouped IID, fixed-horizon temporal,
and exposure OOD. PO-grouped calibration halves keep purchasing documents
together. Other halves retain the original case hash for exact comparability.

Counterfactual masks use permuted consequences, but losses use the actual
unpermuted consequence. Placebo decision masks therefore do not necessarily
satisfy the original expected-loss constraint; they are a candidate-family
diagnostic, not a replacement economic policy. Permutations preserve the full
calibration marginal, not the exact marginal of each subsequent half.

At the nominal 2% target, actual logistic certification and the placebo median
(range) of certifications over twenty calibration seeds are:

| Regime | Actual | Placebo median and range |
| --- | --- | --- |
| Item IID | 19/20 | 0/20 (0-7/20) |
| PO-grouped IID | 7/20 | 2/20 (0-5/20) |
| Fixed-horizon temporal | 15/20 | 0/20 (0-3/20) |
| Exposure OOD | 18/20 | 0/20 (0-4/20) |

Boosted graphs certify 20/20 under every permutation in these regimes. For
logistic regression, random value-induced intermediate sets do not reproduce the
actual alignment result in this finite design. This does not identify a unique
mechanism, establish economic superiority, or overcome correlated-item and shift
limitations of nominal binomial certification. Permutation ranges are descriptive
Monte Carlo summaries, not confidence intervals or independent replications.

`placebo_results.csv` has all 13,440 regime/model/seed/permutation/target rows.
`placebo_permutation_summary.csv` aggregates twenty calibration seeds per
permutation. `placebo_summary.csv` reports actual versus placebo summaries.
`po_grouped_selection_results.csv` and `po_grouped_selection_summary.csv` add the
matched PO-grouped safety, gated-safety, and RCSE eight-chain procedures.

## Repeated vendor holdouts

Ten fixed vendor hash allocations include the original seed 20261023 plus
20261024 through 20261032. The original uses the published
`all-eligible-20261023:` hash prefix; subsequent allocations use
`vendor-heldout-{seed}:`. Uniform group hashes use 70/15/15 cutoffs, without
balancing vendor sizes. Every vendor stays in a single partition. Both estimators
are refitted per new allocation using the original hyperparameters. Isotonic
halves hash `20261004` plus vendor ID, so vendor groups also stay disjoint
between calibration roles. Models and calibrators are supplied.

Complete-evidence boosted-minus-logistic AUROC has median +0.0238 and range
-0.0214 to +0.0774, positive in 8/10 splits. Four individual 95% vendor-cluster
intervals exclude zero in the boosted direction; none exclude zero in the
opposite direction. No multiplicity-controlled conclusion across ten splits is
asserted. Original single-split near-equality is not a general cold-start result.

Largest-vendor share in these complete-evidence resolved tests ranges from
11.1% to 27.4%; effective vendor count is 8.5-25.1. The latter is
`1 / sum(item_share**2)`, not the raw vendor count. Concentration is also released
for every full train/calibration/test partition. Intervals use 1,000 paired
vendor-cluster bootstrap replicates, conditional on each fitted model. Overlap
of vendors and training populations across allocations prevents treating these
ten fits as independent external validations.

PO isolation represents new orders principally from known vendors: 32,039 of
32,154 PO-holdout test items have vendors represented in training. Vendor
isolation instead tests cold-start vendors. Vendor identifiers and names are
excluded from predictors; process fields about vendor invoices are included.

`vendor_repeated_auc.csv`, `vendor_repeated_cohorts.csv`, and
`vendor_repeated_summary.json` contain these results. Split assignments are in
`vendor_repeated_assignments.parquet`. Per-seed models, scores, calibrators and
fixed-policy tables make the nine new fits reproducible.

## Component interpretation

The existing `grouped_component_extension/fixed_policies.csv` and
`loss_contrasts.csv` supply PO-grouped counterparts to the fixed-policy and
component tables. PO-grouped boosted safety-minus-completeness is -0.003418
[-0.006251, -0.000784], smaller than the item-split -0.0093 gain. Logistic
PO-grouped safety also improves loss: -0.003233 [-0.006034, -0.000687]. These
contrasts condition on the given models, test population and log consequence.

The stricter reference constraints jointly cost about 0.002-0.017 mean loss
relative to plug-in cost in item IID and fixed-horizon temporal tests, but save
about 0.04-0.06 under exposure shift. This contrasts both the fixed safety floor
and budget with the plug-in comparator; it does not isolate consequence weighting
alone. Best prespecified fixed policies differ: boosted plug-in in item- and
PO-grouped IID, gate only in fixed-horizon temporal, escalation in exposure OOD.
These are descriptive rankings within the fixed-policy menu, not an exhaustive
comparison with every calibration-selected rule.

Both logistic and boosted deletion diagnostics already appear in the prior
`loss_concentration.csv`. Boosted linear temporal RCSE-minus-safety is -0.003099
[-0.005728, -0.000935]. Its top five beneficial cases contribute 47.1% of the
benefit; omitting the five largest absolute contributions leaves -0.000858.
The negative point estimate survives that deletion, while tail concentration and
unmeasured training uncertainty remain. The logistic linear contrast includes
zero and changes sign after the corresponding five-case deletion.

Manuscripts, review correspondence, editorial logs, document builders and
credentials remain excluded from the public archive.
