# Value-conditional calibration mechanism check

This exploratory extension asks whether the earlier logistic RCSE certification
advantage can be reproduced by calibrating a scalar safety score within PO-value
bins. It preserves the original model fits, partitions, graph starts, risk
targets, and nominal error allocation. It does not select a favorable test split.

## Reproduce

From the repository root, using `requirements-reproduction.txt`:

```bash
python revision_complete/run_value_conditional_calibration.py
```

Inputs are the included base dataset, feature-independent split assignments,
raw scores and calibrators in `robustness_extension/` and
`grouped_component_extension/`. No raw XES download or model retraining is needed.
The script writes all results and consistency checks into this directory.

## Protocol

Four regimes are evaluated: item-hash IID, PO-grouped IID, fixed-horizon temporal,
and exposure OOD. Both frozen logistic and histogram-boosted estimators use the
same twenty calibration-half seeds and four risk caps, 2%, 3%, 5%, and 10%.
PO-grouped calibration halves retain entire purchasing documents together.

Two bin schemes were specified before examining these results:

* P90 bins: absolute PO amount at or below the training P90 versus above it.
* Quartile bins: training-population P25, P50, and P75 boundaries.

Within each bin, isotonic regression maps **raw model scores** to SAFE probability
using only the isotonic-fitting calibration half. The independent second half
performs policy testing, and the fixed test set evaluates the selected policy.
Candidate values are the union of the first-half fitted bin-anchor values. The
eight scalar starts, delta/8 allocation, stopping rule, and maximum-selection-
coverage choice are identical to the released safety graph. Gated and ungated
variants are both reported. Empty fitting bins would use the pooled first-half
calibrator; no fallback occurred. Bin-specific calibrators for the first seed
are supplied, and all other fits can be recreated by the script.

Pooled calibration is rerun as a consistency check; all execution counts,
unsafe counts, and losses match the prior archived scalar-graph results.

## Held-out calibration diagnostics

On the independent selection half at the first seed, logistic UER within the
gate and pooled score at least 0.95 differs by value:

| Regime | At most P90 | Above P90 | High minus low, percentage points, PO-cluster 95% CI |
| --- | --- | --- | --- |
| Item IID | 128/8382 = 1.53% | 14/379 = 3.69% | 2.17 [0.31, 4.23] |
| PO-grouped IID | 222/10532 = 2.11% | 49/827 = 5.93% | 3.82 [1.92, 5.84] |
| Fixed-horizon temporal | 141/8626 = 1.63% | 26/513 = 5.07% | 3.43 [1.59, 5.59] |

The script additionally reports score bands [0.95,0.97), [0.97,0.99), and
[0.99,1]. In the PO-grouped logistic [0.95,0.97) band, low/high mean scores are
0.952474/0.952471 while unsafe rates are 71/1717 = 4.14% and 24/321 = 7.48%.
Thus a broad-band difference is accompanied by residual dependence within a
narrower band at nearly equal mean scores. These diagnostics describe conditional
association rather than a causal effect of transaction value.

High-minus-low intervals use 2,000 paired PO-cluster bootstrap replicates, seed
20261024, conditional on the first fitted pooled calibrator. All twenty seed-level
rates are retained; they are not independent external replications.

## Selection findings

At the nominal two-percent target, **gated logistic scalar graph** certification
over twenty calibration-half seeds is:

| Regime | Pooled | P90 bins | Quartile bins | Actual RCSE graph |
| --- | --- | --- | --- | --- |
| Item IID | 7/20 | 12/20 | 20/20 | 19/20 |
| PO-grouped IID | 1/20 | 4/20 | 9/20 | 7/20 |
| Fixed-horizon temporal | 0/20 | 2/20 | 20/20 | 15/20 |
| Exposure OOD | 1/20 | 10/20 | 20/20 | 18/20 |

The two-bin scheme only partly closes the gap; quartile calibration matches or
exceeds the certification count. The earlier permutation result can therefore
be explained by value-conditional calibration within this design, without a
unique consequence-weighting advantage. Calibration binning also changes score
resolution and candidate geometry, so it is not a unique causal decomposition.
The PO-grouped result remains sensitive: 9/20 is not a stable always-executing
operating point, and its median coverage over all repetitions is zero.

Nominal calibration certification is distinct from test risk under shift.
For example, quartile-calibrated gated logistic OOD median test UER is about
9.09%, despite certification at the 2% calibration target. Item dependence,
unresolved outcomes, and the limitations of the source log remain unchanged.
This extension does not establish improved economic loss or production safety.

## Outputs

* `conditional_calibration_rates.csv`: all model/regime/seed/value/score-band
  counts, UER, mean calibrated score and expected unsafe rate.
* `p90_risk_difference.csv`: first-seed high-minus-low PO-cluster intervals.
* `selection_results.csv`, `selection_summary.csv`: all 3,840 scalar selections
  and twenty-seed summaries, including no-execution repetitions.
* `calibration_fit_audit.csv`, `value_bin_boundaries.csv`: fit counts, fallback
  audit, and training-only cutpoints.
* First-seed `.joblib` calibrator bundles, `protocol.json`, `validation.json`.

Manuscripts, reviewer text, editorial logs, document builders and credentials
remain outside the public research archive.
