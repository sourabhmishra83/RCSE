# RCSE: architecture, benchmark, models, and evaluation

## Decision problem

RCSE evaluates whether an invoice-related process decision should execute, escalate, or abstain given evidence available at time t0, the first invoice-receipt event. Four inputs have different roles:

1. **Evidence validity:** deterministic checks on completeness and selected consistency relations.
2. **Action preference:** a multiclass model's probability of the constructed benchmark reference action.
3. **Execution safety:** a binary model's probability of the event-log D1 safety label.
4. **Consequence:** transaction exposure transformed into a modeled loss scale. The exposure proxy is the absolute PO amount, which is also a raw model input; the inputs are separate in the decision rule but not statistically independent.

Action preference is not the same target as D1 safety. A high model score also does not validate the integrity of its inputs.

## Source populations and interventions

The BPI Challenge 2019 Procure-to-Pay log supplies the source events. Filtering yields 210,692 eligible natural cases. The experiment's union contains 155,821 source cases. Its 246,682 benchmark episodes consist of:

| Episode family | Count | Meaning |
| --- | ---: | --- |
| Complete natural evidence | 50,000 | Natural state with required GR available or not required |
| Delayed natural evidence | 16,682 | Natural state lacking required GR at t0 |
| Monetary contradictions | 120,000 | Two intervention types, four strengths, 15,000 episodes per type/strength |
| Single-field missingness | 45,000 | GR, PO value, or invoice value removed; 15,000 each |
| Severe evidence loss | 15,000 | Multiple evidence fields unavailable |

Synthetic episodes are controlled robustness probes. Their frequency is not an estimate of real production failure prevalence. Some additional source cases enter the experimental union through intervention sampling; the natural safety training population is reconstructed from that union and exceeds the 66,682 natural benchmark episodes.

## Splits and information boundaries

All copies of a source case share a split. Three regimes test different behavior: grouped IID, chronological temporal, and exposure OOD. Temporal allocation orders source cases by decision time and ID in 70/15/15 proportions. Training ends at 2018-09-21 15:10 UTC; calibration ends at 2018-11-20 10:43 UTC; testing starts at 2018-11-20 10:44 UTC. Exposure references are computed from training data within each regime.

An audit found `events_total_case` in the original action predictors. It counts the full future trace. The corrected action experiment excludes it and retains the other 21 numeric and 19 categorical features. Historical action-based results therefore remain descriptive. The frozen original action pipeline is preserved for traceability, not presented as a leakage-free prospective model.

## D1 safety label

For natural cases:

- Required GR missing at t0, or any recorded subsequent adverse signal: NOT_SAFE.
- Evidence complete, eventual clearing recorded, and no recorded adverse signal: SAFE.
- Otherwise: AMBIGUOUS.

The six adverse activities are invoice cancellation, setting a payment block, GR cancellation, price change, quantity change, and subsequent invoice recording. Original D1 uses all available subsequent events without a fixed horizon. These labels are experimental process proxies; they are not adjudicated causal harm labels. Model fitting and natural resolved-score evaluation exclude ambiguous cases.

Evidence-complete subsets remove the directly observable missing-GR part of D1: GR is either not required or is available at t0. Remaining label uncertainty still depends on recorded follow-up.

## Models and calibration

Models use logistic regression with L2 regularization, L-BFGS, C=1, maximum 5,000 iterations, and tolerance 0.0001. Numeric processing uses median imputation and scaling; categorical processing uses imputation and one-hot encoding. The original action objective uses mixed natural/intervention episodes and benchmark action labels. The safety objective uses natural D1-resolved cases. Isotonic calibration is separate from training; multiclass action calibration uses one-vs-rest maps and renormalization.

The mismatch sensitivity adds only `inv_po_abs_rel_diff` and `inv_gr_abs_rel_diff` to the original natural safety predictors. Monetary mismatch is not silently added to the original model. Detailed predictor sets are in feature manifests and experiment manifests.

The shared-target experiment evaluates corrected calibrated action EXECUTE probability and frozen calibrated safety probability against the same SAFE target on the same natural held-out cases. The model objectives and training populations remain different. The comparison fixes evaluation target/population rather than making the training objectives identical.

## Gate v2 and stress tests

Gate v2 assigns VALID, INCOMPLETE, or CONTRADICTORY. Contradiction takes priority over incomplete evidence. Monetary checks use relative invoice-to-PO or invoice-to-GR discrepancies with a 5% boundary and existing denominator conventions. The development footprint audit examined all 180,000 interventions, including held-out source cases, so those corruption families do not provide independent gate validation.

Independent tests hold the trained model and gate fixed. One sets PO-to-invoice age to minus seven days; another multiplies PO/invoice/GR amounts by ten coherently and recomputes ratios, while external evaluation exposure is held fixed. They test omitted integrity checks. They are not observations of actual harm. Natural price/quantity-change subgroups are reported separately.

## Consequence and no-GATHER decision

With absolute exposure v and training P90 reference vref, the main consequence is:

```text
C(v) = log(1 + v) / log(1 + vref)
```

Other saved modes use uniform consequence or training P50/P99 references. Exposure is excluded from corrected predictive features. It enters the decision loss.

For a valid state with safety score s, execution is feasible only if `s >= tau` and `(1-s)*C(v) <= beta`. Feasible execution loss is `(1-s)*C(v)`; infeasible execution has infinite loss. Escalation loss is `cH + rhoH*C(v)`; abstention loss is `cA`. Choose minimum modeled loss, with execution, escalation, then abstention as tie order. The reference costs are cH=.10, rhoH=0, cA=.50, safety threshold .95, and beta=.05. Beta is an expected-loss budget, not an observed UER cap.

The consequence-only ablation changes decision consequence to one while evaluating outcomes on the same log-consequence loss scale. Gate, scores, and other constraints remain fixed. It isolates consequence more closely than comparing policies that change several components.

## GATHER information limitation

Historical simulations compare GATHER at t0 using an observed post-acquisition t1 score. This is retrospective look-ahead. A prospective GATHER policy would need an ex ante distribution over arrivals/transitions and a waiting horizon. That model is not supplied. Additional prospective comparisons disable GATHER. Historical end-to-end loss includes acquisition plus continuation; acquisition-only historical summaries are a different accounting quantity.

## Independent operating-point selection

SHA256 of the fixed seed concatenated with source-case ID, modulo two, divides original calibration cases into isotonic and independent selection halves. Train on original training cases, fit isotonic only on the first half, select thresholds on the second, and evaluate on held-out tests. Corrected raw action scores use the train-fitted pipeline and avoid reusing isotonic fitting labels.

**Learn then Test (LTT):** a fixed 25-threshold family uses exact one-sided binomial tests on conditional unsafe execution, with Bonferroni correction at delta=.05 per family/cap. Select the highest-coverage certified candidate. IID natural-case exchangeability is an assumption; temporal, exposure-OOD, and clustered synthetic cases do not inherit that guarantee.

**Conformal risk control (CRC):** thresholded unsafe-selection indicators are averaged across all natural cases. Selection uses `(unsafe selections + 1)/(selection cases + 1) <= alpha`. The controlled quantity is marginal unsafe incidence, not UER among executed cases.

**Empirical RCSE frontier:** choose the maximum-coverage feasible no-GATHER tau/beta pair on resolved calibration benchmark episodes at fixed risk caps. This is empirical selection, not a finite-sample guarantee for clustered synthetic episodes. Test outcomes do not select the point.

## Metrics, uncertainty, ambiguity, and horizons

Coverage is direct executions divided by evaluated episodes. Conditional UER is unsafe direct executions divided by direct executions; it is undefined with no executions. Exposure-weighted UER weights executions by absolute exposure. Modeled loss uses the declared cost scale and is not measured organizational savings. Action rates sum to one.

Historical full-policy UER maps ambiguous natural cases through benchmark reference action: reference EXECUTE becomes safe; other actions become unsafe. Additional results report resolved-only UER and bounds treating every ambiguous execution as safe or unsafe. Different denominators and conventions must be disclosed.

AUROC comparisons use 2,000 paired natural source-case resamples. Policy/loss intervals resample source-case clusters, preserving synthetic copies together. Models remain fixed, so these intervals do not capture training variability.

Fixed-horizon sensitivity uses events strictly after t0 through t0+30/60/90 days. Missing required GR or an adverse event within the horizon means unsafe; clearing with complete evidence and no adverse event means safe; remaining cases stay ambiguous. The fixed model is not retrained for each horizon. The maximum observed timestamp is only a boundary proxy: the source contains no external outcome-capture-completeness indicator.


## Exposure overlap and loss sensitivity

Exposure is `abs(po_value_initial_eur)`. PO, invoice, and goods-receipt amounts are predictors, so the highest-exposure held-out stratum tests covariate extrapolation together with consequence scaling. Holding fitted scores fixed while changing the decision consequence does not isolate the effect of amounts on model learning.

`revision_complete/run_loss_label_sensitivity.py` evaluates fixed no-GATHER predictions under original fallback, resolved-only, and all-ambiguous-safe/unsafe conventions. Paired source-case bootstrap intervals use 2,000 replicates and seed 20261004. Resolved-only removes unresolved episodes from the loss denominator; the other conventions retain the natural benchmark cohort. These sensitivities are bounds, not new observed outcomes.
