# Results and where to find the evidence

These results concern one Procure-to-Pay log and linear models. Negative findings are retained. They do not establish production safety or universal economic superiority.

## Shared-target discrimination

Calibrated AUROC on identical natural D1-resolved test cases:

| Regime | Action | Safety | Safety minus action, 95% paired interval |
| --- | ---: | ---: | --- |
| IID | .8782 | .9465 | .0682 [.0605, .0765] |
| Temporal | .9224 | .9607 | .0383 [.0318, .0452] |
| Exposure OOD | .7312 | .7494 | .0182 [-.0202, .0595] |

On evidence-complete cases, action/safety AUROC is .4681/.7666, .5892/.7925, and .5910/.6189 respectively. The paired IID and temporal differences remain positive; the OOD interval includes zero. See `rq2_revision/rq2_point_estimates.csv`, `rq2_paired_bootstrap_ci.csv`, and row-level paired Parquet predictions. This replaces inference from comparisons that used different targets and populations.

## Mismatch features and gate integrity

Adding the two monetary-ratio features establishes no AUROC improvement in any regime or evidence-complete subset: every paired difference interval includes zero. See `safety_mismatch_revision/paired_auroc_ci.csv` and `natural_metrics.csv`.

At a fixed .95 score threshold, adding the gate reduces unsafe/selected full-benchmark counts from 16,462/21,320 to 1,084/5,904 IID, 21,517/23,877 to 1,423/3,750 temporal, and 140/166 to 8/33 OOD. Coverage falls alongside unsafe counts. These score-filter diagnostics are not matched-coverage full-policy comparisons.

Both independent corruptions bypass the gate in every originally valid natural state: 20,429 IID, 21,119 temporal, and 1,398 OOD cases per corruption. See `revision_complete/independent_gate_stress.csv` and corresponding stress Parquet files. Development-aligned block rates cannot support a general corruption-detector claim.

## Reference no-GATHER policy and explicit escalation

| Full benchmark regime | Executed | Unsafe | Coverage | Fallback UER | Modeled loss | Escalate-all loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IID | 6,376 | 1,077 | 17.27% | 16.89% | .1035 | .1000 |
| Temporal | 8,232 | 1,417 | 21.91% | 17.21% | .1058 | .1000 |
| Exposure OOD | 23 | 5 | .91% | 21.74% | .1017 | .1000 |

Full-benchmark paired no-GATHER-minus-escalate loss intervals are [.0024,.0046], [.0044,.0071], and [-.0005,.0045]. Natural-only loss is .0547, .0329, and .1028 under the historical fallback. Thus natural savings are conditional, and OOD superiority is not established. All eight historical policies plus always-escalate, raw counts, action rates, and exposure-weighted UER appear in `all_policy_absolute_results.csv`.

For no-GATHER with rhoH=0, execution counts stay fixed at 6,376 IID, 8,232 temporal, and 23 OOD across the human-cost grid. The grid reprices those decisions; its first tested loss crossovers are cH=.20 IID/temporal and .30 OOD. The full 25-setting cost grid is `v2_human_cost_sensitivity.csv`. These are assumed cost scenarios, not observed savings.

The consequence-only OOD ablation changes coverage from .91% to 1.50%, and common-scale loss from .1017 to .1026. Uniform consequence selects 38 executions/eight unsafe versus 23/five with log consequence. IID/temporal effects are small. See `consequence_only_ablation.csv` and the matching figure.

## Ambiguity materially changes risk estimates

| Regime | Ambiguous direct executions | Original fallback UER | Resolved-only UER | All ambiguous unsafe bound |
| --- | ---: | ---: | ---: | ---: |
| IID | 485 | 16.89% | 18.28% | 24.50% |
| Temporal | 4,489 | 17.21% | 37.86% | 71.74% |
| Exposure OOD | 2 | 21.74% | 23.81% | 30.43% |

See `ambiguous_uer_sensitivity.csv` for both safe/unsafe bounds and every policy. The original low temporal risk estimate cannot be interpreted without its fallback convention.

## Calibration-only selection and risk-control baselines

At a 10% natural conditional-risk target, the LTT safety baseline gives IID coverage 87.59% and observed UER 4.12%; temporal 82.09%/4.99%; OOD 81.76%/20.00%. Its nominal guarantee requires the stated IID exchangeability assumptions. CRC instead controls a marginal unsafe-per-case quantity. See `calibration_selected_baselines.csv`, `ltt_candidate_audit.csv`, `fair_threshold_sweeps.csv`, and the disjoint selection/test Parquet files.

No empirical full-benchmark RCSE point is selected at caps through 16%. At 18%, IID test coverage/UER is 15.12%/18.03% and OOD .49%/33.33%. At 20%, temporal gives 9.55%/36.18% and OOD 3.16%/43.59%. Calibration feasibility does not imply test risk control. See `calibration_selected_frontier.csv` and `figures/calibration_frontier.*`.

## Fixed follow-up windows

At both 60 and 90 days, 14,285 of 23,374 temporal held-out natural cases remain ambiguous. See `fixed_horizon_safety_sensitivity.csv`, `fixed_horizon_label_transitions.csv`, `fixed_horizon_targets.parquet`, and `followup_window_manifest.json`. Fixed-window evaluation changes the target/population while holding model scores fixed. It does not supply external proof of complete follow-up.

## Result file map

Paths in this table are relative to `revision_complete/` unless noted.

| Evidence | Files |
| --- | --- |
| Absolute policy outcomes | `all_policy_absolute_results.csv` |
| Paired loss vs escalation | `paired_loss_vs_always_escalate_ci.csv` |
| Original consequence modes | `four_consequence_modes_frozen.csv` |
| Cost sensitivity | `v2_human_cost_sensitivity.csv` |
| Source composition | `benchmark_composition.csv` |
| Train/calibration/test timing | `split_followup_details.csv` |
| Maturity restriction sensitivity | `followup_maturity_sensitivity.csv` |
| Corrected natural action objective | `corrected_action_natural_performance.csv` |
| Independent calibration fit audit | `separate_calibration_training_audit.csv` |
| Original intervals/decomposition | `frozen_evidence/` |
| Feature sets, protocols, input hashes | JSON manifests in all three experiment directories |
| Numerical progress | `analysis_run.log`, `calibration_run.log`, `followup_run.log`, experiment `run.log` files |
| Figures | `figures/*.png` and `figures/*.svg` |

The original deeply nested `rcse_output` tree is retained for reproducibility. Figure 3 shows selected operating points; it is not a guarantee curve. The original GATHER replay is retrospective because it uses observed t1 information at t0.


## Natural-only loss under unresolved-outcome conventions

| Evaluation | Original fallback | Resolved-only | All ambiguous executions unsafe |
| --- | --- | --- | --- |
| IID | 0.0547 | 0.0548 | 0.0873 |
| Temporal | 0.0329 | 0.0570 | 0.3838 |
| Exposure OOD | 0.1028 | 0.1035 | 0.1065 |

Always-escalate costs 0.1000 at the reference settings. IID remains lower even under the pessimistic convention; temporal improvement reverses. Exposure-OOD improvement is not established. The resolved-only cohort uses a different denominator. Exact paired intervals and episode counts are in `revision_complete/natural_loss_label_sensitivity.csv`; its script does not retrain or retune policies.

At the 10% IID target, LTT action/safety choose threshold .05 and gate-safety chooses .00. Sharing a selected grid point does not make LTT and CRC risks equivalent. Those higher-coverage baselines do not establish RCSE superiority.


The immutable research snapshot is [RCSE v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.23273149), DOI 10.5281/zenodo.23273149. It archives the versioned [GitHub release](https://github.com/sourabhmishra83/RCSE/releases/tag/v1.0.0).
