# Risk Calibrated Selective Execution (RCSE)

Research code, derived datasets, saved models, experiment outputs, and technical documentation for Procure-to-Pay decision control using the BPI Challenge 2019 event log.

## Start here

- [Value-conditional calibration check](revision_complete/value_calibration_extension/README.md): held-out risk by value, separate isotonic calibration within training-value bins, and matched scalar selection.
- [Consequence alignment and repeated vendors](revision_complete/alignment_extension/README.md): permuted-consequence placebos, ten vendor holdouts, grouped policy contrasts, and tail diagnostics.
- [Grouped holdouts and component audit](revision_complete/grouped_component_extension/README.md): PO/vendor isolation, matched eight-chain safety selection, plug-in cost policy, cluster intervals, and absolute exposure totals.
- [Current full-population study](docs/ROBUSTNESS_STUDY.md): all eligible cases, repeated sequential selection, earlier temporal refitting, nonlinear models, and cost sensitivity.
- [Reproduction guide](docs/REPRODUCING.md): environment, verified inputs, ordered commands, raw-log access, historical pipeline map, and output comparison.
- [Detailed methods](docs/METHODS.md): labels, source-grouped splits, models/calibration, evidence gate, consequence, policy, baselines, and uncertainty.
- [Results and evidence map](docs/RESULTS.md): numerical findings, saved files, and limitations.
- [Source data](revision_complete/DATA_ACCESS.md): source attribution and raw XES access.

## Quick start

Use Python 3.12. Create and activate a virtual environment, then run from the cloned repository root:

```bash
git clone https://github.com/sourabhmishra83/RCSE.git
cd RCSE
python -m pip install -r requirements-reproduction.txt
python tools/verify_reproduction.py
```

The verifier checks the included scientific evidence without retraining or requiring the raw log. See the full guide for environment activation and the ordered experiment reruns. The included Parquet datasets replace several large duplicate CSV exports. The raw XES file must be downloaded separately for fixed-horizon label reconstruction.

## Repository layout

| Directory or files | Contents |
| --- | --- |
| Numbered root scripts | Historical data construction, modeling, gate/policy evaluation, diagnostics |
| `rcse_output/` | Frozen derived datasets, models/predictions, and original experiment results |
| `bpic2019_output/` | Source-log inspection outputs |
| `rq2_revision/` | Corrected shared-target action/safety comparisons and evidence-complete subsets |
| `safety_mismatch_revision/` | Two-feature sensitivity, trained models, and predictions |
| `revision_complete/` | Stress tests, cost/label/horizon sensitivities, independent calibration baselines/frontiers, and figures |
| `docs/` | Reproduction, methodology, and numerical-result explanations |
| `tools/` | Scientific artifact verification |

The existing directory names preserve script/input paths. Alternative historical script versions are retained for traceability; their numerical prefixes are not a command to run every version in sequence. Manuscripts, reviewer comments, response documents, writing drafts, and document-generation code are excluded from the public repository.

## Main findings and limits

The current study evaluates all 210,692 eligible natural cases, with resolved-case policy comparisons and explicit ambiguity bounds. Complete-evidence AUROC under PO grouping is 0.7406 for logistic regression and 0.8026 for histogram gradient boosting (cluster gap 0.0620 [0.0424, 0.0824]); the original vendor split gives both about 0.805. Across ten vendor allocations, the boosted-minus-logistic gap has median +0.0238 and range -0.0214 to +0.0774, positive in eight splits. A single near-zero result does not establish general absence of a cold-start advantage. Sibling-item overlap therefore does not fully explain the PO-grouped advantage, but it does not generalize across group definitions or shifts. The plug-in cost policy is compared with stricter reference constraints. Cluster intervals support the fixed-horizon temporal gate benefit on log-scale loss; the logistic linear temporal reversal is uncertain and concentrated. Equal eight-chain procedures show selection-power differences. Permuting consequences reduces logistic graph certification from 19/20 to a median 0/20 in item IID and from 15/20 to median 0/20 in fixed-horizon temporal data, Value-quartile isotonic calibration then lets the gated logistic scalar graph certify in 20/20 item-IID and temporal splits and 9/20 PO-grouped splits, compared with RCSE at 19/20, 15/20 and 7/20. The residual value dependence supports a value-conditional calibration account, rather than a unique consequence-weighting advantage. Strict reference constraints cost loss relative to plug-in in IID/temporal tests but reduce loss under exposure shift.

Exact break-even curves distinguish fixed-decision repricing from retuning. Log, linear, and capped-linear consequences and PO-value-weighted risk are provided. The fixed-horizon temporal refit uses 90-day outcomes; capture completeness remains unknown. Integrity probes expose the finite gate specification and score sensitivity, rather than proving general corruption detection. Historical GATHER replay remains retrospective. Older directories retain different benchmark populations and model provenance; use the current study guide for the main full-population analysis.

## Data and code terms

Source: [BPI Challenge 2019, 4TU.ResearchData](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1). The existing source declaration identifies CC BY 4.0; retain source attribution and verify applicable redistribution terms. Raw logs are excluded. Original code and code documentation are MIT licensed; source-data attribution terms are retained separately in LICENSE_SCOPE.md.


Original code and accompanying code documentation are MIT licensed; see `LICENSE` and `LICENSE_SCOPE.md` for the separate source-data attribution terms. The label-loss sensitivity is documented in `docs/RESULTS.md` and reproduced by `revision_complete/run_loss_label_sensitivity.py`.


The current archived reproduction package is [RCSE v1.4.0 on Zenodo](https://doi.org/10.5281/zenodo.23619940), DOI **10.5281/zenodo.23619940**. It includes the full-population and matched-natural analyses, saved models and scores, all numerical outputs, and reproduction instructions. The corresponding [GitHub release is v1.4.0](https://github.com/sourabhmishra83/RCSE/releases/tag/v1.4.0).

Previous versions remain available: [v1.3.1](https://doi.org/10.5281/zenodo.23618321), [v1.2.0](https://doi.org/10.5281/zenodo.23608623), [v1.1.0](https://doi.org/10.5281/zenodo.23608129), [v1.0.1](https://doi.org/10.5281/zenodo.23283511) and [v1.0.0](https://doi.org/10.5281/zenodo.23273149). All belong to the same [Zenodo DOI series](https://doi.org/10.5281/zenodo.23273148).
