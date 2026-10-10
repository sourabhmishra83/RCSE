# Risk Calibrated Selective Execution (RCSE)

Research code, derived datasets, saved models, experiment outputs, and technical documentation for Procure-to-Pay decision control using the BPI Challenge 2019 event log.

## Start here

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

Corrected shared-target evaluation supports an IID/temporal safety-score discrimination advantage; an exposure-OOD advantage is not established. Monetary mismatch features do not establish an AUROC gain. Independent chronology and coherent-value substitution bypass the gate. Always-escalate has lower full-benchmark modeled loss at reference costs, and temporal unsafe-execution estimates are strongly label-dependent.

Historical GATHER uses observed t1 information at t0 and is retrospective. Additional prospective comparisons disable GATHER. Complete outcome capture cannot be established from the log. LTT conditional risk, CRC marginal risk, resolved-only UER, and historical fallback UER have distinct meanings. See the methods and results guides before interpreting them.

## Data and code terms

Source: [BPI Challenge 2019, 4TU.ResearchData](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1). The existing source declaration identifies CC BY 4.0; retain source attribution and verify applicable redistribution terms. Raw logs are excluded. Original code and code documentation are MIT licensed; source-data attribution terms are retained separately in LICENSE_SCOPE.md.


Original code and accompanying code documentation are MIT licensed; see `LICENSE` and `LICENSE_SCOPE.md` for the separate source-data attribution terms. The pre-resubmission label-loss sensitivity is documented in `docs/RESULTS.md` and reproduced by `revision_complete/run_loss_label_sensitivity.py`.


The immutable research snapshot is [RCSE v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.23273149), DOI 10.5281/zenodo.23273149. It archives the versioned [GitHub release](https://github.com/sourabhmishra83/RCSE/releases/tag/v1.0.0).
