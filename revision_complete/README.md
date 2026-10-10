# Additional RCSE experiments

This directory contains executed numerical analyses, saved models/predictions, CSV tables, figures, protocols, and manifests. It contains no public manuscript or reviewer-response deliverables.

Read the repository [reproduction guide](../docs/REPRODUCING.md), [methods](../docs/METHODS.md), and [results](../docs/RESULTS.md). The directory name is preserved because experiment scripts resolve their input/output locations relative to it.

## Entry points

- `run_revision_analyses.py`: policy accounting, escalation/cost comparisons, uncertainty, independent stress, ambiguity bounds, and consequence ablation.
- `run_calibration_baselines.py`: separate isotonic/selection halves, threshold baselines, LTT/CRC, and empirical no-GATHER operating points.
- `run_followup_windows.py`: 30/60/90-day source-event labels using fixed model scores; requires raw XES.
- `build_figures.py`: figures from saved CSV evidence.

Run corrected action and mismatch experiments first when rebuilding their model dependencies. The full ordered commands and dependency pins are documented at the repository root. `reproduction_validation.json` records scientific consistency checks that require no document artifacts.

## Limits

GATHER replay is retrospective, temporal outcomes remain ambiguous, and public risk-control guarantees cannot be transferred automatically to shifted or synthetic-cluster populations. Read the specific metric/label conventions in the methods guide.


Original code and accompanying code documentation are MIT licensed; see `LICENSE` and `LICENSE_SCOPE.md` for the separate source-data attribution terms. The pre-resubmission label-loss sensitivity is documented in `docs/RESULTS.md` and reproduced by `revision_complete/run_loss_label_sensitivity.py`.


The immutable research snapshot is [RCSE v1.0.0 on Zenodo](https://doi.org/10.5281/zenodo.23273149), DOI 10.5281/zenodo.23273149. It archives the versioned [GitHub release](https://github.com/sourabhmishra83/RCSE/releases/tag/v1.0.0).
