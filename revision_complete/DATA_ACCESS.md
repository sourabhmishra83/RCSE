# Source data and included derived inputs

Source: B. van Dongen, *BPI Challenge 2019*, version 1, 4TU.ResearchData (2019), [DOI 10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1). The existing source-data declaration specifies CC BY 4.0. Retain attribution and verify applicable repository redistribution terms for source and derived data.

Download and decompress the source log. Place it at `BPI_Challenge_2019.xes` in the repository root to reconstruct fixed-horizon labels. The executed horizon run used SHA256 `af63bc687fc4152f2123b05c3af7772b37ef3fce2d3f67f812666c9e356baae7`. The raw log and its duplicate copies are excluded from Git.

The public repository includes original frozen derived inputs: `rcse_output/rcse_base_v2.parquet`, `rcse_output/experimental_benchmark/splits/rcse_experimental_with_splits.parquet`, original safety models/predictions, and v2 policy datasets/results. Large duplicate CSV exports are replaced by their included Parquet counterparts. The supported additional-experiment rerun sequence can use these inputs directly; only horizon reconstruction additionally needs the raw XES.

Historical construction scripts, split metadata, feature manifests, specifications, and input hashes are retained. Reconstructing the complete historical pipeline from raw events requires selecting the proper script versions and arguments; it is not a tested one-command build. See [the reproduction guide](../docs/REPRODUCING.md) for ordered additional-experiment commands, required paths, and the historical stage map.
