# Analysis code

This directory is a curated public record of the modelling workflow behind the thesis. It is organised for inspection and explanation rather than as a pip-installable software package. Outputs, restricted data, caches, model checkpoints, and local workstation paths have been removed.

## A guided reading order

The notebooks are numbered in a useful reading order. They are a curated selection, so the numbering is not intended to reproduce the private development history.

1. [`notebooks/01_visual_field_data_exploration.ipynb`](notebooks/01_visual_field_data_exploration.ipynb) introduces the cohort structure, baseline-per-eye summaries, age correction, and visual-field descriptive statistics.
2. [`notebooks/02_autoencoder_baseline_example.ipynb`](notebooks/02_autoencoder_baseline_example.ipynb) gives a compact autoencoder baseline example.
3. [`notebooks/03_build_longitudinal_prediction_sequences.ipynb`](notebooks/03_build_longitudinal_prediction_sequences.ipynb) shows how consecutive historical VF visits become next-visit prediction samples.
4. [`notebooks/04_visual_field_modeling_examples.ipynb`](notebooks/04_visual_field_modeling_examples.ipynb) contains supplementary visualisation, regression-baseline, and temporal or positional-representation examples.
5. [`notebooks/05_grape_cohort_transfer_metadata_workflow.ipynb`](notebooks/05_grape_cohort_transfer_metadata_workflow.ipynb) is the main code-oriented record of the GRAPE transfer-learning and optional-metadata workflow.

## Directory map

| Directory | Purpose |
|---|---|
| `notebooks/` | Selected exploratory, teaching, and modelling notebooks. All saved outputs are cleared. |
| `scripts/` | Small scripts with a relatively self-contained public purpose, including visual-field layout figure generation. |
| `tools/` | Supporting utilities for local baseline calculations, adapted methods, PDF audits, rerun reports, and workbook updates. These are not expected to run from a fresh clone. |

The script-specific notes are in [`scripts/README.md`](scripts/README.md) and [`tools/README.md`](tools/README.md).

## What the code represents

The workflow progresses from simple persistence and pointwise linear references to recurrent LSTM baselines, Bern-to-GRAPE transfer learning, and a six-region VF-only Transformer. Separate optional-modality routes use frozen VF representations and trainable cross-attention modules for RNFL, visit-aligned IOP, or fundus-image inputs.

The central GRAPE notebook is a sanitised code-only version. Its saved outputs and execution records were removed, and local absolute paths were replaced with relative paths. It still refers to authorised Bern/GRAPE data and locally available checkpoints by design; those resources are not included here.

## Running expectations

The public repository is not a fresh-clone reproduction package. The analysis scripts require an authorised local data environment, the corresponding cohort-specific configuration, and—where applicable—locally trained or approved checkpoints. Do not replace the missing resources with guessed or synthetic patient data and do not commit local outputs back to the public repository.

## Terminology

- **VF**: visual field.
- **TD**: total deviation, the pointwise visual-field measure used for modelling.
- **GRAPE**: the target clinical cohort used in the transfer-learning experiments.
- **AE**: autoencoder.
- **RNFL**: retinal nerve fibre layer.
- **IOP**: intraocular pressure.
- **MAE**: mean absolute error, reported in dB after inverse transformation to the original VF scale.

The absence of a runnable result from a fresh clone is deliberate: the source cohorts and derived patient-level artefacts are not part of the public release.
