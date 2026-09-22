# Analysis code

The notebooks and scripts are a curated public record of the modelling workflow. They are intentionally supplied without executed outputs, local workstation paths, restricted data, caches, or model checkpoints.

The early notebooks illustrate public-style visual-field preprocessing and baseline modelling. The selected scripts document the later LSTM, Transformer, transfer-learning, optional-modality, subgroup, and reporting workflows. Running the latter requires authorised data and locally configured paths.

`notebooks/grape_universal_metadata_model_workflow.ipynb` is a sanitised code-only version of the central GRAPE modelling notebook. Its saved outputs and execution records were removed, and local absolute paths were replaced with relative paths. The workflow still refers to authorised Bern/GRAPE data and locally available checkpoints by design; those resources are not included in this repository.

The absence of a runnable result from a fresh clone is deliberate: the source cohorts and derived patient-level artefacts are not part of the public release.
