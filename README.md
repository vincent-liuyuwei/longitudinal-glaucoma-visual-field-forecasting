# Longitudinal Glaucoma Visual-Field Forecasting

This repository is the public-facing research portfolio for Yuwei Liu's master's thesis in Bioinformatics and Computational Biology at the University of Bern. It demonstrates the research question, modelling decisions, analysis workflow, and headline findings without redistributing restricted clinical data.

## Research question

Can longitudinal visual-field (VF) history predict the next available examination, and do baseline retinal nerve fibre layer measurements, visit-aligned intraocular pressure, or longitudinal colour fundus images provide consistent information beyond VF history?

## What the project contains

- `figures/architecture/` — schematics of the longitudinal LSTM baselines, six-region VF-only Transformer, tokenisation, and optional-modality pathways.
- `figures/results/` — aggregate, non-identifying result visualisations used to communicate the study design and findings.
- `analysis_code/` — selected notebooks and scripts documenting preprocessing, baseline modelling, longitudinal modelling, transfer learning, diagnostics, and reporting; start with `analysis_code/README.md` for a guided reading order and terminology.
- `RESULTS.md` — a concise summary of the main evaluation protocol and findings.
- `DATA_AVAILABILITY.md` — the data-sharing and privacy boundary for this public release.

## Methodological progression

1. Persistence and pointwise linear reference predictors.
2. Global, shared-encoder Full-to-Region, and independent Region-specific LSTM baselines.
3. Bern-to-GRAPE transfer under scratch, zero-shot, and fine-tuned conditions.
4. Six independently parameterised regional VF-only Transformers using full 59-point longitudinal input and spatio-temporal point tokens.
5. Separate RNFL, visit-aligned IOP, and fundus-image cross-attention studies with frozen VF backbones and validation-only model selection.

## Main findings

Under the fixed held-out GRAPE evaluation protocol (123 prediction windows from 29 patient-eyes), the final VF-only Transformer reached a sample-level MAE of approximately 2.16 dB. The matched fine-tuned regional LSTM reached approximately 2.22 dB; scratch and zero-shot transfer were approximately 2.30 dB and 2.27 dB. The no-change and pointwise linear baselines were approximately 2.55 dB and 4.61 dB, respectively.

The optional modalities were evaluated separately rather than jointly. Under the controlled Real, Shuffled, and NULL protocol, none produced a consistent additional held-out test improvement over the VF-only reference. These findings are cohort- and protocol-specific; they are not claims of clinical superiority or universal generalisation.

## Reproducibility

The public repository intentionally contains the workflow and aggregate communication artefacts, not the source records. The scripts require an authorised local data environment and will not run from a fresh clone without the corresponding data and configuration. Notebook outputs are cleared to avoid embedding patient-level values or local workstation paths.

## Important limitation

This is a transparent public research record, not a data release. It should not be used to infer patient-level performance, reproduce restricted-cohort results without permission, or make clinical decisions.

## Contact and citation

Please cite the associated master's thesis and contact the author for questions about the public code organisation or the data-access boundary. The repository does not include a software or dataset licence because the applicable institutional and cohort terms must be confirmed first.
