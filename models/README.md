# Model inventory

Trained weights and checkpoints are not included in the public repository. The model designs are documented by the architecture figures and the analysis-code guide; an authorised user must train or obtain any weights within the permitted local environment.

## Model family used in the thesis

1. **Reference predictors** — no-change persistence and pointwise linear extrapolation.
2. **LSTM baselines** — Global, shared-encoder Full-to-Region, and independent Region-specific variants.
3. **Transfer-learning variants** — Bern pretraining followed by scratch, zero-shot, or fine-tuned evaluation on GRAPE.
4. **VF-only Transformer** — six independently parameterised regional models receiving the complete longitudinal 59-point VF history and reconstructing the complete future field after regional reassembly.
5. **Optional-modality routes** — separate RNFL, visit-aligned IOP, and fundus-image cross-attention pathways with the VF backbone frozen.

The public figures show the conceptual differences between these families. They should not be interpreted as a release of the trained clinical models.
