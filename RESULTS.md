# Main results at a glance

The values below are reported aggregate summaries from the thesis evaluation protocol. They are included to communicate the research outcome; no patient-eye-level records are included.

| Component | Reported result | Interpretation |
|---|---:|---|
| No-change baseline | 2.55 dB MAE | Short-term persistence reference |
| Pointwise linear baseline | 4.61 dB MAE | Sensitive two-visit extrapolation reference |
| Regional LSTM, scratch | 2.30 dB MAE | Target-cohort training from random initialisation |
| Regional LSTM, zero-shot transfer | 2.27 dB MAE | Bern checkpoint applied without GRAPE updates |
| Regional LSTM, fine-tuned transfer | 2.22 dB MAE | Bern initialisation adapted on GRAPE |
| Six-region VF-only Transformer | 2.16 dB MAE | Final VF-history backbone |

The comparison used the held-out GRAPE test partition with 123 prediction windows from 29 patient-eyes. Optional modalities were assessed separately using frozen VF-only backbones, validation-only selection, repeated seeds, eye-balanced summaries, and paired patient-eye bootstrap comparisons. The thesis reports no consistent additional test-set improvement from RNFL, IOP, or fundus-image pathways under that protocol.

These are descriptive cohort-level results. The table does not provide patient-level predictions, confidence for a clinical decision, or evidence of superiority outside the prespecified evaluation setting.
