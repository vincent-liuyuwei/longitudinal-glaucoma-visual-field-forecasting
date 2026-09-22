# Architecture figures

These diagrams explain the modelling progression without showing patient measurements.

| Figure group | What it explains |
|---|---|
| `global_lstm_schematic_v2.png`, `full_to_region_lstm_schematic_v2.png`, `region_specific_lstm_schematic_v2.png` | The three longitudinal LSTM baseline designs. |
| `transformer_six_independent_regions.png`, `final_vf_only_transformer_overview.png`, `final_vf_only_transformer_architecture.png`, `final_vf_only_transformer_architecture_detail.png` | The final six-region VF-only Transformer and its regional reassembly. |
| `transformer_tokenisation_self_attention_clean.png` | Point-level spatio-temporal tokenisation and self-attention. |
| `lstm_iop_adaptation_schematic.png` | The exploratory IOP adaptation branch. |
| `glaucoma_progression_ai_irregular.png`, `vf_td_map_schematic_white.png` | Background visual explanations of glaucoma progression and VF/TD representation. |

The six regions are fixed anatomical partitions of the native 59-point VF layout: Supero-Nasal, Supero-Temporal, Macular, Infero-Nasal, Infero-Temporal, and Temporal.
