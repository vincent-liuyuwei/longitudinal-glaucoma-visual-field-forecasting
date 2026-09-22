# Supporting local tools

This directory contains utilities used during development, auditing, or reporting. They are included as a transparent record of the research workflow, but they are not intended to run from a fresh public clone.

| File | Purpose |
|---|---|
| `compute_stage_region_baselines.py` | Computes post-hoc no-change and pointwise-linear baseline summaries for authorised stage-by-region workbooks. |
| `run_grape_method_baseline_adaptations.py` | Runs study-level adaptations of published-method baselines on the native GRAPE 59-point task. These are adaptations, not exact replications of the source papers. |
| `render_local_rerun_report.py` | Renders a local status report for rerun logs and result summaries. |
| `audit_thesis_pdf_structure.py` | Extracts structure and repeated-prose diagnostics from a locally available thesis PDF. |
| `update_stage_region_workbooks_with_baselines.mjs` | Updates authorised stage-by-region workbooks with post-hoc baseline summaries. |

Several tools import thesis-private modules or expect local temporary paths. That is intentional and is described here so that a first-time reader does not mistake this folder for a turnkey package. No restricted inputs, outputs, or workbooks are included.
