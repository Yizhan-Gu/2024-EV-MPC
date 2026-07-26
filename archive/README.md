# Archive

This directory preserves historical research assets that are not part of the
current reproducible charger-MPC pipeline.

Nothing under `archive/` is imported by `src/`, used by the current tests, or
accepted as publication evidence. Files are retained for provenance and may
contain hard-coded paths, incomplete methods, failed runs, or unvalidated
results.

Git tracks the archive notices and selected source/notebook files. Bulk legacy
results, session-level CSV data, model weights, figures, tables, and editor
history are intentionally retained only in the local working copy.

## Contents

- `legacy_2024/`: handwritten Julia/R prototypes, development history, old
  environments, intermediate datasets, and generated outputs.
- `unvalidated_2026_copilot_pipeline/`: the former notebook-led Python pipeline,
  models, tables, figures, and draft text whose fairness and reproducibility
  checks failed the 2026 audit.

Do not copy numerical results from this directory into the paper without
reimplementing the method in the current pipeline and rerunning it.
