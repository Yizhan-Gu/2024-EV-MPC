# Data

## Current processed input

- `processed/clean_charging_sessions_enhanced.csv`: current session table used
  by the deterministic smoke experiment. It is intentionally ignored by Git
  because the current file is about 24 MB and derives from restricted local
  source data.

## Local raw input

- `raw/CP_UCSD_clean_Jul16_Sep24.csv`: large local ChargePoint export. It is
  intentionally ignored by Git.

Legacy train/test splits and earlier cleaned tables are preserved under
`archive/legacy_2024/data/`. All session-level CSV datasets are local-only;
compact aggregate experiment summaries remain eligible for version control.

The current data loader records infeasible rows instead of silently changing
their requested energy. Publication preprocessing will need a documented policy
for partial-slot availability before multi-day experiments.
