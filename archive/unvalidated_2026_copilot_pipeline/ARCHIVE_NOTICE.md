# Unvalidated 2026 notebook pipeline

This directory is an archive of the former `2026_Gu_EV_forecast/` pipeline.

The notebook and its saved outputs must not be used as publication evidence:

- It loads a pre-existing canonical result table instead of reproducing every
  method from raw data.
- Charger energy shortfalls were converted into post-hoc costs while strict
  energy columns were set equal to actual demand.
- A NoForecast dominance guard overwrote losing methods with the NoForecast
  result.
- All saved optimization records failed the original optimization sanity flag.

The cleaned session dataset was separated from this archive and moved to
`data/processed/clean_charging_sessions_enhanced.csv` because it remains the
input to the current smoke experiment.

Git retains the notebook, methodology, and draft text for provenance. Generated
figures, tables, model weights, and optimization exports remain local-only.
