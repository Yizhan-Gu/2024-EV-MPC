# Legacy 2024 EV-MPC prototype

This folder preserves the original handwritten Julia/R research prototype and
its generated artifacts.

## Why it is archived

- `MPC.jl` mixes package installation, preprocessing, forecasting, optimization,
  plotting, and experiment execution in one script.
- The rolling demand-charge state and solver-status handling are incomplete.
- `GMM_forecast.jl` contains incorrect component ranking and swapped ED/PD model
  selection.
- `session_processing.R` expects multiple incompatible generations of input
  columns and cannot run end-to-end against the current root data.
- `MPC.ipynb` is an older divergent implementation with a saved runtime error.
- Saved plots and CSV files were generated before the corrected
  feasibility-preserving charger formulation.

## Layout

- `code/`: source, notebook, Julia environment, and local dependency.
- `development_history/`: editor history retained for provenance.
- `data/`: legacy processed train/test datasets and charger lists.
- `results/`: plots, forecasts, dispatch exports, model files, and debug images.

The source files and this manifest are eligible for Git synchronization.
`data/`, `development_history/`, and `results/` remain local-only to avoid
re-uploading large datasets and thousands of generated artifacts.

These assets are read-only historical references, not validated baselines.
