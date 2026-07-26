# EV Forecasting and Charger-Level MPC

> **Read first:** [`docs/RESEARCH_PROGRESS.md`](docs/RESEARCH_PROGRESS.md)
> records the last verified milestone, exact commands, result boundaries, and
> next experiments. Future work should update that file after every major
> completed phase.

Current research code for forecasting UCSD EV charging sessions and comparing
model-predictive charging strategies at two aggregation levels:

- EV-level forecasting and scheduling, where `driver_id` is the prediction
  entity and each realized session is optimized separately.
- Charger-level forecasting and scheduling, where each physical
  `station_name|port` is the prediction and control entity.

The intended workflow covers charger-level probabilistic forecasting,
feasibility-preserving 15-minute rolling MPC, V0G baselines, tariff and
peak-demand cost calculation, and forecast/control performance comparisons.

## Project layout

- `src/charger_mpc/`: current deterministic EV and charger optimization core.
- `src/charger_forecasting/`: leakage-safe EV/charger panels, metrics, and
  advanced forecasting models.
- `tests/`: fast physical, optimization, and regression checks.
- `experiments/`: small reproducible experiments and their compact outputs.
- `data/`: current processed input and ignored local raw data.
- `docs/`: reference tariff and historical presentation documents.
- `archive/`: preserved legacy code and unvalidated results. Nothing under this
  directory is used by the current pipeline.

## Validation status

The deterministic reference core passes the current unit and smoke tests. The
paper protocol now adds causal historical-median and one-sided conformal
forecasts, exact sparse charger envelopes, continuous executed monthly peak
states, a chronological June validation, and a July--September 2023
out-of-sample pilot. All 552 Q3 day--method runs serve the required energy
without fallback.

The current six-port pilot finds a 28.23% conformal-MPC cost saving relative to
immediate charging and a 1.47% reduction relative to point-forecast MPC.
Because only three billing months are tested, this is a reproducible pilot and
not yet a statistically generalizable publication claim. The complete draft,
formula audit, figures, and commands are under `paper/`.

An advanced forecast-only scaffold is also validated for genuinely distinct
EV- and charger-level tasks. It includes seasonal naive, ridge, decomposition
linear, LSTM, TCN, iTransformer, and a charger graph-temporal regressor. Its
tracked seven-day output is only a pipeline sanity check; it is not a paper
result and has not yet been coupled to MPC.

Historical Julia/R code, notebook pipelines, large debug outputs, old models,
and their result tables are archived with explicit warnings. Their numerical
claims must not be used in the paper without reimplementation and rerunning in
the current pipeline.

## Deterministic charger-MPC reference

The corrected reference implementation is in `src/charger_mpc/`. It provides:

- An exact EV session-level linear program.
- A feasibility-preserving charger-level cumulative-energy envelope.
- Physical disaggregation checks back to individual sessions.
- Rolling execution against actual arrivals with explicit unserved energy.
- A deliberately relaxed legacy model used only to demonstrate false savings.

Fast unit tests:

```bash
MPLCONFIGDIR=/private/tmp/mpl-models \
  PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -v
```

Small six-charger smoke experiment:

```bash
/opt/homebrew/bin/python3 experiments/smoke_charger_mpc.py \
  --chargers 6 \
  --time-limit 1.0
```

Continuous three-day billing experiment:

```bash
/opt/homebrew/bin/python3 experiments/continuous_3day_mpc.py \
  --chargers 6 \
  --time-limit 1.0
```

The one-day script intentionally retains the isolated-day billing proxy for
regression diagnostics. The continuous script is the valid starting point for
longer controller and forecast comparisons.

## Paper-level protocol

Run the full frozen protocol (local processed data required):

```bash
bash experiments/run_paper_protocol.sh
```

The compiled manuscript is `paper/main_segan.pdf`, with source in
`paper/main_segan.tex`. Large source/session CSVs remain ignored; only compact
audited experiment summaries are tracked.

## Advanced forecast-only benchmark

Run the lightweight sanity experiment before any longer training:

```bash
MPLCONFIGDIR=/private/tmp/mpl-models \
  PYTHONPATH=src \
  .venv/bin/python experiments/advanced_forecast_benchmark.py \
  --quick \
  --output experiments/results/advanced_forecast_quick/metrics.csv
```

The full default split ends on 2023-09-30 and is intentionally not run by this
command. See `docs/RESEARCH_PROGRESS.md` for the task definitions, fairness
boundary, and next required experiments.
