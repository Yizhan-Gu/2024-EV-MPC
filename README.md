# EV Forecasting and Charger-Level MPC

Current research code for forecasting UCSD EV charging sessions and comparing
model-predictive charging strategies at two aggregation levels:

- EV-level scheduling, where each charging session is optimized separately.
- Charger-level scheduling, where sessions are aggregated by station and port.

The intended workflow covers charger-level probabilistic forecasting,
feasibility-preserving 15-minute rolling MPC, V0G baselines, tariff and
peak-demand cost calculation, and forecast/control performance comparisons.

## Project layout

- `src/charger_mpc/`: current deterministic EV and charger optimization core.
- `tests/`: fast physical, optimization, and regression checks.
- `experiments/`: small reproducible experiments and their compact outputs.
- `data/`: current processed input and ignored local raw data.
- `docs/`: reference tariff and historical presentation documents.
- `archive/`: preserved legacy code and unvalidated results. Nothing under this
  directory is used by the current pipeline.

## Validation status

The deterministic reference core passes the current unit and smoke tests.
Three independent six-charger runs are summarized in
`experiments/results/README.md`. They are controller checks rather than
publication results and still apply monthly demand-charge rates to isolated
days.

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
PYTHONPATH=src /opt/homebrew/bin/python3 -m unittest discover -s tests -v
```

Small six-charger smoke experiment:

```bash
/opt/homebrew/bin/python3 experiments/smoke_charger_mpc.py \
  --chargers 6 \
  --time-limit 1.0
```

The one-day smoke test applies monthly demand-charge rates to one isolated day
only as a controller check. Publication experiments must carry the executed
monthly peak state across consecutive days.
