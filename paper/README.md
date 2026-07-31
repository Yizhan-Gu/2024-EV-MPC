# Paper artifact

`main_segan.tex` is the current SEGAN-targeted Version 4 research draft. It is compiled
with the locally available standard `article` class because `elsarticle.cls`
is not installed in this environment. The mathematical content, tables, and
figures can later be transferred to Elsevier's submission template without
changing the model.

Main artifacts:

- `main_segan.tex`: manuscript source with fair EV-versus-charger forecasting,
  the physics-constrained additive flexibility target, exact charger
  projection, continuous monthly tariff state, rolling information pattern,
  conformal calibration, and implementation audit.
- `main_segan_v4.pdf`: compiled and visually checked Version 4 draft with the
  full model equations, architecture diagram, task-definition table, and
  model-level diagnostic table.
- `main_segan_v3.pdf`: preserved Version 3 draft.
- `main_segan_v2.pdf`: preserved Version 2 fair-forecast draft.
- `main_segan.pdf`: preserved Version 1 draft from the charger-MPC phase.
- `references.bib`: bibliography.
- `figures/` and `tables/`: generated from the compact audited CSV outputs.
- `results_summary.csv`: paper-level aggregate controller results.

Run the fast tests:

```bash
PYTHONPATH=src /opt/homebrew/bin/python3 -m unittest discover -s tests -v
```

Reproduce the full chronological validation, Q3 test, scalability benchmarks,
figures, tables, and PDF:

```bash
bash experiments/run_paper_protocol.sh
```

Rebuild only the fair forecast tables and figures:

```bash
MPLBACKEND=Agg \
  MPLCONFIGDIR=/private/tmp/mpl-fair \
  PYTHONPATH=src \
  .venv/bin/python experiments/make_fair_forecast_artifacts.py
```

Rebuild only the physics-flexibility tables and figures:

```bash
MPLBACKEND=Agg \
  MPLCONFIGDIR=/private/tmp/mpl-flex \
  PYTHONPATH=src \
  .venv/bin/python experiments/make_flexibility_artifacts.py
```

Compile the versioned draft:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -jobname=main_segan_v4 main_segan.tex
```

The full protocol requires the ignored local file
`data/processed/clean_charging_sessions_enhanced.csv`. It intentionally fails
if that input is absent. The compact paper results are tracked, while the
source session table is never uploaded.

The three advanced forecast entry points default to a July-only test window
for development. The existing Q3 summaries are frozen tracked artifacts, and
the full protocol passes explicit July--September dates only for intentional
paper replication.
