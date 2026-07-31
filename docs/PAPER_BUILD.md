# Paper build and artifact guide

`paper/main_segan.tex` is the current SEGAN-targeted Version 4 research draft.
It is compiled with the locally available standard `article` class because
`elsarticle.cls` is not installed in this environment. The mathematical
content, tables, and figures can later be transferred to Elsevier's submission
template without changing the model.

Main artifacts:

- `paper/main_segan.tex`: manuscript source with fair EV-versus-charger forecasting,
  the physics-constrained additive flexibility target, exact charger
  projection, continuous monthly tariff state, rolling information pattern,
  conformal calibration, and implementation audit.
- `paper/main_segan_v4.pdf`: compiled and visually checked Version 4 draft with the
  full model equations, architecture diagram, task-definition table, and
  model-level diagnostic table.
- `paper/main_segan_v3.pdf`: preserved Version 3 draft.
- `paper/main_segan_v2.pdf`: preserved Version 2 fair-forecast draft.
- `paper/main_segan.pdf`: preserved Version 1 draft from the charger-MPC phase.
- `paper/references.bib`: bibliography.
- `paper/figures/` and `paper/tables/`: generated from compact audited outputs.
- `experiments/results/paper_2023Q3/paper_results_summary.csv`: aggregate
  controller results used by the paper artifact generator.

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
PAPER_BUILD_DIR=/private/tmp/ev_charger_paper_build_v4
mkdir -p "${PAPER_BUILD_DIR}"
(
  cd paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir="${PAPER_BUILD_DIR}" \
    -jobname=main_segan_v4 main_segan.tex
)
cp "${PAPER_BUILD_DIR}/main_segan_v4.pdf" paper/main_segan_v4.pdf
```

The build directory keeps `.aux`, `.bbl`, `.blg`, `.fls`, `.log`, and `.out`
files outside `paper/`; only the completed PDF is copied back. That folder is
reserved for manuscript `.tex`,
versioned `.pdf`, the required `references.bib`, and the `figures/` and
`tables/` subdirectories.

The full protocol requires the ignored local file
`data/processed/clean_charging_sessions_enhanced.csv`. It intentionally fails
if that input is absent. The compact paper results are tracked, while the
source session table is never uploaded.

The three advanced forecast entry points default to a July-only test window
for development. The existing Q3 summaries are frozen tracked artifacts, and
the full protocol passes explicit July--September dates only for intentional
paper replication.
