# Paper artifact

`main_segan.tex` is the current SEGAN-targeted research draft. It is compiled
with the locally available standard `article` class because `elsarticle.cls`
is not installed in this environment. The mathematical content, tables, and
figures can later be transferred to Elsevier's submission template without
changing the model.

Main artifacts:

- `main_segan.tex`: manuscript source with the exact charger projection,
  continuous monthly tariff state, rolling information pattern, conformal
  calibration, and implementation audit.
- `main_segan.pdf`: compiled and visually checked 11-page draft.
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

The full protocol requires the ignored local file
`data/processed/clean_charging_sessions_enhanced.csv`. It intentionally fails
if that input is absent. The compact paper results are tracked, while the
source session table is never uploaded.
