# Research progress: EV versus charger forecasting and MPC

Last verified: 2026-07-30 (Version 4 mathematics, model diagnostics,
paper-figure audit, and clean paper build layout)

This is the read-first handoff for future Codex sessions. Do not infer that a
long experiment, reference-paper reproduction, or advanced-forecast/MPC
coupling is complete
unless it is explicitly listed below with an output file and validation
command.

## Scientific question

The paper asks whether changing the prediction and control entity from an
individual EV to a physical charger can improve forecasting robustness and
economic MPC performance under V0G charging. The study does **not** include
V2G or demand response.

The comparison must use two genuinely distinct forecasting problems:

1. **EV task:** the entity is `driver_id`; predict that driver's next-day
   charging-session attributes.
2. **Charger task:** the entity is `station_name|port`; predict the next-day
   aggregate session attributes at that physical port.

The archived Copilot pipeline did not meet this definition. It predicted one
shared day profile and only changed the downstream optimization backend. It
must remain archived and must not supply paper claims.

## Verified milestones

### Paper-level charger MPC

- Git commit `78305cd` was pushed to `main`.
- The chronological six-port July--September 2023 pilot completed 552
  day--method cases with zero unserved energy and zero fallback runs.
- Conformal charger MPC reduced billed cost by 28.23% relative to immediate
  charging and by 1.47% relative to point-forecast MPC.
- These are a three-month pilot, not a general statistical conclusion.
- The manuscript produced at that milestone is preserved as
  `paper/main_segan.pdf`; the same source has since been expanded into the
  Version 2 draft described below.

### Advanced forecasting scaffold

Implemented in:

- `src/charger_forecasting/panel.py`
- `src/charger_forecasting/models.py`
- `src/charger_forecasting/training.py`
- `experiments/advanced_forecast_benchmark.py`
- `tests/test_advanced_forecasting.py`

Completed features:

- training-only entity selection and normalization;
- separate EV and charger daily panels;
- seasonal-naive and ridge baselines;
- decomposition-linear (DLinear), LSTM, and TCN regressors;
- variate-token iTransformer;
- charger graph-temporal regression with training-only correlation adjacency;
- calendar covariates, active-session target masking, deterministic seeds;
- normalized and aggregate energy metrics plus explicit scope coverage;
- cross-midnight sessions retained and clipped to the 96-slot operating day;
- no model weights or large intermediate arrays written to Git.

### Fair Q3 EV-versus-charger forecast benchmark

Completed outputs:

- `experiments/results/fair_forecast_q3/coverage_sensitivity.csv`
- `experiments/results/fair_forecast_q3/seed_20260726/`
- `experiments/results/fair_forecast_q3/seed_20260727/`
- `experiments/results/fair_forecast_q3/seed_20260728/`
- `experiments/results/fair_forecast_q3/fair_comparison_summary.csv`
- `experiments/results/fair_forecast_q3/full_scope_summary.csv`

Protocol:

- training targets end 2023-03-31;
- validation targets are 2023-04-01 through 2023-06-30;
- untouched Q3 test is 2023-07-01 through 2023-09-30;
- 28-day lookback and 10 maximum epochs;
- main EV cohort contains all 372 drivers with at least three training
  sessions at the six fixed ports;
- `EV` and `ChargerMatched` contain exactly the same realized sessions;
- `ChargerFull` contains all six-port demand;
- DLinear, LSTM, TCN, iTransformer, and GraphGNN use three seeds;
- deterministic/diagnostic baselines use one seed;
- paired daily uncertainty uses 5,000 seven-day moving-block bootstrap
  resamples after averaging seed-specific losses by test day.

Verified Q3 scope:

- full scope: 817 session records and 8,916.115 kWh;
- matched \(h=3\) scope: 239 sessions and 2,654.173 kWh;
- matched energy coverage: 29.768%;
- even all 1,230 drivers seen at least once in training cover only 41.503% of
  Q3 energy.

Matched aggregate daily energy MAE:

| Model | EV (kWh/day) | Charger (kWh/day) | Charger reduction |
|---|---:|---:|---:|
| Seasonal naive | 23.20 | 23.20 | 0.0% |
| Ridge | 95.24 | 21.17 | 77.8% |
| DLinear | 93.64 | 39.37 | 58.0% |
| LSTM | 103.19 | 29.98 | 70.9% |
| TCN | 96.74 | 26.10 | 73.0% |
| iTransformer | 97.91 | 30.20 | 69.2% |

For DLinear, LSTM, TCN, and iTransformer, the 95% paired block-bootstrap
interval for daily EV-minus-charger absolute error is strictly positive.
Seasonal naive is equal to numerical precision, which confirms that matched
targets and dates are identical and that the seven-day lag commutes with
aggregation.

The first hurdle-ridge implementation is a retained negative result. Balanced
participation classification produces small positive predictions across many
inactive drivers, causing EV aggregate MAE of 1,268.99 kWh/day. Do not omit
this from the results table or present it as a successful EV solution.

The correct operational EV metric is **no substitution**:

\[
L^{\mathrm{EV,op}}_t =
\left|E^{\mathrm{match}}_t-\widehat E^{\mathrm{EV}}_t\right|
+\left(E^{\mathrm{full}}_t-E^{\mathrm{match}}_t\right).
\]

Known-driver overprediction cannot cancel demand from an unidentified driver.
Do not replace this metric with
\(\left|E^{\mathrm{full}}_t-\widehat E^{\mathrm{EV}}_t\right|\), which permits
physically invalid identity substitution.

Full-demand charger results:

- unreplicated ridge point estimate: 29.59 kWh/day and 30.53% WAPE;
- three-seed GraphGNN: \(30.40\pm1.48\) kWh/day and 31.37% WAPE;
- iTransformer is less stable across seeds and must not be called the winner.

### Original set-free flexibility-envelope experiment

Implemented in:

- `src/charger_forecasting/envelope.py`;
- `experiments/envelope_forecast_benchmark.py`;
- `experiments/make_flexibility_artifacts.py`;
- `tests/test_advanced_forecasting.py`.

This is an original charger-native formulation. It does **not** reproduce the
reference paper's session-assignment or retrieval procedure and it has no
Hungarian matching metric. Each day is represented directly by six cumulative
feasibility anchors at 08:00, 12:00, 16:00, 20:00, 21:00, and 24:00:

\[
\mathbf s_{t,n}
=
\left[
\mathbf L_{t,n},\,
\mathbf U_{t,n},\,
\mathbf O_{t,n}
\right],
\]

where \(\mathbf L\) and \(\mathbf U\) are the cumulative required- and
deliverable-energy bounds and \(\mathbf O\) is the occupied-port-equivalent
profile. The signature is additive, so summing it over eligible EVs is
mathematically identical to aggregating the same realized sessions at the six
physical ports. The audited maximum absolute identity residual is
\(4.58\times10^{-5}\) in float32 arithmetic.

The differentiable physics output layer guarantees, without inference-time
repair:

\[
0\le L_b\le U_b\le E,\qquad
L_b\le L_{b+1},\qquad
U_b\le U_{b+1},\qquad
L_6=U_6=E,
\]

plus occupancy-limited cumulative capacity and remaining-energy reachability.
Unconstrained models retain a separate projection diagnostic, but their raw
validity is reported so projection cannot be mistaken for native accuracy.

Pre-commit audit on 2026-07-30 found that remaining-capacity reachability had
initially been applied to the upper curve but omitted from the learned lower
curve and validity mask. The hard lower-bound constraint, NumPy projection,
validity audit, and unit test were corrected; the quick run and all three
paper seeds were then rerun from scratch. All numbers below are post-fix
results.

Physical screening is applied before cohort selection and evaluation:

- six fixed ports, January 2022--September 2023;
- 6,238 raw sessions;
- 120 infeasible-energy and 7 invalid/overnight records removed;
- 94 overlapping port-days containing 474 records removed;
- 5,637 physically unambiguous sessions retained;
- 314 recurring drivers with at least three training sessions;
- Q3 matched energy 2,642.384 kWh versus full energy 8,880.536 kWh
  (29.755% coverage).

Paper-level Q3 outputs use seeds 20260730, 20260731, and 20260732. On the
identical matched signature, physics-iTransformer at charger resolution
reduces aggregate MAE relative to the same EV-resolution architecture by:

| Target | EV physics MAE | Charger physics MAE | Reduction | 95% paired block CI for EV minus charger |
|---|---:|---:|---:|---:|
| Terminal energy | 68.74 +/- 6.87 kWh | 21.18 +/- 2.54 kWh | 69.19% | [42.48, 53.07] kWh |
| Lower envelope | 36.73 +/- 4.31 kWh | 14.43 +/- 0.98 kWh | 60.71% | [18.56, 26.14] kWh |
| Upper envelope | 42.62 +/- 3.74 kWh | 15.31 +/- 0.98 kWh | 64.07% | [23.12, 31.68] kWh |

Seasonal naive is exactly equal at EV and charger resolution for all three
targets, which is the matched-scope pipeline audit. Physics-iTransformer has
100% valid raw signatures; the unconstrained iTransformer has 0% raw validity
and is shown only after an explicitly labeled projection.

On full six-port demand, physics-iTransformer's three-seed mean terminal,
lower-, and upper-envelope MAE values are 33.43, 24.57, and 26.19 kWh,
respectively, versus 36.68, 25.50, and 27.56 kWh for seasonal naive. Individual
seeds do not all beat the baseline, so aggregation still does not establish
robust uniform dominance.

Tracked compact outputs:

- `experiments/results/envelope_forecast_quick/`;
- `experiments/results/flexibility_forecast_q3/`;
- `paper/tables/flexibility_matched_summary.tex`;
- `paper/tables/flexibility_full_summary.tex`;
- `paper/figures/flexibility_matched_scope.{pdf,png}`;
- `paper/figures/flexibility_full_scope.{pdf,png}`;
- `paper/figures/flexibility_daily_error.{pdf,png}`;
- `paper/figures/flexibility_envelope_example.{pdf,png}`.

The top-128 development run was moved to the ignored
`archive/paper_development_runs/envelope_q3_top128_20260730/` directory. It is
not a paper result and will not be pushed.

Critical claim boundary: these six-anchor feasibility forecasts have not yet
been lifted to 96 slots, calibrated into uncertainty sets, or passed through
rolling MPC. They establish a fair feasibility-set forecasting advantage, not
a revenue result.

### Version 4 manuscript, formulas, tables, and figures

Current draft:

- source: `paper/main_segan.tex`;
- compiled versioned PDF: `paper/main_segan_v4.pdf`;
- preserved prior draft: `paper/main_segan_v3.pdf`;
- new tables: `paper/tables/forecast_fair_comparison.tex` and
  `paper/tables/forecast_full_scope.tex`, plus
  `flexibility_task_definition.tex`, `flexibility_model_design.tex`,
  `flexibility_model_diagnostics.tex`, and the two summary tables;
- new paper figures:
  `forecast_matched_scope`,
  `forecast_operational_scope`, and
  `forecast_daily_trajectories`, plus the five flexibility figures, in both
  PDF and PNG. The new `physics_itransformer_architecture` figure is an
  original diagram of this repository's implementation.

Version 4 expands the paper from the abbreviated model description to the
complete training-only standardization, variate-token embedding, multi-head
attention, calendar-logit map, occupancy capacity, softmax increment,
remaining-capacity reachability, cumulative closure, and weighted signature
loss equations. The new task table makes the individual-EV, matched-charger,
and full-charger estimands explicit. The diagnostic table jointly reports
accuracy, seed variation, raw physical validity, flexibility-width error, and
parameter count for every task/model combination.

The matched result is now a connected-dot comparison and the full-demand
result is a normalized forest-style plot relative to seasonal naive. Both
retain vector PDF output and colorblind-safe, limited palettes. The final
23-page PDF was rendered page by page with Poppler; the architecture, formula,
task-definition, diagnostic, result, and appendix pages were visually checked.
The LaTeX log has no overfull boxes, unresolved references, or warnings.

Development compute policy from Version 4 onward:

- `advanced_forecast_benchmark.py`,
  `envelope_forecast_benchmark.py`, and
  `forecast_coverage_sensitivity.py` default to test end `2023-07-31`;
- new model development is July-only unless an explicit frozen replication is
  justified;
- the already completed Q3 results remain valid versioned evidence and are not
  rerun merely to update paper layout;
- `run_paper_protocol.sh` retains explicit Q3 dates because it is the frozen
  full replication command, not the default development command.

Paper-directory hygiene from this milestone onward:

- `paper/` contains only manuscript `.tex`, versioned `.pdf`, the required
  `references.bib`, and the `figures/` and `tables/` subdirectories;
- build instructions moved to `docs/PAPER_BUILD.md`;
- the aggregate controller CSV moved to
  `experiments/results/paper_2023Q3/paper_results_summary.csv`, and its
  generator now writes there;
- `run_paper_protocol.sh` builds under
  `${TMPDIR:-/tmp}/ev_charger_paper_build` and copies only the final PDF back,
  so `.aux`, `.bbl`, `.blg`, `.fls`, `.log`, and `.out` files are not
  recreated in `paper/`;
- `.DS_Store` and prior local LaTeX intermediates were removed.

Critical claim boundary: the advanced forecasts remain forecast-only. The
28.23% V0G saving and 1.47% point-forecast improvement come from the separate
historical-median/conformal MPC pilot, not from iTransformer, GraphGNN, or the
new flexibility model.

## Mathematical benchmark definition

For entity \(n\) and day \(t\), the compact target is

\[
\mathbf y_{t,n} =
\left[
c_{t,n},\,
e_{t,n},\,
a_{t,n},\,
d_{t,n},\,
\ell_{t,n}
\right]^\top ,
\]

where \(c\) is session count, \(e\) is total energy in kWh, \(a\) and \(d\)
are mean arrival and departure slots, and \(\ell\) is mean dwell time in
15-minute slots. The same target definition is used at both entity levels so
that the information boundary is auditable.

The EV cohort is selected using only the training interval:

\[
\mathcal N_{\mathrm{EV}}
=
\operatorname{TopK}_{i}
\left\{
\sum_{t\in\mathcal T_{\mathrm{train}}} c_{t,i}
\right\}.
\]

The charger set \(\mathcal N_{\mathrm{C}}\) is the same frozen six-port set as
the current MPC pilot. No validation or test outcomes enter cohort selection,
scaling, or graph construction.

The neural models minimize a weighted masked loss:

\[
\mathcal L(\theta)
=
\frac{
\sum_{t,n,f} m_{t,n,f}
\left(\hat y_{t,n,f}-y_{t,n,f}\right)^2
}{
\sum_{t,n,f}m_{t,n,f}
}.
\]

Count and energy receive extra weight on active days. Timing targets are masked
on inactive days because arrival and departure are undefined there.

For iTransformer, each variable's full lookback vector is embedded as one
token:

\[
\mathbf z_f
=
W_x\,[x_{t-L+1,f},\ldots,x_{t,f}]^\top+\mathbf e_f,
\qquad
\mathbf H
=
\operatorname{TransformerEncoder}
\left([\mathbf z_1,\ldots,\mathbf z_F]\right).
\]

Attention therefore operates across variables, not across timestamps.

For the graph model, the charger adjacency is constructed from training-period
energy series only. After retaining the top-\(k\) absolute correlations and
adding self-loops,

\[
\widetilde A = D^{-1/2}(A+I)D^{-1/2}.
\]

A shared GRU produces node histories \(\mathbf h_{t,n}\), followed by graph
message passing:

\[
\mathbf g_t
=
\sigma\!\left(
W_s\mathbf H_t
+W_g\widetilde A\mathbf H_t
+W_c\mathbf q_t
\right),
\qquad
\widehat{\mathbf Y}_{t+1}=W_o\mathbf g_t.
\]

The normalized aggregate energy error is

\[
\operatorname{WAPE}_{\mathrm{agg}}
=
\frac{
\sum_t
\left|
\sum_n e_{t,n}-\sum_n\hat e_{t,n}
\right|
}{
\sum_t\sum_n e_{t,n}
}.
\]

EV scope coverage is reported separately:

\[
\operatorname{Coverage}_{\mathrm{EV}}
=
\frac{
\sum_{t,i\in\mathcal N_{\mathrm{EV}}} e_{t,i}
}{
\sum_{t,j\in\mathcal N_{\mathrm C}}e_{t,j}
}.
\]

Raw per-entity MAE cannot be compared directly across EV and charger tasks
because their entity counts and energy scales differ.

## Quick sanity experiment

Tracked outputs:

- `experiments/results/advanced_forecast_quick/metrics.csv`
- `experiments/results/advanced_forecast_quick/metrics.json`

Configuration:

- panel start: 2022-01-01;
- train target end: 2023-03-31;
- validation: 2023-04-01 through 2023-06-30;
- quick test: 2023-07-01 through 2023-07-07;
- lookback: 28 days;
- EV entities: top 24 recurrent drivers selected from training only;
- neural training: at most 3 epochs, seed 20260726.

Selected charger-task sanity results:

| Model | Port energy MAE (kWh) | Aggregate daily energy MAE (kWh) |
|---|---:|---:|
| Seasonal naive | 12.012 | 38.001 |
| LSTM | 11.799 | 29.353 |
| TCN | 11.527 | 31.969 |
| iTransformer | 12.617 | 39.283 |
| GraphGNN | 12.535 | 35.624 |

This seven-day, three-epoch run only proves that the data/model/evaluation
pipeline executes. It is **not** a paper result.

The most important finding is methodological: the top-24 EV cohort covers only
1.885% of the six chargers' test energy, while the charger task covers 100%.
The small EV per-entity errors therefore describe a narrow recurring-driver
conditional task and cannot establish superiority over charger forecasting.

## Validation commands

All current tests:

```bash
MPLCONFIGDIR=/private/tmp/mpl-models \
  PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -v
```

Verified result on 2026-07-30: 19 tests passed. Version 4 additionally passed
Python syntax checks, artifact regeneration from tracked compact Q3 summaries,
LaTeX compilation, log inspection, and Poppler page rendering.

Quick benchmark:

```bash
MPLCONFIGDIR=/private/tmp/mpl-models \
  PYTHONPATH=src \
  .venv/bin/python experiments/advanced_forecast_benchmark.py \
  --quick \
  --output experiments/results/advanced_forecast_quick/metrics.csv
```

## Reference-paper boundary

The replacement PDF at `/Users/admin/Desktop/qt4463c0cx.pdf` was reviewed as
background for problem framing. The project is not a continuation or
reproduction of that work. In particular:

- no Hungarian or other session-assignment metric is used;
- no historical-day/session retrieval method is copied;
- no claim depends on reproducing its numerical results;
- the contribution is the additive charger/EV flexibility signature,
  physics-constrained forecasting head, fair matched-scope identity, and
  subsequent charger-native MPC coupling.

## Next experiments, in order

1. Lift the six-anchor physical cone to 96-slot cumulative envelopes using a
   validation-only monotone interpolation/calibration map.
2. Calibrate forecast uncertainty from validation residuals without test
   leakage and preserve the 100% hard-feasibility guarantee.
3. Feed EV and charger forecasts into their corresponding rolling MPCs under
   the same tariff, arrival information, monthly peak state, feasibility
   checks, and random seeds. No cross-entity session assignment is required.
4. Compare avoided charging cost, peak demand, unserved energy, perfect-
   information regret, runtime, and forecast error. Charger superiority must
   emerge from the frozen protocol; it must not be engineered into a metric.
5. Extend to multiple disjoint port cohorts, rolling-origin folds, and at least
   12 billing months before a general economic claim.
6. Add PatchTST or TimeMixer only as targeted forecasting ablations if they
   improve the physical target under the same split and compute budget.

## Non-negotiable boundaries

- V0G only; no V2G and no demand response.
- Chronological splits only.
- No test-period entity selection, scaling, graph construction, or calibration.
- Report EV cold-start/coverage explicitly.
- Do not compare raw per-entity EV and charger MAE as if their scales match.
- Do not use archived Copilot outputs as evidence.
- Do not use Hungarian/session-assignment metrics; compare additive realized
  flexibility sets and downstream MPC outcomes directly.
- Do not commit raw session CSVs, trained model weights, caches, or large debug
  outputs.
