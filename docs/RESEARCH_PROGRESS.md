# Research progress: EV versus charger forecasting and MPC

Last verified: 2026-07-26 (fair Q3 forecast benchmark and Version 2 paper)

This is the read-first handoff for future Codex sessions. Do not infer that a
long experiment, reference-paper reproduction, or MPC coupling is complete
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

### Version 2 manuscript and figures

Current draft:

- source: `paper/main_segan.tex`;
- compiled versioned PDF: `paper/main_segan_v2.pdf`;
- new tables: `paper/tables/forecast_fair_comparison.tex` and
  `paper/tables/forecast_full_scope.tex`;
- new paper figures:
  `forecast_matched_scope`,
  `forecast_operational_scope`, and
  `forecast_daily_trajectories` in both PDF and PNG.

The manuscript now includes related work, the two prediction tasks, matched
and no-substitution estimands, iTransformer/GNN equations, the three-seed Q3
results, bootstrap intervals, negative hurdle results, expanded limitations,
and an implementation-to-equation audit. The PDF was rendered page by page
with Poppler for visual QA. The final local build is 17 pages and has no
overfull boxes, unresolved references, or LaTeX warnings.

Critical claim boundary: the advanced forecasts remain forecast-only. They
have not been calibrated, reconstructed into feasible nonoverlapping sessions,
or sent through MPC. The 28.23% V0G saving and 1.47% point-forecast improvement
come from the separate historical-median/conformal MPC pilot, not from
iTransformer or GraphGNN.

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

Verified result on 2026-07-26: 16 tests passed.

Quick benchmark:

```bash
MPLCONFIGDIR=/private/tmp/mpl-models \
  PYTHONPATH=src \
  .venv/bin/python experiments/advanced_forecast_benchmark.py \
  --quick \
  --output experiments/results/advanced_forecast_quick/metrics.csv
```

## Reference-paper boundary

The requested reference PDF was expected at
`/Users/admin/Downloads/qt4463c0cx.pdf`, but that file was absent and no
matching local file was found on 2026-07-26. No claim has been made that its
methodology was reproduced. Reattach the PDF before locking the final
experiment matrix or writing the related-work comparison against that paper.

## Next experiments, in order

1. Replace the failed class-balanced hurdle with a validation-calibrated
   rare-event objective or hierarchical count model. Preserve the failed
   baseline as a negative result.
2. Add PatchTST and TimeMixer as targeted ablations; consider SAMformer if
   small-sample instability remains visible. Do not add a large pretrained
   model unless it beats strong simple baselines under the same split.
3. Replace or augment correlation edges with physical/spatial charger edges
   if trustworthy metadata are available; compare fixed, learned, and no-graph
   adjacency.
4. Extend beyond the current three-seed Q3 benchmark to rolling-origin folds,
   multiple port cohorts, and at least 12 billing months.
5. Calibrate count, energy, and timing uncertainty using validation-only
   residuals.
6. Reconstruct feasible sessions/envelopes from each task's predictions and
   run both forecasts through their corresponding rolling MPC formulations.
7. Compare revenue/cost, peak demand, unserved energy, runtime, and forecast
   error. Charger-based superiority must emerge from the frozen protocol; it
   must not be assumed or engineered into the metrics.

## Non-negotiable boundaries

- V0G only; no V2G and no demand response.
- Chronological splits only.
- No test-period entity selection, scaling, graph construction, or calibration.
- Report EV cold-start/coverage explicitly.
- Do not compare raw per-entity EV and charger MAE as if their scales match.
- Do not use archived Copilot outputs as evidence.
- Do not commit raw session CSVs, trained model weights, caches, or large debug
  outputs.
