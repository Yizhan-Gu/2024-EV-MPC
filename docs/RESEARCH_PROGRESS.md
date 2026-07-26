# Research progress: EV versus charger forecasting and MPC

Last verified: 2026-07-26

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
- The current manuscript is `paper/main_segan.tex`; compiled output is
  `paper/main_segan.pdf`.

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

Verified result on 2026-07-26: 15 tests passed.

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

1. Add a two-stage EV participation model: predict driver-day activity first,
   then conditional energy/timing. Include a pooled cold-start or residual
   demand component so full charger energy has an honest EV-level comparator.
2. Add PatchTST and TimeMixer as targeted ablations; consider SAMformer if
   small-sample instability remains visible. Do not add a large pretrained
   model unless it beats strong simple baselines under the same split.
3. Replace or augment correlation edges with physical/spatial charger edges
   if trustworthy metadata are available; compare fixed, learned, and no-graph
   adjacency.
4. Use rolling-origin folds and at least three random seeds. Report bootstrap
   confidence intervals and paired loss tests rather than a single split.
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
