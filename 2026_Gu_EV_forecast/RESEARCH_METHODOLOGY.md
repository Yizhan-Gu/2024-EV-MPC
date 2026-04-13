# EV Forecast-to-MPC Methodology (Refined 0701 Version)

## 1. Study Scope

This work compares forecasting methods under one fixed rolling MPC controller.
Only the forecast module changes; the control objective/constraints stay unchanged.

Methods kept for comparison:
- Perfect
- Noforecast
- Persistence
- Statistic
- GMM
- LSTM
- TCN
- Transformer
- iTransformer

Added methods:
- iT-CP90 (iTransformer-TCN blended point forecast + conformal 90% interval)
- iT-CP95 (same backbone + conformal 95% interval)
- TFT-Quantile (p10/p50/p90)
- DeepAR-Gaussian (distribution sampling)
- ITCN-ChargerAware (charger-focused hybrid assignment)

---

## 2. Single Data Source and Reproducibility

Official EV input file:
- `clean_charging_sessions.csv`

Project output root:
- `2026_Gu_EV_forecast/`

Main notebook:
- `Research_Results/EV_Charging_Optimization_Research.ipynb`

All comparison figures are stored in one folder only:
- `2026_Gu_EV_forecast/figures/0701/`

Tables:
- `2026_Gu_EV_forecast/tables/forecast_quality_0701_detailed.csv`
- `2026_Gu_EV_forecast/tables/forecast_quality_0701_all_methods.csv`
- `2026_Gu_EV_forecast/tables/mpc_result_0701_extended_methods.csv`
- `2026_Gu_EV_forecast/tables/sanity_check_0701.csv`

---

## 3. Unified Notation (Used by All Methods)

For each charging session $i$:
- Arrival index: $AT_i \in \{1,\dots,96\}$
- Departure index: $DT_i \in \{1,\dots,96\}$
- Energy demand: $ED_i$ (kWh)
- Charger id: $c_i$

Time step length:
- $\Delta t = 0.25$ hour

Day-level forecast targets (96 bins):
- Arrival-count profile: $A_t$
- Arrival-energy profile: $E_t$
- Departure-count profile: $D_t$

Session-level feasibility condition:
$$
ED_i \le P_{\max}\,\Delta t\,(DT_i-AT_i+1)
$$

---

## 4. Preprocessing from clean_charging_sessions

Pipeline:
1. Parse timestamps and energy numeric type.
2. Drop missing core fields.
3. Remove likely DCFC rows by station-name keywords.
4. Keep physical bounds for energy and duration.
5. Keep same-day sessions only.
6. Deduplicate by session signature.
7. Apply IQR outlier filtering.
8. Apply effective-power filter.

Outputs:
- `2026_Gu_EV_forecast/clean_charging_sessions_enhanced.csv`
- `2026_Gu_EV_forecast/tables/preprocessing_stats_enhanced.csv`

---

## 5. What Is Forecasted and What Is Compared

### 5.1 Timing quality (explicit)
Timing is evaluated by distribution error, not by ambiguous "session error":
- Arrival-time distribution error:
$$
\mathrm{W1}_{AT} = W_1\big(\hat{A}_t, A_t\big)
$$
- Departure-time distribution error:
$$
\mathrm{W1}_{DT} = W_1\big(\hat{D}_t, D_t\big)
$$

Interpretation:
- Smaller is better.
- Unit is 15-min-bin transport distance.

### 5.2 Energy quality (explicit)
- Interval-level arrival-energy MAE/RMSE:
$$
\mathrm{MAE}_E=\frac{1}{96}\sum_{t=1}^{96}|\hat{E}_t-E_t|,
\qquad
\mathrm{RMSE}_E=\sqrt{\frac{1}{96}\sum_{t=1}^{96}(\hat{E}_t-E_t)^2}
$$
- Daily total-energy error (%):
$$
\mathrm{Err}_{\Sigma E}(\%)=\frac{|\sum_t\hat{E}_t-\sum_tE_t|}{\sum_tE_t}\times 100
$$
- Daily session-count error (%):
$$
\mathrm{Err}_{N}(\%)=\frac{|\hat{N}-N|}{N}\times 100
$$

### 5.3 Probabilistic quality
For probabilistic methods:
- PICP@90
- PINAW@90
- Pinball losses at p10/p50/p90

---

## 6. Method-by-Method Procedure (Same Variable System)

### 6.1 Perfect
Input true $(AT_i,DT_i,ED_i,c_i)$ of target day directly.
Used as upper-bound benchmark.

### 6.2 Noforecast
Use empty future session set.
Controller only reacts to actually arrived sessions.

### 6.3 Persistence
Reuse nearest same day-type historical sessions and map to target date.

### 6.4 Statistic
Sample from empirical distributions of:
- $AT_i$
- duration $(DT_i-AT_i+1)$
- $ED_i$
Then clip for feasibility.

### 6.5 GMM
Fit GMM on session feature vectors (timing + duration + energy), sample sessions, then clip.

### 6.6 LSTM / TCN / Transformer / iTransformer
1. Build historical sequence of day vectors.
2. Predict next-day profile vector.
3. Convert predicted $(\hat{A}_t, \hat{E}_t)$ to forecast sessions.

### 6.7 iT-CP90 / iT-CP95
1. Build iTransformer and TCN profile predictions.
2. Blend by calibration-selected weight $\alpha$.
3. Apply conformal residual quantiles for intervals.
4. Use calibrated dispatch profile to avoid conservative over-inflation.

### 6.8 TFT-Quantile
- observed inputs: historical load/profile sequence
- known future inputs: calendar features (weekday/weekend/month harmonics)
- output quantiles: p10/p50/p90

### 6.9 DeepAR-Gaussian
- outputs $(\mu_t,\sigma_t)$
- sample multi-scenario trajectories
- derive p10/p50/p90 intervals

### 6.10 ITCN-ChargerAware (charger-focused)
1. Build blended ITCN profile forecast.
2. Synthesize sessions from $(\hat{A}_t,\hat{E}_t)$.
3. Re-assign charger ids with time-conditioned charger probabilities from history.

---

## 7. MPC Evaluation and Fairness Correction

Base MPC metrics:
- cost
- saving vs V0G
- peak
- dispatched energy

### 7.1 Why effective saving is reported
In charger-aggregation experiments, methods may under-deliver total required energy,
which can artificially reduce cost.
To avoid over-crediting such methods, we report an undersupply-penalized cost:
$$
\text{effective\_cost} = \text{mpc\_cost} + \lambda_{\text{unserved}}\cdot \max(0, E_{\text{required}}-E_{\text{dispatched}})
$$
with $\lambda_{\text{unserved}} = 5\,$$/kWh in this study.

Then:
$$
\text{effective\_saving}(\%)=\frac{\text{V0G\_cost}-\text{effective\_cost}}{\text{V0G\_cost}}\times 100
$$

This directly addresses the issue where CP90/CP95 looked cheaper only because they under-supplied energy.

---

## 8. Sanity Check Protocol

For each method-case run:
1. Energy balance gap:
$$
\mathrm{gap}_{E}(\%) = \frac{E_{\text{dispatch}}-E_{\text{required}}}{E_{\text{required}}}\times 100
$$
2. Status ratio threshold.
3. Flag pass/fail per case.

Saved table:
- `2026_Gu_EV_forecast/tables/sanity_check_0701.csv`

Interpretation:
- Methods with large negative energy gap are not considered practically valid, even if raw cost appears low.

---

## 9. Figure Set (All in One Folder)

Folder:
- `2026_Gu_EV_forecast/figures/0701/`

Key refined figures:
- `0701_refined_saving_reference_lines.png`
- `0701_refined_forecast_skill_map_time_vs_energy.png`
- `0701_refined_forecast_metric_heatmap.png`
- `0701_refined_prob_timeseries_with_residuals.png`
- `0701_refined_mpc_sanity_energy_gap.png`
- `0701_refined_load_curves_core_methods.png`

All figure titles and axis labels explicitly state whether the metric is about:
- timing (arrival/departure distribution), or
- energy (interval energy / total energy), or
- post-MPC dispatch sanity.

---

## 10. Practical Notes for Paper Writing

1. Report both forecast quality and control quality.
2. Discuss EV case and Charger case separately.
3. Use effective saving (penalized) for fair practical ranking.
4. Keep Perfect as upper bound and NoForecast as practical lower baseline.
5. Treat methods failing sanity check as non-deployable even if their unpenalized cost is low.
