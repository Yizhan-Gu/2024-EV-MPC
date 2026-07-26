# Deterministic smoke results

## Fair Q3 EV-versus-charger forecasting

`fair_forecast_q3/` contains the compact forecast-only paper results:

- training through 2023-03-31, April--June validation, and untouched Q3 test;
- a training-selected 372-driver cohort requiring at least three historical
  sessions;
- identical realized sessions for `EV` and `ChargerMatched`;
- all six-port demand for `ChargerFull`;
- three seeds for DLinear, LSTM, TCN, iTransformer, and GraphGNN;
- coverage sensitivity, daily predictions, paired seven-day block-bootstrap
  summaries, and full-scope summaries.

The matched cohort covers 29.77% of full Q3 energy. Operational EV errors use a
no-substitution metric, so excess energy assigned to a known driver cannot
cancel energy belonging to an unidentified driver.

These results have not yet been passed through MPC. Economic results below come
from the separate historical-median/conformal controller experiment.

## Continuous three-day billing experiment

The continuous experiment uses the same six chargers on all three target days.
Each method carries its own executed noncoincident and on-peak month-to-date
peaks into the next day. Reported daily costs therefore contain energy charges
plus only the incremental demand charge above the prior peak.

| Method | Continuous cost | Saving vs. V0G | Final peak / on-peak peak |
| --- | ---: | ---: | ---: |
| V0G | 1148.033 | 0.00% | 26.992 / 13.200 kW |
| Perfect | 675.261 | 41.18% | 15.857 / 7.175 kW |
| NoForecast | 864.021 | 24.74% | 20.037 / 9.803 kW |
| Persistence | 860.601 | 25.04% | 18.820 / 10.722 kW |

If the same executed loads were incorrectly billed as three independent
monthly peaks, costs would be overstated by 1043.998 to 1768.531. This confirms
that isolated-day demand-charge totals are not economically interpretable.

All energy requirements were served within numerical tolerance, all 678
rolling LP solves succeeded, and no fallback was used. Persistence beats
NoForecast by only 0.30 percentage points on this small sample, which is not
evidence of a statistically meaningful forecasting benefit.

Continuous outputs:

- `continuous_2023-07-01_to_2023-07-03_daily.csv`
- `continuous_2023-07-01_to_2023-07-03_summary.csv`

## Independent-day regression diagnostic

These older compact outputs validate the current charger-MPC implementation on
three independent target days. Each run selects six chargers that are active on
both the target day and its same-weekday persistence day one week earlier.

| Method | Mean cost | Mean saving vs. V0G | Mean peak (kW) |
| --- | ---: | ---: | ---: |
| V0G | 1205.292 | 0.00% | 30.233 |
| Perfect | 719.152 | 40.69% | 17.533 |
| NoForecast | 839.255 | 30.68% | 21.800 |
| Persistence | 860.137 | 28.61% | 21.709 |

Across all 12 method-day cases:

- every requested session energy was served within numerical tolerance;
- every rolling LP solve succeeded and no fallback was used;
- each charger-envelope oracle matched the EV-session oracle to within
  `4.547e-13`.

These files remain implementation checks rather than economic results because
each target day applies the monthly demand-charge rate in isolation.

Files:

- `smoke_2023-07-01.csv`
- `smoke_2023-07-02.csv`
- `smoke_2023-07-03.csv`
