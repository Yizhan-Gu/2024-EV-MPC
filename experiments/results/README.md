# Three-day deterministic smoke results

These compact outputs validate the current charger-MPC implementation on three
independent target days. Each run selects six chargers that are active on both
the target day and its same-weekday persistence day one week earlier.

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

These are implementation checks, not publication results. Each target day is
optimized independently and applies the monthly demand-charge rate to that
isolated day. A paper experiment must use a continuous multi-day simulation
with the executed month-to-date peak carried forward.

Files:

- `smoke_2023-07-01.csv`
- `smoke_2023-07-02.csv`
- `smoke_2023-07-03.csv`
