#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PAPER_PYTHON="${PAPER_PYTHON:-/opt/homebrew/bin/python3}"
PLOT_PYTHON="${PLOT_PYTHON:-python}"
DATA_FILE="${PROJECT_ROOT}/data/processed/clean_charging_sessions_enhanced.csv"
FIXED_CHARGERS="UCSD / GILMAN 2-2|2;UCSD / SCHOLARS - 07|2;UCSD / RADY P357 5|2;UCSD / BIRCH AQUARIUM|1;UCSD / SCHOLARS - 01|1;UCSD / SCHOLARS - 08|2"

if [[ ! -f "${DATA_FILE}" ]]; then
  echo "Missing local input: ${DATA_FILE}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

for alpha in 0.1 0.2 0.3 0.4; do
  "${PAPER_PYTHON}" experiments/paper_month_mpc.py \
    --calibration-start 2023-04-01 \
    --calibration-end 2023-05-31 \
    --test-start 2023-06-01 \
    --test-end 2023-06-30 \
    --lookback-weeks 4 \
    --alpha "${alpha}" \
    --fixed-chargers "${FIXED_CHARGERS}" \
    --methods ConformalRobust \
    --time-limit 1.0 \
    --progress-every 30 \
    --data "${DATA_FILE}" \
    --output-dir "experiments/results/validation_alpha_${alpha}"
done

"${PAPER_PYTHON}" experiments/paper_month_mpc.py \
  --calibration-start 2023-05-01 \
  --calibration-end 2023-06-30 \
  --test-start 2023-07-01 \
  --test-end 2023-09-30 \
  --lookback-weeks 4 \
  --alpha 0.1 \
  --fixed-chargers "${FIXED_CHARGERS}" \
  --methods V0G Perfect NoForecast Persistence HistoricalMedian ConformalRobust \
  --time-limit 1.0 \
  --progress-every 10 \
  --data "${DATA_FILE}" \
  --output-dir experiments/results/paper_2023Q3

"${PAPER_PYTHON}" experiments/scalability_benchmark.py \
  --data "${DATA_FILE}" \
  --output experiments/results/paper_2023Q3/scalability.csv
"${PAPER_PYTHON}" experiments/synthetic_scalability.py \
  --output experiments/results/paper_2023Q3/scalability_synthetic.csv

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/charger-mpc-matplotlib}" \
  "${PLOT_PYTHON}" experiments/make_paper_artifacts.py

(
  cd paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_segan.tex
)
