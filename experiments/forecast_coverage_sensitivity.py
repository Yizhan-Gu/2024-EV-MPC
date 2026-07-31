#!/usr/bin/env python3
"""Quantify the open-set coverage limit of individual-EV forecasting."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXED_CHARGERS = {
    "UCSD / GILMAN 2-2|2",
    "UCSD / SCHOLARS - 07|2",
    "UCSD / RADY P357 5|2",
    "UCSD / BIRCH AQUARIUM|1",
    "UCSD / SCHOLARS - 01|1",
    "UCSD / SCHOLARS - 08|2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "clean_charging_sessions_enhanced.csv"
        ),
    )
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2023-03-31")
    parser.add_argument("--test-start", default="2023-07-01")
    parser.add_argument(
        "--test-end",
        default="2023-07-31",
        help=(
            "Development runs default to July only. Pass an explicit later "
            "date only for a frozen paper-scale replication."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 6, 8, 12, 24],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "fair_forecast_q3"
            / "coverage_sensitivity.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_count: Counter[str] = Counter()
    test_energy: Counter[str] = Counter()
    test_sessions: Counter[str] = Counter()
    total_test_energy = 0.0
    total_test_sessions = 0

    with args.data.open(newline="") as handle:
        for row in csv.DictReader(handle):
            charger = f"{row['station_name']}|{row['port']}"
            if charger not in FIXED_CHARGERS:
                continue
            day = row["session_start_time_la"][:10]
            driver = str(row["driver_id"])
            if args.train_start <= day <= args.train_end:
                train_count[driver] += 1
            if args.test_start <= day <= args.test_end:
                energy = float(row["total_energy_dispensed"])
                test_energy[driver] += energy
                test_sessions[driver] += 1
                total_test_energy += energy
                total_test_sessions += 1

    rows = []
    for threshold in sorted(set(args.thresholds)):
        if threshold <= 0:
            raise ValueError("thresholds must be positive")
        cohort = {
            driver
            for driver, count in train_count.items()
            if count >= threshold
        }
        covered_energy = sum(test_energy[driver] for driver in cohort)
        covered_sessions = sum(test_sessions[driver] for driver in cohort)
        rows.append(
            {
                "minimum_training_sessions": threshold,
                "eligible_training_drivers": len(cohort),
                "test_energy_coverage": (
                    covered_energy / total_test_energy
                    if total_test_energy > 0.0
                    else 0.0
                ),
                "test_session_coverage": (
                    covered_sessions / total_test_sessions
                    if total_test_sessions
                    else 0.0
                ),
                "covered_test_energy_kwh": covered_energy,
                "total_test_energy_kwh": total_test_energy,
                "covered_test_sessions": covered_sessions,
                "total_test_sessions": total_test_sessions,
                "unseen_or_ineligible_energy_kwh": (
                    total_test_energy - covered_energy
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print("coverage sensitivity:", args.output)
    for row in rows:
        print(
            f"minimum={row['minimum_training_sessions']:2d} "
            f"drivers={row['eligible_training_drivers']:4d} "
            f"energy coverage={row['test_energy_coverage']:.3f}"
        )


if __name__ == "__main__":
    main()
