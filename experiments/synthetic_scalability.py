#!/usr/bin/env python3
"""Synthetic density benchmark for EV versus charger LP formulations."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_mpc import (  # noqa: E402
    Session,
    disaggregate_nonoverlap,
    solve_charger_envelope,
    solve_ev_dispatch,
    summer_tariff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["6x2", "6x4", "6x8", "12x2", "12x4", "12x8", "24x8"],
        help="Cases formatted as chargers x sessions-per-charger.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "paper_2023Q3"
            / "scalability_synthetic.csv"
        ),
    )
    return parser.parse_args()


def _sessions(chargers: int, sessions_per_charger: int) -> list[Session]:
    if sessions_per_charger > 96:
        raise ValueError("sessions-per-charger cannot exceed 96")
    segment = 96 // sessions_per_charger
    output: list[Session] = []
    for charger_idx in range(chargers):
        charger_id = f"synthetic:{charger_idx:03d}"
        for session_idx in range(sessions_per_charger):
            arrival = session_idx * segment + 1
            departure = (
                96
                if session_idx == sessions_per_charger - 1
                else (session_idx + 1) * segment
            )
            available_slots = departure - arrival + 1
            energy = 0.5 * 6.6 * 0.25 * available_slots
            output.append(
                Session(
                    session_id=f"{charger_id}:{session_idx:03d}",
                    charger_id=charger_id,
                    arrival=arrival,
                    departure=departure,
                    energy_kwh=energy,
                )
            )
    return output


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    tariff = summer_tariff()
    rows: list[dict] = []

    for case in args.cases:
        chargers, sessions_per_charger = (
            int(value) for value in case.lower().split("x", maxsplit=1)
        )
        sessions = _sessions(chargers, sessions_per_charger)
        for repeat in range(1, args.repeats + 1):
            started = time.perf_counter()
            ev_result = solve_ev_dispatch(
                sessions,
                tariff,
                time_limit=30.0,
            )
            ev_runtime = time.perf_counter() - started
            started = time.perf_counter()
            charger_result = solve_charger_envelope(
                sessions,
                tariff,
                time_limit=30.0,
            )
            charger_runtime = time.perf_counter() - started
            disaggregate_nonoverlap(
                sessions,
                charger_result.power_by_unit_kw,
                n_slots=tariff.n_slots,
            )
            rows.append(
                {
                    "case": case,
                    "repeat": repeat,
                    "charger_count": chargers,
                    "sessions_per_charger": sessions_per_charger,
                    "session_count": len(sessions),
                    "ev_variable_count": len(sessions) * 96 + 2,
                    "charger_variable_count": chargers * 96 * 2 + 2,
                    "objective_gap": (
                        charger_result.objective - ev_result.objective
                    ),
                    "max_load_gap_kw": float(
                        np.max(
                            np.abs(
                                charger_result.load_kw
                                - ev_result.load_kw
                            )
                        )
                    ),
                    "ev_runtime_seconds": ev_runtime,
                    "charger_runtime_seconds": charger_runtime,
                    "runtime_ratio_ev_over_charger": (
                        ev_runtime / charger_runtime
                    ),
                }
            )
            print(
                f"{case} repeat={repeat} sessions={len(sessions)} "
                f"gap={rows[-1]['objective_gap']:.3e} "
                f"load-gap={rows[-1]['max_load_gap_kw']:.3e} "
                f"runtime={ev_runtime:.3f}/{charger_runtime:.3f}s",
                flush=True,
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
    print("output:", args.output)


if __name__ == "__main__":
    main()
