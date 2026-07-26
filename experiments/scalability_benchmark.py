#!/usr/bin/env python3
"""Benchmark exact EV- and charger-level day-ahead formulations."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_mpc import (  # noqa: E402
    disaggregate_nonoverlap,
    read_sessions_by_days,
    solve_charger_envelope,
    solve_ev_dispatch,
    summer_tariff,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-days",
        nargs="+",
        default=[
            "2023-07-03",
            "2023-07-17",
            "2023-08-01",
            "2023-08-15",
            "2023-09-01",
            "2023-09-15",
            "2023-09-29",
        ],
    )
    parser.add_argument(
        "--selection-days",
        nargs="+",
        default=["2023-05-01", "2023-06-30"],
        help="Inclusive start/end dates used to rank charger activity.",
    )
    parser.add_argument(
        "--charger-counts",
        nargs="+",
        type=int,
        default=[3, 6, 9, 12],
    )
    parser.add_argument("--time-limit", type=float, default=10.0)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "paper_2023Q3"
            / "scalability.csv"
        ),
    )
    return parser.parse_args()


def _date_range(start: str, end: str) -> list[str]:
    from datetime import date, timedelta

    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def _nonoverlap(sessions: list, charger_id: str) -> bool:
    selected = sorted(
        (x for x in sessions if x.charger_id == charger_id),
        key=lambda x: (x.arrival, x.departure, x.session_id),
    )
    return all(
        current.arrival > previous.departure
        for previous, current in zip(selected, selected[1:])
    )


def main() -> None:
    args = parse_args()
    if len(args.selection_days) != 2:
        raise ValueError("selection-days requires an inclusive start and end")
    if any(count <= 0 for count in args.charger_counts):
        raise ValueError("charger counts must be positive")

    selection_days = _date_range(*args.selection_days)
    loaded_days = selection_days + args.benchmark_days
    sessions_by_day, _ = read_sessions_by_days(args.data, loaded_days)
    counts = Counter(
        session.charger_id
        for day in selection_days
        for session in sessions_by_day[day]
    )
    energy = Counter()
    for day in selection_days:
        for session in sessions_by_day[day]:
            energy[session.charger_id] += session.energy_kwh
    candidates = [
        charger_id
        for charger_id in counts
        if all(
            _nonoverlap(sessions_by_day[day], charger_id)
            for day in args.benchmark_days
        )
    ]
    candidates.sort(key=lambda x: (-counts[x], -energy[x], x))
    maximum = max(args.charger_counts)
    if len(candidates) < maximum:
        raise RuntimeError(
            f"only {len(candidates)} nonoverlapping candidates for "
            f"requested maximum {maximum}"
        )

    tariff = summer_tariff()
    rows: list[dict] = []
    for charger_count in args.charger_counts:
        charger_ids = candidates[:charger_count]
        charger_set = set(charger_ids)
        for target_day in args.benchmark_days:
            sessions = [
                session
                for session in sessions_by_day[target_day]
                if session.charger_id in charger_set
            ]
            started = time.perf_counter()
            ev_result = solve_ev_dispatch(
                sessions,
                tariff,
                time_limit=args.time_limit,
            )
            ev_runtime = time.perf_counter() - started

            started = time.perf_counter()
            charger_result = solve_charger_envelope(
                sessions,
                tariff,
                time_limit=args.time_limit,
            )
            charger_runtime = time.perf_counter() - started
            disaggregate_nonoverlap(
                sessions,
                charger_result.power_by_unit_kw,
                n_slots=tariff.n_slots,
            )
            rows.append(
                {
                    "target_day": target_day,
                    "charger_count": charger_count,
                    "session_count": len(sessions),
                    "ev_variable_count": (
                        len(sessions) * tariff.n_slots + 2
                    ),
                    "charger_variable_count": (
                        2 * charger_count * tariff.n_slots + 2
                    ),
                    "ev_objective": ev_result.objective,
                    "charger_objective": charger_result.objective,
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
                        if charger_runtime > 0.0
                        else float("nan")
                    ),
                    "energy_kwh": ev_result.energy_kwh,
                    "peak_kw": ev_result.peak_kw,
                    "charger_ids": ";".join(charger_ids),
                }
            )
            print(
                f"{target_day} chargers={charger_count:2d} "
                f"sessions={len(sessions):2d} "
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
