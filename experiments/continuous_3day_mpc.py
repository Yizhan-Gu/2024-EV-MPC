#!/usr/bin/env python3
"""Run a small continuous multi-day charger-MPC billing experiment."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_mpc import (  # noqa: E402
    evaluate_cost,
    rolling_charger_mpc,
    summer_tariff,
    v0g_dispatch,
)
from charger_mpc.data import read_day_sessions, with_forecast_ids  # noqa: E402


METHODS = ("V0G", "Perfect", "NoForecast", "Persistence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-days",
        nargs="+",
        default=["2023-07-01", "2023-07-02", "2023-07-03"],
    )
    parser.add_argument(
        "--history-days",
        nargs="+",
        default=["2023-06-24", "2023-06-25", "2023-06-26"],
    )
    parser.add_argument("--chargers", type=int, default=6)
    parser.add_argument("--time-limit", type=float, default=1.0)
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
        "--daily-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "continuous_2023-07-01_to_2023-07-03_daily.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "continuous_2023-07-01_to_2023-07-03_summary.csv"
        ),
    )
    return parser.parse_args()


def _has_nonoverlap(sessions: list, charger_id: str) -> bool:
    selected = sorted(
        (x for x in sessions if x.charger_id == charger_id),
        key=lambda x: (x.arrival, x.departure, x.session_id),
    )
    return all(
        current.arrival > previous.departure
        for previous, current in zip(selected, selected[1:])
    )


def _fixed_busy_chargers(
    target_by_day: dict[str, list],
    history_by_day: dict[str, list],
    *,
    limit: int,
) -> list[str]:
    all_day_lists = list(target_by_day.values()) + list(history_by_day.values())
    common_ids = set.intersection(
        *({session.charger_id for session in sessions}
          for sessions in all_day_lists)
    )
    candidates = [
        charger_id
        for charger_id in common_ids
        if all(
            _has_nonoverlap(sessions, charger_id)
            for sessions in all_day_lists
        )
    ]
    target_count = Counter(
        session.charger_id
        for sessions in target_by_day.values()
        for session in sessions
    )
    target_energy = Counter()
    for sessions in target_by_day.values():
        for session in sessions:
            target_energy[session.charger_id] += session.energy_kwh
    candidates.sort(
        key=lambda x: (-target_count[x], -target_energy[x], x)
    )
    return candidates[:limit]


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if len(args.target_days) != len(args.history_days):
        raise ValueError("target-days and history-days must have equal length")
    if not args.target_days:
        raise ValueError("at least one target/history day pair is required")
    if args.chargers <= 0:
        raise ValueError("chargers must be positive")

    target_by_day = {
        day: read_day_sessions(args.data, day)[0]
        for day in args.target_days
    }
    history_by_day = {
        day: read_day_sessions(args.data, day)[0]
        for day in args.history_days
    }
    charger_ids = _fixed_busy_chargers(
        target_by_day,
        history_by_day,
        limit=args.chargers,
    )
    if len(charger_ids) < args.chargers:
        raise RuntimeError(
            f"requested {args.chargers} fixed chargers but found "
            f"{len(charger_ids)} feasible common chargers"
        )
    charger_set = set(charger_ids)
    tariff = summer_tariff()

    states = {
        method: {"peak": 0.0, "onpeak_peak": 0.0}
        for method in METHODS
    }
    cumulative_cost = {method: 0.0 for method in METHODS}
    daily_rows: list[dict] = []

    for target_day, history_day in zip(
        args.target_days,
        args.history_days,
    ):
        actual = [
            x
            for x in target_by_day[target_day]
            if x.charger_id in charger_set
        ]
        history = [
            x
            for x in history_by_day[history_day]
            if x.charger_id in charger_set
        ]
        forecasts = {
            "Perfect": with_forecast_ids(
                actual,
                f"perfect:{target_day}",
            ),
            "NoForecast": [],
            "Persistence": with_forecast_ids(
                history,
                f"persistence:{history_day}",
            ),
        }

        for method in METHODS:
            start_peak = states[method]["peak"]
            start_onpeak_peak = states[method]["onpeak_peak"]
            if method == "V0G":
                result = v0g_dispatch(
                    actual,
                    tariff,
                    prior_peak_kw=start_peak,
                    prior_onpeak_peak_kw=start_onpeak_peak,
                )
            else:
                result = rolling_charger_mpc(
                    actual,
                    forecasts[method],
                    tariff,
                    method=method,
                    time_limit_per_solve=args.time_limit,
                    prior_peak_kw=start_peak,
                    prior_onpeak_peak_kw=start_onpeak_peak,
                )
            if result.unserved_energy_kwh > 1e-5:
                raise RuntimeError(
                    f"{target_day} {method} left "
                    f"{result.unserved_energy_kwh:.6f} kWh unserved"
                )
            if result.fallback_count:
                raise RuntimeError(
                    f"{target_day} {method} used "
                    f"{result.fallback_count} fallbacks"
                )

            standalone_cost, _, _, _ = evaluate_cost(
                result.load_kw,
                tariff,
            )
            cumulative_cost[method] += result.cost
            states[method] = {
                "peak": result.peak_kw,
                "onpeak_peak": result.onpeak_peak_kw,
            }
            daily_rows.append(
                {
                    "target_day": target_day,
                    "history_day": history_day,
                    "method": method,
                    "incremental_cost": result.cost,
                    "cumulative_cost": cumulative_cost[method],
                    "standalone_cost_same_load": standalone_cost,
                    "daily_proxy_overstatement": (
                        standalone_cost - result.cost
                    ),
                    "energy_kwh": result.energy_kwh,
                    "required_energy_kwh": result.required_energy_kwh,
                    "unserved_energy_kwh": result.unserved_energy_kwh,
                    "start_peak_kw": start_peak,
                    "end_peak_kw": result.peak_kw,
                    "start_onpeak_peak_kw": start_onpeak_peak,
                    "end_onpeak_peak_kw": result.onpeak_peak_kw,
                    "optimal_solve_ratio": result.optimal_solve_ratio,
                    "solve_count": result.solve_count,
                    "fallback_count": result.fallback_count,
                    "dropped_forecast_sessions": (
                        result.dropped_forecast_sessions
                    ),
                    "actual_session_count": len(actual),
                    "forecast_session_count": len(forecasts.get(method, [])),
                    "charger_count": len(charger_ids),
                    "charger_ids": ";".join(charger_ids),
                }
            )

    v0g_cost = cumulative_cost["V0G"]
    summary_rows: list[dict] = []
    for method in METHODS:
        rows = [row for row in daily_rows if row["method"] == method]
        continuous_cost = cumulative_cost[method]
        standalone_sum = sum(
            float(row["standalone_cost_same_load"])
            for row in rows
        )
        total_solves = sum(int(row["solve_count"]) for row in rows)
        optimal_solves = sum(
            float(row["optimal_solve_ratio"]) * int(row["solve_count"])
            for row in rows
        )
        summary_rows.append(
            {
                "method": method,
                "continuous_cost": continuous_cost,
                "saving_vs_v0g": v0g_cost - continuous_cost,
                "saving_pct_vs_v0g": (
                    100.0 * (v0g_cost - continuous_cost) / v0g_cost
                    if v0g_cost > 0.0
                    else 0.0
                ),
                "sum_standalone_cost_same_load": standalone_sum,
                "daily_proxy_overstatement": (
                    standalone_sum - continuous_cost
                ),
                "energy_kwh": sum(float(row["energy_kwh"]) for row in rows),
                "required_energy_kwh": sum(
                    float(row["required_energy_kwh"])
                    for row in rows
                ),
                "unserved_energy_kwh": sum(
                    float(row["unserved_energy_kwh"])
                    for row in rows
                ),
                "final_peak_kw": states[method]["peak"],
                "final_onpeak_peak_kw": states[method]["onpeak_peak"],
                "optimal_solve_ratio": (
                    optimal_solves / total_solves
                    if total_solves
                    else 1.0
                ),
                "solve_count": total_solves,
                "fallback_count": sum(
                    int(row["fallback_count"]) for row in rows
                ),
                "target_day_count": len(args.target_days),
                "charger_count": len(charger_ids),
                "charger_ids": ";".join(charger_ids),
            }
        )

    _write_rows(args.daily_output, daily_rows)
    _write_rows(args.summary_output, summary_rows)

    print("Continuous charger-MPC billing smoke test")
    print("target days:", ", ".join(args.target_days))
    print("history days:", ", ".join(args.history_days))
    print("fixed chargers:", len(charger_ids))
    print("charger ids:", "; ".join(charger_ids))
    print()
    for row in summary_rows:
        print(
            f"{row['method']:11s} "
            f"cost={row['continuous_cost']:.3f} "
            f"saving={row['saving_pct_vs_v0g']:.2f}% "
            f"standalone-sum={row['sum_standalone_cost_same_load']:.3f} "
            f"overstatement={row['daily_proxy_overstatement']:.3f} "
            f"peak={row['final_peak_kw']:.3f}/"
            f"{row['final_onpeak_peak_kw']:.3f} kW "
            f"unserved={row['unserved_energy_kwh']:.6f} "
            f"optimal={row['optimal_solve_ratio']:.3f}"
        )
    print()
    print("daily output:", args.daily_output)
    print("summary output:", args.summary_output)


if __name__ == "__main__":
    main()
