#!/usr/bin/env python3
"""Run a causal full-month charger-MPC evaluation for paper tables."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_mpc import (  # noqa: E402
    calibrate_one_sided_conformal,
    conformal_robust_forecast,
    evaluate_cost,
    historical_median_forecast,
    read_sessions_by_days,
    rolling_charger_mpc,
    session_forecast_metrics,
    summer_tariff,
    v0g_dispatch,
)
from charger_mpc.data import with_forecast_ids  # noqa: E402


METHODS = (
    "V0G",
    "Perfect",
    "NoForecast",
    "Persistence",
    "HistoricalMedian",
    "ConformalRobust",
)
FORECAST_METHODS = (
    "Persistence",
    "HistoricalMedian",
    "ConformalRobust",
)


def _date_range(start: str, end: str) -> list[str]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("end date precedes start date")
    return [
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-start", default="2023-05-01")
    parser.add_argument("--calibration-end", default="2023-06-30")
    parser.add_argument("--test-start", default="2023-07-01")
    parser.add_argument("--test-end", default="2023-07-31")
    parser.add_argument("--lookback-weeks", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--chargers", type=int, default=6)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument(
        "--fixed-chargers",
        default="",
        help="Semicolon-separated charger IDs; bypasses cohort selection.",
    )
    parser.add_argument("--minimum-selection-sessions", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=1)
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
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "paper_2023-07"
        ),
    )
    return parser.parse_args()


def _is_nonoverlap(sessions: list, charger_id: str) -> bool:
    selected = sorted(
        (x for x in sessions if x.charger_id == charger_id),
        key=lambda x: (x.arrival, x.departure, x.session_id),
    )
    return all(
        current.arrival > previous.departure
        for previous, current in zip(selected, selected[1:])
    )


def _select_fixed_chargers(
    sessions_by_day: dict[str, list],
    selection_days: list[str],
    validity_days: list[str],
    *,
    limit: int,
    minimum_sessions: int,
) -> list[str]:
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
        for charger_id, count in counts.items()
        if count >= minimum_sessions
        and all(
            _is_nonoverlap(sessions_by_day[day], charger_id)
            for day in validity_days
        )
    ]
    candidates.sort(key=lambda x: (-counts[x], -energy[x], x))
    if len(candidates) < limit:
        raise RuntimeError(
            f"requested {limit} chargers but only {len(candidates)} "
            "meet the pre-test activity and nonoverlap policy"
        )
    return candidates[:limit]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = list(
            dict.fromkeys(
                key
                for row in rows
                for key in row
            )
        )
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _weighted_mean(
    rows: list[dict],
    value_key: str,
    weight_key: str,
) -> float:
    valid = [
        row
        for row in rows
        if not math.isnan(float(row.get(value_key, math.nan)))
        and float(row[weight_key]) > 0.0
    ]
    denominator = sum(float(row[weight_key]) for row in valid)
    if denominator == 0.0:
        return math.nan
    return sum(
        float(row.get(value_key, math.nan)) * float(row[weight_key])
        for row in valid
    ) / denominator


def main() -> None:
    args = parse_args()
    if args.lookback_weeks <= 0:
        raise ValueError("lookback-weeks must be positive")
    if args.chargers <= 0:
        raise ValueError("chargers must be positive")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    selected_methods = tuple(dict.fromkeys(args.methods))

    calibration_days = _date_range(
        args.calibration_start,
        args.calibration_end,
    )
    test_days = _date_range(args.test_start, args.test_end)
    earliest = (
        date.fromisoformat(args.calibration_start)
        - timedelta(days=7 * args.lookback_weeks)
    ).isoformat()
    loaded_days = _date_range(earliest, args.test_end)
    sessions_by_day, data_stats = read_sessions_by_days(
        args.data,
        loaded_days,
    )
    if args.fixed_chargers:
        charger_ids = [
            item.strip()
            for item in args.fixed_chargers.split(";")
            if item.strip()
        ]
        if not charger_ids:
            raise ValueError("fixed-chargers did not contain any IDs")
        missing = [
            charger_id
            for charger_id in charger_ids
            if not any(
                session.charger_id == charger_id
                for sessions in sessions_by_day.values()
                for session in sessions
            )
        ]
        if missing:
            raise ValueError(f"fixed chargers absent from data: {missing}")
    else:
        charger_ids = _select_fixed_chargers(
            sessions_by_day,
            calibration_days,
            loaded_days,
            limit=args.chargers,
            minimum_sessions=args.minimum_selection_sessions,
        )
    charger_set = set(charger_ids)

    filtered_by_day = {
        day: [
            session
            for session in sessions
            if session.charger_id in charger_set
        ]
        for day, sessions in sessions_by_day.items()
    }
    calibration = calibrate_one_sided_conformal(
        filtered_by_day,
        calibration_days,
        charger_ids,
        lookback_weeks=args.lookback_weeks,
        alpha=args.alpha,
    )

    tariff = summer_tariff()
    states = {
        method: {"month": "", "peak": 0.0, "onpeak_peak": 0.0}
        for method in selected_methods
    }
    monthly_cumulative = defaultdict(lambda: defaultdict(float))
    control_rows: list[dict] = []
    forecast_rows: list[dict] = []
    experiment_start = time.perf_counter()

    for day_index, target_day in enumerate(test_days, start=1):
        month = target_day[:7]
        actual = filtered_by_day[target_day]
        history_day = (
            date.fromisoformat(target_day) - timedelta(days=7)
        ).isoformat()
        persistence = with_forecast_ids(
            filtered_by_day.get(history_day, ()),
            f"persistence:{target_day}",
        )
        median_forecast = historical_median_forecast(
            filtered_by_day,
            target_day,
            charger_ids,
            lookback_weeks=args.lookback_weeks,
        )
        conformal_forecast = conformal_robust_forecast(
            median_forecast,
            calibration,
        )
        forecasts = {
            "Perfect": with_forecast_ids(
                actual,
                f"perfect:{target_day}",
            ),
            "NoForecast": [],
            "Persistence": persistence,
            "HistoricalMedian": median_forecast,
            "ConformalRobust": conformal_forecast,
        }

        for forecast_method in FORECAST_METHODS:
            metric_forecast = (
                median_forecast
                if forecast_method == "HistoricalMedian"
                else forecasts[forecast_method]
            )
            metrics = session_forecast_metrics(
                actual,
                metric_forecast,
                charger_ids,
                calibration=(
                    calibration
                    if forecast_method == "HistoricalMedian"
                    else None
                ),
            )
            forecast_rows.append(
                {
                    "target_day": target_day,
                    "method": forecast_method,
                    **metrics,
                }
            )

        for method in selected_methods:
            if states[method]["month"] != month:
                states[method] = {
                    "month": month,
                    "peak": 0.0,
                    "onpeak_peak": 0.0,
                }
            start_peak = float(states[method]["peak"])
            start_onpeak_peak = float(states[method]["onpeak_peak"])
            started = time.perf_counter()
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
                    allow_fallback=False,
                )
            elapsed = time.perf_counter() - started
            if result.unserved_energy_kwh > 1e-5:
                raise RuntimeError(
                    f"{target_day} {method} left "
                    f"{result.unserved_energy_kwh:.6f} kWh unserved"
                )
            if result.fallback_count:
                raise RuntimeError(
                    f"{target_day} {method} used "
                    f"{result.fallback_count} solver fallbacks"
                )

            standalone_cost, _, _, _ = evaluate_cost(
                result.load_kw,
                tariff,
            )
            monthly_cumulative[month][method] += result.cost
            states[method] = {
                "month": month,
                "peak": result.peak_kw,
                "onpeak_peak": result.onpeak_peak_kw,
            }
            control_rows.append(
                {
                    "target_day": target_day,
                    "month": month,
                    "method": method,
                    "incremental_cost": result.cost,
                    "monthly_cumulative_cost": (
                        monthly_cumulative[month][method]
                    ),
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
                    "runtime_seconds": elapsed,
                    "actual_session_count": len(actual),
                    "forecast_session_count": len(forecasts.get(method, [])),
                    "charger_count": len(charger_ids),
                    "charger_ids": ";".join(charger_ids),
                }
            )

        if (
            day_index % args.progress_every == 0
            or day_index == len(test_days)
        ):
            _write_csv(
                args.output_dir / "control_daily.csv",
                control_rows,
            )
            _write_csv(
                args.output_dir / "forecast_daily.csv",
                forecast_rows,
            )
            elapsed_total = time.perf_counter() - experiment_start
            print(
                f"completed {day_index}/{len(test_days)} days "
                f"({target_day}), elapsed={elapsed_total:.1f}s",
                flush=True,
            )

    months = sorted(monthly_cumulative)
    monthly_rows: list[dict] = []
    for month in months:
        v0g_cost = monthly_cumulative[month].get("V0G", math.nan)
        perfect_cost = monthly_cumulative[month].get("Perfect", math.nan)
        for method in selected_methods:
            rows = [
                row
                for row in control_rows
                if row["month"] == month and row["method"] == method
            ]
            cost = monthly_cumulative[month][method]
            total_solves = sum(int(row["solve_count"]) for row in rows)
            optimal_solves = sum(
                float(row["optimal_solve_ratio"])
                * int(row["solve_count"])
                for row in rows
            )
            monthly_rows.append(
                {
                    "month": month,
                    "method": method,
                    "continuous_cost": cost,
                    "saving_vs_v0g": v0g_cost - cost,
                    "saving_pct_vs_v0g": (
                        100.0 * (v0g_cost - cost) / v0g_cost
                        if math.isfinite(v0g_cost) and v0g_cost > 0.0
                        else math.nan
                    ),
                    "regret_vs_perfect": cost - perfect_cost,
                    "regret_pct_vs_perfect": (
                        100.0 * (cost - perfect_cost) / perfect_cost
                        if (
                            math.isfinite(perfect_cost)
                            and perfect_cost > 0.0
                        )
                        else math.nan
                    ),
                    "standalone_daily_proxy_sum": sum(
                        float(row["standalone_cost_same_load"])
                        for row in rows
                    ),
                    "daily_proxy_overstatement": sum(
                        float(row["daily_proxy_overstatement"])
                        for row in rows
                    ),
                    "energy_kwh": sum(
                        float(row["energy_kwh"]) for row in rows
                    ),
                    "required_energy_kwh": sum(
                        float(row["required_energy_kwh"])
                        for row in rows
                    ),
                    "unserved_energy_kwh": sum(
                        float(row["unserved_energy_kwh"])
                        for row in rows
                    ),
                    "final_peak_kw": rows[-1]["end_peak_kw"],
                    "final_onpeak_peak_kw": (
                        rows[-1]["end_onpeak_peak_kw"]
                    ),
                    "optimal_solve_ratio": (
                        optimal_solves / total_solves
                        if total_solves
                        else 1.0
                    ),
                    "solve_count": total_solves,
                    "fallback_count": sum(
                        int(row["fallback_count"]) for row in rows
                    ),
                    "runtime_seconds": sum(
                        float(row["runtime_seconds"]) for row in rows
                    ),
                    "test_day_count": len(rows),
                    "charger_count": len(charger_ids),
                    "charger_ids": ";".join(charger_ids),
                }
            )

    forecast_summary_rows: list[dict] = []
    for method in FORECAST_METHODS:
        rows = [row for row in forecast_rows if row["method"] == method]
        matched_total = sum(
            float(row["matched_session_count"]) for row in rows
        )
        summary = {
            "method": method,
            "count_mae_per_charger_day": sum(
                float(row["count_mae_per_charger"]) for row in rows
            )
            / len(rows),
            "arrival_mae_slots": _weighted_mean(
                rows,
                "arrival_mae_slots",
                "matched_session_count",
            ),
            "departure_mae_slots": _weighted_mean(
                rows,
                "departure_mae_slots",
                "matched_session_count",
            ),
            "energy_mae_kwh": _weighted_mean(
                rows,
                "energy_mae_kwh",
                "matched_session_count",
            ),
            "matched_session_count": matched_total,
            "actual_session_count": sum(
                float(row["actual_session_count"]) for row in rows
            ),
            "forecast_session_count": sum(
                float(row["forecast_session_count"]) for row in rows
            ),
        }
        if method == "HistoricalMedian":
            summary.update(
                {
                    "arrival_upper_coverage": _weighted_mean(
                        rows,
                        "arrival_upper_coverage",
                        "matched_session_count",
                    ),
                    "departure_lower_coverage": _weighted_mean(
                        rows,
                        "departure_lower_coverage",
                        "matched_session_count",
                    ),
                    "energy_upper_coverage": _weighted_mean(
                        rows,
                        "energy_upper_coverage",
                        "matched_session_count",
                    ),
                }
            )
        else:
            summary.update(
                {
                    "arrival_upper_coverage": math.nan,
                    "departure_lower_coverage": math.nan,
                    "energy_upper_coverage": math.nan,
                }
            )
        forecast_summary_rows.append(summary)

    calibration_rows = [
        {
            "alpha": calibration.alpha,
            "target_marginal_coverage": 1.0 - calibration.alpha,
            "arrival_late_slots": calibration.arrival_late_slots,
            "departure_early_slots": calibration.departure_early_slots,
            "energy_under_kwh": calibration.energy_under_kwh,
            "matched_residual_count": (
                calibration.matched_residual_count
            ),
            "count_under_sessions": calibration.count_under_sessions,
            "count_residual_count": calibration.count_residual_count,
            "calibration_start": args.calibration_start,
            "calibration_end": args.calibration_end,
            "lookback_weeks": args.lookback_weeks,
            "charger_count": len(charger_ids),
            "charger_ids": ";".join(charger_ids),
            "loaded_raw_sessions": sum(
                stats.get("raw", 0) for stats in data_stats.values()
            ),
            "loaded_kept_sessions": sum(
                stats.get("kept", 0) for stats in data_stats.values()
            ),
        }
    ]

    _write_csv(args.output_dir / "control_daily.csv", control_rows)
    _write_csv(args.output_dir / "control_monthly.csv", monthly_rows)
    _write_csv(args.output_dir / "forecast_daily.csv", forecast_rows)
    _write_csv(
        args.output_dir / "forecast_summary.csv",
        forecast_summary_rows,
    )
    _write_csv(args.output_dir / "calibration.csv", calibration_rows)

    print()
    print("Paper-month charger-MPC experiment")
    print("calibration:", args.calibration_start, "to", args.calibration_end)
    print("test:", args.test_start, "to", args.test_end)
    print("chargers:", len(charger_ids))
    print("charger ids:", "; ".join(charger_ids))
    print(
        "conformal corrections:",
        f"arrival +{calibration.arrival_late_slots} slots,",
        f"departure -{calibration.departure_early_slots} slots,",
        f"energy +{calibration.energy_under_kwh:.3f} kWh",
    )
    print()
    for row in monthly_rows:
        print(
            f"{row['month']} {row['method']:16s} "
            f"cost={row['continuous_cost']:.3f} "
            f"saving={row['saving_pct_vs_v0g']:.2f}% "
            f"regret={row['regret_pct_vs_perfect']:.2f}% "
            f"peak={row['final_peak_kw']:.3f}/"
            f"{row['final_onpeak_peak_kw']:.3f} kW "
            f"unserved={row['unserved_energy_kwh']:.6f} "
            f"runtime={row['runtime_seconds']:.2f}s"
        )
    print()
    for row in forecast_summary_rows:
        print(
            f"{row['method']:16s} "
            f"count-MAE={row['count_mae_per_charger_day']:.3f} "
            f"arrival-MAE={row['arrival_mae_slots']:.3f} slots "
            f"departure-MAE={row['departure_mae_slots']:.3f} slots "
            f"energy-MAE={row['energy_mae_kwh']:.3f} kWh"
        )
    print()
    print("output directory:", args.output_dir)


if __name__ == "__main__":
    main()
