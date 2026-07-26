#!/usr/bin/env python3
"""Run one small, reproducible charger-MPC experiment.

The script intentionally uses one target day, one same-weekday persistence day,
and a small charger subset.  It writes only a compact CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_mpc import (  # noqa: E402
    disaggregate_nonoverlap,
    rolling_charger_mpc,
    solve_charger_envelope,
    solve_ev_dispatch,
    summer_tariff,
    v0g_dispatch,
)
from charger_mpc.data import (  # noqa: E402
    common_busy_chargers,
    read_day_sessions,
    with_forecast_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-day", default="2023-07-01")
    parser.add_argument("--history-day", default="2023-06-24")
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
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "smoke_2023-07-01.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tariff = summer_tariff()

    all_target, target_stats = read_day_sessions(
        args.data,
        args.target_day,
    )
    all_history, history_stats = read_day_sessions(
        args.data,
        args.history_day,
    )
    charger_ids = common_busy_chargers(
        all_target,
        all_history,
        limit=args.chargers,
    )
    if not charger_ids:
        raise RuntimeError("No common target/history chargers found")
    charger_set = set(charger_ids)

    actual = [x for x in all_target if x.charger_id in charger_set]
    persistence_source = [
        x for x in all_history if x.charger_id in charger_set
    ]
    perfect_forecast = with_forecast_ids(actual, "perfect")
    persistence_forecast = with_forecast_ids(
        persistence_source,
        "persistence",
    )

    ev_oracle = solve_ev_dispatch(
        actual,
        tariff,
        time_limit=max(2.0, args.time_limit),
    )
    charger_oracle = solve_charger_envelope(
        actual,
        tariff,
        time_limit=max(2.0, args.time_limit),
    )
    disaggregate_nonoverlap(
        actual,
        charger_oracle.power_by_unit_kw,
        n_slots=tariff.n_slots,
    )
    equivalence_gap = charger_oracle.objective - ev_oracle.objective
    if abs(equivalence_gap) > 1e-5:
        raise RuntimeError(
            f"EV/charger oracle equivalence failed: gap={equivalence_gap}"
        )

    results = [
        v0g_dispatch(actual, tariff),
        rolling_charger_mpc(
            actual,
            perfect_forecast,
            tariff,
            method="Perfect",
            time_limit_per_solve=args.time_limit,
        ),
        rolling_charger_mpc(
            actual,
            [],
            tariff,
            method="NoForecast",
            time_limit_per_solve=args.time_limit,
        ),
        rolling_charger_mpc(
            actual,
            persistence_forecast,
            tariff,
            method="Persistence",
            time_limit_per_solve=args.time_limit,
        ),
    ]
    for result in results:
        if result.unserved_energy_kwh > 1e-5:
            raise RuntimeError(
                f"{result.method} left "
                f"{result.unserved_energy_kwh:.6f} kWh unserved"
            )
        if result.fallback_count:
            raise RuntimeError(
                f"{result.method} used {result.fallback_count} fallbacks"
            )
        if (
            result.method != "V0G"
            and result.cost < charger_oracle.objective - 1e-5
        ):
            raise RuntimeError(
                f"{result.method} appears cheaper than the perfect-information "
                "day-ahead oracle"
            )
    perfect_result = next(x for x in results if x.method == "Perfect")
    if abs(perfect_result.cost - charger_oracle.objective) > 1e-5:
        raise RuntimeError(
            "Perfect rolling MPC is not time-consistent with the "
            "day-ahead charger oracle"
        )

    v0g_cost = results[0].cost
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("load_kw")
        row["cost_saving_vs_v0g"] = v0g_cost - result.cost
        row["cost_saving_pct_vs_v0g"] = (
            100.0 * (v0g_cost - result.cost) / v0g_cost
            if v0g_cost > 0
            else 0.0
        )
        row["target_day"] = args.target_day
        row["history_day"] = args.history_day
        row["charger_count"] = len(charger_ids)
        row["actual_session_count"] = len(actual)
        row["forecast_session_count"] = (
            len(perfect_forecast)
            if result.method == "Perfect"
            else len(persistence_forecast)
            if result.method == "Persistence"
            else 0
        )
        row["charger_ids"] = ";".join(charger_ids)
        row["day_ahead_ev_oracle_cost"] = ev_oracle.objective
        row["day_ahead_charger_oracle_cost"] = charger_oracle.objective
        row["oracle_equivalence_gap"] = equivalence_gap
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Deterministic charger-MPC smoke test")
    print("target stats:", target_stats)
    print("history stats:", history_stats)
    print("chargers:", len(charger_ids))
    print("actual sessions:", len(actual))
    print("persistence sessions:", len(persistence_forecast))
    print("daily demand-charge proxy: yes")
    print(
        "EV/charger day-ahead oracle gap:",
        f"{equivalence_gap:.3e}",
    )
    print()
    for row in rows:
        print(
            f"{row['method']:11s} "
            f"cost={row['cost']:.3f} "
            f"saving={row['cost_saving_pct_vs_v0g']:.2f}% "
            f"energy={row['energy_kwh']:.3f}/"
            f"{row['required_energy_kwh']:.3f} kWh "
            f"unserved={row['unserved_energy_kwh']:.6f} "
            f"peak={row['peak_kw']:.3f} kW "
            f"optimal={row['optimal_solve_ratio']:.3f}"
        )
    print()
    print("output:", args.output)


if __name__ == "__main__":
    main()
