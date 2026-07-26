"""CSV loading helpers for the small UCSD charger-MPC experiments."""

from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .core import Session


def _slot_from_timestamp(value: str, *, arrival: bool) -> int:
    timestamp = datetime.fromisoformat(value)
    hour = timestamp.hour + timestamp.minute / 60.0
    scaled = hour / 24.0 * 96
    index = math.ceil(scaled) + 1 if arrival else math.floor(scaled) + 1
    return min(96, max(1, int(index)))


def read_day_sessions(
    path: str | Path,
    day: str,
    *,
    charger_ids: set[str] | None = None,
    delta_t: float = 0.25,
    max_power_kw: float = 6.6,
) -> tuple[list[Session], dict[str, int]]:
    """Read, discretize, and feasibility-filter one day without pandas."""

    sessions_by_day, stats_by_day = read_sessions_by_days(
        path,
        [day],
        charger_ids=charger_ids,
        delta_t=delta_t,
        max_power_kw=max_power_kw,
    )
    return sessions_by_day[day], stats_by_day[day]


def read_sessions_by_days(
    path: str | Path,
    days: Iterable[str],
    *,
    charger_ids: set[str] | None = None,
    delta_t: float = 0.25,
    max_power_kw: float = 6.6,
) -> tuple[dict[str, list[Session]], dict[str, dict[str, int]]]:
    """Read multiple dates in one CSV pass using the day-level policy."""

    requested_days = tuple(dict.fromkeys(days))
    requested_set = set(requested_days)
    sessions_by_day: dict[str, list[Session]] = {
        day: [] for day in requested_days
    }
    counters = {day: Counter() for day in requested_days}
    with Path(path).open(newline="") as handle:
        for row_idx, row in enumerate(csv.DictReader(handle), start=2):
            day = row["session_start_time_la"][:10]
            if day not in requested_set:
                continue
            stats = counters[day]
            stats["raw"] += 1
            charger_id = f"{row['station_name']}|{row['port']}"
            if charger_ids is not None and charger_id not in charger_ids:
                continue
            arrival = _slot_from_timestamp(
                row["session_start_time_la"],
                arrival=True,
            )
            departure = _slot_from_timestamp(
                row["session_end_time_la"],
                arrival=False,
            )
            energy = float(row["total_energy_dispensed"])
            if departure < arrival:
                stats["invalid_window"] += 1
                continue
            capacity = max_power_kw * delta_t * (departure - arrival + 1)
            if energy <= 0.0 or energy > capacity + 1e-9:
                stats["infeasible_energy"] += 1
                continue
            sessions_by_day[day].append(
                Session(
                    session_id=f"{day}:{row_idx}",
                    charger_id=charger_id,
                    arrival=arrival,
                    departure=departure,
                    energy_kwh=energy,
                    max_power_kw=max_power_kw,
                )
            )
            stats["kept"] += 1
    for sessions in sessions_by_day.values():
        sessions.sort(
            key=lambda x: (
                x.charger_id,
                x.arrival,
                x.departure,
                x.session_id,
            )
        )
    return (
        sessions_by_day,
        {day: dict(counters[day]) for day in requested_days},
    )


def common_busy_chargers(
    target_sessions: Iterable[Session],
    history_sessions: Iterable[Session],
    *,
    limit: int,
) -> list[str]:
    """Choose busy target-day chargers that also exist in the history day."""

    target_counts = Counter(x.charger_id for x in target_sessions)
    target_energy = Counter()
    for session in target_sessions:
        target_energy[session.charger_id] += session.energy_kwh
    history_ids = {x.charger_id for x in history_sessions}
    candidates = [x for x in target_counts if x in history_ids]
    candidates.sort(
        key=lambda x: (-target_counts[x], -target_energy[x], x)
    )
    return candidates[:limit]


def with_forecast_ids(
    sessions: Iterable[Session],
    prefix: str,
) -> list[Session]:
    """Copy sessions into a forecast namespace to avoid identity leakage."""

    return [
        Session(
            session_id=f"{prefix}:{idx}",
            charger_id=session.charger_id,
            arrival=session.arrival,
            departure=session.departure,
            energy_kwh=session.energy_kwh,
            max_power_kw=session.max_power_kw,
            source="forecast",
        )
        for idx, session in enumerate(sessions)
    ]
