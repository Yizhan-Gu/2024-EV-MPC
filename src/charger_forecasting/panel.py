"""Daily panel construction for distinct EV- and charger-level tasks."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np


FEATURE_NAMES = (
    "session_count",
    "energy_kwh",
    "mean_arrival_slot",
    "mean_departure_slot",
    "mean_dwell_slots",
)
Level = Literal["ev", "charger"]


@dataclass(frozen=True)
class PanelData:
    """Dense daily entity panel with compact session-derived features."""

    level: Level
    dates: tuple[str, ...]
    entity_ids: tuple[str, ...]
    values: np.ndarray
    selection_end: str

    def __post_init__(self) -> None:
        expected = (
            len(self.dates),
            len(self.entity_ids),
            len(FEATURE_NAMES),
        )
        if self.values.shape != expected:
            raise ValueError(
                f"panel has shape {self.values.shape}, expected {expected}"
            )


def _dates(start: str, end: str) -> tuple[str, ...]:
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("panel end precedes start")
    return tuple(
        (first + timedelta(days=offset)).isoformat()
        for offset in range((last - first).days + 1)
    )


def _charger_id(row: dict[str, str]) -> str:
    return f"{row['station_name']}|{row['port']}"


def _entity_id(row: dict[str, str], level: Level) -> str:
    return str(row["driver_id"]) if level == "ev" else _charger_id(row)


def _session_slots(
    start_value: str,
    end_value: str,
) -> tuple[float, float] | None:
    """Return within-day arrival/departure slots without dropping overnights."""

    start = datetime.fromisoformat(start_value)
    end = datetime.fromisoformat(end_value)
    if end < start:
        return None
    midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
    arrival = 1.0 + (start - midnight).total_seconds() / 900.0
    departure = 1.0 + (end - midnight).total_seconds() / 900.0
    return (
        min(96.0, max(1.0, arrival)),
        min(96.0, max(1.0, departure)),
    )


def _select_entities(
    path: Path,
    *,
    level: Level,
    selection_start: str,
    selection_end: str,
    top_k: int | None,
    minimum_sessions: int,
    charger_filter: set[str] | None,
    driver_filter: set[str] | None,
) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    energy: Counter[str] = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            day = row["session_start_time_la"][:10]
            if day < selection_start or day > selection_end:
                continue
            charger_id = _charger_id(row)
            if charger_filter is not None and charger_id not in charger_filter:
                continue
            if (
                driver_filter is not None
                and str(row["driver_id"]) not in driver_filter
            ):
                continue
            entity_id = _entity_id(row, level)
            counts[entity_id] += 1
            energy[entity_id] += float(row["total_energy_dispensed"])

    candidates = [
        entity_id
        for entity_id, count in counts.items()
        if count >= minimum_sessions
    ]
    candidates.sort(key=lambda x: (-counts[x], -energy[x], x))
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        candidates = candidates[:top_k]
    if not candidates:
        raise ValueError("no entities satisfy the training-only selection")
    return tuple(candidates)


def build_daily_panel(
    path: str | Path,
    *,
    level: Level,
    start: str,
    end: str,
    selection_end: str,
    selection_start: str | None = None,
    entity_ids: Sequence[str] | None = None,
    top_k: int | None = None,
    minimum_sessions: int = 1,
    charger_filter: Iterable[str] | None = None,
    driver_filter: Iterable[str] | None = None,
) -> PanelData:
    """Build a daily panel without using post-selection outcomes.

    EV panels use ``driver_id`` as the entity. Charger panels use the physical
    ``station_name|port`` identifier. Automatic entity ranking ends at
    ``selection_end``; test-period activity therefore cannot determine the
    cohort.
    """

    if level not in ("ev", "charger"):
        raise ValueError("level must be 'ev' or 'charger'")
    if selection_end > end:
        raise ValueError("selection_end cannot follow panel end")
    panel_path = Path(path)
    date_ids = _dates(start, end)
    charger_set = (
        None if charger_filter is None else set(charger_filter)
    )
    driver_set = None if driver_filter is None else set(driver_filter)
    if entity_ids is None:
        selected = _select_entities(
            panel_path,
            level=level,
            selection_start=selection_start or start,
            selection_end=selection_end,
            top_k=top_k,
            minimum_sessions=minimum_sessions,
            charger_filter=charger_set,
            driver_filter=driver_set,
        )
    else:
        selected = tuple(dict.fromkeys(str(value) for value in entity_ids))
        if not selected:
            raise ValueError("entity_ids cannot be empty")

    day_index = {day: idx for idx, day in enumerate(date_ids)}
    entity_index = {
        entity_id: idx for idx, entity_id in enumerate(selected)
    }
    values = np.zeros(
        (len(date_ids), len(selected), len(FEATURE_NAMES)),
        dtype=np.float32,
    )

    with panel_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            day = row["session_start_time_la"][:10]
            day_idx = day_index.get(day)
            if day_idx is None:
                continue
            charger_id = _charger_id(row)
            if charger_set is not None and charger_id not in charger_set:
                continue
            if (
                driver_set is not None
                and str(row["driver_id"]) not in driver_set
            ):
                continue
            entity_id = _entity_id(row, level)
            entity_idx = entity_index.get(entity_id)
            if entity_idx is None:
                continue
            slots = _session_slots(
                row["session_start_time_la"],
                row["session_end_time_la"],
            )
            if slots is None:
                continue
            arrival, departure = slots
            energy = float(row["total_energy_dispensed"])
            cell = values[day_idx, entity_idx]
            cell[0] += 1.0
            cell[1] += energy
            cell[2] += arrival
            cell[3] += departure
            cell[4] += departure - arrival

    active = values[:, :, 0] > 0.0
    for feature_idx in (2, 3, 4):
        values[:, :, feature_idx] = np.divide(
            values[:, :, feature_idx],
            values[:, :, 0],
            out=np.zeros_like(values[:, :, feature_idx]),
            where=active,
        )
    return PanelData(
        level=level,
        dates=date_ids,
        entity_ids=selected,
        values=values,
        selection_end=selection_end,
    )
