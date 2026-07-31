"""Leakage-safe flexibility-envelope panels and projection utilities.

The target is intentionally set-valued rather than session matched.  At a
small number of intraday anchor slots, every entity is represented by:

* cumulative energy that must already have been delivered (lower envelope);
* cumulative energy that could have been delivered (upper envelope); and
* occupied-port equivalents in the preceding time block.

Summing these quantities across drivers or across physical chargers produces
the same realized target when both panels contain the same sessions.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from torch import nn


EnvelopeLevel = Literal["ev", "charger"]
DEFAULT_ANCHOR_SLOTS = (32, 48, 64, 80, 84, 96)
DEFAULT_MAX_POWER_KW = 6.6
DEFAULT_DELTA_T = 0.25
_TOL = 1e-7


def envelope_feature_names(
    anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
) -> tuple[str, ...]:
    anchors = _validate_anchors(anchor_slots)
    return tuple(
        [f"lower_cumulative_kwh_slot_{slot}" for slot in anchors]
        + [f"upper_cumulative_kwh_slot_{slot}" for slot in anchors]
        + [f"occupied_port_equivalent_to_slot_{slot}" for slot in anchors]
    )


@dataclass(frozen=True)
class EnvelopePanel:
    """Dense daily panel of compressed cumulative feasibility envelopes."""

    level: EnvelopeLevel
    dates: tuple[str, ...]
    entity_ids: tuple[str, ...]
    values: np.ndarray
    selection_end: str
    anchor_slots: tuple[int, ...]
    quality_stats: dict[str, int]

    def __post_init__(self) -> None:
        expected = (
            len(self.dates),
            len(self.entity_ids),
            3 * len(self.anchor_slots),
        )
        if self.values.shape != expected:
            raise ValueError(
                f"panel has shape {self.values.shape}, expected {expected}"
            )


@dataclass(frozen=True)
class EnvelopeScaler:
    """Feature-wise scaler without session-specific masking assumptions."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "EnvelopeScaler":
        if train_values.ndim != 3:
            raise ValueError("expected [day, entity, feature] values")
        mean = train_values.mean(axis=(0, 1), dtype=np.float64)
        scale = train_values.std(axis=(0, 1), dtype=np.float64)
        scale = np.where(scale > 1e-6, scale, 1.0)
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)


class FeasibleEnvelopeOutput(nn.Module):
    """Differentiable physical output layer for any entity regressor.

    The wrapped model emits unconstrained latent values. Occupancy is mapped
    through a sigmoid, total energy is bounded by forecast occupancy capacity,
    and cumulative upper/lower curves are monotonized while preserving
    terminal equality, ``lower <= upper``, cumulative capacity, and enough
    delivered energy to finish within the remaining capacity. The layer
    returns values in the same standardized coordinates used by the training
    loss, so feasibility is part of the learned architecture rather than an
    inference-only repair.
    """

    def __init__(
        self,
        base_model: nn.Module,
        scaler: EnvelopeScaler,
        *,
        anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
        delta_t: float = DEFAULT_DELTA_T,
        max_power_kw: float = DEFAULT_MAX_POWER_KW,
    ) -> None:
        super().__init__()
        anchors = _validate_anchors(anchor_slots)
        expected = 3 * len(anchors)
        if scaler.mean.shape != (expected,) or scaler.scale.shape != (
            expected,
        ):
            raise ValueError("scaler does not match envelope feature count")
        self.base_model = base_model
        self.n_anchors = len(anchors)
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(scaler.mean, dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(scaler.scale, dtype=torch.float32),
        )
        self.register_buffer(
            "block_capacity_kwh",
            torch.as_tensor(
                np.diff((0,) + anchors)
                * delta_t
                * max_power_kw,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        calendar: torch.Tensor,
    ) -> torch.Tensor:
        latent = self.base_model(x, calendar)
        n = self.n_anchors
        if latent.shape[-1] != 3 * n:
            raise ValueError("base model output does not match envelope")

        lower_logits = latent[..., :n]
        upper_logits = latent[..., n : 2 * n]
        occupied = torch.sigmoid(latent[..., 2 * n :])
        block_capacity = occupied * self.block_capacity_kwh
        cumulative_capacity = torch.cumsum(block_capacity, dim=-1)
        total_capacity = cumulative_capacity[..., -1]
        energy_logit = 0.5 * (
            lower_logits[..., -1] + upper_logits[..., -1]
        )
        terminal = total_capacity * torch.sigmoid(energy_logit)

        upper_target = terminal.unsqueeze(-1) * torch.cumsum(
            torch.softmax(upper_logits, dim=-1),
            dim=-1,
        )
        remaining_capacity = (
            total_capacity.unsqueeze(-1) - cumulative_capacity
        )
        minimum_to_finish = torch.relu(
            terminal.unsqueeze(-1) - remaining_capacity
        )
        upper = torch.minimum(
            torch.maximum(upper_target, minimum_to_finish),
            cumulative_capacity,
        )
        upper = torch.cummax(upper, dim=-1).values
        upper = torch.cat(
            (upper[..., :-1], terminal.unsqueeze(-1)),
            dim=-1,
        )

        lower_target = terminal.unsqueeze(-1) * torch.cumsum(
            torch.softmax(lower_logits, dim=-1),
            dim=-1,
        )
        lower = torch.maximum(lower_target, minimum_to_finish)
        lower = torch.minimum(lower, upper)
        lower = torch.cummax(lower, dim=-1).values
        lower = torch.cat(
            (lower[..., :-1], terminal.unsqueeze(-1)),
            dim=-1,
        )
        physical = torch.cat((lower, upper, occupied), dim=-1)
        return (
            (physical - self.feature_mean) / self.feature_scale
        )


@dataclass(frozen=True)
class _Record:
    day: str
    driver_id: str
    charger_id: str
    arrival: int
    departure: int
    energy_kwh: float
    max_power_kw: float


def _validate_anchors(anchor_slots: Sequence[int]) -> tuple[int, ...]:
    anchors = tuple(int(slot) for slot in anchor_slots)
    if (
        not anchors
        or anchors[-1] != 96
        or any(slot < 1 or slot > 96 for slot in anchors)
        or any(right <= left for left, right in zip(anchors, anchors[1:]))
    ):
        raise ValueError(
            "anchor slots must be strictly increasing, within 1..96, "
            "and end at slot 96"
        )
    return anchors


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


def _discrete_slots(
    start_value: str,
    end_value: str,
) -> tuple[int, int] | None:
    start = datetime.fromisoformat(start_value)
    end = datetime.fromisoformat(end_value)
    if end.date() != start.date() or end < start:
        return None
    arrival_float = (start.hour + start.minute / 60.0) / 24.0 * 96
    departure_float = (end.hour + end.minute / 60.0) / 24.0 * 96
    arrival = min(96, max(1, math.ceil(arrival_float) + 1))
    departure = min(96, max(1, math.floor(departure_float) + 1))
    if departure < arrival:
        return None
    return int(arrival), int(departure)


def _read_screened_records(
    path: Path,
    *,
    start: str,
    end: str,
    charger_filter: set[str] | None,
    delta_t: float,
    max_power_kw: float,
) -> tuple[list[_Record], dict[str, int]]:
    grouped: dict[tuple[str, str], list[_Record]] = defaultdict(list)
    stats: Counter[str] = Counter()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            day = row["session_start_time_la"][:10]
            if day < start or day > end:
                continue
            charger_id = _charger_id(row)
            if charger_filter is not None and charger_id not in charger_filter:
                continue
            stats["raw_sessions"] += 1
            slots = _discrete_slots(
                row["session_start_time_la"],
                row["session_end_time_la"],
            )
            if slots is None:
                stats["invalid_window_sessions"] += 1
                continue
            arrival, departure = slots
            energy = float(row["total_energy_dispensed"])
            capacity = (
                max_power_kw * delta_t * (departure - arrival + 1)
            )
            if energy <= 0.0 or energy > capacity + _TOL:
                stats["infeasible_energy_sessions"] += 1
                continue
            grouped[(day, charger_id)].append(
                _Record(
                    day=day,
                    driver_id=str(row["driver_id"]),
                    charger_id=charger_id,
                    arrival=arrival,
                    departure=departure,
                    energy_kwh=energy,
                    max_power_kw=max_power_kw,
                )
            )

    records: list[_Record] = []
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda item: (
                item.arrival,
                item.departure,
                item.driver_id,
            ),
        )
        if any(
            current.arrival <= previous.departure
            for previous, current in zip(ordered, ordered[1:])
        ):
            stats["overlap_port_days"] += 1
            stats["overlap_sessions_dropped"] += len(ordered)
            continue
        records.extend(ordered)
    stats["kept_sessions"] = len(records)
    return records, dict(stats)


def _select_entities(
    records: Sequence[_Record],
    *,
    level: EnvelopeLevel,
    selection_start: str,
    selection_end: str,
    top_k: int | None,
    minimum_sessions: int,
    driver_filter: set[str] | None,
) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    energy: Counter[str] = Counter()
    for record in records:
        if not selection_start <= record.day <= selection_end:
            continue
        if (
            driver_filter is not None
            and record.driver_id not in driver_filter
        ):
            continue
        entity_id = (
            record.driver_id
            if level == "ev"
            else record.charger_id
        )
        counts[entity_id] += 1
        energy[entity_id] += record.energy_kwh
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
        raise ValueError("no entities satisfy training-only selection")
    return tuple(candidates)


def build_envelope_panel(
    path: str | Path,
    *,
    level: EnvelopeLevel,
    start: str,
    end: str,
    selection_end: str,
    selection_start: str | None = None,
    entity_ids: Sequence[str] | None = None,
    top_k: int | None = None,
    minimum_sessions: int = 1,
    charger_filter: Iterable[str] | None = None,
    driver_filter: Iterable[str] | None = None,
    anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
    delta_t: float = DEFAULT_DELTA_T,
    max_power_kw: float = DEFAULT_MAX_POWER_KW,
) -> EnvelopePanel:
    """Build EV- or charger-level panels from physically screened sessions."""

    if level not in ("ev", "charger"):
        raise ValueError("level must be 'ev' or 'charger'")
    if selection_end > end:
        raise ValueError("selection_end cannot follow panel end")
    if delta_t <= 0.0 or max_power_kw <= 0.0:
        raise ValueError("delta_t and max_power_kw must be positive")
    anchors = _validate_anchors(anchor_slots)
    dates = _dates(start, end)
    charger_set = (
        None if charger_filter is None else set(charger_filter)
    )
    driver_set = None if driver_filter is None else set(driver_filter)
    records, quality_stats = _read_screened_records(
        Path(path),
        start=start,
        end=end,
        charger_filter=charger_set,
        delta_t=delta_t,
        max_power_kw=max_power_kw,
    )
    if entity_ids is None:
        selected = _select_entities(
            records,
            level=level,
            selection_start=selection_start or start,
            selection_end=selection_end,
            top_k=top_k,
            minimum_sessions=minimum_sessions,
            driver_filter=driver_set,
        )
    else:
        selected = tuple(dict.fromkeys(str(value) for value in entity_ids))
        if not selected:
            raise ValueError("entity_ids cannot be empty")

    date_index = {day: idx for idx, day in enumerate(dates)}
    entity_index = {
        entity_id: idx for idx, entity_id in enumerate(selected)
    }
    n_anchors = len(anchors)
    values = np.zeros(
        (len(dates), len(selected), 3 * n_anchors),
        dtype=np.float32,
    )
    block_starts = (1,) + tuple(slot + 1 for slot in anchors[:-1])

    for record in records:
        if (
            driver_set is not None
            and record.driver_id not in driver_set
        ):
            continue
        entity_id = (
            record.driver_id
            if level == "ev"
            else record.charger_id
        )
        entity_idx = entity_index.get(entity_id)
        if entity_idx is None:
            continue
        day_idx = date_index[record.day]
        cell = values[day_idx, entity_idx]
        for anchor_idx, (block_start, anchor) in enumerate(
            zip(block_starts, anchors)
        ):
            elapsed_slots = max(
                0,
                min(anchor, record.departure) - record.arrival + 1,
            )
            future_slots = max(
                0,
                record.departure
                - max(anchor + 1, record.arrival)
                + 1,
            )
            cell[anchor_idx] += max(
                0.0,
                record.energy_kwh
                - record.max_power_kw * delta_t * future_slots,
            )
            cell[n_anchors + anchor_idx] += min(
                record.energy_kwh,
                record.max_power_kw * delta_t * elapsed_slots,
            )
            occupied_slots = max(
                0,
                min(anchor, record.departure)
                - max(block_start, record.arrival)
                + 1,
            )
            block_length = anchor - block_start + 1
            cell[2 * n_anchors + anchor_idx] += (
                occupied_slots / block_length
            )

    return EnvelopePanel(
        level=level,
        dates=dates,
        entity_ids=selected,
        values=values,
        selection_end=selection_end,
        anchor_slots=anchors,
        quality_stats=quality_stats,
    )


def envelope_validity_mask(
    values: np.ndarray,
    *,
    anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
    delta_t: float = DEFAULT_DELTA_T,
    max_power_kw: float = DEFAULT_MAX_POWER_KW,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Return one validity flag per signature in ``values``."""

    anchors = _validate_anchors(anchor_slots)
    if values.shape[-1] != 3 * len(anchors):
        raise ValueError("unexpected envelope feature count")
    n = len(anchors)
    lower = values[..., :n]
    upper = values[..., n : 2 * n]
    occupied = values[..., 2 * n :]
    block_lengths = np.diff((0,) + anchors)
    capacity = occupied * (
        block_lengths * delta_t * max_power_kw
    )
    cumulative_capacity = np.cumsum(capacity, axis=-1)
    finite = np.isfinite(values).all(axis=-1)
    nonnegative = (
        (lower >= -tolerance).all(axis=-1)
        & (upper >= -tolerance).all(axis=-1)
        & (occupied >= -tolerance).all(axis=-1)
    )
    occupied_valid = (occupied <= 1.0 + tolerance).all(axis=-1)
    monotone = (
        (np.diff(lower, axis=-1) >= -tolerance).all(axis=-1)
        & (np.diff(upper, axis=-1) >= -tolerance).all(axis=-1)
    )
    ordered = (lower <= upper + tolerance).all(axis=-1)
    terminal = (
        np.abs(lower[..., -1] - upper[..., -1]) <= tolerance
    )
    remaining_capacity = (
        cumulative_capacity[..., -1, None] - cumulative_capacity
    )
    minimum_to_finish = np.maximum(
        0.0,
        lower[..., -1, None] - remaining_capacity,
    )
    capacity_valid = (
        (upper <= cumulative_capacity + tolerance).all(axis=-1)
        & (
            lower[..., -1]
            <= cumulative_capacity[..., -1] + tolerance
        )
        & (
            lower >= minimum_to_finish - tolerance
        ).all(axis=-1)
    )
    return (
        finite
        & nonnegative
        & occupied_valid
        & monotone
        & ordered
        & terminal
        & capacity_valid
    )


def project_envelope_signatures(
    values: np.ndarray,
    *,
    anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
    delta_t: float = DEFAULT_DELTA_T,
    max_power_kw: float = DEFAULT_MAX_POWER_KW,
) -> np.ndarray:
    """Project raw model outputs into a compact physical envelope cone."""

    anchors = _validate_anchors(anchor_slots)
    if values.shape[-1] != 3 * len(anchors):
        raise ValueError("unexpected envelope feature count")
    n = len(anchors)
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 3 * n)
    output = np.zeros_like(flat)
    block_lengths = np.diff((0,) + anchors).astype(np.float64)

    for row_idx, row in enumerate(flat):
        raw_lower = np.nan_to_num(row[:n], nan=0.0)
        raw_upper = np.nan_to_num(row[n : 2 * n], nan=0.0)
        occupied = np.clip(
            np.nan_to_num(row[2 * n :], nan=0.0),
            0.0,
            1.0,
        )
        block_capacity = (
            occupied * block_lengths * delta_t * max_power_kw
        )
        cumulative_capacity = np.cumsum(block_capacity)
        raw_terminal = max(
            0.0,
            0.5 * (raw_lower[-1] + raw_upper[-1]),
        )
        terminal = min(raw_terminal, cumulative_capacity[-1])

        upper_target = np.maximum.accumulate(
            np.clip(raw_upper, 0.0, terminal)
        )
        remaining_capacity = (
            cumulative_capacity[-1] - cumulative_capacity
        )
        minimum_to_finish = np.maximum(
            0.0,
            terminal - remaining_capacity,
        )
        upper = np.maximum(upper_target, minimum_to_finish)
        upper = np.minimum(upper, cumulative_capacity)
        upper = np.minimum(upper, terminal)
        upper = np.maximum.accumulate(upper)

        lower = np.maximum.accumulate(
            np.clip(raw_lower, 0.0, terminal)
        )
        lower = np.maximum(lower, minimum_to_finish)
        lower = np.minimum(lower, upper)
        lower = np.maximum.accumulate(lower)
        lower[-1] = terminal
        upper[-1] = terminal

        output[row_idx, :n] = lower
        output[row_idx, n : 2 * n] = upper
        output[row_idx, 2 * n :] = occupied
    return output.reshape(values.shape).astype(np.float32)


def envelope_target_weights(
    raw_target: np.ndarray,
    *,
    active_weight: float = 3.0,
    anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
) -> np.ndarray:
    """Weight active entity-days without discarding informative zeros."""

    anchors = _validate_anchors(anchor_slots)
    if raw_target.shape[-1] != 3 * len(anchors):
        raise ValueError("unexpected envelope feature count")
    if active_weight < 1.0:
        raise ValueError("active_weight must be at least one")
    terminal = raw_target[..., len(anchors) - 1]
    weights = np.ones_like(raw_target, dtype=np.float32)
    weights *= np.where(
        terminal[..., None] > 0.0,
        active_weight,
        1.0,
    )
    return weights


def envelope_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    target_dates: Sequence[str],
    anchor_slots: Sequence[int] = DEFAULT_ANCHOR_SLOTS,
) -> dict[str, float]:
    """Set-free metrics on aggregate flexibility envelopes."""

    anchors = _validate_anchors(anchor_slots)
    if actual.shape != prediction.shape or actual.ndim != 3:
        raise ValueError("actual/prediction shape mismatch")
    if actual.shape[-1] != 3 * len(anchors):
        raise ValueError("unexpected envelope feature count")
    if actual.shape[0] != len(target_dates):
        raise ValueError("one target date is required per panel day")
    n = len(anchors)
    aggregate_actual = actual.sum(axis=1)
    aggregate_prediction = prediction.sum(axis=1)
    lower_error = np.abs(
        aggregate_actual[:, :n] - aggregate_prediction[:, :n]
    )
    upper_error = np.abs(
        aggregate_actual[:, n : 2 * n]
        - aggregate_prediction[:, n : 2 * n]
    )
    occupied_error = np.abs(
        aggregate_actual[:, 2 * n :]
        - aggregate_prediction[:, 2 * n :]
    )
    actual_energy = aggregate_actual[:, n - 1]
    predicted_energy = aggregate_prediction[:, n - 1]
    energy_error = np.abs(actual_energy - predicted_energy)
    actual_width = aggregate_actual[:, n : 2 * n] - aggregate_actual[:, :n]
    predicted_width = (
        aggregate_prediction[:, n : 2 * n]
        - aggregate_prediction[:, :n]
    )
    actual_active = actual[..., n - 1] > 0.0
    predicted_active = prediction[..., n - 1] >= 0.5
    true_positive = float(
        np.logical_and(actual_active, predicted_active).sum()
    )
    false_positive = float(
        np.logical_and(~actual_active, predicted_active).sum()
    )
    false_negative = float(
        np.logical_and(actual_active, ~predicted_active).sum()
    )
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    return {
        "aggregate_lower_curve_mae_kwh": float(lower_error.mean()),
        "aggregate_upper_curve_mae_kwh": float(upper_error.mean()),
        "aggregate_occupied_equivalent_mae": float(
            occupied_error.mean()
        ),
        "aggregate_terminal_energy_mae_kwh": float(energy_error.mean()),
        "aggregate_terminal_energy_wape": float(
            energy_error.sum() / max(1e-9, actual_energy.sum())
        ),
        "aggregate_flexibility_width_mae_kwh": float(
            np.abs(actual_width - predicted_width).mean()
        ),
        "scope_test_energy_kwh": float(actual_energy.sum()),
        "actual_active_rate": float(actual_active.mean()),
        "predicted_active_rate": float(predicted_active.mean()),
        "participation_precision": precision,
        "participation_recall": recall,
        "participation_f1": (
            2.0 * precision * recall
            / max(1e-9, precision + recall)
        ),
    }
