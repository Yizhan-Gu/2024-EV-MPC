"""Causal nonparametric and conformal charger-session forecasts."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping, Sequence

from .core import Session


@dataclass(frozen=True)
class ConformalCalibration:
    """One-sided split-conformal corrections for matched sessions."""

    alpha: float
    arrival_late_slots: int
    departure_early_slots: int
    energy_under_kwh: float
    matched_residual_count: int
    count_residual_count: int
    count_under_sessions: int


def _previous_same_weekdays(target_day: str, lookback_weeks: int) -> list[str]:
    target = date.fromisoformat(target_day)
    return [
        (target - timedelta(days=7 * lag)).isoformat()
        for lag in range(1, lookback_weeks + 1)
    ]


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _charger_sessions(
    sessions: Sequence[Session],
    charger_id: str,
) -> list[Session]:
    return sorted(
        (session for session in sessions if session.charger_id == charger_id),
        key=lambda x: (x.arrival, x.departure, x.session_id),
    )


def historical_median_forecast(
    sessions_by_day: Mapping[str, Sequence[Session]],
    target_day: str,
    charger_ids: Sequence[str],
    *,
    lookback_weeks: int = 4,
    n_slots: int = 96,
    delta_t: float = 0.25,
) -> list[Session]:
    """Forecast each charger from ordinal-session historical medians."""

    if lookback_weeks <= 0:
        raise ValueError("lookback_weeks must be positive")
    history_days = _previous_same_weekdays(target_day, lookback_weeks)
    forecast: list[Session] = []

    for charger_id in charger_ids:
        samples_by_day = [
            _charger_sessions(sessions_by_day.get(day, ()), charger_id)
            for day in history_days
        ]
        counts = [len(samples) for samples in samples_by_day]
        forecast_count = _round_half_up(float(statistics.median(counts)))
        next_available = 1

        for ordinal in range(forecast_count):
            samples = [
                day_sessions[ordinal]
                for day_sessions in samples_by_day
                if ordinal < len(day_sessions)
            ]
            if not samples:
                continue
            arrival = _round_half_up(
                float(statistics.median(x.arrival for x in samples))
            )
            departure = _round_half_up(
                float(statistics.median(x.departure for x in samples))
            )
            energy = float(statistics.median(x.energy_kwh for x in samples))
            max_power = float(
                statistics.median(x.max_power_kw for x in samples)
            )

            arrival = max(1, min(n_slots, arrival, departure))
            departure = max(arrival, min(n_slots, departure))
            arrival = max(arrival, next_available)
            if arrival > departure:
                continue
            capacity = max_power * delta_t * (departure - arrival + 1)
            energy = min(max(0.0, energy), capacity)
            if energy <= 0.0:
                continue
            forecast.append(
                Session(
                    session_id=(
                        f"histmedian:{target_day}:{charger_id}:{ordinal}"
                    ),
                    charger_id=charger_id,
                    arrival=arrival,
                    departure=departure,
                    energy_kwh=energy,
                    max_power_kw=max_power,
                    source="historical_median",
                )
            )
            next_available = departure + 1

    return forecast


def _matched_pairs(
    actual: Sequence[Session],
    forecast: Sequence[Session],
    charger_ids: Sequence[str],
) -> list[tuple[Session, Session]]:
    pairs: list[tuple[Session, Session]] = []
    for charger_id in charger_ids:
        actual_charger = _charger_sessions(actual, charger_id)
        forecast_charger = _charger_sessions(forecast, charger_id)
        pairs.extend(zip(actual_charger, forecast_charger))
    return pairs


def _finite_sample_quantile(values: Sequence[float], alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    if not values:
        raise ValueError("conformal calibration requires residuals")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    return ordered[min(len(ordered), max(1, rank)) - 1]


def calibrate_one_sided_conformal(
    sessions_by_day: Mapping[str, Sequence[Session]],
    calibration_days: Sequence[str],
    charger_ids: Sequence[str],
    *,
    lookback_weeks: int = 4,
    alpha: float = 0.1,
) -> ConformalCalibration:
    """Calibrate marginal upper-risk corrections without test-day data."""

    arrival_late: list[float] = []
    departure_early: list[float] = []
    energy_under: list[float] = []
    count_under: list[float] = []

    for calibration_day in calibration_days:
        actual = list(sessions_by_day.get(calibration_day, ()))
        forecast = historical_median_forecast(
            sessions_by_day,
            calibration_day,
            charger_ids,
            lookback_weeks=lookback_weeks,
        )
        for observed, predicted in _matched_pairs(
            actual,
            forecast,
            charger_ids,
        ):
            arrival_late.append(observed.arrival - predicted.arrival)
            departure_early.append(
                predicted.departure - observed.departure
            )
            energy_under.append(
                observed.energy_kwh - predicted.energy_kwh
            )
        actual_counts = Counter(x.charger_id for x in actual)
        forecast_counts = Counter(x.charger_id for x in forecast)
        for charger_id in charger_ids:
            count_under.append(
                actual_counts[charger_id] - forecast_counts[charger_id]
            )

    return ConformalCalibration(
        alpha=alpha,
        arrival_late_slots=max(
            0,
            int(math.ceil(_finite_sample_quantile(arrival_late, alpha))),
        ),
        departure_early_slots=max(
            0,
            int(math.ceil(_finite_sample_quantile(departure_early, alpha))),
        ),
        energy_under_kwh=max(
            0.0,
            _finite_sample_quantile(energy_under, alpha),
        ),
        matched_residual_count=len(arrival_late),
        count_residual_count=len(count_under),
        count_under_sessions=max(
            0,
            int(math.ceil(_finite_sample_quantile(count_under, alpha))),
        ),
    )


def conformal_robust_forecast(
    point_forecast: Sequence[Session],
    calibration: ConformalCalibration,
    *,
    n_slots: int = 96,
    delta_t: float = 0.25,
) -> list[Session]:
    """Tighten matched-session flexibility using one-sided quantiles.

    Session-count uncertainty is reported by the calibration object but is not
    synthesized into fictitious extra sessions. This limitation is explicit so
    that physical windows are never invented without empirical attributes.
    """

    robust: list[Session] = []
    for idx, session in enumerate(point_forecast):
        arrival = min(
            n_slots,
            session.arrival + calibration.arrival_late_slots,
        )
        departure = max(
            arrival,
            session.departure - calibration.departure_early_slots,
        )
        capacity = (
            session.max_power_kw
            * delta_t
            * (departure - arrival + 1)
        )
        energy = min(
            capacity,
            session.energy_kwh + calibration.energy_under_kwh,
        )
        if energy <= 0.0:
            continue
        robust.append(
            Session(
                session_id=f"conformal:{idx}:{session.session_id}",
                charger_id=session.charger_id,
                arrival=arrival,
                departure=departure,
                energy_kwh=energy,
                max_power_kw=session.max_power_kw,
                source="conformal_robust",
            )
        )
    return robust


def session_forecast_metrics(
    actual: Sequence[Session],
    forecast: Sequence[Session],
    charger_ids: Sequence[str],
    *,
    calibration: ConformalCalibration | None = None,
) -> dict[str, float]:
    """Return count and ordinal-matched session forecast diagnostics."""

    pairs = _matched_pairs(actual, forecast, charger_ids)
    actual_counts = Counter(x.charger_id for x in actual)
    forecast_counts = Counter(x.charger_id for x in forecast)
    count_absolute_error = sum(
        abs(actual_counts[charger_id] - forecast_counts[charger_id])
        for charger_id in charger_ids
    )
    metrics: dict[str, float] = {
        "actual_session_count": float(len(actual)),
        "forecast_session_count": float(len(forecast)),
        "matched_session_count": float(len(pairs)),
        "count_mae_per_charger": (
            count_absolute_error / len(charger_ids)
            if charger_ids
            else 0.0
        ),
        "arrival_mae_slots": (
            sum(abs(a.arrival - f.arrival) for a, f in pairs) / len(pairs)
            if pairs
            else math.nan
        ),
        "departure_mae_slots": (
            sum(abs(a.departure - f.departure) for a, f in pairs)
            / len(pairs)
            if pairs
            else math.nan
        ),
        "energy_mae_kwh": (
            sum(abs(a.energy_kwh - f.energy_kwh) for a, f in pairs)
            / len(pairs)
            if pairs
            else math.nan
        ),
    }
    if calibration is not None:
        metrics.update(
            {
                "arrival_upper_coverage": (
                    sum(
                        a.arrival
                        <= f.arrival + calibration.arrival_late_slots
                        for a, f in pairs
                    )
                    / len(pairs)
                    if pairs
                    else math.nan
                ),
                "departure_lower_coverage": (
                    sum(
                        a.departure
                        >= f.departure - calibration.departure_early_slots
                        for a, f in pairs
                    )
                    / len(pairs)
                    if pairs
                    else math.nan
                ),
                "energy_upper_coverage": (
                    sum(
                        a.energy_kwh
                        <= f.energy_kwh + calibration.energy_under_kwh
                        for a, f in pairs
                    )
                    / len(pairs)
                    if pairs
                    else math.nan
                ),
            }
        )
    return metrics
