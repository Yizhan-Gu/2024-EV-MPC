"""Small, auditable optimization core for charger-based V1G experiments.

The module deliberately contains no machine-learning code.  It establishes the
deterministic reference problem that future forecasts must feed without changing
the controller, tariff, or service requirements.

Slots are one-indexed in :class:`Session` to match the original Julia code.
NumPy arrays remain zero-indexed internally.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import lil_matrix


_TOL = 1e-7


@dataclass(frozen=True)
class Session:
    """One unidirectional EV charging request."""

    session_id: str
    charger_id: str
    arrival: int
    departure: int
    energy_kwh: float
    max_power_kw: float = 6.6
    source: str = "actual"

    def validate(self, n_slots: int, delta_t: float) -> None:
        if not 1 <= self.arrival <= self.departure <= n_slots:
            raise ValueError(
                f"{self.session_id}: invalid slots "
                f"[{self.arrival}, {self.departure}] for horizon {n_slots}"
            )
        if self.energy_kwh < -_TOL:
            raise ValueError(f"{self.session_id}: negative energy")
        capacity = (
            self.max_power_kw
            * delta_t
            * (self.departure - self.arrival + 1)
        )
        if self.energy_kwh > capacity + _TOL:
            raise ValueError(
                f"{self.session_id}: {self.energy_kwh:.6f} kWh exceeds "
                f"{capacity:.6f} kWh availability capacity"
            )


@dataclass(frozen=True)
class Tariff:
    """Linear energy plus noncoincident/on-peak demand-charge proxy."""

    energy_price_per_kwh: tuple[float, ...]
    demand_charge_all_per_kw: float
    demand_charge_onpeak_per_kw: float
    onpeak_slots: frozenset[int]
    other_fraction: float = 0.0
    other_per_kwh: float = 0.0

    @property
    def n_slots(self) -> int:
        return len(self.energy_price_per_kwh)


@dataclass(frozen=True)
class DispatchResult:
    """Optimal full-horizon dispatch and its physical/economic summary."""

    objective: float
    load_kw: np.ndarray
    power_by_unit_kw: Mapping[str, np.ndarray]
    energy_kwh: float
    peak_kw: float
    onpeak_peak_kw: float
    solver_message: str


@dataclass(frozen=True)
class RollingResult:
    """Actually executed rolling-horizon charger dispatch."""

    method: str
    load_kw: np.ndarray
    cost: float
    energy_kwh: float
    required_energy_kwh: float
    unserved_energy_kwh: float
    peak_kw: float
    onpeak_peak_kw: float
    optimal_solve_ratio: float
    solve_count: int
    fallback_count: int
    dropped_forecast_sessions: int


def summer_tariff(n_slots: int = 96, delta_t: float = 0.25) -> Tariff:
    """Return the tariff coefficients used by the original Julia experiment.

    Demand rates are monthly billing rates.  Applying them to one isolated day
    is therefore only a smoke-test proxy; publication experiments must carry
    the executed monthly peak state across days.
    """

    on_start = int(16 / delta_t) + 1
    on_end = int(21 / delta_t)
    onpeak = frozenset(range(on_start, on_end + 1))
    on_rate = 0.11957 + 0.00671
    off_rate = 0.10008 + 0.00671
    prices = tuple(
        on_rate if slot in onpeak else off_rate
        for slot in range(1, n_slots + 1)
    )
    return Tariff(
        energy_price_per_kwh=prices,
        demand_charge_all_per_kw=24.48,
        demand_charge_onpeak_per_kw=9.78 + 19.14,
        onpeak_slots=onpeak,
        other_fraction=0.0578,
        other_per_kwh=0.0058 + 0.00058 + 0.0003,
    )


def _validate_inputs(
    sessions: Sequence[Session],
    tariff: Tariff,
    delta_t: float,
) -> None:
    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    for session in sessions:
        session.validate(tariff.n_slots, delta_t)


def _assert_unique_ids(sessions: Sequence[Session]) -> None:
    identifiers = [session.session_id for session in sessions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("session_id values must be unique")


def _group_by_charger(
    sessions: Sequence[Session],
) -> dict[str, list[Session]]:
    grouped: dict[str, list[Session]] = {}
    for session in sessions:
        grouped.setdefault(session.charger_id, []).append(session)
    for charger_sessions in grouped.values():
        charger_sessions.sort(key=lambda x: (x.arrival, x.departure, x.session_id))
    return grouped


def _assert_nonoverlap(grouped: Mapping[str, Sequence[Session]]) -> None:
    for charger_id, charger_sessions in grouped.items():
        for previous, current in zip(charger_sessions, charger_sessions[1:]):
            if current.arrival <= previous.departure:
                raise ValueError(
                    f"charger {charger_id} has overlapping sessions "
                    f"{previous.session_id} and {current.session_id}"
                )


def _objective_vector(
    n_units: int,
    tariff: Tariff,
    delta_t: float,
) -> np.ndarray:
    n_slots = tariff.n_slots
    n_power = n_units * n_slots
    objective = np.zeros(n_power + 2, dtype=float)
    for unit_idx in range(n_units):
        offset = unit_idx * n_slots
        for slot_idx, price in enumerate(tariff.energy_price_per_kwh):
            objective[offset + slot_idx] = (
                (1.0 + tariff.other_fraction) * price * delta_t
                + tariff.other_per_kwh * delta_t
            )
    objective[n_power] = (
        (1.0 + tariff.other_fraction)
        * tariff.demand_charge_all_per_kw
    )
    objective[n_power + 1] = (
        (1.0 + tariff.other_fraction)
        * tariff.demand_charge_onpeak_per_kw
    )
    return objective


def _load_peak_constraints(
    n_units: int,
    tariff: Tariff,
) -> tuple[lil_matrix, np.ndarray]:
    n_slots = tariff.n_slots
    n_power = n_units * n_slots
    n_rows = n_slots + len(tariff.onpeak_slots)
    matrix = lil_matrix((n_rows, n_power + 2), dtype=float)
    rhs = np.zeros(n_rows, dtype=float)

    row = 0
    for slot_idx in range(n_slots):
        for unit_idx in range(n_units):
            matrix[row, unit_idx * n_slots + slot_idx] = 1.0
        matrix[row, n_power] = -1.0
        row += 1

    for slot in sorted(tariff.onpeak_slots):
        slot_idx = slot - 1
        for unit_idx in range(n_units):
            matrix[row, unit_idx * n_slots + slot_idx] = 1.0
        matrix[row, n_power + 1] = -1.0
        row += 1

    return matrix, rhs


def _bounds(
    power_bounds: Sequence[Sequence[tuple[float, float]]],
    prior_peak_kw: float,
    prior_onpeak_peak_kw: float,
) -> list[tuple[float, float | None]]:
    flat: list[tuple[float, float | None]] = [
        bound for unit in power_bounds for bound in unit
    ]
    flat.append((max(0.0, prior_peak_kw), None))
    flat.append((max(0.0, prior_onpeak_peak_kw), None))
    return flat


def _result_from_solution(
    solution: np.ndarray,
    unit_ids: Sequence[str],
    tariff: Tariff,
    delta_t: float,
    objective: float,
    message: str,
) -> DispatchResult:
    n_slots = tariff.n_slots
    power = solution[: len(unit_ids) * n_slots].reshape(len(unit_ids), n_slots)
    load = power.sum(axis=0)
    onpeak_indices = [slot - 1 for slot in tariff.onpeak_slots]
    onpeak_peak = (
        float(np.max(load[onpeak_indices])) if onpeak_indices else 0.0
    )
    return DispatchResult(
        objective=float(objective),
        load_kw=load,
        power_by_unit_kw={
            unit_id: power[idx].copy()
            for idx, unit_id in enumerate(unit_ids)
        },
        energy_kwh=float(load.sum() * delta_t),
        peak_kw=float(np.max(load)) if len(load) else 0.0,
        onpeak_peak_kw=onpeak_peak,
        solver_message=message,
    )


def _solve(
    objective: np.ndarray,
    bounds: Sequence[tuple[float, float | None]],
    a_ub: lil_matrix,
    b_ub: np.ndarray,
    a_eq: lil_matrix | None,
    b_eq: np.ndarray | None,
    time_limit: float,
) -> OptimizeResult:
    result = linprog(
        objective,
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=None if a_eq is None else a_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"time_limit": max(0.1, float(time_limit))},
    )
    if not result.success:
        raise RuntimeError(f"HiGHS failed: {result.message}")
    return result


def solve_ev_dispatch(
    sessions: Iterable[Session],
    tariff: Tariff,
    *,
    delta_t: float = 0.25,
    prior_peak_kw: float = 0.0,
    prior_onpeak_peak_kw: float = 0.0,
    time_limit: float = 5.0,
) -> DispatchResult:
    """Solve the exact session-level deterministic charging LP."""

    session_list = list(sessions)
    if not session_list:
        return _empty_dispatch(tariff)
    _validate_inputs(session_list, tariff, delta_t)
    _assert_unique_ids(session_list)

    n_slots = tariff.n_slots
    n_sessions = len(session_list)
    n_power = n_sessions * n_slots
    objective = _objective_vector(n_sessions, tariff, delta_t)
    a_ub, b_ub = _load_peak_constraints(n_sessions, tariff)

    a_eq = lil_matrix((n_sessions, n_power + 2), dtype=float)
    b_eq = np.zeros(n_sessions, dtype=float)
    power_bounds: list[list[tuple[float, float]]] = []
    for session_idx, session in enumerate(session_list):
        unit_bounds: list[tuple[float, float]] = []
        for slot in range(1, n_slots + 1):
            available = session.arrival <= slot <= session.departure
            unit_bounds.append(
                (0.0, session.max_power_kw if available else 0.0)
            )
            if available:
                a_eq[session_idx, session_idx * n_slots + slot - 1] = delta_t
        b_eq[session_idx] = session.energy_kwh
        power_bounds.append(unit_bounds)

    result = _solve(
        objective,
        _bounds(
            power_bounds,
            prior_peak_kw,
            prior_onpeak_peak_kw,
        ),
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        time_limit,
    )
    return _result_from_solution(
        result.x,
        [session.session_id for session in session_list],
        tariff,
        delta_t,
        result.fun,
        result.message,
    )


def _charger_envelopes(
    charger_sessions: Sequence[Session],
    n_slots: int,
    delta_t: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    lower = np.zeros(n_slots, dtype=float)
    upper = np.zeros(n_slots, dtype=float)
    slot_bounds: list[tuple[float, float]] = []

    for slot in range(1, n_slots + 1):
        active = [
            session
            for session in charger_sessions
            if session.arrival <= slot <= session.departure
        ]
        slot_bounds.append(
            (0.0, active[0].max_power_kw if active else 0.0)
        )

        for session in charger_sessions:
            elapsed_slots = max(
                0,
                min(slot, session.departure) - session.arrival + 1,
            )
            future_slots = max(
                0,
                session.departure - max(slot + 1, session.arrival) + 1,
            )
            upper[slot - 1] += min(
                session.energy_kwh,
                session.max_power_kw * delta_t * elapsed_slots,
            )
            lower[slot - 1] += max(
                0.0,
                session.energy_kwh
                - session.max_power_kw * delta_t * future_slots,
            )

    return lower, upper, slot_bounds


def solve_charger_envelope(
    sessions: Iterable[Session],
    tariff: Tariff,
    *,
    delta_t: float = 0.25,
    prior_peak_kw: float = 0.0,
    prior_onpeak_peak_kw: float = 0.0,
    time_limit: float = 5.0,
) -> DispatchResult:
    """Solve the exact charger projection for nonoverlapping sessions.

    Every charger has cumulative lower/upper energy envelopes.  For physically
    nonoverlapping sessions on a port, these bounds preserve every individual
    arrival, deadline, and energy requirement.
    """

    session_list = list(sessions)
    if not session_list:
        return _empty_dispatch(tariff)
    _validate_inputs(session_list, tariff, delta_t)
    _assert_unique_ids(session_list)
    grouped = _group_by_charger(session_list)
    _assert_nonoverlap(grouped)

    charger_ids = sorted(grouped)
    n_chargers = len(charger_ids)
    n_slots = tariff.n_slots
    n_power = n_chargers * n_slots
    objective = _objective_vector(n_chargers, tariff, delta_t)
    peak_matrix, peak_rhs = _load_peak_constraints(n_chargers, tariff)

    envelope_rows = 2 * n_chargers * n_slots
    a_ub = lil_matrix(
        (peak_matrix.shape[0] + envelope_rows, n_power + 2),
        dtype=float,
    )
    a_ub[: peak_matrix.shape[0], :] = peak_matrix
    b_ub = np.zeros(a_ub.shape[0], dtype=float)
    b_ub[: len(peak_rhs)] = peak_rhs

    power_bounds: list[list[tuple[float, float]]] = []
    row = peak_matrix.shape[0]
    for charger_idx, charger_id in enumerate(charger_ids):
        lower, upper, charger_bounds = _charger_envelopes(
            grouped[charger_id],
            n_slots,
            delta_t,
        )
        power_bounds.append(charger_bounds)
        offset = charger_idx * n_slots
        for slot_idx in range(n_slots):
            for prior_idx in range(slot_idx + 1):
                a_ub[row, offset + prior_idx] = delta_t
                a_ub[row + 1, offset + prior_idx] = -delta_t
            b_ub[row] = upper[slot_idx]
            b_ub[row + 1] = -lower[slot_idx]
            row += 2

    result = _solve(
        objective,
        _bounds(
            power_bounds,
            prior_peak_kw,
            prior_onpeak_peak_kw,
        ),
        a_ub,
        b_ub,
        None,
        None,
        time_limit,
    )
    return _result_from_solution(
        result.x,
        charger_ids,
        tariff,
        delta_t,
        result.fun,
        result.message,
    )


def solve_charger_total_energy_relaxation(
    sessions: Iterable[Session],
    tariff: Tariff,
    *,
    delta_t: float = 0.25,
    time_limit: float = 5.0,
) -> DispatchResult:
    """Diagnostic legacy relaxation using only charger daily energy.

    This intentionally omits cumulative session envelopes.  It exists only to
    demonstrate why the former charger formulation can report infeasible cost
    savings and must never be used for publication results.
    """

    session_list = list(sessions)
    if not session_list:
        return _empty_dispatch(tariff)
    _validate_inputs(session_list, tariff, delta_t)
    _assert_unique_ids(session_list)
    grouped = _group_by_charger(session_list)
    _assert_nonoverlap(grouped)

    charger_ids = sorted(grouped)
    n_chargers = len(charger_ids)
    n_slots = tariff.n_slots
    n_power = n_chargers * n_slots
    objective = _objective_vector(n_chargers, tariff, delta_t)
    a_ub, b_ub = _load_peak_constraints(n_chargers, tariff)
    a_eq = lil_matrix((n_chargers, n_power + 2), dtype=float)
    b_eq = np.zeros(n_chargers, dtype=float)
    power_bounds: list[list[tuple[float, float]]] = []

    for charger_idx, charger_id in enumerate(charger_ids):
        occupied = set()
        max_power_by_slot: dict[int, float] = {}
        for session in grouped[charger_id]:
            b_eq[charger_idx] += session.energy_kwh
            for slot in range(session.arrival, session.departure + 1):
                occupied.add(slot)
                max_power_by_slot[slot] = session.max_power_kw
        unit_bounds = []
        for slot in range(1, n_slots + 1):
            upper = max_power_by_slot.get(slot, 0.0)
            unit_bounds.append((0.0, upper))
            if slot in occupied:
                a_eq[
                    charger_idx,
                    charger_idx * n_slots + slot - 1,
                ] = delta_t
        power_bounds.append(unit_bounds)

    result = _solve(
        objective,
        _bounds(power_bounds, 0.0, 0.0),
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        time_limit,
    )
    return _result_from_solution(
        result.x,
        charger_ids,
        tariff,
        delta_t,
        result.fun,
        result.message,
    )


def disaggregate_nonoverlap(
    sessions: Iterable[Session],
    charger_power_kw: Mapping[str, np.ndarray],
    *,
    n_slots: int,
    delta_t: float = 0.25,
    tolerance: float = 1e-5,
) -> dict[str, np.ndarray]:
    """Map a charger dispatch back to its unique active session per slot."""

    session_list = list(sessions)
    grouped = _group_by_charger(session_list)
    _assert_nonoverlap(grouped)
    output = {
        session.session_id: np.zeros(n_slots, dtype=float)
        for session in session_list
    }

    for charger_id, power in charger_power_kw.items():
        if len(power) != n_slots:
            raise ValueError(f"{charger_id}: unexpected dispatch length")
        charger_sessions = grouped.get(charger_id, [])
        for slot in range(1, n_slots + 1):
            active = [
                session
                for session in charger_sessions
                if session.arrival <= slot <= session.departure
            ]
            if len(active) > 1:
                raise ValueError(f"{charger_id}: overlapping active sessions")
            if active:
                output[active[0].session_id][slot - 1] = power[slot - 1]
            elif abs(power[slot - 1]) > tolerance:
                raise ValueError(
                    f"{charger_id}: {power[slot - 1]:.6f} kW "
                    f"scheduled while vacant at slot {slot}"
                )

    for session in session_list:
        delivered = output[session.session_id].sum() * delta_t
        if abs(delivered - session.energy_kwh) > tolerance:
            raise ValueError(
                f"{session.session_id}: delivered {delivered:.6f}, "
                f"required {session.energy_kwh:.6f}"
            )
    return output


def evaluate_cost(
    load_kw: Sequence[float],
    tariff: Tariff,
    *,
    delta_t: float = 0.25,
    prior_peak_kw: float = 0.0,
    prior_onpeak_peak_kw: float = 0.0,
) -> tuple[float, float, float, float]:
    """Evaluate actual executed load without post-hoc energy replacement."""

    load = np.asarray(load_kw, dtype=float)
    if len(load) != tariff.n_slots:
        raise ValueError("load length does not match tariff")
    if np.any(load < -_TOL):
        raise ValueError("load cannot be negative")

    peak = max(float(np.max(load)), float(prior_peak_kw))
    onpeak_indices = [slot - 1 for slot in tariff.onpeak_slots]
    onpeak_peak = max(
        float(np.max(load[onpeak_indices])) if onpeak_indices else 0.0,
        float(prior_onpeak_peak_kw),
    )
    energy_kwh = float(load.sum() * delta_t)
    energy_charge = float(
        np.dot(load, np.asarray(tariff.energy_price_per_kwh)) * delta_t
    )
    demand_charge = (
        tariff.demand_charge_all_per_kw * peak
        + tariff.demand_charge_onpeak_per_kw * onpeak_peak
    )
    cost = (
        (1.0 + tariff.other_fraction) * (energy_charge + demand_charge)
        + tariff.other_per_kwh * energy_kwh
    )
    return float(cost), energy_kwh, peak, onpeak_peak


def evaluate_incremental_cost(
    load_kw: Sequence[float],
    tariff: Tariff,
    *,
    delta_t: float = 0.25,
    prior_peak_kw: float = 0.0,
    prior_onpeak_peak_kw: float = 0.0,
) -> tuple[float, float, float, float]:
    """Evaluate one day's incremental contribution to a monthly bill.

    Energy charges are always incremental. Demand charges include only the
    increase above the executed month-to-date peak supplied by the caller.
    The returned peaks are the updated states to carry into the next day.
    """

    prior_peak = max(0.0, float(prior_peak_kw))
    prior_onpeak_peak = max(0.0, float(prior_onpeak_peak_kw))
    total_cost, energy_kwh, peak, onpeak_peak = evaluate_cost(
        load_kw,
        tariff,
        delta_t=delta_t,
        prior_peak_kw=prior_peak,
        prior_onpeak_peak_kw=prior_onpeak_peak,
    )
    prior_demand_cost = (1.0 + tariff.other_fraction) * (
        tariff.demand_charge_all_per_kw * prior_peak
        + tariff.demand_charge_onpeak_per_kw * prior_onpeak_peak
    )
    incremental_cost = total_cost - prior_demand_cost
    return float(incremental_cost), energy_kwh, peak, onpeak_peak


def v0g_dispatch(
    sessions: Iterable[Session],
    tariff: Tariff,
    *,
    delta_t: float = 0.25,
    prior_peak_kw: float = 0.0,
    prior_onpeak_peak_kw: float = 0.0,
) -> RollingResult:
    """Execute immediate charging and return its incremental billing cost."""

    session_list = list(sessions)
    _validate_inputs(session_list, tariff, delta_t)
    grouped = _group_by_charger(session_list)
    _assert_nonoverlap(grouped)
    load = np.zeros(tariff.n_slots, dtype=float)
    unserved = 0.0
    for session in session_list:
        remaining = session.energy_kwh
        for slot in range(session.arrival, session.departure + 1):
            power = min(session.max_power_kw, remaining / delta_t)
            load[slot - 1] += power
            remaining -= power * delta_t
            if remaining <= _TOL:
                remaining = 0.0
                break
        unserved += max(0.0, remaining)
    cost, energy, peak, onpeak_peak = evaluate_incremental_cost(
        load,
        tariff,
        delta_t=delta_t,
        prior_peak_kw=prior_peak_kw,
        prior_onpeak_peak_kw=prior_onpeak_peak_kw,
    )
    required = float(sum(x.energy_kwh for x in session_list))
    return RollingResult(
        method="V0G",
        load_kw=load,
        cost=cost,
        energy_kwh=energy,
        required_energy_kwh=required,
        unserved_energy_kwh=unserved,
        peak_kw=peak,
        onpeak_peak_kw=onpeak_peak,
        optimal_solve_ratio=1.0,
        solve_count=0,
        fallback_count=0,
        dropped_forecast_sessions=0,
    )


def _overlap(left: Session, right: Session) -> bool:
    return (
        left.charger_id == right.charger_id
        and left.arrival <= right.departure
        and right.arrival <= left.departure
    )


def _reconcile_work_set(
    actual_active: Sequence[Session],
    future_forecast: Sequence[Session],
) -> tuple[list[Session], int]:
    """Prefer arrived truth and greedily retain nonoverlapping forecasts."""

    accepted = list(actual_active)
    dropped = 0
    for forecast in sorted(
        future_forecast,
        key=lambda x: (x.arrival, x.departure, x.charger_id, x.session_id),
    ):
        if any(_overlap(forecast, existing) for existing in accepted):
            dropped += 1
            continue
        accepted.append(forecast)
    return accepted, dropped


def rolling_charger_mpc(
    actual_sessions: Iterable[Session],
    forecast_sessions: Iterable[Session],
    tariff: Tariff,
    *,
    method: str,
    delta_t: float = 0.25,
    time_limit_per_solve: float = 1.0,
    prior_peak_kw: float = 0.0,
    prior_onpeak_peak_kw: float = 0.0,
) -> RollingResult:
    """Execute receding-horizon charger MPC against actual arrivals.

    Once a session arrives, its true energy and departure are assumed known.
    Forecast sessions are used only while their arrival remains in the future.
    Planned power is executed only when a real EV is physically connected.
    Prior peaks are executed month-to-date states. ``cost`` reports only this
    day's incremental contribution to the bill, while returned peaks are the
    updated states for the following day.
    """

    actual = list(actual_sessions)
    forecast = list(forecast_sessions)
    _validate_inputs(actual, tariff, delta_t)
    _validate_inputs(forecast, tariff, delta_t)
    _assert_unique_ids(actual)
    _assert_unique_ids(forecast)
    _assert_nonoverlap(_group_by_charger(actual))
    _assert_nonoverlap(_group_by_charger(forecast))

    delivered = {session.session_id: 0.0 for session in actual}
    load = np.zeros(tariff.n_slots, dtype=float)
    initial_peak = max(0.0, float(prior_peak_kw))
    initial_onpeak_peak = max(0.0, float(prior_onpeak_peak_kw))
    prior_peak = initial_peak
    prior_onpeak_peak = initial_onpeak_peak
    solve_count = 0
    optimal_count = 0
    fallback_count = 0
    dropped_total = 0

    for slot in range(1, tariff.n_slots + 1):
        active_actual: list[Session] = []
        active_by_charger: dict[str, Session] = {}
        for session in actual:
            remaining = max(
                0.0,
                session.energy_kwh - delivered[session.session_id],
            )
            if (
                remaining > _TOL
                and session.arrival <= slot <= session.departure
            ):
                rebased = replace(
                    session,
                    arrival=slot,
                    energy_kwh=remaining,
                    source="actual",
                )
                active_actual.append(rebased)
                if rebased.charger_id in active_by_charger:
                    raise ValueError(
                        f"multiple actual sessions active on "
                        f"{rebased.charger_id} at slot {slot}"
                    )
                active_by_charger[rebased.charger_id] = rebased

        future_forecast = [
            session
            for session in forecast
            if session.arrival > slot
        ]
        work, dropped = _reconcile_work_set(
            active_actual,
            future_forecast,
        )
        dropped_total += dropped

        planned_now: dict[str, float] = {}
        if work:
            solve_count += 1
            try:
                plan = solve_charger_envelope(
                    work,
                    tariff,
                    delta_t=delta_t,
                    prior_peak_kw=prior_peak,
                    prior_onpeak_peak_kw=prior_onpeak_peak,
                    time_limit=time_limit_per_solve,
                )
                optimal_count += 1
                planned_now = {
                    charger_id: float(power[slot - 1])
                    for charger_id, power in plan.power_by_unit_kw.items()
                }
            except RuntimeError:
                fallback_count += 1
                planned_now = {
                    charger_id: min(
                        session.max_power_kw,
                        session.energy_kwh / delta_t,
                    )
                    for charger_id, session in active_by_charger.items()
                }

        executed_slot = 0.0
        for charger_id, session in active_by_charger.items():
            power = min(
                session.max_power_kw,
                max(0.0, planned_now.get(charger_id, 0.0)),
                session.energy_kwh / delta_t,
            )
            delivered[session.session_id] += power * delta_t
            executed_slot += power

        load[slot - 1] = executed_slot
        prior_peak = max(prior_peak, executed_slot)
        if slot in tariff.onpeak_slots:
            prior_onpeak_peak = max(prior_onpeak_peak, executed_slot)

    required = float(sum(session.energy_kwh for session in actual))
    served = float(sum(delivered.values()))
    unserved = max(0.0, required - served)
    cost, energy, peak, onpeak_peak = evaluate_incremental_cost(
        load,
        tariff,
        delta_t=delta_t,
        prior_peak_kw=initial_peak,
        prior_onpeak_peak_kw=initial_onpeak_peak,
    )
    return RollingResult(
        method=method,
        load_kw=load,
        cost=cost,
        energy_kwh=energy,
        required_energy_kwh=required,
        unserved_energy_kwh=unserved,
        peak_kw=peak,
        onpeak_peak_kw=onpeak_peak,
        optimal_solve_ratio=(
            optimal_count / solve_count if solve_count else 1.0
        ),
        solve_count=solve_count,
        fallback_count=fallback_count,
        dropped_forecast_sessions=dropped_total,
    )


def _empty_dispatch(tariff: Tariff) -> DispatchResult:
    load = np.zeros(tariff.n_slots, dtype=float)
    return DispatchResult(
        objective=0.0,
        load_kw=load,
        power_by_unit_kw={},
        energy_kwh=0.0,
        peak_kw=0.0,
        onpeak_peak_kw=0.0,
        solver_message="empty problem",
    )
