"""Fast deterministic checks for the feasibility-preserving charger model."""

from __future__ import annotations

import unittest

import numpy as np

from charger_mpc import (
    Session,
    Tariff,
    disaggregate_nonoverlap,
    evaluate_incremental_cost,
    rolling_charger_mpc,
    solve_charger_envelope,
    solve_charger_total_energy_relaxation,
    solve_ev_dispatch,
)


def test_tariff(prices: tuple[float, ...]) -> Tariff:
    return Tariff(
        energy_price_per_kwh=prices,
        demand_charge_all_per_kw=0.3,
        demand_charge_onpeak_per_kw=0.2,
        onpeak_slots=frozenset({3, 4}),
    )


class ChargerEnvelopeTests(unittest.TestCase):
    def test_incremental_cost_charges_only_new_monthly_peak(self) -> None:
        tariff = Tariff(
            energy_price_per_kwh=(0.0, 0.0, 0.0, 0.0),
            demand_charge_all_per_kw=2.0,
            demand_charge_onpeak_per_kw=3.0,
            onpeak_slots=frozenset({3, 4}),
        )

        below_cost, _, below_peak, below_onpeak = evaluate_incremental_cost(
            (3.0, 3.0, 1.0, 1.0),
            tariff,
            delta_t=1.0,
            prior_peak_kw=4.0,
            prior_onpeak_peak_kw=2.0,
        )
        above_cost, _, above_peak, above_onpeak = evaluate_incremental_cost(
            (5.0, 1.0, 4.0, 1.0),
            tariff,
            delta_t=1.0,
            prior_peak_kw=4.0,
            prior_onpeak_peak_kw=2.0,
        )

        self.assertAlmostEqual(below_cost, 0.0)
        self.assertAlmostEqual(below_peak, 4.0)
        self.assertAlmostEqual(below_onpeak, 2.0)
        self.assertAlmostEqual(above_cost, 8.0)
        self.assertAlmostEqual(above_peak, 5.0)
        self.assertAlmostEqual(above_onpeak, 4.0)

    def test_sequential_incremental_costs_equal_one_month_bill(self) -> None:
        tariff = Tariff(
            energy_price_per_kwh=(1.0, 1.0, 1.0, 1.0),
            demand_charge_all_per_kw=2.0,
            demand_charge_onpeak_per_kw=3.0,
            onpeak_slots=frozenset({3, 4}),
        )

        day_one_cost, _, peak, onpeak_peak = evaluate_incremental_cost(
            (4.0, 0.0, 2.0, 0.0),
            tariff,
            delta_t=1.0,
        )
        day_two_cost, _, peak, onpeak_peak = evaluate_incremental_cost(
            (1.0, 0.0, 3.0, 0.0),
            tariff,
            delta_t=1.0,
            prior_peak_kw=peak,
            prior_onpeak_peak_kw=onpeak_peak,
        )

        expected_month_bill = 10.0 + 2.0 * 4.0 + 3.0 * 3.0
        self.assertAlmostEqual(day_one_cost + day_two_cost, expected_month_bill)
        self.assertAlmostEqual(peak, 4.0)
        self.assertAlmostEqual(onpeak_peak, 3.0)

    def test_exact_charger_projection_matches_ev_optimum(self) -> None:
        sessions = [
            Session("a1", "A", 1, 2, 1.5, max_power_kw=1.0),
            Session("a2", "A", 3, 4, 1.0, max_power_kw=1.0),
            Session("b1", "B", 1, 4, 2.0, max_power_kw=1.0),
        ]
        tariff = test_tariff((0.30, 0.10, 0.25, 0.08))

        ev = solve_ev_dispatch(
            sessions,
            tariff,
            delta_t=1.0,
            time_limit=2.0,
        )
        charger = solve_charger_envelope(
            sessions,
            tariff,
            delta_t=1.0,
            time_limit=2.0,
        )
        disaggregated = disaggregate_nonoverlap(
            sessions,
            charger.power_by_unit_kw,
            n_slots=4,
            delta_t=1.0,
        )

        self.assertAlmostEqual(ev.objective, charger.objective, places=7)
        self.assertAlmostEqual(ev.energy_kwh, 4.5, places=7)
        self.assertAlmostEqual(charger.energy_kwh, 4.5, places=7)
        np.testing.assert_allclose(
            ev.load_kw,
            charger.load_kw,
            atol=1e-6,
        )
        for session in sessions:
            delivered = disaggregated[session.session_id].sum()
            self.assertAlmostEqual(
                delivered,
                session.energy_kwh,
                places=7,
            )

    def test_daily_energy_relaxation_can_violate_session_deadline(self) -> None:
        sessions = [
            Session("early", "A", 1, 1, 1.0, max_power_kw=2.0),
            Session("late", "A", 3, 3, 1.0, max_power_kw=2.0),
        ]
        tariff = Tariff(
            energy_price_per_kwh=(1.0, 0.0, 0.1),
            demand_charge_all_per_kw=0.0,
            demand_charge_onpeak_per_kw=0.0,
            onpeak_slots=frozenset(),
        )

        exact = solve_charger_envelope(
            sessions,
            tariff,
            delta_t=1.0,
        )
        relaxed = solve_charger_total_energy_relaxation(
            sessions,
            tariff,
            delta_t=1.0,
        )

        self.assertAlmostEqual(exact.power_by_unit_kw["A"][0], 1.0)
        self.assertAlmostEqual(relaxed.power_by_unit_kw["A"][0], 0.0)
        self.assertLess(relaxed.objective, exact.objective)
        with self.assertRaises(ValueError):
            disaggregate_nonoverlap(
                sessions,
                relaxed.power_by_unit_kw,
                n_slots=3,
                delta_t=1.0,
            )

    def test_overlapping_sessions_are_rejected(self) -> None:
        sessions = [
            Session("one", "A", 1, 2, 1.0),
            Session("two", "A", 2, 3, 1.0),
        ]
        tariff = test_tariff((0.1, 0.1, 0.1, 0.1))
        with self.assertRaisesRegex(ValueError, "overlapping"):
            solve_charger_envelope(sessions, tariff, delta_t=1.0)

    def test_rolling_execution_serves_actual_energy_without_postprocessing(
        self,
    ) -> None:
        actual = [
            Session("a1", "A", 1, 2, 1.0, max_power_kw=1.0),
            Session("a2", "A", 3, 4, 1.0, max_power_kw=1.0),
            Session("b1", "B", 2, 4, 1.5, max_power_kw=1.0),
        ]
        forecast = [
            Session(
                f"forecast:{idx}",
                session.charger_id,
                session.arrival,
                session.departure,
                session.energy_kwh,
                max_power_kw=session.max_power_kw,
                source="forecast",
            )
            for idx, session in enumerate(actual)
        ]
        tariff = test_tariff((0.30, 0.10, 0.25, 0.08))

        perfect = rolling_charger_mpc(
            actual,
            forecast,
            tariff,
            method="Perfect",
            delta_t=1.0,
        )
        noforecast = rolling_charger_mpc(
            actual,
            [],
            tariff,
            method="NoForecast",
            delta_t=1.0,
        )

        self.assertAlmostEqual(perfect.required_energy_kwh, 3.5)
        self.assertAlmostEqual(perfect.energy_kwh, 3.5)
        self.assertAlmostEqual(perfect.unserved_energy_kwh, 0.0)
        self.assertAlmostEqual(noforecast.energy_kwh, 3.5)
        self.assertAlmostEqual(noforecast.unserved_energy_kwh, 0.0)
        self.assertEqual(perfect.fallback_count, 0)
        self.assertEqual(noforecast.fallback_count, 0)

    def test_rolling_carries_month_to_date_peak_without_recharging_it(
        self,
    ) -> None:
        sessions = [
            Session("a", "A", 1, 2, 1.0, max_power_kw=1.0),
        ]
        tariff = Tariff(
            energy_price_per_kwh=(0.0, 0.0),
            demand_charge_all_per_kw=2.0,
            demand_charge_onpeak_per_kw=3.0,
            onpeak_slots=frozenset({1, 2}),
        )

        result = rolling_charger_mpc(
            sessions,
            [],
            tariff,
            method="NoForecast",
            delta_t=1.0,
            prior_peak_kw=4.0,
            prior_onpeak_peak_kw=3.0,
        )

        self.assertAlmostEqual(result.energy_kwh, 1.0)
        self.assertAlmostEqual(result.unserved_energy_kwh, 0.0)
        self.assertAlmostEqual(result.cost, 0.0)
        self.assertAlmostEqual(result.peak_kw, 4.0)
        self.assertAlmostEqual(result.onpeak_peak_kw, 3.0)


if __name__ == "__main__":
    unittest.main()
