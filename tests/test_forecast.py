"""Fast checks for causal median and conformal session forecasts."""

from __future__ import annotations

import math
import unittest

from charger_mpc import (
    Session,
    calibrate_one_sided_conformal,
    conformal_robust_forecast,
    historical_median_forecast,
    session_forecast_metrics,
)


class ForecastTests(unittest.TestCase):
    def test_historical_median_uses_only_previous_same_weekdays(self) -> None:
        charger = "A"
        history_days = (
            "2023-06-24",
            "2023-06-17",
            "2023-06-10",
            "2023-06-03",
        )
        sessions_by_day = {
            day: [
                Session(
                    f"{day}:a",
                    charger,
                    arrival,
                    departure,
                    energy,
                    max_power_kw=10.0,
                )
            ]
            for day, arrival, departure, energy in zip(
                history_days,
                (1, 2, 3, 4),
                (5, 6, 7, 8),
                (0.5, 1.0, 1.0, 1.5),
            )
        }

        forecast = historical_median_forecast(
            sessions_by_day,
            "2023-07-01",
            [charger],
            lookback_weeks=4,
        )

        self.assertEqual(len(forecast), 1)
        self.assertEqual(forecast[0].arrival, 3)
        self.assertEqual(forecast[0].departure, 7)
        self.assertAlmostEqual(forecast[0].energy_kwh, 1.0)

    def test_one_sided_conformal_tightens_forecast_flexibility(self) -> None:
        charger = "A"
        sessions_by_day = {
            "2023-06-24": [
                Session("h1", charger, 1, 5, 0.5, max_power_kw=10.0)
            ],
            "2023-06-17": [
                Session("h2", charger, 2, 6, 1.0, max_power_kw=10.0)
            ],
            "2023-06-10": [
                Session("h3", charger, 3, 7, 1.0, max_power_kw=10.0)
            ],
            "2023-06-03": [
                Session("h4", charger, 4, 8, 1.5, max_power_kw=10.0)
            ],
            "2023-07-01": [
                Session("actual", charger, 5, 5, 1.5, max_power_kw=10.0)
            ],
        }
        point = historical_median_forecast(
            sessions_by_day,
            "2023-07-01",
            [charger],
        )
        calibration = calibrate_one_sided_conformal(
            sessions_by_day,
            ["2023-07-01"],
            [charger],
            alpha=0.1,
        )
        robust = conformal_robust_forecast(point, calibration)
        metrics = session_forecast_metrics(
            sessions_by_day["2023-07-01"],
            point,
            [charger],
            calibration=calibration,
        )

        self.assertEqual(calibration.arrival_late_slots, 2)
        self.assertEqual(calibration.departure_early_slots, 2)
        self.assertAlmostEqual(calibration.energy_under_kwh, 0.5)
        self.assertEqual(robust[0].arrival, 5)
        self.assertEqual(robust[0].departure, 5)
        self.assertAlmostEqual(robust[0].energy_kwh, 1.5)
        self.assertAlmostEqual(metrics["arrival_upper_coverage"], 1.0)
        self.assertAlmostEqual(metrics["departure_lower_coverage"], 1.0)
        self.assertAlmostEqual(metrics["energy_upper_coverage"], 1.0)

    def test_zero_matches_report_nan_coverage(self) -> None:
        calibration = calibrate_one_sided_conformal(
            {
                "2023-06-24": [
                    Session("h1", "A", 1, 2, 1.0)
                ],
                "2023-06-17": [
                    Session("h2", "A", 1, 2, 1.0)
                ],
                "2023-06-10": [
                    Session("h3", "A", 1, 2, 1.0)
                ],
                "2023-06-03": [
                    Session("h4", "A", 1, 2, 1.0)
                ],
                "2023-07-01": [
                    Session("actual", "A", 1, 2, 1.0)
                ],
            },
            ["2023-07-01"],
            ["A"],
        )
        metrics = session_forecast_metrics(
            [],
            [],
            ["A"],
            calibration=calibration,
        )

        self.assertTrue(math.isnan(metrics["arrival_upper_coverage"]))
        self.assertTrue(math.isnan(metrics["departure_lower_coverage"]))
        self.assertTrue(math.isnan(metrics["energy_upper_coverage"]))


if __name__ == "__main__":
    unittest.main()
