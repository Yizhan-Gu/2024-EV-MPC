import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from charger_forecasting import (
    DLinearRegressor,
    GraphTemporalRegressor,
    ITransformerRegressor,
    PanelScaler,
    build_daily_panel,
    correlation_adjacency,
)
from charger_forecasting.training import masked_mse, target_mask


class AdvancedForecastingTests(unittest.TestCase):
    def test_ev_entity_selection_uses_training_period_only(self) -> None:
        fields = [
            "driver_id",
            "session_start_time_la",
            "session_end_time_la",
            "total_energy_dispensed",
            "station_name",
            "port",
        ]
        rows = [
            {
                "driver_id": "training-driver",
                "session_start_time_la": "2023-01-01T08:00:00",
                "session_end_time_la": "2023-01-01T10:00:00",
                "total_energy_dispensed": "5.0",
                "station_name": "Station A",
                "port": "1",
            },
            {
                "driver_id": "test-only-driver",
                "session_start_time_la": "2023-01-03T09:00:00",
                "session_end_time_la": "2023-01-03T11:00:00",
                "total_energy_dispensed": "7.0",
                "station_name": "Station A",
                "port": "1",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            panel = build_daily_panel(
                path,
                level="ev",
                start="2023-01-01",
                end="2023-01-03",
                selection_end="2023-01-02",
                top_k=10,
            )
        self.assertEqual(panel.entity_ids, ("training-driver",))
        self.assertEqual(panel.values[0, 0, 0], 1.0)
        self.assertAlmostEqual(panel.values[0, 0, 1], 5.0)
        self.assertEqual(panel.values[2, 0, 0], 0.0)

    def test_itransformer_and_graph_shapes(self) -> None:
        dlinear = DLinearRegressor(
            context_length=14,
            n_features=5,
        )
        dlinear_output = dlinear(
            torch.zeros(3, 14, 5),
            torch.zeros(3, 4),
        )
        self.assertEqual(tuple(dlinear_output.shape), (3, 5))

        entity_model = ITransformerRegressor(
            context_length=14,
            n_features=5,
            d_model=16,
            n_heads=4,
        )
        entity_output = entity_model(
            torch.zeros(3, 14, 5),
            torch.zeros(3, 4),
        )
        self.assertEqual(tuple(entity_output.shape), (3, 5))

        adjacency = np.eye(4, dtype=np.float32)
        graph_model = GraphTemporalRegressor(
            n_features=5,
            adjacency=adjacency,
            hidden_size=8,
        )
        graph_output = graph_model(
            torch.zeros(2, 14, 4, 5),
            torch.zeros(2, 4),
        )
        self.assertEqual(tuple(graph_output.shape), (2, 4, 5))

    def test_training_only_adjacency_is_symmetric_and_finite(self) -> None:
        values = np.zeros((20, 4, 5), dtype=np.float32)
        base = np.arange(20, dtype=np.float32)
        values[:, 0, 1] = base
        values[:, 1, 1] = base * 2
        values[:, 2, 1] = base[::-1]
        adjacency = correlation_adjacency(
            values,
            feature_index=1,
            top_k=2,
        )
        self.assertTrue(np.all(np.isfinite(adjacency)))
        np.testing.assert_allclose(adjacency, adjacency.T, atol=1e-7)
        self.assertTrue(np.all(np.diag(adjacency) > 0.0))

    def test_scaling_and_mask_ignore_inactive_timing_targets(self) -> None:
        values = np.zeros((3, 2, 5), dtype=np.float32)
        values[0, 0] = [1.0, 8.0, 33.0, 45.0, 12.0]
        values[1, 0] = [1.0, 10.0, 35.0, 47.0, 12.0]
        scaler = PanelScaler.fit(values)
        restored = scaler.inverse_transform(scaler.transform(values))
        np.testing.assert_allclose(restored, values, atol=1e-5)

        mask = target_mask(values)
        self.assertEqual(mask[2, 1, 2], 0.0)
        prediction = torch.ones(3, 2, 5)
        target = torch.zeros(3, 2, 5)
        loss = masked_mse(
            prediction,
            target,
            torch.from_numpy(mask),
        )
        self.assertGreater(float(loss), 0.0)

    def test_overnight_session_is_clipped_not_dropped(self) -> None:
        fields = [
            "driver_id",
            "session_start_time_la",
            "session_end_time_la",
            "total_energy_dispensed",
            "station_name",
            "port",
        ]
        row = {
            "driver_id": "overnight-driver",
            "session_start_time_la": "2023-01-01T23:00:00",
            "session_end_time_la": "2023-01-02T02:00:00",
            "total_energy_dispensed": "9.0",
            "station_name": "Station A",
            "port": "1",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            panel = build_daily_panel(
                path,
                level="charger",
                start="2023-01-01",
                end="2023-01-01",
                selection_end="2023-01-01",
            )
        self.assertEqual(panel.values[0, 0, 0], 1.0)
        self.assertAlmostEqual(panel.values[0, 0, 1], 9.0)
        self.assertAlmostEqual(panel.values[0, 0, 2], 93.0)
        self.assertAlmostEqual(panel.values[0, 0, 3], 96.0)
        self.assertAlmostEqual(panel.values[0, 0, 4], 3.0)

    def test_matched_charger_scope_contains_same_driver_energy(self) -> None:
        fields = [
            "driver_id",
            "session_start_time_la",
            "session_end_time_la",
            "total_energy_dispensed",
            "station_name",
            "port",
        ]
        rows = [
            {
                "driver_id": "included",
                "session_start_time_la": "2023-01-01T08:00:00",
                "session_end_time_la": "2023-01-01T10:00:00",
                "total_energy_dispensed": "5.0",
                "station_name": "Station A",
                "port": "1",
            },
            {
                "driver_id": "excluded",
                "session_start_time_la": "2023-01-01T12:00:00",
                "session_end_time_la": "2023-01-01T14:00:00",
                "total_energy_dispensed": "9.0",
                "station_name": "Station A",
                "port": "1",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            ev_panel = build_daily_panel(
                path,
                level="ev",
                start="2023-01-01",
                end="2023-01-01",
                selection_end="2023-01-01",
                entity_ids=("included",),
            )
            charger_panel = build_daily_panel(
                path,
                level="charger",
                start="2023-01-01",
                end="2023-01-01",
                selection_end="2023-01-01",
                entity_ids=("Station A|1",),
                driver_filter=ev_panel.entity_ids,
            )
        self.assertAlmostEqual(ev_panel.values[..., 1].sum(), 5.0)
        self.assertAlmostEqual(charger_panel.values[..., 1].sum(), 5.0)


if __name__ == "__main__":
    unittest.main()
