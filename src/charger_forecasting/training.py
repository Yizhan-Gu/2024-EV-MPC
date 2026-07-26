"""Normalization, masked objectives, and metrics for forecast panels."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import torch


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def calendar_features(days: Sequence[str]) -> np.ndarray:
    output = np.zeros((len(days), 4), dtype=np.float32)
    for idx, value in enumerate(days):
        current = date.fromisoformat(value)
        weekday_angle = 2.0 * math.pi * current.weekday() / 7.0
        year_angle = 2.0 * math.pi * current.timetuple().tm_yday / 365.25
        output[idx] = (
            math.sin(weekday_angle),
            math.cos(weekday_angle),
            math.sin(year_angle),
            math.cos(year_angle),
        )
    return output


@dataclass(frozen=True)
class PanelScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "PanelScaler":
        if train_values.ndim != 3 or train_values.shape[2] != 5:
            raise ValueError("expected [day, entity, 5] training values")
        mean = np.zeros(5, dtype=np.float32)
        scale = np.ones(5, dtype=np.float32)
        active = train_values[:, :, 0] > 0.0
        for feature_idx in range(5):
            sample = (
                train_values[:, :, feature_idx][active]
                if feature_idx >= 2
                else train_values[:, :, feature_idx].reshape(-1)
            )
            if sample.size:
                mean[feature_idx] = float(sample.mean())
                std = float(sample.std())
                scale[feature_idx] = std if std > 1e-6 else 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (values - self.mean) / self.scale
        inactive = values[..., 0] <= 0.0
        transformed[..., 2:] = np.where(
            inactive[..., None],
            0.0,
            transformed[..., 2:],
        )
        return transformed.astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        output = values * self.scale + self.mean
        output[..., 0:2] = np.maximum(0.0, output[..., 0:2])
        output[..., 2:4] = np.clip(output[..., 2:4], 1.0, 96.0)
        output[..., 4] = np.maximum(0.0, output[..., 4])
        inactive = output[..., 0] < 0.5
        output[..., 2:] = np.where(
            inactive[..., None],
            0.0,
            output[..., 2:],
        )
        return output.astype(np.float32)


def target_mask(
    raw_target: np.ndarray,
    *,
    active_weight: float = 3.0,
) -> np.ndarray:
    if active_weight < 1.0:
        raise ValueError("active_weight must be at least one")
    mask = np.ones_like(raw_target, dtype=np.float32)
    active = raw_target[..., 0] > 0.0
    mask[..., 0:2] = np.where(
        active[..., None],
        active_weight,
        1.0,
    )
    mask[..., 2:] = active[..., None]
    return mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weighted = (prediction - target).square() * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def regression_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    target_dates: Sequence[str],
) -> dict[str, float]:
    if (
        actual.shape != prediction.shape
        or actual.ndim != 3
        or actual.shape[-1] != 5
    ):
        raise ValueError("actual/prediction shape mismatch")
    if actual.shape[0] != len(target_dates):
        raise ValueError("one target date is required per panel day")
    active = actual[..., 0] > 0.0
    predicted_active = prediction[..., 0] >= 0.5
    count_error = np.abs(actual[..., 0] - prediction[..., 0])
    energy_error = np.abs(actual[..., 1] - prediction[..., 1])
    metrics = {
        "count_mae": float(count_error.mean()),
        "energy_mae_kwh": float(energy_error.mean()),
        "energy_wape": float(
            energy_error.sum() / max(1e-9, actual[..., 1].sum())
        ),
        "actual_active_rate": float(active.mean()),
        "predicted_active_rate": float(predicted_active.mean()),
    }
    true_positive = float(np.logical_and(active, predicted_active).sum())
    false_positive = float(
        np.logical_and(~active, predicted_active).sum()
    )
    false_negative = float(
        np.logical_and(active, ~predicted_active).sum()
    )
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    metrics["participation_precision"] = precision
    metrics["participation_recall"] = recall
    metrics["participation_f1"] = (
        2.0 * precision * recall / max(1e-9, precision + recall)
    )
    for name, feature_idx in (
        ("arrival_mae_slots", 2),
        ("departure_mae_slots", 3),
        ("dwell_mae_slots", 4),
    ):
        metrics[name] = (
            float(
                np.abs(
                    actual[..., feature_idx][active]
                    - prediction[..., feature_idx][active]
                ).mean()
            )
            if np.any(active)
            else math.nan
        )

    daily_actual = actual[..., 1].sum(axis=1)
    daily_prediction = prediction[..., 1].sum(axis=1)
    daily_error = np.abs(daily_actual - daily_prediction)
    metrics["aggregate_daily_energy_mae_kwh"] = float(daily_error.mean())
    metrics["aggregate_daily_energy_wape"] = float(
        daily_error.sum() / max(1e-9, daily_actual.sum())
    )
    metrics["scope_test_energy_kwh"] = float(daily_actual.sum())
    return metrics
