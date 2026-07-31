#!/usr/bin/env python3
"""Forecast physical charging-flexibility signatures without session matching."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_forecasting import (  # noqa: E402
    DEFAULT_ANCHOR_SLOTS,
    DLinearRegressor,
    EnvelopeScaler,
    FeasibleEnvelopeOutput,
    ITransformerRegressor,
    LSTMRegressor,
    TCNRegressor,
    build_envelope_panel,
    calendar_features,
    envelope_metrics,
    envelope_target_weights,
    envelope_validity_mask,
    masked_mse,
    project_envelope_signatures,
    set_deterministic_seed,
)


FIXED_CHARGERS = (
    "UCSD / GILMAN 2-2|2",
    "UCSD / SCHOLARS - 07|2",
    "UCSD / RADY P357 5|2",
    "UCSD / BIRCH AQUARIUM|1",
    "UCSD / SCHOLARS - 01|1",
    "UCSD / SCHOLARS - 08|2",
)
MODELS = (
    "SeasonalNaive",
    "Ridge",
    "DLinear",
    "LSTM",
    "TCN",
    "iTransformer",
    "FeasibleDLinear",
    "FeasibleTCN",
    "FeasibleITransformer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--panel-start", default="2022-01-01")
    parser.add_argument("--train-end", default="2023-03-31")
    parser.add_argument("--validation-end", default="2023-06-30")
    parser.add_argument(
        "--test-end",
        default="2023-07-31",
        help=(
            "Development runs default to July only. Pass an explicit later "
            "date only for a frozen paper-scale replication."
        ),
    )
    parser.add_argument("--context-length", type=int, default=28)
    parser.add_argument("--ev-top-k", type=int, default=512)
    parser.add_argument("--ev-minimum-sessions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "envelope_forecast_quick"
            / "metrics.csv"
        ),
    )
    return parser.parse_args()


def _indices(
    dates: tuple[str, ...],
    *,
    context: int,
    lower_exclusive: str | None,
    upper_inclusive: str,
) -> list[int]:
    return [
        idx
        for idx, day in enumerate(dates)
        if idx >= context
        and (lower_exclusive is None or day > lower_exclusive)
        and day <= upper_inclusive
    ]


def _window_raw(
    values: np.ndarray,
    indices: list[int],
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [values[idx - context : idx] for idx in indices],
        axis=0,
    )
    y = np.stack([values[idx] for idx in indices], axis=0)
    return x, y


def _entity_arrays(
    scaled_values: np.ndarray,
    raw_values: np.ndarray,
    dates: tuple[str, ...],
    indices: list[int],
    context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _window_raw(scaled_values, indices, context)
    _, raw_y = _window_raw(raw_values, indices, context)
    days, lookback, entities, features = x.shape
    return (
        x.transpose(0, 2, 1, 3)
        .reshape(days * entities, lookback, features)
        .astype(np.float32),
        y.reshape(days * entities, features).astype(np.float32),
        np.repeat(
            calendar_features([dates[idx] for idx in indices]),
            entities,
            axis=0,
        ).astype(np.float32),
        envelope_target_weights(
            raw_y.reshape(days * entities, features),
            anchor_slots=DEFAULT_ANCHOR_SLOTS,
        ),
    )


def _tensor_dataset(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> TensorDataset:
    x, target, calendar, weight = arrays
    return TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(calendar),
        torch.from_numpy(target),
        torch.from_numpy(weight),
    )


def _fit_torch(
    model: nn.Module,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[nn.Module, float, int]:
    train_loader = DataLoader(
        _tensor_dataset(train_arrays),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(8128),
    )
    validation_loader = DataLoader(
        _tensor_dataset(validation_arrays),
        batch_size=batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    best_loss = math.inf
    best_state = deepcopy(model.state_dict())
    stale = 0
    completed_epochs = 0
    for epoch in range(epochs):
        model.train()
        for x, calendar, target, weight in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(
                model(x, calendar),
                target,
                weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = 0.0
        denominator = 0.0
        with torch.no_grad():
            for x, calendar, target, weight in validation_loader:
                prediction = model(x, calendar)
                total += float(
                    ((prediction - target).square() * weight).sum()
                )
                denominator += float(weight.sum())
        validation_loss = total / max(1.0, denominator)
        completed_epochs = epoch + 1
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    return model, best_loss, completed_epochs


def _predict_torch(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        _tensor_dataset(arrays),
        batch_size=batch_size,
        shuffle=False,
    )
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, calendar, _, _ in loader:
            output.append(model(x, calendar).numpy())
    return np.concatenate(output, axis=0)


def _ridge_prediction(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    train_x, train_y, train_calendar, train_weight = train_arrays
    test_x, _, test_calendar, _ = test_arrays
    design = np.concatenate(
        (train_x.reshape(len(train_x), -1), train_calendar),
        axis=1,
    )
    test_design = np.concatenate(
        (test_x.reshape(len(test_x), -1), test_calendar),
        axis=1,
    )
    prediction = np.zeros_like(test_arrays[1])
    for feature_idx in range(train_y.shape[1]):
        model = Ridge(alpha=1.0)
        model.fit(
            design,
            train_y[:, feature_idx],
            sample_weight=train_weight[:, feature_idx],
        )
        prediction[:, feature_idx] = model.predict(test_design)
    return prediction.astype(np.float32)


def _model(
    name: str,
    *,
    context: int,
    features: int,
    hidden: int,
    scaler: EnvelopeScaler,
) -> nn.Module:
    feasible = name.startswith("Feasible")
    base_name = name.removeprefix("Feasible") if feasible else name
    if base_name == "DLinear":
        model: nn.Module = DLinearRegressor(context, features)
    elif base_name == "LSTM":
        model = LSTMRegressor(features, hidden_size=hidden)
    elif base_name == "TCN":
        model = TCNRegressor(features, hidden_size=hidden)
    elif base_name == "ITransformer":
        heads = 4 if hidden % 4 == 0 else 1
        model = ITransformerRegressor(
            context,
            features,
            d_model=hidden,
            n_heads=heads,
        )
    elif base_name == "iTransformer":
        heads = 4 if hidden % 4 == 0 else 1
        model = ITransformerRegressor(
            context,
            features,
            d_model=hidden,
            n_heads=heads,
        )
    else:
        raise ValueError(f"unknown neural model: {name}")
    if feasible:
        return FeasibleEnvelopeOutput(
            model,
            scaler,
            anchor_slots=DEFAULT_ANCHOR_SLOTS,
        )
    return model


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _daily_records(
    *,
    task: str,
    model: str,
    dates: list[str],
    actual: np.ndarray,
    raw_prediction: np.ndarray,
    prediction: np.ndarray,
    seed: int,
) -> list[dict]:
    n = len(DEFAULT_ANCHOR_SLOTS)
    actual_aggregate = actual.sum(axis=1)
    raw_aggregate = raw_prediction.sum(axis=1)
    aggregate = prediction.sum(axis=1)
    output = []
    for day_idx, day in enumerate(dates):
        for anchor_idx, anchor in enumerate(DEFAULT_ANCHOR_SLOTS):
            output.append(
                {
                    "date": day,
                    "task": task,
                    "model": model,
                    "anchor_slot": anchor,
                    "actual_lower_kwh": float(
                        actual_aggregate[day_idx, anchor_idx]
                    ),
                    "actual_upper_kwh": float(
                        actual_aggregate[day_idx, n + anchor_idx]
                    ),
                    "actual_occupied_equivalent": float(
                        actual_aggregate[day_idx, 2 * n + anchor_idx]
                    ),
                    "raw_predicted_lower_kwh": float(
                        raw_aggregate[day_idx, anchor_idx]
                    ),
                    "raw_predicted_upper_kwh": float(
                        raw_aggregate[day_idx, n + anchor_idx]
                    ),
                    "predicted_lower_kwh": float(
                        aggregate[day_idx, anchor_idx]
                    ),
                    "predicted_upper_kwh": float(
                        aggregate[day_idx, n + anchor_idx]
                    ),
                    "predicted_occupied_equivalent": float(
                        aggregate[day_idx, 2 * n + anchor_idx]
                    ),
                    "seed": seed,
                }
            )
    return output


def main() -> None:
    args = parse_args()
    if args.context_length < 7:
        raise ValueError("context-length must be at least seven")
    if args.quick:
        args.test_end = min(args.test_end, "2023-07-07")
        args.ev_top_k = min(args.ev_top_k, 24)
        args.epochs = min(args.epochs, 3)
    set_deterministic_seed(args.seed)
    torch.set_num_threads(1)

    ev_panel = build_envelope_panel(
        args.data,
        level="ev",
        start=args.panel_start,
        end=args.test_end,
        selection_end=args.train_end,
        top_k=args.ev_top_k,
        minimum_sessions=args.ev_minimum_sessions,
        charger_filter=FIXED_CHARGERS,
    )
    panels = {
        "EV": ev_panel,
        "ChargerMatched": build_envelope_panel(
            args.data,
            level="charger",
            start=args.panel_start,
            end=args.test_end,
            selection_end=args.train_end,
            entity_ids=FIXED_CHARGERS,
            charger_filter=FIXED_CHARGERS,
            driver_filter=ev_panel.entity_ids,
        ),
        "ChargerFull": build_envelope_panel(
            args.data,
            level="charger",
            start=args.panel_start,
            end=args.test_end,
            selection_end=args.train_end,
            entity_ids=FIXED_CHARGERS,
            charger_filter=FIXED_CHARGERS,
        ),
    }
    identity_difference = float(
        np.max(
            np.abs(
                panels["EV"].values.sum(axis=1)
                - panels["ChargerMatched"].values.sum(axis=1)
            )
        )
    )
    if identity_difference > 1e-4:
        raise AssertionError(
            "matched EV and charger panels have different envelopes: "
            f"{identity_difference}"
        )
    for task, panel in panels.items():
        validity = envelope_validity_mask(
            panel.values,
            anchor_slots=panel.anchor_slots,
        )
        if not validity.all():
            raise AssertionError(
                f"{task} contains {int((~validity).sum())} invalid "
                "actual signatures"
            )

    rows: list[dict] = []
    daily_rows: list[dict] = []
    actual_by_task: dict[str, np.ndarray] = {}
    prediction_by_task_model: dict[
        tuple[str, str],
        tuple[list[str], np.ndarray],
    ] = {}

    for task, panel in panels.items():
        train_indices = _indices(
            panel.dates,
            context=args.context_length,
            lower_exclusive=None,
            upper_inclusive=args.train_end,
        )
        validation_indices = _indices(
            panel.dates,
            context=args.context_length,
            lower_exclusive=args.train_end,
            upper_inclusive=args.validation_end,
        )
        test_indices = _indices(
            panel.dates,
            context=args.context_length,
            lower_exclusive=args.validation_end,
            upper_inclusive=args.test_end,
        )
        if not train_indices or not validation_indices or not test_indices:
            raise ValueError(f"{task} has an empty chronological split")
        train_day_mask = np.asarray(
            [day <= args.train_end for day in panel.dates]
        )
        scaler = EnvelopeScaler.fit(panel.values[train_day_mask])
        scaled_values = scaler.transform(panel.values)
        train_arrays = _entity_arrays(
            scaled_values,
            panel.values,
            panel.dates,
            train_indices,
            args.context_length,
        )
        validation_arrays = _entity_arrays(
            scaled_values,
            panel.values,
            panel.dates,
            validation_indices,
            args.context_length,
        )
        test_arrays = _entity_arrays(
            scaled_values,
            panel.values,
            panel.dates,
            test_indices,
            args.context_length,
        )
        test_dates = [panel.dates[idx] for idx in test_indices]
        test_actual = panel.values[test_indices]
        actual_by_task[task] = test_actual
        n_test_days, n_entities, n_features = test_actual.shape

        for name in args.models:
            started = time.perf_counter()
            validation_loss = math.nan
            epochs_completed = 0
            parameter_count = 0
            if name == "SeasonalNaive":
                raw_prediction = panel.values[
                    [idx - 7 for idx in test_indices]
                ].copy()
            elif name == "Ridge":
                prediction_scaled = _ridge_prediction(
                    train_arrays,
                    test_arrays,
                )
                raw_prediction = scaler.inverse_transform(
                    prediction_scaled.reshape(test_actual.shape)
                )
            else:
                set_deterministic_seed(
                    args.seed + 100 * MODELS.index(name)
                )
                model = _model(
                    name,
                    context=args.context_length,
                    features=n_features,
                    hidden=args.hidden_size,
                    scaler=scaler,
                )
                parameter_count = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                model, validation_loss, epochs_completed = _fit_torch(
                    model,
                    train_arrays,
                    validation_arrays,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
                prediction_scaled = _predict_torch(
                    model,
                    test_arrays,
                    batch_size=args.batch_size,
                )
                raw_prediction = scaler.inverse_transform(
                    prediction_scaled.reshape(test_actual.shape)
                )
            prediction = (
                raw_prediction
                if name.startswith("Feasible")
                else project_envelope_signatures(
                    raw_prediction,
                    anchor_slots=panel.anchor_slots,
                )
            )
            raw_validity = envelope_validity_mask(
                raw_prediction,
                anchor_slots=panel.anchor_slots,
            )
            projected_validity = envelope_validity_mask(
                prediction,
                anchor_slots=panel.anchor_slots,
            )
            if not projected_validity.all():
                raise AssertionError(
                    f"projection failed for {task}/{name}"
                )
            raw_metrics = envelope_metrics(
                test_actual,
                raw_prediction,
                target_dates=test_dates,
                anchor_slots=panel.anchor_slots,
            )
            metrics = envelope_metrics(
                test_actual,
                prediction,
                target_dates=test_dates,
                anchor_slots=panel.anchor_slots,
            )
            rows.append(
                {
                    "task": task,
                    "model": name,
                    "comparison_scope": (
                        "matched_recurring_ev"
                        if task != "ChargerFull"
                        else "full_six_charger"
                    ),
                    "entity_count": n_entities,
                    "train_target_days": len(train_indices),
                    "validation_target_days": len(validation_indices),
                    "test_target_days": len(test_indices),
                    "context_days": args.context_length,
                    "anchor_slots": ";".join(
                        str(slot) for slot in panel.anchor_slots
                    ),
                    "scope_energy_coverage": math.nan,
                    "operational_terminal_energy_mae_kwh": math.nan,
                    "operational_terminal_energy_wape": math.nan,
                    "raw_valid_signature_rate": float(
                        raw_validity.mean()
                    ),
                    "projected_valid_signature_rate": float(
                        projected_validity.mean()
                    ),
                    "projection_mean_absolute_adjustment": float(
                        np.abs(prediction - raw_prediction).mean()
                    ),
                    **metrics,
                    **{
                        f"raw_{key}": value
                        for key, value in raw_metrics.items()
                        if key.startswith("aggregate_")
                    },
                    "validation_weighted_mse": validation_loss,
                    "epochs_completed": epochs_completed,
                    "parameter_count": parameter_count,
                    "runtime_seconds": time.perf_counter() - started,
                    "seed": args.seed,
                }
            )
            daily_rows.extend(
                _daily_records(
                    task=task,
                    model=name,
                    dates=test_dates,
                    actual=test_actual,
                    raw_prediction=raw_prediction,
                    prediction=prediction,
                    seed=args.seed,
                )
            )
            prediction_by_task_model[(task, name)] = (
                test_dates,
                prediction,
            )

    full_energy = rows[
        next(
            idx
            for idx, row in enumerate(rows)
            if row["task"] == "ChargerFull"
        )
    ]["scope_test_energy_kwh"]
    for row in rows:
        row["scope_energy_coverage"] = (
            row["scope_test_energy_kwh"] / full_energy
            if full_energy > 0.0
            else math.nan
        )
        if row["task"] == "ChargerFull":
            row["operational_terminal_energy_mae_kwh"] = row[
                "aggregate_terminal_energy_mae_kwh"
            ]
            row["operational_terminal_energy_wape"] = row[
                "aggregate_terminal_energy_wape"
            ]
        elif row["task"] == "EV":
            dates, prediction = prediction_by_task_model[
                ("EV", row["model"])
            ]
            n = len(DEFAULT_ANCHOR_SLOTS)
            matched_actual = actual_by_task["EV"][
                ..., n - 1
            ].sum(axis=1)
            full_actual = actual_by_task["ChargerFull"][
                ..., n - 1
            ].sum(axis=1)
            predicted = prediction[..., n - 1].sum(axis=1)
            error = (
                np.abs(matched_actual - predicted)
                + np.maximum(0.0, full_actual - matched_actual)
            )
            row["operational_terminal_energy_mae_kwh"] = float(
                error.mean()
            )
            row["operational_terminal_energy_wape"] = float(
                error.sum() / max(1e-9, full_actual.sum())
            )

    _write_csv(args.output, rows)
    daily_output = args.output.with_name("daily_signatures.csv")
    _write_csv(daily_output, daily_rows)
    metadata = {
        "status": "quick_sanity" if args.quick else "full_forecast_only",
        "scientific_target": (
            "Forecast cumulative lower/upper deliverable-energy envelopes "
            "and block occupancy directly. No EV-session assignment or "
            "Hungarian metric is used."
        ),
        "data": _portable_path(args.data),
        "metrics": _portable_path(args.output),
        "daily_signatures": _portable_path(daily_output),
        "matched_identity_max_abs_difference": identity_difference,
        "quality_stats": {
            task: panel.quality_stats for task, panel in panels.items()
        },
        "claim_boundary": (
            "This experiment evaluates forecasted flexibility signatures. "
            "The projected signatures have not yet been connected to the "
            "rolling MPC controller."
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("flexibility-envelope forecast benchmark")
    print("matched identity max abs difference:", identity_difference)
    for row in rows:
        print(
            f"{row['task']:15s} {row['model']:13s} "
            f"energy-MAE="
            f"{row['aggregate_terminal_energy_mae_kwh']:.3f} "
            f"lower-MAE={row['aggregate_lower_curve_mae_kwh']:.3f} "
            f"upper-MAE={row['aggregate_upper_curve_mae_kwh']:.3f} "
            f"raw-valid={row['raw_valid_signature_rate']:.3f}",
            flush=True,
        )
    print("output:", args.output)
    print("daily:", daily_output)


if __name__ == "__main__":
    main()
