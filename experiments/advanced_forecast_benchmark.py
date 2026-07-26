#!/usr/bin/env python3
"""Compare genuinely distinct EV- and charger-level forecast tasks."""

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
from sklearn.linear_model import LogisticRegression, Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from charger_forecasting import (  # noqa: E402
    DLinearRegressor,
    FEATURE_NAMES,
    GraphTemporalRegressor,
    ITransformerRegressor,
    LSTMRegressor,
    PanelScaler,
    TCNRegressor,
    build_daily_panel,
    calendar_features,
    correlation_adjacency,
    masked_mse,
    regression_metrics,
    set_deterministic_seed,
)
from charger_forecasting.training import target_mask  # noqa: E402


FIXED_CHARGERS = (
    "UCSD / GILMAN 2-2|2",
    "UCSD / SCHOLARS - 07|2",
    "UCSD / RADY P357 5|2",
    "UCSD / BIRCH AQUARIUM|1",
    "UCSD / SCHOLARS - 01|1",
    "UCSD / SCHOLARS - 08|2",
)
ENTITY_MODELS = (
    "SeasonalNaive",
    "Ridge",
    "HurdleRidge",
    "DLinear",
    "LSTM",
    "TCN",
    "iTransformer",
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
    parser.add_argument("--test-end", default="2023-09-30")
    parser.add_argument("--context-length", type=int, default=28)
    parser.add_argument("--ev-top-k", type=int, default=512)
    parser.add_argument("--ev-minimum-sessions", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ENTITY_MODELS + ("GraphGNN",),
        default=list(ENTITY_MODELS) + ["GraphGNN"],
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "results"
            / "advanced_forecast_quick"
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
    x_entity = x.transpose(0, 2, 1, 3).reshape(
        days * entities,
        lookback,
        features,
    )
    y_entity = y.reshape(days * entities, features)
    raw_entity = raw_y.reshape(days * entities, features)
    calendar = np.repeat(
        calendar_features([dates[idx] for idx in indices]),
        entities,
        axis=0,
    )
    return (
        x_entity.astype(np.float32),
        y_entity.astype(np.float32),
        calendar.astype(np.float32),
        target_mask(raw_entity),
    )


def _graph_arrays(
    scaled_values: np.ndarray,
    raw_values: np.ndarray,
    dates: tuple[str, ...],
    indices: list[int],
    context: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _window_raw(scaled_values, indices, context)
    _, raw_y = _window_raw(raw_values, indices, context)
    return (
        x.astype(np.float32),
        y.astype(np.float32),
        calendar_features([dates[idx] for idx in indices]),
        target_mask(raw_y),
    )


def _tensor_dataset(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> TensorDataset:
    x, y, calendar, mask = arrays
    return TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(calendar),
        torch.from_numpy(y),
        torch.from_numpy(mask),
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
) -> tuple[nn.Module, float]:
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
    for _ in range(epochs):
        model.train()
        for x, calendar, y, mask in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(x, calendar), y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        total = 0.0
        denominator = 0.0
        with torch.no_grad():
            for x, calendar, y, mask in validation_loader:
                prediction = model(x, calendar)
                total += float(
                    ((prediction - y).square() * mask).sum()
                )
                denominator += float(mask.sum())
        validation_loss = total / max(1.0, denominator)
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    return model, best_loss


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
    train_x, train_y, train_calendar, train_mask = train_arrays
    test_x, _, test_calendar, _ = test_arrays
    design = np.concatenate(
        (train_x.reshape(len(train_x), -1), train_calendar),
        axis=1,
    )
    test_design = np.concatenate(
        (test_x.reshape(len(test_x), -1), test_calendar),
        axis=1,
    )
    prediction = np.zeros((len(test_x), train_y.shape[1]), dtype=np.float32)
    for feature_idx in range(train_y.shape[1]):
        selected = train_mask[:, feature_idx] > 0.0
        model = Ridge(alpha=1.0)
        model.fit(
            design[selected],
            train_y[selected, feature_idx],
            sample_weight=train_mask[selected, feature_idx],
        )
        prediction[:, feature_idx] = model.predict(test_design)
    return prediction


def _hurdle_ridge_prediction(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test_arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    scaler: PanelScaler,
    *,
    seed: int,
) -> np.ndarray:
    """Two-stage participation and conditional-attribute regression."""

    train_x, train_y, train_calendar, train_mask = train_arrays
    test_x, _, test_calendar, _ = test_arrays
    design = np.concatenate(
        (train_x.reshape(len(train_x), -1), train_calendar),
        axis=1,
    )
    test_design = np.concatenate(
        (test_x.reshape(len(test_x), -1), test_calendar),
        axis=1,
    )
    active = train_mask[:, 2] > 0.0
    if active.all():
        probability = np.ones(len(test_x), dtype=np.float32)
    elif not active.any():
        probability = np.zeros(len(test_x), dtype=np.float32)
    else:
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=seed,
        )
        classifier.fit(design, active.astype(np.int8))
        probability = classifier.predict_proba(test_design)[:, 1]

    conditional_scaled = np.zeros(
        (len(test_x), train_y.shape[1]),
        dtype=np.float32,
    )
    if active.any():
        for feature_idx in range(train_y.shape[1]):
            model = Ridge(alpha=1.0)
            model.fit(
                design[active],
                train_y[active, feature_idx],
            )
            conditional_scaled[:, feature_idx] = model.predict(test_design)
    conditional_raw = scaler.inverse_transform(conditional_scaled)
    conditional_raw[:, 0:2] *= probability[:, None]
    conditional_raw[probability < 0.5, 2:] = 0.0
    return conditional_raw


def _model(
    name: str,
    *,
    context: int,
    features: int,
    hidden: int,
) -> nn.Module:
    if name == "DLinear":
        return DLinearRegressor(context, features)
    if name == "LSTM":
        return LSTMRegressor(features, hidden_size=hidden)
    if name == "TCN":
        return TCNRegressor(features, hidden_size=hidden)
    if name == "iTransformer":
        heads = 4 if hidden % 4 == 0 else 1
        return ITransformerRegressor(
            context,
            features,
            d_model=hidden,
            n_heads=heads,
        )
    raise ValueError(f"unknown neural model: {name}")


def _write_csv(path: Path, rows: list[dict]) -> None:
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


def _daily_energy_records(
    *,
    task: str,
    model: str,
    dates: list[str],
    actual: np.ndarray,
    prediction: np.ndarray,
    seed: int,
) -> list[dict]:
    actual_energy = actual[..., 1].sum(axis=1)
    predicted_energy = prediction[..., 1].sum(axis=1)
    return [
        {
            "date": day,
            "task": task,
            "model": model,
            "actual_scope_energy_kwh": float(observed),
            "predicted_scope_energy_kwh": float(predicted),
            "absolute_scope_error_kwh": float(abs(observed - predicted)),
            "full_actual_energy_kwh": math.nan,
            "unmodelled_energy_kwh": math.nan,
            "no_substitution_operational_error_kwh": math.nan,
            "seed": seed,
        }
        for day, observed, predicted in zip(
            dates,
            actual_energy,
            predicted_energy,
        )
    ]


def main() -> None:
    args = parse_args()
    if args.context_length < 7:
        raise ValueError("context length must be at least seven days")
    if args.quick:
        args.test_end = min(args.test_end, "2023-07-07")
        args.ev_top_k = min(args.ev_top_k, 24)
        args.epochs = min(args.epochs, 3)
        args.hidden_size = min(args.hidden_size, 16)

    set_deterministic_seed(args.seed)
    torch.set_num_threads(1)
    ev_panel = build_daily_panel(
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
        "ChargerMatched": build_daily_panel(
            args.data,
            level="charger",
            start=args.panel_start,
            end=args.test_end,
            selection_end=args.train_end,
            entity_ids=FIXED_CHARGERS,
            charger_filter=FIXED_CHARGERS,
            driver_filter=ev_panel.entity_ids,
        ),
        "ChargerFull": build_daily_panel(
            args.data,
            level="charger",
            start=args.panel_start,
            end=args.test_end,
            selection_end=args.train_end,
            entity_ids=FIXED_CHARGERS,
            charger_filter=FIXED_CHARGERS,
        ),
    }
    np.testing.assert_allclose(
        panels["EV"].values[..., 1].sum(axis=1),
        panels["ChargerMatched"].values[..., 1].sum(axis=1),
        atol=1e-4,
        err_msg="matched EV and charger scopes contain different energy",
    )
    rows: list[dict] = []
    daily_rows: list[dict] = []
    task_test_energy: dict[str, float] = {}
    predictions_by_task_model: dict[
        tuple[str, str],
        tuple[list[str], np.ndarray],
    ] = {}
    full_actual_by_date: dict[str, float] = {}
    actual_energy_by_task_date: dict[str, dict[str, float]] = {}

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
            raise ValueError(f"{task}: an experiment split is empty")
        train_day_mask = np.asarray(
            [day <= args.train_end for day in panel.dates]
        )
        scaler = PanelScaler.fit(panel.values[train_day_mask])
        scaled = scaler.transform(panel.values)
        train_arrays = _entity_arrays(
            scaled,
            panel.values,
            panel.dates,
            train_indices,
            args.context_length,
        )
        validation_arrays = _entity_arrays(
            scaled,
            panel.values,
            panel.dates,
            validation_indices,
            args.context_length,
        )
        test_arrays = _entity_arrays(
            scaled,
            panel.values,
            panel.dates,
            test_indices,
            args.context_length,
        )
        test_dates = [panel.dates[idx] for idx in test_indices]
        test_actual = panel.values[test_indices]
        task_test_energy[task] = float(test_actual[..., 1].sum())
        actual_energy_by_task_date[task] = dict(
            zip(
                test_dates,
                test_actual[..., 1].sum(axis=1).astype(float),
            )
        )
        if task == "ChargerFull":
            full_actual_by_date = actual_energy_by_task_date[task]

        for name in args.models:
            if name == "GraphGNN":
                continue
            started = time.perf_counter()
            validation_loss = math.nan
            parameter_count = 0
            if name == "SeasonalNaive":
                raw_x, _ = _window_raw(
                    panel.values,
                    test_indices,
                    args.context_length,
                )
                prediction = raw_x[:, -7]
            elif name == "Ridge":
                prediction_scaled = _ridge_prediction(
                    train_arrays,
                    test_arrays,
                )
                prediction = scaler.inverse_transform(
                    prediction_scaled.reshape(test_actual.shape)
                )
            elif name == "HurdleRidge":
                prediction = _hurdle_ridge_prediction(
                    train_arrays,
                    test_arrays,
                    scaler,
                    seed=args.seed,
                ).reshape(test_actual.shape)
            else:
                set_deterministic_seed(
                    args.seed + 100 * ENTITY_MODELS.index(name)
                )
                model = _model(
                    name,
                    context=args.context_length,
                    features=panel.values.shape[2],
                    hidden=args.hidden_size,
                )
                parameter_count = sum(
                    parameter.numel() for parameter in model.parameters()
                )
                model, validation_loss = _fit_torch(
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
                prediction = scaler.inverse_transform(
                    prediction_scaled.reshape(test_actual.shape)
                )
            metrics = regression_metrics(
                test_actual,
                prediction,
                target_dates=test_dates,
            )
            rows.append(
                {
                    "task": task,
                    "model": name,
                    "entity_count": len(panel.entity_ids),
                    "train_target_days": len(train_indices),
                    "validation_target_days": len(validation_indices),
                    "test_target_days": len(test_indices),
                    "context_days": args.context_length,
                    "comparison_scope": (
                        "matched_recurring_ev"
                        if task != "ChargerFull"
                        else "full_six_charger"
                    ),
                    "scope_energy_coverage": math.nan,
                    "operational_daily_energy_mae_kwh": math.nan,
                    "operational_daily_energy_wape": math.nan,
                    **metrics,
                    "validation_masked_mse": validation_loss,
                    "parameter_count": parameter_count,
                    "runtime_seconds": time.perf_counter() - started,
                    "seed": args.seed,
                }
            )
            daily_rows.extend(
                _daily_energy_records(
                    task=task,
                    model=name,
                    dates=test_dates,
                    actual=test_actual,
                    prediction=prediction,
                    seed=args.seed,
                )
            )
            predictions_by_task_model[(task, name)] = (
                test_dates,
                prediction[..., 1].sum(axis=1),
            )

        if task.startswith("Charger") and "GraphGNN" in args.models:
            started = time.perf_counter()
            adjacency = correlation_adjacency(
                panel.values[train_day_mask],
                feature_index=1,
                top_k=min(3, len(panel.entity_ids) - 1),
            )
            graph_train = _graph_arrays(
                scaled,
                panel.values,
                panel.dates,
                train_indices,
                args.context_length,
            )
            graph_validation = _graph_arrays(
                scaled,
                panel.values,
                panel.dates,
                validation_indices,
                args.context_length,
            )
            graph_test = _graph_arrays(
                scaled,
                panel.values,
                panel.dates,
                test_indices,
                args.context_length,
            )
            set_deterministic_seed(args.seed + 1000)
            model = GraphTemporalRegressor(
                panel.values.shape[2],
                adjacency,
                hidden_size=args.hidden_size,
            )
            parameter_count = sum(
                parameter.numel() for parameter in model.parameters()
            )
            model, validation_loss = _fit_torch(
                model,
                graph_train,
                graph_validation,
                epochs=args.epochs,
                batch_size=min(args.batch_size, 64),
                learning_rate=args.learning_rate,
            )
            prediction_scaled = _predict_torch(
                model,
                graph_test,
                batch_size=min(args.batch_size, 64),
            )
            prediction = scaler.inverse_transform(prediction_scaled)
            metrics = regression_metrics(
                test_actual,
                prediction,
                target_dates=test_dates,
            )
            rows.append(
                {
                    "task": task,
                    "model": "GraphGNN",
                    "entity_count": len(panel.entity_ids),
                    "train_target_days": len(train_indices),
                    "validation_target_days": len(validation_indices),
                    "test_target_days": len(test_indices),
                    "context_days": args.context_length,
                    "comparison_scope": (
                        "matched_recurring_ev"
                        if task == "ChargerMatched"
                        else "full_six_charger"
                    ),
                    "scope_energy_coverage": math.nan,
                    "operational_daily_energy_mae_kwh": math.nan,
                    "operational_daily_energy_wape": math.nan,
                    **metrics,
                    "validation_masked_mse": validation_loss,
                    "parameter_count": parameter_count,
                    "runtime_seconds": time.perf_counter() - started,
                    "seed": args.seed,
                }
            )
            daily_rows.extend(
                _daily_energy_records(
                    task=task,
                    model="GraphGNN",
                    dates=test_dates,
                    actual=test_actual,
                    prediction=prediction,
                    seed=args.seed,
                )
            )
            predictions_by_task_model[(task, "GraphGNN")] = (
                test_dates,
                prediction[..., 1].sum(axis=1),
            )

    full_energy = task_test_energy["ChargerFull"]
    for row in rows:
        task = row["task"]
        row["scope_energy_coverage"] = (
            task_test_energy[task] / full_energy
            if full_energy > 0.0
            else math.nan
        )
        if task == "ChargerFull":
            row["operational_daily_energy_mae_kwh"] = row[
                "aggregate_daily_energy_mae_kwh"
            ]
            row["operational_daily_energy_wape"] = row[
                "aggregate_daily_energy_wape"
            ]
        elif task == "EV":
            dates, predicted = predictions_by_task_model[
                (task, row["model"])
            ]
            matched_actual = np.asarray(
                [
                    actual_energy_by_task_date["EV"][day]
                    for day in dates
                ],
                dtype=np.float64,
            )
            full_actual = np.asarray(
                [full_actual_by_date[day] for day in dates],
                dtype=np.float64,
            )
            unmodelled = np.maximum(0.0, full_actual - matched_actual)
            error = np.abs(matched_actual - predicted) + unmodelled
            row["operational_daily_energy_mae_kwh"] = float(error.mean())
            row["operational_daily_energy_wape"] = float(
                error.sum() / max(1e-9, full_actual.sum())
            )
    for daily in daily_rows:
        full_actual = full_actual_by_date[daily["date"]]
        daily["full_actual_energy_kwh"] = full_actual
        if daily["task"] == "EV":
            unmodelled = max(
                0.0,
                full_actual - daily["actual_scope_energy_kwh"],
            )
            daily["unmodelled_energy_kwh"] = unmodelled
            daily["no_substitution_operational_error_kwh"] = (
                daily["absolute_scope_error_kwh"] + unmodelled
            )
    _write_csv(args.output, rows)
    daily_output = args.output.with_name("daily_predictions.csv")
    _write_csv(daily_output, daily_rows)
    metadata = {
        "status": "quick_sanity" if args.quick else "full_forecast_only",
        "data": _portable_path(args.data),
        "panel_start": args.panel_start,
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "test_end": args.test_end,
        "fixed_chargers": list(FIXED_CHARGERS),
        "ev_top_k": args.ev_top_k,
        "ev_minimum_sessions": args.ev_minimum_sessions,
        "target_features": list(FEATURE_NAMES),
        "models": args.models,
        "epochs": args.epochs,
        "seed": args.seed,
        "important_boundary": (
            "Forecast-only benchmark; predictions are not yet reconstructed "
            "into feasible sessions or evaluated in MPC."
        ),
        "coverage_warning": (
            "EV and ChargerMatched contain exactly the same sessions. "
            "ChargerFull covers all six-charger demand. Operational EV error "
            "adds matched-scope absolute error and unmodelled demand without "
            "allowing overprediction of one driver to replace another."
        ),
        "daily_predictions": _portable_path(daily_output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("advanced forecast benchmark")
    for row in rows:
        print(
            f"{row['task']:15s} {row['model']:13s} "
            f"energy-MAE={row['energy_mae_kwh']:.3f} "
            f"aggregate-day-MAE="
            f"{row['aggregate_daily_energy_mae_kwh']:.3f} "
            f"coverage={row['scope_energy_coverage']:.3f}",
            flush=True,
        )
    print("output:", args.output)
    print("daily:", daily_output)


if __name__ == "__main__":
    main()
