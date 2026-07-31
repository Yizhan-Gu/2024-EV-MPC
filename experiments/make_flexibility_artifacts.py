#!/usr/bin/env python3
"""Summarize and plot the physical flexibility-envelope experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    PROJECT_ROOT / "experiments" / "results" / "flexibility_forecast_q3"
)
PAPER_ROOT = PROJECT_ROOT / "paper"
FIGURE_ROOT = PAPER_ROOT / "figures"
TABLE_ROOT = PAPER_ROOT / "tables"

PHYSICS_MODEL = "FeasibleITransformer"
MODEL_LABELS = {
    "SeasonalNaive": "Seasonal naive",
    "iTransformer": "iTransformer + projection",
    PHYSICS_MODEL: "Physics-iTransformer",
}
TASK_LABELS = {
    "EV": "Individual EV",
    "ChargerMatched": "Physical charger",
    "ChargerFull": "Full charger demand",
}
METRICS = {
    "terminal": "Terminal energy",
    "lower": "Lower envelope",
    "upper": "Upper envelope",
}
EV_COLOR = "#D55E00"
CHARGER_COLOR = "#0072B2"
PHYSICS_COLOR = "#009E73"
SEASONAL_COLOR = "#6B7280"
PROJECTED_COLOR = "#CC79A7"
ACTUAL_COLOR = "#222222"
GRID_COLOR = "#D8DEE8"


def _load(pattern: str) -> pd.DataFrame:
    paths = sorted(RESULT_ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    return pd.concat(
        [
            pd.read_csv(path).assign(run=path.parent.name)
            for path in paths
        ],
        ignore_index=True,
    )


def _daily_errors(daily: pd.DataFrame) -> pd.DataFrame:
    working = daily.copy()
    working["lower_error"] = np.abs(
        working["actual_lower_kwh"] - working["predicted_lower_kwh"]
    )
    working["upper_error"] = np.abs(
        working["actual_upper_kwh"] - working["predicted_upper_kwh"]
    )
    working["actual_width"] = (
        working["actual_upper_kwh"] - working["actual_lower_kwh"]
    )
    working["predicted_width"] = (
        working["predicted_upper_kwh"]
        - working["predicted_lower_kwh"]
    )
    working["width_error"] = np.abs(
        working["actual_width"] - working["predicted_width"]
    )
    curve = (
        working.groupby(["run", "date", "task", "model"], as_index=False)
        .agg(
            lower=("lower_error", "mean"),
            upper=("upper_error", "mean"),
            width=("width_error", "mean"),
        )
    )
    terminal = (
        working[working["anchor_slot"] == 96]
        .assign(
            terminal=lambda frame: np.abs(
                frame["actual_lower_kwh"]
                - frame["predicted_lower_kwh"]
            )
        )[["run", "date", "task", "model", "terminal"]]
    )
    return curve.merge(
        terminal,
        on=["run", "date", "task", "model"],
        validate="one_to_one",
    )


def _block_bootstrap_mean(
    values: np.ndarray,
    *,
    block_length: int = 7,
    samples: int = 5000,
    seed: int = 20260730,
) -> tuple[float, float, float]:
    if values.ndim != 1 or len(values) < block_length:
        raise ValueError("insufficient one-dimensional bootstrap values")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(len(values) / block_length)
    maximum_start = len(values) - block_length
    boot = np.empty(samples, dtype=np.float64)
    for sample_idx in range(samples):
        starts = rng.integers(
            0,
            maximum_start + 1,
            size=n_blocks,
        )
        sample = np.concatenate(
            [
                values[start : start + block_length]
                for start in starts
            ]
        )[: len(values)]
        boot[sample_idx] = sample.mean()
    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _model_summary(metrics: pd.DataFrame) -> list[dict]:
    rows = []
    columns = {
        "terminal": "aggregate_terminal_energy_mae_kwh",
        "lower": "aggregate_lower_curve_mae_kwh",
        "upper": "aggregate_upper_curve_mae_kwh",
        "width": "aggregate_flexibility_width_mae_kwh",
    }
    for (task, model), group in metrics.groupby(["task", "model"]):
        row = {
            "task": task,
            "model": model,
            "seeds": int(group["seed"].nunique()),
            "scope_energy_coverage": float(
                group["scope_energy_coverage"].mean()
            ),
            "raw_valid_signature_rate": float(
                group["raw_valid_signature_rate"].mean()
            ),
        }
        for label, column in columns.items():
            row[f"{label}_mae_mean"] = float(group[column].mean())
            row[f"{label}_mae_sd"] = float(
                group[column].std(ddof=1)
                if len(group) > 1
                else 0.0
            )
        rows.append(row)
    return rows


def _matched_summary(
    model_summary: pd.DataFrame,
    daily_error: pd.DataFrame,
) -> list[dict]:
    physics = model_summary[
        model_summary["model"] == PHYSICS_MODEL
    ].set_index("task")
    seasonal = model_summary[
        model_summary["model"] == "SeasonalNaive"
    ].set_index("task")
    rows = []
    for metric in ("terminal", "lower", "upper"):
        ev = float(physics.loc["EV", f"{metric}_mae_mean"])
        charger = float(
            physics.loc["ChargerMatched", f"{metric}_mae_mean"]
        )
        seasonal_value = float(
            seasonal.loc["ChargerMatched", f"{metric}_mae_mean"]
        )
        selected = daily_error[
            daily_error["model"] == PHYSICS_MODEL
        ]
        averaged = (
            selected.groupby(["date", "task"])[metric]
            .mean()
            .unstack("task")
            .sort_index()
        )
        difference = (
            averaged["EV"] - averaged["ChargerMatched"]
        ).to_numpy()
        _, ci_low, ci_high = _block_bootstrap_mean(difference)
        rows.append(
            {
                "metric": metric,
                "ev_physics_mae": ev,
                "ev_physics_sd": float(
                    physics.loc["EV", f"{metric}_mae_sd"]
                ),
                "charger_physics_mae": charger,
                "charger_physics_sd": float(
                    physics.loc[
                        "ChargerMatched",
                        f"{metric}_mae_sd",
                    ]
                ),
                "charger_reduction_vs_ev_pct": (
                    100.0 * (ev - charger) / ev
                ),
                "ev_minus_charger_block_ci_low": ci_low,
                "ev_minus_charger_block_ci_high": ci_high,
                "seasonal_matched_mae": seasonal_value,
                "charger_reduction_vs_seasonal_pct": (
                    100.0
                    * (seasonal_value - charger)
                    / seasonal_value
                ),
            }
        )
    return rows


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


def _write_tables(
    matched: list[dict],
    model_summary: pd.DataFrame,
) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        (
            r"Target & EV physics & Charger physics & Reduction "
            r"& 95\% block CI & Seasonal \\"
        ),
        (
            r" & (kWh) & (kWh) & (\%) & "
            r"(EV minus charger, kWh) & (kWh) \\"
        ),
        r"\midrule",
    ]
    for row in matched:
        lines.append(
            f"{METRICS[row['metric']]} & "
            f"{row['ev_physics_mae']:.2f} $\\pm$ "
            f"{row['ev_physics_sd']:.2f} & "
            f"{row['charger_physics_mae']:.2f} $\\pm$ "
            f"{row['charger_physics_sd']:.2f} & "
            f"{row['charger_reduction_vs_ev_pct']:.1f} & "
            f"[{row['ev_minus_charger_block_ci_low']:.2f}, "
            f"{row['ev_minus_charger_block_ci_high']:.2f}] & "
            f"{row['seasonal_matched_mae']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "flexibility_matched_summary.tex").write_text(
        "\n".join(lines)
    )

    selected = model_summary[
        model_summary["task"] == "ChargerFull"
    ].set_index("model")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Full-demand model & Seeds & Terminal energy & "
            r"Lower envelope & Upper envelope \\"
        ),
        r" & & (kWh/day) & (kWh/anchor) & (kWh/anchor) \\",
        r"\midrule",
    ]
    for model in ("SeasonalNaive", "iTransformer", PHYSICS_MODEL):
        row = selected.loc[model]
        suffix = (
            f" $\\pm$ {row['terminal_mae_sd']:.2f}"
            if int(row["seeds"]) > 1
            else ""
        )
        lower_suffix = (
            f" $\\pm$ {row['lower_mae_sd']:.2f}"
            if int(row["seeds"]) > 1
            else ""
        )
        upper_suffix = (
            f" $\\pm$ {row['upper_mae_sd']:.2f}"
            if int(row["seeds"]) > 1
            else ""
        )
        lines.append(
            f"{MODEL_LABELS[model]} & {int(row['seeds'])} & "
            f"{row['terminal_mae_mean']:.2f}{suffix} & "
            f"{row['lower_mae_mean']:.2f}{lower_suffix} & "
            f"{row['upper_mae_mean']:.2f}{upper_suffix} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "flexibility_full_summary.tex").write_text(
        "\n".join(lines)
    )


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color=GRID_COLOR,
        linewidth=0.75,
        alpha=0.9,
    )
    ax.set_axisbelow(True)


def _save_figure(name: str) -> None:
    for suffix in ("pdf", "png"):
        plt.savefig(
            FIGURE_ROOT / f"{name}.{suffix}",
            dpi=320,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()


def _plot_matched(matched: pd.DataFrame) -> None:
    x = np.arange(len(matched))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.25, 3.9))
    ax.bar(
        x - width,
        matched["ev_physics_mae"],
        yerr=matched["ev_physics_sd"],
        width=width,
        color=EV_COLOR,
        label="Individual-EV physics model",
        capsize=3,
    )
    ax.bar(
        x,
        matched["charger_physics_mae"],
        yerr=matched["charger_physics_sd"],
        width=width,
        color=CHARGER_COLOR,
        label="Charger physics model",
        capsize=3,
    )
    ax.bar(
        x + width,
        matched["seasonal_matched_mae"],
        width=width,
        color=SEASONAL_COLOR,
        alpha=0.82,
        label="Seasonal naive (both resolutions)",
    )
    ax.set_xticks(
        x,
        [METRICS[value] for value in matched["metric"]],
    )
    ax.set_ylabel("Aggregate Q3 MAE (kWh)")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8.5,
    )
    _style_axis(ax)
    for index, row in matched.iterrows():
        ax.text(
            index,
            row["charger_physics_mae"] + row["charger_physics_sd"] + 1.8,
            f"{row['charger_reduction_vs_ev_pct']:.0f}% lower",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=CHARGER_COLOR,
        )
    fig.tight_layout()
    _save_figure("flexibility_matched_scope")


def _plot_full(model_summary: pd.DataFrame) -> None:
    selected = model_summary[
        model_summary["task"] == "ChargerFull"
    ].set_index("model")
    models = ("SeasonalNaive", "iTransformer", PHYSICS_MODEL)
    metrics = ("terminal", "lower", "upper")
    colors = (SEASONAL_COLOR, PROJECTED_COLOR, PHYSICS_COLOR)
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.25, 3.9))
    for model_idx, (model, color) in enumerate(zip(models, colors)):
        row = selected.loc[model]
        offset = (model_idx - 1) * width
        ax.bar(
            x + offset,
            [row[f"{metric}_mae_mean"] for metric in metrics],
            yerr=[row[f"{metric}_mae_sd"] for metric in metrics],
            width=width,
            color=color,
            label=MODEL_LABELS[model],
            capsize=3,
        )
    ax.set_xticks(x, [METRICS[metric] for metric in metrics])
    ax.set_ylabel("Full-demand Q3 MAE (kWh)")
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8.5,
    )
    _style_axis(ax)
    fig.tight_layout()
    _save_figure("flexibility_full_scope")


def _plot_daily(daily_error: pd.DataFrame) -> None:
    physics = daily_error[
        daily_error["model"] == PHYSICS_MODEL
    ]
    mean_seed = (
        physics.groupby(["date", "task"])["terminal"]
        .mean()
        .unstack("task")
    )
    seasonal = (
        daily_error[
            (daily_error["model"] == "SeasonalNaive")
            & (daily_error["task"] == "ChargerMatched")
        ]
        .set_index("date")["terminal"]
        .sort_index()
    )
    index = pd.to_datetime(mean_seed.index)
    fig, ax = plt.subplots(figsize=(7.25, 3.8))
    ax.plot(
        index,
        mean_seed["EV"].rolling(7, min_periods=1).mean(),
        color=EV_COLOR,
        linewidth=1.8,
        label="Individual-EV physics model",
    )
    ax.plot(
        index,
        mean_seed["ChargerMatched"].rolling(7, min_periods=1).mean(),
        color=CHARGER_COLOR,
        linewidth=1.8,
        label="Charger physics model",
    )
    ax.plot(
        index,
        seasonal.rolling(7, min_periods=1).mean(),
        color=SEASONAL_COLOR,
        linewidth=1.4,
        linestyle="--",
        label="Seasonal naive",
    )
    ax.set_ylabel("7-day mean terminal-energy\nabsolute error (kWh)")
    ax.set_xlabel("Q3 2023 test date")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8.5,
    )
    _style_axis(ax)
    fig.autofmt_xdate(rotation=0)
    fig.tight_layout()
    _save_figure("flexibility_daily_error")


def _plot_example(daily: pd.DataFrame) -> None:
    selected = daily[
        (daily["model"] == PHYSICS_MODEL)
        & (daily["task"] == "ChargerFull")
    ].copy()
    selected["terminal_error"] = np.where(
        selected["anchor_slot"] == 96,
        np.abs(
            selected["actual_lower_kwh"]
            - selected["predicted_lower_kwh"]
        ),
        np.nan,
    )
    error_by_day = (
        selected.groupby(["run", "date"])["terminal_error"]
        .max()
        .groupby("date")
        .mean()
        .sort_values()
    )
    example_day = error_by_day.index[len(error_by_day) // 2]
    day = selected[selected["date"] == example_day]
    averaged = (
        day.groupby("anchor_slot", as_index=False)
        .agg(
            actual_lower=("actual_lower_kwh", "first"),
            actual_upper=("actual_upper_kwh", "first"),
            predicted_lower=("predicted_lower_kwh", "mean"),
            predicted_upper=("predicted_upper_kwh", "mean"),
        )
        .sort_values("anchor_slot")
    )
    hour = averaged["anchor_slot"].to_numpy() / 4.0
    fig, ax = plt.subplots(figsize=(7.25, 3.9))
    ax.fill_between(
        hour,
        averaged["actual_lower"],
        averaged["actual_upper"],
        color=SEASONAL_COLOR,
        alpha=0.22,
        label="Realized feasible-energy band",
    )
    ax.plot(
        hour,
        averaged["actual_lower"],
        color=ACTUAL_COLOR,
        linewidth=1.6,
    )
    ax.plot(
        hour,
        averaged["actual_upper"],
        color=ACTUAL_COLOR,
        linewidth=1.6,
    )
    ax.fill_between(
        hour,
        averaged["predicted_lower"],
        averaged["predicted_upper"],
        color=PHYSICS_COLOR,
        alpha=0.25,
        label="Physics-iTransformer forecast band",
    )
    ax.plot(
        hour,
        averaged["predicted_lower"],
        color=PHYSICS_COLOR,
        linewidth=1.8,
    )
    ax.plot(
        hour,
        averaged["predicted_upper"],
        color=PHYSICS_COLOR,
        linewidth=1.8,
    )
    ax.set_title(f"Full six-charger flexibility envelope: {example_day}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Cumulative feasible energy (kWh)")
    ax.set_xticks(hour)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save_figure("flexibility_envelope_example")


def main() -> None:
    metrics = _load("seed_*/metrics.csv")
    daily = _load("seed_*/daily_signatures.csv")
    model_rows = _model_summary(metrics)
    model_summary = pd.DataFrame(model_rows)
    daily_error = _daily_errors(daily)
    matched_rows = _matched_summary(model_summary, daily_error)
    matched = pd.DataFrame(matched_rows)

    _write_csv(RESULT_ROOT / "model_summary.csv", model_rows)
    _write_csv(RESULT_ROOT / "matched_summary.csv", matched_rows)
    _write_tables(matched_rows, model_summary)
    _plot_matched(matched)
    _plot_full(model_summary)
    _plot_daily(daily_error)
    _plot_example(daily)

    print("flexibility artifacts")
    print(matched.to_string(index=False))
    print("figures:", FIGURE_ROOT)
    print("tables:", TABLE_ROOT)


if __name__ == "__main__":
    main()
