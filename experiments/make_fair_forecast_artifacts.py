#!/usr/bin/env python3
"""Audit, summarize, and plot the fair EV-versus-charger benchmark."""

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
    PROJECT_ROOT / "experiments" / "results" / "fair_forecast_q3"
)
PAPER_ROOT = PROJECT_ROOT / "paper"
FIGURE_ROOT = PAPER_ROOT / "figures"
TABLE_ROOT = PAPER_ROOT / "tables"

COMMON_MODELS = (
    "SeasonalNaive",
    "Ridge",
    "HurdleRidge",
    "DLinear",
    "LSTM",
    "TCN",
    "iTransformer",
)
PLOT_MODELS = (
    "SeasonalNaive",
    "Ridge",
    "DLinear",
    "LSTM",
    "TCN",
    "iTransformer",
)
LABELS = {
    "SeasonalNaive": "Seasonal naive",
    "Ridge": "Ridge",
    "HurdleRidge": "Hurdle ridge",
    "DLinear": "DLinear",
    "LSTM": "LSTM",
    "TCN": "TCN",
    "iTransformer": "iTransformer",
    "GraphGNN": "GraphGNN",
}
EV_COLOR = "#D55E00"
CHARGER_COLOR = "#0072B2"
GRAPH_COLOR = "#009E73"
ACTUAL_COLOR = "#262626"
GRID_COLOR = "#D7DCE2"


def _load(pattern: str) -> pd.DataFrame:
    paths = sorted(RESULT_ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    return pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )


def _audit_matched_scope(daily: pd.DataFrame) -> None:
    for (seed, model), group in daily.groupby(["seed", "model"]):
        ev = (
            group[group["task"] == "EV"]
            .sort_values("date")
            .reset_index(drop=True)
        )
        charger = (
            group[group["task"] == "ChargerMatched"]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if ev.empty or charger.empty:
            continue
        if not ev["date"].equals(charger["date"]):
            raise AssertionError(
                f"date mismatch for seed={seed}, model={model}"
            )
        np.testing.assert_allclose(
            ev["actual_scope_energy_kwh"],
            charger["actual_scope_energy_kwh"],
            atol=1e-5,
            err_msg=(
                f"matched energy mismatch for seed={seed}, model={model}"
            ),
        )


def _block_bootstrap_mean(
    values: np.ndarray,
    *,
    block_length: int = 7,
    samples: int = 5000,
    seed: int = 20260726,
) -> tuple[float, float, float]:
    if values.ndim != 1 or len(values) < block_length:
        raise ValueError("insufficient one-dimensional bootstrap values")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(len(values) / block_length)
    maximum_start = len(values) - block_length
    boot = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        starts = rng.integers(0, maximum_start + 1, size=blocks)
        sample = np.concatenate(
            [values[start : start + block_length] for start in starts]
        )[: len(values)]
        boot[index] = sample.mean()
    return (
        float(values.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _model_daily_average(
    daily: pd.DataFrame,
    *,
    task: str,
    model: str,
    value: str,
) -> pd.Series:
    selected = daily[
        (daily["task"] == task) & (daily["model"] == model)
    ].copy()
    if selected.empty:
        raise ValueError(f"missing {task}/{model}")
    return selected.groupby("date")[value].mean().sort_index()


def _comparison_summary(
    metrics: pd.DataFrame,
    daily: pd.DataFrame,
) -> list[dict]:
    rows: list[dict] = []
    for model in COMMON_MODELS:
        ev_metrics = metrics[
            (metrics["task"] == "EV") & (metrics["model"] == model)
        ].set_index("seed")
        charger_metrics = metrics[
            (metrics["task"] == "ChargerMatched")
            & (metrics["model"] == model)
        ].set_index("seed")
        seeds = ev_metrics.index.intersection(charger_metrics.index)
        if seeds.empty:
            continue
        ev_metrics = ev_metrics.loc[seeds]
        charger_metrics = charger_metrics.loc[seeds]

        ev_error = _model_daily_average(
            daily[daily["seed"].isin(seeds)],
            task="EV",
            model=model,
            value="absolute_scope_error_kwh",
        )
        charger_error = _model_daily_average(
            daily[daily["seed"].isin(seeds)],
            task="ChargerMatched",
            model=model,
            value="absolute_scope_error_kwh",
        )
        difference = (ev_error - charger_error).to_numpy()
        mean_difference, ci_low, ci_high = _block_bootstrap_mean(
            difference
        )
        ev_mae = float(
            ev_metrics["aggregate_daily_energy_mae_kwh"].mean()
        )
        charger_mae = float(
            charger_metrics["aggregate_daily_energy_mae_kwh"].mean()
        )
        rows.append(
            {
                "model": model,
                "seeds": len(seeds),
                "ev_matched_daily_mae_kwh": ev_mae,
                "ev_matched_daily_mae_sd_kwh": float(
                    ev_metrics["aggregate_daily_energy_mae_kwh"].std(ddof=1)
                    if len(seeds) > 1
                    else 0.0
                ),
                "charger_matched_daily_mae_kwh": charger_mae,
                "charger_matched_daily_mae_sd_kwh": float(
                    charger_metrics[
                        "aggregate_daily_energy_mae_kwh"
                    ].std(ddof=1)
                    if len(seeds) > 1
                    else 0.0
                ),
                "charger_reduction_pct": (
                    100.0 * (ev_mae - charger_mae) / ev_mae
                    if ev_mae > 0.0
                    else 0.0
                ),
                "paired_ev_minus_charger_mae_kwh": mean_difference,
                "paired_block_ci_low_kwh": ci_low,
                "paired_block_ci_high_kwh": ci_high,
                "ev_operational_no_substitution_wape": float(
                    ev_metrics["operational_daily_energy_wape"].mean()
                ),
                "scope_energy_coverage": float(
                    ev_metrics["scope_energy_coverage"].iloc[0]
                ),
            }
        )
    return rows


def _full_summary(metrics: pd.DataFrame) -> list[dict]:
    rows = []
    selected = metrics[metrics["task"] == "ChargerFull"]
    for model, group in selected.groupby("model"):
        rows.append(
            {
                "model": model,
                "seeds": int(group["seed"].nunique()),
                "full_daily_mae_kwh": float(
                    group["aggregate_daily_energy_mae_kwh"].mean()
                ),
                "full_daily_mae_sd_kwh": float(
                    group["aggregate_daily_energy_mae_kwh"].std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "full_energy_wape": float(
                    group["aggregate_daily_energy_wape"].mean()
                ),
                "participation_f1": float(
                    group["participation_f1"].mean()
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def _write_tables(
    comparison: list[dict],
    full: list[dict],
) -> None:
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (
            r"Model & EV matched MAE & Charger matched MAE & "
            r"Reduction & 95\% block CI \\"
        ),
        r" & (kWh/day) & (kWh/day) & (\%) & (kWh/day) \\",
        r"\midrule",
    ]
    for row in comparison:
        lines.append(
            f"{_latex_escape(LABELS[row['model']])} & "
            f"{row['ev_matched_daily_mae_kwh']:.2f} & "
            f"{row['charger_matched_daily_mae_kwh']:.2f} & "
            f"{row['charger_reduction_pct']:.1f} & "
            f"[{row['paired_block_ci_low_kwh']:.2f}, "
            f"{row['paired_block_ci_high_kwh']:.2f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "forecast_fair_comparison.tex").write_text(
        "\n".join(lines)
    )

    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        (
            r"Charger model & Seeds & Full-demand MAE "
            r"(kWh/day) & Energy WAPE \\"
        ),
        r"\midrule",
    ]
    for row in sorted(full, key=lambda item: item["full_daily_mae_kwh"]):
        lines.append(
            f"{_latex_escape(LABELS[row['model']])} & "
            f"{row['seeds']} & "
            f"{row['full_daily_mae_kwh']:.2f} & "
            f"{100.0 * row['full_energy_wape']:.1f}\\% \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "forecast_full_scope.tex").write_text(
        "\n".join(lines)
    )


def _save_figure(name: str) -> None:
    for suffix in ("pdf", "png"):
        plt.savefig(
            FIGURE_ROOT / f"{name}.{suffix}",
            dpi=320,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close()


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def _plot_matched(comparison: pd.DataFrame) -> None:
    selected = (
        comparison.set_index("model").loc[list(PLOT_MODELS)].reset_index()
    )
    y = np.arange(len(selected))
    height = 0.34
    fig, ax = plt.subplots(figsize=(7.25, 4.15))
    ax.barh(
        y + height / 2,
        selected["ev_matched_daily_mae_kwh"],
        xerr=selected["ev_matched_daily_mae_sd_kwh"],
        height=height,
        color=EV_COLOR,
        label="Individual-EV prediction",
        capsize=2.5,
    )
    ax.barh(
        y - height / 2,
        selected["charger_matched_daily_mae_kwh"],
        xerr=selected["charger_matched_daily_mae_sd_kwh"],
        height=height,
        color=CHARGER_COLOR,
        label="Charger prediction",
        capsize=2.5,
    )
    ax.set_yticks(y, [LABELS[model] for model in selected["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate daily energy MAE on identical sessions (kWh)")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    _style_axis(ax)
    for index, row in selected.iterrows():
        if row["charger_reduction_pct"] > 0.05:
            ax.text(
                max(
                    row["ev_matched_daily_mae_kwh"],
                    row["charger_matched_daily_mae_kwh"],
                )
                + 2.0,
                index,
                f"{row['charger_reduction_pct']:.0f}% lower",
                va="center",
                fontsize=8,
                color=ACTUAL_COLOR,
            )
    ax.set_xlim(
        0.0,
        1.25
        * max(
            selected["ev_matched_daily_mae_kwh"].max(),
            selected["charger_matched_daily_mae_kwh"].max(),
        ),
    )
    fig.tight_layout()
    _save_figure("forecast_matched_scope")


def _plot_operational(
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55))
    x = np.arange(len(coverage))
    axes[0].plot(
        x,
        100.0 * coverage["test_energy_coverage"],
        color=EV_COLOR,
        marker="o",
        linewidth=2.0,
    )
    axes[0].set_xticks(
        x,
        coverage["minimum_training_sessions"].astype(str),
    )
    axes[0].set_xlabel("Minimum training sessions per driver")
    axes[0].set_ylabel("Q3 energy represented by EV cohort (%)")
    axes[0].set_ylim(0.0, 46.0)
    axes[0].grid(axis="y", color=GRID_COLOR, linewidth=0.7)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    chosen = coverage[
        coverage["minimum_training_sessions"] == 3
    ].iloc[0]
    chosen_index = int(
        coverage.index[
            coverage["minimum_training_sessions"] == 3
        ][0]
    )
    axes[0].annotate(
        f"Paper cohort: {100.0 * chosen['test_energy_coverage']:.1f}%",
        xy=(chosen_index, 100.0 * chosen["test_energy_coverage"]),
        xytext=(chosen_index + 0.8, 36.5),
        arrowprops={"arrowstyle": "-", "color": ACTUAL_COLOR},
        fontsize=8,
    )

    comparison_plot = (
        comparison.set_index("model").loc[list(PLOT_MODELS)].reset_index()
    )
    full = (
        metrics[
            (metrics["task"] == "ChargerFull")
            & (metrics["model"].isin(PLOT_MODELS))
        ]
        .groupby("model")["operational_daily_energy_wape"]
        .mean()
    )
    y = np.arange(len(comparison_plot))
    height = 0.34
    axes[1].barh(
        y + height / 2,
        100.0
        * comparison_plot["ev_operational_no_substitution_wape"],
        height=height,
        color=EV_COLOR,
        label="EV, no substitution",
    )
    axes[1].barh(
        y - height / 2,
        100.0 * np.asarray([full[model] for model in PLOT_MODELS]),
        height=height,
        color=CHARGER_COLOR,
        label="Charger, full demand",
    )
    axes[1].set_yticks(
        y,
        [LABELS[model] for model in comparison_plot["model"]],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Operational energy WAPE (%)")
    axes[1].legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    _style_axis(axes[1])
    fig.tight_layout(w_pad=2.4)
    _save_figure("forecast_operational_scope")


def _plot_daily(daily: pd.DataFrame) -> None:
    dates = pd.to_datetime(
        sorted(daily[daily["task"] == "ChargerFull"]["date"].unique())
    )
    full_actual = _model_daily_average(
        daily,
        task="ChargerFull",
        model="GraphGNN",
        value="full_actual_energy_kwh",
    )
    graph_prediction = _model_daily_average(
        daily,
        task="ChargerFull",
        model="GraphGNN",
        value="predicted_scope_energy_kwh",
    )
    matched_actual = _model_daily_average(
        daily,
        task="EV",
        model="iTransformer",
        value="actual_scope_energy_kwh",
    )
    ev_prediction = _model_daily_average(
        daily,
        task="EV",
        model="iTransformer",
        value="predicted_scope_energy_kwh",
    )
    charger_prediction = _model_daily_average(
        daily,
        task="ChargerMatched",
        model="iTransformer",
        value="predicted_scope_energy_kwh",
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.25, 5.15), sharex=True)
    axes[0].plot(
        dates,
        full_actual,
        color=ACTUAL_COLOR,
        linewidth=1.7,
        label="Actual full demand",
    )
    axes[0].plot(
        dates,
        graph_prediction,
        color=GRAPH_COLOR,
        linewidth=1.4,
        label="GraphGNN charger forecast",
    )
    axes[0].set_ylabel("Full energy (kWh/day)")
    axes[0].legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )

    axes[1].plot(
        dates,
        matched_actual,
        color=ACTUAL_COLOR,
        linewidth=1.7,
        label="Actual matched scope",
    )
    axes[1].plot(
        dates,
        ev_prediction,
        color=EV_COLOR,
        linewidth=1.2,
        alpha=0.9,
        label="EV iTransformer",
    )
    axes[1].plot(
        dates,
        charger_prediction,
        color=CHARGER_COLOR,
        linewidth=1.4,
        label="Charger iTransformer",
    )
    axes[1].set_ylabel("Matched energy (kWh/day)")
    axes[1].legend(
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8,
    )
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].set_xlabel("Q3 2023 test day")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.75)
        ax.set_axisbelow(True)
    fig.tight_layout(h_pad=1.2)
    _save_figure("forecast_daily_trajectories")


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    metrics = _load("seed_*/metrics.csv")
    daily = _load("seed_*/daily_predictions.csv")
    coverage = pd.read_csv(RESULT_ROOT / "coverage_sensitivity.csv")
    _audit_matched_scope(daily)

    comparison_rows = _comparison_summary(metrics, daily)
    full_rows = _full_summary(metrics)
    _write_csv(
        RESULT_ROOT / "fair_comparison_summary.csv",
        comparison_rows,
    )
    _write_csv(RESULT_ROOT / "full_scope_summary.csv", full_rows)
    _write_tables(comparison_rows, full_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    comparison = pd.DataFrame(comparison_rows)
    _plot_matched(comparison)
    _plot_operational(comparison, metrics, coverage)
    _plot_daily(daily)
    print("fair forecast artifacts:", RESULT_ROOT)


if __name__ == "__main__":
    main()
