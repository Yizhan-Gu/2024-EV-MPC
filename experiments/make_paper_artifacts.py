#!/usr/bin/env python3
"""Generate paper figures and LaTeX tables from audited compact CSVs."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "experiments" / "results" / "paper_2023Q3"
PAPER_ROOT = PROJECT_ROOT / "paper"
FIGURE_ROOT = PAPER_ROOT / "figures"
TABLE_ROOT = PAPER_ROOT / "tables"

METHODS = (
    "V0G",
    "Perfect",
    "NoForecast",
    "Persistence",
    "HistoricalMedian",
    "ConformalRobust",
)
LABELS = {
    "V0G": "V0G",
    "Perfect": "Perfect",
    "NoForecast": "No forecast",
    "Persistence": "Persistence",
    "HistoricalMedian": "Hist. median",
    "ConformalRobust": "Conformal",
}
COLORS = {
    "V0G": "#7f7f7f",
    "Perfect": "#2a6fbb",
    "NoForecast": "#d98c2b",
    "Persistence": "#b34d4d",
    "HistoricalMedian": "#6a51a3",
    "ConformalRobust": "#238b45",
}


def _read(name: str) -> list[dict[str, str]]:
    with (RESULT_ROOT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _save_figure(name: str) -> None:
    for suffix in ("pdf", "png"):
        plt.savefig(
            FIGURE_ROOT / f"{name}.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close()


def _latex_escape(value: str) -> str:
    return value.replace("%", r"\%").replace("_", r"\_")


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    monthly = _read("control_monthly.csv")
    forecast = _read("forecast_summary.csv")
    synthetic = _read("scalability_synthetic.csv")
    real_scalability = _read("scalability.csv")

    by_month_method = {
        (row["month"], row["method"]): row for row in monthly
    }
    months = sorted({row["month"] for row in monthly})
    aggregate_cost = {
        method: sum(
            float(by_month_method[(month, method)]["continuous_cost"])
            for month in months
        )
        for method in METHODS
    }
    v0g_cost = aggregate_cost["V0G"]
    perfect_cost = aggregate_cost["Perfect"]
    aggregate_saving = {
        method: 100.0 * (v0g_cost - cost) / v0g_cost
        for method, cost in aggregate_cost.items()
    }
    aggregate_regret = {
        method: 100.0 * (cost - perfect_cost) / perfect_cost
        for method, cost in aggregate_cost.items()
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(months))
    width = 0.13
    offsets = (
        np.arange(len(METHODS)) - (len(METHODS) - 1) / 2.0
    ) * width
    for offset, method in zip(offsets, METHODS):
        values = [
            float(by_month_method[(month, method)]["continuous_cost"])
            for month in months
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            label=LABELS[method],
            color=COLORS[method],
        )
    ax.set_xticks(x, [month.replace("2023-", "") for month in months])
    ax.set_xlabel("2023 billing month")
    ax.set_ylabel("Continuous charging cost (USD)")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=3, frameon=False, loc="upper center")
    _save_figure("monthly_control_cost")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    median_forecast = next(
        row for row in forecast if row["method"] == "HistoricalMedian"
    )
    coverage_names = ("Arrival", "Departure", "Energy")
    coverage_values = [
        100.0 * float(median_forecast["arrival_upper_coverage"]),
        100.0 * float(median_forecast["departure_lower_coverage"]),
        100.0 * float(median_forecast["energy_upper_coverage"]),
    ]
    axes[0].bar(
        coverage_names,
        coverage_values,
        color=COLORS["ConformalRobust"],
        width=0.62,
    )
    axes[0].axhline(
        90.0,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label="90% target",
    )
    axes[0].set_ylim(85.0, 96.0)
    axes[0].set_ylabel("One-sided empirical coverage (%)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    causal = (
        "NoForecast",
        "Persistence",
        "HistoricalMedian",
        "ConformalRobust",
    )
    values = [aggregate_saving[method] for method in causal]
    axes[1].bar(
        [LABELS[method] for method in causal],
        values,
        color=[COLORS[method] for method in causal],
        width=0.62,
    )
    axes[1].axhline(
        aggregate_saving["Perfect"],
        color=COLORS["Perfect"],
        linestyle="--",
        linewidth=1.0,
        label="Perfect",
    )
    axes[1].set_ylabel("Q3 saving relative to V0G (%)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    _save_figure("coverage_and_control_value")

    grouped_speedup: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in synthetic:
        grouped_speedup[
            (
                int(row["charger_count"]),
                int(row["sessions_per_charger"]),
            )
        ].append(float(row["runtime_ratio_ev_over_charger"]))
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for chargers in sorted({key[0] for key in grouped_speedup}):
        points = sorted(
            (
                sessions_per_charger,
                statistics.median(values),
            )
            for (count, sessions_per_charger), values
            in grouped_speedup.items()
            if count == chargers
        )
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=f"{chargers} chargers",
        )
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Sessions per charger per day")
    ax.set_ylabel("Runtime ratio: EV LP / charger LP")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    _save_figure("synthetic_scalability")

    summary_rows = []
    for method in METHODS:
        summary_rows.append(
            {
                "method": method,
                "q3_cost": aggregate_cost[method],
                "saving_pct_vs_v0g": aggregate_saving[method],
                "regret_pct_vs_perfect": aggregate_regret[method],
            }
        )
    with (PAPER_ROOT / "results_summary.csv").open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    control_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Q3 cost (USD) & Saving vs. V0G & Regret vs. perfect \\",
        r"\midrule",
    ]
    for row in summary_rows:
        control_lines.append(
            f"{_latex_escape(LABELS[row['method']])} & "
            f"{row['q3_cost']:.2f} & "
            f"{row['saving_pct_vs_v0g']:.2f}\\% & "
            f"{row['regret_pct_vs_perfect']:.2f}\\% \\\\"
        )
    control_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "control_summary.tex").write_text(
        "\n".join(control_lines)
    )

    forecast_lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & Count MAE & Arrival MAE & Departure MAE & Energy MAE \\",
        r" & (sessions) & (slots) & (slots) & (kWh) \\",
        r"\midrule",
    ]
    for row in forecast:
        forecast_lines.append(
            f"{_latex_escape(LABELS[row['method']])} & "
            f"{float(row['count_mae_per_charger_day']):.3f} & "
            f"{float(row['arrival_mae_slots']):.3f} & "
            f"{float(row['departure_mae_slots']):.3f} & "
            f"{float(row['energy_mae_kwh']):.3f} \\\\"
        )
    forecast_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "forecast_summary.tex").write_text(
        "\n".join(forecast_lines)
    )

    validation_rows = []
    for alpha in ("0.1", "0.2", "0.3", "0.4"):
        path = (
            PROJECT_ROOT
            / "experiments"
            / "results"
            / f"validation_alpha_{alpha}"
            / "control_monthly.csv"
        )
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        validation_rows.append((alpha, float(row["continuous_cost"])))
    validation_lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"$\alpha$ & Nominal coverage & June validation cost (USD) \\",
        r"\midrule",
    ]
    for alpha, cost in validation_rows:
        validation_lines.append(
            f"{float(alpha):.1f} & "
            f"{100.0 * (1.0 - float(alpha)):.0f}\\% & "
            f"{cost:.2f} \\\\"
        )
    validation_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "validation_alpha.tex").write_text(
        "\n".join(validation_lines)
    )

    max_objective_gap = max(
        abs(float(row["objective_gap"])) for row in real_scalability
    )
    max_load_gap = max(
        float(row["max_load_gap_kw"]) for row in real_scalability
    )
    speedup_8 = statistics.median(
        float(row["runtime_ratio_ev_over_charger"])
        for row in synthetic
        if int(row["sessions_per_charger"]) == 8
    )
    invariant_lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Diagnostic & Value \\",
        r"\midrule",
        f"Maximum absolute objective gap & {max_objective_gap:.2e} \\\\",
        f"Maximum aggregate-load gap (kW) & {max_load_gap:.2e} \\\\",
        f"Median synthetic speedup at 8 sessions/charger & "
        f"{speedup_8:.2f}$\\times$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (TABLE_ROOT / "invariance_scalability.tex").write_text(
        "\n".join(invariant_lines)
    )
    print("paper artifacts:", PAPER_ROOT)


if __name__ == "__main__":
    main()
