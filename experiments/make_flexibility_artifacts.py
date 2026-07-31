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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
INK_COLOR = "#1F2937"
PALE_BLUE = "#E8F1F8"
PALE_GREEN = "#E7F5EF"
PALE_ORANGE = "#FBEDE3"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9.5,
        "axes.labelcolor": INK_COLOR,
        "axes.edgecolor": INK_COLOR,
        "axes.titleweight": "semibold",
        "xtick.color": INK_COLOR,
        "ytick.color": INK_COLOR,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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
        "occupancy": "aggregate_occupied_equivalent_mae",
    }
    for (task, model), group in metrics.groupby(["task", "model"]):
        row = {
            "task": task,
            "model": model,
            "seeds": int(group["seed"].nunique()),
            "entity_count": int(group["entity_count"].max()),
            "scope_test_energy_kwh": float(
                group["scope_test_energy_kwh"].mean()
            ),
            "scope_energy_coverage": float(
                group["scope_energy_coverage"].mean()
            ),
            "raw_valid_signature_rate": float(
                group["raw_valid_signature_rate"].mean()
            ),
            "projected_valid_signature_rate": float(
                group["projected_valid_signature_rate"].mean()
            ),
            "parameter_count": int(group["parameter_count"].max()),
            "runtime_seconds_mean": float(group["runtime_seconds"].mean()),
            "projection_adjustment_mean": float(
                group["projection_mean_absolute_adjustment"].mean()
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

    scope = (
        model_summary.groupby("task", as_index=True)
        .agg(
            entity_count=("entity_count", "max"),
            scope_test_energy_kwh=("scope_test_energy_kwh", "mean"),
            scope_energy_coverage=("scope_energy_coverage", "mean"),
        )
    )
    task_lines = [
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        (
            r"Task & Forecast entity & Realized test scope & Entities & "
            r"Q3 energy & Coverage \\"
        ),
        r" & & & & (kWh) & (\%) \\",
        r"\midrule",
        (
            r"Individual EV & driver ID & recurring-driver sessions & "
            f"{int(scope.loc['EV', 'entity_count'])} & "
            f"{scope.loc['EV', 'scope_test_energy_kwh']:.2f} & "
            f"{100.0 * scope.loc['EV', 'scope_energy_coverage']:.2f} \\\\"
        ),
        (
            r"Matched charger & physical port & same recurring-driver "
            r"sessions & "
            f"{int(scope.loc['ChargerMatched', 'entity_count'])} & "
            f"{scope.loc['ChargerMatched', 'scope_test_energy_kwh']:.2f} & "
            f"{100.0 * scope.loc['ChargerMatched', 'scope_energy_coverage']:.2f} \\\\"
        ),
        (
            r"Full charger & physical port & all screened port sessions & "
            f"{int(scope.loc['ChargerFull', 'entity_count'])} & "
            f"{scope.loc['ChargerFull', 'scope_test_energy_kwh']:.2f} & "
            f"{100.0 * scope.loc['ChargerFull', 'scope_energy_coverage']:.2f} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (TABLE_ROOT / "flexibility_task_definition.tex").write_text(
        "\n".join(task_lines)
    )

    design_lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Stage & Mathematical object & Implementation & Role \\",
        r"\midrule",
        (
            r"Target & $[\mathbf L,\mathbf U,\mathbf O]\in"
            r"\mathbb R^{18}$ & six anchors, three channels & "
            r"additive feasible set \\"
        ),
        (
            r"History & $\widetilde{\mathbf X}\in"
            r"\mathbb R^{28\times18}$ & training-only scaling & "
            r"leakage-safe context \\"
        ),
        (
            r"Backbone & 18 variate tokens in $\mathbb R^{32}$ & "
            r"one 4-head encoder layer & cross-feature dependence \\"
        ),
        (
            r"Physical head & $(\widehat{\mathbf L},"
            r"\widehat{\mathbf U},\widehat{\mathbf O})$ & "
            r"sigmoid, softmax, cumulative closure & hard feasibility \\"
        ),
        (
            r"Objective & weighted standardized MSE & active weight 3 & "
            r"retain informative zero days \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (TABLE_ROOT / "flexibility_model_design.tex").write_text(
        "\n".join(design_lines)
    )

    diagnostic = model_summary.copy()
    diagnostic["task_order"] = diagnostic["task"].map(
        {"EV": 0, "ChargerMatched": 1, "ChargerFull": 2}
    )
    diagnostic["model_order"] = diagnostic["model"].map(
        {"SeasonalNaive": 0, "iTransformer": 1, PHYSICS_MODEL: 2}
    )
    diagnostic = diagnostic.sort_values(["task_order", "model_order"])

    rank_maps: dict[tuple[str, str], tuple[float, float]] = {}
    for task, group in diagnostic.groupby("task"):
        for metric in ("terminal", "lower", "upper"):
            values = sorted(group[f"{metric}_mae_mean"].tolist())
            rank_maps[(task, metric)] = (
                values[0],
                values[1] if len(values) > 1 else values[0],
            )

    def metric_cell(row: pd.Series, metric: str) -> str:
        mean = float(row[f"{metric}_mae_mean"])
        sd = float(row[f"{metric}_mae_sd"])
        value = (
            f"{mean:.2f} $\\pm$ {sd:.2f}"
            if int(row["seeds"]) > 1
            else f"{mean:.2f}"
        )
        best, second = rank_maps[(str(row["task"]), metric)]
        if math.isclose(mean, best, rel_tol=0.0, abs_tol=1e-8):
            return f"\\textbf{{{value}}}"
        if math.isclose(mean, second, rel_tol=0.0, abs_tol=1e-8):
            return f"\\underline{{{value}}}"
        return value

    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        (
            r"Task & Model & Seeds & Terminal & Lower & Upper & "
            r"Width & Raw valid & Parameters \\"
        ),
        (
            r" & & & (kWh/day) & (kWh) & (kWh) & (kWh) & "
            r"(\%) & \\"
        ),
        r"\midrule",
    ]
    previous_task = None
    for _, row in diagnostic.iterrows():
        if previous_task is not None and row["task"] != previous_task:
            lines.append(r"\addlinespace")
        lines.append(
            f"{TASK_LABELS[row['task']]} & "
            f"{MODEL_LABELS[row['model']]} & "
            f"{int(row['seeds'])} & "
            f"{metric_cell(row, 'terminal')} & "
            f"{metric_cell(row, 'lower')} & "
            f"{metric_cell(row, 'upper')} & "
            f"{row['width_mae_mean']:.2f} & "
            f"{100.0 * row['raw_valid_signature_rate']:.0f} & "
            f"{int(row['parameter_count']):,} \\\\"
        )
        previous_task = row["task"]
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_ROOT / "flexibility_model_diagnostics.tex").write_text(
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


def _diagram_box(
    ax: plt.Axes,
    *,
    x: float,
    width: float,
    text: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    y = 0.34
    height = 0.34
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.15,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=8.25,
        color=INK_COLOR,
        linespacing=1.25,
    )


def _diagram_arrow(
    ax: plt.Axes,
    *,
    start: float,
    end: float,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (start, 0.51),
            (end, 0.51),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color="#64748B",
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def _plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.25, 2.75))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    boxes = (
        (
            0.02,
            0.13,
            "28-day history\n$\\mathbf{X}\\in\\mathbb{R}^{28\\times18}$",
            PALE_ORANGE,
            EV_COLOR,
            "raw target history",
        ),
        (
            0.18,
            0.13,
            "Training-only\nstandardization",
            PALE_ORANGE,
            EV_COLOR,
            "$\\widetilde{\\mathbf{X}}$",
        ),
        (
            0.34,
            0.17,
            "iTransformer\n18 variate tokens\n4-head attention + FFN",
            PALE_BLUE,
            CHARGER_COLOR,
            "$\\mathbf{H}\\in\\mathbb{R}^{18\\times32}$",
        ),
        (
            0.54,
            0.10,
            "Latent\nlogits",
            PALE_BLUE,
            CHARGER_COLOR,
            "$\\mathbf{z}\\in\\mathbb{R}^{18}$",
        ),
        (
            0.67,
            0.18,
            "Physical output layer\nsigmoid occupancy + energy\n"
            "softmax increments\nmonotone capacity closure",
            PALE_GREEN,
            PHYSICS_COLOR,
            "hard feasibility",
        ),
        (
            0.88,
            0.10,
            "Feasible\nsignature\n$[\\widehat{\\mathbf{L}},"
            "\\widehat{\\mathbf{U}},"
            "\\widehat{\\mathbf{O}}]$",
            PALE_GREEN,
            PHYSICS_COLOR,
            "$18$ outputs",
        ),
    )
    for x, width, label, face, edge, dimension in boxes:
        _diagram_box(
            ax,
            x=x,
            width=width,
            text=label,
            facecolor=face,
            edgecolor=edge,
        )
        ax.text(
            x + width / 2.0,
            0.24,
            dimension,
            ha="center",
            va="top",
            fontsize=7.6,
            color="#64748B",
        )

    for left, right in zip(boxes, boxes[1:]):
        _diagram_arrow(
            ax,
            start=left[0] + left[1] + 0.004,
            end=right[0] - 0.004,
        )

    stage_headers = (
        (
            0.02,
            0.31,
            "DATA AND CAUSAL PREPROCESSING",
            EV_COLOR,
        ),
        (0.34, 0.64, "LEARNED BACKBONE", CHARGER_COLOR),
        (0.67, 0.98, "PHYSICAL DECODER", PHYSICS_COLOR),
    )
    for start, end, label, color in stage_headers:
        ax.text(
            0.5 * (start + end),
            0.82,
            label,
            ha="center",
            va="center",
            fontsize=8.1,
            weight="semibold",
            color=color,
        )
        ax.plot(
            [start, end],
            [0.77, 0.77],
            color=color,
            linewidth=1.0,
        )

    fig.tight_layout(pad=0.2)
    _save_figure("physics_itransformer_architecture")


def _plot_matched(matched: pd.DataFrame) -> None:
    y = np.arange(len(matched))
    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    for position, (_, row) in enumerate(matched.iterrows()):
        ax.plot(
            [row["charger_physics_mae"], row["ev_physics_mae"]],
            [position, position],
            color="#CBD5E1",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=1,
        )
        ax.errorbar(
            row["ev_physics_mae"],
            position,
            xerr=row["ev_physics_sd"],
            fmt="o",
            markersize=7,
            color=EV_COLOR,
            ecolor=EV_COLOR,
            elinewidth=1.5,
            capsize=3,
            zorder=3,
        )
        ax.errorbar(
            row["charger_physics_mae"],
            position,
            xerr=row["charger_physics_sd"],
            fmt="o",
            markersize=7,
            color=CHARGER_COLOR,
            ecolor=CHARGER_COLOR,
            elinewidth=1.5,
            capsize=3,
            zorder=3,
        )
        ax.scatter(
            row["seasonal_matched_mae"],
            position,
            marker="D",
            s=35,
            color=SEASONAL_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        midpoint = 0.5 * (
            row["ev_physics_mae"] + row["charger_physics_mae"]
        )
        ax.text(
            midpoint,
            position - 0.18,
            f"{row['charger_reduction_vs_ev_pct']:.0f}% lower",
            ha="center",
            va="center",
            fontsize=8.3,
            color=INK_COLOR,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
        )
    ax.set_yticks(
        y,
        [METRICS[value] for value in matched["metric"]],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate Q3 MAE (kWh; lower is better)")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=EV_COLOR,
                markeredgecolor=EV_COLOR,
                label="Individual-EV physics",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=CHARGER_COLOR,
                markeredgecolor=CHARGER_COLOR,
                label="Charger physics",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=SEASONAL_COLOR,
                markeredgecolor=SEASONAL_COLOR,
                label="Seasonal naive",
            ),
        ],
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=8.5,
    )
    _style_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save_figure("flexibility_matched_scope")


def _plot_full(model_summary: pd.DataFrame) -> None:
    selected = model_summary[
        model_summary["task"] == "ChargerFull"
    ].set_index("model")
    metrics = ("terminal", "lower", "upper")
    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.25, 3.25))
    ax.axvspan(0.75, 1.0, color=PALE_GREEN, alpha=0.7, zorder=0)
    ax.axvline(
        1.0,
        color=SEASONAL_COLOR,
        linewidth=1.4,
        linestyle="--",
        zorder=1,
    )
    for model, color, offset in (
        ("iTransformer", PROJECTED_COLOR, -0.11),
        (PHYSICS_MODEL, PHYSICS_COLOR, 0.11),
    ):
        row = selected.loc[model]
        ratios = np.array(
            [
                row[f"{metric}_mae_mean"]
                / selected.loc["SeasonalNaive", f"{metric}_mae_mean"]
                for metric in metrics
            ]
        )
        errors = np.array(
            [
                row[f"{metric}_mae_sd"]
                / selected.loc["SeasonalNaive", f"{metric}_mae_mean"]
                for metric in metrics
            ]
        )
        ax.errorbar(
            ratios,
            y + offset,
            xerr=errors,
            fmt="o",
            markersize=7,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            label=MODEL_LABELS[model],
            zorder=3,
        )
    ax.set_yticks(y, [METRICS[metric] for metric in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("MAE relative to seasonal naive (lower is better)")
    ax.set_xlim(0.75, 1.08)
    ax.text(
        0.985,
        1.02,
        "better",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.2,
        color=PHYSICS_COLOR,
    )
    ax.text(
        1.005,
        1.02,
        "seasonal reference",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=SEASONAL_COLOR,
    )
    ax.legend(
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.09),
        fontsize=8.5,
    )
    _style_axis(ax, grid_axis="x")
    ax.tick_params(axis="y", length=0)
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
    ax.set_xticklabels(
        [f"{int(value):02d}:00" for value in hour],
        rotation=30,
        ha="right",
    )
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
    _plot_architecture()
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
