"""Create the E2 core-lineup paired-effect forest plot for notebook 14."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "e2_contextual_hybrid_screen" / "summary.json"
OUTPUT_DIR = ROOT / "results" / "e2_contextual_hybrid_screen"
PNG = OUTPUT_DIR / "figure6_e2_core_paired_effects.png"
PDF = OUTPUT_DIR / "figure6_e2_core_paired_effects.pdf"

CORE_LINEUPS = ("external_mixed", "frozen_dddqn")
HYBRIDS = (
    ("original_hybrid", "Original hybrid", "o", "#0072B2"),
    ("phase_productive_hybrid", "Phase/productive", "s", "#009E73"),
    ("armed_threat_hybrid", "Armed threat", "D", "#E6AB02"),
)
METRICS = (
    (
        "score",
        "Score Δ",
        "↑ better",
    ),
    (
        "bombs",
        "Bombs / round Δ",
        "Exposure only",
    ),
    (
        "confirmed_catastrophic_self_trap",
        "Confirmed traps / round Δ",
        "↓ safer",
    ),
)


def paired(summary: dict, lineup: str, hybrid: str, metric: str) -> tuple[float, float, float]:
    """Return hybrid-minus-relaxed effect and interval from stored relaxed-minus-hybrid rows."""
    contrast = f"one_branch_relaxed-{hybrid}"
    row = next(
        item for item in summary["paired_bootstrap"]
        if item["lineup"] == lineup
        and item["contrast"] == contrast
        and item["metric"] == metric
    )
    return -row["mean_difference"], -row["ci_high"], -row["ci_low"]


def metric_limits(summary: dict, metric: str) -> tuple[float, float]:
    values = [0.0]
    for lineup in CORE_LINEUPS:
        for hybrid, *_ in HYBRIDS:
            estimate, low, high = paired(summary, lineup, hybrid, metric)
            values.extend((estimate, low, high))
    span = max(values) - min(values)
    padding = max(span * 0.12, 0.025 if metric == "confirmed_catastrophic_self_trap" else 0.1)
    return min(values) - padding, max(values) + padding


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    missing = set(CORE_LINEUPS) - set(summary["protocol"]["lineups"])
    if missing:
        raise ValueError(f"Missing core lineups: {sorted(missing)}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 6.25), constrained_layout=False)
    y_positions = list(range(len(HYBRIDS)))[::-1]
    x_limits = {metric: metric_limits(summary, metric) for metric, *_ in METRICS}

    for row_index, lineup in enumerate(CORE_LINEUPS):
        for column_index, (metric, title, direction) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            axis.axvline(0, color="#8A8A8A", linewidth=1.0, linestyle="--", zorder=0)
            axis.axhline(-0.5, color="#D8D8D8", linewidth=0.6, zorder=0)
            for y, (hybrid, label, marker, color) in zip(y_positions, HYBRIDS):
                estimate, low, high = paired(summary, lineup, hybrid, metric)
                axis.errorbar(
                    estimate,
                    y,
                    xerr=[[estimate - low], [high - estimate]],
                    fmt=marker,
                    markersize=6.2,
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    ecolor=color,
                    elinewidth=1.7,
                    capsize=3.0,
                    capthick=1.2,
                    zorder=3,
                )
            axis.set_xlim(*x_limits[metric])
            axis.set_ylim(-0.55, len(HYBRIDS) - 0.45)
            axis.set_yticks(y_positions)
            if column_index == 0:
                axis.set_yticklabels([item[1] for item in HYBRIDS])
            else:
                axis.set_yticklabels([])
                axis.tick_params(axis="y", length=0)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.grid(axis="x", color="#E6E6E6", linewidth=0.7, zorder=0)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
            axis.set_xlabel(r"$\Delta$ (hybrid $-$ relaxed)")
            if row_index == 0:
                axis.set_title(f"{title}\n{direction}", pad=10, fontweight="semibold")

    legend_handles = [
        Line2D(
            [0], [0], marker=marker, color=color, label=label, linestyle="none",
            markerfacecolor=color, markeredgecolor="white", markersize=7,
        )
        for _, label, marker, color in HYBRIDS
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.005),
        columnspacing=2.2,
    )
    figure.suptitle(
        "E2: Paired effects in the core lineups",
        x=0.55,
        y=0.985,
        fontsize=13,
        fontweight="bold",
    )
    figure.text(
        0.55,
        0.947,
        "Points: mean seed differences; bars: paired 95% CIs.",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#555555",
    )
    figure.subplots_adjust(left=0.21, right=0.985, top=0.84, bottom=0.16, wspace=0.28, hspace=0.56)
    for row_index, lineup in enumerate(CORE_LINEUPS):
        box = axes[row_index, 0].get_position()
        figure.text(
            0.025,
            (box.y0 + box.y1) / 2,
            lineup,
            rotation=90,
            va="center",
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#333333",
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(PDF, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
