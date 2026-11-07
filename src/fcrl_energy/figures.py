from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import FCRL, FORECAST, HINDSIGHT, STANDARD

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b3"
COLORS = {FCRL: "#2a78d6", STANDARD: "#eb6834", FORECAST: "#1baf7a"}
SHORT = {
    "equal amount per resident": "Equal amount per resident",
    STANDARD: "Equal amount per resident\nat risk of poverty",
    "proportional to last year's need": "Proportional to\nlast year's need",
    FORECAST: "Proportional to\nforecast need",
    FCRL: "FCRL-energy",
    HINDSIGHT: "Hindsight benchmark\n(not implementable)",
}
LEGEND = {FCRL: "FCRL-energy", STANDARD: "Equal amount per resident at risk of poverty", FORECAST: "Proportional to forecast need"}


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT,
            "axes.titlecolor": TEXT,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def _axes(fig, ax, title, subtitle):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    fig.text(0.02, 0.97, title, fontsize=13.5, fontweight="bold", color=TEXT, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=10, color=TEXT_SECONDARY, va="top")


def _percent(ax, upper=1.0):
    ticks = np.linspace(0, upper, int(round(upper / 0.2)) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlim(0, upper)


def worst_off(performance, path):
    _style()
    rows = performance.set_index("strategy")
    order = [name for name in SHORT if name in rows.index]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.8, bottom=0.12)
    for i, name in enumerate(reversed(order)):
        value = rows.loc[name, "worst_off_coverage"]
        color = COLORS[FCRL] if name == FCRL else MUTED
        ax.barh(i, value, height=0.55, color=color, linewidth=0)
        low, high = rows.loc[name, "worst_off_coverage_ci_low"], rows.loc[name, "worst_off_coverage_ci_high"]
        ax.plot([low, high], [i, i], color=TEXT_SECONDARY, linewidth=1.2, solid_capstyle="butt")
        ax.text(max(value, high) + 0.012, i, f"{value:.0%}", va="center", ha="left", fontsize=10.5, color=TEXT, fontweight="bold" if name == FCRL else "normal")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT[name] for name in reversed(order)])
    _percent(ax, 0.8)
    ax.set_xlabel("Share of energy need covered, worst-off tenth of regions (mean over test seasons)")
    _axes(fig, ax, "The same budget protects the worst-off regions far better", "Coverage of the energy need of residents at risk of poverty in the worst-covered 10% of regions. Lines show 95% intervals.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def distribution(regional, path, seed=0):
    _style()
    names = [STANDARD, FORECAST, FCRL]
    names = [name for name in names if name in regional.columns]
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.27, right=0.95, top=0.8, bottom=0.13)
    for i, name in enumerate(reversed(names)):
        values = regional[name].to_numpy()
        jitter = rng.uniform(-0.22, 0.22, len(values))
        ax.scatter(values, i + jitter, s=22, color=COLORS[name], alpha=0.55, linewidths=0.6, edgecolors=SURFACE)
        lowest = values.min()
        ax.plot([lowest, lowest], [i - 0.32, i + 0.32], color=TEXT, linewidth=2)
        ax.text(lowest - 0.012, i + 0.36, f"lowest {lowest:.0%}", ha="right", va="bottom", fontsize=9.5, color=TEXT)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([SHORT[name] for name in reversed(names)])
    ax.set_ylim(-0.6, len(names) - 0.4)
    _percent(ax, 1.0)
    ax.set_xlabel("Share of energy need covered, one dot per region and season")
    _axes(fig, ax, "Support is shared more evenly across regions", "Each dot is one region in one test season; the black mark shows the lowest coverage.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def stress(table, path, metric="lowest_coverage"):
    _style()
    names = [FCRL, STANDARD, FORECAST]
    scenarios = list(dict.fromkeys(table["scenario"]))
    labels = {
        "as observed": "As observed",
        "unexpected 30% rise in energy prices": "Prices +30%",
        "20% cut in the support budget": "Budget −20%",
        "price rise and budget cut together": "Prices +30% and\nbudget −20%",
    }
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.76, bottom=0.12)
    height = 0.24
    for s, scenario in enumerate(reversed(scenarios)):
        block = table[table["scenario"] == scenario].set_index("strategy")
        for k, name in enumerate(names):
            if name not in block.index:
                continue
            y = s + (1 - k) * (height + 0.03)
            value = block.loc[name, metric]
            ax.barh(y, value, height=height, color=COLORS[name], linewidth=0)
            ax.text(value + 0.008, y, f"{value:.0%}", va="center", ha="left", fontsize=9.5, color=TEXT, fontweight="bold" if name == FCRL else "normal")
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([labels.get(name, name) for name in reversed(scenarios)])
    _percent(ax, 0.6)
    ax.set_xlabel("Lowest coverage of any region in any test season")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[name]) for name in names]
    fig.legend(handles, [LEGEND[name] for name in names], loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=3, frameon=False, fontsize=9.5, handlelength=1.0, handleheight=1.0)
    _axes(fig, ax, "Protection holds when prices rise or the budget is cut", "Lowest regional coverage under each shock, for the same budget shared by three rules.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def by_country(regional, path):
    _style()
    data = regional.groupby("country")[[STANDARD, FCRL]].min()
    names = {"EL": "Greece", "BG": "Bulgaria", "ES": "Spain", "PT": "Portugal", "IT": "Italy"}
    data = data.sort_values(FCRL)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.74, bottom=0.14)
    for i, (country, row) in enumerate(data.iterrows()):
        start, end = row[STANDARD], row[FCRL]
        ax.plot([start, end], [i, i], color=MUTED, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter([start], [i], s=70, color=COLORS[STANDARD], edgecolors=SURFACE, linewidths=2, zorder=2)
        ax.scatter([end], [i], s=70, color=COLORS[FCRL], edgecolors=SURFACE, linewidths=2, zorder=3)
        for value, bold in ((start, False), (end, True)):
            other = end if value == start else start
            side = -0.015 if value <= other else 0.015
            ax.text(value + side, i, f"{value:.0%}", ha="right" if side < 0 else "left", va="center", fontsize=9.5, color=TEXT if bold else TEXT_SECONDARY, fontweight="bold" if bold else "normal")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([names.get(code, code) for code in data.index])
    ax.set_ylim(-0.6, len(data) - 0.4)
    _percent(ax, 0.8)
    ax.set_xlabel("Lowest coverage of any region of the country in any test season")
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor=COLORS[name], markeredgecolor=SURFACE) for name in (STANDARD, FCRL)]
    fig.legend(handles, [LEGEND[STANDARD], LEGEND[FCRL]], loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=2, frameon=False, fontsize=9.5)
    gains = int((data[FCRL] > data[STANDARD]).sum())
    title = "The least protected region of every country gains" if gains == len(data) else f"The least protected region gains in {gains} of {len(data)} countries"
    _axes(fig, ax, title, "Lowest regional coverage per country, from an equal amount per resident at risk of poverty to FCRL-energy, same budget.")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_all(performance, regional, stress_table, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "worst_off_coverage.png": lambda p: worst_off(performance, p),
        "regional_coverage.png": lambda p: distribution(regional, p),
        "stress_tests.png": lambda p: stress(stress_table, p),
        "by_country.png": lambda p: by_country(regional, p),
    }
    written = []
    for name, draw in paths.items():
        path = directory / name
        draw(path)
        written.append(path)
    return written
