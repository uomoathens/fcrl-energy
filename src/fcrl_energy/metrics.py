import numpy as np
import pandas as pd

METRICS = [
    "overall_coverage",
    "worst_off_coverage",
    "lowest_coverage",
    "p10_coverage",
    "coverage_gap",
    "coverage_gini",
    "leakage_share",
    "guarantee_attainment",
]

LABELS = {
    "overall_coverage": "share of vulnerable energy need covered",
    "worst_off_coverage": "coverage of the worst-off tenth of regions (mean over seasons)",
    "lowest_coverage": "lowest regional coverage in any season (descriptive, no interval)",
    "p10_coverage": "coverage at the 10th percentile of regions (mean over seasons)",
    "coverage_gap": "gap between the 90th and 10th percentile of regional coverage (mean over seasons)",
    "coverage_gini": "inequality of coverage across regions, Gini (mean over seasons)",
    "leakage_share": "share of support exceeding need",
    "guarantee_attainment": "share of region-months meeting the guaranteed minimum",
}

HIGHER_IS_BETTER = {
    "overall_coverage": True,
    "worst_off_coverage": True,
    "lowest_coverage": True,
    "p10_coverage": True,
    "coverage_gap": False,
    "coverage_gini": False,
    "leakage_share": False,
    "guarantee_attainment": True,
}


def weighted_gini(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) < 2 or weights.sum() <= 0:
        return 0.0
    mean = np.average(values, weights=weights)
    if mean <= 0:
        return 0.0
    diff = np.abs(values[:, None] - values[None, :])
    return float((weights[:, None] * weights[None, :] * diff).sum() / (2 * weights.sum() ** 2 * mean))


def _lower_tail(values, share):
    values = np.sort(values)
    count = max(1, int(np.ceil(share * len(values))))
    return float(values[:count].mean())


def season_coverage(frame, column):
    covered = np.minimum(frame[column], frame["need"])
    table = frame.assign(covered=covered).groupby(["region", "season_label"]).agg(
        covered=("covered", "sum"), need=("need", "sum"), vulnerable=("vulnerable", "mean"), allocation=(column, "sum")
    )
    table["coverage"] = table["covered"] / table["need"]
    return table.reset_index()


def summarise(frame, column, guarantee_level):
    allocation = frame[column].to_numpy()
    need = frame["need"].to_numpy()
    covered = np.minimum(allocation, need)
    seasons = season_coverage(frame, column)
    per_season = seasons.groupby("season_label")
    minimum = per_season["coverage"].min()
    p10 = per_season["coverage"].quantile(0.1)
    gap = per_season["coverage"].quantile(0.9) - p10
    tail = pd.Series([_lower_tail(block["coverage"].to_numpy(), 0.1) for _, block in per_season])
    gini = pd.Series([weighted_gini(block["coverage"], block["vulnerable"]) for _, block in per_season])
    result = {
        "overall_coverage": float(covered.sum() / need.sum()),
        "worst_off_coverage": float(tail.mean()),
        "lowest_coverage": float(minimum.min()),
        "p10_coverage": float(p10.mean()),
        "coverage_gap": float(gap.mean()),
        "coverage_gini": float(gini.mean()),
        "leakage_share": float(np.maximum(allocation - need, 0).sum() / allocation.sum()),
        "guarantee_attainment": float(np.mean(covered / need >= guarantee_level - 1e-9)) if guarantee_level > 0 else float("nan"),
    }
    return result


def bootstrap(frame, columns, guarantee_level, samples, seed):
    regions = np.array(sorted(frame["region"].unique()))
    blocks = {region: block for region, block in frame.groupby("region")}
    rng = np.random.default_rng(seed)
    draws = {column: [] for column in columns}
    for _ in range(samples):
        chosen = rng.choice(regions, size=len(regions), replace=True)
        parts = []
        for position, region in enumerate(chosen):
            parts.append(blocks[region].assign(region=f"{region}#{position}"))
        sample = pd.concat(parts, ignore_index=True)
        for column in columns:
            draws[column].append(summarise(sample, column, guarantee_level))
    return {column: pd.DataFrame(values) for column, values in draws.items()}


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
