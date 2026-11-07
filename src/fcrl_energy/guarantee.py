import math

import numpy as np
import pandas as pd


def conformal_quantile(values, level):
    values = np.sort(np.asarray(values, dtype=float))
    m = len(values)
    if m == 0:
        return float("nan")
    if level <= 0:
        return 0.0
    rank = math.ceil((m + 1) * level)
    if rank > m:
        return float(values[-1])
    return float(values[max(rank, 1) - 1])


def calibration_ratios(panel, need):
    ratios = need / panel["forecast"]
    train = panel["split"] == "train"
    ratios[train] = panel.loc[train, "ratio"] * need[train] / panel.loc[train, "need"]
    return ratios


def adaptive_quantiles(panel, need, delta, gamma):
    ratios = calibration_ratios(panel, need)
    train = panel["split"] == "train"
    q = pd.Series(np.nan, index=panel.index)
    path = []
    for season in sorted(panel["season"].unique()):
        in_season = panel["season"] == season
        pool = list(ratios[train & in_season])
        static = conformal_quantile(pool, 1 - delta)
        q[train & in_season] = static
        alpha = delta
        later = panel[~train & in_season]
        for ordinal, block in later.groupby("ordinal", sort=True):
            level = min(max(1 - alpha, 0.0), 1.0)
            current = conformal_quantile(pool, level) if level < 1 else float(np.max(pool))
            q[block.index] = current
            realised = ratios[block.index].to_numpy()
            miss = float(np.mean(realised > current))
            path.append(
                {
                    "season": season,
                    "season_label": block["season_label"].iloc[0],
                    "split": block["split"].iloc[0],
                    "year": int(block["year"].iloc[0]),
                    "month": int(block["month"].iloc[0]),
                    "target_miss_rate": delta,
                    "working_miss_rate": alpha,
                    "need_to_forecast_bound": current,
                    "realised_miss_rate": miss,
                }
            )
            alpha = alpha + gamma * (delta - miss)
            pool.extend(realised.tolist())
    return q, pd.DataFrame(path)
