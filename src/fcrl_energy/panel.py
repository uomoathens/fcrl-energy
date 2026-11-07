import numpy as np
import pandas as pd

FEATURES = ["poverty_rate", "share65", "lag_anomaly", "price_momentum", "region_bias"]
FEATURE_LABELS = {
    "poverty_rate": "share of residents at risk of poverty",
    "share65": "share of residents aged 65 or over",
    "lag_anomaly": "last month's heating or cooling need relative to normal",
    "price_momentum": "energy price change over the last three months",
    "region_bias": "past tendency of the region's need to exceed its forecast",
}


def _lagged_annual(annual, years):
    rows = []
    for region, block in annual.groupby("region"):
        block = block.sort_values("year").set_index("year")
        span = range(min(min(block.index), min(years) - 1), max(max(block.index), max(years)) + 1)
        filled = block.reindex(span)[["population", "share65", "poverty_rate"]].ffill().bfill()
        for year in years:
            source = filled.loc[year - 1] if (year - 1) in filled.index else filled.iloc[0]
            rows.append({"region": region, "year": year, **source.to_dict()})
    return pd.DataFrame(rows)


def _price_lags(prices):
    frames = []
    for country, block in prices.groupby("country"):
        block = block.copy()
        block["ordinal"] = block["year"] * 12 + block["month"] - 1
        block = block.set_index("ordinal").sort_index()
        full = block.reindex(range(block.index.min(), block.index.max() + 1))
        full["price"] = full["price"].interpolate(limit_area="inside")
        out = pd.DataFrame({"ordinal": full.index, "price": full["price"].values})
        out["price_lag1"] = out["price"].shift(1)
        out["price_lag4"] = out["price"].shift(4)
        out["country"] = country
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def _season(month, program):
    if month in program["winter_months"]:
        return "winter"
    if month in program["summer_months"]:
        return "summer"
    return None


def build_panel(raw, config):
    program = config["program"]
    splits = config["splits"]
    kappa = float(program.get("cooling_weight", 1.0))
    base = float(program.get("base_need_degree_days", 10.0))
    train_years = set(splits["train_season_years"])
    regions = raw["regions"]
    degree = raw["degree"].merge(regions[["region", "name", "country"]], on="region")
    degree["ordinal"] = degree["year"] * 12 + degree["month"] - 1
    degree = degree.sort_values(["region", "ordinal"]).reset_index(drop=True)
    lag = degree[["region", "ordinal", "hdd", "cdd"]].copy()
    lag["ordinal"] = lag["ordinal"] + 1
    degree = degree.merge(lag.rename(columns={"hdd": "hdd_lag", "cdd": "cdd_lag"}), on=["region", "ordinal"], how="left")
    first_train, last_train = min(train_years), max(train_years)
    calendar = degree[(degree["year"] >= first_train) & (degree["year"] <= last_train)]
    clim_calendar = calendar.groupby(["region", "month"])[["hdd", "cdd"]].mean().rename(columns={"hdd": "hdd_clim_cal", "cdd": "cdd_clim_cal"}).reset_index()
    degree["lag_month"] = (degree["month"] - 2) % 12 + 1
    degree = degree.merge(clim_calendar.rename(columns={"month": "lag_month", "hdd_clim_cal": "hdd_lag_clim", "cdd_clim_cal": "cdd_lag_clim"}), on=["region", "lag_month"], how="left")
    degree["season"] = degree["month"].map(lambda m: _season(m, program))
    panel = degree[degree["season"].notna()].copy()
    panel["season_year"] = np.where((panel["season"] == "winter") & (panel["month"] <= 6), panel["year"] - 1, panel["year"])
    winter = panel["season"] == "winter"
    panel["degree"] = np.where(winter, panel["hdd"], kappa * panel["cdd"])
    panel["degree_lag"] = np.where(winter, panel["hdd_lag"], kappa * panel["cdd_lag"])
    panel["degree_lag_clim"] = np.where(winter, panel["hdd_lag_clim"], kappa * panel["cdd_lag_clim"])
    in_train = panel["season_year"].isin(train_years)
    groups = panel[in_train].groupby(["region", "month"])["degree"]
    stats = pd.DataFrame({"clim_sum": groups.sum(), "clim_count": groups.count()}).reset_index()
    panel = panel.merge(stats, on=["region", "month"], how="left")
    panel["clim"] = panel["clim_sum"] / panel["clim_count"]
    loo = (panel["clim_sum"] - panel["degree"]) / (panel["clim_count"] - 1)
    panel["clim_calibration"] = np.where(panel["season_year"].isin(train_years) & (panel["clim_count"] > 1), loo, panel["clim"])
    prices = _price_lags(raw["prices"])
    panel = panel.merge(prices, on=["country", "ordinal"], how="left")
    years = sorted(panel["year"].unique())
    annual = _lagged_annual(raw["annual"], years)
    panel = panel.merge(annual, on=["region", "year"], how="left")
    panel["vulnerable"] = panel["population"] * panel["poverty_rate"] / 100.0
    panel["need"] = panel["vulnerable"] * (panel["degree"] + base) * panel["price"]
    panel["forecast"] = panel["vulnerable"] * (panel["clim"] + base) * panel["price_lag1"]
    calibration_forecast = panel["vulnerable"] * (panel["clim_calibration"] + base) * panel["price_lag1"]
    panel["ratio"] = panel["need"] / calibration_forecast
    panel["lag_anomaly"] = np.log((panel["degree_lag"] + base) / (panel["degree_lag_clim"] + base))
    panel["price_momentum"] = np.log(panel["price_lag1"] / panel["price_lag4"])
    required = ["need", "forecast", "ratio", "lag_anomaly", "price_momentum", "poverty_rate", "share65", "vulnerable"]
    panel = panel.dropna(subset=required)
    panel = panel[(panel["forecast"] > 0) & (panel["need"] > 0)]
    bias = panel[panel["season_year"].isin(train_years)].assign(log_ratio=lambda f: np.log(f["ratio"])).groupby(["region", "season"])["log_ratio"].mean().rename("region_bias").reset_index()
    panel = panel.merge(bias, on=["region", "season"], how="left")
    panel["region_bias"] = panel["region_bias"].fillna(0.0)
    expected = {"winter": len(program["winter_months"]), "summer": len(program["summer_months"])}
    sizes = panel.groupby(["region", "season", "season_year"])["month"].transform("count")
    panel = panel[sizes == panel["season"].map(expected)]
    regions_per_season = panel.groupby(["season", "season_year"])["region"].transform("nunique")
    panel = panel[regions_per_season == panel["region"].nunique()].copy()
    split = pd.Series("unused", index=panel.index)
    split[panel["season_year"].isin(splits["train_season_years"])] = "train"
    split[panel["season_year"].isin(splits["validation_season_years"])] = "validation"
    split[panel["season_year"].isin(splits["test_season_years"])] = "test"
    panel["split"] = split
    panel = panel[panel["split"] != "unused"].copy()
    panel["season_label"] = np.where(
        panel["season"] == "winter",
        "winter " + panel["season_year"].astype(str) + "-" + ((panel["season_year"] + 1) % 100).astype(str).str.zfill(2),
        "summer " + panel["season_year"].astype(str),
    )
    train = panel["split"] == "train"
    for name in FEATURES:
        mean = panel.loc[train, name].mean()
        std = panel.loc[train, name].std()
        panel[f"z_{name}"] = (panel[name] - mean) / (std if std and std > 0 else 1.0)
    keep = [
        "region", "name", "country", "year", "month", "ordinal", "season", "season_year", "season_label", "split",
        "hdd", "cdd", "degree", "clim", "price", "price_lag1", "population", "poverty_rate", "share65", "vulnerable",
        "need", "forecast", "ratio", *FEATURES, *[f"z_{name}" for name in FEATURES],
    ]
    return panel[keep].sort_values(["ordinal", "region"]).reset_index(drop=True)


def coverage_table(panel):
    rows = []
    for split, block in panel.groupby("split"):
        rows.append(
            {
                "split": split,
                "seasons": ", ".join(sorted(block["season_label"].unique())),
                "regions": int(block["region"].nunique()),
                "region_months": int(len(block)),
                "mean_need_to_forecast_ratio": float(block["ratio"].mean()),
                "share_of_months_need_above_forecast": float((block["ratio"] > 1).mean()),
            }
        )
    order = {"train": 0, "validation": 1, "test": 2}
    return pd.DataFrame(rows).sort_values("split", key=lambda s: s.map(order)).reset_index(drop=True)
