import numpy as np
import pandas as pd


def generate(config):
    settings = config["data"]["synthetic"]
    rng = np.random.default_rng(int(settings.get("seed", 0)))
    countries = list(settings.get("countries", ["AA", "BB", "CC", "DD"]))
    per_country = int(settings.get("regions_per_country", 10))
    first_year = int(settings.get("first_year", 2014))
    last_year = int(settings.get("last_year", 2025))
    persistence = float(settings.get("persistence", 0.6))
    warming = float(settings.get("summer_warming_per_year", 0.03))
    regions = []
    for c_index, country in enumerate(countries):
        for r_index in range(per_country):
            north = rng.uniform(0, 1)
            regions.append(
                {
                    "region": f"{country}{r_index + 1:02d}",
                    "name": f"Region {country}-{r_index + 1}",
                    "country": country,
                    "nuts_version": 2021,
                    "north": north,
                    "poverty_base": float(np.clip(rng.normal(18 + 8 * (1 - north) + 3 * c_index, 5), 6, 45)),
                    "population_base": float(rng.lognormal(13.5, 0.7)),
                    "share65_base": float(np.clip(rng.normal(21, 3), 14, 30)),
                }
            )
    regions = pd.DataFrame(regions)
    months = [(year, month) for year in range(first_year, last_year + 1) for month in range(1, 13)]
    heating_shape = np.array([1.0, 0.85, 0.6, 0.3, 0.08, 0.0, 0.0, 0.0, 0.03, 0.2, 0.5, 0.85])
    cooling_shape = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.5, 1.0, 0.95, 0.4, 0.05, 0.0, 0.0])
    degree_rows = []
    for row in regions.itertuples():
        heat_peak = 250 + 300 * row.north
        cool_peak = 40 + 160 * (1 - row.north)
        anomaly = 0.0
        for year, month in months:
            anomaly = persistence * anomaly + rng.normal(0, 0.18)
            trend = 1 + warming * (year - first_year)
            hdd = max(0.0, heat_peak * heating_shape[month - 1] * (1 - anomaly) * (2 - trend) ** 0.5)
            cdd = max(0.0, cool_peak * cooling_shape[month - 1] * (1 + anomaly) * trend)
            degree_rows.append({"region": row.region, "year": year, "month": month, "hdd": hdd, "cdd": cdd})
    degree = pd.DataFrame(degree_rows)
    price_rows = []
    for c_index, country in enumerate(countries):
        shock = 0.4 + 0.15 * c_index
        level = 1.0
        for year, month in months:
            level *= float(np.exp(rng.normal(0.0015, 0.008)))
            bump = 0.0
            if year == 2022:
                bump = shock * min(month, 9) / 9
            elif year >= 2023:
                bump = shock * max(0.35, 1 - 0.25 * (year - 2022) - 0.02 * month)
            price_rows.append({"country": country, "year": year, "month": month, "price": level * (1 + bump)})
    prices = pd.DataFrame(price_rows)
    annual_rows = []
    for row in regions.itertuples():
        for year in range(first_year, last_year + 1):
            crisis = 2.0 if year in (2022, 2023) else 0.0
            annual_rows.append(
                {
                    "region": row.region,
                    "year": year,
                    "population": row.population_base * (1 + 0.002 * (year - first_year)),
                    "share65": row.share65_base + 0.2 * (year - first_year),
                    "poverty_rate": float(np.clip(row.poverty_base + crisis + rng.normal(0, 0.8), 3, 60)),
                }
            )
    annual = pd.DataFrame(annual_rows)
    log = [{"source": "synthetic", "regions": len(regions), "countries": len(countries)}]
    return {"regions": regions[["region", "name", "country", "nuts_version"]], "degree": degree, "annual": annual, "prices": prices, "log": log}
