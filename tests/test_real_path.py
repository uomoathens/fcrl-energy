import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcrl_energy import pipeline, sources
from fcrl_energy.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]


def jsonstat(dimensions, rows):
    ids = list(dimensions)
    categories = {name: sorted({row[name] for row in rows}) for name in ids}
    sizes = [len(categories[name]) for name in ids]
    index = {name: {code: i for i, code in enumerate(categories[name])} for name in ids}
    values = {}
    for row in rows:
        position = 0
        for name, size in zip(ids, sizes):
            position = position * size + index[name][row[name]]
        values[str(position)] = row["value"]
    return {
        "id": ids,
        "size": sizes,
        "dimension": {name: {"category": {"index": index[name]}} for name in ids},
        "value": values,
    }


def fake_network(raw, missing_nuts2_degree, missing_poverty):
    regions = raw["regions"].copy()
    codes = {"AA": "EL", "BB": "IT", "CC": "ES", "DD": "PT"}
    regions["code"] = [codes[c] + f"{i + 1:02d}" for c, i in zip(regions["country"], regions.groupby("country").cumcount())]
    rename = dict(zip(regions["region"], regions["code"]))
    degree = raw["degree"].assign(region=raw["degree"]["region"].map(rename))
    annual = raw["annual"].assign(region=raw["annual"]["region"].map(rename))
    prices = raw["prices"].assign(country=raw["prices"]["country"].map(codes))
    calls = []

    def respond(url, cache, key, pause=1.0):
        calls.append(url)
        if "nutspt_2" in url or "nutspt_3" in url:
            features = []
            for row in regions.itertuples():
                if "nutspt_2" in url:
                    features.append({"properties": {"id": row.code, "na": row.name}, "geometry": {"coordinates": [0, 0]}})
                else:
                    for child in ("1", "2"):
                        features.append({"properties": {"id": row.code + child, "na": row.name}, "geometry": {"coordinates": [0, 0]}})
            return {"features": features}
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        dataset = parsed.path.rsplit("/", 1)[-1]
        geo = query["geo"][0]
        if dataset == "nrg_chddr2_m":
            assert query["sinceTimePeriod"][0] in ("2014-01", "2014M01")
            if geo == missing_nuts2_degree:
                raise sources.SourceError("400")
            parent = geo[:4]
            block = degree[degree["region"] == parent]
            rows = []
            for r in block.itertuples():
                for indicator, value in (("HDD", r.hdd), ("CDD", r.cdd)):
                    rows.append({"unit": "NR", "indic_nrg": indicator, "geo": geo, "time": f"{r.year}-{r.month:02d}", "value": float(value)})
            return jsonstat(["unit", "indic_nrg", "geo", "time"], rows)
        if dataset == "prc_hicp_midx":
            block = prices[prices["country"] == geo]
            rows = [{"unit": "I15", "coicop": "CP045", "geo": geo, "time": f"{r.year}-{r.month:02d}", "value": float(r.price * 100)} for r in block.itertuples()]
            return jsonstat(["unit", "coicop", "geo", "time"], rows)
        column = {"demo_r_d2jan": "population", "demo_r_pjanind2": "share65", "ilc_li41": "poverty_rate"}.get(dataset)
        if dataset == "ilc_li02":
            block = annual[annual["region"].str[:2] == geo].groupby("year")["poverty_rate"].mean()
            rows = [{"geo": geo, "time": str(year), "value": float(value)} for year, value in block.items()]
            return jsonstat(["geo", "time"], rows)
        if dataset == "ilc_li41" and geo == missing_poverty:
            raise sources.SourceError("400")
        block = annual[annual["region"] == geo]
        rows = [{"geo": geo, "time": str(r.year), "value": float(getattr(r, column))} for r in block.itertuples()]
        return jsonstat(["geo", "time"], rows)

    return respond, calls, regions


def test_real_path_with_fallbacks(monkeypatch, tmp_path):
    synthetic = yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text(encoding="utf-8"))
    synthetic["data"]["synthetic"]["regions_per_country"] = 4
    raw = generate(synthetic)
    respond, calls, regions = fake_network(raw, "EL02", "IT03")
    monkeypatch.setattr(sources, "fetch_json", respond)
    config = yaml.safe_load((ROOT / "configs" / "europe_south.yaml").read_text(encoding="utf-8"))
    config["data"]["countries"] = ["EL", "IT", "ES", "PT"]
    config["data"]["nuts_versions"] = [2021]
    config["data"]["cache"] = str(tmp_path)
    config["model"]["iterations"] = 30
    config["evaluation"]["bootstrap_samples"] = 20
    result = pipeline.run(config)
    assert result.panel["region"].nunique() == 16
    log = pd.DataFrame(result.raw_log)
    assert "mean of 2 NUTS 3 series" in set(log.get("degree_days_source", pd.Series(dtype=str)).dropna())
    assert "ilc_li02 (national, used for the region)" in set(log.get("poverty_source", pd.Series(dtype=str)).dropna())
    assert np.isfinite(result.performance["overall_coverage"]).all()
    assert any("prc_hicp_midx" in url for url in calls)


def test_month_codes_in_both_formats():
    assert sources.parse_month("2022-07") == (2022, 7)
    assert sources.parse_month("2022M07") == (2022, 7)
