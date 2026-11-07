import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .jsonstat import to_frame

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
NUTS_POINTS = "https://raw.githubusercontent.com/eurostat/Nuts2json/master/pub/v2/{version}/4326/nutspt_{level}.json"
USER_AGENT = "fcrl-energy research software (https://github.com/uomoathens/fcrl-energy)"


class SourceError(RuntimeError):
    pass


def _cache_path(cache, key):
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in key)
    return Path(cache) / f"{safe[:180]}.json"


def fetch_json(url, cache, key, retries=4, pause=1.0):
    path = _cache_path(cache, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(pause)
            return payload
        except urllib.error.HTTPError as error:
            last = error
            if error.code in (400, 404):
                raise SourceError(f"{error.code} for {url}") from error
            time.sleep(pause * (2 ** attempt) * 5)
        except (urllib.error.URLError, TimeoutError) as error:
            last = error
            time.sleep(pause * (2 ** attempt) * 5)
    raise SourceError(f"failed after {retries} attempts: {url}: {last}")


def eurostat(code, filters, cache):
    query = [("format", "JSON"), ("lang", "EN")]
    for name, value in filters.items():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            query.append((name, str(item)))
    url = EUROSTAT + code + "?" + urllib.parse.urlencode(query)
    key = "eurostat_" + code + "_" + "_".join(f"{k}-{v}" for k, v in query[2:])
    return to_frame(fetch_json(url, cache, key))


def nuts_points(level, countries, cache, version=2021):
    payload = fetch_json(NUTS_POINTS.format(level=level, version=version), cache, f"nuts_points_{version}_{level}")
    rows = []
    for feature in payload["features"]:
        code = feature["properties"]["id"]
        if code[:2] in countries:
            rows.append({"region": code, "name": feature["properties"].get("na", code), "country": code[:2]})
    return pd.DataFrame(rows, columns=["region", "name", "country"])


def parse_month(code):
    text = str(code)
    if "M" in text:
        year, month = text.split("M")
    else:
        year, month = text.split("-")[:2]
    return int(year), int(month)


def _monthly(frame, value_name):
    if frame.empty:
        return pd.DataFrame(columns=["year", "month", value_name])
    parsed = frame["time"].map(parse_month)
    return pd.DataFrame({"year": [p[0] for p in parsed], "month": [p[1] for p in parsed], value_name: frame["value"].values})


def _since_variants(since):
    year, month = parse_month(since)
    return [f"{year}-{month:02d}", f"{year}M{month:02d}"]


def monthly_series(code, filters, since, cache):
    last = None
    for variant in _since_variants(since):
        try:
            return eurostat(code, {**filters, "sinceTimePeriod": variant}, cache)
        except SourceError as error:
            last = error
    raise last


def degree_days(region, children, since, cache, log):
    def fetch(geo):
        try:
            frame = monthly_series("nrg_chddr2_m", {"unit": "NR", "indic_nrg": ["HDD", "CDD"], "geo": geo}, since, cache)
        except SourceError:
            return pd.DataFrame()
        if frame.empty:
            return frame
        parts = []
        for indicator, column in (("HDD", "hdd"), ("CDD", "cdd")):
            block = _monthly(frame[frame["indic_nrg"] == indicator], column)
            parts.append(block.set_index(["year", "month"]))
        if len(parts[0]) == 0 or len(parts[1]) == 0:
            return pd.DataFrame()
        return parts[0].join(parts[1], how="inner").reset_index()

    direct = fetch(region)
    if len(direct) > 0:
        log.append({"region": region, "degree_days_source": "NUTS 2 series"})
        return direct.assign(region=region)
    pieces = [fetch(child) for child in children]
    pieces = [piece for piece in pieces if len(piece) > 0]
    if pieces and len(pieces) == len(children):
        merged = pd.concat(pieces)
        counts = merged.groupby(["year", "month"]).size()
        complete = counts[counts == len(pieces)].index
        averaged = merged.groupby(["year", "month"])[["hdd", "cdd"]].mean().loc[complete].reset_index()
        log.append({"region": region, "degree_days_source": f"mean of {len(pieces)} NUTS 3 series"})
        return averaged.assign(region=region)
    log.append({"region": region, "degree_days_source": "unavailable"})
    return pd.DataFrame(columns=["year", "month", "hdd", "cdd", "region"])


def annual_series(code, filters, geo, cache):
    try:
        frame = eurostat(code, {**filters, "geo": geo}, cache)
    except SourceError:
        return pd.Series(dtype=float)
    if frame.empty:
        return pd.Series(dtype=float)
    return pd.Series(frame["value"].values, index=frame["time"].astype(int).values, dtype=float).sort_index()


def price_index(country, since, cache, coicop="CP045"):
    try:
        frame = monthly_series("prc_hicp_midx", {"unit": "I15", "coicop": coicop, "geo": country}, since, cache)
    except SourceError:
        return pd.DataFrame(columns=["country", "year", "month", "price"])
    block = _monthly(frame, "price")
    block["price"] = block["price"] / 100.0
    return block.assign(country=country)


def _annual_block(region, cache, log):
    population = annual_series("demo_r_d2jan", {"sex": "T", "age": "TOTAL", "unit": "NR"}, region, cache)
    share65 = annual_series("demo_r_pjanind2", {"indic_de": "PC_Y65_MAX"}, region, cache)
    poverty = annual_series("ilc_li41", {"unit": "PC"}, region, cache)
    source = "ilc_li41 (regional)"
    if poverty.empty:
        poverty = annual_series("ilc_li02", {"indic_il": "LI_R_MD60", "unit": "PC", "age": "TOTAL", "sex": "T"}, region[:2], cache)
        source = "ilc_li02 (national, used for the region)"
    log.append({"region": region, "poverty_source": source, "poverty_years": len(poverty), "population_years": len(population), "share65_years": len(share65)})
    years = sorted(set(population.index) | set(share65.index) | set(poverty.index))
    if not years:
        return pd.DataFrame(columns=["region", "year", "population", "share65", "poverty_rate"])
    frame = pd.DataFrame({"year": years})
    frame["population"] = frame["year"].map(population)
    frame["share65"] = frame["year"].map(share65)
    frame["poverty_rate"] = frame["year"].map(poverty)
    return frame.assign(region=region)


def _complete(degree, annual, required):
    if degree.empty:
        return set()
    keys = degree.groupby("region").apply(lambda block: set(zip(block["year"], block["month"])))
    usable = set()
    for region, present in keys.items():
        if not set(required) <= present:
            continue
        block = annual[annual["region"] == region]
        if block["population"].notna().any() and block["poverty_rate"].notna().any():
            usable.add(region)
    return usable


def _country(country, versions, exclude, since, required, cache, log):
    best = None
    seen = []
    for version in versions:
        regions = nuts_points(2, [country], cache, version)
        regions = regions[~regions["region"].isin(exclude)].reset_index(drop=True)
        codes = tuple(sorted(regions["region"]))
        if not codes or codes in seen:
            continue
        seen.append(codes)
        level3 = nuts_points(3, [country], cache, version)
        attempt = []
        degree_frames = []
        annual_frames = []
        for code in codes:
            children = sorted(level3.loc[level3["region"].str[:4] == code, "region"])
            degree_frames.append(degree_days(code, children, since, cache, attempt))
            annual_frames.append(_annual_block(code, cache, attempt))
        degree = pd.concat(degree_frames, ignore_index=True)
        annual = pd.concat(annual_frames, ignore_index=True)
        usable = _complete(degree, annual, required)
        log.append({"country": country, "nuts_version": version, "regions": len(codes), "regions_complete": len(usable)})
        if best is None or len(usable) >= best[0]:
            best = (len(usable), version, regions, degree, annual, attempt, usable)
    if best is None or best[0] == 0:
        return None
    _, version, regions, degree, annual, attempt, usable = best
    log.extend(attempt)
    log.append({"country": country, "nuts_version_selected": version})
    regions = regions[regions["region"].isin(usable)].assign(nuts_version=version)
    return regions, degree[degree["region"].isin(usable)], annual[annual["region"].isin(usable)]


def required_months(config):
    start_year, start_month = parse_month(config["data"]["program_start"])
    end_year, end_month = parse_month(config["data"]["program_end"])
    program = set(config["program"]["winter_months"]) | set(config["program"]["summer_months"])
    months = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        if month in program:
            months.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def load_real(config, cache):
    data = config["data"]
    countries = list(data["countries"])
    versions = list(data.get("nuts_versions", [2021]))
    exclude = set(data.get("exclude_regions", []))
    required = required_months(config)
    log = []
    found = [_country(country, versions, exclude, data["since"], required, cache, log) for country in countries]
    found = [item for item in found if item is not None]
    if not found:
        raise SourceError("no region had complete degree-day, population and poverty data")
    regions = pd.concat([item[0] for item in found], ignore_index=True)
    degree = pd.concat([item[1] for item in found], ignore_index=True)
    annual = pd.concat([item[2] for item in found], ignore_index=True)
    prices = pd.concat([price_index(country, data["since"], cache) for country in regions["country"].unique()], ignore_index=True)
    log.append({"price_source": "prc_hicp_midx, CP045 electricity, gas and other fuels, 2015 = 100", "countries_with_prices": int(prices["country"].nunique())})
    regions = regions[regions["country"].isin(prices["country"].unique())].reset_index(drop=True)
    return {"regions": regions[["region", "name", "country", "nuts_version"]], "degree": degree, "annual": annual, "prices": prices, "log": log}
