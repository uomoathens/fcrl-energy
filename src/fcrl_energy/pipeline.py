from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as m
from .guarantee import adaptive_quantiles
from .panel import FEATURE_LABELS, FEATURES, build_panel, coverage_table
from .policy import allocate, default_settings, make_block, train, water_fill
from .sources import load_real
from .synthetic import generate

STANDARD = "equal amount per resident at risk of poverty"
FORECAST = "proportional to forecast need"
FCRL = "FCRL-energy: guaranteed minimum, learned policy and emergency reserve"
HINDSIGHT = "hindsight benchmark (needs future data, not implementable)"
STRATEGIES = [
    ("a_per_resident", "equal amount per resident"),
    ("a_per_poor_resident", STANDARD),
    ("a_last_year", "proportional to last year's need"),
    ("a_forecast", FORECAST),
    ("a_fcrl", FCRL),
    ("a_hindsight", HINDSIGHT),
]
NAMES = dict(STRATEGIES)
SEASONS = ("winter", "summer")


@dataclass
class RunResult:
    config: dict
    panel: pd.DataFrame
    raw_log: list
    selection: pd.DataFrame
    selected: dict
    performance: pd.DataFrame
    differences: pd.DataFrame
    stress: pd.DataFrame
    weights: pd.DataFrame
    statements: pd.DataFrame
    regional: pd.DataFrame
    guarantee_path: pd.DataFrame
    history: pd.DataFrame
    coverage: pd.DataFrame
    reserve: pd.DataFrame
    notes: dict = field(default_factory=dict)


def _budget(panel, share):
    return (panel.groupby("ordinal")["forecast"].sum() * share).to_dict()


def _proportional(frame, weights, budget):
    totals = weights.groupby(frame["ordinal"]).transform("sum")
    return frame["ordinal"].map(budget) * weights / totals


def _last_year(panel, frame):
    lookup = panel.set_index(["region", "ordinal"])["need"]
    keys = list(zip(frame["region"], frame["ordinal"] - 12))
    values = pd.Series([lookup.get(key, np.nan) for key in keys], index=frame.index)
    return values.fillna(frame["forecast"])


def fcrl_allocation(frame, budget, q, plan):
    columns = {"final": pd.Series(0.0, index=frame.index), "ex_ante": pd.Series(0.0, index=frame.index), "top_up": pd.Series(0.0, index=frame.index), "floor": pd.Series(0.0, index=frame.index)}
    unspent = {}
    for season, part in frame.groupby("season"):
        setting = plan[season]
        reserve_share = setting["reserve"]
        ex_ante_budget = {ordinal: value * (1 - reserve_share) for ordinal, value in budget.items()}
        floor = (setting["level"] * q.loc[part.index] * part["forecast"]).to_numpy()
        block = make_block(part, ex_ante_budget, floor)
        ex_ante = allocate(setting["theta"], block)
        floor_total = np.bincount(block.group, weights=floor, minlength=block.groups)
        scale = np.where(floor_total > block.budget, block.budget / np.maximum(floor_total, 1e-12), 1.0)
        reserve = block.budget / max(1 - reserve_share, 1e-12) * reserve_share
        top_up, left = water_fill(ex_ante, block.need, block.group, block.groups, reserve)
        ordinals = np.unique(part["ordinal"].to_numpy())
        for ordinal, value in zip(ordinals, left):
            unspent[ordinal] = float(value)
        columns["ex_ante"].loc[part.index] = ex_ante
        columns["top_up"].loc[part.index] = top_up
        columns["final"].loc[part.index] = ex_ante + top_up
        columns["floor"].loc[part.index] = floor * scale[block.group]
    return columns, unspent


def _allocations(panel, frame, budget, q, plan):
    out = pd.DataFrame(index=frame.index)
    out["a_per_resident"] = _proportional(frame, frame["population"], budget)
    out["a_per_poor_resident"] = _proportional(frame, frame["vulnerable"], budget)
    out["a_last_year"] = _proportional(frame, _last_year(panel, frame), budget)
    out["a_forecast"] = _proportional(frame, frame["forecast"], budget)
    parts, unspent = fcrl_allocation(frame, budget, q, plan)
    out["a_fcrl"] = parts["final"]
    out["a_hindsight"] = _proportional(frame, frame["need"], budget)
    return out, parts, unspent


def _train_policies(panel, budget, settings):
    thetas = {}
    histories = []
    for season in SEASONS:
        part = panel[(panel["split"] == "train") & (panel["season"] == season)]
        if part.empty:
            thetas[season] = np.zeros(len(FEATURES))
            continue
        block = make_block(part, budget, np.zeros(len(part)))
        theta, history = train(block, settings)
        thetas[season] = theta
        histories.append(pd.DataFrame(history).assign(season=season))
    return thetas, pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()


def _evaluate(part, budget, q, season, theta, level, reserve):
    plan = {season: {"theta": theta, "level": level, "reserve": reserve}}
    parts, unspent = fcrl_allocation(part, budget, q, plan)
    summary = m.summarise(part.assign(a=parts["final"]), "a", level)
    reserve_total = sum(budget[o] * reserve for o in part["ordinal"].unique())
    summary["reserve_spent_share"] = 1 - sum(unspent.values()) / reserve_total if reserve_total > 0 else float("nan")
    summary["months_floor_affordable"] = float(np.mean([(level * q.loc[g.index] * g["forecast"]).sum() <= budget[o] * (1 - reserve) * (1 + 1e-9) for o, g in part.groupby("ordinal")]))
    return summary


def _select(panel, budget, q, thetas, guarantee):
    levels = list(guarantee["minimum_coverage_grid"])
    reserves = list(guarantee["reserve_grid"])
    tolerance = float(guarantee["overall_tolerance"])
    worst_tolerance = float(guarantee["worst_off_tolerance"])
    rows = []
    plan = {}
    for season in SEASONS:
        part = panel[(panel["split"] == "validation") & (panel["season"] == season)]
        if part.empty:
            plan[season] = {"theta": thetas[season], "level": 0.0, "reserve": 0.0, "reason": "no validation data for this season"}
            continue
        reference = m.summarise(part.assign(a=_proportional(part, part["forecast"], budget)), "a", 0.0)
        grid = {}
        for level in levels:
            for reserve in reserves:
                grid[(level, reserve)] = _evaluate(part, budget, q, season, thetas[season], level, reserve)
        no_harm = {key: value["overall_coverage"] >= reference["overall_coverage"] - tolerance for key, value in grid.items()}
        reserve_options = [r for r in reserves if no_harm[(0.0, r)]] or [0.0]
        chosen_reserve = max(reserve_options, key=lambda r: (round(grid[(0.0, r)]["worst_off_coverage"], 6), -r))
        anchor = grid[(0.0, chosen_reserve)]["worst_off_coverage"]
        level_options = [l for l in levels if no_harm[(l, chosen_reserve)] and grid[(l, chosen_reserve)]["worst_off_coverage"] >= anchor - worst_tolerance and grid[(l, chosen_reserve)]["months_floor_affordable"] >= 1.0]
        chosen_level = max(level_options) if level_options else 0.0
        reason = (
            f"reserve: the share that maximised coverage of the worst-off tenth of regions on validation without lowering total coverage; "
            f"guaranteed minimum: the largest that the budget could always afford and that kept both total coverage and worst-off coverage within tolerance"
        )
        plan[season] = {"theta": thetas[season], "level": float(chosen_level), "reserve": float(chosen_reserve), "reason": reason}
        for (level, reserve), summary in grid.items():
            rows.append(
                {
                    "season": season,
                    "guaranteed_minimum": level,
                    "emergency_reserve_share": reserve,
                    "months_floor_affordable": summary["months_floor_affordable"],
                    "validation_overall_coverage": summary["overall_coverage"],
                    "validation_worst_off_coverage": summary["worst_off_coverage"],
                    "validation_lowest_coverage": summary["lowest_coverage"],
                    "validation_coverage_gap": summary["coverage_gap"],
                    "validation_guarantee_attainment": summary["guarantee_attainment"],
                    "validation_reserve_spent_share": summary["reserve_spent_share"],
                    "reference_overall_coverage": reference["overall_coverage"],
                    "passes_no_harm_check": bool(no_harm[(level, reserve)]),
                    "selected": bool(level == chosen_level and reserve == chosen_reserve),
                }
            )
    return pd.DataFrame(rows), plan


def _performance(frame, allocations, level, samples, seed):
    data = frame.join(allocations)
    columns = [key for key, _ in STRATEGIES]
    draws = m.bootstrap(data, columns, level, samples, seed)
    rows = []
    for key, name in STRATEGIES:
        point = m.summarise(data, key, level)
        row = {"strategy": name}
        for metric in m.METRICS:
            low, high = m.interval(draws[key][metric]) if metric != "lowest_coverage" else (float("nan"), float("nan"))
            row[metric] = point[metric]
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    differences = []
    fcrl_point = m.summarise(data, "a_fcrl", level)
    for baseline in ("a_per_poor_resident", "a_forecast"):
        baseline_point = m.summarise(data, baseline, level)
        for metric in [name for name in m.METRICS if name not in ("lowest_coverage", "guarantee_attainment")]:
            diff = draws["a_fcrl"][metric] - draws[baseline][metric]
            low, high = m.interval(diff)
            point = fcrl_point[metric] - baseline_point[metric]
            differences.append(
                {
                    "comparison": f"{FCRL} minus {NAMES[baseline]}",
                    "quantity": m.LABELS[metric],
                    "metric": metric,
                    "difference": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def _guarantee_level(plan):
    levels = [plan[s]["level"] for s in SEASONS if s in plan]
    return min(levels) if levels else 0.0


def _attainment(frame, allocation, plan):
    level = frame["season"].map({s: plan[s]["level"] for s in plan})
    covered = np.minimum(allocation, frame["need"]) / frame["need"]
    mask = level > 0
    return float((covered[mask] >= level[mask] - 1e-9).mean()) if mask.any() else float("nan")


def _stress(panel, budget, plan, delta, gamma, scenarios):
    rows = []
    test_index = panel.index[panel["split"] == "test"]
    test_ordinals = set(panel.loc[test_index, "ordinal"])
    for scenario in scenarios:
        shocked = panel.copy()
        shocked.loc[test_index, "need"] = shocked.loc[test_index, "need"] * (1 + scenario["price_shock"])
        shocked_budget = {o: v * (1 - scenario["budget_cut"]) if o in test_ordinals else v for o, v in budget.items()}
        q, _ = adaptive_quantiles(shocked, shocked["need"], delta, gamma)
        frame = shocked.loc[test_index]
        allocations, _, _ = _allocations(shocked, frame, shocked_budget, q, plan)
        data = frame.join(allocations)
        for key in ("a_per_poor_resident", "a_forecast", "a_fcrl"):
            summary = m.summarise(data, key, 0.0)
            summary["guarantee_attainment"] = _attainment(frame, data[key], plan)
            rows.append({"scenario": scenario["name"], "strategy": NAMES[key], **summary})
    return pd.DataFrame(rows)


def _weights(thetas):
    rows = []
    for season in SEASONS:
        for name, weight in zip(FEATURES, thetas[season]):
            rows.append({"season": season, "feature": name, "meaning": FEATURE_LABELS[name], "weight": float(weight), "effect_of_one_standard_deviation_on_claim": float(np.exp(weight))})
    return pd.DataFrame(rows)


def _statements(frame, allocations, parts, budget, plan):
    latest = {}
    for season in SEASONS:
        block = frame[frame["season"] == season]
        if len(block):
            latest[season] = block.loc[block["ordinal"].idxmax(), "season_label"]
    rows = []
    for season, label in latest.items():
        block = frame[frame["season_label"] == label]
        data = block.join(allocations)
        data["budget"] = data["ordinal"].map(budget)
        for key in ("ex_ante", "top_up", "floor"):
            data[key] = parts[key].loc[block.index]
        setting = plan[season]
        for region, part in data.groupby("region"):
            received = part["a_fcrl"].sum()
            share = received / part["budget"].sum()
            floor_share = part["floor"].sum() / received if received > 0 else 0.0
            top_share = part["top_up"].sum() / received if received > 0 else 0.0
            factors = {name: float(np.exp(np.mean(part[f"z_{name}"]) * w)) for name, w in zip(FEATURES, setting["theta"])}
            realised = np.minimum(part["a_fcrl"], part["need"]).sum() / part["need"].sum()
            standard = np.minimum(part["a_per_poor_resident"], part["need"]).sum() / part["need"].sum()
            adjustments = [item for item in sorted(factors.items(), key=lambda item: abs(np.log(item[1])), reverse=True)[:2] if abs(item[1] - 1) >= 0.005]
            parts_text = []
            if setting["level"] > 0:
                parts_text.append(f"{floor_share:.0%} came from the guaranteed minimum of {setting['level']:.0%} of need")
            if top_share > 0.0005:
                parts_text.append(f"{top_share:.0%} came from the emergency reserve, paid after the month to lift the regions with the lowest coverage to a common level")
            elif setting["reserve"] > 0:
                parts_text.append("no emergency top-up was needed")
            adjustment_text = "; ".join(f"{FEATURE_LABELS[name]} {'raised' if f >= 1 else 'lowered'} its claim by {abs(f - 1):.0%}" for name, f in adjustments) or "no indicator changed its claim by more than 0.5%"
            sentence = (
                f"{part['name'].iloc[0]} received {share:.1%} of the support budget in {label}. "
                + (f"Of this, {' and '.join(parts_text)}. " if parts_text else "")
                + f"The rest followed its forecast need ({part['forecast'].sum() / data['forecast'].sum():.1%} of the total), where {adjustment_text}. "
                + f"Support covered {realised:.0%} of the energy need of residents at risk of poverty, against {standard:.0%} under an equal amount per resident at risk of poverty."
            )
            rows.append(
                {
                    "region": region,
                    "name": part["name"].iloc[0],
                    "country": part["country"].iloc[0],
                    "season": label,
                    "share_of_budget": share,
                    "share_of_forecast_need": part["forecast"].sum() / data["forecast"].sum(),
                    "guaranteed_part": floor_share,
                    "emergency_top_up_part": top_share,
                    **{f"factor_{name}": value for name, value in factors.items()},
                    "realised_coverage": realised,
                    "realised_coverage_under_standard_rule": standard,
                    "statement": sentence,
                }
            )
    return pd.DataFrame(rows).sort_values(["season", "realised_coverage"]).reset_index(drop=True)


def _regional(frame, allocations):
    data = frame.join(allocations)
    tables = []
    for key, name in STRATEGIES:
        table = m.season_coverage(data, key)[["region", "season_label", "coverage"]].rename(columns={"coverage": name})
        tables.append(table.set_index(["region", "season_label"]))
    regional = pd.concat(tables, axis=1).reset_index()
    names = frame.drop_duplicates("region").set_index("region")[["name", "country"]]
    return regional.join(names, on="region")[["region", "name", "country", "season_label", *[name for _, name in STRATEGIES]]]


def _reserve_table(frame, parts, unspent, budget, plan):
    rows = []
    for (label, season), block in frame.groupby(["season_label", "season"]):
        reserve = sum(budget[o] * plan[season]["reserve"] for o in block["ordinal"].unique())
        left = sum(unspent.get(o, 0.0) for o in block["ordinal"].unique())
        top = parts["top_up"].loc[block.index]
        rows.append(
            {
                "season": label,
                "reserve_share_of_budget": plan[season]["reserve"],
                "reserve_spent_share": 1 - left / reserve if reserve > 0 else float("nan"),
                "regions_receiving_top_up": int((top.groupby(block["region"]).sum() > 0).sum()),
                "regions": int(block["region"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def run(config):
    data = config["data"]
    if data["source"] == "synthetic":
        raw = generate(config)
    else:
        cache = Path(data.get("cache", ".cache/fcrl"))
        raw = load_real(config, cache)
    panel = build_panel(raw, config)
    program = config["program"]
    guarantee = config["guarantee"]
    budget = _budget(panel, float(program["budget_share"]))
    delta = float(guarantee["miss_rate"])
    gamma = float(guarantee["adaptation_rate"])
    q, path = adaptive_quantiles(panel, panel["need"], delta, gamma)
    settings = default_settings(config["model"])
    thetas, history = _train_policies(panel, budget, settings)
    selection, plan = _select(panel, budget, q, thetas, guarantee)
    test = panel[panel["split"] == "test"]
    allocations, parts, unspent = _allocations(panel, test, budget, q, plan)
    evaluation = config.get("evaluation", {})
    level = _guarantee_level(plan)
    performance, differences = _performance(test, allocations, 0.0, int(evaluation.get("bootstrap_samples", 300)), int(evaluation.get("seed", 0)))
    attainment = {key: _attainment(test, allocations[key], plan) for key, _ in STRATEGIES}
    performance["guarantee_attainment"] = [attainment[key] for key, _ in STRATEGIES]
    scenarios = config.get("stress_tests", [
        {"name": "as observed", "price_shock": 0.0, "budget_cut": 0.0},
        {"name": "unexpected 30% rise in energy prices", "price_shock": 0.3, "budget_cut": 0.0},
        {"name": "20% cut in the support budget", "price_shock": 0.0, "budget_cut": 0.2},
        {"name": "price rise and budget cut together", "price_shock": 0.3, "budget_cut": 0.2},
    ])
    stress = _stress(panel, budget, plan, delta, gamma, scenarios)
    test_path = path[path["split"] == "test"]
    notes = {
        "budget_share": float(program["budget_share"]),
        "miss_rate": delta,
        "adaptation_rate": gamma,
        "guarantee_test_miss_rate": float(test_path["realised_miss_rate"].mean()) if len(test_path) else float("nan"),
        "validation_miss_rate": float(path.loc[path["split"] == "validation", "realised_miss_rate"].mean()) if len(path) else float("nan"),
        "minimum_level": level,
    }
    return RunResult(
        config=config,
        panel=panel,
        raw_log=raw["log"],
        selection=selection,
        selected=plan,
        performance=performance,
        differences=differences,
        stress=stress,
        weights=_weights(thetas),
        statements=_statements(test, allocations, parts, budget, plan),
        regional=_regional(test, allocations),
        guarantee_path=path,
        history=history,
        coverage=coverage_table(panel),
        reserve=_reserve_table(test, parts, unspent, budget, plan),
        notes=notes,
    )
