from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as m
from .guarantee import adaptive_quantiles
from .panel import FEATURE_LABELS, FEATURES, build_panel, coverage_table
from .policy import allocate, default_settings, floor_feasible, make_block, train
from .sources import load_real
from .synthetic import generate

STANDARD = "equal amount per resident at risk of poverty"
FORECAST = "proportional to forecast need"
FCRL = "FCRL-energy: guaranteed minimum and learned policy"
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


def _policy_allocation(frame, budget, theta, floor):
    block = make_block(frame, budget, floor.loc[frame.index].to_numpy())
    return pd.Series(allocate(theta, block), index=frame.index), block


def _allocations(panel, frame, budget, theta, floor):
    out = pd.DataFrame(index=frame.index)
    out["a_per_resident"] = _proportional(frame, frame["population"], budget)
    out["a_per_poor_resident"] = _proportional(frame, frame["vulnerable"], budget)
    out["a_last_year"] = _proportional(frame, _last_year(panel, frame), budget)
    out["a_forecast"] = _proportional(frame, frame["forecast"], budget)
    out["a_fcrl"], _ = _policy_allocation(frame, budget, theta, floor)
    out["a_hindsight"] = _proportional(frame, frame["need"], budget)
    return out


def _select(panel, budget, q, grid, settings, tolerance):
    train_frame = panel[panel["split"] == "train"]
    valid = panel[panel["split"] == "validation"]
    reference = m.summarise(valid.assign(a=_proportional(valid, valid["forecast"], budget)), "a", 0.0)
    rows = []
    fitted = {}
    for level in grid:
        floor = level * q * panel["forecast"]
        block = make_block(train_frame, budget, floor.loc[train_frame.index].to_numpy())
        theta, history = train(block, settings)
        fitted[level] = (theta, history)
        allocation, valid_block = _policy_allocation(valid, budget, theta, floor)
        summary = m.summarise(valid.assign(a=allocation), "a", level)
        feasible = float(np.mean(floor_feasible(valid_block)))
        passes = summary["overall_coverage"] >= reference["overall_coverage"] - tolerance
        rows.append(
            {
                "guaranteed_minimum": level,
                "months_floor_affordable": feasible,
                "validation_overall_coverage": summary["overall_coverage"],
                "validation_worst_off_coverage": summary["worst_off_coverage"],
                "validation_coverage_gap": summary["coverage_gap"],
                "validation_leakage_share": summary["leakage_share"],
                "validation_guarantee_attainment": summary["guarantee_attainment"],
                "reference_overall_coverage": reference["overall_coverage"],
                "passes_no_harm_check": bool(passes),
                **{f"weight_{name}": float(w) for name, w in zip(FEATURES, theta)},
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["passes_no_harm_check"]]
    if len(eligible):
        best = eligible.sort_values(["validation_worst_off_coverage", "guaranteed_minimum"], ascending=[False, True]).iloc[0]
        reason = "highest worst-off coverage on validation among settings that do not lower overall coverage relative to forecast-proportional allocation"
    else:
        best = table.sort_values("validation_overall_coverage", ascending=False).iloc[0]
        reason = "no setting kept overall coverage on validation; the setting with the highest overall coverage was kept"
    level = float(best["guaranteed_minimum"])
    table["selected"] = table["guaranteed_minimum"] == level
    theta, history = fitted[level]
    return table, {"level": level, "theta": theta, "reason": reason, "reference": reference}, pd.DataFrame(history)


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
    for baseline in ("a_per_poor_resident", "a_forecast"):
        for metric in [name for name in m.METRICS if name != "lowest_coverage"]:
            diff = draws["a_fcrl"][metric] - draws[baseline][metric]
            low, high = m.interval(diff)
            point = m.summarise(data, "a_fcrl", level)[metric] - m.summarise(data, baseline, level)[metric]
            differences.append(
                {
                    "comparison": f"{FCRL} minus {NAMES[baseline]}",
                    "quantity": m.LABELS[metric],
                    "metric": metric,
                    "difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "better_for_citizens": bool((point > 0) == m.HIGHER_IS_BETTER[metric]) if point != 0 else False,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(differences)


def _stress(panel, budget, theta, level, delta, gamma, scenarios):
    rows = []
    test_index = panel.index[panel["split"] == "test"]
    for scenario in scenarios:
        shocked = panel.copy()
        shocked.loc[test_index, "need"] = shocked.loc[test_index, "need"] * (1 + scenario["price_shock"])
        shocked_budget = {
            ordinal: value * (1 - scenario["budget_cut"]) if ordinal in set(shocked.loc[test_index, "ordinal"]) else value
            for ordinal, value in budget.items()
        }
        q, _ = adaptive_quantiles(shocked, shocked["need"], delta, gamma)
        floor = level * q * shocked["forecast"]
        frame = shocked.loc[test_index]
        allocations = _allocations(shocked, frame, shocked_budget, theta, floor)
        data = frame.join(allocations)
        for key in ("a_per_poor_resident", "a_forecast", "a_fcrl"):
            summary = m.summarise(data, key, level)
            rows.append({"scenario": scenario["name"], "strategy": NAMES[key], **summary})
    return pd.DataFrame(rows)


def _weights(theta):
    return pd.DataFrame(
        {
            "feature": FEATURES,
            "meaning": [FEATURE_LABELS[name] for name in FEATURES],
            "weight": theta,
            "effect_of_one_standard_deviation_on_claim": np.exp(theta),
        }
    )


def _statements(panel, frame, allocations, budget, theta, floor, level):
    latest_label = frame.loc[frame["ordinal"].idxmax(), "season_label"]
    block = frame[frame["season_label"] == latest_label]
    data = block.join(allocations)
    data["budget"] = data["ordinal"].map(budget)
    data["floor_part"] = floor.loc[block.index]
    floor_total = data.groupby("ordinal")["floor_part"].transform("sum")
    scale = np.minimum(1.0, data["budget"] / floor_total.where(floor_total > 0, np.nan)).fillna(1.0)
    data["floor_part"] = data["floor_part"] * scale
    data["forecast_share"] = data["forecast"] / data.groupby("ordinal")["forecast"].transform("sum")
    rows = []
    for region, part in data.groupby("region"):
        share = part["a_fcrl"].sum() / part["budget"].sum()
        floor_share = part["floor_part"].sum() / part["a_fcrl"].sum() if part["a_fcrl"].sum() > 0 else 0.0
        factors = {name: float(np.exp(np.mean(part[f"z_{name}"]) * w)) for name, w in zip(FEATURES, theta)}
        expected = part["a_fcrl"].sum() / part["forecast"].sum()
        realised = np.minimum(part["a_fcrl"], part["need"]).sum() / part["need"].sum()
        standard = np.minimum(part["a_per_poor_resident"], part["need"]).sum() / part["need"].sum()
        adjustments = sorted(factors.items(), key=lambda item: abs(np.log(item[1])), reverse=True)[:2]
        adjustment_text = "; ".join(f"{FEATURE_LABELS[name]} {'raised' if f >= 1 else 'lowered'} its claim by {abs(f - 1):.0%}" for name, f in adjustments)
        guarantee_text = f"a guaranteed minimum of {level:.0%} of its vulnerable households' need" if level > 0 else "no guaranteed minimum"
        rows.append(
            {
                "region": region,
                "name": part["name"].iloc[0],
                "country": part["country"].iloc[0],
                "season": latest_label,
                "share_of_budget": share,
                "share_of_forecast_need": part["forecast"].sum() / data["forecast"].sum(),
                "floor_part_of_allocation": floor_share,
                **{f"factor_{name}": value for name, value in factors.items()},
                "expected_coverage_of_forecast_need": expected,
                "realised_coverage": realised,
                "realised_coverage_under_standard_rule": standard,
                "statement": (
                    f"{part['name'].iloc[0]} received {share:.1%} of the support budget in {latest_label}. "
                    f"The allocation includes {guarantee_text}, which made up {floor_share:.0%} of what the region received. "
                    f"The rest followed its forecast need ({part['forecast'].sum() / data['forecast'].sum():.1%} of the total), where {adjustment_text}. "
                    f"Support covered {realised:.0%} of the energy need of residents at risk of poverty, against {standard:.0%} under an equal amount per resident at risk of poverty."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("realised_coverage").reset_index(drop=True)


def _regional(frame, allocations):
    data = frame.join(allocations)
    tables = []
    for key, name in STRATEGIES:
        table = m.season_coverage(data, key)[["region", "season_label", "coverage"]].rename(columns={"coverage": name})
        tables.append(table.set_index(["region", "season_label"]))
    regional = pd.concat(tables, axis=1).reset_index()
    names = frame.drop_duplicates("region").set_index("region")[["name", "country"]]
    return regional.join(names, on="region")[["region", "name", "country", "season_label", *[name for _, name in STRATEGIES]]]


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
    selection, selected, history = _select(panel, budget, q, list(guarantee["minimum_coverage_grid"]), settings, float(guarantee["overall_tolerance"]))
    level = selected["level"]
    theta = selected["theta"]
    floor = level * q * panel["forecast"]
    test = panel[panel["split"] == "test"]
    allocations = _allocations(panel, test, budget, theta, floor)
    evaluation = config.get("evaluation", {})
    performance, differences = _performance(test, allocations, level, int(evaluation.get("bootstrap_samples", 300)), int(evaluation.get("seed", 0)))
    scenarios = config.get("stress_tests", [
        {"name": "as observed", "price_shock": 0.0, "budget_cut": 0.0},
        {"name": "unexpected 30% rise in energy prices", "price_shock": 0.3, "budget_cut": 0.0},
        {"name": "20% cut in the support budget", "price_shock": 0.0, "budget_cut": 0.2},
        {"name": "price rise and budget cut together", "price_shock": 0.3, "budget_cut": 0.2},
    ])
    stress = _stress(panel, budget, theta, level, delta, gamma, scenarios)
    statements = _statements(panel, test, allocations, budget, theta, floor, level)
    regional = _regional(test, allocations)
    test_path = path[path["split"] == "test"]
    notes = {
        "budget_share": float(program["budget_share"]),
        "miss_rate": delta,
        "adaptation_rate": gamma,
        "guarantee_test_miss_rate": float(test_path["realised_miss_rate"].mean()) if len(test_path) else float("nan"),
        "validation_miss_rate": float(path.loc[path["split"] == "validation", "realised_miss_rate"].mean()) if len(path) else float("nan"),
    }
    return RunResult(
        config=config,
        panel=panel,
        raw_log=raw["log"],
        selection=selection,
        selected=selected,
        performance=performance,
        differences=differences,
        stress=stress,
        weights=_weights(theta),
        statements=statements,
        regional=regional,
        guarantee_path=path,
        history=history,
        coverage=coverage_table(panel),
        notes=notes,
    )
