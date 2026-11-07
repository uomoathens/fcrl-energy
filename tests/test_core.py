from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcrl_energy.guarantee import adaptive_quantiles, conformal_quantile
from fcrl_energy.metrics import summarise, weighted_gini
from fcrl_energy.panel import build_panel
from fcrl_energy.policy import allocate, default_settings, make_block, objective, water_fill
from fcrl_energy.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]


def config():
    return yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text(encoding="utf-8"))


def panel():
    cfg = config()
    return build_panel(generate(cfg), cfg), cfg


def test_conformal_quantile_finite_sample_rank():
    values = list(range(1, 11))
    assert conformal_quantile(values, 0.9) == 10
    assert conformal_quantile(values, 0.5) == 6
    assert conformal_quantile(values, 0.0) == 0.0


def test_panel_splits_and_climatology_use_training_only():
    frame, cfg = panel()
    assert set(frame["split"]) == {"train", "validation", "test"}
    train = frame[frame["split"] == "train"]
    clim = train.groupby(["region", "month"])["degree"].mean()
    later = frame[frame["split"] != "train"].set_index(["region", "month"])["clim"]
    joined = later.to_frame().join(clim.rename("expected"))
    assert np.allclose(joined["clim"], joined["expected"])
    assert (frame["forecast"] > 0).all() and (frame["need"] > 0).all()
    sizes = frame.groupby(["region", "season_label"]).size()
    assert set(sizes.unique()) <= {4, 5}


def test_allocation_spends_exactly_the_budget_and_respects_floor():
    frame, _ = panel()
    block_frame = frame[frame["split"] == "validation"]
    budget = (block_frame.groupby("ordinal")["forecast"].sum() * 0.6).to_dict()
    floor = 0.3 * block_frame["forecast"].to_numpy()
    block = make_block(block_frame, budget, floor)
    theta = np.array([0.2, -0.1, 0.3, 0.0, 0.1])
    allocation = allocate(theta, block)
    spent = np.bincount(block.group, weights=allocation)
    assert np.allclose(spent, block.budget)
    assert (allocation >= floor - 1e-9).all()


def test_floor_is_scaled_when_budget_cannot_afford_it():
    frame, _ = panel()
    block_frame = frame[frame["split"] == "validation"]
    budget = (block_frame.groupby("ordinal")["forecast"].sum() * 0.2).to_dict()
    block = make_block(block_frame, budget, 0.5 * block_frame["forecast"].to_numpy())
    allocation = allocate(np.zeros(5), block)
    assert np.allclose(np.bincount(block.group, weights=allocation), block.budget)


def test_objective_prefers_equal_coverage():
    frame, cfg = panel()
    block_frame = frame[frame["split"] == "validation"]
    budget = (block_frame.groupby("ordinal")["need"].sum() * 0.6).to_dict()
    block = make_block(block_frame, budget, np.zeros(len(block_frame)))
    settings = default_settings(cfg["model"])
    equal = block_frame.assign(forecast=block_frame["need"])
    equal_block = make_block(equal, budget, np.zeros(len(equal)))
    assert objective(np.zeros(5), equal_block, settings) > objective(np.zeros(5), block, settings)


def test_adaptive_guarantee_tracks_target_miss_rate():
    frame, _ = panel()
    _, path = adaptive_quantiles(frame, frame["need"], 0.1, 0.05)
    assert abs(path["realised_miss_rate"].mean() - 0.1) < 0.06
    shocked = frame["need"].where(frame["split"] != "test", frame["need"] * 1.3)
    _, shocked_path = adaptive_quantiles(frame, shocked, 0.1, 0.05)
    test = shocked_path[shocked_path["split"] == "test"]
    assert test["need_to_forecast_bound"].iloc[-1] > test["need_to_forecast_bound"].iloc[0]


def test_metrics_on_known_values():
    frame = pd.DataFrame(
        {
            "region": ["A", "B", "A", "B"],
            "season_label": ["s", "s", "s", "s"],
            "need": [10.0, 10.0, 10.0, 10.0],
            "vulnerable": [1.0, 1.0, 1.0, 1.0],
            "a": [5.0, 15.0, 5.0, 5.0],
        }
    )
    summary = summarise(frame, "a", 0.5)
    assert np.isclose(summary["overall_coverage"], 25 / 40)
    assert np.isclose(summary["leakage_share"], 5 / 30)
    assert np.isclose(summary["worst_off_coverage"], 0.5)
    assert np.isclose(summary["guarantee_attainment"], 1.0)
    assert np.isclose(weighted_gini([1, 1, 1], [1, 2, 3]), 0.0)


def test_reserve_raises_the_lowest_coverage_first():
    allocation = np.array([1.0, 5.0, 2.0])
    need = np.array([10.0, 10.0, 4.0])
    group = np.zeros(3, dtype=int)
    top_up, unspent = water_fill(allocation, need, group, 1, np.array([4.0]))
    assert np.allclose(top_up, [4.0, 0.0, 0.0])
    assert np.allclose(unspent, [0.0])
    top_up, unspent = water_fill(allocation, need, group, 1, np.array([100.0]))
    assert np.allclose(allocation + top_up, need)
    assert np.allclose(unspent, [84.0])


def test_reserve_never_lowers_any_region():
    frame, cfg = panel()
    from fcrl_energy.guarantee import adaptive_quantiles
    from fcrl_energy.pipeline import fcrl_allocation
    q, _ = adaptive_quantiles(frame, frame["need"], 0.1, 0.05)
    part = frame[frame["split"] == "test"]
    budget = (frame.groupby("ordinal")["forecast"].sum() * 0.6).to_dict()
    theta = np.zeros(5)
    plan = {s: {"theta": theta, "level": 0.2, "reserve": 0.1} for s in ("winter", "summer")}
    parts, _ = fcrl_allocation(part, budget, q, plan)
    spent = parts["final"].groupby(part["ordinal"]).sum()
    assert (spent <= part.groupby("ordinal")["ordinal"].first().map(budget) * (1 + 1e-9)).all()
    assert (parts["top_up"] >= -1e-12).all()
    assert (parts["final"] >= parts["floor"] - 1e-9).all()
