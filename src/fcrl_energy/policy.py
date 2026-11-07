from dataclasses import dataclass

import numpy as np

from .panel import FEATURES


@dataclass
class Block:
    group: np.ndarray
    forecast: np.ndarray
    need: np.ndarray
    vulnerable: np.ndarray
    features: np.ndarray
    budget: np.ndarray
    floor: np.ndarray

    @property
    def groups(self):
        return int(self.group.max()) + 1 if len(self.group) else 0


def make_block(frame, budget_by_ordinal, floor):
    ordinals = frame["ordinal"].to_numpy()
    unique, group = np.unique(ordinals, return_inverse=True)
    budget = np.array([budget_by_ordinal[o] for o in unique], dtype=float)
    return Block(
        group=group,
        forecast=frame["forecast"].to_numpy(dtype=float),
        need=frame["need"].to_numpy(dtype=float),
        vulnerable=frame["vulnerable"].to_numpy(dtype=float),
        features=frame[[f"z_{name}" for name in FEATURES]].to_numpy(dtype=float),
        budget=budget,
        floor=np.asarray(floor, dtype=float),
    )


def _sum(values, block):
    return np.bincount(block.group, weights=values, minlength=block.groups)


def allocate(theta, block, floor=None):
    floor = block.floor if floor is None else floor
    claims = block.forecast * np.exp(np.clip(block.features @ theta, -6, 6))
    floor_total = _sum(floor, block)
    scale = np.where(floor_total > block.budget, block.budget / np.maximum(floor_total, 1e-12), 1.0)
    floor_used = floor * scale[block.group]
    rest = block.budget - floor_total * scale
    claim_total = _sum(claims, block)
    return floor_used + rest[block.group] * claims / claim_total[block.group]


def floor_feasible(block, floor=None):
    floor = block.floor if floor is None else floor
    return _sum(floor, block) <= block.budget * (1 + 1e-9)


def _softplus(x):
    return np.logaddexp(0.0, x)


def _group_logsumexp(values, block, sharpness):
    peak = np.full(block.groups, -np.inf)
    np.maximum.at(peak, block.group, values)
    total = _sum(np.exp(sharpness * (values - peak[block.group])), block)
    return peak + np.log(total) / sharpness


def objective(theta, block, settings):
    k = settings["smoothing"]
    a = allocate(theta, block)
    x = a / block.need
    covered = 1.0 - _softplus(k * (1.0 - x)) / k
    eta = settings["inequality_aversion"]
    c = np.maximum(covered, 0.0) + settings["epsilon"]
    utility = np.log(c) if abs(eta - 1.0) < 1e-9 else (c ** (1 - eta) - 1) / (1 - eta)
    welfare = _sum(block.vulnerable * utility, block) / _sum(block.vulnerable, block)
    leak = _sum(block.need * _softplus(k * (x - 1.0)) / k, block) / block.budget
    spread = _group_logsumexp(covered, block, settings["gap_sharpness"]) + _group_logsumexp(-covered, block, settings["gap_sharpness"])
    excess = np.maximum(spread - settings["max_gap"], 0.0)
    value = np.mean(welfare - settings["leakage_weight"] * leak) - settings["gap_penalty"] * np.mean(excess ** 2)
    return float(value - settings["ridge"] * np.sum(theta ** 2))


def train(block, settings):
    theta = np.zeros(block.features.shape[1])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    rate = settings["learning_rate"]
    step = settings["finite_difference"]
    history = []
    for iteration in range(1, settings["iterations"] + 1):
        gradient = np.zeros_like(theta)
        for j in range(len(theta)):
            shift = np.zeros_like(theta)
            shift[j] = step
            gradient[j] = (objective(theta + shift, block, settings) - objective(theta - shift, block, settings)) / (2 * step)
        m = 0.9 * m + 0.1 * gradient
        v = 0.999 * v + 0.001 * gradient ** 2
        theta = theta + rate * (m / (1 - 0.9 ** iteration)) / (np.sqrt(v / (1 - 0.999 ** iteration)) + 1e-8)
        if iteration % 10 == 0 or iteration == 1:
            history.append({"iteration": iteration, "objective": objective(theta, block, settings), **{f"weight_{name}": float(w) for name, w in zip(FEATURES, theta)}})
    return theta, history


def default_settings(model):
    return {
        "smoothing": float(model.get("smoothing", 20.0)),
        "inequality_aversion": float(model.get("inequality_aversion", 2.0)),
        "epsilon": float(model.get("epsilon", 0.02)),
        "leakage_weight": float(model.get("leakage_weight", 0.5)),
        "max_gap": float(model.get("max_gap", 0.3)),
        "gap_penalty": float(model.get("gap_penalty", 5.0)),
        "gap_sharpness": float(model.get("gap_sharpness", 30.0)),
        "ridge": float(model.get("ridge", 0.01)),
        "learning_rate": float(model.get("learning_rate", 0.05)),
        "finite_difference": float(model.get("finite_difference", 1e-4)),
        "iterations": int(model.get("iterations", 250)),
    }


def water_fill(allocation, need, group, groups, reserve):
    topup = np.zeros_like(allocation)
    unspent = np.zeros(groups)
    for g in range(groups):
        rows = np.flatnonzero(group == g)
        if len(rows) == 0 or reserve[g] <= 0:
            unspent[g] = max(reserve[g], 0.0)
            continue
        a = allocation[rows]
        n = need[rows]
        shortfall = np.maximum(n - a, 0.0).sum()
        if shortfall <= reserve[g]:
            topup[rows] = np.maximum(n - a, 0.0)
            unspent[g] = reserve[g] - shortfall
            continue
        low, high = 0.0, 1.0
        for _ in range(60):
            level = (low + high) / 2
            if np.maximum(level * n - a, 0.0).sum() > reserve[g]:
                high = level
            else:
                low = level
        topup[rows] = np.maximum(low * n - a, 0.0)
        unspent[g] = reserve[g] - topup[rows].sum()
    return topup, unspent
