from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fcrl_energy.cli import main
from fcrl_energy.pipeline import FCRL, FORECAST, HINDSIGHT, STANDARD, run

ROOT = Path(__file__).resolve().parents[1]


def quick_config():
    config = yaml.safe_load((ROOT / "configs" / "synthetic.yaml").read_text(encoding="utf-8"))
    config["model"]["iterations"] = 40
    config["evaluation"]["bootstrap_samples"] = 30
    return config


def test_pipeline_selects_on_validation_and_reports_all_rules():
    result = run(quick_config())
    performance = result.performance.set_index("strategy")
    assert {STANDARD, FORECAST, FCRL, HINDSIGHT} <= set(performance.index)
    assert result.selection.groupby("season")["selected"].sum().eq(1).all()
    for season, setting in result.selected.items():
        assert setting["level"] in set(result.selection["guaranteed_minimum"])
        assert setting["reserve"] in set(result.selection["emergency_reserve_share"])
    for name in performance.index:
        assert performance.loc[name, "overall_coverage"] <= performance.loc[HINDSIGHT, "overall_coverage"] + 1e-9
    assert performance.loc[FCRL, "worst_off_coverage"] > performance.loc[STANDARD, "worst_off_coverage"]
    assert len(result.statements) == 2 * result.panel["region"].nunique()
    assert result.statements["statement"].str.len().min() > 100
    assert set(result.stress["scenario"]) >= {"as observed", "20% cut in the support budget"}


def test_stress_tests_lower_coverage():
    result = run(quick_config())
    stress = result.stress.set_index(["scenario", "strategy"])
    for strategy in (STANDARD, FORECAST, FCRL):
        assert stress.loc[("price rise and budget cut together", strategy), "overall_coverage"] < stress.loc[("as observed", strategy), "overall_coverage"]


def test_cli_writes_report(tmp_path):
    config = quick_config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary = tmp_path / "summary.md"
    assert main(["run", "--config", str(path), "--output", str(tmp_path / "out"), "--summary", str(summary)]) == 0
    sheets = pd.read_excel(tmp_path / "out" / "energy_support_report.xlsx", sheet_name=None)
    assert {"Summary", "Strategies", "Allocation statements", "Stress tests", "Guarantee", "Emergency reserve", "Data sources"} <= set(sheets)
    assert "FCRL-energy" in summary.read_text(encoding="utf-8")
    assert np.isfinite(sheets["Strategies"]["overall_coverage"]).all()
