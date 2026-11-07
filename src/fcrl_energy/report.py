from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import metrics as m
from .pipeline import FCRL, FORECAST, STANDARD

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    return f"{value:.{digits}f}"


def _ci(row, metric):
    return f"{_fmt(row[metric])} (95% CI {_fmt(row[metric + '_ci_low'])} to {_fmt(row[metric + '_ci_high'])})"


def _row(result, name):
    table = result.performance.set_index("strategy")
    return table.loc[name]


def summary_rows(result):
    config = result.config
    panel = result.panel
    plan = result.selected
    notes = result.notes
    standard = _row(result, STANDARD)
    forecast = _row(result, FORECAST)
    fcrl = _row(result, FCRL)
    rows = [
        ("Data source", config["data"]["source"]),
        ("Regions", int(panel["region"].nunique())),
        ("Countries", ", ".join(sorted(panel["country"].unique()))),
        ("Test seasons", ", ".join(sorted(panel.loc[panel["split"] == "test", "season_label"].unique()))),
        ("Support budget", f"{notes['budget_share']:.0%} of the forecast energy need of residents at risk of poverty, set before each month"),
        ("Need definition", "residents at risk of poverty x (heating degree days in winter, cooling degree days in summer, plus a base need) x energy price index"),
    ]
    for season in ("winter", "summer"):
        if season in plan:
            rows.append((f"Guaranteed minimum, {season} (selected on validation)", f"{plan[season]['level']:.0%} of each region's need, protected against forecast error with a target miss rate of {notes['miss_rate']:.0%}"))
            rows.append((f"Emergency reserve, {season} (selected on validation)", f"{plan[season]['reserve']:.0%} of the budget, paid after the month to the regions with the lowest coverage"))
    reason = next((plan[s]["reason"] for s in ("winter", "summer") if s in plan), "")
    rows += [
        ("Selection rule", reason),
        ("Forecast bound miss rate on validation", _fmt(notes["validation_miss_rate"])),
        ("Forecast bound miss rate on test", _fmt(notes["guarantee_test_miss_rate"])),
        ("Guarantee attainment on test, FCRL-energy", _fmt(fcrl["guarantee_attainment"])),
    ]
    for label, row in ((STANDARD, standard), (FORECAST, forecast), (FCRL, fcrl)):
        rows.append((f"Worst-off coverage, {label}", _ci(row, "worst_off_coverage")))
        rows.append((f"Overall coverage, {label}", _ci(row, "overall_coverage")))
        rows.append((f"Coverage gap, {label}", _ci(row, "coverage_gap")))
        rows.append((f"Lowest coverage, {label}", _fmt(row["lowest_coverage"])))
    for comparison, block in result.differences.groupby("comparison", sort=False):
        for metric in ("worst_off_coverage", "overall_coverage", "coverage_gap", "leakage_share"):
            row = block[block["metric"] == metric].iloc[0]
            rows.append((f"{m.LABELS[metric].capitalize()}: {comparison}", f"{_fmt(row['difference'])} (95% CI {_fmt(row['ci_low'])} to {_fmt(row['ci_high'])})"))
    return rows


def _format(sheet):
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for column in sheet.columns:
        width = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
        sheet.column_dimensions[get_column_letter(column[0].column)].width = min(max(12, width + 2), 90)
    sheet.freeze_panes = "A2"


def write_excel(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = pd.DataFrame(result.raw_log)
    sheets = {
        "Summary": pd.DataFrame(summary_rows(result), columns=["Item", "Value"]),
        "Strategies": result.performance,
        "Compared with other rules": result.differences,
        "Selection on validation": result.selection,
        "Emergency reserve": result.reserve,
        "Stress tests": result.stress,
        "Allocation statements": result.statements,
        "Policy weights": result.weights,
        "Regional coverage": result.regional,
        "Guarantee": result.guarantee_path,
        "Training history": result.history,
        "Data coverage": result.coverage,
        "Data sources": sources if len(sources) else pd.DataFrame({"note": ["no log"]}),
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            _format(writer.sheets[name[:31]])
    return path


def markdown(result, config_name):
    lines = [f"## FCRL-energy: `{config_name}`", ""]
    plan = result.selected
    notes = result.notes
    for season in ("winter", "summer"):
        if season in plan:
            lines.append(f"{season.capitalize()}: guaranteed minimum **{plan[season]['level']:.0%}** of need, emergency reserve **{plan[season]['reserve']:.0%}** of the budget.")
    lines.append("")
    lines.append(f"Forecast bound miss rate: validation {_fmt(notes['validation_miss_rate'])}, test {_fmt(notes['guarantee_test_miss_rate'])} (target {notes['miss_rate']:.2f}).")
    lines.append("")
    lines.append("| Strategy | Overall coverage | Worst-off coverage | Lowest coverage | Coverage gap | Leakage |")
    lines.append("|---|---|---|---|---|---|")
    for row in result.performance.to_dict("records"):
        lines.append(
            f"| {row['strategy']} | {_fmt(row['overall_coverage'])} | {_fmt(row['worst_off_coverage'])} ({_fmt(row['worst_off_coverage_ci_low'])} to {_fmt(row['worst_off_coverage_ci_high'])}) | {_fmt(row['lowest_coverage'])} | {_fmt(row['coverage_gap'])} | {_fmt(row['leakage_share'])} |"
        )
    lines.append("")
    lines.append("| Stress test | Strategy | Overall coverage | Worst-off coverage | Lowest coverage | Guarantee attainment |")
    lines.append("|---|---|---|---|---|---|")
    for row in result.stress.to_dict("records"):
        lines.append(f"| {row['scenario']} | {row['strategy']} | {_fmt(row['overall_coverage'])} | {_fmt(row['worst_off_coverage'])} | {_fmt(row['lowest_coverage'])} | {_fmt(row['guarantee_attainment'])} |")
    lines.append("")
    return "\n".join(lines) + "\n"
