import argparse
import json
import sys
from pathlib import Path

import yaml

from .pipeline import run
from .report import markdown, write_excel


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fcrl-energy")
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--config", default="configs/synthetic.yaml")
    runner.add_argument("--output", default="outputs")
    runner.add_argument("--summary", default=None)
    arguments = parser.parse_args(argv)
    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    result = run(config)
    report = write_excel(result, Path(arguments.output) / "energy_support_report.xlsx")
    print(
        json.dumps(
            {
                "report": str(report),
                "regions": int(result.panel["region"].nunique()),
                "plan": {season: {"guaranteed_minimum": value["level"], "emergency_reserve": value["reserve"]} for season, value in result.selected.items()},
                "strategies": {
                    row["strategy"]: {"overall": row["overall_coverage"], "worst_off": row["worst_off_coverage"], "gap": row["coverage_gap"]}
                    for row in result.performance.to_dict("records")
                },
            },
            indent=2,
            default=float,
        )
    )
    if arguments.summary:
        with Path(arguments.summary).open("a", encoding="utf-8") as handle:
            handle.write(markdown(result, arguments.config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
