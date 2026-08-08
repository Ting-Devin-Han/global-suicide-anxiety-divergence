import argparse
import subprocess
import sys
from pathlib import Path

from common import default_input_path, ensure_output_dir


def execute(script, input_path, output_dir, additional=None):
    command = [sys.executable, str(Path(__file__).resolve().parent / script), "--input", str(input_path), "--output-dir", str(output_dir)]
    if additional:
        command.extend(additional)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-projection", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--skip-backtests", action="store_true")
    args = parser.parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    for script in [
        "validate_panel.py",
        "observed_trends.py",
        "burden_rankings.py",
        "long_term_contextual_models.py",
        "within_country_models.py",
        "country_specific_models.py",
    ]:
        execute(script, args.input, output_dir)
    if args.include_projection:
        projection_arguments = ["--device", args.device, "--estimators", str(args.estimators)]
        if args.skip_backtests:
            projection_arguments.append("--skip-backtests")
        execute("recursive_projection.py", args.input, output_dir, projection_arguments)
        execute("projection_scenarios.py", args.input, output_dir, ["--forecast", str(output_dir / "continuation_country_projections.csv")])
        execute("regional_heterogeneity.py", args.input, output_dir, ["--projections", str(output_dir / "scenario_country_projections.csv")])
    else:
        execute("regional_heterogeneity.py", args.input, output_dir)


if __name__ == "__main__":
    main()

