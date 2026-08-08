import argparse
from pathlib import Path

import pandas as pd

from common import ENTITY, OUTCOMES, TIME, ensure_output_dir, load_panel


def observed_regional_summary(panel):
    first_year = int(panel[TIME].min())
    last_year = int(panel[TIME].max())
    first = panel[panel[TIME] == first_year][[ENTITY, "continent", *OUTCOMES.values()]].copy()
    last = panel[panel[TIME] == last_year][[ENTITY, *OUTCOMES.values()]].copy()
    merged = first.merge(last, on=ENTITY, suffixes=("_first", "_last"), validate="one_to_one")
    records = []
    for label, column in OUTCOMES.items():
        merged["relative_change_percent"] = (merged[f"{column}_last"] / merged[f"{column}_first"] - 1) * 100
        for continent, group in merged.groupby("continent", sort=True):
            records.append(
                {
                    "continent": continent,
                    "outcome": label,
                    "start_year": first_year,
                    "end_year": last_year,
                    "median_within_country_change_percent": float(group["relative_change_percent"].median()),
                    "countries_or_territories": int(len(group)),
                    "countries_or_territories_increasing": int((group["relative_change_percent"] > 0).sum()),
                }
            )
    return pd.DataFrame(records)


def projected_regional_summary(projections):
    endpoint = projections[projections[TIME] == projections[TIME].max()].copy()
    endpoint["relative_change_percent"] = (endpoint["prediction"] / endpoint["baseline_2019"] - 1) * 100
    return endpoint.groupby(["scenario", "continent", "outcome"], as_index=False).agg(
        median_within_country_change_percent=("relative_change_percent", "median"),
        countries_or_territories=(ENTITY, "nunique"),
        countries_or_territories_increasing=("relative_change_percent", lambda values: int((values > 0).sum())),
    )


def run(input_path=None, projection_path=None, output_dir=None):
    destination = ensure_output_dir(output_dir)
    observed = observed_regional_summary(load_panel(input_path))
    observed.to_csv(destination / "observed_regional_heterogeneity.csv", index=False)
    projected = None
    candidate = Path(projection_path) if projection_path else destination / "scenario_country_projections.csv"
    if candidate.exists():
        projected = projected_regional_summary(pd.read_csv(candidate))
        projected.to_csv(destination / "projected_regional_heterogeneity.csv", index=False)
    return observed, projected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--projections", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    observed, projected = run(args.input, args.projections, args.output_dir)
    print(observed.to_string(index=False))
    if projected is not None:
        print(projected.to_string(index=False))


if __name__ == "__main__":
    main()

