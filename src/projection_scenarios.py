import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ENTITY, NONFATAL_OUTCOMES, OUTCOMES, TIME, ensure_output_dir, load_panel


SCENARIOS = {
    "Trend continuation": 1.0,
    "Attenuated burden divergence": None,
    "Accelerated burden divergence": None,
}


def divergence_calibration(panel):
    working = panel.sort_values([ENTITY, TIME]).copy()
    for label, column in OUTCOMES.items():
        key = label.lower()
        working[f"log_{key}"] = np.log(np.clip(working[column].to_numpy(dtype=float), 1e-8, None))
        working[f"change_{key}"] = working.groupby(ENTITY)[f"log_{key}"].diff()
    nonfatal_change = working[[f"change_{label.lower()}" for label in NONFATAL_OUTCOMES]].mean(axis=1)
    intensity = (nonfatal_change - working["change_suicide"]).abs().dropna()
    quantiles = intensity.quantile([0.25, 0.50, 0.75])
    calibration = pd.DataFrame(
        {
            "statistic": ["q25", "q50", "q75"],
            "annual_divergence_intensity": [quantiles.loc[0.25], quantiles.loc[0.50], quantiles.loc[0.75]],
        }
    )
    multipliers = {
        "Trend continuation": 1.0,
        "Attenuated burden divergence": float(quantiles.loc[0.25] / quantiles.loc[0.50]),
        "Accelerated burden divergence": float(quantiles.loc[0.75] / quantiles.loc[0.50]),
    }
    return calibration, multipliers


def normalize_continuation(forecast):
    if "scenario" in forecast.columns and (forecast["scenario"] == "Continuation").any():
        forecast = forecast[forecast["scenario"] == "Continuation"].copy()
    required = [ENTITY, "outcome", TIME, "prediction"]
    missing = [column for column in required if column not in forecast.columns]
    if missing:
        raise ValueError(f"Continuation forecast is missing columns: {missing}")
    if forecast.duplicated([ENTITY, "outcome", TIME]).any():
        raise ValueError("Continuation forecast contains duplicate country-outcome-year rows")
    return forecast.copy()


def apply_scenarios(panel, continuation):
    calibration, multipliers = divergence_calibration(panel)
    baseline = panel[panel[TIME] == panel[TIME].max()][[ENTITY, "country_name", "continent", *OUTCOMES.values()]].copy()
    baseline = baseline.melt(
        id_vars=[ENTITY, "country_name", "continent"],
        value_vars=list(OUTCOMES.values()),
        var_name="outcome_column",
        value_name="baseline_2019",
    )
    inverse = {column: label for label, column in OUTCOMES.items()}
    baseline["outcome"] = baseline["outcome_column"].map(inverse)
    base_columns = baseline[[ENTITY, "country_name", "continent", "outcome", "baseline_2019"]]
    continuation = normalize_continuation(continuation).drop(columns=["country_name", "continent"], errors="ignore")
    continuation = continuation.merge(base_columns, on=[ENTITY, "outcome"], how="left", validate="many_to_one")
    if continuation["baseline_2019"].isna().any():
        raise ValueError("Forecast contains unmatched country-outcome rows")
    frames = []
    for scenario, multiplier in multipliers.items():
        frame = continuation.copy()
        frame["scenario"] = scenario
        frame["divergence_multiplier"] = multiplier
        frame["continuation_scaffold"] = frame["prediction"].astype(float)
        if scenario != "Trend continuation":
            log_change = np.log(
                np.clip(frame["continuation_scaffold"].to_numpy(dtype=float), 1e-8, None)
                / np.clip(frame["baseline_2019"].to_numpy(dtype=float), 1e-8, None)
            )
            frame["prediction"] = frame["baseline_2019"].to_numpy(dtype=float) * np.exp(multiplier * log_change)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    return result, calibration, multipliers


def summarize_scenarios(projections):
    endpoint = projections[projections[TIME] == projections[TIME].max()].copy()
    endpoint["relative_change_percent"] = (endpoint["prediction"] / endpoint["baseline_2019"] - 1) * 100
    summary = endpoint.groupby(["scenario", "outcome"], as_index=False).agg(
        median_within_country_change_percent=("relative_change_percent", "median"),
        q25_within_country_change_percent=("relative_change_percent", lambda values: values.quantile(0.25)),
        q75_within_country_change_percent=("relative_change_percent", lambda values: values.quantile(0.75)),
        countries_or_territories_increasing=("relative_change_percent", lambda values: int((values > 0).sum())),
        countries_or_territories_decreasing=("relative_change_percent", lambda values: int((values < 0).sum())),
    )
    profiles = endpoint.pivot_table(index=[ENTITY, "country_name", "continent", "scenario"], columns="outcome", values="relative_change_percent").reset_index()
    profiles["nonfatal_change_percent"] = profiles[list(NONFATAL_OUTCOMES)].mean(axis=1)
    profiles["projected_profile"] = np.select(
        [
            (profiles["Suicide"] < 0) & (profiles["nonfatal_change_percent"] > 0),
            (profiles["Suicide"] > 0) & (profiles["nonfatal_change_percent"] > 0),
            (profiles["Suicide"] < 0) & (profiles["nonfatal_change_percent"] < 0),
        ],
        ["Fatal down and non-fatal up", "Both increasing", "Both decreasing"],
        default="Fatal up and non-fatal down",
    )
    counts = profiles.groupby(["scenario", "projected_profile"], as_index=False).size().rename(columns={"size": "countries_or_territories"})
    return summary, profiles, counts


def run(input_path=None, forecast_path=None, output_dir=None):
    destination = ensure_output_dir(output_dir)
    if forecast_path is None:
        forecast_path = destination / "continuation_country_projections.csv"
    panel = load_panel(input_path)
    continuation = pd.read_csv(forecast_path)
    projections, calibration, multipliers = apply_scenarios(panel, continuation)
    summary, profiles, counts = summarize_scenarios(projections)
    projections.to_csv(destination / "scenario_country_projections.csv", index=False)
    calibration.to_csv(destination / "divergence_intensity_calibration.csv", index=False)
    summary.to_csv(destination / "scenario_summary.csv", index=False)
    profiles.to_csv(destination / "scenario_country_profiles.csv", index=False)
    counts.to_csv(destination / "scenario_profile_counts.csv", index=False)
    (destination / "scenario_multipliers.json").write_text(json.dumps(multipliers, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--forecast", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.forecast, args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
