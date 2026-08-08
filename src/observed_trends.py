import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from common import ENTITY, OUTCOMES, TIME, ensure_output_dir, load_panel


def slope(values, years):
    return float(linregress(np.asarray(years, dtype=float), np.asarray(values, dtype=float)).slope)


def annualized_log_change(values, years):
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0):
        return np.nan
    return float(np.expm1(slope(np.log(array), years)) * 100)


def bootstrap_median_interval(values, repetitions=5000, seed=42):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(repetitions, len(array)), replace=True)
    medians = np.median(draws, axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def country_trends(panel):
    records = []
    for code, group in panel.groupby(ENTITY, sort=True):
        group = group.sort_values(TIME)
        base = {
            ENTITY: code,
            "country_name": group["country_name"].iloc[0],
            "continent": group["continent"].iloc[0],
        }
        for label, column in OUTCOMES.items():
            values = group[column].to_numpy(dtype=float)
            years = group[TIME].to_numpy(dtype=float)
            records.append(
                {
                    **base,
                    "outcome": label,
                    "value_2000": float(values[0]),
                    "value_2019": float(values[-1]),
                    "absolute_change": float(values[-1] - values[0]),
                    "relative_change_percent": float((values[-1] / values[0] - 1) * 100),
                    "linear_slope": slope(values, years),
                    "annualized_log_change_percent": annualized_log_change(values, years),
                    "increased": bool(values[-1] > values[0]),
                }
            )
    return pd.DataFrame(records)


def annual_summary(panel):
    records = []
    for label, column in OUTCOMES.items():
        for year, group in panel.groupby(TIME, sort=True):
            values = group[column].to_numpy(dtype=float)
            records.append(
                {
                    "outcome": label,
                    TIME: int(year),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "countries_or_territories": int(len(values)),
                }
            )
    return pd.DataFrame(records)


def long_term_profiles(panel, trends):
    country = panel.groupby([ENTITY, "country_name", "continent"], as_index=False)[list(OUTCOMES.values())].mean()
    country = country.rename(columns={value: f"{label.lower()}_mean" for label, value in OUTCOMES.items()})
    trend_wide = trends.pivot(index=ENTITY, columns="outcome", values="linear_slope").reset_index()
    trend_wide = trend_wide.rename(columns={"Suicide": "suicide_slope", "Anxiety": "anxiety_slope"})
    result = country.merge(trend_wide, on=ENTITY, how="left")
    suicide_median = result["suicide_mean"].median()
    anxiety_median = result["anxiety_mean"].median()
    result["level_profile"] = np.select(
        [
            (result["suicide_mean"] >= suicide_median) & (result["anxiety_mean"] >= anxiety_median),
            (result["suicide_mean"] >= suicide_median) & (result["anxiety_mean"] < anxiety_median),
            (result["suicide_mean"] < suicide_median) & (result["anxiety_mean"] >= anxiety_median),
        ],
        ["High suicide and high anxiety", "High suicide only", "High anxiety only"],
        default="Low suicide and low anxiety",
    )
    result["change_profile"] = np.select(
        [
            (result["suicide_slope"] >= 0) & (result["anxiety_slope"] >= 0),
            (result["suicide_slope"] >= 0) & (result["anxiety_slope"] < 0),
            (result["suicide_slope"] < 0) & (result["anxiety_slope"] >= 0),
        ],
        ["Both increasing", "Suicide increasing and anxiety decreasing", "Suicide decreasing and anxiety increasing"],
        default="Both decreasing",
    )
    return result


def observed_summary(panel, trends, profiles):
    records = []
    slope_matrix = trends.pivot(index=ENTITY, columns="outcome", values="linear_slope")
    level_correlation = spearmanr(profiles["suicide_mean"], profiles["anxiety_mean"])
    slope_correlation = spearmanr(slope_matrix["Suicide"], slope_matrix["Anxiety"])
    for label in OUTCOMES:
        subset = trends[trends["outcome"] == label]
        lower, upper = bootstrap_median_interval(subset["linear_slope"])
        records.append(
            {
                "outcome": label,
                "median_country_relative_change_percent": float(subset["relative_change_percent"].median()),
                "countries_or_territories_increasing": int(subset["increased"].sum()),
                "countries_or_territories_decreasing": int((~subset["increased"]).sum()),
                "median_country_linear_slope": float(subset["linear_slope"].median()),
                "bootstrap_ci_low": lower,
                "bootstrap_ci_high": upper,
                "median_annualized_log_change_percent": float(subset["annualized_log_change_percent"].median()),
            }
        )
    metadata = {
        "countries_or_territories": int(panel[ENTITY].nunique()),
        "years": [int(panel[TIME].min()), int(panel[TIME].max())],
        "spearman_long_term_level_correlation": float(level_correlation.statistic),
        "spearman_long_term_level_p": float(level_correlation.pvalue),
        "spearman_slope_correlation": float(slope_correlation.statistic),
        "spearman_slope_p": float(slope_correlation.pvalue),
    }
    return pd.DataFrame(records), metadata


def run(input_path=None, output_dir=None):
    panel = load_panel(input_path)
    destination = ensure_output_dir(output_dir)
    trends = country_trends(panel)
    annual = annual_summary(panel)
    profiles = long_term_profiles(panel, trends)
    summary, metadata = observed_summary(panel, trends, profiles)
    profile_counts = pd.concat(
        [
            profiles["level_profile"].value_counts().rename_axis("profile").reset_index(name="countries_or_territories").assign(profile_type="level"),
            profiles["change_profile"].value_counts().rename_axis("profile").reset_index(name="countries_or_territories").assign(profile_type="change"),
        ],
        ignore_index=True,
    )
    annual.to_csv(destination / "annual_outcome_summary.csv", index=False)
    trends.to_csv(destination / "country_outcome_trends.csv", index=False)
    profiles.to_csv(destination / "long_term_profiles.csv", index=False)
    profile_counts.to_csv(destination / "profile_counts.csv", index=False)
    summary.to_csv(destination / "observed_trend_summary.csv", index=False)
    (destination / "observed_trend_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

