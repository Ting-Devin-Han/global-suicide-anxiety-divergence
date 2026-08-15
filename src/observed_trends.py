import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

from common import ENTITY, NONFATAL_OUTCOMES, OUTCOMES, TIME, ensure_output_dir, load_panel, zscore


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
    linear = trends.pivot(index=ENTITY, columns="outcome", values="linear_slope").rename(
        columns={label: f"{label.lower()}_slope" for label in OUTCOMES}
    )
    annualized = trends.pivot(index=ENTITY, columns="outcome", values="annualized_log_change_percent").rename(
        columns={label: f"{label.lower()}_annualized_log_change" for label in OUTCOMES}
    )
    relative = trends.pivot(index=ENTITY, columns="outcome", values="relative_change_percent").rename(
        columns={label: f"{label.lower()}_relative_change" for label in OUTCOMES}
    )
    result = country.merge(linear.reset_index(), on=ENTITY, how="left").merge(
        annualized.reset_index(), on=ENTITY, how="left"
    ).merge(relative.reset_index(), on=ENTITY, how="left")
    for label in OUTCOMES:
        result[f"{label.lower()}_level_z"] = zscore(result[f"{label.lower()}_mean"])
    result["nonfatal_long_term_score"] = result[
        [f"{label.lower()}_level_z" for label in NONFATAL_OUTCOMES]
    ].mean(axis=1)
    result["nonfatal_annualized_log_change"] = result[
        [f"{label.lower()}_annualized_log_change" for label in NONFATAL_OUTCOMES]
    ].mean(axis=1)
    result["nonfatal_relative_change"] = result[
        [f"{label.lower()}_relative_change" for label in NONFATAL_OUTCOMES]
    ].mean(axis=1)
    fatal_cut = result["suicide_level_z"].median()
    nonfatal_cut = result["nonfatal_long_term_score"].median()
    result["level_profile"] = np.select(
        [
            (result["suicide_level_z"] < fatal_cut) & (result["nonfatal_long_term_score"] < nonfatal_cut),
            (result["suicide_level_z"] >= fatal_cut) & (result["nonfatal_long_term_score"] < nonfatal_cut),
            (result["suicide_level_z"] < fatal_cut) & (result["nonfatal_long_term_score"] >= nonfatal_cut),
        ],
        ["Low fatal and low non-fatal", "High fatal and low non-fatal", "Low fatal and high non-fatal"],
        default="High fatal and high non-fatal",
    )
    result["change_profile"] = np.select(
        [
            (result["suicide_relative_change"] < 0) & (result["nonfatal_relative_change"] > 0),
            (result["suicide_relative_change"] > 0) & (result["nonfatal_relative_change"] > 0),
            (result["suicide_relative_change"] < 0) & (result["nonfatal_relative_change"] < 0),
        ],
        ["Fatal down and non-fatal up", "Both increasing", "Both decreasing"],
        default="Fatal up and non-fatal down",
    )
    return result


def observed_summary(panel, trends, profiles):
    records = []
    slope_matrix = trends.pivot(index=ENTITY, columns="outcome", values="linear_slope")
    correlations = []
    labels = list(OUTCOMES)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            level = spearmanr(profiles[f"{left.lower()}_mean"], profiles[f"{right.lower()}_mean"])
            trend = spearmanr(slope_matrix[left], slope_matrix[right])
            correlations.append(
                {
                    "outcome_1": left,
                    "outcome_2": right,
                    "long_term_spearman_rho": float(level.statistic),
                    "long_term_two_sided_p": float(level.pvalue),
                    "slope_spearman_rho": float(trend.statistic),
                    "slope_two_sided_p": float(trend.pvalue),
                }
            )
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
        "pairwise_correlations": correlations,
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
