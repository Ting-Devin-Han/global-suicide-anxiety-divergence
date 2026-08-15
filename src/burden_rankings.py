import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from common import ENTITY, NONFATAL_OUTCOMES, OUTCOMES, TIME, ensure_output_dir, load_panel, zscore


def annualized_log_change(values, years):
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=float)
    valid = np.isfinite(values) & np.isfinite(years) & (values > 0)
    if valid.sum() < 3:
        return np.nan
    return float(linregress(years[valid], np.log(values[valid])).slope)


def robust_minmax(values, lower=5, upper=95):
    series = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    low, high = np.nanpercentile(series, [lower, upper])
    if not np.isfinite(high - low) or high == low:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - low) / (high - low)).clip(0, 1)


def calculate_rankings(panel):
    records = []
    for code, group in panel.groupby(ENTITY, sort=True):
        group = group.sort_values(TIME)
        record = {
            ENTITY: code,
            "country_name": group["country_name"].iloc[0],
            "continent": group["continent"].iloc[0],
        }
        for label, column in OUTCOMES.items():
            key = label.lower()
            record[f"{key}_mean"] = float(group[column].mean())
            record[f"{key}_annualized_log_change"] = annualized_log_change(group[column], group[TIME])
            record[f"{key}_relative_change_percent"] = float((group[column].iloc[-1] / group[column].iloc[0] - 1) * 100)
        records.append(record)
    ranking = pd.DataFrame(records)
    labels = [label.lower() for label in OUTCOMES]
    for label in labels:
        ranking[f"{label}_level_z"] = zscore(ranking[f"{label}_mean"])
        ranking[f"{label}_long_term_component"] = robust_minmax(ranking[f"{label}_mean"])
        ranking[f"{label}_change_z"] = zscore(ranking[f"{label}_annualized_log_change"])
    ranking["long_term_joint_burden"] = ranking[[f"{label}_level_z" for label in labels]].mean(axis=1)
    component_columns = [f"{label}_long_term_component" for label in labels]
    component_total = ranking[component_columns].sum(axis=1).replace(0, np.nan)
    for label in labels:
        ranking[f"{label}_long_term_share"] = (
            ranking[f"{label}_long_term_component"] / component_total
        ).fillna(1 / len(labels))
    ranking["joint_increase_score"] = ranking[[f"{label}_change_z" for label in labels]].mean(axis=1)
    ranking["burden_ranking_score"] = 0.60 * ranking["long_term_joint_burden"] + 0.40 * ranking["joint_increase_score"]
    ranking["composite_rank"] = ranking["burden_ranking_score"].rank(ascending=False, method="first").astype(int)
    ranking["long_term_rank"] = ranking["long_term_joint_burden"].rank(ascending=False, method="first").astype(int)
    ranking["nonfatal_long_term_score"] = ranking[
        [f"{label.lower()}_level_z" for label in NONFATAL_OUTCOMES]
    ].mean(axis=1)
    ranking["nonfatal_annualized_log_change"] = ranking[
        [f"{label.lower()}_annualized_log_change" for label in NONFATAL_OUTCOMES]
    ].mean(axis=1)
    ranking["nonfatal_relative_change_percent"] = ranking[
        [f"{label.lower()}_relative_change_percent" for label in NONFATAL_OUTCOMES]
    ].mean(axis=1)
    fatal_cut = ranking["suicide_level_z"].median()
    nonfatal_cut = ranking["nonfatal_long_term_score"].median()
    ranking["level_profile"] = np.select(
        [
            (ranking["suicide_level_z"] < fatal_cut) & (ranking["nonfatal_long_term_score"] < nonfatal_cut),
            (ranking["suicide_level_z"] >= fatal_cut) & (ranking["nonfatal_long_term_score"] < nonfatal_cut),
            (ranking["suicide_level_z"] < fatal_cut) & (ranking["nonfatal_long_term_score"] >= nonfatal_cut),
        ],
        ["Low fatal and low non-fatal", "High fatal and low non-fatal", "Low fatal and high non-fatal"],
        default="High fatal and high non-fatal",
    )
    ranking["change_profile"] = np.select(
        [
            (ranking["suicide_relative_change_percent"] < 0) & (ranking["nonfatal_relative_change_percent"] > 0),
            (ranking["suicide_relative_change_percent"] > 0) & (ranking["nonfatal_relative_change_percent"] > 0),
            (ranking["suicide_relative_change_percent"] < 0) & (ranking["nonfatal_relative_change_percent"] < 0),
        ],
        ["Fatal down and non-fatal up", "Both increasing", "Both decreasing"],
        default="Fatal up and non-fatal down",
    )
    for label in labels:
        ranking[f"{label}_rank"] = ranking[f"{label}_mean"].rank(ascending=False, method="min").astype(int)
    return ranking.sort_values("composite_rank").reset_index(drop=True)


def run(input_path=None, output_dir=None):
    destination = ensure_output_dir(output_dir)
    result = calculate_rankings(load_panel(input_path))
    result.to_csv(destination / "burden_rankings.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
