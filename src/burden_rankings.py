import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from common import ENTITY, OUTCOMES, TIME, ensure_output_dir, load_panel, zscore


def annualized_log_change(values, years):
    values = np.asarray(values, dtype=float)
    years = np.asarray(years, dtype=float)
    valid = np.isfinite(values) & np.isfinite(years) & (values > 0)
    if valid.sum() < 3:
        return np.nan
    return float(linregress(years[valid], np.log(values[valid])).slope)


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
        records.append(record)
    ranking = pd.DataFrame(records)
    ranking["suicide_level_z"] = zscore(ranking["suicide_mean"])
    ranking["anxiety_level_z"] = zscore(ranking["anxiety_mean"])
    ranking["long_term_joint_burden"] = (ranking["suicide_level_z"] + ranking["anxiety_level_z"]) / 2
    ranking["suicide_change_z"] = zscore(ranking["suicide_annualized_log_change"])
    ranking["anxiety_change_z"] = zscore(ranking["anxiety_annualized_log_change"])
    ranking["joint_increase_score"] = (ranking["suicide_change_z"] + ranking["anxiety_change_z"]) / 2
    ranking["burden_ranking_score"] = 0.60 * ranking["long_term_joint_burden"] + 0.40 * ranking["joint_increase_score"]
    ranking["composite_rank"] = ranking["burden_ranking_score"].rank(ascending=False, method="min").astype(int)
    ranking["long_term_rank"] = ranking["long_term_joint_burden"].rank(ascending=False, method="min").astype(int)
    ranking["level_profile"] = np.select(
        [
            (ranking["suicide_level_z"] >= 0) & (ranking["anxiety_level_z"] >= 0),
            (ranking["suicide_level_z"] >= 0) & (ranking["anxiety_level_z"] < 0),
            (ranking["suicide_level_z"] < 0) & (ranking["anxiety_level_z"] >= 0),
        ],
        ["High suicide and high anxiety", "High suicide only", "High anxiety only"],
        default="Low suicide and low anxiety",
    )
    ranking["change_profile"] = np.select(
        [
            (ranking["suicide_annualized_log_change"] >= 0) & (ranking["anxiety_annualized_log_change"] >= 0),
            (ranking["suicide_annualized_log_change"] >= 0) & (ranking["anxiety_annualized_log_change"] < 0),
            (ranking["suicide_annualized_log_change"] < 0) & (ranking["anxiety_annualized_log_change"] >= 0),
        ],
        ["Both increasing", "Suicide increasing and anxiety decreasing", "Suicide decreasing and anxiety increasing"],
        default="Both decreasing",
    )
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
