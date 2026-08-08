import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from common import ENTITY, OUTCOMES, PREDICTORS, PREDICTOR_LABELS, TIME, bh_adjust, ensure_output_dir, load_panel, zscore


def fit_country_models(panel):
    working = panel.copy()
    working["joint_burden"] = (zscore(working[OUTCOMES["Suicide"]]) + zscore(working[OUTCOMES["Anxiety"]])) / 2
    records = []
    for code, group in working.groupby(ENTITY, sort=True):
        group = group.sort_values(TIME)
        centered_year = group[TIME] - group[TIME].mean()
        standardized_outcome = zscore(group["joint_burden"])
        for predictor in PREDICTORS:
            design = pd.DataFrame(
                {
                    "predictor": zscore(group[predictor]).to_numpy(),
                    "year": centered_year.to_numpy(dtype=float),
                }
            )
            fitted = sm.OLS(standardized_outcome.to_numpy(), sm.add_constant(design)).fit()
            records.append(
                {
                    ENTITY: code,
                    "country_name": group["country_name"].iloc[0],
                    "continent": group["continent"].iloc[0],
                    "dimension": predictor,
                    "dimension_label": PREDICTOR_LABELS[predictor],
                    "estimate": float(fitted.params["predictor"]),
                    "standard_error": float(fitted.bse["predictor"]),
                    "ci_low": float(fitted.conf_int().loc["predictor", 0]),
                    "ci_high": float(fitted.conf_int().loc["predictor", 1]),
                    "p_value": float(fitted.pvalues["predictor"]),
                    "observations": int(fitted.nobs),
                    "adjusted_r_squared": float(fitted.rsquared_adj),
                }
            )
    result = pd.DataFrame(records)
    result["fdr_q_value_within_country"] = np.nan
    for code, index in result.groupby(ENTITY).groups.items():
        result.loc[index, "fdr_q_value_within_country"] = bh_adjust(result.loc[index, "p_value"])
    return result


def run(input_path=None, output_dir=None):
    destination = ensure_output_dir(output_dir)
    result = fit_country_models(load_panel(input_path))
    result.to_csv(destination / "country_specific_contextual_associations.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()

