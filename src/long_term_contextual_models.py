import argparse
import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from common import ENTITY, OUTCOMES, PREDICTORS, PREDICTOR_LABELS, bh_adjust, ensure_output_dir, load_panel, zscore


def prepare_country_means(panel):
    columns = [*OUTCOMES.values(), *PREDICTORS]
    country = panel.groupby([ENTITY, "country_name", "continent"], as_index=False)[columns].mean()
    for column in columns:
        country[f"z_{column}"] = zscore(country[column])
    return country


def stack_outcomes(country):
    records = []
    for label, outcome in OUTCOMES.items():
        frame = country[[ENTITY, "country_name", "continent", *[f"z_{item}" for item in PREDICTORS]]].copy()
        frame["outcome"] = label
        frame["anxiety_outcome"] = int(label == "Anxiety")
        frame["value"] = country[f"z_{outcome}"].to_numpy()
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def fit_model(country):
    stacked = stack_outcomes(country)
    main_terms = [f"z_{item}" for item in PREDICTORS]
    interaction_terms = [f"anxiety_outcome:z_{item}" for item in PREDICTORS]
    formula = "value ~ anxiety_outcome + " + " + ".join(main_terms + interaction_terms)
    fitted = smf.ols(formula, data=stacked).fit(cov_type="cluster", cov_kwds={"groups": stacked[ENTITY]})
    records = []
    for predictor in PREDICTORS:
        main = f"z_{predictor}"
        interaction = f"anxiety_outcome:z_{predictor}"
        anxiety_test = fitted.t_test(f"{main} + {interaction} = 0")
        records.extend(
            [
                {
                    "dimension": predictor,
                    "dimension_label": PREDICTOR_LABELS[predictor],
                    "estimate_type": "Suicide association",
                    "estimate": float(fitted.params[main]),
                    "ci_low": float(fitted.conf_int().loc[main, 0]),
                    "ci_high": float(fitted.conf_int().loc[main, 1]),
                    "p_value": float(fitted.pvalues[main]),
                },
                {
                    "dimension": predictor,
                    "dimension_label": PREDICTOR_LABELS[predictor],
                    "estimate_type": "Anxiety association",
                    "estimate": float(anxiety_test.effect.item()),
                    "ci_low": float(anxiety_test.conf_int()[0, 0]),
                    "ci_high": float(anxiety_test.conf_int()[0, 1]),
                    "p_value": float(anxiety_test.pvalue.item()),
                },
                {
                    "dimension": predictor,
                    "dimension_label": PREDICTOR_LABELS[predictor],
                    "estimate_type": "Anxiety minus suicide contrast",
                    "estimate": float(fitted.params[interaction]),
                    "ci_low": float(fitted.conf_int().loc[interaction, 0]),
                    "ci_high": float(fitted.conf_int().loc[interaction, 1]),
                    "p_value": float(fitted.pvalues[interaction]),
                },
            ]
        )
    result = pd.DataFrame(records)
    contrast = result["estimate_type"] == "Anxiety minus suicide contrast"
    result["fdr_q_value"] = pd.NA
    result.loc[contrast, "fdr_q_value"] = bh_adjust(result.loc[contrast, "p_value"])
    return fitted, stacked, result


def run(input_path=None, output_dir=None):
    destination = ensure_output_dir(output_dir)
    country = prepare_country_means(load_panel(input_path))
    fitted, stacked, coefficients = fit_model(country)
    country.to_csv(destination / "long_term_country_means.csv", index=False)
    coefficients.to_csv(destination / "long_term_contextual_coefficients.csv", index=False)
    metadata = {
        "country_observations": int(country[ENTITY].nunique()),
        "stacked_observations": int(len(stacked)),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "covariance": "Country-clustered",
        "multiple_testing": "Benjamini-Hochberg across ten outcome contrasts",
    }
    (destination / "long_term_model_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return coefficients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

