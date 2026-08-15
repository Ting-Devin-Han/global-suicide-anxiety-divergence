import argparse
import json
from pathlib import Path

import numpy as np
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
        frame["value"] = country[f"z_{outcome}"].to_numpy()
        records.append(frame)
    stacked = pd.concat(records, ignore_index=True)
    stacked["outcome"] = pd.Categorical(stacked["outcome"], categories=list(OUTCOMES), ordered=True)
    return stacked


def linear_estimate(fitted, terms):
    vector = np.zeros(len(fitted.params), dtype=float)
    for term, weight in terms.items():
        vector[fitted.params.index.get_loc(term)] = weight
    test = fitted.t_test(vector)
    interval = np.asarray(test.conf_int()).reshape(-1)
    return {
        "estimate": float(np.asarray(test.effect).reshape(-1)[0]),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "p_value": float(np.asarray(test.pvalue).reshape(-1)[0]),
    }


def fit_model(country):
    stacked = stack_outcomes(country)
    main_terms = [f"z_{item}" for item in PREDICTORS]
    interaction_terms = [f"C(outcome):z_{item}" for item in PREDICTORS]
    formula = "value ~ C(outcome) + " + " + ".join(main_terms + interaction_terms)
    fitted = smf.ols(formula, data=stacked).fit(cov_type="cluster", cov_kwds={"groups": stacked[ENTITY]})
    records = []
    for predictor in PREDICTORS:
        main = f"z_{predictor}"
        outcome_terms = {"Suicide": {main: 1.0}}
        for label in list(OUTCOMES)[1:]:
            outcome_terms[label] = {main: 1.0, f"C(outcome)[T.{label}]:z_{predictor}": 1.0}
        for label, terms in outcome_terms.items():
            records.append(
                {
                    "dimension": predictor,
                    "dimension_label": PREDICTOR_LABELS[predictor],
                    "estimate_type": f"{label} association",
                    **linear_estimate(fitted, terms),
                }
            )
        for left, right in [("Anxiety", "Suicide"), ("Depression", "Suicide"), ("Depression", "Anxiety")]:
            terms = outcome_terms[left].copy()
            for term, weight in outcome_terms[right].items():
                terms[term] = terms.get(term, 0.0) - weight
            records.append(
                {
                    "dimension": predictor,
                    "dimension_label": PREDICTOR_LABELS[predictor],
                    "estimate_type": f"{left} minus {right.lower()} contrast",
                    **linear_estimate(fitted, terms),
                }
            )
    result = pd.DataFrame(records)
    contrast = result["estimate_type"].str.endswith("contrast")
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
        "multiple_testing": "Benjamini-Hochberg across thirty pairwise outcome contrasts",
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
