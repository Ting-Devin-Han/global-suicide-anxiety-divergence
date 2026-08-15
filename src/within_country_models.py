import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import ENTITY, OUTCOMES, PREDICTORS, PREDICTOR_LABELS, TIME, bh_adjust, ensure_output_dir, load_panel, zscore


def prepare_stacked_panel(panel, lag):
    working = panel.copy()
    if lag:
        working[PREDICTORS] = working.groupby(ENTITY)[PREDICTORS].shift(lag)
        working = working.dropna(subset=PREDICTORS).copy()
    for column in [*OUTCOMES.values(), *PREDICTORS]:
        working[f"z_{column}"] = zscore(working[column])
    frames = []
    for label, outcome in OUTCOMES.items():
        frame = working[[ENTITY, "country_name", TIME, *[f"z_{item}" for item in PREDICTORS]]].copy()
        frame["outcome"] = label
        frame["anxiety_outcome"] = int(label == "Anxiety")
        frame["depression_outcome"] = int(label == "Depression")
        frame["value"] = working[f"z_{outcome}"].to_numpy()
        frame["outcome_country"] = label + ":" + frame[ENTITY].astype(str)
        frame["outcome_year"] = label + ":" + frame[TIME].astype(str)
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)
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


def fit_model(panel, lag):
    stacked = prepare_stacked_panel(panel, lag)
    main_terms = [f"z_{item}" for item in PREDICTORS]
    interaction_terms = [f"{indicator}:z_{item}" for indicator in ["anxiety_outcome", "depression_outcome"] for item in PREDICTORS]
    formula = "value ~ " + " + ".join(main_terms + interaction_terms) + " + C(outcome_country) + C(outcome_year)"
    fitted = smf.ols(formula, data=stacked).fit(cov_type="cluster", cov_kwds={"groups": stacked[ENTITY]})
    records = []
    for predictor in PREDICTORS:
        main = f"z_{predictor}"
        outcome_terms = {
            "Suicide": {main: 1.0},
            "Anxiety": {main: 1.0, f"anxiety_outcome:z_{predictor}": 1.0},
            "Depression": {main: 1.0, f"depression_outcome:z_{predictor}": 1.0},
        }
        for label, terms in outcome_terms.items():
            records.append(
                {
                    "lag_years": lag,
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
                    "lag_years": lag,
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
    metadata = {
        "lag_years": lag,
        "stacked_observations": int(len(stacked)),
        "countries_or_territories": int(stacked[ENTITY].nunique()),
        "r_squared": float(fitted.rsquared),
        "adjusted_r_squared": float(fitted.rsquared_adj),
        "fixed_effects": "Outcome-specific country and year fixed effects",
        "covariance": "Country-clustered",
        "multiple_testing": "Benjamini-Hochberg across thirty pairwise outcome contrasts",
    }
    return result, metadata


def run(input_path=None, output_dir=None):
    panel = load_panel(input_path)
    destination = ensure_output_dir(output_dir)
    tables = []
    summaries = []
    for lag in [0, 1]:
        coefficients, metadata = fit_model(panel, lag)
        tables.append(coefficients)
        summaries.append(metadata)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(destination / "within_country_coefficients.csv", index=False)
    (destination / "within_country_model_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
