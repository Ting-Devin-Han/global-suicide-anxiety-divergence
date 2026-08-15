import argparse
from math import exp, lgamma, log, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln, hyp2f1
from scipy.stats import spearmanr

from common import ENTITY, OUTCOMES, ensure_output_dir, load_panel


PRIOR_SCALES = {
    "Medium-narrow": 1 / sqrt(27),
    "Medium": 1 / 3,
    "Wide": 1 / sqrt(3),
    "Ultrawide": 1.0,
}


def bayesfactor_pearson_ly(correlation, sample_size, kappa):
    if not np.isfinite(correlation) or sample_size < 3 or not -1 <= correlation <= 1:
        raise ValueError("Invalid correlation or sample size")
    hyperterm = hyp2f1(
        (sample_size - 1) / 2,
        (sample_size - 1) / 2,
        (sample_size + 2 / kappa) / 2,
        correlation**2,
    )
    if not np.isfinite(hyperterm) or hyperterm <= 0:
        raise ValueError("Bayes-factor calculation did not converge")
    return exp(
        (1 - 2 / kappa) * log(2)
        + 0.5 * log(pi)
        - betaln(1 / kappa, 1 / kappa)
        + lgamma((sample_size + 2 / kappa - 1) / 2)
        - lgamma((sample_size + 2 / kappa) / 2)
        + log(hyperterm)
    )


def calculate(panel):
    means = panel.groupby([ENTITY, "country_name", "continent"], as_index=False)[list(OUTCOMES.values())].mean()
    means = means.rename(columns={column: label for label, column in OUTCOMES.items()})
    records = []
    labels = list(OUTCOMES)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            correlation, p_value = spearmanr(means[left], means[right])
            for prior_name, kappa in PRIOR_SCALES.items():
                bf10 = bayesfactor_pearson_ly(float(correlation), len(means), kappa)
                records.append(
                    {
                        "outcome_1": left,
                        "outcome_2": right,
                        "countries_or_territories": int(len(means)),
                        "spearman_rho": float(correlation),
                        "two_sided_p": float(p_value),
                        "prior_scale": prior_name,
                        "kappa": float(kappa),
                        "bf10": float(bf10),
                        "bf01": float(1 / bf10),
                    }
                )
    return means, pd.DataFrame(records)


def run(input_path=None, output_dir=None):
    destination = ensure_output_dir(output_dir)
    means, statistics = calculate(load_panel(input_path))
    means.to_csv(destination / "country_long_term_outcome_means.csv", index=False)
    statistics.to_csv(destination / "bayesian_rank_concordance.csv", index=False)
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.output_dir)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
