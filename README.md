# Global Suicide–Anxiety Divergence

Core reproducibility code for the cross-national study of divergent suicide mortality and anxiety-disorder burden, 2000–2030.

### Scope

This repository contains the core data-processing and statistical-analysis code used in the manuscript.

The repository reproduces:

- validation of the 231 × 20 country-year panel;
- country-specific observed trends and bootstrap intervals;
- long-term outcome profiles and trend concordance;
- the composite burden ranking based on 60% long-term joint burden and 40% joint increase;
- paired cross-country contextual models for the two outcomes;
- contemporaneous and one-year-lagged two-way fixed-effects models;
- exploratory country-specific contextual associations;
- temporally backtested TabPFN recursive projections through 2030;
- empirically calibrated attenuated, continuation, and accelerated divergence scenarios; and
- observed and projected regional heterogeneity summaries.

### Repository structure

```text
global-suicide-anxiety-divergence/
├── CODE_AVAILABILITY.tex
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── input/
│   └── output/
└── src/
    ├── common.py
    ├── validate_panel.py
    ├── observed_trends.py
    ├── burden_rankings.py
    ├── long_term_contextual_models.py
    ├── within_country_models.py
    ├── country_specific_models.py
    ├── recursive_projection.py
    ├── projection_scenarios.py
    ├── regional_heterogeneity.py
    └── run_all.py
```

### Input data

Place the fixed analysis panel at:

`data/input/main_country_year_panel_231_2000_2019_FINAL.csv`

The required schema and data-boundary notes are provided in `data/README.md`. The data file is not included because the underlying sources retain their original access and redistribution conditions.

### Environment

Python 3.10.16 was used for the archived analysis environment. Create an isolated environment and install the pinned packages:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`. A CUDA-capable device is recommended for the full TabPFN projection workflow. A compatible PyTorch build may need to be installed from the official PyTorch index for the local CUDA version.

### Run the core statistical analysis

```bash
python src/run_all.py
```

This command runs all non-projection analyses and writes machine-readable outputs to `data/output/`.

To run the complete projection workflow with the original eight-estimator specification and strict temporal backtests:

```bash
python src/run_all.py --include-projection --device cuda --estimators 8
```

For a faster projection-only verification that omits repeated backtest fitting:

```bash
python src/run_all.py --include-projection --device cuda --estimators 8 --skip-backtests
```

Use `--device cpu` only when a supported CPU configuration is available; the full workflow is substantially slower without a GPU.

### Analysis-to-output map

| Analysis | Script | Main output |
|---|---|---|
| Panel audit | `validate_panel.py` | `panel_audit.json` |
| Observed outcome trends | `observed_trends.py` | `country_outcome_trends.csv`, `observed_trend_summary.csv` |
| Composite burden ranking | `burden_rankings.py` | `burden_rankings.csv` |
| Long-term cross-country model | `long_term_contextual_models.py` | `long_term_contextual_coefficients.csv` |
| Current and lagged fixed-effects models | `within_country_models.py` | `within_country_coefficients.csv` |
| Country-specific associations | `country_specific_models.py` | `country_specific_contextual_associations.csv` |
| Recursive 2030 continuation forecast and backtests | `recursive_projection.py` | `continuation_country_projections.csv`, `projection_backtest_summary.csv` |
| Projection scenarios | `projection_scenarios.py` | `scenario_summary.csv`, `scenario_country_projections.csv` |
| Regional heterogeneity | `regional_heterogeneity.py` | `observed_regional_heterogeneity.csv`, `projected_regional_heterogeneity.csv` |

### Reproducibility details

Observed country slopes use all 20 annual observations. The confidence interval for the median country slope uses 5,000 country-level bootstrap resamples with seed 42. Long-term paired models average outcomes and predictors over 2000–2019, standardize all variables across 231 countries or territories, cluster standard errors by country, and control the false-discovery rate across the ten outcome contrasts using Benjamini–Hochberg adjustment.

The within-country models jointly estimate both standardized outcomes with outcome-specific country and year fixed effects, all ten contextual predictors entered simultaneously, country-clustered standard errors, and the same ten-contrast false-discovery-rate correction. The lagged model shifts all contextual predictors by one year within country.

The projection model predicts annual outcome changes recursively. Its features include outcome levels at lags 1, 2, 3, and 5; the most recent annual change; three- and five-year slopes; five-year mean and variability; early-period level, mean, and slope; the ten contextual dimensions and their annual changes; continent indicators; and Global North status. Contextual covariates continue according to clipped country-specific ten-year slopes. Strict temporal backtests use cutoffs in 2009, 2014, and 2016 and compare TabPFN with persistence and country-specific linear-trend benchmarks.

Projection-scenario multipliers are calculated from the 25th, 50th, and 75th percentiles of the observed absolute difference between annual log changes in anxiety prevalence and suicide mortality. They are derived from the input panel at runtime rather than hard-coded.

### Creating the GitHub repository

After checking the contents and replacing the username placeholder:

```bash
git init
git add .
git commit -m "Release core reproducibility code"
gh repo create global-suicide-anxiety-divergence --public --source . --remote origin --push
```


