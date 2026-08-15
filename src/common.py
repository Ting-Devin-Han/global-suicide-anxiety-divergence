from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ENTITY = "country_code"
TIME = "year"
EXPECTED_COUNTRIES = 200
EXPECTED_YEARS = tuple(range(2000, 2020))
OUTCOMES = {
    "Suicide": "suicide_rate",
    "Anxiety": "anxiety_disorder_prevalence",
    "Depression": "depression_as_prevalence",
}
FATAL_OUTCOME = "Suicide"
NONFATAL_OUTCOMES = ("Anxiety", "Depression")
PREDICTORS = [
    "economic_development",
    "social_inequality_index",
    "education_human_capital_index",
    "life_expectancy",
    "population_growth",
    "urban_activity",
    "resource_heat_pressure_index",
    "ecological_natural_exposure_index",
    "built_environment_index",
    "policy_social_context_index",
]
PREDICTOR_LABELS = {
    "economic_development": "Economic development",
    "social_inequality_index": "Social inequality",
    "education_human_capital_index": "Education and human capital",
    "life_expectancy": "Life expectancy",
    "population_growth": "Population growth",
    "urban_activity": "Activity",
    "resource_heat_pressure_index": "Pollution",
    "ecological_natural_exposure_index": "Ecological exposure",
    "built_environment_index": "Built-up surface",
    "policy_social_context_index": "Policy-social context",
}


def repository_root():
    return Path(__file__).resolve().parents[1]


def default_input_path():
    return repository_root() / "data" / "input" / "main_country_year_panel_200_2000_2019_FINAL.csv"


def default_output_dir():
    return repository_root() / "data" / "output"


def ensure_output_dir(path=None):
    destination = Path(path) if path else default_output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def zscore(values):
    series = pd.Series(values, copy=False).astype(float)
    scale = series.std(ddof=0)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / scale


def bh_adjust(values):
    array = np.asarray(values, dtype=float)
    adjusted = np.full(array.shape, np.nan)
    valid = np.isfinite(array)
    if valid.any():
        adjusted[valid] = multipletests(array[valid], method="fdr_bh")[1]
    return adjusted


def load_panel(path=None):
    source = Path(path) if path else default_input_path()
    panel = pd.read_csv(source)
    required = [
        ENTITY,
        "country_name",
        TIME,
        "continent",
        *OUTCOMES.values(),
        *PREDICTORS,
    ]
    missing_columns = [column for column in required if column not in panel.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    panel[TIME] = panel[TIME].astype(int)
    panel = panel.sort_values([ENTITY, TIME]).reset_index(drop=True)
    if panel.duplicated([ENTITY, TIME]).any():
        raise ValueError("Duplicate country-year rows were found")
    if panel[required].isna().any().any():
        missing = panel[required].isna().sum()
        raise ValueError(f"Missing values were found: {missing[missing > 0].to_dict()}")
    years = tuple(sorted(panel[TIME].unique()))
    counts = panel.groupby(ENTITY)[TIME].nunique()
    if panel[ENTITY].nunique() != EXPECTED_COUNTRIES or years != EXPECTED_YEARS:
        raise ValueError("The fixed analysis requires 200 countries or territories observed from 2000 through 2019")
    if counts.nunique() != 1 or counts.iloc[0] != len(EXPECTED_YEARS):
        raise ValueError("The fixed analysis panel must contain 20 annual observations for every country or territory")
    return panel
