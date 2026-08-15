# Data directory

## English

Place the analysis-ready panel at:

`data/input/main_country_year_panel_200_2000_2019_FINAL.csv`

The file must contain one row per country or territory and year for 200 geographical units over 2000–2019. The required fields are:

- Identifiers: `country_code`, `country_name`, `year`, `continent`, and `global_north_south_group`.
- Outcomes: `suicide_rate`, `anxiety_disorder_prevalence`, and `depression_as_prevalence`.
- Contextual dimensions: `economic_development`, `social_inequality_index`, `education_human_capital_index`, `life_expectancy`, `population_growth`, `urban_activity`, `resource_heat_pressure_index`, `ecological_natural_exposure_index`, `built_environment_index`, and `policy_social_context_index`.

The analysis-ready data are not committed to this repository because the component datasets remain subject to their original provider terms. The repository does not treat repeated rows from different source files as independent evidence records. In particular, governance and happiness source counts must not be obtained by summing overlapping hierarchy levels, and age-standardized suicide-rate observations must not be described as direct suicide-prevention policy evidence.

All generated analysis outputs are written to `data/output/`.
