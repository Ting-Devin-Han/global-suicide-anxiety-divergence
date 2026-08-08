import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabpfn import TabPFNRegressor

from common import ENTITY, OUTCOMES, PREDICTORS, TIME, ensure_output_dir, load_panel


RANDOM_SEED = 20260723
BACKTEST_CUTOFFS = [2009, 2014, 2016]
FINAL_YEAR = 2030


@dataclass
class FittedOutcomeModel:
    label: str
    outcome: str
    model: TabPFNRegressor
    feature_names: list[str]
    early_stats: pd.DataFrame
    cutoff: int
    lower_bound: float
    upper_bound: float


def linear_slope(years, values):
    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    return float(np.polyfit(years, values, 1)[0])


def projected_context(panel, cutoff, end_year):
    history = panel[panel[TIME] <= cutoff].copy()
    panel_sd = history[PREDICTORS].std(ddof=0)
    bounds = {}
    raw_slopes = {column: {} for column in PREDICTORS}
    for column in PREDICTORS:
        values = history[column].to_numpy(dtype=float)
        low, high = np.quantile(values, [0.005, 0.995])
        span = max(high - low, panel_sd[column], 1e-9)
        if column == "economic_development":
            bounds[column] = (0.0, high + 0.50 * span)
        elif column == "life_expectancy":
            bounds[column] = (max(20.0, low - 0.25 * span), min(100.0, high + 0.25 * span))
        else:
            bounds[column] = (low - 0.50 * span, high + 0.50 * span)
        for code, group in history.groupby(ENTITY, sort=False):
            tail = group.nlargest(min(10, len(group)), TIME).sort_values(TIME)
            outcome = tail[column].to_numpy(dtype=float)
            if column == "economic_development":
                outcome = np.log1p(np.clip(outcome, 0, None))
            raw_slopes[column][code] = linear_slope(tail[TIME], outcome)
    clipped_slopes = {column: {} for column in PREDICTORS}
    for column in PREDICTORS:
        values = np.asarray(list(raw_slopes[column].values()), dtype=float)
        low, high = np.quantile(values, [0.05, 0.95])
        for code, value in raw_slopes[column].items():
            clipped_slopes[column][code] = float(np.clip(value, low, high))
    records = []
    last_rows = history.sort_values(TIME).groupby(ENTITY, as_index=False).tail(1)
    for _, last in last_rows.iterrows():
        code = last[ENTITY]
        for year in range(cutoff + 1, end_year + 1):
            horizon = year - cutoff
            record = {
                ENTITY: code,
                "country_name": last["country_name"],
                "continent": last["continent"],
                "global_north_south_group": last["global_north_south_group"],
                TIME: year,
            }
            for column in PREDICTORS:
                slope = clipped_slopes[column][code]
                if column == "economic_development":
                    value = np.expm1(np.log1p(max(float(last[column]), 0.0)) + slope * horizon)
                else:
                    value = float(last[column]) + slope * horizon
                record[column] = float(np.clip(value, *bounds[column]))
            records.append(record)
    return pd.DataFrame(records)


def early_country_stats(panel, outcome, cutoff):
    end = min(cutoff, 2004)
    early = panel[(panel[TIME] >= 2000) & (panel[TIME] <= end)].copy()
    records = []
    for code, group in early.groupby(ENTITY, sort=False):
        values = group[outcome].to_numpy(dtype=float)
        records.append(
            {
                ENTITY: code,
                "early_level": float(values[0]),
                "early_mean": float(np.mean(values)),
                "early_slope": linear_slope(group[TIME], values),
            }
        )
    return pd.DataFrame(records).set_index(ENTITY)


def static_features(panel):
    metadata = panel.sort_values(TIME).groupby(ENTITY, as_index=False).first()[[ENTITY, "continent", "global_north_south_group"]].set_index(ENTITY)
    continents = pd.get_dummies(metadata["continent"], prefix="continent", dtype=float)
    metadata = pd.concat([metadata, continents], axis=1)
    metadata["global_north"] = metadata["global_north_south_group"].str.contains("North", case=False, na=False).astype(float)
    return metadata.drop(columns=["continent", "global_north_south_group"])


def feature_row(code, target_year, outcome_history, context, early_stats, static):
    history = outcome_history[code]
    required_years = [target_year - offset for offset in [1, 2, 3, 4, 5]]
    if any(year not in history for year in required_years):
        raise ValueError(f"Insufficient outcome history for {code}, target {target_year}")
    levels = np.asarray([history[year] for year in required_years], dtype=float)
    recent_years = np.asarray(required_years[::-1], dtype=float)
    recent_values = levels[::-1]
    current_context = context.loc[(code, target_year)]
    previous_context = context.loc[(code, target_year - 1)]
    record = {
        "time_saturation": min(max((target_year - 2000) / 20.0, 0.0), 1.0),
        "level_lag1": levels[0],
        "level_lag2": levels[1],
        "level_lag3": levels[2],
        "level_lag5": levels[4],
        "change_lag1": levels[0] - levels[1],
        "slope_3yr": linear_slope(recent_years[-3:], recent_values[-3:]),
        "slope_5yr": linear_slope(recent_years, recent_values),
        "mean_5yr": float(np.mean(recent_values)),
        "sd_5yr": float(np.std(recent_values, ddof=0)),
        "early_level": float(early_stats.loc[code, "early_level"]),
        "early_mean": float(early_stats.loc[code, "early_mean"]),
        "early_slope": float(early_stats.loc[code, "early_slope"]),
    }
    for column in PREDICTORS:
        record[f"context_{column}"] = float(current_context[column])
        record[f"context_change_{column}"] = float(current_context[column] - previous_context[column])
    for column, value in static.loc[code].items():
        record[column] = float(value)
    return record


def history_dictionary(panel, outcome, end_year):
    history = {}
    for code, group in panel[panel[TIME] <= end_year].groupby(ENTITY, sort=False):
        history[code] = dict(zip(group[TIME].astype(int), group[outcome].astype(float)))
    return history


def context_lookup(panel, cutoff, end_year):
    historical = panel[panel[TIME] <= cutoff][[ENTITY, TIME, *PREDICTORS]].copy()
    future = projected_context(panel, cutoff, end_year)
    combined = pd.concat([historical, future[[ENTITY, TIME, *PREDICTORS]]], ignore_index=True)
    return combined.set_index([ENTITY, TIME]).sort_index(), future


def training_matrix(panel, label, outcome, cutoff):
    historical = panel[panel[TIME] <= cutoff].copy()
    context = historical.set_index([ENTITY, TIME])[PREDICTORS].sort_index()
    history = history_dictionary(historical, outcome, cutoff)
    early = early_country_stats(historical, outcome, cutoff)
    static = static_features(historical)
    records = []
    targets = []
    for code in sorted(history):
        for target_year in range(2005, cutoff + 1):
            records.append(feature_row(code, target_year, history, context, early, static))
            targets.append(history[code][target_year] - history[code][target_year - 1])
    features = pd.DataFrame(records)
    target = np.asarray(targets, dtype=float)
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target).all():
        raise ValueError(f"Non-finite training values for {label}")
    return features, target, early


def fit_outcome_model(panel, label, outcome, cutoff, device, estimators):
    features, target, early = training_matrix(panel, label, outcome, cutoff)
    model = TabPFNRegressor(
        n_estimators=estimators,
        random_state=RANDOM_SEED + cutoff + (0 if label == "Suicide" else 1000),
        device=device,
        ignore_pretraining_limits=True,
        fit_mode="fit_preprocessors",
        memory_saving_mode="auto",
    )
    started = time.time()
    model.fit(features, target)
    print(f"Fitted {label} through {cutoff}: n={len(features)}, p={features.shape[1]}, seconds={time.time() - started:.1f}")
    observed = panel.loc[panel[TIME] <= cutoff, outcome].to_numpy(dtype=float)
    upper = float(max(np.quantile(observed, 0.999) * 1.35, observed.max() * 1.05))
    return FittedOutcomeModel(label, outcome, model, features.columns.tolist(), early, cutoff, 0.0, upper)


def forecast_outcome(panel, fitted, end_year):
    cutoff = fitted.cutoff
    history = history_dictionary(panel, fitted.outcome, cutoff)
    static = static_features(panel[panel[TIME] <= cutoff])
    context, future_context = context_lookup(panel, cutoff, end_year)
    metadata = panel[panel[TIME] == cutoff][[ENTITY, "country_name", "continent"]].drop_duplicates(ENTITY).set_index(ENTITY)
    records = []
    for year in range(cutoff + 1, end_year + 1):
        codes = sorted(history)
        features = pd.DataFrame([feature_row(code, year, history, context, fitted.early_stats, static) for code in codes])
        features = features[fitted.feature_names]
        changes = np.asarray(fitted.model.predict(features, output_type="mean"), dtype=float)
        previous = np.asarray([history[code][year - 1] for code in codes], dtype=float)
        predictions = np.clip(previous + changes, fitted.lower_bound, fitted.upper_bound)
        for code, prediction, change in zip(codes, predictions, changes):
            history[code][year] = float(prediction)
            records.append(
                {
                    ENTITY: code,
                    "country_name": metadata.loc[code, "country_name"],
                    "continent": metadata.loc[code, "continent"],
                    TIME: year,
                    "outcome": fitted.label,
                    "prediction": float(prediction),
                    "predicted_annual_change": float(change),
                }
            )
    return pd.DataFrame(records), future_context


def persistence_forecast(panel, label, outcome, cutoff, end_year):
    last = panel[panel[TIME] == cutoff].set_index(ENTITY)
    records = []
    for code, row in last.iterrows():
        for year in range(cutoff + 1, end_year + 1):
            records.append({ENTITY: code, TIME: year, "outcome": label, "prediction": float(row[outcome]), "model": "Persistence"})
    return pd.DataFrame(records)


def linear_trend_forecast(panel, label, outcome, cutoff, end_year):
    historical = panel[panel[TIME] <= cutoff]
    upper = float(historical[outcome].max() * 1.35)
    records = []
    for code, group in historical.groupby(ENTITY, sort=False):
        tail = group.nlargest(min(10, len(group)), TIME).sort_values(TIME)
        trend = linear_slope(tail[TIME], tail[outcome])
        last = float(tail.iloc[-1][outcome])
        for year in range(cutoff + 1, end_year + 1):
            prediction = float(np.clip(last + trend * (year - cutoff), 0.0, upper))
            records.append({ENTITY: code, TIME: year, "outcome": label, "prediction": prediction, "model": "Linear trend"})
    return pd.DataFrame(records)


def score_predictions(predictions, actual, cutoff, model):
    merged = predictions.merge(actual[[ENTITY, TIME, "outcome", "actual"]], on=[ENTITY, TIME, "outcome"], how="left", validate="one_to_one")
    merged["cutoff"] = cutoff
    merged["horizon"] = merged[TIME] - cutoff
    merged["model"] = model
    merged["error"] = merged["actual"] - merged["prediction"]
    records = []
    for (outcome, horizon), group in merged.groupby(["outcome", "horizon"], sort=True):
        interquartile_range = float(np.subtract(*np.quantile(group["actual"], [0.75, 0.25])))
        rmse = float(mean_squared_error(group["actual"], group["prediction"]) ** 0.5)
        records.append(
            {
                "cutoff": cutoff,
                "horizon": int(horizon),
                "outcome": outcome,
                "model": model,
                "n": int(len(group)),
                "mae": float(mean_absolute_error(group["actual"], group["prediction"])),
                "rmse": rmse,
                "nrmse_iqr": rmse / interquartile_range if interquartile_range > 0 else np.nan,
                "bias_actual_minus_prediction": float(group["error"].mean()),
                "r2": float(r2_score(group["actual"], group["prediction"])),
            }
        )
    return merged, records


def run_backtests(panel, device, estimators):
    actual = panel.melt(id_vars=[ENTITY, TIME], value_vars=list(OUTCOMES.values()), var_name="outcome_variable", value_name="actual")
    actual["outcome"] = actual["outcome_variable"].map({value: key for key, value in OUTCOMES.items()})
    prediction_tables = []
    summary_records = []
    for cutoff in BACKTEST_CUTOFFS:
        end_year = min(2019, cutoff + 10)
        for label, outcome in OUTCOMES.items():
            fitted = fit_outcome_model(panel, label, outcome, cutoff, device, estimators)
            tabpfn, _ = forecast_outcome(panel, fitted, end_year)
            tabpfn = tabpfn[[ENTITY, TIME, "outcome", "prediction"]].copy()
            tabpfn["model"] = "TabPFN"
            candidates = [
                ("TabPFN", tabpfn),
                ("Persistence", persistence_forecast(panel, label, outcome, cutoff, end_year)),
                ("Linear trend", linear_trend_forecast(panel, label, outcome, cutoff, end_year)),
            ]
            for model, prediction in candidates:
                scored, summary = score_predictions(prediction, actual, cutoff, model)
                prediction_tables.append(scored)
                summary_records.extend(summary)
    return pd.concat(prediction_tables, ignore_index=True), pd.DataFrame(summary_records)


def run(input_path=None, output_dir=None, device="cuda", estimators=8, skip_backtests=False, final_year=FINAL_YEAR):
    panel = load_panel(input_path)
    if "global_north_south_group" not in panel.columns:
        raise ValueError("Projection requires global_north_south_group")
    destination = ensure_output_dir(output_dir)
    metadata = {
        "model": "TabPFNRegressor panel autoregression of annual outcome changes",
        "random_seed": RANDOM_SEED,
        "estimators": estimators,
        "device": device,
        "backtest_cutoffs": BACKTEST_CUTOFFS,
        "projection_years": [2020, final_year],
        "context_continuation": "Clipped country-specific ten-year linear trends",
    }
    if not skip_backtests:
        predictions, summary = run_backtests(panel, device, estimators)
        predictions.to_csv(destination / "projection_backtest_predictions.csv", index=False)
        summary.to_csv(destination / "projection_backtest_summary.csv", index=False)
    projection_tables = []
    context_tables = []
    for label, outcome in OUTCOMES.items():
        fitted = fit_outcome_model(panel, label, outcome, 2019, device, estimators)
        projection, context = forecast_outcome(panel, fitted, final_year)
        projection_tables.append(projection)
        context_tables.append(context)
    projections = pd.concat(projection_tables, ignore_index=True)
    future_context = pd.concat(context_tables, ignore_index=True).drop_duplicates([ENTITY, TIME])
    projections.to_csv(destination / "continuation_country_projections.csv", index=False)
    future_context.to_csv(destination / "projection_future_context.csv", index=False)
    (destination / "projection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return projections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--estimators", type=int, default=8)
    parser.add_argument("--skip-backtests", action="store_true")
    parser.add_argument("--final-year", type=int, default=FINAL_YEAR)
    args = parser.parse_args()
    result = run(args.input, args.output_dir, args.device, args.estimators, args.skip_backtests, args.final_year)
    print(result.tail(20).to_string(index=False))


if __name__ == "__main__":
    main()

