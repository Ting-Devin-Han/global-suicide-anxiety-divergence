import argparse
import json
from pathlib import Path

from common import ENTITY, OUTCOMES, PREDICTORS, TIME, ensure_output_dir, load_panel


def validate(input_path=None, output_dir=None):
    panel = load_panel(input_path)
    years = sorted(panel[TIME].unique().tolist())
    country_counts = panel.groupby(ENTITY)[TIME].nunique()
    audit = {
        "rows": int(len(panel)),
        "countries_or_territories": int(panel[ENTITY].nunique()),
        "start_year": int(min(years)),
        "end_year": int(max(years)),
        "number_of_years": int(len(years)),
        "balanced_panel": bool(country_counts.nunique() == 1 and country_counts.iloc[0] == len(years)),
        "duplicate_country_year_rows": int(panel.duplicated([ENTITY, TIME]).sum()),
        "missing_analysis_values": int(panel[[*OUTCOMES.values(), *PREDICTORS]].isna().sum().sum()),
        "outcomes": list(OUTCOMES.values()),
        "contextual_dimensions": PREDICTORS,
    }
    destination = ensure_output_dir(output_dir) / "panel_audit.json"
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = validate(args.input, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

