"""Validate wearable timeseries CSV against schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from jsonschema import Draft7Validator

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "oura_timeseries.schema.json"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "sample_wearable_timeseries.csv"


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_file(csv_path: Path, schema: dict) -> int:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")
    array_schema = {"type": "array", "items": schema}
    validator = Draft7Validator(array_schema)
    errors = list(validator.iter_errors(records))
    if errors:
        print(f"Found {len(errors)} schema violations in {csv_path}:")
        for err in errors[:5]:
            print(f" - {err.message} at {list(err.path)}")
        return 1
    print(f"Schema validation passed for {csv_path} ({len(records)} rows).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate wearable CSV against schema")
    parser.add_argument("--in", dest="input_path", default=str(DEFAULT_DATA), help="CSV file to validate")
    args = parser.parse_args()
    schema = load_schema(SCHEMA_PATH)
    exit(validate_file(Path(args.input_path), schema))


if __name__ == "__main__":  # pragma: no cover
    main()
