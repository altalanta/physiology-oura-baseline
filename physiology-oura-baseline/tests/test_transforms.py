import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_analysis import build_features, train_eval

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "oura_timeseries.schema.json"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_wearable_timeseries.csv"


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_on_sample():
    schema = load_schema()
    df = pd.read_csv(DATA_PATH)
    records = df.to_dict(orient="records")
    validator = Draft7Validator({"type": "array", "items": schema})
    errors = list(validator.iter_errors(records))
    assert not errors, f"Schema violations: {errors[:3]}"


def test_feature_qc_no_nans():
    df = pd.read_csv(DATA_PATH)
    X, y = build_features(df)
    assert not np.isnan(X.to_numpy()).any()
    assert len(X) == len(y)


def test_calibration_sanity(tmp_path):
    df = pd.read_csv(DATA_PATH)
    metrics = train_eval(df, out_dir=tmp_path)
    assert 0 <= metrics["brier"] <= 1
    assert metrics["reliability_bins"] == 10
    assert np.isfinite(metrics["reliability_slope"]) or np.isnan(metrics["reliability_slope"])


def test_subgroup_metrics(tmp_path):
    df = pd.read_csv(DATA_PATH)
    metrics = train_eval(df, out_dir=tmp_path)
    assert "subgroup_delta" in metrics
    assert np.isfinite(metrics["subgroup_delta"]) or np.isnan(metrics["subgroup_delta"])
