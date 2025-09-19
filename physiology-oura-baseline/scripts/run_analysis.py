"""Lightweight feature engineering and evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "steps",
    "sleep_hours",
    "rhr",
    "hrv",
    "temp_dev",
]
TARGET_COLUMN = "readiness_score"
TARGET_THRESHOLD = 75


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return engineered features X and binary target y."""
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    X = df[FEATURE_COLUMNS].copy()
    # scale basic features
    X["sleep_deficit"] = np.clip(7.0 - df["sleep_hours"], 0, None)
    X["temp_dev_abs"] = df["temp_dev"].abs()
    X["hrv_log"] = np.log1p(df["hrv"])
    y = (df[TARGET_COLUMN] >= TARGET_THRESHOLD).astype(int)
    return X, y


def _reliability_stats(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    if len(frac_pos) > 1:
        slope, intercept = np.polyfit(mean_pred, frac_pos, deg=1)
    else:
        slope, intercept = float("nan"), float("nan")
    return {
        "reliability_bins": n_bins,
        "reliability_slope": float(slope),
        "reliability_intercept": float(intercept),
    }


def train_eval(df: pd.DataFrame, out_dir: Path | None = None, seed: int = 42) -> Dict[str, float]:
    """Train a logistic model and return evaluation metrics."""
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=min(0.4, max(0.2, 4 / len(X))),
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "brier": float(brier_score_loss(y_test, probs)),
        "roc_auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_test, probs)),
    }
    metrics.update(_reliability_stats(y_test.to_numpy(), probs))

    low_sleep_mask = X_test["sleep_hours"] < 6.0
    subgroup_mean_low = float(probs[low_sleep_mask].mean()) if low_sleep_mask.any() else float("nan")
    subgroup_mean_rest = float(probs[~low_sleep_mask].mean()) if (~low_sleep_mask).any() else float("nan")
    metrics["subgroup_delta"] = float(subgroup_mean_low - subgroup_mean_rest)

    if out_dir is None:
        out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:  # pragma: no cover
    df = pd.read_csv(Path("data") / "sample_wearable_timeseries.csv")
    metrics = train_eval(df)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
