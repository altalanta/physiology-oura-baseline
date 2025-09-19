"""Render model card from metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

METRICS_PATH = Path("outputs/metrics.json")
CARD_PATH = Path("reports/model_card.md")


TEMPLATE = """# Model Card: physiology-oura-baseline

- Generated: {timestamp}
- Commit: `{commit}`

## Overview
Lightweight wearable-readiness baseline model trained on synthetic Oura-style features.

## Intended Use
- Educational/demo of power-aware wellness scoring.
- Not for medical diagnosis or treatment decisions.

## Data
- Source: `data/sample_wearable_timeseries.csv`
- Schema: `schema/oura_timeseries.schema.json`
- Target: readiness score >= {threshold}

## Features
- steps, sleep hours, resting heart rate, HRV, temperature deviation
- engineered features: sleep deficit, absolute temperature deviation, log HRV

## Training
- Model: LogisticRegression (scikit-learn)
- Train/test split with random_state=42

## Metrics
{metrics_block}

## Assumptions
- Synthetic data approximates realistic ranges.
- Logistic link adequate for binary readiness classification.

## Limitations
- No personalization or temporal modeling.
- Synthetic quality may not reflect real-device noise.

## Ethical / Privacy Notes
- No PII; data synthetic. Always safeguard real wearable exports.

## Versioning
- Metrics source: `{metrics_path}`
"""


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_metrics(metrics: dict | None) -> str:
    if metrics is None:
        return "Metrics not yet generated. Run `python -m scripts.run_analysis` first."
    lines = []
    for key in ["brier", "roc_auc", "pr_auc", "reliability_slope", "reliability_intercept", "subgroup_delta"]:
        value = metrics.get(key, "n/a")
        if isinstance(value, float):
            lines.append(f"- **{key}**: {value:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")
    bins = metrics.get("reliability_bins")
    if bins is not None:
        lines.append(f"- reliability bins: {bins}")
    return "\n".join(lines)


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render model card")
    parser.add_argument("--metrics", default=str(METRICS_PATH), help="Metrics JSON path")
    parser.add_argument("--out", default=str(CARD_PATH), help="Model card output path")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics))
    timestamp = datetime.utcnow().isoformat() + "Z"
    content = TEMPLATE.format(
        timestamp=timestamp,
        commit=git_sha(),
        metrics_block=format_metrics(metrics),
        metrics_path=args.metrics,
        threshold=75,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Model card written to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
