# Model Card: physiology-oura-baseline

- Generated: 2025-09-19T19:14:39.835895Z
- Commit: `60694b1`

## Overview
Lightweight wearable-readiness baseline model trained on synthetic Oura-style features.

## Intended Use
- Educational/demo of power-aware wellness scoring.
- Not for medical diagnosis or treatment decisions.

## Data
- Source: `data/sample_wearable_timeseries.csv`
- Schema: `schema/oura_timeseries.schema.json`
- Target: readiness score >= 75

## Features
- steps, sleep hours, resting heart rate, HRV, temperature deviation
- engineered features: sleep deficit, absolute temperature deviation, log HRV

## Training
- Model: LogisticRegression (scikit-learn)
- Train/test split with random_state=42

## Metrics
- **brier**: 0.0002
- **roc_auc**: 1.0000
- **pr_auc**: 1.0000
- **reliability_slope**: 1.0133
- **reliability_intercept**: -0.0133
- **subgroup_delta**: -0.6404
- reliability bins: 10

## Assumptions
- Synthetic data approximates realistic ranges.
- Logistic link adequate for binary readiness classification.

## Limitations
- No personalization or temporal modeling.
- Synthetic quality may not reflect real-device noise.

## Ethical / Privacy Notes
- No PII; data synthetic. Always safeguard real wearable exports.

## Versioning
- Metrics source: `outputs/metrics.json`
