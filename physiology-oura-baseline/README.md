# physiology-oura-baseline

<!-- BADGES:BEGIN -->
[![CI](https://img.shields.io/github/actions/workflow/status/OWNER/REPO/ci.yml?branch=main)](https://github.com/OWNER/REPO/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- BADGES:END -->

Wearable-derived baseline pipeline with schema checks, reproducible evaluation, and a model report card.

## Quickstart
```bash
make env
pytest -q
make schema-check
make report-card
```

- Schema: [`schema/oura_timeseries.schema.json`](schema/oura_timeseries.schema.json)
- Model card: [`reports/model_card.md`](reports/model_card.md)

## Project structure
- `data/` synthetic sample timeseries
- `scripts/run_analysis.py` feature engineering + modeling helpers
- `tools/` schema validation and report-card rendering utilities
- `tests/` unit tests covering schema adherence and metrics sanity

## Outputs
Generated metrics live in `outputs/metrics.json` (created by `train_eval`). Render the model card with `make report-card`.
