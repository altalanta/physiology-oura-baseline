# Contributing

Thanks for helping improve the physiology-oura-baseline project!

## Prereqs
- Python 3.10+
- `make` (optional but recommended)
- Recommended packages listed in `requirements.txt`

## Environment
```bash
make env
```
This will create `.venv/` and install dependencies.

## Run tests
```bash
make test
```
Add unit tests under `tests/` whenever you add or modify logic (feature engineering, modeling, schema tools, etc.).

## Code style
- Follow PEP8/black-style formatting.
- Prefer explicit, typed helper functions.
- Keep notebooks in `notebooks/` and avoid committing large outputs.

## Commit messages
- Use concise, imperative subject lines (e.g., `feat: add subgroup calibration test`).
- Reference issues/links where relevant.

## Branching
- Create topic branches from `main`.
- Keep branches focused and short-lived.
- Rebase onto `main` before opening a PR when possible.

## PR checklist
- [ ] `make test`
- [ ] `make schema-check`
- [ ] `make report-card`
- [ ] Updated docs/model card as needed
- [ ] Added/updated unit tests
- [ ] Added or updated schema documentation if data contract changed
