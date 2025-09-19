.PHONY: env test lint schema-check report-card

env:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

test:
	. .venv/bin/activate || true; pytest -q

lint:
	echo "No lint step configured; add your preferred linter."

schema-check:
	python -m tools.schema_check

report-card:
	python -m tools.render_card
