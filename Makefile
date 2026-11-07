.PHONY: install format lint typecheck test check-all run bench calibrate train-models validate

install:
	uv sync

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

typecheck:
	uv run mypy extractor

test:
	uv run pytest

check-all: format lint typecheck test

run:
	uv run uvicorn extractor.service:app --port 8100

bench:
	uv run python -m bench.harness

calibrate:
	uv run python -m extractor.calibrate

train-models:
	uv run python -m extractor.train_models

# bench + calibration + model card + validation report in one pass
validate: bench calibrate
	uv run python -c "from bench.harness import run_bench; from extractor.calibrate import load_calibration; from extractor.artefacts import write_artefacts; b = run_bench(); print(write_artefacts(b.model_dump() if hasattr(b, 'model_dump') else b, load_calibration()))"
