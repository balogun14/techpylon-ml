.PHONY: install format lint typecheck test check-all run

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
