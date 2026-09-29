.PHONY: install test test-live lint format typecheck security check build run clean

install:
	uv sync

test:
	uv run pytest

# The opt-in contract suite against real Yahoo endpoints (tests/live/). `-m live` overrides
# the `-m 'not live'` in addopts, and --no-cov drops the coverage gate: these tests exercise
# the source's agreement with a live third party, not its line coverage, so counting them
# would inflate the same percentage `make test` already gates. Not part of `make check`,
# which stays offline; CI runs this nightly (.github/workflows/live.yml).
test-live:
	uv run pytest -m live --no-cov

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

security:
	uv run bandit -c pyproject.toml -r src

check: lint typecheck security test

build:
	uv build

run:
	uv run mcp-finance

clean:
	rm -rf .coverage .mypy_cache .pytest_cache .ruff_cache htmlcov dist build
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
