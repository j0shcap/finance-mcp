.PHONY: install test test-live e2e snapshot lint format typecheck security check build run clean

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

# The built wheel, installed into a fresh venv and driven over stdio (tests/e2e/): the
# artifact users run, not the source tree. Offline apart from the package index - the
# Yahoo-backed e2e tests are also `live`, so they run with test-live instead. --no-cov
# because the server under test runs in a subprocess, out of coverage's reach.
e2e:
	uv run pytest -m "e2e and not live" --no-cov

# Rewrite the committed model-facing contract (tests/snapshots/contract/) from the server,
# after an intended change to a tool, prompt, description or schema - or a fastmcp, mcp or
# pydantic bump. Review the result with `git diff --word-diff` before committing it.
snapshot:
	uv run pytest tests/test_contract_snapshot.py --update-snapshots --no-cov -q -rs

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
