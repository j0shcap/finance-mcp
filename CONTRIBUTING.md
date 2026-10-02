# Contributing

Thanks for helping. Bug reports and feature ideas go in
[issues](https://github.com/j0shcap/finance-mcp/issues); security problems go through
[SECURITY.md](SECURITY.md), not a public issue.

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/j0shcap/finance-mcp
cd finance-mcp
make install        # uv sync: the package plus dev tools, from uv.lock
uv run pre-commit install   # optional: ruff, mypy and bandit on every commit
```

## Checks

| Command | What it runs | When |
| --- | --- | --- |
| `make check` | ruff, `mypy --strict`, bandit, and the offline test suite with the coverage gate | before every push; CI runs it on Python 3.13 and 3.14 |
| `make test` | just the offline test suite | while iterating |
| `make e2e` | builds the wheel and drives it over stdio | when touching packaging, the entry point or anything printed |
| `make test-live` | contract tests against real Yahoo endpoints | when touching `data/providers/yahoo.py` or `data/service.py` |
| `make snapshot` | rewrites the model-facing contract snapshot | after an intended change to anything a client sees |

`make check` never touches the network: tests mock yfinance through `tests/fakes.py`. Tests that
need the network or a built wheel are marked `live` or `e2e` and are deselected by default;
`--strict-markers` makes a misspelled marker an error.

## How we work

- **Tests first.** Write the failing test, watch it fail for the right reason, then make it pass.
  Bug fixes start with a regression test that reproduces the bug.
- **Coverage.** The gate in `pyproject.toml` fails below 90%; the suite sits at 100%, and new
  code should keep it there.
- **Units and signs.** Rates are decimals and the cashflow tools follow Excel's sign convention.
  Say per-period or annual, and which currency, in every new `Field` description; the shared
  wording lives in `src/finance_mcp/conventions.py`.
- **Errors.** Domain errors raise `InvalidInput`, `DataUnavailable` or `SymbolNotFound`, and tools
  translate them through `run_calc` / `run_data` (`tools/_dispatch.py`) so the message reaches the
  model. Don't swallow exceptions or return placeholder values: missing data is `null` or an
  error, never a guess.

## Providers and the logic layer

Market data reaches the tools through three layers, so a new data or news source is a new
adapter, not a change to the logic:

- **Ports** (`data/providers/ports.py`): `typing.Protocol` interfaces, one per source that could
  be replaced on its own (`MarketDataProvider`, `NewsProvider`). They speak only this package's
  models and errors.
- **Adapters** (`data/providers/`): one module per provider. `yahoo.py` (`YahooProvider`) holds
  every yfinance call, Yahoo field name and Yahoo quirk, and implements every port.
- **Logic** (`data/service.py`, `DataService`): what is fetched together, what is cached and
  for how long, and the pure computations (`analytics.py`, `performance.py`, `risk_free.py`,
  `relevance.py`). It never names a provider.

`server.py` is the composition root: it builds the providers and hands each port its
implementation. `tests/test_architecture.py` fails if anything else imports yfinance, its
transport, pandas or the Yahoo adapter. `tests/data/test_service_ports.py` runs the logic on a
plain in-memory provider.

## Yahoo payload shapes

`make record-shapes` records the *shape* of every kind of payload `data/providers/yahoo.py` parses -
index kinds and timezones, column dtypes, the type of each key it reads (found by scanning
`yahoo.py`), what an unknown symbol raises - into `tests/shapes/yahoo.json`, with none of
Yahoo's values. `tests/test_fakes_match_shapes.py` holds the fakes in `tests/fakes.py` to
it, so the unit suite parses realistic payloads. The nightly live job runs
`python -m scripts.record_shapes --check` and fails when Yahoo, yfinance or pandas starts returning
something new; then run `make record-shapes` (it merges with what is committed), update the
fakes the shape test names, and fix any parser that needs it.

## The contract snapshot

`tests/snapshots/contract/` holds everything a client receives that steers the model: the
server instructions, every tool's description, annotations and input/output schemas, every
prompt rendered with its `tests/prompt_samples.py` arguments, and the conventions resource.
`make check` fails on any difference. When the change is intended, run `make snapshot` and
review the result with `git diff --word-diff` before committing it. The `.md` files hold the
long prose (tool descriptions, rendered prompts) so it diffs line by line.

A `fastmcp`, `mcp` or `pydantic` bump can change generated schemas or appended text, and
that fails the snapshot too, on purpose. Dependabot groups those packages into one PR:
check it out, run `make snapshot`, review and push. Pushing to a Dependabot branch stops
its automatic rebases, which is fine for a one-off. `mcp` and `pydantic-core` are transitive,
so they usually move with a `fastmcp` or `pydantic` bump; a manual `uv lock --upgrade` can
move them too, and needs `make snapshot` like any other change to the lock.

## Adding a tool

1. Put the logic in `src/finance_mcp/data/` and its result model in `data/models.py`.
   Market-data models subclass `MarketData`, which rounds their output.
   - A calculator goes in `calculators.py` (pure math, no network).
   - A market-data tool is split across the layers above: a method on the matching port in
     `providers/ports.py`, its Yahoo fetch-and-parse in `providers/yahoo.py` (never cached),
     any computation over fetched data in a pure module, and the orchestration - what is
     fetched together and cached, for how long - in a `DataService` method in `service.py`.
2. Register it in the matching `tools/` module with `@mcp.tool(annotations=calculator(...))` or
   `market_data(...)` from `tools/_annotations.py`, and bound every input with `Field`.
3. Add its name to `MARKET_DATA_TOOLS` or `CALCULATOR_TOOLS` in `conventions.py`; the server
   instructions list tools from there, and a test checks the tuples against the registry.
4. Market-data tools also need a live contract test in `tests/live/`; calculators need a
   published golden input in `GOLDEN` (`tests/e2e/golden.py`).
5. Add a README bullet under `## Tools` in the form ``- `tool_name` — what it does``.
   `tests/test_readme_drift.py` fails until the README and the registry agree.
6. Run `make snapshot` and commit the new contract files.

## Adding a prompt

Register it in `prompts/analysis.py` or `prompts/calculations.py`, add sample arguments to
`SAMPLE_ARGS` in `tests/prompt_samples.py` (the drift test then checks every tool, parameter and
field the prompt names), add a README bullet under `## Prompts`, and run `make snapshot`.

## Commits and pull requests

- [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `chore:`, `ci:`, with a scope where it helps (`fix(calculators): …`).
  One logical change per commit.
- Open the PR against `master`, fill in the template, and add an entry under `[Unreleased]` in
  [CHANGELOG.md](CHANGELOG.md) for anything a user would notice, under **Breaking changes**
  if it changes a parameter, a result shape or a field's meaning.

## Releasing (maintainers)

The version comes from the git tag (hatch-vcs): `vX.Y.Z` builds `X.Y.Z`, and any untagged
commit builds a `.devN` version that can't pass for a release. There is no version to bump.

1. `make release-prep VERSION=X.Y.Z` moves the `[Unreleased]` entries under
   `## [X.Y.Z] - <today>` and updates the compare links. Open a PR with it and merge.
2. Optional rehearsal: `gh workflow run release.yml -f version=X.Y.Z` runs the whole
   verification - quality gate, build, version check, e2e against the built wheel - on a
   local-only tag and publishes nothing. (Dispatch it with `--ref <branch>` to rehearse
   before merging.)
3. `make release VERSION=X.Y.Z` publishes the GitHub Release `vX.Y.Z` on `origin/master`, with
   that CHANGELOG section as its notes. The Publish workflow then:
   - refuses a tag that isn't `vX.Y.Z` or isn't on `master`;
   - runs `make check` and builds from the clean tag;
   - refuses any artifact that isn't exactly `X.Y.Z`;
   - runs e2e against that wheel and uploads it to PyPI through trusted publishing;
   - checks that PyPI serves the same files, and installs the release by name and calls a
     tool.

   PyPI versions are immutable, which is why all of that happens before the upload.

The release must be created by a person (or a personal token), not by another workflow's
`GITHUB_TOKEN`, which can't trigger the Publish workflow. Pre-release tags aren't published.

If verification fails, nothing reached PyPI, but the GitHub Release and its tag exist. Remove
both with `gh release delete vX.Y.Z --cleanup-tag --yes`, fix `master`, and run
`make release` again. Deleting only the release keeps the tag, and a new release would then
build the old commit; `make release` refuses to start while the tag exists.
