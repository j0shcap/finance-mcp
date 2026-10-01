# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); before 1.0, a minor version may
contain breaking changes, and each one is listed under **Breaking changes**.

## [Unreleased]

### Added
- README sections on conventions & units and on the data source, with Claude Code and Claude
  Desktop setup; `CONTRIBUTING.md`, `SECURITY.md`, this changelog, issue and pull-request
  templates, and a test that fails when the README's tool, prompt, resource or configuration
  lists drift from the server.

### Changed
- Yahoo requests are capped at 8 in flight across all concurrent tool calls
  (`FINANCE_MCP_MAX_CONCURRENT_REQUESTS`), and a request Yahoo throttles or the network drops
  is retried up to twice with backoff (`FINANCE_MCP_REQUEST_RETRIES`, 0 to disable), within
  about 15 seconds. Unknown symbols and slow timeouts still fail at once.
- The package version now comes from the git tag, so an install from an untagged commit reports
  a `.devN` version rather than the last release's number. Releases are cut from this changelog,
  and the publish workflow tests the exact wheel before uploading it and checks afterwards that
  PyPI serves those same files.
- Calculator rate parameters and results state per-period or annual and "as a decimal"
  consistently, and bond prices are stated in the units of `face`; descriptions only, no schema
  change.
- Requires fastmcp 4.x (`>=4.0.10,<5`, which brings the MCP Python SDK 2.x) and pydantic
  `>=2.12`; supersedes #16. A result model rejecting a value inside a tool still reaches the
  model as the masked "Error calling tool" error, not as invalid arguments.

## [0.4.2] - 2026-10-01

0.4.1 was bumped in #43 but never tagged or published; its changes ship here.

### Breaking changes
- `analyze_performance`, `compare_to_benchmark` and `compare_tickers` default `risk_free_rate`
  to the 13-week US T-bill yield (Yahoo `^IRX`) averaged over the dates measured, instead of 0,
  so the default Sharpe, Sortino, downside deviation and alpha are excess over cash. Pass
  `risk_free_rate=0` for the old figures. When the T-bill average can't be formed, the
  rate-dependent figures are `null` with a `risk_free_rate_note`; they never fall back to 0
  (#45).
- `compare_tickers`' table-level `risk_free_rate` is the caller's rate or `null`; each row
  carries the rate used for its own dates (#45).

### Added
- `risk_free_rate_source` (`caller`, `treasury_bill` or `unavailable`) on every result that
  takes a risk-free rate (#45).
- `get_news` flags `mentions_company` on each article for a stock, with `relevance_check` and
  `relevance_note` explaining a null flag; articles are flagged, not filtered (#45).
- `loan_schedule` reports `interest_saved` and `payments_saved` for an extra monthly payment
  (#45).
- `make e2e`: the built wheel is installed into a fresh venv, launched as the `mcp-finance`
  console script, `python -m finance_mcp` and `uvx --from <wheel> mcp-finance`, and driven over
  stdio; runs in CI on Python 3.13 and 3.14 (#44).
- Property-based cross-checks of every calculator against `numpy-financial` and `scipy` in
  `make check` (#44).

### Changed
- Market-data results are rounded on output to 7 significant digits, the precision Yahoo
  provides (integer digits are never rounded away); computations still use full precision, and
  calculator results are unrounded (#43).
- `time_value_of_money` defaults an omitted `fv` to 0, as Excel does (#43).
- Invalid tool arguments come back as one clean message (`Invalid arguments for get_quote:
  tickers: …`) instead of pydantic's raw error dump (#45).
- The server launches without fastmcp's banner and its PyPI update check (#44).
- fastmcp is locked at 3.4.7; units and nullability are worded the same way across every result
  field (#45).

### Fixed
- With the network down, six market-data tools still reported a good ticker as "invalid or
  delisted", because yfinance hid the error behind an empty result (#44).
- `time_value_of_money` at rates below ~1e-16 dropped the payment term, or raised a division
  error when solving for `pmt` (#44).
- `irr` could report a spurious root next to a sign change, or pass a local minimum off as a
  double root (#44).
- The conventions glossary and prompts no longer steer cross-listings (ADRs) onto the
  valuation ratios Yahoo computes across two currencies (#44).

## [0.4.0] - 2026-10-01

### Added
- `bond_price_dated` and `bond_ytm_dated`: price a bond, or solve its yield, for a settlement
  date between coupon dates; clean and dirty price, accrued interest, duration and convexity, per
  face and per 100; Actual/Actual ICMA or 30/360 US, and a `first_period_discount` choice (#36).
- Prompts `compare_stocks`, `loan_planner`, `bond_analysis` and `investment_cashflows` (#37).
- A test that checks every tool, parameter and result field a prompt names against the tool
  registry (#38).

### Changed
- `irr`, `xirr` and `bond_ytm` solve on the raw pricing functions (about 1.7x and 2.7x faster
  for `irr` and `xirr`); internal cleanup of the calculators, data layer and tests with no change
  to results (#38).

### Fixed
- `bond_price_dated` rejects a yield that would give a non-positive clean price, instead of
  returning a negative price or a bare division error (#36).
- Descriptions stated the annualization gate as 90 days (it is 85), and the dated-bond
  duration convention wrongly (#38).

### Dependencies
- pydantic-settings 2.15.0, starlette 1.3.1, pyjwt 2.15.0, urllib3 2.8.0, virtualenv 21.7.12 and
  dev tooling (#14, #15, #20, #39, #41, #42).

## [0.3.0] - 2026-09-29

### Breaking changes
- `get_quote` returns `{quotes, errors}` instead of a list of quotes: one bad or unreachable
  ticker no longer fails the batch, and lands in `errors` with its reason. It takes 1–25
  tickers (#27).
- Intraday `get_price_history` bars (1m–1h) carry a full timestamp with the exchange's UTC
  offset instead of a date; daily and longer bars stay dated (#27).
- `analyze_performance`'s `annualized_return_percent` and `annualized_volatility_percent` can be
  `null`: windows under ~3 months are not annualized (#26, #33).
- A network or source failure is reported as a data-unavailable error, no longer as an unknown
  symbol (#27).
- Tool inputs are bounded in the schema: malformed tickers, empty `line_items`, and oversized
  cashflow lists, bond terms and loan terms are rejected before any work is done (#27, #30).
- Settings are read from the environment only; a `.env` file is no longer loaded (#30).

### Added
- `compare_to_benchmark`: beta, correlation, Jensen's alpha, tracking error, information ratio
  and excess return against a benchmark (default `SPY`) (#32).
- `compare_tickers`: side-by-side performance and valuation for 2–10 tickers, with partial
  results and per-row currency flags (#32).
- Sharpe, Sortino, downside deviation and Calmar in `analyze_performance`, with an explicit
  annual `risk_free_rate` (default 0) echoed in the result (#32).
- `periods_per_year` in `analyze_performance`, showing the calendar the figures were annualized
  on (#26).
- Currency labels: `currency` on `get_financials`, `currency` and `financial_currency` on
  `get_key_metrics`; unmatched `line_items` are reported with close-match suggestions (#27).
- `get_news` reports which Yahoo endpoint served it (#34).
- Tool annotations and titles, server instructions, the `finance://conventions` resource, and
  `FINANCE_MCP_MAX_HISTORY_BARS` (#30).
- Opt-in live contract tests against Yahoo (`make test-live`), run nightly in CI (#31).

### Changed
- Annualization uses elapsed calendar time rather than 252 bars a year, so a 24/7 instrument
  such as BTC-USD annualizes correctly (#26, #29).
- `get_quote` fetches tickers concurrently; symbols are normalized before caching; the cache
  evicts least-recently-used entries, and long price histories are not cached (#27, #33).
- Dependency floors match what the code calls (`yfinance>=1.0`, `fastmcp>=3.0,<4`,
  `pydantic>=2.11.7`); CI runs on Python 3.13 and 3.14; a release publishes only if its tag
  matches `project.version` and `make check` passes (#13).

### Fixed
- `time_value_of_money` rejects impossible inputs (zero periods, `rate <= -1`, `pv = pmt = 0`
  when solving for `nper`) with a clear message instead of a raw arithmetic error (#25).
- `irr` and `xirr` find double (tangent) roots, closely spaced roots, and roots above 1000% (up
  to 1,000,000%) (#25).
- Overflows in `convert_rate` and `loan_schedule` are reported as input errors; `bond_price`
  accepts any yield above `-frequency`; `loan_schedule` pays off the final balance exactly (#25).
- A one-year window's annualized return now equals its total return, including for 24/7
  instruments; short windows no longer extrapolate a few days' move to a year (#26).
- `period="3mo"` no longer drops below the annualization gate on some calendar dates (#33).
- Intraday bars no longer all share one date (#27).
- A broken Yahoo news endpoint is no longer reported as "no recent news"; news falls back to
  Yahoo search (#34).
- `bond_price` with a very long maturity could hang the server (#30).

### Dependencies
- yfinance 1.7.0, mcp 1.28.1, pydantic 2.13.5, pydantic-settings 2.14.2, anyio 4.14.2,
  cryptography 50.0.0, soupsieve 2.9, python-multipart 0.0.31 (#17, #18, #19, #21, #22, #23,
  #24, #34).

## [0.2.0] - 2026-06-02

Baseline for this changelog: market-data, analytics and calculator tools, and the
`analyze_stock` prompt.

[Unreleased]: https://github.com/j0shcap/finance-mcp/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/j0shcap/finance-mcp/compare/v0.4.0...v0.4.2
[0.4.0]: https://github.com/j0shcap/finance-mcp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/j0shcap/finance-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/j0shcap/finance-mcp/releases/tag/v0.2.0
