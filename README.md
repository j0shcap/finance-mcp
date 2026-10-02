# Finance MCP

[![CI](https://github.com/j0shcap/finance-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/j0shcap/finance-mcp/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/j0shcap/finance-mcp/branch/master/graph/badge.svg)](https://codecov.io/gh/j0shcap/finance-mcp)
[![PyPI](https://img.shields.io/pypi/v/mcp-finance)](https://pypi.org/project/mcp-finance/)
[![Python](https://img.shields.io/pypi/pyversions/mcp-finance)](https://pypi.org/project/mcp-finance/)
[![License: MIT](https://img.shields.io/pypi/l/mcp-finance)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/types-mypy-blue)](https://mypy-lang.org/)

An MCP server of finance tools: a `yfinance` market-data wrapper, computed analytics,
and deterministic financial calculators.

## Tools

**Market data (yfinance)**

- `get_quote` — current price snapshots for 1–25 tickers: price, change, day/52-week ranges, market cap (fetched in parallel; cached briefly). Results are partial: successful quotes plus a per-ticker `errors` list, so one bad symbol doesn't sink the batch.
- `get_price_history` — OHLCV bars plus a computed summary for a ticker, by period and interval (long windows are truncated; the summary covers the full window). Intraday bars carry a full timestamp with the exchange's UTC offset; daily and longer bars are dated.
- `get_financials` — income statement, balance sheet, or cash flow (annual or quarterly) as line items by period; values labelled with the company's reporting `currency`, with an optional line-item filter that reports any labels it couldn't match (plus close-match suggestions).
- `get_company_profile` — sector, industry, market cap, P/E, beta, business summary, plus recent dividends and stock splits.
- `get_analyst_data` — sell-side analyst consensus: price targets, consensus recommendation, and the recent rating trend (analyst counts over the last four months).
- `get_earnings` — the next earnings date (confirmed or estimated, in the exchange's time), EPS and revenue consensus for the coming quarters and fiscal years, and the last four quarters' EPS surprises.
- `search_symbols` — resolve a company or instrument name to ticker symbol(s), best match first, across all instrument types (equity, ETF, crypto, …).
- `get_news` — recent news headlines for a ticker, newest first: title, publisher, link, publish time, and a short summary (no news returns an empty list, not an error). For a stock, each article is flagged with whether it names the company or its ticker, since Yahoo files market-wide stories under tickers too.

**Analytics**

- `get_key_metrics` — valuation/profitability/leverage ratios (P/E, EV/EBITDA, margins, ROE, debt/equity, FCF, EPS, …) as reported by Yahoo; units noted per field, and absolute amounts labelled with the quote vs. reporting currency they're in.
- `analyze_performance` — total & annualized return, annualized volatility, max drawdown, 50/200-day SMAs, and risk-adjusted statistics (Sharpe, Sortino, downside deviation, Calmar) from the daily price series. Annualized figures are null for windows under 85 days.

  `risk_free_rate` (an annual decimal, also taken by `compare_to_benchmark` and `compare_tickers`) defaults to the 13-week US T-bill yield (Yahoo `^IRX`) averaged over the dates measured and converted to an effective annual rate, so the ratios are excess over cash; pass `0` for raw return per unit of risk. Each result echoes the rate and its source: `caller`, `treasury_bill`, or `unavailable` (with a note), in which case the rate-dependent figures are null rather than computed at 0.
- `compare_to_benchmark` — beta, correlation, Jensen's alpha, tracking error, information ratio and excess return versus a benchmark ticker (default `SPY`). The two daily series are inner-joined on date, so a 24/7 instrument compared with an equity benchmark contributes only its weekday closes; the overlapping observation count is reported.
- `compare_tickers` — side-by-side performance plus key valuation metrics for 2–10 tickers, fetched in parallel. Partial like `get_quote`: a ticker with no usable history lands in `errors`, and a row whose valuation metrics failed keeps its performance figures with `metrics_error` set. Rows not denominated in the table's base currency are flagged.

**Time value & loans**

- `time_value_of_money` — solve any one of present/future value, payment, rate, or periods (compound interest, annuities, CAGR); supports begin-of-period (annuity-due).
- `loan_schedule` — monthly payment, total interest, and optional amortization schedule for a fixed-rate loan/mortgage (nominal APR compounded monthly); with an extra monthly payment, also the interest and payments it saves.

**Cashflow valuation**

- `npv` — net present value of equally-spaced cashflows (cashflow[0] at t=0).
- `irr` — internal rate of return; returns all real roots and flags non-uniqueness.
- `mirr` — modified IRR; single-valued given finance and reinvestment rates.
- `xnpv` / `xirr` — NPV/IRR of cashflows on actual calendar dates (Actual/365).

**Rates & fixed income**

- `convert_rate` — nominal ↔ effective annual rate (discrete or continuous compounding).
- `bond_price` — price plus Macaulay/modified duration and convexity at a given yield (priced on a coupon date).
- `bond_ytm` — yield to maturity from a bond's market price.
- `bond_price_dated` — price a bond for a settlement date that may fall **between** coupon dates: clean and dirty price, accrued interest, duration and convexity, per face and per 100 (Actual/Actual ICMA or 30/360 US). Defaults to the street convention for the part period; `first_period_discount="simple"` matches the US Treasury's own formulas, and also matches Excel inside the final coupon period.
- `bond_ytm_dated` — yield to maturity from a clean price for a given settlement date.

The calculators are pure and deterministic; the market-data and analytics tools fetch live
data (briefly cached). Every tool returns a typed, structured result. Market-data floats are
rounded to 7 significant digits, the precision Yahoo actually provides (whole numbers such as
volumes keep every digit); calculator results are returned at full precision.

## Prompts

Reusable analysis templates the client exposes for you to invoke (Claude Code shows them as
slash commands).

- `analyze_stock` (arguments: `ticker`, optional `horizon`, default `12mo`) — single-stock deep-dive: fundamentals, growth-adjusted peer valuation, risk-adjusted and benchmark-relative risk posture, analyst view, and news catalysts → bull/bear cases and a fair-value range with a horizon-framed verdict, citing the data behind each claim.
- `compare_stocks` (arguments: `tickers` — 2-10 symbols separated by commas or spaces, optional `horizon`, default `12mo`) — ranks a peer group: a comparability screen, a rubric fixed before the results are read, growth-adjusted valuation derived from the data rather than Yahoo's PEG, risk-adjusted performance checked for stability across a 1y and a 5y window, and currency caveats → a ranked verdict that keeps ties and data gaps apart from real differences.
- `loan_planner` (arguments: `principal`, `annual_rate`, `term_months`, optional `extra_payment`) — fixed-rate loan or mortgage: payment and total interest cross-checked through a second tool, note rate vs. a fee-loaded APR vs. the effective annual rate, non-monthly compounding, what extra payments are worth, and a refinance break-even that accounts for a reset term.
- `bond_analysis` (arguments: `bond` — a plain-words description, optional `shock_bp`, default `100`) — price, yield, accrued interest, duration, convexity and DV01 with the day count and clean/dirty basis made explicit, then a ± rate shock estimated from duration and convexity and cross-checked by exact repricing, and where option-free analytics break.
- `investment_cashflows` (arguments: `cashflows`, optional `discount_rate`, `reinvest_rate`) — NPV/IRR/MIRR/XIRR: timing and rate-period conventions, sign-pattern diagnosis (multiple IRRs, borrowing-type flows), NPV as the decision rule with an NPV profile, and when MIRR is the better single figure.

Every prompt points at the `finance://conventions` resource for unit rules (`analyze_stock`
also embeds the market-data glossary). A test renders each prompt and checks every tool call,
keyword argument and snake_case field name it mentions against the tool registry.

## Resources

- `finance://conventions` — the units, sign, and rate conventions every result follows: which
  Yahoo fields are fractions vs. already percents, which currency each absolute amount is in,
  and the Excel sign convention (cash received positive, cash paid negative) the cashflow and
  TVM tools use. A condensed version ships as the server's MCP `instructions`.

Every tool is annotated read-only; the calculators are additionally marked idempotent and
closed-world, the market-data tools open-world.

## Conventions & units

The short version for people reading results; `finance://conventions` is the authoritative,
per-field glossary.

- **Signs follow Excel.** Cash you receive is positive, cash you pay is negative: a loan
  principal or a deposit is a negative `pv`, the balance you get back a positive `fv`. Mixed-up
  signs flip the answer or leave a rate solve with no solution.
- **Rates are decimals**: `0.05` means 5%, as an input and in calculator results.
- **Per-period vs. annual.** `npv`, `irr`, `mirr` and `time_value_of_money` take a per-period
  rate matching the cashflow spacing (monthly flows → a monthly rate). `xnpv`, `xirr`,
  `loan_schedule`, `convert_rate` and the four bond tools take annual rates; `loan_schedule`'s
  is a nominal APR compounded monthly.
- **Bonds: clean vs. dirty.** The clean price is what the market quotes; the dirty price
  (clean + accrued interest) is what a buyer pays. `bond_price_dated` reports both, and
  `bond_ytm_dated` solves from the clean price.
- **Market-data fields named `*_percent` are percents** (`12.5` = 12.5%); the risk ratios
  (Sharpe, Sortino, Calmar, beta, …) are plain numbers.
- **Yahoo's own units are inconsistent**, and are passed through as reported: margins, ROE and
  ROA are fractions (`0.27` = 27%), but `debt_to_equity` and `dividend_yield` are already
  percents, and `recommendation_mean` runs from 1 (strong buy) to 5 (strong sell).
- **Currencies.** Prices are in the quote currency. Statement figures are in the reporting
  currency (`get_financials` labels it `currency`), as are several `get_key_metrics` amounts
  (labelled `financial_currency`). The two differ for ADRs and other cross-listings, which also
  make several Yahoo valuation ratios unreliable.
- **Annualization** is over calendar time from auto-adjusted (dividend-inclusive) prices.
  Windows under 85 days return `null` for annualized figures rather than extrapolating.

## Configuration

Optional, read from the environment only — there is no `.env` support, because an MCP client
launches the server in whatever working directory it chooses.

| Variable | Default | Meaning |
| --- | --- | --- |
| `FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS` | `30` | Quote cache lifetime. |
| `FINANCE_MCP_HISTORY_CACHE_TTL_SECONDS` | `300` | Price-history cache lifetime. |
| `FINANCE_MCP_FUNDAMENTALS_CACHE_TTL_SECONDS` | `3600` | Fundamentals/profile cache lifetime. |
| `FINANCE_MCP_MAX_HISTORY_BARS` | `260` | Most bars `get_price_history` returns before truncating (the summary still covers the full window). |
| `FINANCE_MCP_MAX_CONCURRENT_REQUESTS` | `8` | Most Yahoo requests in flight at once, across all concurrent tool calls. |
| `FINANCE_MCP_REQUEST_RETRIES` | `2` | Retries, with backoff, of a request Yahoo throttled or the network dropped; `0` disables them. |

## Install

```bash
uvx mcp-finance        # run without installing (recommended)
# or
pip install mcp-finance
```

## Usage

Run the stdio server directly:

```bash
mcp-finance
```

Or add it to an MCP client. Claude Code:

```bash
claude mcp add finance -- uvx mcp-finance
```

Claude Desktop and other JSON-configured clients (Claude Desktop's file is
`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{ "mcpServers": { "finance": { "command": "uvx", "args": ["mcp-finance"] } } }
```

Settings from [Configuration](#configuration) go in the server's `env`:

```json
{
  "mcpServers": {
    "finance": {
      "command": "uvx",
      "args": ["mcp-finance"],
      "env": { "FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS": "60" }
    }
  }
}
```

## Data source & disclaimer

Market data comes from Yahoo Finance through [`yfinance`](https://github.com/ranaroussi/yfinance),
an unofficial library that is not affiliated with, endorsed or vetted by Yahoo. Quotes may be
delayed, and fields can change or disappear without notice when Yahoo changes its endpoints.
The data is intended for personal research and educational use; refer to Yahoo's terms of use
for what you may do with it. Nothing this server returns is investment advice, and the software
is provided as is, without warranty (see [LICENSE](LICENSE)).

## Development

```bash
git clone https://github.com/j0shcap/finance-mcp
cd finance-mcp
uv sync
make check   # ruff, mypy --strict, bandit, pytest (enforces the 90% coverage gate)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a tool or prompt,
[CHANGELOG.md](CHANGELOG.md) for what changed between releases, and
[SECURITY.md](SECURITY.md) to report a vulnerability.

`make check` is fully offline: every test in it mocks yfinance.

### Live contract tests

`tests/live/` calls real Yahoo endpoints to catch what mocks cannot: a renamed `info` key, a
restructured news payload, a field switching between a fraction and a percent.

```bash
make test-live   # opt-in: hits the network, excluded from the coverage gate
```

They are marked `live` and deselected by default, so they never run in `make check` or on a
PR; CI runs them nightly (`.github/workflows/live.yml`) and a failure opens or comments on a
single rolling issue.

Assertions cover shape and unit plausibility rather than exact values — margins as fractions,
`debt_to_equity` and `dividend_yield` as percents, `period_ends` descending — and each tool is
exercised both through `DataService` and through an in-process MCP client. Throttling is
retried and then reported as a skip, so it never reads as a contract failure.

### End-to-end tests

`tests/e2e/` tests the artifact rather than the source: it builds the wheel, installs it into a
fresh venv and launches the server the three ways a client config can — the `mcp-finance`
console script, `python -m finance_mcp`, and `uvx --from <wheel> mcp-finance` — then speaks MCP
to it over stdio. That catches what an in-process client cannot: a broken entry point, a module
or dependency missing from the wheel, or a stray `print` corrupting the JSON-RPC stream.

```bash
make e2e   # builds the wheel; needs uv on PATH and access to the package index, not Yahoo
```

It checks the handshake, every tool's annotations and schemas, every prompt and the
conventions resource, calls each calculator with a published golden input, and cuts the
server's network to prove an outage is never reported as an unknown symbol. The Yahoo-backed
half is also marked `live`, so it runs with `make test-live`. Set `FINANCE_MCP_E2E_WHEEL` to
test a prebuilt wheel instead of building one.

The calculators are also cross-checked in `make check` against independent implementations
(`numpy-financial`, `scipy`) with `hypothesis` property tests: TVM round-trips, NPV at the
IRR, bond price ↔ yield, and loan schedules that amortize to zero.
