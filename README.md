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
- `search_symbols` — resolve a company or instrument name to ticker symbol(s), best match first, across all instrument types (equity, ETF, crypto, …).
- `get_news` — recent news headlines for a ticker, newest first: title, publisher, link, publish time, and a short summary (no news returns an empty list, not an error).

**Analytics**

- `get_key_metrics` — valuation/profitability/leverage ratios (P/E, EV/EBITDA, margins, ROE, debt/equity, FCF, EPS, …) as reported by Yahoo; units noted per field, and absolute amounts labelled with the quote vs. reporting currency they're in.
- `analyze_performance` — total & annualized return, annualized volatility, max drawdown, and 50/200-day SMAs computed from the daily price series.

**Time value & loans**

- `time_value_of_money` — solve any one of present/future value, payment, rate, or periods (compound interest, annuities, CAGR); supports begin-of-period (annuity-due).
- `loan_schedule` — monthly payment, total interest, and optional amortization schedule for a fixed-rate loan/mortgage (nominal APR compounded monthly).

**Cashflow valuation**

- `npv` — net present value of equally-spaced cashflows (cashflow[0] at t=0).
- `irr` — internal rate of return; returns all real roots and flags non-uniqueness.
- `mirr` — modified IRR; single-valued given finance and reinvestment rates.
- `xnpv` / `xirr` — NPV/IRR of cashflows on actual calendar dates (Actual/365).

**Rates & fixed income**

- `convert_rate` — nominal ↔ effective annual rate (discrete or continuous compounding).
- `bond_price` — price plus Macaulay/modified duration and convexity at a given yield (priced on a coupon date).
- `bond_ytm` — yield to maturity from a bond's market price.

The calculators are pure and deterministic; the market-data and analytics tools fetch
live data (briefly cached) and surface source errors clearly. All tools return typed,
structured results and report invalid inputs as clear errors.

## Prompts

MCP prompts — reusable analysis templates the client exposes for you to invoke (Claude Code shows
them as slash commands; other clients surface them their own way).

- `analyze_stock` (arguments: `ticker`, optional `horizon`, default `12mo`) — single-stock deep-dive: fundamentals, growth-adjusted peer valuation, performance, analyst view, and news catalysts → bull/bear cases and a fair-value range with a horizon-framed verdict, citing the data behind each claim.

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

Or add it to an MCP client (e.g. Claude Desktop, Claude Code):

```json
{ "mcpServers": { "finance": { "command": "uvx", "args": ["mcp-finance"] } } }
```

## Development

```bash
git clone https://github.com/j0shcap/finance-mcp
cd finance-mcp
uv sync
make check   # ruff, mypy --strict, bandit, pytest (enforces the coverage gate)
```
