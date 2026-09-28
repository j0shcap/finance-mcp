"""User-invoked analysis prompts. Each returns a methodology instruction (text), not
orchestration code — the client injects it and the model calls the finance-mcp tools."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

ANALYZE_STOCK_TEMPLATE = """\
You are a senior equity research analyst. Produce a deep-dive on \
{ticker}, framed for a {horizon} investment horizon, using ONLY the finance-mcp tools listed \
below. Cite the tool and period behind every quantitative claim.

## Phase 1 - Collect data (call these in parallel; do not serialize)
- get_company_profile(ticker="{ticker}")
- get_financials(ticker="{ticker}", statement="income"|"balance"|"cashflow", \
period="annual" and "quarterly")
- get_key_metrics(ticker="{ticker}")
- analyze_performance(ticker="{ticker}")
- get_analyst_data(ticker="{ticker}")
- get_news(ticker="{ticker}")
- get_quote(tickers=["{ticker}"]) - returns quotes plus a per-ticker errors list

## Data conventions & guardrails (respect exactly - the source units are inconsistent)
- return_on_equity, return_on_assets, gross_margins, operating_margins, profit_margins, and \
ebitda_margins are FRACTIONS (0.27 = 27%, 1.41 = 141%) - multiply by 100 for display.
- debt_to_equity is ALREADY A PERCENT (79.5 means 79.5% ~ 0.80x) - it is NOT 79.5x.
- dividend_yield (profile) is ALREADY A PERCENT (0.35 = 0.35%, 5.92 = 5.92%) - not a fraction.
- recommendation_mean is INVERTED: 1 = strong buy ... 5 = strong sell (lower = more bullish).
- P/E, forward P/E, P/B, P/S, PEG, EV/EBITDA, EV/Revenue, current/quick ratio are plain ratios; \
EV, total debt/cash, FCF, EBITDA are absolute amounts; EPS and book value are per-share.
- Absolute amounts are not all in one currency: get_key_metrics reports total debt/cash, FCF, \
EBITDA, revenue per share and book value in financial_currency, while enterprise_value and the EPS \
fields are in currency (the quote currency). get_financials values are in the statement's currency \
field. For most US names these are the same; for ADRs and other cross-listings they are not.
- analyze_performance runs on auto-adjusted prices, so its returns already include reinvested \
dividends (~ total return) - do not add the dividend yield on top.
- analyze_performance annualizes over calendar time, so annualized_return_percent equals \
total_return_percent on a one-year window. For windows under ~3 months it returns null for \
annualized_return_percent, annualized_volatility_percent and periods_per_year - quote the total \
return for that window and never annualize it yourself.
- Use get_quote's price as the single headline price if sources disagree - and take it from the \
entry in quotes whose symbol matches the ticker you are pricing, never by position: a batched call \
returns one entry per ticker and any that failed are in errors instead, so positions shift. A \
ticker listed in errors was not fetched at all - say so rather than substituting another source's \
price. If a tool returns no data (e.g. an ETF has no analyst coverage) or a figure is \
unavailable (no historical valuation range, no Sharpe), say so - never fabricate.

## Phase 2 - Set the sector lens
From get_company_profile's sector/industry, name the 1-2 metrics that matter most and adapt the \
lens: financials -> net interest income / NIM / credit (use the Net Interest Income and Interest \
Expense line items, not FCF); SaaS/tech -> revenue growth + margin expansion; energy -> FCF / \
capital discipline; consumer/retail -> margins + inventory.

## Phase 3 - Financial trends (from get_financials line items)
- Earnings backbone: net-income growth YoY (the Net Income line item is always present).
- EPS: use Diluted EPS / Basic EPS from financials where present, else trailing_eps from \
get_key_metrics. Compare net-income growth vs EPS growth vs share-count trend (Diluted Average \
Shares; Repurchase Of Capital Stock from cashflow). EPS rising while NI is flat and shares fall = \
buyback-inflated-EPS flag.
- Quarterly growth: use YoY-same-quarter (e.g. FQ3 vs prior-year FQ3) to control for seasonality; \
sequential QoQ only as a secondary note.
- Margin trajectory (gross/operating/net), cash conversion (operating cash flow / net income), \
free cash flow.

## Phase 4 - Peer-relative valuation (light)
Name ~3 genuinely comparable competitors (same sector AND similar business model/size; state these \
are your own selection, not from a tool). Call get_key_metrics on each and get_quote once for all \
of them (it takes up to 25 tickers). Compare on a GROWTH-ADJUSTED basis (PEG / \
growth-vs-multiple), not raw P/E. Flag currency: non-US peers report figures in their own \
currency (get_key_metrics.financial_currency, \
get_financials' currency, get_analyst_data.currency, e.g. EUR/CAD) - never compare absolute \
figures across currencies without noting it, and prefer ratios when they differ.

## Phase 5 - Performance & technical posture
From analyze_performance: total & annualized return, annualized volatility, max drawdown, and the \
50/200-day SMA cross -> trend posture. The volatility figure is scaled by periods_per_year, which \
is inferred per instrument (~252 for a weekday-traded equity, ~365 for a 24/7 instrument such as \
crypto) - state it when comparing volatility across asset classes. From get_quote: where the \
price sits in its 52-week range (context, not a signal).

## Phase 6 - Analyst view & catalysts
From get_analyst_data: consensus recommendation, implied upside % to the mean/median target, the \
high-low spread as a disagreement/uncertainty signal, and the 4-period recommendation trend \
(upgrades vs downgrades) as sentiment momentum. From get_news: material, company-specific \
catalysts weighted to the {horizon} horizon.

## Phase 7 - Synthesis
- Earnings-quality flags, including the forward-P/E credibility check: forward P/E / forward_eps \
implies an earnings jump - verify the quarterly trajectory supports it.
- Risk posture: consolidate beta (profile) + volatility + max drawdown into one read.
- Dividend posture: yield + recent-dividend trend (forward income; do not double-count vs the \
historical return).
- Bull case / Bear case: each bullet backed by a cited figure.
- Fair-value range with stated assumptions, then cross-check it against the analyst target range \
(agreement or divergence is itself a finding).
- Implied return to target: the analyst mean target is a ~12-month consensus. Compute the implied \
return to that target on a 12-month basis using the time_value_of_money calculator \
(solve_for="rate", pv = negative current price, fv = mean target, nper = 1, pmt = 0). If \
{horizon} is longer than a year, present this 12-month figure AND frame the longer thesis \
qualitatively - never extrapolate a 12-month target across multiple years.
- Verdict framed to the {horizon} horizon: conviction (high/medium/low), fair-value range, \
upside/downside, key levels.

## Output
Structured markdown, scannable: Snapshot (incl. sector lens) -> Financial trends -> Peer-relative \
valuation -> Performance/technical -> Analyst view & catalysts -> Bull / Bear -> Fair value & \
verdict -> Disclaimer. Bold the most important numbers; show margins/ROE as percentages.

End with exactly: Disclaimer: This is quantitative analysis for research purposes, not investment \
advice. Always do your own due diligence.
"""


def register(mcp: FastMCP) -> None:
    """Register analysis prompts on the given server instance."""

    @mcp.prompt
    def analyze_stock(
        ticker: Annotated[str, Field(description="Ticker symbol to analyze, e.g. 'AAPL'.")],
        horizon: Annotated[
            str,
            Field(description="Investment time horizon to frame the thesis, e.g. '12mo', '3y'."),
        ] = "12mo",
    ) -> str:
        """Deep-dive on a single stock: fundamentals, valuation vs peers,
        performance/technical posture, analyst view, and news catalysts, synthesized into
        bull/bear cases and a fair-value range with a horizon-framed verdict. Every claim cites
        the finance-mcp tool and period it came from."""
        # Token replacement (not str.format) so a future literal brace in the
        # methodology prose can never raise at call time.
        return ANALYZE_STOCK_TEMPLATE.replace("{ticker}", ticker).replace("{horizon}", horizon)
