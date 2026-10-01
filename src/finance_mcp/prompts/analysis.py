"""User-invoked analysis prompts. Each returns a methodology instruction (text), not
orchestration code — the client injects it and the model calls the finance-mcp tools."""

import json
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.conventions import UNITS_GLOSSARY
from finance_mcp.prompts._render import parse_tickers, render
from finance_mcp.tools._inputs import MAX_COMPARE_TICKERS

_ANALYZE_STOCK_TEMPLATE = """\
You are a senior equity research analyst. Produce a deep-dive on \
{ticker}, framed for a {horizon} investment horizon, using ONLY the finance-mcp tools listed \
below. Cite the tool and period behind every quantitative claim.

## Phase 1 - Collect data (call these in parallel; do not serialize)
Leave risk_free_rate out of analyze_performance, compare_to_benchmark and compare_tickers: each \
then measures its Sharpe, Sortino and alpha against the 13-week T-bill yield averaged over its \
own dates, and echoes the rate with risk_free_rate_source "treasury_bill". If you override it, \
pass the SAME value to all three, or Phase 7 will consolidate figures measured against different \
hurdles. A result whose risk_free_rate_source is "unavailable" has those figures null - report \
that; never substitute a 0 rate.
- get_company_profile(ticker="{ticker}")
- get_financials(ticker="{ticker}", statement="income"|"balance"|"cashflow", \
period="annual" and "quarterly")
- get_key_metrics(ticker="{ticker}")
- analyze_performance(ticker="{ticker}")
- compare_to_benchmark(ticker="{ticker}", benchmark="SPY") - swap SPY for a \
benchmark that fits the listing (QQQ for US tech, a local index for a non-US line)
- get_analyst_data(ticker="{ticker}")
- get_news(ticker="{ticker}")
- get_quote(tickers=["{ticker}"]) - returns quotes plus a per-ticker errors list

## Data conventions & guardrails (respect exactly - the source units are inconsistent)
{units_glossary}
- If a tool returns no data (e.g. an ETF has no analyst coverage) or a figure is unavailable (no \
historical valuation range, a null ratio on a short window), say so - never fabricate. A null \
ratio means "not computable over this window", never zero.
- The same glossary is available as the {conventions_uri} resource if you need it again later.

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

## Phase 4 - Peer-relative valuation
Name ~3 genuinely comparable competitors (same sector AND similar business model/size; state \
these are your own selection, not from a tool). Call compare_tickers once with {ticker}, those \
peers (and the same risk_free_rate, if you overrode it) - it returns performance and valuation \
side by side for up to 10 tickers \
in one call, so do not loop get_key_metrics over them. Compare on a GROWTH-ADJUSTED basis (PEG / \
growth-vs-multiple), not raw P/E, and rank risk-adjusted return (sharpe_ratio) rather than raw \
return. Read the table's errors list and any row's metrics_error before treating a blank cell as \
a finding, and each row's periods_per_year before ranking volatility or sharpe_ratio - rows on \
different calendars are not directly comparable. Currency: any row flagged currency_differs (or \
a table with mixed_currencies true) is not denominated in base_currency - its returns carry an \
FX move the others do not, so compare those rows on ratios and say so. A row \
whose financial_currency differs from its own currency is a cross-listing: its price_to_sales, \
price_to_book and EV multiples mix two currencies, so rank it on P/E, PEG and the margins only \
(the glossary's cross-listing rule).

## Phase 5 - Performance, risk-adjusted return & technical posture
From analyze_performance: total & annualized return, annualized volatility, max drawdown, the \
50/200-day SMA cross -> trend posture, and the risk-adjusted set - sharpe_ratio (return per unit \
of total risk), sortino_ratio (per unit of DOWNSIDE risk; on a positive Sharpe, sitting above it \
means the swings were mostly upward - the comparison inverts when the Sharpe is negative), \
downside_deviation_percent and calmar_ratio (CAGR per unit of worst drawdown). State the \
risk_free_rate the result echoes and its source: by default these are excess over the T-bill \
yield of the same window; a rate of 0 would make them raw return per unit of risk. \
The volatility figure is scaled by periods_per_year, which is inferred per instrument (~252 for a \
weekday-traded equity, ~365 for a 24/7 instrument such as crypto) - state it when comparing \
volatility across asset classes.
From compare_to_benchmark: beta (market sensitivity) read together with correlation (how much of \
the move the benchmark actually explains - a big beta at a low correlation explains little), \
alpha_percent (annualized outperformance beyond what beta predicted, in percentage points), \
tracking_error_percent, information_ratio (how RELIABLY it out/under-performed) and \
excess_return_percent. Check overlapping_observations first: a thin overlap (a recent listing, a \
halt, or a 7-day instrument against a 5-day benchmark) makes beta and alpha noise, and a \
cross-currency pair folds an FX move into both.
From get_quote: where the price sits in its 52-week range (context, not a signal).

## Phase 6 - Analyst view & catalysts
From get_analyst_data: consensus recommendation, implied upside % to the mean/median target, the \
high-low spread as a disagreement/uncertainty signal, and the 4-period recommendation trend \
(upgrades vs downgrades) as sentiment momentum. From get_news: material, company-specific \
catalysts weighted to the {horizon} horizon. Articles with mentions_company false are usually \
market-wide stories filed under the ticker - read the title, and use one only if it bears on \
{ticker} specifically.

## Phase 7 - Synthesis
- Earnings-quality flags, including the forward-P/E credibility check: forward_eps above \
trailing_eps implies expected earnings growth - verify the quarterly trajectory supports it, and \
treat an unsupported gap as a flag. When trailing_eps is positive, a forward P/E below the \
trailing P/E says the same thing; when trailing_eps is zero or negative the trailing P/E is \
meaningless, so compare the EPS figures directly instead.
- Risk posture: consolidate compare_to_benchmark's beta and correlation (prefer them over the \
profile's ~5-year beta, and say which window each covers), sharpe_ratio/sortino_ratio, volatility \
and max drawdown into one read: how much risk was taken, and whether the return paid for it.
- Dividend posture: yield + recent-dividend trend (forward income; do not double-count vs the \
historical return).
- Bull case / Bear case: each bullet backed by a cited figure.
- Fair-value range with stated assumptions, then cross-check it against the analyst target range \
(agreement or divergence is itself a finding).
- Implied return to target: the analyst mean target is a ~12-month consensus, so the implied \
return is simply (mean target / current price) - 1, using get_quote's price. If {horizon} is \
longer than a year, present this 12-month figure AND frame the longer thesis qualitatively - \
never extrapolate a 12-month target across multiple years.
- Verdict framed to the {horizon} horizon: conviction (high/medium/low), fair-value range, \
upside/downside, key levels.

## Output
Structured markdown, scannable: Snapshot (incl. sector lens) -> Financial trends -> Peer-relative \
valuation -> Performance/technical -> Analyst view & catalysts -> Bull / Bear -> Fair value & \
verdict -> Disclaimer. Bold the most important numbers; show margins/ROE as percentages.

End with exactly: {disclaimer}
"""


_COMPARE_STOCKS_TEMPLATE = """\
You are a buy-side analyst ranking a peer group for a {horizon} investment horizon: {tickers}. \
Take every figure from the finance-mcp tools below - never do that arithmetic yourself when a \
tool reports the number, and show the formula for any simple ratio you derive. Cite the tool and \
period behind every figure. Units differ by field (fractions vs percents, per-row currencies): \
read the {conventions_uri} resource before converting or comparing anything.

## Phase 1 - Collect (call these in parallel)
Leave risk_free_rate out of both compare_tickers calls: each row's Sharpe and Sortino are then \
excess over the 13-week T-bill yield averaged over that row's own dates (its risk_free_rate and \
risk_free_rate_source say so), so the 1y and 5y windows each use their own period's cash return. \
A row whose risk_free_rate_source is "unavailable" has those ratios null - rank it on the rest \
and say why.
- compare_tickers(tickers={tickers_list}, period="1y")
- compare_tickers(tickers={tickers_list}, period="5y") - the robustness window
- get_quote(tickers={tickers_list}) - read each entry by its symbol, never by position
- get_company_profile(ticker=T) for each ticker - sector, industry, market cap, currency
- get_financials(ticker=T, statement="income", period="annual", line_items=["Total Revenue", \
"Diluted EPS", "Net Income"]) for each ticker - the growth history

## Phase 2 - Comparability screen (before any ranking)
- Same sector AND a similar business model, with scale within an order of magnitude? If not, a \
ranking on multiples is meaningless: split the list into comparable groups, rank within each, and \
say why.
- Sector lens: banks and insurers -> price_to_book beside return_on_equity (EV-based multiples \
mean nothing for them); capital-light growth -> growth-adjusted P/E and margins; cyclicals -> \
think mid-cycle (a cyclical at peak earnings looks cheapest exactly when it is riskiest).
- Read both tables' errors and every row's metrics_error first: a missing ticker or blank \
metrics is a data gap, not a finding.

## Phase 3 - Declare the rubric BEFORE reading the results
Write down the criteria and their weights, tied to the {horizon} horizon, before ranking - this \
stops the conclusion from choosing its own evidence. A multi-year horizon weights valuation \
against growth durability and business quality most; a horizon of a year or less gives more \
weight to drawdown and risk-adjusted return. State the weights you chose, then keep them.

## Phase 4 - Growth-adjusted valuation: derive it, do not trust it blindly
- peg_ratio from Yahoo is often null or stale and its growth basis is undisclosed - use it as one \
input, never the only one.
- Implied forward EPS growth = trailing_pe / forward_pe - 1, meaningful ONLY when both are \
positive.
- Historical growth from get_financials: revenue and Diluted EPS CAGR = (latest / earliest) ^ \
(1 / years) - 1, where years is the number of annual periods returned minus 1 (period_ends run \
most recent first). It is undefined when either endpoint is zero or negative - say so rather \
than computing it.
- Growth-adjusted P/E = forward_pe / (expected growth in percent). With negative or near-zero \
earnings the P/E is meaningless: fall back to ev_to_ebitda or price_to_sales, read beside \
profit_margins.
- Value-trap check: a low multiple with falling revenue or margins is usually cheap for a reason.

## Phase 5 - Risk-adjusted performance, and whether it is robust
- Rank on sharpe_ratio, sortino_ratio and calmar_ratio alongside max_drawdown_percent - not on \
raw total_return_percent.
- Before comparing rows, check each row's periods_per_year (a 24/7 instrument is annualized on a \
different calendar) and start_date (a recent listing covers less history than its peers).
- Compare the 1y and 5y orderings. If the ranking flips between windows, say the performance \
ranking is regime-dependent and weight it less. Past returns are context, not a forecast.

## Phase 6 - Currency
If a table reports mixed_currencies, or a row is flagged currency_differs, those rows' returns \
carry an FX move the others do not and their absolute amounts are in another currency: rank them \
on ratios only, and say so. A row whose financial_currency differs from its \
currency is a cross-listing: Yahoo computes its price_to_sales, price_to_book and EV multiples \
across two currencies, so rank it on P/E, PEG and the margins only, as the cross-listing rule in \
{conventions_uri} sets out.

## Phase 7 - Verdict
- Score each ticker against the rubric you declared. Tickers missing too many inputs go in an \
"insufficient data" bucket - not last place. When two scores are within the noise of their \
inputs, call it a tie; do not manufacture precision.
- Ranked table: rank, ticker, one-line thesis, the key risk, conviction (high / medium / low), \
and what would change its rank.
- Then short per-ticker notes, each claim backed by a cited figure, and the caveats that apply: \
comparability splits, currency, data gaps, rank instability.

End with exactly: {disclaimer}
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
        return render(
            _ANALYZE_STOCK_TEMPLATE,
            ticker=ticker,
            horizon=horizon,
            units_glossary=UNITS_GLOSSARY,
        )

    @mcp.prompt
    def compare_stocks(
        tickers: Annotated[
            str,
            Field(
                description=f"2-{MAX_COMPARE_TICKERS} ticker symbols separated by commas or "
                "spaces, e.g. 'AAPL, MSFT, GOOGL'."
            ),
        ],
        horizon: Annotated[
            str,
            Field(description="Investment time horizon to frame the ranking, e.g. '12mo', '3y'."),
        ] = "12mo",
    ) -> str:
        """Rank a peer group side by side: a comparability screen, a rubric fixed before the
        results are read, growth-adjusted valuation derived from the data, risk-adjusted
        performance checked across two windows, and currency caveats - ending in a ranked
        verdict that separates ties and data gaps from real differences."""
        symbols = parse_tickers(tickers)
        return render(
            _COMPARE_STOCKS_TEMPLATE,
            tickers=", ".join(symbols),
            tickers_list=json.dumps(symbols),
            horizon=horizon,
        )
