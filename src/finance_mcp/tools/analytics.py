"""MCP tools for computed analytics, backed by YFinanceClient."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.data.models import (
    BenchmarkComparison,
    HistoryPeriod,
    KeyMetrics,
    PerformanceStats,
    TickerComparison,
)
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.tools._annotations import market_data
from finance_mcp.tools._dispatch import run_data
from finance_mcp.tools._inputs import MAX_COMPARE_TICKERS, RiskFreeRate, Ticker


def register(mcp: FastMCP, client: YFinanceClient) -> None:
    """Register analytics tools bound to a YFinanceClient."""

    @mcp.tool(annotations=market_data("Key Valuation & Profitability Metrics"))
    async def get_key_metrics(
        ticker: Ticker,
    ) -> KeyMetrics:
        """Valuation, profitability, and leverage ratios (as reported by Yahoo).

        Note units differ by field: P/E, P/B, P/S, EV/EBITDA, PEG are plain ratios;
        margins and ROE/ROA are fractions (0.27 = 27%); debt_to_equity is a percent
        (79.5 = 79.5%); EV, total debt/cash, FCF, EBITDA are absolute amounts. Those
        amounts are not all in one currency: debt/cash/FCF/EBITDA and the per-share
        revenue/book value are in `financial_currency`, the EPS fields in `currency`. They
        differ for ADRs and other cross-listings, where Yahoo computes P/S, P/B, EV and the
        EV multiples across both currencies: use P/E, PEG and the margins for those.
        """
        return await run_data(lambda: client.get_key_metrics(ticker))

    @mcp.tool(annotations=market_data("Return & Risk Statistics"))
    async def analyze_performance(
        ticker: Ticker,
        period: Annotated[
            HistoryPeriod, Field(description="Look-back window for the statistics.")
        ] = "1y",
        risk_free_rate: RiskFreeRate = None,
    ) -> PerformanceStats:
        """Return and risk stats from daily auto-adjusted closes over the window.

        Includes total and annualized return, annualized volatility, max drawdown
        (negative percent), 50/200-day SMAs (null if insufficient history), and the
        risk-adjusted set: Sharpe, Sortino, downside deviation and Calmar (CAGR per unit
        of max drawdown).

        Annualized figures use the actual calendar span between the first and last bar,
        so over a one-year window the annualized return equals the total return for any
        instrument. Volatility is scaled by an observations-per-year factor inferred from
        the data and reported as periods_per_year (roughly 252 for a weekday-traded
        equity, 365 for a 24/7 instrument such as crypto). The annualized figures,
        periods_per_year and every risk-adjusted ratio are null when the window spans
        under 85 days (just under three months), because annualizing a sub-quarter move
        extrapolates noise into a yearly rate.

        Left out, risk_free_rate is the 13-week US T-bill yield averaged over the same
        dates, so Sharpe, Sortino and downside deviation are excess over cash; pass 0 for
        raw figures. The rate and its source are echoed in the result, and if the T-bill
        average cannot be formed those three are null with risk_free_rate_note saying why.
        For beta, alpha or a comparison against an index, use compare_to_benchmark.
        """
        return await run_data(lambda: client.analyze_performance(ticker, period, risk_free_rate))

    @mcp.tool(annotations=market_data("Benchmark-Relative Statistics"))
    async def compare_to_benchmark(
        ticker: Ticker,
        benchmark: Annotated[
            Ticker,
            Field(
                description="Benchmark to measure against; defaults to SPY (S&P 500). Use a "
                "benchmark that matches the asset: ^GSPC or SPY for US large-cap, QQQ for "
                "US tech, a local index for a non-US listing (returns are compared in each "
                "instrument's own quote currency)."
            ),
        ] = "SPY",
        period: Annotated[
            HistoryPeriod, Field(description="Look-back window for the comparison.")
        ] = "1y",
        risk_free_rate: RiskFreeRate = None,
    ) -> BenchmarkComparison:
        """Beta, correlation, Jensen's alpha, tracking error, information ratio and excess
        return versus a benchmark, over the dates the two instruments share.

        The two daily close series are inner-joined on date, so a 24/7 instrument compared
        against an equity benchmark contributes only its weekday closes (the weekend move
        lands in the Monday return). overlapping_observations reports how many dates were
        actually used - a thin overlap makes every figure noisy, so read it first.

        Annualized figures (both CAGRs, alpha, tracking error, information ratio) are null
        when the overlap spans under 85 days; beta, correlation and excess return are not,
        since they need no annualization. risk_free_rate only affects alpha; left out, it is
        the 13-week T-bill yield averaged over the overlapping dates (alpha is null if that
        cannot be formed). Returns are in each instrument's own quote currency, so a cross-currency
        pair folds an FX move into every figure - say so rather than reading it straight.
        """
        return await run_data(
            lambda: client.compare_to_benchmark(ticker, benchmark, period, risk_free_rate)
        )

    @mcp.tool(annotations=market_data("Side-by-Side Ticker Comparison"))
    async def compare_tickers(
        tickers: Annotated[
            list[Ticker],
            Field(
                min_length=2,
                max_length=MAX_COMPARE_TICKERS,
                description=(
                    f"2-{MAX_COMPARE_TICKERS} ticker symbols to compare, e.g. "
                    "['AAPL', 'MSFT', 'GOOGL']. For one ticker use analyze_performance."
                ),
            ),
        ],
        period: Annotated[
            HistoryPeriod, Field(description="Look-back window applied to every row.")
        ] = "1y",
        risk_free_rate: RiskFreeRate = None,
    ) -> TickerComparison:
        """Side-by-side performance and key valuation metrics for 2-10 tickers.

        Each row carries total/annualized return, volatility, max drawdown and the
        risk-adjusted ratios over `period` - measured against the caller's risk_free_rate,
        or by default the 13-week T-bill yield over that row's own dates (each row echoes
        its rate) - plus Yahoo's valuation metrics (P/E, forward
        P/E, P/B, P/S, PEG, EV/EBITDA, margins, ROE, debt/equity) in their as-reported
        units - margins and ROE are fractions, debt_to_equity is already a percent. Rank
        peers on PEG or growth-vs-multiple rather than raw P/E.

        Tickers are fetched in parallel and results are partial: a ticker whose price
        history could not be fetched is named in `errors` with the reason and has no row,
        and a row whose valuation metrics failed is still present with those fields null
        and `metrics_error` set. One bad ticker never invalidates the rest.

        Rows whose quote currency differs from the table's base_currency are flagged with
        currency_differs, and mixed_currencies summarises it: those returns carry an FX
        component the other rows do not, so compare such rows on ratios and say so. A row
        whose financial_currency differs from its currency is a cross-listing: its P/S, P/B
        and EV multiples mix two currencies, so rank it on P/E, PEG and the margins.
        """
        return await run_data(lambda: client.compare_tickers(tickers, period, risk_free_rate))
