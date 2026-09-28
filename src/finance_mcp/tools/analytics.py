"""MCP tools for computed analytics, backed by YFinanceClient."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.data.models import HistoryPeriod, KeyMetrics, PerformanceStats
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.tools._dispatch import run_data


def register(mcp: FastMCP, client: YFinanceClient) -> None:
    """Register analytics tools bound to a YFinanceClient."""

    @mcp.tool
    async def get_key_metrics(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
    ) -> KeyMetrics:
        """Valuation, profitability, and leverage ratios (as reported by Yahoo).

        Note units differ by field: P/E, P/B, P/S, EV/EBITDA, PEG are plain ratios;
        margins and ROE/ROA are fractions (0.27 = 27%); debt_to_equity is a percent
        (79.5 = 79.5%); EV, total debt/cash, FCF, EBITDA are absolute amounts. Those
        amounts are not all in one currency: debt/cash/FCF/EBITDA and the per-share
        revenue/book value are in `financial_currency`, EV and the EPS fields in
        `currency`. They differ for ADRs and other cross-listings.
        """
        return await run_data(lambda: client.get_key_metrics(ticker))

    @mcp.tool
    async def analyze_performance(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
        period: Annotated[
            HistoryPeriod, Field(description="Look-back window for the statistics.")
        ] = "1y",
    ) -> PerformanceStats:
        """Return and risk stats from daily auto-adjusted closes over the window.

        Includes total and annualized return, annualized volatility, max drawdown
        (negative percent), and 50/200-day SMAs (null if insufficient history).

        Annualized figures use the actual calendar span between the first and last bar,
        so over a one-year window the annualized return equals the total return for any
        instrument. Volatility is scaled by an observations-per-year factor inferred from
        the data and reported as periods_per_year (roughly 261 for a weekday-traded
        equity, 365 for a 24/7 instrument such as crypto). Both annualized figures and
        periods_per_year are null when the window spans under 90 days, because
        annualizing a sub-quarter move extrapolates noise into a yearly rate.
        """
        return await run_data(lambda: client.analyze_performance(ticker, period))
