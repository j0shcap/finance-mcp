"""MCP tools for equities market data, backed by YFinanceClient."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    FinancialStatement,
    HistoryInterval,
    HistoryPeriod,
    NewsResult,
    PriceHistory,
    QuoteResult,
    Statement,
    StatementPeriod,
    SymbolSearchResult,
)
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.tools._dispatch import run_data


def register(mcp: FastMCP, client: YFinanceClient) -> None:
    """Register equities tools bound to a YFinanceClient."""

    @mcp.tool
    async def get_quote(
        tickers: Annotated[
            list[str],
            Field(
                min_length=1,
                max_length=25,
                description="1-25 ticker symbols, e.g. ['AAPL', 'MSFT'].",
            ),
        ],
    ) -> QuoteResult:
        """Current price snapshots for up to 25 tickers (price, change, ranges, market cap).

        Tickers are fetched in parallel and results are partial: successful quotes come back
        in `quotes`, and any ticker that could not be fetched is named in `errors` with the
        reason (invalid/delisted symbol vs. a source failure). One bad ticker does not
        invalidate the rest, so there is no need to retry the whole batch.
        """
        return await run_data(lambda: client.get_quote(tickers))

    @mcp.tool
    async def get_price_history(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
        period: Annotated[HistoryPeriod, Field(description="Look-back window.")] = "1mo",
        interval: Annotated[
            HistoryInterval,
            Field(
                description=(
                    "Bar interval. Intraday intervals (1m-1h) only support short look-backs "
                    "(1m ~ 7 days, sub-daily ~ 60 days); use 1d+ for long periods."
                )
            ),
        ] = "1d",
    ) -> PriceHistory:
        """Historical OHLCV bars plus a summary; long windows are truncated (summary is full)."""
        return await run_data(lambda: client.get_price_history(ticker, period, interval))

    @mcp.tool
    async def get_financials(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
        statement: Annotated[
            Statement, Field(description="Which statement: 'income', 'balance', or 'cashflow'.")
        ],
        period: Annotated[
            StatementPeriod, Field(description="Reporting period granularity.")
        ] = "annual",
        line_items: Annotated[
            list[str] | None,
            Field(
                description="Specific line-item labels to return (as they appear in the statement, "
                "e.g. 'Total Revenue'; case and extra whitespace are ignored); omit for the full "
                "statement."
            ),
        ] = None,
    ) -> FinancialStatement:
        """Income statement, balance sheet, or cash flow.

        Returns line items by period (most recent first); values are in the currency named by
        `currency` (the company's reporting currency, which can differ from the currency its
        shares trade in) in absolute units (e.g. 416161000000 = 416.161 billion), null where not
        reported. Filtered labels that do not exist are reported in `missing_line_items`, with
        `available_line_items` and `line_item_suggestions` to retry from.
        """
        return await run_data(lambda: client.get_financials(ticker, statement, period, line_items))

    @mcp.tool
    async def get_company_profile(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
    ) -> CompanyProfile:
        """Company profile and key stats.

        Includes sector, industry, market cap and P/E, with recent dividends and splits.
        Note: dividend_yield is a percent (e.g. 5.92 means 5.92%).
        """
        return await run_data(lambda: client.get_company_profile(ticker))

    @mcp.tool
    async def get_analyst_data(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
    ) -> AnalystData:
        """Sell-side analyst consensus: price targets, the consensus recommendation, and the
        recent rating trend (analyst counts over the last four months).

        recommendation_mean runs 1.0 (strong buy) to 5.0 (strong sell). Price targets and
        current_price are in the result's currency. ETFs, indices, and crypto have no
        analyst coverage and return an error.
        """
        return await run_data(lambda: client.get_analyst_data(ticker))

    @mcp.tool
    async def get_news(
        ticker: Annotated[str, Field(description="Ticker symbol, e.g. 'AAPL'.")],
        count: Annotated[
            int, Field(ge=1, le=50, description="Maximum number of articles to return.")
        ] = 10,
    ) -> NewsResult:
        """Recent news headlines for a symbol, newest first.

        Each article has a title, publisher, link, publish time (ISO8601 UTC), and a short summary.
        Works for stocks, ETFs, and crypto. A symbol with no news (or an unknown symbol) returns an
        empty article list rather than an error.
        """
        return await run_data(lambda: client.get_news(ticker, count))

    @mcp.tool
    async def search_symbols(
        query: Annotated[
            str, Field(description="Company or instrument name to resolve, e.g. 'Apple'.")
        ],
        max_results: Annotated[
            int, Field(ge=1, le=20, description="Maximum number of matches to return.")
        ] = 8,
    ) -> SymbolSearchResult:
        """Resolve a company or instrument name to ticker symbol(s), best match first.

        Returns all instrument types (EQUITY, ETF, CRYPTOCURRENCY, FUTURE, INDEX, …); use each
        match's quote_type to choose. Use this to find a symbol before calling the other tools.
        An unmatched query returns an empty match list (not an error).
        """
        return await run_data(lambda: client.search_symbols(query, max_results))
