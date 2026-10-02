"""MCP tools for equities market data, backed by DataService."""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    Earnings,
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
from finance_mcp.data.service import DataService
from finance_mcp.tools._annotations import market_data
from finance_mcp.tools._dispatch import run_data
from finance_mcp.tools._inputs import MAX_LINE_ITEMS, MAX_QUOTE_TICKERS, Ticker


def register(mcp: FastMCP, service: DataService) -> None:
    """Register equities tools bound to a DataService."""

    @mcp.tool(annotations=market_data("Stock Quotes"))
    async def get_quote(
        tickers: Annotated[
            list[Ticker],
            Field(
                min_length=1,
                max_length=MAX_QUOTE_TICKERS,
                description=f"1-{MAX_QUOTE_TICKERS} ticker symbols, e.g. ['AAPL', 'MSFT'].",
            ),
        ],
    ) -> QuoteResult:
        """Current price snapshots for up to 25 tickers (price, change, ranges, market cap).

        Tickers are fetched in parallel and results are partial: successful quotes come back
        in `quotes`, and any ticker that could not be fetched is named in `errors` with the
        reason (invalid/delisted symbol vs. a source failure). One bad ticker does not
        invalidate the rest, so there is no need to retry the whole batch.
        """
        return await run_data(lambda: service.get_quote(tickers))

    @mcp.tool(annotations=market_data("Price History (OHLCV)"))
    async def get_price_history(
        ticker: Ticker,
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
        return await run_data(lambda: service.get_price_history(ticker, period, interval))

    @mcp.tool(annotations=market_data("Financial Statements"))
    async def get_financials(
        ticker: Ticker,
        statement: Annotated[
            Statement, Field(description="Which statement: 'income', 'balance', or 'cashflow'.")
        ],
        period: Annotated[
            StatementPeriod, Field(description="Reporting period granularity.")
        ] = "annual",
        line_items: Annotated[
            list[Annotated[str, Field(min_length=1, max_length=120)]] | None,
            Field(
                min_length=1,
                max_length=MAX_LINE_ITEMS,
                description="Specific line-item labels to return (as they appear in the statement, "
                "e.g. 'Total Revenue'; case and extra whitespace are ignored); omit for the full "
                "statement.",
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
        return await run_data(lambda: service.get_financials(ticker, statement, period, line_items))

    @mcp.tool(annotations=market_data("Company Profile"))
    async def get_company_profile(
        ticker: Ticker,
    ) -> CompanyProfile:
        """Company profile and key stats.

        Includes sector, industry, market cap and P/E, with recent dividends and splits.
        Note: dividend_yield is a percent (e.g. 5.92 means 5.92%).
        """
        return await run_data(lambda: service.get_company_profile(ticker))

    @mcp.tool(annotations=market_data("Analyst Ratings & Price Targets"))
    async def get_analyst_data(
        ticker: Ticker,
    ) -> AnalystData:
        """Sell-side analyst consensus: price targets, the consensus recommendation, and the
        recent rating trend (analyst counts over the last four months).

        recommendation_mean runs 1.0 (strong buy) to 5.0 (strong sell). Price targets and
        current_price are in the result's currency. ETFs, indices, and crypto have no
        analyst coverage and return an error.
        """
        return await run_data(lambda: service.get_analyst_data(ticker))

    @mcp.tool(annotations=market_data("Earnings Dates & Estimates"))
    async def get_earnings(
        ticker: Ticker,
    ) -> Earnings:
        """When a company reports next, what analysts expect, and whether it beat lately.

        next_report: the next report's date-time in the exchange's timezone, and whether the
        company has confirmed it (date_is_estimate false; true or null means it hasn't).
        estimates: EPS and revenue consensus (average, low, high, analyst count, year-ago value,
        growth) for the quarter the next report covers, the quarter after, and their fiscal
        years, each with fiscal_period_end. history: the last four quarters' EPS against the
        consensus, oldest first, with surprise_percent. growth_percent and surprise_percent are
        PERCENTS; EPS, revenue and history can each be in a different currency, so check
        eps_currency, revenue_currency and history_currency. Companies only: ETFs, funds,
        indices, currencies and crypto return an error.
        """
        return await run_data(lambda: service.get_earnings(ticker))

    @mcp.tool(annotations=market_data("Company News"))
    async def get_news(
        ticker: Ticker,
        count: Annotated[
            int, Field(ge=1, le=50, description="Maximum number of articles to return.")
        ] = 10,
    ) -> NewsResult:
        """Recent news headlines for a symbol, newest first.

        Each article has a title, publisher, link and publish time (ISO8601 UTC). Summaries come
        from the per-symbol news stream only: when `source` is "search" that stream returned
        nothing and this fell back to a news search, which carries no summary, so every
        summary is null for a reason unrelated to the stories. Works for stocks, ETFs, and crypto.
        A symbol with no news (or an unknown symbol) returns an empty article list, not an error.

        A ticker's feed can include market-wide stories too, so for a stock each article's
        mentions_company says whether its title or summary names the company or its ticker.
        It is a text match (brand and executive names are not), so it flags rather than
        filters: every article is returned, in order. relevance_check says when the flags are
        null (not a stock, no company name is known for it, or fetching the name failed, with
        relevance_note saying why).
        """
        return await run_data(lambda: service.get_news(ticker, count))

    @mcp.tool(annotations=market_data("Ticker Symbol Search"))
    async def search_symbols(
        query: Annotated[
            str,
            Field(
                min_length=1,
                max_length=128,
                pattern=r"\S",
                description="Company or instrument name to resolve, e.g. 'Apple'.",
            ),
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
        return await run_data(lambda: service.search_symbols(query, max_results))
