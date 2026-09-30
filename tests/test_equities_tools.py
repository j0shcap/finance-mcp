from typing import Any

import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFException

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.server import create_server
from tests.fakes import (
    QUOTE_FI,
    FakeSearch,
    fake_symbol_ticker_factory,
    fake_ticker_factory,
    make_client,
    make_financials_df,
    make_history_df,
    make_news_item,
    make_recommendations_df,
    make_series,
)

INCOME = {"Total Revenue": [400.0, 380.0], "Net Income": [100.0, 90.0]}
FULL_INFO = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "currency": "USD",
    "marketCap": 4.5e12,
    "trailingPE": 37.7,
}


async def test_get_quote_tool() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(fast_info=QUOTE_FI)))
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
        assert {"get_quote", "get_price_history"} <= names
        result = await client.call_tool("get_quote", {"tickers": ["AAPL"]})
        assert result.data.quotes[0].price == 190.0
        assert result.data.errors == []


async def test_get_quote_tool_multiple_tickers() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(fast_info=QUOTE_FI)))
    async with Client(server) as client:
        result = await client.call_tool("get_quote", {"tickers": ["AAPL", "MSFT"]})
        assert [q.symbol for q in result.data.quotes] == ["AAPL", "MSFT"]


async def test_get_price_history_tool() -> None:
    df = make_history_df([100.0, 101.0, 102.0])
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(history_df=df)))
    async with Client(server) as client:
        result = await client.call_tool(
            "get_price_history", {"ticker": "AAPL", "period": "1mo", "interval": "1d"}
        )
        assert result.data.summary.bars == 3


async def test_get_quote_tool_surfaces_yfinance_message_per_ticker() -> None:
    # A source failure is reported against the ticker it belongs to, not as a tool error:
    # the rest of the batch is still worth returning.
    server = create_server(
        yf_client=make_client(
            factory=fake_ticker_factory(fast_info_error=YFException("yahoo: blocked"))
        )
    )
    async with Client(server) as client:
        result = await client.call_tool("get_quote", {"tickers": ["AAPL"]})
        assert result.data.quotes == []
        assert result.data.errors[0].symbol == "AAPL"
        assert "yahoo: blocked" in result.data.errors[0].error


async def test_get_price_history_tool_surfaces_yfinance_message() -> None:
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(error=RuntimeError("yahoo: down")))
    )
    async with Client(server) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool(
                "get_price_history", {"ticker": "AAPL", "period": "1mo", "interval": "1d"}
            )
        assert "yahoo: down" in str(exc.value)


async def test_fundamentals_tools_registered() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(info=FULL_INFO)))
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
        assert {"get_financials", "get_company_profile"} <= names


async def test_get_financials_tool() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    )
    async with Client(server) as client:
        result = await client.call_tool(
            "get_financials", {"ticker": "AAPL", "statement": "income", "period": "annual"}
        )
        assert result.data.period_ends == ["2024-09-30", "2023-09-30"]
        assert result.data.line_items["Total Revenue"] == [400.0, 380.0]


async def test_get_financials_tool_line_items_filter() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    )
    async with Client(server) as client:
        result = await client.call_tool(
            "get_financials",
            {
                "ticker": "AAPL",
                "statement": "income",
                "period": "annual",
                "line_items": ["Net Income"],
            },
        )
        assert list(result.data.line_items.keys()) == ["Net Income"]


async def test_get_company_profile_tool() -> None:
    div = make_series(["2024-02-01", "2024-05-01"], [0.24, 0.25])
    spl = make_series(["2020-08-31"], [4.0])
    server = create_server(
        yf_client=make_client(
            factory=fake_ticker_factory(info=FULL_INFO, dividends=div, splits=spl)
        )
    )
    async with Client(server) as client:
        result = await client.call_tool("get_company_profile", {"ticker": "AAPL"})
        assert result.data.sector == "Technology"
        assert len(result.data.recent_dividends) == 2 and result.data.splits[0].ratio == 4.0


async def test_get_financials_tool_invalid_errors() -> None:
    server = create_server(
        yf_client=make_client(
            factory=fake_ticker_factory(financials={"income_stmt": pd.DataFrame()})
        )
    )
    async with Client(server) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool(
                "get_financials", {"ticker": "BAD", "statement": "income", "period": "annual"}
            )
        # The actionable, model-facing message must survive the run_data -> ToolError hop.
        assert "No income statement available for 'BAD'" in str(exc.value)


async def test_get_company_profile_tool_invalid_errors() -> None:
    def _factory(_symbol: str) -> object:
        raise DataUnavailable("no profile")

    server = create_server(yf_client=make_client(factory=_factory))
    async with Client(server) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool("get_company_profile", {"ticker": "BAD"})
        assert "no profile" in str(exc.value)


ANALYST_INFO = {
    "longName": "Apple Inc.",
    "currency": "USD",
    "currentPrice": 190.0,
    "recommendationKey": "buy",
    "recommendationMean": 1.9,
    "numberOfAnalystOpinions": 40,
    "targetMeanPrice": 210.0,
    "targetMedianPrice": 208.0,
    "targetHighPrice": 250.0,
    "targetLowPrice": 170.0,
}


async def test_get_analyst_data_tool() -> None:
    recs = make_recommendations_df(
        [
            ("0m", 12, 20, 8, 0, 0),
            ("-1m", 11, 21, 8, 0, 0),
            ("-2m", 10, 20, 9, 1, 0),
            ("-3m", 10, 19, 9, 1, 0),
        ]
    )
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(info=ANALYST_INFO, recommendations=recs))
    )
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
        assert {"get_analyst_data", "search_symbols"} <= names
        result = await client.call_tool("get_analyst_data", {"ticker": "AAPL"})
        assert result.data.recommendation_mean == 1.9
        assert result.data.target_mean_price == 210.0
        assert result.data.currency == "USD"
        assert len(result.data.recommendation_trend) == 4
        assert result.data.recommendation_trend[0].period == "0m"


async def test_get_analyst_data_tool_no_coverage_errors() -> None:
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(info={"longName": "SPDR ETF"}))
    )
    async with Client(server) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool("get_analyst_data", {"ticker": "SPY"})
        assert "No analyst coverage for 'SPY'" in str(exc.value)


async def test_search_symbols_tool() -> None:
    quotes: list[dict[str, Any]] = [
        {
            "symbol": "AAPL",
            "longname": "Apple Inc.",
            "quoteType": "EQUITY",
            "exchDisp": "NASDAQ",
            "score": 9000.0,
        },
        {"symbol": "APLE", "shortname": "Apple Hospitality", "quoteType": "REIT"},
    ]
    server = create_server(
        yf_client=make_client(
            factory=fake_ticker_factory(), search_factory=FakeSearch(quotes=quotes)
        )
    )
    async with Client(server) as client:
        result = await client.call_tool("search_symbols", {"query": "Apple"})
        assert [m.symbol for m in result.data.matches] == ["AAPL", "APLE"]
        assert result.data.matches[0].quote_type == "EQUITY"


async def test_search_symbols_tool_empty_is_not_error() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory()))
    async with Client(server) as client:
        result = await client.call_tool("search_symbols", {"query": "zzzznope"})
        assert result.data.matches == []


async def test_search_symbols_tool_surfaces_error() -> None:
    server = create_server(
        yf_client=make_client(
            factory=fake_ticker_factory(),
            search_factory=FakeSearch(error=RuntimeError("yahoo: search down")),
        )
    )
    async with Client(server) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool("search_symbols", {"query": "Apple"})
        assert "yahoo: search down" in str(exc.value)


async def test_get_news_tool() -> None:
    items = [
        make_news_item(
            "Apple hits record high",
            publisher="Yahoo Finance",
            link="https://finance.yahoo.com/news/a",
            published="2026-05-31T11:44:34Z",
            summary="Shares rally.",
        ),
        make_news_item(
            "Analysts upgrade Apple",
            publisher="Reuters",
            link="https://finance.yahoo.com/news/b",
            published="2026-05-30T09:00:00Z",
        ),
    ]
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(news=items)))
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
        assert "get_news" in names
        result = await client.call_tool("get_news", {"ticker": "AAPL"})
        assert result.data.symbol == "AAPL"
        assert [a.title for a in result.data.articles] == [
            "Apple hits record high",
            "Analysts upgrade Apple",
        ]
        assert result.data.articles[0].publisher == "Yahoo Finance"


async def test_get_news_tool_empty_is_not_error() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(news=[])))
    async with Client(server) as client:
        result = await client.call_tool("get_news", {"ticker": "ZZZZ"})
        assert result.data.articles == []


async def test_get_news_tool_surfaces_error() -> None:
    server = create_server(
        yf_client=make_client(
            factory=fake_ticker_factory(news_error=RuntimeError("yahoo: news down"))
        )
    )
    async with Client(server) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool("get_news", {"ticker": "AAPL"})
        assert "yahoo: news down" in str(exc.value)


async def test_get_quote_tool_returns_partial_results() -> None:
    server = create_server(
        yf_client=make_client(
            factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI, "MSFT": QUOTE_FI})
        )
    )
    async with Client(server) as client:
        result = await client.call_tool("get_quote", {"tickers": ["AAPL", "NOPE", "MSFT"]})
        assert [q.symbol for q in result.data.quotes] == ["AAPL", "MSFT"]
        assert [e.symbol for e in result.data.errors] == ["NOPE"]


async def test_get_quote_tool_rejects_an_empty_ticker_list() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(fast_info=QUOTE_FI)))
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_quote", {"tickers": []})


async def test_get_quote_tool_rejects_more_than_25_tickers() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(fast_info=QUOTE_FI)))
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_quote", {"tickers": [f"SYM{i}" for i in range(26)]})


async def test_get_quote_tool_accepts_25_tickers() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(fast_info=QUOTE_FI)))
    async with Client(server) as client:
        result = await client.call_tool("get_quote", {"tickers": [f"SYM{i}" for i in range(25)]})
        assert len(result.data.quotes) == 25


async def test_get_financials_tool_rejects_an_empty_line_items_filter() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    )
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "get_financials", {"ticker": "AAPL", "statement": "income", "line_items": []}
            )
