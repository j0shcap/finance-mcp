from typing import Any

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFException

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.tools._inputs import MAX_QUOTE_TICKERS
from tests.fakes import (
    QUOTE_FI,
    FakeSearch,
    connect,
    fake_symbol_ticker_factory,
    fake_ticker_factory,
    make_financials_df,
    make_history_df,
    make_news_item,
    make_recommendations_df,
    make_series,
)

INCOME = {"Total Revenue": [400.0, 380.0], "Net Income": [100.0, 90.0]}
INCOME_STATEMENT = {"income_stmt": make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])}
FULL_INFO = {
    "longName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "currency": "USD",
    "marketCap": 4.5e12,
    "trailingPE": 37.7,
}
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


def _profile_unavailable(_symbol: str) -> object:
    raise DataUnavailable("no profile")


async def test_get_quote_tool() -> None:
    async with connect(fake_ticker_factory(fast_info=QUOTE_FI)) as client:
        result = await client.call_tool("get_quote", {"tickers": ["AAPL", "MSFT"]})
    assert [q.symbol for q in result.data.quotes] == ["AAPL", "MSFT"]
    assert result.data.quotes[0].price == 190.0
    assert result.data.errors == []


async def test_get_quote_tool_accepts_the_largest_batch() -> None:
    tickers = [f"SYM{i}" for i in range(MAX_QUOTE_TICKERS)]
    async with connect(fake_ticker_factory(fast_info=QUOTE_FI)) as client:
        result = await client.call_tool("get_quote", {"tickers": tickers})
    assert len(result.data.quotes) == MAX_QUOTE_TICKERS


async def test_get_quote_tool_returns_partial_results() -> None:
    factory = fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI, "MSFT": QUOTE_FI})
    async with connect(factory) as client:
        result = await client.call_tool("get_quote", {"tickers": ["AAPL", "NOPE", "MSFT"]})
    assert [q.symbol for q in result.data.quotes] == ["AAPL", "MSFT"]
    assert [e.symbol for e in result.data.errors] == ["NOPE"]


async def test_get_quote_tool_reports_a_source_failure_against_its_ticker() -> None:
    # Not a tool error: the rest of the batch is still worth returning.
    factory = fake_ticker_factory(fast_info_error=YFException("yahoo: blocked"))
    async with connect(factory) as client:
        result = await client.call_tool("get_quote", {"tickers": ["AAPL"]})
    assert result.data.quotes == []
    assert result.data.errors[0].symbol == "AAPL"
    assert "yahoo: blocked" in result.data.errors[0].error


async def test_get_price_history_tool() -> None:
    factory = fake_ticker_factory(history_df=make_history_df([100.0, 101.0, 102.0]))
    async with connect(factory) as client:
        result = await client.call_tool(
            "get_price_history", {"ticker": "AAPL", "period": "1mo", "interval": "1d"}
        )
    assert result.data.summary.bars == 3


async def test_get_financials_tool() -> None:
    async with connect(fake_ticker_factory(financials=INCOME_STATEMENT)) as client:
        result = await client.call_tool(
            "get_financials", {"ticker": "AAPL", "statement": "income", "period": "annual"}
        )
    assert result.data.period_ends == ["2024-09-30", "2023-09-30"]
    assert result.data.line_items["Total Revenue"] == [400.0, 380.0]


async def test_get_financials_tool_line_items_filter() -> None:
    async with connect(fake_ticker_factory(financials=INCOME_STATEMENT)) as client:
        result = await client.call_tool(
            "get_financials",
            {"ticker": "AAPL", "statement": "income", "line_items": ["Net Income"]},
        )
    assert list(result.data.line_items.keys()) == ["Net Income"]


async def test_get_company_profile_tool() -> None:
    factory = fake_ticker_factory(
        info=FULL_INFO,
        dividends=make_series(["2024-02-01", "2024-05-01"], [0.24, 0.25]),
        splits=make_series(["2020-08-31"], [4.0]),
    )
    async with connect(factory) as client:
        result = await client.call_tool("get_company_profile", {"ticker": "AAPL"})
    assert result.data.sector == "Technology"
    assert len(result.data.recent_dividends) == 2
    assert result.data.splits[0].ratio == 4.0


async def test_get_analyst_data_tool() -> None:
    recs = make_recommendations_df(
        [
            ("0m", 12, 20, 8, 0, 0),
            ("-1m", 11, 21, 8, 0, 0),
            ("-2m", 10, 20, 9, 1, 0),
            ("-3m", 10, 19, 9, 1, 0),
        ]
    )
    async with connect(fake_ticker_factory(info=ANALYST_INFO, recommendations=recs)) as client:
        result = await client.call_tool("get_analyst_data", {"ticker": "AAPL"})
    assert result.data.recommendation_mean == 1.9
    assert result.data.target_mean_price == 210.0
    assert result.data.currency == "USD"
    assert [row.period for row in result.data.recommendation_trend] == ["0m", "-1m", "-2m", "-3m"]


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
    async with connect(search_factory=FakeSearch(quotes=quotes)) as client:
        result = await client.call_tool("search_symbols", {"query": "Apple"})
    assert [m.symbol for m in result.data.matches] == ["AAPL", "APLE"]
    assert result.data.matches[0].quote_type == "EQUITY"


async def test_search_symbols_tool_empty_is_not_error() -> None:
    async with connect() as client:
        result = await client.call_tool("search_symbols", {"query": "zzzznope"})
    assert result.data.matches == []


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
    async with connect(fake_ticker_factory(news=items)) as client:
        result = await client.call_tool("get_news", {"ticker": "AAPL"})
    assert result.data.symbol == "AAPL"
    assert [a.title for a in result.data.articles] == [
        "Apple hits record high",
        "Analysts upgrade Apple",
    ]
    assert result.data.articles[0].publisher == "Yahoo Finance"


async def test_get_news_tool_empty_is_not_error() -> None:
    async with connect(fake_ticker_factory(news=[])) as client:
        result = await client.call_tool("get_news", {"ticker": "ZZZZ"})
    assert result.data.articles == []


@pytest.mark.parametrize(
    ("tool", "arguments", "options", "message"),
    [
        (
            "get_price_history",
            {"ticker": "AAPL"},
            {"factory": fake_ticker_factory(error=RuntimeError("yahoo: down"))},
            "yahoo: down",
        ),
        (
            "get_financials",
            {"ticker": "BAD", "statement": "income"},
            {"factory": fake_ticker_factory(financials={"income_stmt": pd.DataFrame()})},
            "No income statement available for 'BAD'",
        ),
        (
            "get_company_profile",
            {"ticker": "BAD"},
            {"factory": _profile_unavailable},
            "no profile",
        ),
        (
            "get_analyst_data",
            {"ticker": "SPY"},
            {"factory": fake_ticker_factory(info={"longName": "SPDR ETF"})},
            "No analyst coverage for 'SPY'",
        ),
        (
            "search_symbols",
            {"query": "Apple"},
            {"search_factory": FakeSearch(error=RuntimeError("yahoo: search down"))},
            "yahoo: search down",
        ),
        (
            "get_news",
            {"ticker": "AAPL"},
            {"factory": fake_ticker_factory(news_error=RuntimeError("yahoo: news down"))},
            "yahoo: news down",
        ),
    ],
)
async def test_data_layer_message_reaches_the_model_as_a_tool_error(
    tool: str, arguments: dict[str, Any], options: dict[str, Any], message: str
) -> None:
    async with connect(**options) as client:
        with pytest.raises(ToolError) as exc:
            await client.call_tool(tool, arguments)
    assert message in str(exc.value)
