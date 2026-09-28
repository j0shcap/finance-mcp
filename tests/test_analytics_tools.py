import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from finance_mcp.server import create_server
from tests.conftest import fake_ticker_factory, make_client, make_history_df

METRICS_INFO = {
    "longName": "Apple Inc.",
    "trailingPE": 37.73,
    "profitMargins": 0.271,
    "returnOnEquity": 1.41,
    "debtToEquity": 79.55,
    "freeCashflow": 101090746368,
}


async def test_analytics_tools_registered() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(info=METRICS_INFO)))
    async with Client(server) as client:
        names = {t.name for t in await client.list_tools()}
        assert {"get_key_metrics", "analyze_performance"} <= names
        assert len(names) == 19  # 18 prior + get_news


async def test_get_key_metrics_tool() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(info=METRICS_INFO)))
    async with Client(server) as client:
        result = await client.call_tool("get_key_metrics", {"ticker": "AAPL"})
        assert result.data.trailing_pe == 37.73 and result.data.profit_margins == 0.271


async def test_analyze_performance_tool() -> None:
    df = make_history_df([100.0, 110.0, 99.0])
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(history_df=df)))
    async with Client(server) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "1mo"})
        assert result.data.bars == 3
        assert result.data.total_return_percent == pytest.approx(-1.0)


async def test_analyze_performance_tool_reports_the_annualization_factor() -> None:
    df = make_history_df([100.0 + i for i in range(200)])
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(history_df=df)))
    async with Client(server) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "1y"})
        assert result.data.periods_per_year == pytest.approx(365.25, rel=0.02)
        assert result.data.annualized_return_percent is not None
        assert result.data.annualized_volatility_percent is not None


async def test_analyze_performance_tool_nulls_annualized_fields_on_short_window() -> None:
    df = make_history_df([100.0, 101.0, 102.0, 103.0, 104.0])
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(history_df=df)))
    async with Client(server) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "5d"})
        assert result.data.annualized_return_percent is None
        assert result.data.annualized_volatility_percent is None
        assert result.data.periods_per_year is None
        assert result.data.total_return_percent == pytest.approx(4.0)


async def test_analyze_performance_schema_documents_the_short_window_null() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory()))
    async with Client(server) as client:
        [tool] = [t for t in await client.list_tools() if t.name == "analyze_performance"]
        schema = (tool.outputSchema or {})["properties"]
        assert "90 days" in schema["annualized_return_percent"]["description"]
        assert "252" not in schema["annualized_return_percent"]["description"]
        assert "252" not in schema["annualized_volatility_percent"]["description"]
        assert "periods_per_year" in schema
        # The tool description is what a model reads before choosing the tool; it must not
        # still advertise the fixed trading-day convention this change removed.
        assert tool.description is not None and "252" not in tool.description


async def test_get_key_metrics_tool_invalid_errors() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(info={"x": 1})))
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_key_metrics", {"ticker": "BAD"})


async def test_analyze_performance_tool_invalid_errors() -> None:
    server = create_server(
        yf_client=make_client(factory=fake_ticker_factory(history_df=pd.DataFrame()))
    )
    async with Client(server) as client:
        with pytest.raises(ToolError):
            await client.call_tool("analyze_performance", {"ticker": "BAD", "period": "1y"})
