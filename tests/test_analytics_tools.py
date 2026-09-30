import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from finance_mcp.server import create_server
from tests.fakes import (
    fake_multi_ticker_factory,
    fake_ticker_factory,
    make_client,
    make_history_df,
)

METRICS_INFO = {
    "longName": "Apple Inc.",
    "currency": "USD",
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
        # 19 prior + compare_to_benchmark + compare_tickers + the two dated bond tools.
        assert len(names) == 23


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
        assert "85 days" in schema["annualized_return_percent"]["description"]
        assert "252 trading days" not in schema["annualized_return_percent"]["description"]
        assert "252-day" not in schema["annualized_volatility_percent"]["description"]
        assert "periods_per_year" in schema
        # The tool description is what a model reads before choosing the tool; it must not
        # still advertise the fixed trading-day convention this change removed.
        assert tool.description is not None and "252 trading days" not in tool.description


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


async def test_analyze_performance_tool_reports_risk_adjusted_stats() -> None:
    df = make_history_df([100.0 + (i % 5) + i * 0.3 for i in range(120)])
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(history_df=df)))
    async with Client(server) as client:
        result = await client.call_tool(
            "analyze_performance", {"ticker": "AAPL", "period": "6mo", "risk_free_rate": 0.04}
        )
        assert result.data.risk_free_rate == 0.04
        assert result.data.sharpe_ratio is not None
        assert result.data.downside_deviation_percent is not None


async def test_analyze_performance_tool_defaults_the_risk_free_rate_to_zero() -> None:
    df = make_history_df([100.0 + (i % 5) + i * 0.3 for i in range(120)])
    server = create_server(yf_client=make_client(factory=fake_ticker_factory(history_df=df)))
    async with Client(server) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "6mo"})
        assert result.data.risk_free_rate == 0.0


async def test_analyze_performance_schema_states_the_risk_free_default_in_the_output() -> None:
    """The default must be legible from the RESULT, not just the input schema, so a model
    reading a Sharpe knows whether it is an excess figure."""
    server = create_server(yf_client=make_client(factory=fake_ticker_factory()))
    async with Client(server) as client:
        [tool] = [t for t in await client.list_tools() if t.name == "analyze_performance"]
        properties = (tool.outputSchema or {})["properties"]
        assert "Defaults to 0" in properties["risk_free_rate"]["description"]
        assert "RAW" in properties["risk_free_rate"]["description"]
        assert "downside_deviation" in properties["sortino_ratio"]["description"]
        assert tool.description is not None and "Sharpe" in tool.description


async def test_compare_to_benchmark_tool() -> None:
    closes = [100.0 + i * 0.3 for i in range(200)]
    bench = [400.0 + i * 0.8 for i in range(200)]
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(closes)},
            "SPY": {"history_df": make_history_df(bench)},
        }
    )
    server = create_server(yf_client=make_client(factory=factory))
    async with Client(server) as client:
        result = await client.call_tool("compare_to_benchmark", {"ticker": "AAPL"})
        assert result.data.benchmark == "SPY"  # the default
        assert result.data.period == "1y"  # the default
        assert result.data.overlapping_observations == 200
        assert result.data.beta is not None


async def test_compare_to_benchmark_tool_rejects_a_self_comparison_as_a_tool_error() -> None:
    factory = fake_multi_ticker_factory(
        {"SPY": {"history_df": make_history_df([400.0 + i for i in range(200)])}}
    )
    server = create_server(yf_client=make_client(factory=factory))
    async with Client(server) as client:
        with pytest.raises(ToolError, match="two different"):
            await client.call_tool("compare_to_benchmark", {"ticker": "SPY", "benchmark": "SPY"})


async def test_compare_to_benchmark_schema_explains_the_inner_join() -> None:
    server = create_server(yf_client=make_client(factory=fake_ticker_factory()))
    async with Client(server) as client:
        [tool] = [t for t in await client.list_tools() if t.name == "compare_to_benchmark"]
        properties = (tool.outputSchema or {})["properties"]
        assert "inner join" in properties["overlapping_observations"]["description"]
        assert tool.description is not None
        assert "inner-joined" in tool.description
        assert "quote currency" in tool.description


async def test_compare_tickers_tool() -> None:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {
                "history_df": make_history_df([100.0 + i * 0.3 for i in range(200)]),
                "info": METRICS_INFO,
            },
            "MSFT": {
                "history_df": make_history_df([200.0 + i * 0.5 for i in range(200)]),
                "info": METRICS_INFO,
            },
        }
    )
    server = create_server(yf_client=make_client(factory=factory))
    async with Client(server) as client:
        result = await client.call_tool("compare_tickers", {"tickers": ["AAPL", "MSFT"]})
        assert [row.symbol for row in result.data.rows] == ["AAPL", "MSFT"]
        assert result.data.period == "1y"
        assert result.data.errors == []
        assert result.data.base_currency == "USD"
        assert result.data.mixed_currencies is False


async def test_compare_tickers_tool_returns_partial_results() -> None:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {
                "history_df": make_history_df([100.0 + i * 0.3 for i in range(200)]),
                "info": METRICS_INFO,
            }
        }
    )
    server = create_server(yf_client=make_client(factory=factory))
    async with Client(server) as client:
        result = await client.call_tool("compare_tickers", {"tickers": ["AAPL", "NOPE"]})
        assert [row.symbol for row in result.data.rows] == ["AAPL"]
        assert [err.symbol for err in result.data.errors] == ["NOPE"]
