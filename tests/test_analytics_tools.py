from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from fastmcp.exceptions import ToolError

from finance_mcp.data import analytics
from finance_mcp.data.yfinance_client import TREASURY_BILL_SYMBOL
from tests.fakes import connect, fake_multi_ticker_factory, fake_ticker_factory, make_history_df

METRICS_INFO = {
    "longName": "Apple Inc.",
    "currency": "USD",
    "trailingPE": 37.73,
    "profitMargins": 0.271,
    "returnOnEquity": 1.41,
    "debtToEquity": 79.55,
    "freeCashflow": 101090746368,
}
#: 120 daily closes trending up with regular dips, so the downside deviation is non-zero.
CHOPPY_CLOSES = [100.0 + (i % 5) + i * 0.3 for i in range(120)]


async def test_get_key_metrics_tool() -> None:
    async with connect(fake_ticker_factory(info=METRICS_INFO)) as client:
        result = await client.call_tool("get_key_metrics", {"ticker": "AAPL"})
        assert result.data.trailing_pe == 37.73 and result.data.profit_margins == 0.271


async def test_analyze_performance_tool() -> None:
    df = make_history_df([100.0, 110.0, 99.0])
    async with connect(fake_ticker_factory(history_df=df)) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "1mo"})
        assert result.data.bars == 3
        assert result.data.total_return_percent == pytest.approx(-1.0)


async def test_analyze_performance_tool_reports_the_annualization_factor() -> None:
    df = make_history_df([100.0 + i for i in range(200)])
    async with connect(fake_ticker_factory(history_df=df)) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "1y"})
        assert result.data.periods_per_year == pytest.approx(365.25, rel=0.02)
        assert result.data.annualized_return_percent is not None
        assert result.data.annualized_volatility_percent is not None


async def test_analyze_performance_tool_nulls_annualized_fields_on_short_window() -> None:
    df = make_history_df([100.0, 101.0, 102.0, 103.0, 104.0])
    async with connect(fake_ticker_factory(history_df=df)) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "5d"})
        assert result.data.annualized_return_percent is None
        assert result.data.annualized_volatility_percent is None
        assert result.data.periods_per_year is None
        assert result.data.total_return_percent == pytest.approx(4.0)


async def test_analyze_performance_schema_documents_the_short_window_null() -> None:
    async with connect(fake_ticker_factory()) as client:
        [tool] = [t for t in await client.list_tools() if t.name == "analyze_performance"]
        schema = (tool.outputSchema or {})["properties"]
        assert "85 days" in schema["annualized_return_percent"]["description"]
        assert "252 trading days" not in schema["annualized_return_percent"]["description"]
        assert "252-day" not in schema["annualized_volatility_percent"]["description"]
        assert "periods_per_year" in schema
        # Annualization infers periods_per_year from the data, so the tool description must
        # not advertise a fixed 252-trading-day year.
        assert tool.description is not None and "252 trading days" not in tool.description


async def test_get_key_metrics_tool_invalid_errors() -> None:
    async with connect(fake_ticker_factory(info={"x": 1})) as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_key_metrics", {"ticker": "BAD"})


async def test_analyze_performance_tool_invalid_errors() -> None:
    async with connect(fake_ticker_factory(history_df=pd.DataFrame())) as client:
        with pytest.raises(ToolError):
            await client.call_tool("analyze_performance", {"ticker": "BAD", "period": "1y"})


async def test_analyze_performance_tool_reports_risk_adjusted_stats() -> None:
    df = make_history_df(CHOPPY_CLOSES)
    async with connect(fake_ticker_factory(history_df=df)) as client:
        result = await client.call_tool(
            "analyze_performance", {"ticker": "AAPL", "period": "6mo", "risk_free_rate": 0.04}
        )
        assert result.data.risk_free_rate == 0.04
        assert result.data.sharpe_ratio is not None
        assert result.data.downside_deviation_percent is not None


def _performance_factory() -> Callable[[str], Any]:
    """An asset plus the T-bill history the default risk-free rate is drawn from."""
    return fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(CHOPPY_CLOSES)},
            TREASURY_BILL_SYMBOL: {"history_df": make_history_df([4.03] * 120)},
        }
    )


@pytest.mark.parametrize("arguments", [{}, {"risk_free_rate": None}])
async def test_analyze_performance_tool_defaults_to_the_treasury_bill_rate(
    arguments: dict[str, Any],
) -> None:
    async with connect(_performance_factory()) as client:
        result = await client.call_tool(
            "analyze_performance", {"ticker": "AAPL", "period": "6mo", **arguments}
        )
        assert result.data.risk_free_rate == pytest.approx(
            analytics.treasury_bill_effective_rate(4.03), rel=1e-6
        )
        assert result.data.risk_free_rate_source == "treasury_bill"
        assert result.data.sharpe_ratio is not None


async def test_analyze_performance_tool_reports_an_unavailable_rate_without_failing() -> None:
    df = make_history_df(CHOPPY_CLOSES)
    factory = fake_multi_ticker_factory({"AAPL": {"history_df": df}})  # no T-bill history
    async with connect(factory) as client:
        result = await client.call_tool("analyze_performance", {"ticker": "AAPL", "period": "6mo"})
        assert result.data.risk_free_rate is None
        assert result.data.risk_free_rate_source == "unavailable"
        assert result.data.risk_free_rate_note is not None
        assert result.data.sharpe_ratio is None
        assert result.data.annualized_return_percent is not None


async def test_analyze_performance_schema_states_the_risk_free_default_in_the_output() -> None:
    # The default must be legible from the RESULT, not just the input schema, so a model
    # reading a Sharpe knows whether it is an excess figure.
    async with connect(fake_ticker_factory()) as client:
        [tool] = [t for t in await client.list_tools() if t.name == "analyze_performance"]
        properties = (tool.outputSchema or {})["properties"]
        assert "T-bill" in properties["risk_free_rate"]["description"]
        assert "RAW" in properties["risk_free_rate"]["description"]
        assert "unavailable" in properties["risk_free_rate_source"]["description"]
        assert "downside_deviation" in properties["sortino_ratio"]["description"]
        assert tool.description is not None and "Sharpe" in tool.description
        input_rate = tool.inputSchema["properties"]["risk_free_rate"]
        assert input_rate.get("default") is None
        assert "T-bill" in input_rate["description"]


async def test_compare_to_benchmark_tool() -> None:
    closes = [100.0 + i * 0.3 for i in range(200)]
    bench = [400.0 + i * 0.8 for i in range(200)]
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(closes)},
            "SPY": {"history_df": make_history_df(bench)},
        }
    )
    async with connect(factory) as client:
        result = await client.call_tool("compare_to_benchmark", {"ticker": "AAPL"})
        assert result.data.benchmark == "SPY"  # the default
        assert result.data.period == "1y"  # the default
        assert result.data.overlapping_observations == 200
        assert result.data.beta is not None


async def test_compare_to_benchmark_tool_rejects_a_self_comparison_as_a_tool_error() -> None:
    factory = fake_multi_ticker_factory(
        {"SPY": {"history_df": make_history_df([400.0 + i for i in range(200)])}}
    )
    async with connect(factory) as client:
        with pytest.raises(ToolError, match="two different"):
            await client.call_tool("compare_to_benchmark", {"ticker": "SPY", "benchmark": "SPY"})


async def test_compare_to_benchmark_schema_explains_the_inner_join() -> None:
    async with connect(fake_ticker_factory()) as client:
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
    async with connect(factory) as client:
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
    async with connect(factory) as client:
        result = await client.call_tool("compare_tickers", {"tickers": ["AAPL", "NOPE"]})
        assert [row.symbol for row in result.data.rows] == ["AAPL"]
        assert [err.symbol for err in result.data.errors] == ["NOPE"]
