"""Bounds enforced at the tool boundary, before any fetch or numeric work runs.

Each test that rejects an argument also asserts the data layer was never reached, which
is the point of validating at the boundary: a malformed ticker should cost nothing.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError

from finance_mcp.tools._inputs import (
    MAX_CASHFLOWS,
    MAX_COMPARE_TICKERS,
    MAX_LINE_ITEMS,
    MAX_QUOTE_TICKERS,
    TICKER_PATTERN,
)
from tests.fakes import connect, counting, fake_ticker_factory


@pytest.fixture
async def fetches() -> AsyncIterator[tuple[Client[FastMCPTransport], list[str]]]:
    """A client over a fake data layer, plus the list of symbols it was asked to fetch."""
    factory, calls = counting(
        fake_ticker_factory(
            info={"longName": "Some Instrument", "currency": "USD"},
            fast_info={"last_price": 190.0, "currency": "USD"},
        )
    )
    async with connect(factory) as client:
        yield client, calls


@pytest.mark.parametrize(
    "ticker",
    [
        "",  # empty
        "   ",  # whitespace only
        "AAPL MSFT",  # two symbols in one argument
        "AA;DROP",  # punctuation Yahoo symbols never contain
        "A" * 40,  # implausibly long
    ],
)
async def test_malformed_ticker_is_rejected_without_a_fetch(
    fetches: tuple[Client[FastMCPTransport], list[str]], ticker: str
) -> None:
    client, calls = fetches
    with pytest.raises(ToolError):
        await client.call_tool("get_company_profile", {"ticker": ticker})
    assert calls == []


@pytest.mark.parametrize("ticker", ["BRK-B", "^GSPC", "RY.TO", "BTC-USD", "EURUSD=X", "005930.KS"])
async def test_yahoo_symbol_shapes_pass_validation(
    fetches: tuple[Client[FastMCPTransport], list[str]], ticker: str
) -> None:
    client, calls = fetches
    result = await client.call_tool("get_company_profile", {"ticker": ticker})
    assert result.data.symbol == ticker
    assert calls == [ticker]


async def test_untidy_ticker_is_still_accepted_and_normalized(
    fetches: tuple[Client[FastMCPTransport], list[str]],
) -> None:
    client, _ = fetches
    result = await client.call_tool("get_company_profile", {"ticker": " aapl "})
    assert result.data.symbol == "AAPL"


@pytest.mark.parametrize(
    "tickers",
    [
        [],
        [f"SYM{i}" for i in range(MAX_QUOTE_TICKERS + 1)],
        ["AAPL", "not a ticker!"],
    ],
)
async def test_bad_quote_batch_is_rejected_without_a_fetch(
    fetches: tuple[Client[FastMCPTransport], list[str]], tickers: list[str]
) -> None:
    client, calls = fetches
    with pytest.raises(ToolError):
        await client.call_tool("get_quote", {"tickers": tickers})
    assert calls == []


async def test_blank_search_query_is_rejected_without_a_fetch(
    fetches: tuple[Client[FastMCPTransport], list[str]],
) -> None:
    client, _ = fetches
    with pytest.raises(ToolError):
        await client.call_tool("search_symbols", {"query": "   "})


@pytest.mark.parametrize("line_items", [[], [f"Item {i}" for i in range(MAX_LINE_ITEMS + 1)]])
async def test_bad_line_items_filter_is_rejected_without_a_fetch(
    fetches: tuple[Client[FastMCPTransport], list[str]], line_items: list[str]
) -> None:
    client, calls = fetches
    with pytest.raises(ToolError):
        await client.call_tool(
            "get_financials", {"ticker": "AAPL", "statement": "income", "line_items": line_items}
        )
    assert calls == []


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("npv", {"rate": 0.1, "cashflows": []}),
        ("npv", {"rate": -1.0, "cashflows": [-100.0, 120.0]}),
        ("xnpv", {"rate": -1.5, "cashflows": [{"date": "2024-01-01", "amount": -100.0}]}),
        ("irr", {"cashflows": [-100.0]}),
        ("irr", {"cashflows": [-100.0] + [1.0] * 5000}),
        ("mirr", {"cashflows": [-100.0, 120.0], "finance_rate": -1.0, "reinvest_rate": 0.1}),
        ("mirr", {"cashflows": [-100.0, 120.0], "finance_rate": 0.1, "reinvest_rate": -2.0}),
        ("xirr", {"cashflows": [{"date": "2024-01-01", "amount": -100.0}]}),
        (
            # A zero-coupon, zero-yield bond: nothing overflows, so only a bound on
            # years_to_maturity stops the coupon loop running for billions of periods.
            "bond_price",
            {"face": 1000.0, "coupon_rate": 0.0, "years_to_maturity": 1e9, "ytm": 0.0},
        ),
        (
            "bond_ytm",
            {"face": 1000.0, "coupon_rate": 0.05, "years_to_maturity": 1e9, "price": 900.0},
        ),
        (
            # The dated tools have no years_to_maturity field to bound, so the coupon loop
            # is bounded by the settlement-to-maturity span instead -- a relationship
            # between two arguments, checked in the calculator.
            "bond_price_dated",
            {
                "settlement": "2024-01-01",
                "maturity": "3024-01-01",
                "coupon_rate": 0.05,
                "ytm": 0.05,
            },
        ),
        (
            "bond_ytm_dated",
            {
                "settlement": "2024-01-01",
                "maturity": "3024-01-01",
                "coupon_rate": 0.05,
                "clean_price": 100.0,
            },
        ),
        (
            # Settlement on or after maturity: also a two-argument relationship.
            "bond_price_dated",
            {
                "settlement": "2030-01-01",
                "maturity": "2024-01-01",
                "coupon_rate": 0.05,
                "ytm": 0.05,
            },
        ),
        (
            # A frequency that does not divide 12 cannot produce a whole-month schedule.
            "bond_price_dated",
            {
                "settlement": "2024-01-01",
                "maturity": "2034-01-01",
                "coupon_rate": 0.05,
                "ytm": 0.05,
                "frequency": 5,
            },
        ),
        (
            # Over MAX_COUPON_FREQUENCY, so the schema rejects it before the calculator.
            "bond_price_dated",
            {
                "settlement": "2024-01-01",
                "maturity": "2034-01-01",
                "coupon_rate": 0.05,
                "ytm": 0.05,
                "frequency": 13,
            },
        ),
        (
            "bond_ytm_dated",
            {
                "settlement": "2024-01-01",
                "maturity": "2034-01-01",
                "coupon_rate": 0.05,
                "clean_price": 0.0,
            },
        ),
        ("loan_schedule", {"principal": 1000.0, "annual_rate": 0.05, "term_months": 10_000_000}),
        (
            "convert_rate",
            {"rate": 0.1, "periods_per_year": 10**9, "direction": "nominal_to_effective"},
        ),
        ("time_value_of_money", {"solve_for": "fv", "pv": -100.0, "rate": -1.0, "nper": 5.0}),
    ],
)
async def test_out_of_range_calculator_arguments_are_rejected(
    client: Client[FastMCPTransport], tool: str, args: dict[str, Any]
) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(tool, args)


async def test_a_long_but_allowed_cashflow_series_still_computes(
    client: Client[FastMCPTransport],
) -> None:
    cashflows = [-1000.0] + [12.0] * 199
    result = await client.call_tool("npv", {"rate": 0.01, "cashflows": cashflows})
    # -1000 + 12 * (1 - 1.01**-199) / 0.01
    assert result.data.npv == pytest.approx(34.3361, rel=1e-4)


async def test_rate_bounds_are_visible_in_the_tool_schema(
    client: Client[FastMCPTransport],
) -> None:
    # A model should see 'rate must exceed -1' in the schema, not discover it by erroring.
    by_name = {tool.name: tool for tool in await client.list_tools()}
    for tool, field in [
        ("npv", "rate"),
        ("xnpv", "rate"),
        ("mirr", "finance_rate"),
        ("mirr", "reinvest_rate"),
        ("time_value_of_money", "rate"),
    ]:
        schema = by_name[tool].inputSchema["properties"][field]
        # time_value_of_money's rate is optional, so its constraint sits in the union branch.
        branches = schema.get("anyOf", [schema])
        assert any(branch.get("exclusiveMinimum") == -1 for branch in branches), (tool, field)


async def test_cashflow_list_bounds_are_visible_in_the_tool_schema(
    client: Client[FastMCPTransport],
) -> None:
    by_name = {tool.name: tool for tool in await client.list_tools()}
    for tool, min_items in [("npv", 1), ("irr", 2), ("mirr", 2), ("xnpv", 1), ("xirr", 2)]:
        schema = by_name[tool].inputSchema["properties"]["cashflows"]
        assert schema["minItems"] == min_items, tool
        assert schema["maxItems"] == MAX_CASHFLOWS, tool


async def test_ticker_pattern_is_visible_in_the_tool_schema(
    client: Client[FastMCPTransport],
) -> None:
    by_name = {tool.name: tool for tool in await client.list_tools()}
    assert by_name["get_company_profile"].inputSchema["properties"]["ticker"]["pattern"] == (
        TICKER_PATTERN
    )


@pytest.mark.parametrize("rate", [-0.51, 1.01, 5.0])
async def test_out_of_range_risk_free_rate_is_rejected_without_a_fetch(
    fetches: tuple[Client[FastMCPTransport], list[str]], rate: float
) -> None:
    client, calls = fetches
    with pytest.raises(ToolError):
        await client.call_tool("analyze_performance", {"ticker": "AAPL", "risk_free_rate": rate})
    assert calls == []


@pytest.mark.parametrize("rate", [-0.5, 0.0, 0.0425, 1.0])
async def test_plausible_risk_free_rates_pass_validation(
    fetches: tuple[Client[FastMCPTransport], list[str]], rate: float
) -> None:
    client, _ = fetches
    # The stub has no history, so the fetch itself fails - the point is that validation let
    # the call through rather than rejecting the rate.
    with pytest.raises(ToolError, match="price history"):
        await client.call_tool("analyze_performance", {"ticker": "AAPL", "risk_free_rate": rate})


@pytest.mark.parametrize(
    "tickers",
    [
        ["AAPL"],  # one ticker is analyze_performance's job
        [f"SYM{i}" for i in range(MAX_COMPARE_TICKERS + 1)],  # over the bound
        ["AAPL", "A B"],  # a malformed member
    ],
)
async def test_bad_compare_tickers_list_is_rejected_without_a_fetch(
    fetches: tuple[Client[FastMCPTransport], list[str]], tickers: list[str]
) -> None:
    client, calls = fetches
    with pytest.raises(ToolError):
        await client.call_tool("compare_tickers", {"tickers": tickers})
    assert calls == []
