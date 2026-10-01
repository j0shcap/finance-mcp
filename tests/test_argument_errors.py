"""Argument-validation failures reach the model as one clean, self-correctable message."""

from typing import Annotated, Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent
from pydantic import BaseModel, Field

from finance_mcp.server import create_server
from finance_mcp.tools._argument_errors import ArgumentErrorMiddleware


def _text(result: CallToolResult) -> str:
    assert result.is_error
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


async def _error_text(name: str, arguments: dict[str, Any]) -> str:
    async with Client(create_server()) as client:
        return _text(await client.call_tool(name, arguments, raise_on_error=False))


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        (
            "get_quote",
            {"tickers": []},
            "Invalid arguments for get_quote: tickers: List should have at least 1 item "
            "after validation, not 0.",
        ),
        (
            "npv",
            {"cashflows": [-100.0, 110.0]},
            "Invalid arguments for npv: rate: Missing required argument.",
        ),
        (
            "get_quote",
            {"tickers": ["AAPL"], "foo": 1},
            "Invalid arguments for get_quote: foo: Unexpected keyword argument.",
        ),
        (
            "get_price_history",
            {"ticker": "AAPL", "period": "2w"},
            "Invalid arguments for get_price_history: period: Input should be '1d', '5d', "
            "'1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd' or 'max'.",
        ),
        (
            # Nested locations read as an index path, not pydantic's dotted tuple.
            "xirr",
            {
                "cashflows": [
                    {"date": "2024-13-01", "amount": -1.0},
                    {"date": "2025-01-01", "amount": 2.0},
                ]
            },
            "Invalid arguments for xirr: cashflows[0].date: Input should be a valid date or "
            "datetime, month value is outside expected range of 1-12.",
        ),
        (
            # A pattern failure names the value instead of printing the regex.
            "get_quote",
            {"tickers": ["AA PL"]},
            "Invalid arguments for get_quote: tickers[0]: 'AA PL' is not in the expected "
            "format; see the parameter's description.",
        ),
    ],
)
async def test_argument_errors_are_clean(
    name: str, arguments: dict[str, Any], expected: str
) -> None:
    assert await _error_text(name, arguments) == expected


async def test_every_argument_error_is_listed() -> None:
    text = await _error_text("time_value_of_money", {"solve_for": "pmt", "pv": "abc", "nper": []})
    assert text == (
        "Invalid arguments for time_value_of_money: pv: Input should be a valid number, "
        "unable to parse string as a number; nper: Input should be a valid number."
    )


async def test_a_long_offending_value_is_truncated() -> None:
    # 22 characters: inside the 24-character length cap, past the pattern's 20.
    text = await _error_text("get_quote", {"tickers": ["X" * 22]})
    assert "'XXXXXXXXXXXXXXXXXXXX...'" in text
    assert "X" * 21 not in text


async def test_no_raw_pydantic_detail_reaches_the_model() -> None:
    text = await _error_text("compare_tickers", {"tickers": ["A"] * 11})
    for raw in ("validation error for", "[type=", "input_value", "errors.pydantic.dev"):
        assert raw not in text


async def test_every_tool_rejects_an_unknown_argument_cleanly() -> None:
    # Pins that each tool's validator is titled after the tool, which is how the
    # middleware tells an argument error from one raised inside a tool body.
    async with Client(create_server()) as client:
        names = [tool.name for tool in await client.list_tools()]
        for name in names:
            text = _text(await client.call_tool(name, {"not_a_parameter": 1}, raise_on_error=False))
            assert text.startswith(f"Invalid arguments for {name}: "), text
            assert "not_a_parameter: Unexpected keyword argument" in text, text


class _Positive(BaseModel):
    value: Annotated[float, Field(gt=0)]


def _assert_masked_server_error(text: str) -> None:
    assert text == "Error calling tool 'broken'"


async def test_a_validation_error_from_a_tool_body_is_not_reworded() -> None:
    # A result model rejecting a value is a server bug, not a bad call: it must not be
    # reported as the caller's invalid arguments. fastmcp 4 would turn it into a
    # JSON-RPC "Invalid request parameters" error, which call_tool raises as MCPError
    # rather than returning as a tool error.
    mcp = FastMCP("body-error", mask_error_details=True)
    mcp.add_middleware(ArgumentErrorMiddleware())

    @mcp.tool
    def broken() -> _Positive:
        return _Positive(value=-1.0)

    async with Client(mcp) as client:
        result = await client.call_tool("broken", {}, raise_on_error=False)
    _assert_masked_server_error(_text(result))


async def test_a_validation_error_from_a_tool_body_is_masked_by_the_server() -> None:
    server = create_server()

    @server.tool
    def broken() -> _Positive:
        return _Positive(value=-1.0)

    async with Client(server) as client:
        result = await client.call_tool("broken", {}, raise_on_error=False)
    _assert_masked_server_error(_text(result))


async def test_other_tool_errors_pass_through_unchanged() -> None:
    text = await _error_text("irr", {"cashflows": [100.0, 200.0]})
    assert text == "irr needs at least one sign change in the cashflows."
