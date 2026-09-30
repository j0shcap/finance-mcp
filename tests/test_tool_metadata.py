"""Protocol-level tests for the metadata a client sees: annotations, titles,
server instructions/version, the conventions resource, and error masking."""

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError
from mcp.types import TextResourceContents, Tool, ToolAnnotations

from finance_mcp import __version__, conventions
from finance_mcp.server import create_server


def annotations_of(tool: Tool) -> ToolAnnotations:
    """The tool's annotations, asserting they exist (every tool here is annotated)."""
    assert tool.annotations is not None, tool.name
    return tool.annotations


CALCULATOR_TOOLS = {
    "time_value_of_money",
    "bond_price",
    "bond_ytm",
    "bond_price_dated",
    "bond_ytm_dated",
    "loan_schedule",
    "npv",
    "irr",
    "mirr",
    "xnpv",
    "xirr",
    "convert_rate",
}
MARKET_DATA_TOOLS = {
    "get_quote",
    "get_price_history",
    "get_financials",
    "get_company_profile",
    "get_analyst_data",
    "get_news",
    "search_symbols",
    "get_key_metrics",
    "analyze_performance",
    "compare_to_benchmark",
    "compare_tickers",
}


async def test_every_tool_is_annotated_read_only(client: Client[FastMCPTransport]) -> None:
    tools = await client.list_tools()
    assert {t.name for t in tools} == CALCULATOR_TOOLS | MARKET_DATA_TOOLS
    for tool in tools:
        assert annotations_of(tool).readOnlyHint is True, tool.name


async def test_every_tool_has_a_human_readable_title(client: Client[FastMCPTransport]) -> None:
    for tool in await client.list_tools():
        title = annotations_of(tool).title
        assert title, tool.name
        assert title != tool.name, tool.name  # a real title, not the snake_case name


async def test_calculators_are_idempotent_and_closed_world(
    client: Client[FastMCPTransport],
) -> None:
    by_name = {t.name: t for t in await client.list_tools()}
    for name in CALCULATOR_TOOLS:
        annotations = annotations_of(by_name[name])
        assert annotations.idempotentHint is True, name
        assert annotations.openWorldHint is False, name


async def test_market_data_tools_are_open_world_and_not_idempotent(
    client: Client[FastMCPTransport],
) -> None:
    by_name = {t.name: t for t in await client.list_tools()}
    for name in MARKET_DATA_TOOLS:
        annotations = annotations_of(by_name[name])
        assert annotations.openWorldHint is True, name
        assert annotations.idempotentHint is not True, name  # live prices move


async def test_no_tool_claims_to_be_destructive(client: Client[FastMCPTransport]) -> None:
    for tool in await client.list_tools():
        assert annotations_of(tool).destructiveHint is not True, tool.name


async def test_server_advertises_instructions_over_the_protocol(
    client: Client[FastMCPTransport],
) -> None:
    initialized = client.initialize_result
    assert initialized is not None
    instructions = initialized.instructions
    assert instructions
    assert "finance://conventions" in instructions
    # The sign convention and the Yahoo unit quirks are the two things a model
    # gets wrong without orientation.
    assert "received is positive" in instructions
    assert "debt_to_equity" in instructions


async def test_server_reports_its_package_version(client: Client[FastMCPTransport]) -> None:
    initialized = client.initialize_result
    assert initialized is not None
    assert initialized.serverInfo.version == __version__


async def test_conventions_resource_is_listed(client: Client[FastMCPTransport]) -> None:
    uris = {str(resource.uri) for resource in await client.list_resources()}
    assert "finance://conventions" in uris


async def test_conventions_resource_serves_the_units_glossary(
    client: Client[FastMCPTransport],
) -> None:
    contents = await client.read_resource("finance://conventions")
    assert isinstance(contents[0], TextResourceContents)
    text = contents[0].text
    assert "debt_to_equity" in text
    assert "recommendation_mean" in text
    assert "financial_currency" in text


async def test_tool_error_messages_still_reach_the_client(
    client: Client[FastMCPTransport],
) -> None:
    with pytest.raises(ToolError, match="cashflows"):
        await client.call_tool("npv", {"rate": 0.1, "cashflows": []})


async def test_unexpected_exceptions_do_not_leak_internals() -> None:
    server = create_server()

    @server.tool
    def explode() -> str:
        raise RuntimeError("connection string postgres://user:hunter2@internal")

    async with Client(server) as connected:
        with pytest.raises(ToolError) as excinfo:
            await connected.call_tool("explode", {})
    assert "hunter2" not in str(excinfo.value)


async def test_instructions_name_exactly_the_registered_tools(
    client: Client[FastMCPTransport],
) -> None:
    """The instructions map the server for the model by naming both tool families. A tool
    added, removed, or renamed without updating those lists would leave a stale map."""
    initialized = client.initialize_result
    assert initialized is not None
    instructions = initialized.instructions or ""
    families = set(conventions.MARKET_DATA_TOOLS) | set(conventions.CALCULATOR_TOOLS)
    assert families == {tool.name for tool in await client.list_tools()}
    for name in families:
        assert name in instructions, name


async def test_conventions_resource_explains_the_risk_adjusted_conventions(
    client: Client[FastMCPTransport],
) -> None:
    contents = await client.read_resource("finance://conventions")
    assert isinstance(contents[0], TextResourceContents)
    text = contents[0].text
    assert "risk_free_rate" in text
    assert "sharpe_ratio" in text
    # The two figures a model is most likely to mis-scale: a decimal rate and a
    # percentage-point difference.
    assert "DECIMAL" in text
    assert "percentage POINTS" in text


async def test_conventions_resource_explains_the_benchmark_alignment(
    client: Client[FastMCPTransport],
) -> None:
    contents = await client.read_resource("finance://conventions")
    assert isinstance(contents[0], TextResourceContents)
    text = contents[0].text
    assert "overlapping_observations" in text
    assert "compare_to_benchmark" in text
    assert "compare_tickers" in text
