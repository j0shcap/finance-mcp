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


async def test_every_tool_is_annotated_read_only(client: Client[FastMCPTransport]) -> None:
    for tool in await client.list_tools():
        assert annotations_of(tool).read_only_hint is True, tool.name


async def test_every_tool_has_a_human_readable_title(client: Client[FastMCPTransport]) -> None:
    for tool in await client.list_tools():
        title = annotations_of(tool).title
        assert title, tool.name
        assert title != tool.name, tool.name  # a real title, not the snake_case name


async def test_calculators_are_idempotent_and_closed_world(
    client: Client[FastMCPTransport],
) -> None:
    by_name = {t.name: t for t in await client.list_tools()}
    for name in conventions.CALCULATOR_TOOLS:
        annotations = annotations_of(by_name[name])
        assert annotations.idempotent_hint is True, name
        assert annotations.open_world_hint is False, name


async def test_market_data_tools_are_open_world_and_not_idempotent(
    client: Client[FastMCPTransport],
) -> None:
    by_name = {t.name: t for t in await client.list_tools()}
    for name in conventions.MARKET_DATA_TOOLS:
        annotations = annotations_of(by_name[name])
        assert annotations.open_world_hint is True, name
        assert annotations.idempotent_hint is not True, name  # live prices move


async def test_no_tool_claims_to_be_destructive(client: Client[FastMCPTransport]) -> None:
    for tool in await client.list_tools():
        assert annotations_of(tool).destructive_hint is not True, tool.name


async def test_server_advertises_instructions_over_the_protocol(
    client: Client[FastMCPTransport],
) -> None:
    instructions = client.instructions
    assert instructions
    assert "finance://conventions" in instructions
    # The sign convention and the Yahoo unit quirks are the two things a model
    # gets wrong without orientation.
    assert "received is positive" in instructions
    assert "debt_to_equity" in instructions


async def test_server_reports_its_package_version(client: Client[FastMCPTransport]) -> None:
    assert client.server_info is not None
    assert client.server_info.version == __version__


async def test_conventions_resource_is_listed(client: Client[FastMCPTransport]) -> None:
    uris = {str(resource.uri) for resource in await client.list_resources()}
    assert "finance://conventions" in uris


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
    instructions = client.instructions or ""
    families = set(conventions.MARKET_DATA_TOOLS) | set(conventions.CALCULATOR_TOOLS)
    assert families == {tool.name for tool in await client.list_tools()}
    for name in families:
        assert name in instructions, name


@pytest.mark.parametrize(
    "phrase",
    [
        # The units glossary.
        "debt_to_equity",
        "recommendation_mean",
        "financial_currency",
        # The risk-adjusted conventions, including the two figures a model most often
        # mis-scales: a decimal rate and a percentage-point difference.
        "risk_free_rate",
        "sharpe_ratio",
        "DECIMAL",
        "percentage POINTS",
        # How the benchmark comparison aligns two calendars.
        "overlapping_observations",
        "compare_to_benchmark",
        "compare_tickers",
    ],
)
async def test_conventions_resource_explains(client: Client[FastMCPTransport], phrase: str) -> None:
    contents = await client.read_resource("finance://conventions")
    assert isinstance(contents[0], TextResourceContents)
    assert phrase in contents[0].text
