"""Guards prompt/tool drift: every tool, parameter and result field a prompt names must
exist on the server.

Prompts are prose, so nothing type-checks them against the tools they orchestrate. A tool
renamed, a parameter dropped, or a result field reshaped would leave a prompt steering
the model toward a call that fails or a field that is never there. This checks every
rendered prompt against the live registry, from two angles:

- call sites: each ``identifier(`` must be a registered tool;
- vocabulary: each snake_case token (one containing ``_``) must be a tool name, a tool
  input parameter, a property somewhere in a tool's output schema, or a string enum
  literal from a schema (e.g. ``nominal_to_effective``).

The vocabulary check is what catches field and parameter drift. Prose therefore avoids
``f(x)`` notation and made-up snake_case words - use a real field name or plain English.
"""

import re
from collections.abc import Iterator
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from mcp.types import TextContent, Tool

#: Arguments to render each prompt with. Must name every registered prompt, so a new
#: prompt cannot bypass the guard by never being rendered.
SAMPLE_ARGS: dict[str, dict[str, str]] = {
    "analyze_stock": {"ticker": "AAPL", "horizon": "3y"},
    "investment_cashflows": {
        "cashflows": "-1000, 300, 300, 400",
        "discount_rate": "8%",
        "reinvest_rate": "6%",
    },
    "bond_analysis": {"bond": "UST 4% due 2036-01-15, clean 92.30", "shock_bp": "50"},
}

_CALL_SITE = re.compile(r"\b([a-z][a-z0-9_]*)\(")
_SNAKE_CASE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def _schema_words(node: Any) -> Iterator[str]:
    """Every property name and string enum/const literal anywhere in a JSON schema."""
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            yield from properties
        for literal in node.get("enum", []):
            if isinstance(literal, str):
                yield literal
        if isinstance(node.get("const"), str):
            yield node["const"]
        for value in node.values():
            yield from _schema_words(value)
    elif isinstance(node, list):
        for item in node:
            yield from _schema_words(item)


def vocabulary(tools: list[Tool]) -> set[str]:
    """The identifiers a prompt may legitimately name, derived from the registry."""
    words = {tool.name for tool in tools}
    for tool in tools:
        words.update(_schema_words(tool.inputSchema))
        words.update(_schema_words(tool.outputSchema or {}))
    return words


def unknown_call_sites(text: str, tool_names: set[str]) -> set[str]:
    """``name(`` occurrences in ``text`` whose name is not a registered tool."""
    return set(_CALL_SITE.findall(text)) - tool_names


def unknown_identifiers(text: str, words: set[str]) -> set[str]:
    """snake_case tokens in ``text`` that are not a tool, parameter, field or literal."""
    return set(_SNAKE_CASE.findall(text)) - words


async def _rendered_prompts(client: Client[FastMCPTransport]) -> dict[str, str]:
    rendered = {}
    for prompt in await client.list_prompts():
        result = await client.get_prompt(prompt.name, SAMPLE_ARGS[prompt.name])
        parts = [m.content.text for m in result.messages if isinstance(m.content, TextContent)]
        rendered[prompt.name] = "\n".join(parts)
    return rendered


# --- the guard's own behaviour: it must actually fire -------------------------------


def test_call_site_check_flags_a_tool_that_does_not_exist() -> None:
    text = "Call get_quote(tickers=[...]) and get_price_targets(ticker='AAPL')."
    assert unknown_call_sites(text, {"get_quote"}) == {"get_price_targets"}


def test_vocabulary_check_flags_a_field_that_does_not_exist() -> None:
    text = "Read modified_duration and yield_to_worst from the result."
    assert unknown_identifiers(text, {"modified_duration"}) == {"yield_to_worst"}


def test_vocabulary_ignores_plain_words_and_uris() -> None:
    text = "Read finance://conventions first; a 50/200-day SMA is context, not a signal."
    assert unknown_identifiers(text, set()) == set()


def test_schema_words_walk_properties_enums_and_nested_defs() -> None:
    schema = {
        "properties": {"direction": {"enum": ["nominal_to_effective", 3]}},
        "$defs": {"Row": {"properties": {"accrued_interest": {}}, "const": "fixed_value"}},
    }
    assert set(_schema_words(schema)) == {
        "direction",
        "nominal_to_effective",
        "accrued_interest",
        "fixed_value",
    }


# --- the guard over every registered prompt -----------------------------------------


async def test_every_registered_prompt_has_sample_arguments(
    client: Client[FastMCPTransport],
) -> None:
    assert {p.name for p in await client.list_prompts()} == set(SAMPLE_ARGS)


async def test_prompts_only_call_registered_tools(client: Client[FastMCPTransport]) -> None:
    tool_names = {tool.name for tool in await client.list_tools()}
    for name, text in (await _rendered_prompts(client)).items():
        unknown = unknown_call_sites(text, tool_names)
        assert not unknown, f"{name} calls unregistered tools: {sorted(unknown)}"


async def test_prompts_only_name_registered_tools_parameters_and_fields(
    client: Client[FastMCPTransport],
) -> None:
    words = vocabulary(await client.list_tools())
    for name, text in (await _rendered_prompts(client)).items():
        unknown = unknown_identifiers(text, words)
        assert not unknown, f"{name} names identifiers no tool defines: {sorted(unknown)}"


@pytest.mark.parametrize("prompt", sorted(SAMPLE_ARGS))
async def test_every_prompt_orchestrates_at_least_one_tool(
    client: Client[FastMCPTransport], prompt: str
) -> None:
    tool_names = {tool.name for tool in await client.list_tools()}
    text = (await _rendered_prompts(client))[prompt]
    assert set(_CALL_SITE.findall(text)) & tool_names, prompt
