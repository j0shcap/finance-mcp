"""The installed server's MCP surface, over stdio, for every way a user launches it.

Each test runs once per launcher (console script, `python -m`, `uvx --from <wheel>`): the
handshake, the advertised tools, prompts and resources, and a stderr free of tracebacks.
The expected values come from the in-repo registry, so this pins that the *artifact*
matches the source rather than restating either.
"""

import re

from jsonschema import Draft202012Validator
from mcp.types import TextResourceContents

from finance_mcp.conventions import (
    CALCULATOR_TOOLS,
    CONVENTIONS_DOC,
    CONVENTIONS_URI,
    MARKET_DATA_TOOLS,
    SERVER_INSTRUCTIONS,
)
from tests.e2e.conftest import PROJECT_VERSION, Server
from tests.test_prompt_tool_drift import SAMPLE_ARGS

#: A `{placeholder}` the prompt template failed to fill.
_UNFILLED = re.compile(r"\{[a-z_]+\}")


async def test_initialize_reports_name_version_and_instructions(server: Server) -> None:
    init = server.client.initialize_result
    assert init is not None

    assert init.serverInfo.name == "finance-mcp"
    # Read from the installed distribution's metadata; a mismatch means a stale wheel.
    assert init.serverInfo.version == PROJECT_VERSION
    assert init.instructions == SERVER_INSTRUCTIONS
    assert CONVENTIONS_URI in init.instructions


async def test_lists_every_tool_with_annotations_and_valid_schemas(server: Server) -> None:
    tools = {tool.name: tool for tool in await server.client.list_tools()}

    assert set(tools) == {*MARKET_DATA_TOOLS, *CALCULATOR_TOOLS}
    assert len(tools) == len(MARKET_DATA_TOOLS) + len(CALCULATOR_TOOLS)
    for name, tool in tools.items():
        hints = tool.annotations
        assert hints is not None and hints.title, f"{name} has no annotations/title"
        assert hints.readOnlyHint is True and hints.destructiveHint is False, name
        is_calculator = name in CALCULATOR_TOOLS
        assert hints.idempotentHint is is_calculator, name
        assert hints.openWorldHint is not is_calculator, name
        assert tool.description, f"{name} has no description"
        Draft202012Validator.check_schema(tool.inputSchema)
        assert tool.outputSchema is not None, f"{name} advertises no output schema"
        Draft202012Validator.check_schema(tool.outputSchema)


async def test_every_prompt_renders(server: Server) -> None:
    prompts = await server.client.list_prompts()

    assert {p.name for p in prompts} == set(SAMPLE_ARGS)
    for name, args in SAMPLE_ARGS.items():
        result = await server.client.get_prompt(name, args)
        (message,) = result.messages
        text = getattr(message.content, "text", "")
        assert CONVENTIONS_URI in text, name
        assert not _UNFILLED.search(text), f"{name} left a placeholder unfilled"


async def test_conventions_resource_is_listed_and_readable(server: Server) -> None:
    (resource,) = await server.client.list_resources()
    assert str(resource.uri) == CONVENTIONS_URI
    assert resource.mimeType == "text/markdown"

    (contents,) = await server.client.read_resource(CONVENTIONS_URI)
    assert isinstance(contents, TextResourceContents)
    assert contents.text == CONVENTIONS_DOC


async def test_starts_cleanly(server: Server) -> None:
    """Nothing but JSON-RPC on stdout and no crash on stderr.

    Deliberately not "stderr is empty": fastmcp may log a startup line there, which is
    exactly where a stdio server's logs belong.
    """
    assert await server.client.ping()
    assert not server.protocol.errors, f"non-JSON-RPC output on stdout: {server.protocol.errors}"
    stderr = server.stderr()
    assert "Traceback" not in stderr, stderr
    assert "ERROR" not in stderr, stderr
