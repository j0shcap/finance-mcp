"""Guards README drift: the tools, prompts, resources and settings the README documents
must be exactly the ones the server registers.

The README is the first thing a user reads, and nothing else ties it to the code: a tool
added without a README line, or removed while its line stays, would go unnoticed. Each
check compares sets both ways, so a missing and a stale entry both fail.

Convention the parser relies on: within a section, each entry is a bullet that starts with
the backticked name(s), e.g. ``- `npv` — ...`` or ``- `xnpv` / `xirr` — ...``; settings
are rows of the Configuration table, ``| `FINANCE_MCP_...` | `default` | ... |``.
"""

import re
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from finance_mcp.settings import Settings

README = Path(__file__).resolve().parents[1] / "README.md"

_LEADING_NAMES = re.compile(r"^- ((?:`[^`]+`(?:\s*/\s*)?)+)", re.MULTILINE)
_BACKTICKED = re.compile(r"`([^`]+)`")
_CONFIG_ROW = re.compile(r"^\| `(FINANCE_MCP_[A-Z_]+)` \| `([^`]*)` \|", re.MULTILINE)


def section(markdown: str, title: str) -> str:
    """The body of the ``## title`` section, up to the next ``## `` heading."""
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", markdown, re.M | re.S)
    assert match, f"README has no '## {title}' section"
    return match.group(1)


def documented_names(markdown: str, title: str) -> set[str]:
    """The backticked names leading each bullet of a section."""
    body = section(markdown, title)
    return {name for lead in _LEADING_NAMES.findall(body) for name in _BACKTICKED.findall(lead)}


def drift_message(kind: str, documented: set[str], registered: set[str]) -> str:
    return (
        f"README {kind} drifted from the server. "
        f"Add a bullet for: {sorted(registered - documented)}; "
        f"remove the bullet for: {sorted(documented - registered)}"
    )


def test_documented_names_reads_every_name_leading_a_bullet() -> None:
    markdown = "## Tools\n\n- `npv` — a\n- `xnpv` / `xirr` — b `not_this`\n\n## Next\n- `other`\n"
    assert documented_names(markdown, "Tools") == {"npv", "xnpv", "xirr"}


def test_drift_message_names_missing_and_stale_entries() -> None:
    message = drift_message("tools", {"old_tool", "npv"}, {"npv", "new_tool"})
    assert "['new_tool']" in message
    assert "['old_tool']" in message


async def test_readme_tools_match_registry(client: Client[FastMCPTransport]) -> None:
    documented = documented_names(README.read_text(), "Tools")
    registered = {tool.name for tool in await client.list_tools()}
    assert documented == registered, drift_message("tools", documented, registered)


async def test_readme_prompts_match_registry(client: Client[FastMCPTransport]) -> None:
    documented = documented_names(README.read_text(), "Prompts")
    registered = {prompt.name for prompt in await client.list_prompts()}
    assert documented == registered, drift_message("prompts", documented, registered)


async def test_readme_resources_match_registry(client: Client[FastMCPTransport]) -> None:
    documented = documented_names(README.read_text(), "Resources")
    registered = {str(resource.uri) for resource in await client.list_resources()}
    assert documented == registered, drift_message("resources", documented, registered)


def test_readme_config_table_matches_settings() -> None:
    rows = dict(_CONFIG_ROW.findall(section(README.read_text(), "Configuration")))
    prefix = Settings.model_config["env_prefix"]
    expected = {
        f"{prefix}{name}".upper(): str(field.default)
        for name, field in Settings.model_fields.items()
    }
    assert rows == expected
