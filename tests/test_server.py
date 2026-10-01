"""Tests for server assembly and settings."""

from pathlib import Path

import pytest
from fastmcp import FastMCP
from pydantic import ValidationError

from finance_mcp.server import build_default_client, create_server, main
from finance_mcp.settings import get_settings


def test_build_default_client_uses_settings_defaults() -> None:
    client = build_default_client()
    assert client._quote_ttl == 30.0
    assert client._history_ttl == 300.0
    assert client._fundamentals_ttl == 3600.0
    assert client._max_bars == 260


@pytest.mark.parametrize(
    ("variable", "value", "attribute", "expected"),
    [
        ("FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS", "45", "_quote_ttl", 45.0),
        ("FINANCE_MCP_HISTORY_CACHE_TTL_SECONDS", "120", "_history_ttl", 120.0),
        ("FINANCE_MCP_FUNDAMENTALS_CACHE_TTL_SECONDS", "7200", "_fundamentals_ttl", 7200.0),
        ("FINANCE_MCP_MAX_HISTORY_BARS", "40", "_max_bars", 40),
    ],
)
def test_build_default_client_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, attribute: str, expected: float
) -> None:
    monkeypatch.setenv(variable, value)
    assert getattr(build_default_client(), attribute) == expected


def test_settings_ignore_a_dotenv_file_in_the_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An MCP client launches the server in whatever CWD it chooses, so a stray .env there
    # must not silently reconfigure the server.
    (tmp_path / ".env").write_text("FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS=999\n")
    monkeypatch.chdir(tmp_path)
    assert get_settings().quote_cache_ttl_seconds == 30


@pytest.mark.parametrize(
    ("variable", "value"),
    [("FINANCE_MCP_MAX_HISTORY_BARS", "0"), ("FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS", "-1")],
)
def test_settings_reject_a_nonsensical_value(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError):
        get_settings()


def test_create_server_returns_fastmcp() -> None:
    assert isinstance(create_server(), FastMCP)


def test_main_runs_without_the_fastmcp_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    # The banner would be written into the client's log on every launch, with a PyPI update
    # check whose advice ("pip install --upgrade fastmcp") steps outside pyproject's fastmcp<4.
    kwargs_seen: list[dict[str, object]] = []

    def fake_run(self: FastMCP, *args: object, **kwargs: object) -> None:
        kwargs_seen.append(kwargs)

    monkeypatch.setattr(FastMCP, "run", fake_run)
    main()
    assert kwargs_seen == [{"show_banner": False}]
