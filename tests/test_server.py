"""Tests for server assembly and settings."""

from pathlib import Path

import pytest
from fastmcp import FastMCP
from pydantic import ValidationError

from finance_mcp.server import build_default_client, create_server, main
from finance_mcp.settings import get_settings


def test_settings_defaults() -> None:
    settings = get_settings()
    assert settings.quote_cache_ttl_seconds == 30
    assert settings.fundamentals_cache_ttl_seconds == 3600


def test_quote_ttl_default_is_30() -> None:
    settings = get_settings()
    assert settings.quote_cache_ttl_seconds == 30
    assert settings.history_cache_ttl_seconds == 300
    assert settings.fundamentals_cache_ttl_seconds == 3600


def test_build_default_client_uses_settings_defaults() -> None:
    client = build_default_client()
    assert client._quote_ttl == 30.0
    assert client._history_ttl == 300.0


def test_build_default_client_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS", "45")
    client = build_default_client()
    assert client._quote_ttl == 45.0


def test_build_default_client_uses_fundamentals_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_MCP_FUNDAMENTALS_CACHE_TTL_SECONDS", "7200")
    client = build_default_client()
    assert client._fundamentals_ttl == 7200.0


def test_build_default_client_honors_history_ttl_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_MCP_HISTORY_CACHE_TTL_SECONDS", "120")
    assert build_default_client()._history_ttl == 120.0


def test_create_server_returns_fastmcp() -> None:
    assert isinstance(create_server(), FastMCP)


def test_main_invokes_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def fake_run(self: FastMCP, *args: object, **kwargs: object) -> None:
        calls.append(True)

    monkeypatch.setattr(FastMCP, "run", fake_run)
    main()
    assert calls == [True]


def test_max_history_bars_default_is_260() -> None:
    assert get_settings().max_history_bars == 260


def test_build_default_client_honors_max_history_bars_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FINANCE_MCP_MAX_HISTORY_BARS", "40")
    assert build_default_client()._max_bars == 40


def test_settings_ignore_a_dotenv_file_in_the_launch_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An MCP client launches the server in whatever CWD it chooses, so a stray .env
    there must not silently reconfigure the server."""
    (tmp_path / ".env").write_text("FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS=999\n")
    monkeypatch.chdir(tmp_path)
    assert get_settings().quote_cache_ttl_seconds == 30


def test_settings_reject_a_nonsensical_max_history_bars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_MCP_MAX_HISTORY_BARS", "0")
    with pytest.raises(ValidationError):
        get_settings()


def test_settings_reject_a_negative_cache_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_MCP_QUOTE_CACHE_TTL_SECONDS", "-1")
    with pytest.raises(ValidationError):
        get_settings()


def test_main_runs_without_the_fastmcp_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The banner is the one thing the server writes on every launch, into the client's log,
    and it comes with a PyPI update check whose advice ("pip install --upgrade fastmcp")
    would step outside the fastmcp<4 range pyproject pins."""
    kwargs_seen: list[dict[str, object]] = []

    def fake_run(self: FastMCP, *args: object, **kwargs: object) -> None:
        kwargs_seen.append(kwargs)

    monkeypatch.setattr(FastMCP, "run", fake_run)
    main()
    assert kwargs_seen == [{"show_banner": False}]
