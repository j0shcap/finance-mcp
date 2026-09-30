"""A network outage must read as "the source is unavailable", never as "invalid symbol".

The 2026-09-27 review found transport failures reported as unknown tickers - which tells
the model to drop a perfectly good symbol instead of retrying. The mocked suites pin the
classification of an exception, but only the real yfinance can show whether a failure
arrives as one: its price and statement fetches hide transport errors by default and hand
back an empty frame. So this runs the installed server with its network cut (a proxy on a
port nothing listens on) and calls every market-data tool. Offline and deterministic.
"""

from typing import Any

import pytest

from finance_mcp.conventions import MARKET_DATA_TOOLS
from tests.e2e.conftest import Server

#: What each tool is told to look up. AAPL is a real, well-covered symbol, so any "no data"
#: answer here can only be a misread transport failure.
CALLS: dict[str, dict[str, Any]] = {
    "search_symbols": {"query": "Apple"},
    "get_quote": {"tickers": ["AAPL"]},
    "get_price_history": {"ticker": "AAPL"},
    "get_financials": {"ticker": "AAPL", "statement": "income"},
    "get_company_profile": {"ticker": "AAPL"},
    "get_key_metrics": {"ticker": "AAPL"},
    "get_analyst_data": {"ticker": "AAPL"},
    "analyze_performance": {"ticker": "AAPL"},
    "compare_to_benchmark": {"ticker": "AAPL"},
    "compare_tickers": {"tickers": ["AAPL", "MSFT"]},
    "get_news": {"ticker": "AAPL"},
}

#: Phrasings of "this symbol has no data" across the client's no-data messages.
_SYMBOL_BLAME = ("invalid", "delisted", "no price history", "no quote data", "not available")


def test_every_market_data_tool_is_covered() -> None:
    assert set(CALLS) == set(MARKET_DATA_TOOLS)


def _failure_text(result: Any) -> str:
    """The failure a tool reported: its error, or its per-ticker errors if it is partial."""
    if result.is_error:
        return " ".join(getattr(block, "text", "") for block in result.content)
    errors = (result.structured_content or {}).get("errors") or []
    assert errors, f"expected a failure with the network down, got {result.structured_content}"
    return " ".join(e["error"] for e in errors)


@pytest.mark.parametrize("tool", sorted(CALLS))
async def test_network_failure_is_not_reported_as_an_unknown_symbol(
    offline_server: Server, tool: str
) -> None:
    result = await offline_server.client.call_tool(tool, CALLS[tool], raise_on_error=False)
    text = _failure_text(result)

    assert "failed" in text.lower(), f"{tool} should report a failed fetch, got: {text}"
    for phrase in _SYMBOL_BLAME:
        assert phrase not in text.lower(), (
            f"{tool} blamed the symbol ({phrase!r}) for a network outage: {text}"
        )
