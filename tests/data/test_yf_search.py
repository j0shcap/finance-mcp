"""YFinanceClient.search_symbols."""

from types import SimpleNamespace
from typing import Any

import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.models import (
    SymbolSearchResult,
)
from finance_mcp.data.yfinance_client import (
    YFinanceClient,
)
from tests.fakes import (
    FakeClock,
    fake_search_factory,
    fake_ticker_factory,
    make_client,
)

SEARCH_QUOTES: list[dict[str, Any]] = [
    {
        "symbol": "AAPL",
        "longname": "Apple Inc.",
        "shortname": "Apple",
        "quoteType": "EQUITY",
        "exchDisp": "NASDAQ",
        "sectorDisp": "Technology",
        "industryDisp": "Consumer Electronics",
        "score": 12345.6,
    },
    {
        "symbol": "APLE",
        "shortname": "Apple Hospitality REIT",  # only shortname
        "quoteType": "EQUITY",
        "exchDisp": "NYSE",
    },
]


def test_search_symbols_happy_path_maps_fields() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=SEARCH_QUOTES),
    )
    result = client.search_symbols("apple")
    assert isinstance(result, SymbolSearchResult)
    assert result.query == "apple"
    assert [m.symbol for m in result.matches] == ["AAPL", "APLE"]
    first = result.matches[0]
    assert first.name == "Apple Inc." and first.quote_type == "EQUITY"
    assert first.exchange == "NASDAQ" and first.sector == "Technology"
    assert first.industry == "Consumer Electronics" and first.score == 12345.6
    second = result.matches[1]
    assert second.name == "Apple Hospitality REIT"  # longname missing -> shortname
    assert second.sector is None and second.score is None


def test_search_symbols_empty_quotes_returns_empty_no_raise() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=[]),
    )
    result = client.search_symbols("zzzznope")
    assert result.query == "zzzznope" and result.matches == []


def test_search_symbols_typed_error_is_data_unavailable() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(error=YFException("search rate limited")),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "search rate limited" in str(exc.value)


def test_search_symbols_raw_error_is_data_unavailable() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(error=RuntimeError("boom")),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "boom" in str(exc.value)


def test_search_symbols_passes_max_results() -> None:
    captured: dict[str, Any] = {}

    def search(query: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(quotes=SEARCH_QUOTES)

    client = make_client(factory=fake_ticker_factory(), search_factory=search)
    client.search_symbols("apple", max_results=3)
    assert captured["max_results"] == 3
    assert captured["news_count"] == 0 and captured["lists_count"] == 0


def test_search_symbols_skips_quote_without_symbol() -> None:
    quotes: list[dict[str, Any]] = [
        {"shortname": "No Symbol Co"},
        {"symbol": "AAPL", "longname": "Apple Inc."},
    ]
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=quotes),
    )
    result = client.search_symbols("apple")
    assert [m.symbol for m in result.matches] == ["AAPL"]


def test_search_symbols_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting_search(query: str, **kwargs: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(quotes=SEARCH_QUOTES)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=fake_ticker_factory(),
        search_factory=counting_search,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.search_symbols("apple")
    client.search_symbols("apple")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.search_symbols("apple")
    assert calls["n"] == 2


def test_search_symbols_parse_error_is_data_unavailable() -> None:
    # A quote whose score is a non-coercible object survives mapping until SymbolMatch
    # construction; force a parse failure via a bad value type for a typed field.
    quotes: list[dict[str, Any]] = [{"symbol": "AAPL", "score": object()}]
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=quotes),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "Failed to parse search results for 'apple'" in str(exc.value)
