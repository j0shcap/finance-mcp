"""DataService.search_symbols."""

from typing import Any

import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.models import (
    SymbolSearchResult,
)
from tests.fakes import (
    FakeClock,
    FakeSearch,
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
        search_factory=FakeSearch(quotes=SEARCH_QUOTES),
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
        search_factory=FakeSearch(quotes=[]),
    )
    result = client.search_symbols("zzzznope")
    assert result.query == "zzzznope" and result.matches == []


@pytest.mark.parametrize("error", [YFException("rate limited"), RuntimeError("boom")])
def test_search_symbols_failure_is_data_unavailable(error: Exception) -> None:
    client = make_client(search_factory=FakeSearch(error=error))
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert type(exc.value) is DataUnavailable
    assert f"Search failed for 'apple': {error}" == str(exc.value)


def test_search_symbols_passes_max_results() -> None:
    search = FakeSearch(quotes=SEARCH_QUOTES)
    make_client(search_factory=search).search_symbols("apple", max_results=3)
    assert search.calls == [{"query": "apple", "max_results": 3, "news_count": 0, "lists_count": 0}]


def test_search_symbols_skips_quote_without_symbol() -> None:
    quotes: list[dict[str, Any]] = [
        {"shortname": "No Symbol Co"},
        {"symbol": "AAPL", "longname": "Apple Inc."},
    ]
    client = make_client(
        search_factory=FakeSearch(quotes=quotes),
    )
    result = client.search_symbols("apple")
    assert [m.symbol for m in result.matches] == ["AAPL"]


def test_search_symbols_caches_within_ttl() -> None:
    search = FakeSearch(quotes=SEARCH_QUOTES)
    clock = FakeClock()
    client = make_client(search_factory=search, clock=clock, fundamentals_ttl=3600.0)
    client.search_symbols("apple")
    client.search_symbols("apple")
    assert len(search.calls) == 1
    clock.advance(3601.0)
    client.search_symbols("apple")
    assert len(search.calls) == 2


def test_search_symbols_parse_error_is_data_unavailable() -> None:
    quotes: list[dict[str, Any]] = [{"symbol": "AAPL", "score": object()}]
    client = make_client(
        search_factory=FakeSearch(quotes=quotes),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "Failed to parse search results for 'apple'" in str(exc.value)
