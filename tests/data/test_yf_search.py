"""DataService.search_symbols."""

import json
from typing import Any

import pytest
from yfinance.exceptions import (
    YFDataException,
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


def test_search_symbols_empty_quotes_returns_empty_no_raise(evictions: list[None]) -> None:
    search = FakeSearch(quotes=[])
    client = make_client(search_factory=search)
    result = client.search_symbols("zzzznope")
    assert result.query == "zzzznope" and result.matches == []
    # A real no-match isn't retried, but the source must not keep serving it.
    assert len(search.calls) == 1 and len(evictions) == 1


#: Replies that carry no quotes list: an empty body, Yahoo's error body, and an error object
#: where the list belongs.
ERROR_REPLIES = [
    {},
    {"finance": {"result": None, "error": {"code": "Internal Server Error"}}},
    {"quotes": {"error": "Internal Server Error"}},
]


@pytest.mark.parametrize("reply", ERROR_REPLIES)
def test_search_symbols_retries_a_reply_without_quotes(
    reply: dict[str, Any], evictions: list[None]
) -> None:
    search = FakeSearch(quotes=SEARCH_QUOTES, responses=[reply])
    client = make_client(search_factory=search)
    result = client.search_symbols("apple")
    assert [m.symbol for m in result.matches] == ["AAPL", "APLE"]
    # Let go of the bad reply first, or the retry would be answered with it again.
    assert len(search.calls) == 2 and len(evictions) == 1


@pytest.mark.parametrize("reply", ERROR_REPLIES)
def test_search_symbols_reply_without_quotes_every_time_is_data_unavailable(
    reply: dict[str, Any], evictions: list[None]
) -> None:
    search = FakeSearch(responses=[reply] * 3)
    client = make_client(search_factory=search, request_retries=2)
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert type(exc.value) is DataUnavailable
    assert str(exc.value).startswith("Search failed for 'apple': ")
    assert len(search.calls) == 3 and len(evictions) == 3


@pytest.mark.parametrize("error", [YFException("rate limited"), RuntimeError("boom")])
def test_search_symbols_failure_is_data_unavailable(error: Exception) -> None:
    client = make_client(search_factory=FakeSearch(error=error))
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert type(exc.value) is DataUnavailable
    assert f"Search failed for 'apple': {error}" == str(exc.value)


@pytest.mark.parametrize(
    "error",
    [
        YFDataException("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"),
        json.JSONDecodeError("Expecting value", "<html>502 Bad Gateway</html>", 0),
    ],
)
def test_search_symbols_error_page_is_data_unavailable_and_evicted(
    error: Exception, evictions: list[None]
) -> None:
    # yfinance keeps the page it raises on, so every later search would hit it again.
    client = make_client(search_factory=FakeSearch(error=error))
    with pytest.raises(DataUnavailable, match="Search failed for 'apple'"):
        client.search_symbols("apple")
    assert len(evictions) == 1


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


def test_search_symbols_does_not_cache_an_empty_result(evictions: list[None]) -> None:
    search = FakeSearch(quotes=[])
    client = make_client(search_factory=search, fundamentals_ttl=3600.0)
    client.search_symbols("apple")
    client.search_symbols("apple")
    # An empty result may be a source failure that looks like no matches; ask again.
    assert len(search.calls) == 2


def test_search_symbols_parse_error_is_data_unavailable() -> None:
    quotes: list[dict[str, Any]] = [{"symbol": "AAPL", "score": object()}]
    client = make_client(
        search_factory=FakeSearch(quotes=quotes),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "Failed to parse search results for 'apple'" in str(exc.value)
