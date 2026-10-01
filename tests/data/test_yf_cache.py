"""The TTL and LRU cache behind every YFinanceClient method."""

from types import SimpleNamespace
from typing import Any

import pytest

from finance_mcp.data.cache import TTLCache
from tests.fakes import (
    QUOTE_FI,
    FakeClock,
    counting,
    fake_ticker_factory,
    make_client,
)


def test_cache_evicts_least_recently_used_entry_over_max() -> None:
    factory, calls = counting(fake_ticker_factory(fast_info=QUOTE_FI))
    client = make_client(factory, quote_ttl=30.0, cache_max_entries=2)
    client.get_quote(["AAA"])
    client.get_quote(["BBB"])
    client.get_quote(["AAA"])  # cache hit -> AAA becomes the most recently USED entry
    client.get_quote(["CCC"])  # inserting a third entry evicts BBB, not AAA
    assert len(client._cache) == 2
    calls.clear()
    client.get_quote(["AAA"])
    assert calls == []  # still cached
    client.get_quote(["BBB"])
    assert calls == ["BBB"]  # was evicted, so refetched


def test_cache_purges_expired_entries_on_insert() -> None:
    clock = FakeClock()
    client = make_client(clock=clock, quote_ttl=30.0)
    client.get_quote(["AAA"])
    clock.advance(31.0)
    client.get_quote(["BBB"])
    # AAA is past its own TTL, so it is dropped rather than squatting on the bound.
    assert len(client._cache) == 1
    assert ("quote", "BBB") in client._cache


def test_cache_never_exceeds_max_entries() -> None:
    client = make_client(cache_max_entries=3)
    for i in range(20):
        client.get_quote([f"SYM{i}"])
        assert len(client._cache) <= 3


def test_cache_entry_is_timestamped_after_the_fetch_completes() -> None:
    clock = FakeClock()
    calls: list[str] = []

    def slow_factory(symbol: str) -> Any:
        class _Ticker:
            @property
            def fast_info(self) -> Any:
                calls.append(symbol)
                clock.advance(35.0)  # the fetch itself outlasts the 30s quote TTL
                return SimpleNamespace(**QUOTE_FI)

        return _Ticker()

    client = make_client(slow_factory, clock=clock, quote_ttl=30.0)
    client.get_quote(["AAPL"])
    client.get_quote(["AAPL"])
    # Timestamping at the start would insert the entry already expired, making the cache
    # a no-op during exactly the slowdown it exists to absorb.
    assert calls == ["AAPL"]


def test_refreshing_a_present_key_makes_it_most_recently_used() -> None:
    cache = TTLCache(FakeClock(), max_entries=2)

    def racing_fetch() -> str:
        # What concurrent get_quote calls do: another thread caches this key while this
        # fetch is in flight (so the write below lands on a key already present), and a
        # second key is cached after it.
        cache.get_or_fetch(("k", "A"), 30.0, lambda: "stale")
        cache.get_or_fetch(("k", "B"), 30.0, lambda: "B")
        return "fresh"

    assert cache.get_or_fetch(("k", "A"), 30.0, racing_fetch) == "fresh"
    cache.get_or_fetch(("k", "C"), 30.0, lambda: "C")  # over the bound: evict the LRU entry
    # Refreshing A must make it most recently used; leaving it in the racing thread's
    # older slot would evict the entry that was just written.
    assert cache.get_or_fetch(("k", "A"), 30.0, lambda: "refetched") == "fresh"
    assert ("k", "B") not in cache


def test_ttl_cache_refetches_an_expired_entry() -> None:
    clock = FakeClock()
    cache = TTLCache(clock, max_entries=4)
    assert cache.get_or_fetch(("k",), 30.0, lambda: "first") == "first"
    clock.advance(29.0)
    assert cache.get_or_fetch(("k",), 30.0, lambda: "second") == "first"
    clock.advance(1.0)
    assert cache.get_or_fetch(("k",), 30.0, lambda: "second") == "second"


def test_ttl_cache_evicts_the_least_recently_used_entry() -> None:
    cache = TTLCache(FakeClock(), max_entries=2)
    cache.get_or_fetch(("a",), 30.0, lambda: "a")
    cache.get_or_fetch(("b",), 30.0, lambda: "b")
    cache.get_or_fetch(("a",), 30.0, lambda: "unused")  # a is now the most recently used
    cache.get_or_fetch(("c",), 30.0, lambda: "c")
    assert ("a",) in cache and ("c",) in cache and ("b",) not in cache
    assert len(cache) == 2


def test_a_value_the_predicate_rejects_is_returned_but_not_kept() -> None:
    cache = TTLCache(FakeClock(), max_entries=4)
    assert cache.get_or_fetch(("k",), 30.0, lambda: "big", cacheable=lambda v: v != "big") == "big"
    assert ("k",) not in cache and len(cache) == 0


@pytest.mark.parametrize("fetch", ["get_company_profile", "get_key_metrics", "get_analyst_data"])
def test_fundamentals_are_cached_for_the_fundamentals_ttl(fetch: str) -> None:
    info = {"longName": "Apple Inc.", "recommendationMean": 2.0}
    factory, calls = counting(fake_ticker_factory(info=info))
    clock = FakeClock()
    get = getattr(make_client(factory, clock=clock, fundamentals_ttl=3600.0), fetch)
    get("AAPL")
    get("AAPL")
    assert len(calls) == 1
    clock.advance(3601.0)
    get("AAPL")
    assert len(calls) == 2
