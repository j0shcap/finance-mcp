"""The TTL and LRU cache behind every YFinanceClient method."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from finance_mcp.data.yfinance_client import (
    DEFAULT_CACHE_MAX_ENTRIES,
    YFinanceClient,
)
from tests.fakes import (
    QUOTE_FI,
    FakeClock,
    fake_ticker_factory,
)

# --- bounded LRU cache (item 7) ---


def _counting_quote_factory(calls: list[str]) -> Callable[[str], Any]:
    def factory(symbol: str) -> Any:
        calls.append(symbol)
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    return factory


def test_cache_evicts_least_recently_used_entry_over_max() -> None:
    calls: list[str] = []
    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=_counting_quote_factory(calls),
        time_fn=clock,
        quote_ttl=30.0,
        cache_max_entries=2,
    )
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
    calls: list[str] = []
    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=_counting_quote_factory(calls), time_fn=clock, quote_ttl=30.0
    )
    client.get_quote(["AAA"])
    clock.advance(31.0)
    client.get_quote(["BBB"])
    # AAA is past its own TTL, so it is dropped rather than squatting on the bound.
    assert len(client._cache) == 1
    assert ("quote", "BBB") in client._cache


def test_cache_never_exceeds_max_entries() -> None:
    calls: list[str] = []
    client = YFinanceClient(
        ticker_factory=_counting_quote_factory(calls),
        time_fn=FakeClock(),
        quote_ttl=30.0,
        cache_max_entries=3,
    )
    for i in range(20):
        client.get_quote([f"SYM{i}"])
        assert len(client._cache) <= 3


def test_cache_default_max_entries_is_bounded() -> None:
    client = YFinanceClient(ticker_factory=fake_ticker_factory(fast_info=QUOTE_FI))
    assert client._cache_max_entries == DEFAULT_CACHE_MAX_ENTRIES
    assert DEFAULT_CACHE_MAX_ENTRIES > 0


# --- cache entries are timestamped when the fetch completes ---


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

    client = YFinanceClient(ticker_factory=slow_factory, time_fn=clock, quote_ttl=30.0)
    client.get_quote(["AAPL"])
    client.get_quote(["AAPL"])
    # Timestamping at the start would insert the entry already expired, making the cache
    # a no-op during exactly the slowdown it exists to absorb.
    assert calls == ["AAPL"]


def test_refreshing_a_present_key_makes_it_most_recently_used() -> None:
    client = YFinanceClient(
        ticker_factory=fake_ticker_factory(fast_info=QUOTE_FI),
        time_fn=FakeClock(),
        cache_max_entries=2,
    )

    def racing_fetch() -> str:
        # What concurrent get_quote calls do: another thread inserts this key while this
        # fetch is in flight (so the write below lands on a key already present), and a
        # third key is cached after it.
        client._cache[("k", "A")] = (client._now(), 30.0, "stale")
        client._cache[("k", "B")] = (client._now(), 30.0, "B")
        return "fresh"

    assert client._cached(("k", "A"), 30.0, racing_fetch) == "fresh"
    client._cached(("k", "C"), 30.0, lambda: "C")  # over the bound: evict the LRU entry
    # Refreshing A must make it most recently used; leaving it in the racing thread's
    # older slot would evict the entry that was just written.
    assert client._cache[("k", "A")][2] == "fresh"
    assert ("k", "B") not in client._cache
