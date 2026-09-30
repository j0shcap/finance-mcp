"""YFinanceClient.get_quote: parsing, caching and partial batches."""

import threading
from types import SimpleNamespace
from typing import Any

import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    QuoteResult,
)
from finance_mcp.data.yfinance_client import (
    QUOTE_MAX_WORKERS,
    YFinanceClient,
)
from tests.fakes import (
    QUOTE_FI,
    FakeClock,
    fake_symbol_ticker_factory,
    fake_ticker_factory,
    make_client,
    make_history_df,
)


def test_get_quote_parses_and_computes_change() -> None:
    [q] = make_client().get_quote(["AAPL"]).quotes
    assert q.symbol == "AAPL"
    assert q.price == 190.0
    assert q.change == pytest.approx(2.0)
    assert q.change_percent == pytest.approx(2.0 / 188.0 * 100.0)
    assert q.currency == "USD"


def test_get_quote_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting_factory(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting_factory, time_fn=clock, quote_ttl=30.0, history_ttl=300.0
    )
    client.get_quote(["AAPL"])
    client.get_quote(["AAPL"])
    assert calls["n"] == 1
    clock.advance(31.0)
    client.get_quote(["AAPL"])
    assert calls["n"] == 2


def test_get_quote_cache_expires_exactly_at_ttl() -> None:
    # The cache hit test is `now - hit[0] < ttl`, so a sample taken exactly `ttl` later
    # is a MISS. Pins the strict inequality (a flip to `<=` would extend staleness).
    calls = {"n": 0}

    def counting_factory(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    clock = FakeClock()
    client = YFinanceClient(ticker_factory=counting_factory, time_fn=clock, quote_ttl=30.0)
    client.get_quote(["AAPL"])
    clock.advance(30.0)  # exactly at the TTL boundary -> not strictly within -> refetch
    client.get_quote(["AAPL"])
    assert calls["n"] == 2


def test_get_quote_missing_price_raises_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(fast_info={"last_price": None}))
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("BADSYM")


def test_get_quote_surfaces_yfinance_error_message() -> None:
    client = make_client(
        factory=fake_ticker_factory(fast_info_error=YFException("yahoo says: rate limited"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client._fetch_quote("AAPL")
    assert "yahoo says: rate limited" in str(exc.value)


def test_get_quote_invalid_symbol_returns_clean_symbol_not_found() -> None:
    client = make_client(
        factory=fake_ticker_factory(fast_info_error=KeyError("exchangeTimezoneName"))
    )
    with pytest.raises(SymbolNotFound) as exc:
        client._fetch_quote("BAD")
    assert "No quote data for 'BAD'" in str(exc.value)
    assert "exchangeTimezoneName" not in str(exc.value)


def test_get_quote_none_price_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(fast_info={"last_price": None}))
    with pytest.raises(SymbolNotFound) as exc:
        client._fetch_quote("BAD")
    assert "No quote data for" in str(exc.value)


def test_get_quote_no_second_network_call_on_failure() -> None:
    calls = {"history": 0}

    class _Ticker:
        @property
        def fast_info(self) -> Any:
            raise KeyError("x")

        def history(self, **_kwargs: Any) -> Any:
            calls["history"] += 1
            return None

    def factory(_symbol: str) -> Any:
        return _Ticker()

    client = make_client(factory=factory)
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("BAD")
    assert calls["history"] == 0


def test_get_quote_nan_price_raises_symbol_not_found() -> None:
    client = make_client(
        factory=fake_ticker_factory(fast_info={"last_price": float("nan"), "previous_close": 188.0})
    )
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("AAPL")


def test_get_quote_nan_previous_close_yields_none_change() -> None:
    fi = {**QUOTE_FI, "previous_close": float("nan")}
    client = make_client(factory=fake_ticker_factory(fast_info=fi))
    [q] = client.get_quote(["AAPL"]).quotes
    assert q.price == 190.0
    assert q.change is None
    assert q.change_percent is None


def test_get_quote_inf_price_raises_symbol_not_found() -> None:
    client = make_client(
        factory=fake_ticker_factory(fast_info={"last_price": float("inf"), "previous_close": 188.0})
    )
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("X")


class _RaisingCurrencyFastInfo:
    """fast_info stub whose `currency` property raises, last_price is fine."""

    last_price = 190.0
    previous_close = 188.0

    @property
    def currency(self) -> str:
        raise YFException("boom")


def test_get_quote_fast_info_attr_error_becomes_data_unavailable() -> None:
    df = make_history_df([100.0])

    def factory(_symbol: str) -> Any:
        return SimpleNamespace(fast_info=_RaisingCurrencyFastInfo(), history=lambda **_k: df)

    client = make_client(factory=factory)
    with pytest.raises(DataUnavailable) as exc:
        client._fetch_quote("X")
    assert "boom" in str(exc.value)


def test_get_quote_distinct_symbols_cached_independently() -> None:
    calls = {"n": 0}

    def counting_factory(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    client = YFinanceClient(
        ticker_factory=counting_factory, time_fn=FakeClock(), quote_ttl=30.0, history_ttl=300.0
    )
    results = client.get_quote(["AAPL", "MSFT"]).quotes
    assert [r.symbol for r in results] == ["AAPL", "MSFT"]
    assert calls["n"] == 2


def test_get_quote_zero_previous_close_change_pct_none() -> None:
    fi = {**QUOTE_FI, "previous_close": 0.0}
    [q] = make_client(factory=fake_ticker_factory(fast_info=fi)).get_quote(["AAPL"]).quotes
    assert q.change == pytest.approx(190.0) and q.change_percent is None and q.previous_close == 0.0


def test_get_quote_batch_keeps_good_tickers_when_one_is_missing() -> None:
    def factory(symbol: str) -> object:
        if symbol == "AAPL":
            return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)
        return fake_ticker_factory(fast_info_error=KeyError("exchangeTimezoneName"))(symbol)

    result = make_client(factory=factory).get_quote(["AAPL", "MSFT"])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert [e.symbol for e in result.errors] == ["MSFT"]


def test_get_quote_non_price_nan_fields_nulled() -> None:
    fi = {**QUOTE_FI, "market_cap": float("nan"), "last_volume": float("nan")}
    [q] = make_client(factory=fake_ticker_factory(fast_info=fi)).get_quote(["AAPL"]).quotes
    assert q.price == 190.0 and q.market_cap is None and q.volume is None


# --- get_quote: concurrent fetch with partial results (item 5) ---


def test_get_quote_returns_partial_results_instead_of_failing_the_batch() -> None:
    client = make_client(
        factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI, "MSFT": QUOTE_FI})
    )
    result = client.get_quote(["AAPL", "BADSYM", "MSFT"])
    assert [q.symbol for q in result.quotes] == ["AAPL", "MSFT"]  # request order preserved
    assert [e.symbol for e in result.errors] == ["BADSYM"]
    assert result.errors[0].error == (
        "No quote data for 'BADSYM'. The symbol may be invalid or delisted."
    )


def test_get_quote_error_entry_carries_a_transport_failure_message() -> None:
    client = make_client(
        factory=fake_symbol_ticker_factory(
            fast_info={"AAPL": QUOTE_FI}, errors={"MSFT": ConnectionError("connection reset")}
        )
    )
    result = client.get_quote(["AAPL", "MSFT"])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert "connection reset" in result.errors[0].error


def test_get_quote_all_failing_returns_no_quotes_and_all_errors() -> None:
    client = make_client(factory=fake_symbol_ticker_factory())
    result = client.get_quote(["NOPE1", "NOPE2"])
    assert result.quotes == []
    assert [e.symbol for e in result.errors] == ["NOPE1", "NOPE2"]


def test_get_quote_deduplicates_equivalent_symbols() -> None:
    calls: list[str] = []
    client = make_client(
        factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI}, calls=calls)
    )
    result = client.get_quote(["AAPL", "aapl", " AAPL "])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert calls == ["AAPL"]


def test_get_quote_blank_symbol_becomes_an_error_entry_not_an_exception() -> None:
    client = make_client(factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI}))
    result = client.get_quote(["AAPL", "  "])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert result.errors[0].symbol == "  "
    assert "Empty ticker symbol" in result.errors[0].error


def test_get_quote_empty_list_returns_empty_result() -> None:
    assert make_client(factory=fake_symbol_ticker_factory()).get_quote([]) == QuoteResult(
        quotes=[], errors=[]
    )


def test_get_quote_fetches_tickers_concurrently() -> None:
    # Every fetch waits on the barrier, so this only completes if all the symbols are in
    # flight at once; a sequential fetcher would block until the timeout. The party count
    # tracks the worker bound so lowering QUOTE_MAX_WORKERS cannot deadlock the test.
    symbols = [f"SYM{i}" for i in range(min(QUOTE_MAX_WORKERS, 4))]
    gate = threading.Barrier(len(symbols), timeout=10)
    client = make_client(
        factory=fake_symbol_ticker_factory(fast_info=dict.fromkeys(symbols, QUOTE_FI), gate=gate)
    )
    result = client.get_quote(symbols)
    assert [q.symbol for q in result.quotes] == symbols
    assert result.errors == []


def test_get_quote_concurrency_is_bounded() -> None:
    assert QUOTE_MAX_WORKERS > 1
    symbols = [f"SYM{i}" for i in range(QUOTE_MAX_WORKERS + 5)]
    client = make_client(
        factory=fake_symbol_ticker_factory(fast_info=dict.fromkeys(symbols, QUOTE_FI))
    )
    # More tickers than workers still completes: the pool queues the overflow.
    assert len(client.get_quote(symbols).quotes) == len(symbols)
