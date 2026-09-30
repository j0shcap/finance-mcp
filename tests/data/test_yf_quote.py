"""YFinanceClient.get_quote: parsing, caching and partial batches."""

import threading
from types import SimpleNamespace

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
)
from tests.fakes import (
    QUOTE_FI,
    FakeClock,
    counting,
    fake_symbol_ticker_factory,
    fake_ticker_factory,
    make_client,
)


def test_get_quote_parses_and_computes_change() -> None:
    [q] = make_client().get_quote(["AAPL"]).quotes
    assert q.symbol == "AAPL"
    assert q.price == 190.0
    assert q.change == pytest.approx(2.0)
    assert q.change_percent == pytest.approx(2.0 / 188.0 * 100.0)
    assert q.currency == "USD"


def test_get_quote_caches_within_ttl() -> None:
    factory, calls = counting(fake_ticker_factory(fast_info=QUOTE_FI))
    clock = FakeClock()
    client = make_client(factory, clock=clock, quote_ttl=30.0)
    client.get_quote(["AAPL"])
    client.get_quote(["AAPL"])
    assert len(calls) == 1
    clock.advance(29.9)
    client.get_quote(["AAPL"])
    assert len(calls) == 1
    # An entry exactly ttl old is stale.
    clock.advance(0.1)
    client.get_quote(["AAPL"])
    assert len(calls) == 2


@pytest.mark.parametrize("price", [None, float("nan"), float("inf")])
def test_get_quote_without_a_finite_price_is_symbol_not_found(price: float | None) -> None:
    client = make_client(
        fake_ticker_factory(fast_info={"last_price": price, "previous_close": 1.0})
    )
    with pytest.raises(SymbolNotFound, match="No quote data for 'BAD'"):
        client._fetch_quote("BAD")


def test_get_quote_surfaces_yfinance_error_message() -> None:
    client = make_client(
        factory=fake_ticker_factory(fast_info_error=YFException("yahoo says: rate limited"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client._fetch_quote("AAPL")
    assert "yahoo says: rate limited" in str(exc.value)


def test_get_quote_nan_previous_close_yields_none_change() -> None:
    fi = {**QUOTE_FI, "previous_close": float("nan")}
    client = make_client(factory=fake_ticker_factory(fast_info=fi))
    [q] = client.get_quote(["AAPL"]).quotes
    assert q.price == 190.0
    assert q.change is None
    assert q.change_percent is None


class _RaisingCurrencyFastInfo:
    """fast_info stub whose `currency` property raises, last_price is fine."""

    last_price = 190.0
    previous_close = 188.0

    @property
    def currency(self) -> str:
        raise YFException("boom")


def test_get_quote_fast_info_attr_error_becomes_data_unavailable() -> None:
    client = make_client(lambda _symbol: SimpleNamespace(fast_info=_RaisingCurrencyFastInfo()))
    with pytest.raises(DataUnavailable) as exc:
        client._fetch_quote("X")
    assert "boom" in str(exc.value)


def test_get_quote_distinct_symbols_cached_independently() -> None:
    factory, calls = counting(fake_ticker_factory(fast_info=QUOTE_FI))
    client = make_client(factory)
    results = client.get_quote(["AAPL", "MSFT"]).quotes
    client.get_quote(["AAPL", "MSFT"])
    assert [r.symbol for r in results] == ["AAPL", "MSFT"]
    assert calls == ["AAPL", "MSFT"]


def test_get_quote_zero_previous_close_change_pct_none() -> None:
    fi = {**QUOTE_FI, "previous_close": 0.0}
    [q] = make_client(factory=fake_ticker_factory(fast_info=fi)).get_quote(["AAPL"]).quotes
    assert q.change == pytest.approx(190.0) and q.change_percent is None and q.previous_close == 0.0


def test_get_quote_non_price_nan_fields_nulled() -> None:
    fi = {**QUOTE_FI, "market_cap": float("nan"), "last_volume": float("nan")}
    [q] = make_client(factory=fake_ticker_factory(fast_info=fi)).get_quote(["AAPL"]).quotes
    assert q.price == 190.0 and q.market_cap is None and q.volume is None


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
    factory, calls = counting(fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI}))
    client = make_client(factory)
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


def test_get_quote_queues_more_tickers_than_workers() -> None:
    symbols = [f"SYM{i}" for i in range(QUOTE_MAX_WORKERS + 5)]
    client = make_client(
        factory=fake_symbol_ticker_factory(fast_info=dict.fromkeys(symbols, QUOTE_FI))
    )
    assert len(client.get_quote(symbols).quotes) == len(symbols)
