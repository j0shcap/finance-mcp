"""Which yfinance failures become SymbolNotFound and which stay DataUnavailable."""

import pytest
import yfinance as yf
from yfinance.exceptions import (
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
    YFTzMissingError,
)

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.providers.yahoo import YahooProvider, is_transient
from tests.fakes import (
    fake_ticker_factory,
    make_client,
)


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeHTTPError(Exception):
    """Shaped like the curl_cffi/requests HTTPError yfinance lets escape raise_for_status()."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _FakeResponse(status_code)


TRANSPORT_ERRORS = [
    ConnectionError("Connection reset by peer"),
    TimeoutError("timed out after 30s"),
    OSError("network is unreachable"),
    _FakeHTTPError("500 Server Error: Internal Server Error for url: ...", 500),
    _FakeHTTPError("503 Server Error: Service Unavailable for url: ...", 503),
    ValueError("Expecting value: line 1 column 1 (char 0)"),
]


@pytest.mark.parametrize("exc", TRANSPORT_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_quote_transport_failure_is_data_unavailable_not_symbol_not_found(
    exc: Exception, instant_backoff: list[float]
) -> None:
    source = YahooProvider(fake_ticker_factory(fast_info_error=exc))
    with pytest.raises(DataUnavailable) as raised:
        source.quote("AAPL")
    # Dropped connections and 5xx are retried before failing; the rest fail at once.
    assert instant_backoff == ([2.0, 6.0] if is_transient(exc) else [])
    assert not isinstance(raised.value, SymbolNotFound)
    assert str(exc) in str(raised.value)  # underlying message preserved
    assert "may be invalid or delisted" not in str(raised.value)


@pytest.mark.parametrize("exc", TRANSPORT_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_info_transport_failure_is_data_unavailable_not_symbol_not_found(exc: Exception) -> None:
    client = make_client(factory=fake_ticker_factory(info_error=exc))
    with pytest.raises(DataUnavailable) as raised:
        client.get_company_profile("AAPL")
    assert not isinstance(raised.value, SymbolNotFound)
    assert str(exc) in str(raised.value)


NO_DATA_ERRORS = [
    KeyError("exchangeTimezoneName"),
    YFTickerMissingError("NOPE", "possibly delisted; no price data found"),
    YFTzMissingError("NOPE"),
    YFPricesMissingError("NOPE", ""),
    _FakeHTTPError("404 Client Error: Not Found for url: ...", 404),
    Exception("Quote not found for ticker symbol: NOPE"),
]


@pytest.mark.parametrize("exc", NO_DATA_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_quote_no_data_signals_are_symbol_not_found(exc: Exception) -> None:
    source = YahooProvider(fake_ticker_factory(fast_info_error=exc))
    with pytest.raises(SymbolNotFound) as raised:
        source.quote("NOPE")
    assert str(raised.value) == "No quote data for 'NOPE'. The symbol may be invalid or delisted."


@pytest.mark.parametrize("exc", NO_DATA_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_info_no_data_signals_are_symbol_not_found(exc: Exception) -> None:
    client = make_client(factory=fake_ticker_factory(info_error=exc))
    with pytest.raises(SymbolNotFound) as raised:
        client.get_key_metrics("NOPE")
    assert "No metrics data for 'NOPE'" in str(raised.value)


def test_rate_limit_error_stays_data_unavailable() -> None:
    source = YahooProvider(fake_ticker_factory(fast_info_error=YFRateLimitError()))
    with pytest.raises(DataUnavailable) as raised:
        source.quote("AAPL")
    assert not isinstance(raised.value, SymbolNotFound)
    assert "Rate limited" in str(raised.value)


@pytest.mark.parametrize("exc", TRANSPORT_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_history_transport_failure_is_data_unavailable_not_symbol_not_found(
    exc: Exception,
) -> None:
    client = make_client(factory=fake_ticker_factory(history_error=exc))
    with pytest.raises(DataUnavailable) as raised:
        client.get_price_history("AAPL", period="1mo", interval="1d")
    assert not isinstance(raised.value, SymbolNotFound)
    assert str(exc) in str(raised.value)


@pytest.mark.parametrize("exc", NO_DATA_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_history_no_data_signals_are_symbol_not_found(exc: Exception) -> None:
    # Unhidden, Yahoo's 404 for an unknown symbol raises rather than returning an empty
    # frame, so it must be classified, not wrapped.
    client = make_client(factory=fake_ticker_factory(history_error=exc))
    with pytest.raises(SymbolNotFound, match="No price history for 'NOPE'"):
        client.get_price_history("NOPE", period="1mo", interval="1d")


def test_client_makes_yfinance_raise_instead_of_hiding_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # By default a transport error comes back as an empty result, which reads exactly
    # like an unknown symbol.
    monkeypatch.setattr(yf.config.debug, "hide_exceptions", True)
    make_client(factory=fake_ticker_factory())
    assert yf.config.debug.hide_exceptions is False
