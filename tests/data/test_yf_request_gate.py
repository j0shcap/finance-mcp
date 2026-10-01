"""YahooSource's request gate: a cap on concurrent Yahoo requests, and retries with backoff."""

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from curl_cffi.const import CurlECode
from curl_cffi.requests.exceptions import code2error
from yfinance.exceptions import YFRateLimitError

from finance_mcp.data import yahoo
from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.yahoo import YahooSource, is_transient
from finance_mcp.data.yahoo import _jitter as real_jitter  # bound before the test patch
from finance_mcp.server import build_default_client
from tests.fakes import (
    QUOTE_FI,
    FakeHTTPError,
    fake_multi_ticker_factory,
    make_client,
    make_history_df,
)

#: Long enough that a regression fails the test rather than hanging the run.
TIMEOUT = 10.0


def _flaky_quotes(failures: list[Exception]) -> tuple[Callable[[str], Any], list[str]]:
    """A ticker factory whose fast_info raises each of ``failures`` in turn, then answers."""
    attempts: list[str] = []

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def fast_info(self) -> Any:
            attempts.append(self._symbol)
            if failures:
                raise failures.pop(0)
            return SimpleNamespace(**QUOTE_FI)

    return _Ticker, attempts


def _run(target: Callable[[], Any]) -> Any:
    """Run ``target`` on a thread and fail, rather than hang, if it doesn't finish."""
    result: list[Any] = []
    thread = threading.Thread(target=lambda: result.append(target()), daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    assert not thread.is_alive(), "deadlocked or hung"
    return result[0]


# --- the cap ------------------------------------------------------------------------------


def test_no_more_than_the_cap_of_requests_run_at_once() -> None:
    lock = threading.Lock()
    in_flight, peak = 0, 0
    cap_reached = threading.Event()

    class _Ticker:
        def __init__(self, _symbol: str) -> None:
            pass

        @property
        def fast_info(self) -> Any:
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
                if in_flight >= 3:
                    cap_reached.set()
            cap_reached.wait(TIMEOUT)  # hold the slot until the cap has been reached
            with lock:
                in_flight -= 1
            return SimpleNamespace(**QUOTE_FI)

    client = make_client(_Ticker, max_concurrent_requests=3)
    symbols = [f"SYM{i}" for i in range(10)]
    result = _run(lambda: client.get_quote(symbols))
    assert [q.symbol for q in result.quotes] == symbols
    assert cap_reached.is_set() and peak == 3  # reached the cap, never exceeded it


def test_nested_fetches_do_not_deadlock_at_a_cap_of_one() -> None:
    history = make_history_df([100.0 + i for i in range(200)])
    info = {"longName": "Apple Inc.", "currency": "USD"}
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": history, "info": info},
            "MSFT": {"history_df": history, "info": info},
            "SPY": {"history_df": history},
            "^IRX": {"history_df": make_history_df([4.0] * 200)},
        }
    )
    client = make_client(factory, max_concurrent_requests=1)
    # asset || benchmark || bills, nested two deep
    assert _run(lambda: client.compare_to_benchmark("AAPL", "SPY", "1y")).beta is not None
    # rows that wait on a background T-bill fetch
    assert len(_run(lambda: client.compare_tickers(["AAPL", "MSFT"], "1y")).rows) == 2


def test_a_request_inside_a_request_is_refused() -> None:
    source = YahooSource()
    with pytest.raises(RuntimeError, match="nested Yahoo request"):
        source._request(lambda: source._request(lambda: 1))


# --- retries ------------------------------------------------------------------------------


def test_a_throttled_request_is_retried_with_backoff(instant_backoff: list[float]) -> None:
    factory, attempts = _flaky_quotes([YFRateLimitError(), YFRateLimitError()])
    quote = YahooSource(factory).quote("AAPL")
    assert quote.price == QUOTE_FI["last_price"]
    assert len(attempts) == 3
    assert instant_backoff == [2.0, 6.0]


def test_retries_give_up_with_the_original_error(instant_backoff: list[float]) -> None:
    factory, attempts = _flaky_quotes([YFRateLimitError() for _ in range(5)])
    with pytest.raises(DataUnavailable, match="Too Many Requests"):
        YahooSource(factory).quote("AAPL")
    assert len(attempts) == 3
    assert instant_backoff == [2.0, 6.0]


def test_an_unknown_symbol_is_not_retried(instant_backoff: list[float]) -> None:
    factory, attempts = _flaky_quotes([FakeHTTPError(404)])
    with pytest.raises(SymbolNotFound):
        YahooSource(factory).quote("NOPE")
    assert len(attempts) == 1
    assert instant_backoff == []


def test_retries_can_be_turned_off(instant_backoff: list[float]) -> None:
    factory, attempts = _flaky_quotes([YFRateLimitError()])
    with pytest.raises(DataUnavailable):
        YahooSource(factory, request_retries=0).quote("AAPL")
    assert len(attempts) == 1


def test_no_retry_starts_past_the_time_budget(
    monkeypatch: pytest.MonkeyPatch, instant_backoff: list[float]
) -> None:
    monkeypatch.setattr(yahoo, "RETRY_BUDGET_SECONDS", 1.0)  # shorter than the first 2s wait
    factory, attempts = _flaky_quotes([YFRateLimitError()])
    with pytest.raises(DataUnavailable):
        YahooSource(factory).quote("AAPL")
    assert len(attempts) == 1


def test_a_best_effort_read_is_not_retried(instant_backoff: list[float]) -> None:
    reads: list[str] = []

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def info(self) -> Any:
            reads.append(self._symbol)
            raise YFRateLimitError()

    with pytest.raises(YFRateLimitError):
        YahooSource(_Ticker).statement_currency("SAP")
    assert reads == ["SAP"]


def test_the_gate_is_released_while_backing_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cap of one: if the throttled request kept its slot through the backoff, the request
    # made during that backoff could never start.
    factory, _ = _flaky_quotes([YFRateLimitError()])
    source = YahooSource(factory, max_concurrent_requests=1)
    during_backoff: list[float] = []

    def sleep_while_another_request_runs(_delay: float) -> None:
        during_backoff.append(_run(lambda: source.quote("OTHER")).price)

    monkeypatch.setattr(yahoo, "_sleep", sleep_while_another_request_runs)
    assert source.quote("AAPL").price == QUOTE_FI["last_price"]
    assert during_backoff == [QUOTE_FI["last_price"]]


def test_jitter_spreads_a_delay_by_up_to_half_either_way() -> None:
    samples = [real_jitter(2.0) for _ in range(200)]
    assert all(1.0 <= sample <= 3.0 for sample in samples)
    assert len(set(samples)) > 1


# --- what counts as transient -------------------------------------------------------------


def _curl(code: CurlECode) -> Exception:
    """The exception curl_cffi itself raises for a curl error code."""
    error: Exception = code2error(code, "message")("message", code)
    return error


def _status(code: int) -> Exception:
    exc = RuntimeError(f"HTTP Error {code}")
    exc.response = SimpleNamespace(status_code=code)  # type: ignore[attr-defined]
    return exc


@pytest.mark.parametrize(
    ("exc", "transient"),
    [
        (YFRateLimitError(), True),
        (_curl(CurlECode.COULDNT_CONNECT), True),
        (_curl(CurlECode.RECV_ERROR), True),
        (_curl(CurlECode.COULDNT_RESOLVE_HOST), True),  # DNSError
        (_curl(CurlECode.HTTP2_STREAM), True),
        (_curl(CurlECode.HTTP2), True),
        (_curl(CurlECode.PARTIAL_FILE), True),  # IncompleteRead
        (_status(429), True),
        (_status(503), True),
        # a timeout has already spent yfinance's 30s
        (_curl(CurlECode.OPERATION_TIMEDOUT), False),
        # SSL failures subclass ConnectionError but don't heal
        (_curl(CurlECode.SSL_CONNECT_ERROR), False),
        (_curl(CurlECode.PEER_FAILED_VERIFICATION), False),
        (_curl(CurlECode.COULDNT_RESOLVE_PROXY), False),
        (_status(404), False),
        (FakeHTTPError(404), False),
        (KeyError("exchangeTimezoneName"), False),
    ],
    ids=lambda value: str(value) if isinstance(value, bool) else repr(value)[:40],
)
def test_is_transient_for_what_curl_cffi_really_raises(exc: Exception, transient: bool) -> None:
    assert is_transient(exc) is transient


# --- configuration ------------------------------------------------------------------------


def test_settings_reach_the_request_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCE_MCP_MAX_CONCURRENT_REQUESTS", "3")
    monkeypatch.setenv("FINANCE_MCP_REQUEST_RETRIES", "0")
    source = build_default_client()._source
    assert source._gate._initial_value == 3  # type: ignore[attr-defined]
    assert source._retry_delays == ()


@pytest.mark.parametrize("value", ["0", "33"])
def test_an_unusable_request_cap_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("FINANCE_MCP_MAX_CONCURRENT_REQUESTS", value)
    with pytest.raises(ValidationError):
        build_default_client()
