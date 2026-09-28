"""Tests for the live suite's own harness, with no network involved.

The rate-limit handling in tests/live/conftest.py only runs when Yahoo is actually
throttling us, which is exactly when nobody is watching and a wrong branch turns into
either a red nightly build or - worse - a green one that asserted nothing. So the retry,
skip and error-matching paths are exercised here, offline, with fakes.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFRateLimitError

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from tests.live import conftest as live
from tests.live.conftest import Layer

#: yfinance's own message, as it reaches us wrapped by the client.
THROTTLED = "Too Many Requests. Rate limited. Try after a while."


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry path instant; the delays themselves are not what is under test."""
    monkeypatch.setattr(live, "BACKOFF_SECONDS", (0.0, 0.0))


class FakeClient:
    """Stands in for YFinanceClient, failing a configurable number of times first."""

    def __init__(self, failures: int, exc: Exception, result: Any = "ok") -> None:
        self.remaining = failures
        self.exc = exc
        self.result = result
        self.calls = 0

    def get_quote(self, symbols: list[str]) -> Any:
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.exc
        return self.result


def _layer(name: str, client: Any = None, mcp_client: Any = None) -> Layer:
    """A Layer over fakes; the constructor's types are widened to Any for the stand-ins."""
    return Layer(name, client, mcp_client)


def _direct(client: Any = None) -> Layer:
    return _layer("direct", client)


@pytest.mark.parametrize(
    "exc",
    [
        YFRateLimitError(),
        # What the client actually raises: the type is gone, only the message survives.
        DataUnavailable(f"Failed to fetch quote for 'AAPL': {THROTTLED}"),
        ToolError("Failed to fetch metrics for 'AAPL': rate limit exceeded"),
        DataUnavailable("Failed to fetch news for 'AAPL': 429 Client Error"),
    ],
)
def test_rate_limit_is_recognised_however_it_arrives(exc: Exception) -> None:
    assert live._is_rate_limited(exc)


@pytest.mark.parametrize(
    "exc",
    [
        SymbolNotFound("No quote data for 'NOTATICKER.XX'. The symbol may be invalid or delisted."),
        DataUnavailable("No analyst coverage for 'SPY'."),
        ToolError("Failed to parse profile for 'AAPL': unexpected payload"),
    ],
)
def test_real_failures_are_not_mistaken_for_throttling(exc: Exception) -> None:
    """A contract failure must stay a failure, or this suite would skip its way to green."""
    assert not live._is_rate_limited(exc)


async def test_call_retries_a_throttled_request_and_succeeds() -> None:
    client = FakeClient(failures=1, exc=YFRateLimitError())

    result = await _direct(client).call("get_quote", tickers=["AAPL"])

    assert result == "ok"
    assert client.calls == 2, "the first attempt was throttled, so it must be retried"


async def test_call_skips_after_exhausting_its_attempts() -> None:
    """A persistently throttled call is a skip, not a failure - the whole point of requirement 4."""
    client = FakeClient(failures=99, exc=DataUnavailable(f"Failed to fetch quote: {THROTTLED}"))

    with pytest.raises(pytest.skip.Exception, match="rate-limited"):
        await _direct(client).call("get_quote", tickers=["AAPL"])

    assert client.calls == live.MAX_ATTEMPTS


async def test_call_reraises_a_genuine_failure_without_retrying() -> None:
    """A real error must surface immediately, not be retried three times and then skipped."""
    client = FakeClient(failures=99, exc=DataUnavailable("Failed to parse profile for 'AAPL': x"))

    with pytest.raises(DataUnavailable, match="Failed to parse profile"):
        await _direct(client).call("get_quote", tickers=["AAPL"])

    assert client.calls == 1


async def test_call_skips_when_throttling_hides_in_the_quote_errors_list() -> None:
    """get_quote reports per-symbol failures instead of raising, so it needs its own path.

    Without this, a fully throttled batch reads as a contract failure ("expected a quote for
    AAPL, got an error") rather than as Yahoo rate-limiting us.
    """

    throttled = SimpleNamespace(
        symbol="AAPL", error=f"Failed to fetch quote for 'AAPL': {THROTTLED}"
    )
    result = SimpleNamespace(errors=[throttled], quotes=[])
    client = FakeClient(failures=0, exc=YFRateLimitError(), result=result)

    with pytest.raises(pytest.skip.Exception, match="rate-limited"):
        await _direct(client).call("get_quote", tickers=["AAPL"])


async def test_call_returns_a_partial_quote_result_with_ordinary_errors() -> None:
    """An invalid-symbol error in the same list must NOT trigger a skip."""

    invalid = SimpleNamespace(
        symbol="NOTATICKER.XX",
        error="No quote data for 'NOTATICKER.XX'. The symbol may be invalid or delisted.",
    )
    client = FakeClient(
        failures=0,
        exc=YFRateLimitError(),
        result=SimpleNamespace(errors=[invalid], quotes=[]),
    )

    result = await _direct(client).call("get_quote", tickers=["NOTATICKER.XX"])

    assert result.errors[0].symbol == "NOTATICKER.XX"


async def test_expect_error_matches_the_message() -> None:
    async with _direct(None).expect_error("No analyst coverage"):
        raise DataUnavailable("No analyst coverage for 'SPY'.")


async def test_expect_error_rejects_a_different_error() -> None:
    """A wrong-but-real error is a failure: the message is part of the contract."""
    with pytest.raises(AssertionError, match="expected a DataUnavailable"):
        async with _direct(None).expect_error("No analyst coverage"):
            raise DataUnavailable("Failed to fetch analyst data for 'SPY': timeout")


async def test_expect_error_fails_when_nothing_is_raised() -> None:
    """If Yahoo starts covering ETFs, the SPY test has to go red rather than quietly pass."""
    with pytest.raises(pytest.fail.Exception, match="but the call succeeded"):
        async with _direct(None).expect_error("No analyst coverage"):
            pass


async def test_expect_error_skips_when_throttled_instead_of_matching() -> None:
    """A rate limit arrives as the same type, so it must not be counted as the expected error."""
    with pytest.raises(pytest.skip.Exception, match="rate-limited"):
        async with _direct(None).expect_error("No analyst coverage"):
            raise DataUnavailable(f"Failed to fetch analyst data for 'SPY': {THROTTLED}")


def test_mcp_layer_expects_tool_error() -> None:
    """The two layers surface the same domain failure as different types."""
    assert _layer("direct").error_type is DataUnavailable
    assert _layer("mcp", mcp_client=object()).error_type is ToolError
