"""Offline tests for the live suite's own harness.

The rate-limit handling in tests/live/conftest.py only executes while Yahoo is throttling
us - precisely when nobody is watching, and where a wrong branch means either a red nightly
build or a green one that asserted nothing. So it is exercised here with fakes.
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
    """Keep the retries instant; the delays are not what is under test."""
    monkeypatch.setattr(live, "BACKOFF_SECONDS", (0.0, 0.0))


class FakeClient:
    """Stands in for YFinanceClient, failing a set number of times before succeeding."""

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


def _layer(client: Any = None, mcp_client: Any = None) -> Layer:
    """A Layer over fakes, with the constructor's types widened for the stand-ins."""
    return Layer(client, mcp_client)


@pytest.mark.parametrize(
    "exc",
    [
        YFRateLimitError(),
        # What the client actually raises: the type is gone, only the message survives.
        DataUnavailable(f"Failed to fetch quote for 'AAPL': {THROTTLED}"),
        ToolError("Failed to fetch metrics for 'AAPL': rate limit exceeded"),
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
        # A bare number must not read as a 429: line-item values and prices contain digits.
        DataUnavailable("Failed to parse income statement for 'AAPL': bad value 4290000"),
    ],
)
def test_real_failures_are_not_mistaken_for_throttling(exc: Exception) -> None:
    """A contract failure must stay a failure, or the suite could skip its way to green."""
    assert not live._is_rate_limited(exc)


async def test_call_retries_a_throttled_request_and_succeeds() -> None:
    client = FakeClient(failures=1, exc=YFRateLimitError())

    result = await _layer(client).call("get_quote", tickers=["AAPL"])

    assert result == "ok"
    assert client.calls == 2, "the first attempt was throttled, so it must be retried"


async def test_call_skips_after_exhausting_its_attempts() -> None:
    """A persistently throttled call is a skip, not a failure."""
    client = FakeClient(failures=99, exc=DataUnavailable(f"Failed to fetch quote: {THROTTLED}"))

    with pytest.raises(pytest.skip.Exception, match="rate-limited"):
        await _layer(client).call("get_quote", tickers=["AAPL"])

    assert client.calls == live.MAX_ATTEMPTS


async def test_call_reraises_a_genuine_failure_without_retrying() -> None:
    """A real error surfaces at once rather than being retried and then skipped."""
    client = FakeClient(failures=99, exc=DataUnavailable("Failed to parse profile for 'AAPL': x"))

    with pytest.raises(DataUnavailable, match="Failed to parse profile"):
        await _layer(client).call("get_quote", tickers=["AAPL"])

    assert client.calls == 1


async def test_call_skips_when_throttling_hides_in_the_quote_errors_list() -> None:
    """get_quote reports per-symbol failures instead of raising, so it needs its own path."""
    throttled = SimpleNamespace(
        symbol="AAPL", error=f"Failed to fetch quote for 'AAPL': {THROTTLED}"
    )
    result = SimpleNamespace(errors=[throttled], quotes=[])
    client = FakeClient(failures=0, exc=YFRateLimitError(), result=result)

    with pytest.raises(pytest.skip.Exception, match="rate-limited"):
        await _layer(client).call("get_quote", tickers=["AAPL"])


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

    result = await _layer(client).call("get_quote", tickers=["NOTATICKER.XX"])

    assert result.errors[0].symbol == "NOTATICKER.XX"


async def test_expect_error_matches_the_message() -> None:
    async with _layer().expect_error("No analyst coverage"):
        raise DataUnavailable("No analyst coverage for 'SPY'.")


async def test_expect_error_rejects_a_different_error() -> None:
    """A wrong-but-real error still fails: the message is part of the contract."""
    with pytest.raises(AssertionError, match="expected a DataUnavailable"):
        async with _layer().expect_error("No analyst coverage"):
            raise DataUnavailable("Failed to fetch analyst data for 'SPY': timeout")


async def test_expect_error_fails_when_nothing_is_raised() -> None:
    """If Yahoo starts covering ETFs, the SPY test goes red rather than quietly passing."""
    with pytest.raises(pytest.fail.Exception, match="but the call succeeded"):
        async with _layer().expect_error("No analyst coverage"):
            pass


async def test_expect_error_skips_when_throttled_instead_of_matching() -> None:
    """Throttling arrives as the same type, so it must not count as the expected error."""
    with pytest.raises(pytest.skip.Exception, match="rate-limited"):
        async with _layer().expect_error("No analyst coverage"):
            raise DataUnavailable(f"Failed to fetch analyst data for 'SPY': {THROTTLED}")


def test_the_layers_surface_the_same_failure_as_different_types() -> None:
    assert _layer().error_type is DataUnavailable
    assert _layer().name == "direct"
    assert _layer(mcp_client=object()).error_type is ToolError
    assert _layer(mcp_client=object()).name == "mcp"
