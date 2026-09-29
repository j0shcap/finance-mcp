"""Harness for the live Yahoo contract suite.

The mocked suites pin our parsing of a payload shape recorded once; this one treats Yahoo
as the system under test and the units our tool descriptions, `conventions.UNITS_GLOSSARY`
and the analyze_stock prompt promise as the contract.

Every test asserts presence AND range, because either alone is blind. `_opt(info.get(...))`
maps a missing key to None, so `x is None or 0 < x < 200` still holds after Yahoo renames
the key - the field is silently gone and the suite stays green. `require_present` is what
catches that; the ranges catch a unit flip in a field that is still there.

The `layer` fixture runs each test body twice, against the real client and over MCP, both
sharing one session-scoped client so the second pass comes from its cache rather than a
second Yahoo call.
"""

import asyncio
import datetime
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFRateLimitError

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.server import build_default_client, create_server

_LIVE_DIR = Path(__file__).parent

#: Few symbols, reused, so the shared cache absorbs most of the traffic.
AAPL = "AAPL"  # the best-covered ticker Yahoo has: every field we read is populated
BTC = "BTC-USD"  # 24/7, so intraday assertions do not depend on market hours
SAP = "SAP"  # reports in EUR, quotes in USD: the cross-listing currency invariant
SPY = "SPY"  # an ETF: no analyst coverage, so it must produce a clear error
UNKNOWN = "NOTATICKER.XX"  # never a real symbol: the partial-error path

#: One delay per retry, so the attempt count follows from the backoff schedule.
BACKOFF_SECONDS = (2.0, 8.0)
MAX_ATTEMPTS = len(BACKOFF_SECONDS) + 1

#: Tool argument -> client argument. The tools say "ticker(s)", the client says "symbol(s)".
_ARG_ALIASES = {"ticker": "symbol", "tickers": "symbols"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the tests in this directory `live`, so none can leak into the offline gate.

    The path filter is load-bearing: pytest calls this hook once with EVERY item in the
    session, including those outside this directory. Unfiltered, it marks the whole offline
    suite live and `-m 'not live'` then deselects all of it, leaving a gate that passes
    having run nothing. tests/test_live_marker.py runs a real collection pass against that.
    """
    for item in items:
        if item.path.is_relative_to(_LIVE_DIR):
            item.add_marker(pytest.mark.live)


def _is_throttle_message(text: str) -> bool:
    lowered = text.lower()
    return "rate limit" in lowered or "too many requests" in lowered


def _is_rate_limited(exc: BaseException) -> bool:
    """True when Yahoo is throttling us rather than reporting a real contract failure.

    Matching on text as well as type is required, not belt-and-braces: the client catches
    YFRateLimitError and re-raises it as DataUnavailable, so by the time a test sees it the
    type is gone and only the message survives.
    """
    return isinstance(exc, YFRateLimitError) or _is_throttle_message(str(exc))


def _throttled_quote_errors(result: Any) -> list[str]:
    """Throttle messages hiding in a QuoteResult.errors list.

    get_quote is partial by design, so a throttled symbol becomes an errors entry rather
    than an exception - which would otherwise read as a contract failure.
    """
    errors = getattr(result, "errors", None) or []
    return [e.error for e in errors if _is_throttle_message(e.error)]


@pytest.fixture(scope="session")
def yf_client() -> YFinanceClient:
    """One real client per session, built as production builds it.

    Session-scoped so its TTL cache spans the whole run; via build_default_client so the
    settings-to-client wiring is exercised too.
    """
    return build_default_client()


@pytest.fixture(scope="session")
def mcp_server(yf_client: YFinanceClient) -> Any:
    """A server on the SAME client, so the mcp layer reuses the direct layer's cache."""
    return create_server(yf_client=yf_client)


class Layer:
    """One of the two ways to call a tool, behind a single interface.

    Passing an mcp client selects the MCP layer; omitting it calls the data layer directly.
    Everything else about the layer follows from that, so the two cannot disagree.
    """

    def __init__(self, yf_client: YFinanceClient, mcp_client: Client[Any] | None = None) -> None:
        self._yf = yf_client
        self._mcp = mcp_client

    @property
    def name(self) -> str:
        return "mcp" if self._mcp is not None else "direct"

    @property
    def error_type(self) -> type[Exception]:
        """How a data-layer failure surfaces here: tools/_dispatch turns it into ToolError."""
        return ToolError if self._mcp is not None else DataUnavailable

    async def _invoke(self, tool: str, kwargs: dict[str, Any]) -> Any:
        if self._mcp is not None:
            result = await self._mcp.call_tool(tool, kwargs)
            return result.data
        renamed = {_ARG_ALIASES.get(k, k): v for k, v in kwargs.items()}
        # Off the event loop, as tools/_dispatch does: these are blocking HTTP calls.
        return await asyncio.to_thread(partial(getattr(self._yf, tool), **renamed))

    async def call(self, tool: str, **kwargs: Any) -> Any:
        """Call `tool`, retrying while throttled and skipping if it never clears.

        Returns the same model at either layer, so one test body covers both.
        """
        for delay in (*BACKOFF_SECONDS, None):  # None marks the final attempt
            try:
                result = await self._invoke(tool, kwargs)
                throttled = _throttled_quote_errors(result)
                if not throttled:
                    return result
                reason = throttled[0]
            except Exception as exc:
                if not _is_rate_limited(exc):
                    raise
                reason = str(exc)
            if delay is None:
                pytest.skip(f"Yahoo rate-limited {tool} after {MAX_ATTEMPTS} attempts: {reason}")
            await asyncio.sleep(delay)

    @asynccontextmanager
    async def expect_error(self, match: str) -> AsyncIterator[None]:
        """Assert the body fails with this layer's error type and a matching message.

        Not pytest.raises: throttling arrives as the same type as the failure under test, so
        it has to be filtered here rather than counted as a pass or reported as a break.
        """
        try:
            yield
        except self.error_type as exc:
            if _is_rate_limited(exc):
                pytest.skip(f"Yahoo rate-limited; cannot assert the {match!r} error: {exc}")
            assert re.search(match, str(exc)), (
                f"expected a {self.error_type.__name__} matching {match!r} at the "
                f"{self.name} layer, got: {exc}"
            )
            return
        pytest.fail(
            f"expected a {self.error_type.__name__} matching {match!r} at the "
            f"{self.name} layer, but the call succeeded"
        )


@pytest.fixture(params=["direct", "mcp"])
async def layer(
    request: pytest.FixtureRequest, yf_client: YFinanceClient, mcp_server: Any
) -> AsyncIterator[Layer]:
    """Runs each dependent test twice, once per layer.

    The mcp client is per-test because its connection is an async context manager and the
    default asyncio loop scope is per-test; that is free, since the transport is in-process
    and the Yahoo call underneath is cached.
    """
    if request.param == "direct":
        yield Layer(yf_client)
        return
    async with Client(mcp_server) as connected:
        yield Layer(yf_client, connected)


def require_present(model: Any, fields: Sequence[str]) -> None:
    """Assert every named field is populated - the drift detector.

    A field going null is how a renamed Yahoo key reaches us, and no range assertion can
    see it. Fix the client when this fails; relaxing it to tolerate None restores the blind
    spot the suite exists to remove.
    """
    missing = [f for f in fields if getattr(model, f) is None]
    assert not missing, (
        f"{type(model).__name__} fields unexpectedly null: {missing}. Yahoo most likely "
        f"renamed or dropped the underlying key(s); fix the client's mapping."
    )


def iso_dates_descending(dates: Sequence[str]) -> bool:
    """True when ISO date strings are strictly newest-first (lexicographic == chronological)."""
    return all(newer > older for newer, older in pairwise(dates))


def assert_distinct_intraday_timestamps(dates: Sequence[str]) -> None:
    """Intraday bars are moments: each needs a distinct, offset-bearing timestamp.

    Date-only values here would mean the daily-bar formatting has leaked into the intraday
    path, collapsing a session's bars onto one timestamp.
    """
    assert len(set(dates)) == len(dates), f"duplicate intraday timestamps in {dates[:5]}"
    assert list(dates) == sorted(dates), "intraday bars must be chronological"
    for date in dates:
        assert len(date) > 10, f"intraday bars need a full timestamp, got {date!r}"
        assert datetime.datetime.fromisoformat(date).tzinfo is not None, (
            f"intraday timestamps must carry a UTC offset: {date}"
        )
