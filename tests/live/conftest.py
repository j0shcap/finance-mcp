"""Harness for the live Yahoo contract suite.

WHAT THIS SUITE IS FOR
----------------------
Every other test in this repo mocks yfinance, so it pins *our parsing* of a payload shape
we wrote down once. None of them can see Yahoo change that shape: a renamed ``info`` key,
``pegRatio`` becoming ``trailingPegRatio``, ``dividendYield`` flipping between a fraction
and a percent, the news item's nested ``content`` moving. This suite treats Yahoo as the
system under test and the units/shape our tool descriptions, ``conventions.UNITS_GLOSSARY``
and the ``analyze_stock`` prompt promise as the contract.

TWO KINDS OF ASSERTION - BOTH ARE LOAD-BEARING
----------------------------------------------
``_opt(info.get(...))`` maps every missing key to ``None``, so a range check alone passes
*vacuously* on exactly the drift we are hunting: if Yahoo renames ``pegRatio``, then
``peg_ratio is None or 0 < peg_ratio < 200`` is satisfied and the suite stays green while
the field is silently gone. So every test pairs:

* ``require_present`` - for a large, well-covered name (AAPL) these fields MUST be
  non-null. This is the drift detector.
* range/shape assertions - when present, the value must sit in the band its documented
  unit implies. This is the units detector.

If a ``require_present`` call starts failing, the field really did disappear. Fix the
client (or the docs it contradicts); do NOT relax the assertion to ``is None or ...``,
which would restore the blind spot this suite exists to remove.

RATE LIMITS READ AS SKIPS, NEVER FAILURES
-----------------------------------------
``YFRateLimitError`` never escapes ``YFinanceClient``: the client wraps it into
``DataUnavailable`` with the message preserved verbatim, and ``tools/_dispatch`` re-wraps
that into ``ToolError``. Detection is therefore textual as well as type-based (see
``_is_rate_limited``), and ``get_quote`` needs a third path because it reports per-symbol
failures in ``errors`` instead of raising at all.

TWO LAYERS, ONE ASSERTION BODY
------------------------------
The ``layer`` fixture is parametrized over ``direct`` (the real ``YFinanceClient``) and
``mcp`` (an in-process ``fastmcp.Client`` over ``create_server``), so each contract is
asserted at both layers from a single test body. Both layers share ONE session-scoped
client, so the ``mcp`` pass is served from that client's cache and costs almost no extra
Yahoo traffic - which is what keeps this suite under the rate limit.
"""

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFRateLimitError

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.server import build_default_client, create_server

#: Symbols are deliberately few and reused across tests so the shared cache absorbs most
#: of the traffic. Each one is here for a reason the tests name.
AAPL = "AAPL"  # the best-covered ticker Yahoo has: every field we read is populated
BTC = "BTC-USD"  # 24/7, so intraday/calendar assertions do not depend on market hours
SAP = "SAP"  # reports in EUR, quotes in USD: the cross-listing currency invariant
SPY = "SPY"  # an ETF: no analyst coverage, so it must produce a clear error
UNKNOWN = "NOTATICKER.XX"  # never a real symbol: the partial-error path

#: Up to three attempts, backing off between them, before a rate limit becomes a skip.
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2.0, 8.0)

#: Tool argument name -> client argument name. The tools speak "ticker(s)" (what a user
#: types), the client speaks "symbol(s)"; everything else is spelled identically.
_ARG_ALIASES = {"ticker": "symbol", "tickers": "symbols"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark everything under ``tests/live/`` as ``live``, so nothing here can leak.

    Directory-based rather than a per-test decorator on purpose: a forgotten decorator
    would put a real network call inside ``make check``, and there is no way to forget
    this.
    """
    for item in items:
        item.add_marker(pytest.mark.live)


def _is_rate_limited(exc: BaseException) -> bool:
    """True when ``exc`` is Yahoo throttling us rather than a real contract failure.

    The type check alone is not enough: the client catches ``YFRateLimitError`` and
    re-raises it as ``DataUnavailable`` carrying the original message, so by the time a
    test sees it the type is gone and only the text remains.
    """
    if isinstance(exc, YFRateLimitError):
        return True
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text or "429" in text


def _rate_limited_quote_errors(result: Any) -> list[str]:
    """Rate-limit messages hiding in a ``QuoteResult.errors`` list.

    ``get_quote`` is partial by design: a throttled symbol becomes a ``QuoteError`` entry
    instead of an exception. Without this, a fully rate-limited batch would look like a
    contract failure ("expected a quote for AAPL, got an error").
    """
    errors = getattr(result, "errors", None)
    if not errors:
        return []
    return [e.error for e in errors if _is_rate_limited(Exception(e.error))]


@pytest.fixture(scope="session")
def yf_client() -> YFinanceClient:
    """One real client for the whole session, built exactly as production builds it.

    Session-scoped for two reasons: its TTL cache (30s quotes / 300s history / 1h
    fundamentals) then spans the whole run, and going through ``build_default_client``
    means this suite also exercises the real settings-to-client wiring.
    """
    return build_default_client()


@pytest.fixture(scope="session")
def mcp_server(yf_client: YFinanceClient) -> Any:
    """A server bound to the SAME client, so the mcp layer reuses the direct layer's cache."""
    return create_server(yf_client=yf_client)


class Layer:
    """One of the two ways a tool can be called, behind a single interface.

    ``name`` is ``"direct"`` or ``"mcp"``; tests that need to branch on the layer read it,
    but the point of this class is that almost none of them do.
    """

    def __init__(self, name: str, yf_client: YFinanceClient, mcp_client: Client[Any] | None):
        self.name = name
        self._yf = yf_client
        self._mcp = mcp_client
        #: The exception a data-layer failure surfaces as at this layer: the client raises
        #: DataUnavailable, and tools/_dispatch translates that into ToolError for MCP.
        self.error_type: type[Exception] = DataUnavailable if name == "direct" else ToolError

    async def _invoke(self, tool: str, kwargs: dict[str, Any]) -> Any:
        if self._mcp is not None:
            result = await self._mcp.call_tool(tool, kwargs)
            return result.data
        renamed = {_ARG_ALIASES.get(k, k): v for k, v in kwargs.items()}
        method = getattr(self._yf, tool)
        # Off the event loop, as tools/_dispatch does: these are blocking HTTP calls.
        return await asyncio.to_thread(lambda: method(**renamed))

    async def call(self, tool: str, **kwargs: Any) -> Any:
        """Call ``tool`` at this layer, retrying a rate limit and finally skipping on one.

        Returns the same pydantic model either way, so one assertion body covers both
        layers.
        """
        for attempt in range(MAX_ATTEMPTS):
            last = attempt == MAX_ATTEMPTS - 1
            try:
                result = await self._invoke(tool, kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised below unless throttled
                if not _is_rate_limited(exc):
                    raise
                if last:
                    pytest.skip(f"Yahoo rate-limited {tool} after {MAX_ATTEMPTS} attempts: {exc}")
            else:
                throttled = _rate_limited_quote_errors(result)
                if not throttled:
                    return result
                if last:
                    pytest.skip(f"Yahoo rate-limited {tool} after {MAX_ATTEMPTS} attempts: "
                                f"{throttled[0]}")
            await asyncio.sleep(BACKOFF_SECONDS[attempt])
        raise AssertionError("unreachable: the final attempt either returns or skips")

    @asynccontextmanager
    async def expect_error(self, match: str) -> AsyncIterator[None]:
        """Assert the body fails with this layer's error type and a message matching ``match``.

        Not ``pytest.raises``: a rate limit arrives as the same exception type as the
        failure we are asserting, so it has to be filtered out here rather than counted as
        a pass (which would make the test vacuous) or a failure (which would be a false
        alarm).
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
    """Every test that uses this runs twice: once per layer.

    The mcp client is function-scoped because its connection is an async context manager
    and the default pytest-asyncio loop is per-test; that costs nothing, since the
    transport is in-process and the expensive part (the Yahoo call) is cached on the
    session-scoped client underneath.
    """
    if request.param == "direct":
        yield Layer("direct", yf_client, None)
        return
    async with Client(mcp_server) as connected:
        yield Layer("mcp", yf_client, connected)


def require_present(model: Any, fields: Sequence[str]) -> None:
    """Assert every named field is non-null. The drift detector - see the module docstring.

    A field going null is how Yahoo renaming a key reaches us, and it is invisible to a
    range assertion. Do not relax this to tolerate ``None``; fix the client instead.
    """
    missing = [f for f in fields if getattr(model, f) is None]
    assert not missing, (
        f"{type(model).__name__} fields unexpectedly null: {missing}. Yahoo most likely "
        f"renamed or dropped the underlying key(s) - fix the client's mapping rather than "
        f"relaxing this assertion."
    )


def iso_dates_descending(dates: Sequence[str]) -> bool:
    """True when ISO date strings are strictly newest-first (lexicographic == chronological)."""
    return all(a > b for a, b in zip(dates, dates[1:], strict=False))
