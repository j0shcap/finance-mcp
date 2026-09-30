"""The market-data findings from the 2026-09-27 review, re-proved on the installed server.

Live as well as e2e: these hit Yahoo, so they run with `make test-live` (nightly), not with
`make e2e` on every PR. tests/live/ already asserts the same contracts in-process; this
file adds the one thing it cannot show - that the shipped wheel, speaking stdio, still
honours them. Throttling is retried and then skipped exactly as in tests/live/, through
the same Layer.
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from finance_mcp.server import build_default_client
from tests.e2e.conftest import Server
from tests.live.conftest import (
    AAPL,
    BTC,
    SAP,
    UNKNOWN,
    Layer,
    assert_distinct_intraday_timestamps,
)
from tests.live.test_live_derived import ANNUALIZED_FIELDS

pytestmark = pytest.mark.live


@pytest_asyncio.fixture(loop_scope="session")
async def stdio(default_server: Server) -> AsyncIterator[Layer]:
    """The live suite's Layer, speaking to the installed server over stdio.

    Layer calls tools through whatever MCP client it is given; the direct-layer client it
    also takes is never used once an MCP client is passed.
    """
    yield Layer(build_default_client(), default_server.client)


async def test_crypto_annualized_return_matches_total_return_over_one_year(
    stdio: Layer,
) -> None:
    """Over ~1y of a 24/7 calendar, annualizing must be close to a no-op.

    The finding: BTC-USD was annualized on a 252-day calendar, inflating the figure.
    """
    stats = await stdio.call("analyze_performance", ticker=BTC, period="1y")

    assert stats.annualized_return_percent == pytest.approx(stats.total_return_percent, rel=0.05)


async def test_a_five_day_window_is_not_annualized(stdio: Layer) -> None:
    stats = await stdio.call("analyze_performance", ticker=AAPL, period="5d")

    for field in ANNUALIZED_FIELDS:
        assert getattr(stats, field) is None, f"{field} must be null on a 5-day window"


async def test_intraday_bars_have_distinct_timestamps(stdio: Layer) -> None:
    history = await stdio.call("get_price_history", ticker=BTC, period="5d", interval="5m")

    assert len(history.bars) > 1
    assert_distinct_intraday_timestamps([b.date for b in history.bars])


async def test_sap_statements_report_eur(stdio: Layer) -> None:
    statement = await stdio.call("get_financials", ticker=SAP, statement="income")

    assert statement.currency == "EUR"


async def test_unknown_line_items_are_reported(stdio: Layer) -> None:
    statement = await stdio.call(
        "get_financials", ticker=AAPL, statement="income", line_items=["Total Revenue", "Revenue"]
    )

    assert statement.missing_line_items == ["Revenue"]
    assert "Total Revenue" in statement.line_items


async def test_get_quote_returns_partial_results(stdio: Layer) -> None:
    result = await stdio.call("get_quote", tickers=[AAPL, UNKNOWN])

    assert [q.symbol for q in result.quotes] == [AAPL]
    assert [e.symbol for e in result.errors] == [UNKNOWN]
    assert "invalid or delisted" in result.errors[0].error


async def test_an_unknown_symbol_still_reads_as_one(stdio: Layer) -> None:
    """The other half of the network-failure fix (tests/e2e/test_stdio_network_failure.py).

    With yfinance's exceptions unhidden, Yahoo's 404 for an unknown symbol raises out of the
    history fetch instead of arriving as an empty frame; it must still be classified as a
    missing symbol, not as an outage.
    """
    async with stdio.expect_error("No price history"):
        await stdio.call("get_price_history", ticker=UNKNOWN)
