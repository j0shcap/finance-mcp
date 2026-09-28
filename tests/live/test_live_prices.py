"""Live contract: get_quote and get_price_history.

Each test runs at both layers (real YFinanceClient, in-process MCP client) via ``layer``.
See tests/live/conftest.py for why presence and range assertions are both needed.
"""

import datetime

import pytest

from tests.live.conftest import AAPL, BTC, UNKNOWN, Layer, require_present

#: Every quote field Yahoo populates for a large US listing. A null here means the
#: fast_info attribute we read was renamed or dropped.
QUOTE_FIELDS = (
    "price",
    "currency",
    "previous_close",
    "change",
    "change_percent",
    "day_high",
    "day_low",
    "year_high",
    "year_low",
    "market_cap",
    "volume",
)


async def test_quote_aapl_shape_and_units(layer: Layer) -> None:
    result = await layer.call("get_quote", tickers=[AAPL])

    assert result.errors == [], f"unexpected per-ticker errors: {result.errors}"
    assert len(result.quotes) == 1
    quote = result.quotes[0]
    assert quote.symbol == AAPL
    require_present(quote, QUOTE_FIELDS)

    assert quote.price > 0
    assert quote.previous_close is not None and quote.previous_close > 0
    assert quote.day_low is not None and quote.day_high is not None
    assert quote.day_low <= quote.day_high
    assert quote.year_low is not None and quote.year_high is not None
    assert quote.year_low <= quote.year_high
    # Widened by 10%: a session that sets a new 52-week extreme can be ahead of the
    # year_high/year_low Yahoo reports, and that lag is not a contract violation.
    assert quote.year_low * 0.9 <= quote.price <= quote.year_high * 1.1, (
        f"price {quote.price} is outside the 52-week range "
        f"[{quote.year_low}, {quote.year_high}] by more than the tolerated lag"
    )
    # AAPL trades in USD and is a multi-hundred-billion-dollar company. A currency flip or
    # a market cap three orders of magnitude off would mean a unit change, not a price move.
    assert quote.currency == "USD"
    assert quote.market_cap is not None and quote.market_cap > 1e11
    assert quote.volume is not None and quote.volume > 0

    # change/change_percent are computed by the client, so this checks our arithmetic
    # against live inputs rather than Yahoo's.
    assert quote.change == pytest.approx(quote.price - quote.previous_close)
    assert quote.change_percent == pytest.approx(
        (quote.price - quote.previous_close) / quote.previous_close * 100.0
    )
    # ...and pins it as a percent, not a fraction, which is what the field description says.
    assert -50 < quote.change_percent < 50


async def test_quote_crypto_shape_and_units(layer: Layer) -> None:
    """BTC-USD: a non-equity instrument still has to satisfy the quote contract."""
    result = await layer.call("get_quote", tickers=[BTC])

    assert result.errors == []
    quote = result.quotes[0]
    assert quote.symbol == BTC
    # Not market_cap: Yahoo does not report one for crypto, which is exactly why the field
    # is optional on the model. The equity case asserts it instead.
    require_present(quote, ("price", "currency", "previous_close", "day_high", "day_low", "volume"))
    assert quote.price > 0
    assert quote.currency == "USD"
    assert quote.volume is not None and quote.volume > 0


async def test_quote_batch_is_partial_and_keyed_by_symbol(layer: Layer) -> None:
    """One bad ticker reports itself in errors without discarding the others.

    Quotes are read by symbol rather than by position, because a failed ticker is absent
    from ``quotes`` and every later position shifts - the bug the tool description and the
    conventions glossary both warn the model about.
    """
    result = await layer.call("get_quote", tickers=[AAPL, UNKNOWN, BTC])

    by_symbol = {q.symbol: q for q in result.quotes}
    assert set(by_symbol) == {AAPL, BTC}, f"expected AAPL and BTC-USD, got {list(by_symbol)}"
    assert by_symbol[AAPL].price > 0
    assert by_symbol[BTC].price > 0

    assert len(result.errors) == 1
    failure = result.errors[0]
    assert failure.symbol == UNKNOWN
    # The message has to tell the model retrying will not help.
    assert "invalid or delisted" in failure.error.lower(), (
        f"the error should identify an unusable symbol, got: {failure.error}"
    )


async def test_price_history_daily_shape(layer: Layer) -> None:
    history = await layer.call("get_price_history", ticker=AAPL, period="1mo", interval="1d")

    assert history.symbol == AAPL
    assert history.period == "1mo"
    assert history.interval == "1d"
    assert history.bars, "a one-month daily window must contain bars"

    dates = [b.date for b in history.bars]
    # Daily bars cover whole sessions, so they are date-only: a timestamp would imply a
    # trade time that does not exist.
    for date in dates:
        assert len(date) == 10, f"daily bars must be date-only, got {date!r}"
        datetime.date.fromisoformat(date)
    assert dates == sorted(dates), "bars must be chronological"
    assert len(set(dates)) == len(dates), "daily bars must not repeat a session"

    for bar in history.bars:
        assert bar.low <= bar.open <= bar.high, f"open outside the bar's range: {bar}"
        assert bar.low <= bar.close <= bar.high, f"close outside the bar's range: {bar}"
        assert bar.low > 0 and bar.volume >= 0

    summary = history.summary
    assert summary.bars == len(history.bars)
    assert history.truncated is False
    assert summary.start_date == dates[0]
    assert summary.end_date == dates[-1]
    assert summary.period_low <= min(b.low for b in history.bars)
    assert summary.period_high >= max(b.high for b in history.bars)
    assert summary.start_close > 0 and summary.end_close > 0


async def test_price_history_long_window_truncates_but_summarizes_fully(layer: Layer) -> None:
    """A long window caps `bars` while `summary` still covers the whole period."""
    history = await layer.call("get_price_history", ticker=AAPL, period="5y", interval="1d")

    assert history.truncated is True
    # The cap is configurable, so this asserts the relationship rather than the number.
    assert history.summary.bars > len(history.bars)
    assert history.summary.start_date < history.bars[0].date, (
        "the summary must start before the first returned bar when truncated"
    )
    assert history.summary.end_date == history.bars[-1].date


async def test_price_history_intraday_timestamps_are_distinct(layer: Layer) -> None:
    """Intraday bars are moments, so each carries a distinct offset-bearing timestamp.

    BTC-USD rather than AAPL: it trades 24/7, so this holds on a weekend, a holiday, and
    overnight, with no market-hours branch in the test.
    """
    history = await layer.call("get_price_history", ticker=BTC, period="5d", interval="5m")

    assert len(history.bars) > 1
    dates = [b.date for b in history.bars]
    assert len(set(dates)) == len(dates), (
        "intraday bars collapsed to duplicate timestamps - the date-only formatting used "
        "for daily bars has leaked into the intraday path"
    )
    for date in dates:
        assert len(date) > 10, f"intraday bars need a full timestamp, got {date!r}"
        parsed = datetime.datetime.fromisoformat(date)
        assert parsed.tzinfo is not None, f"intraday timestamps must carry a UTC offset: {date}"
    assert dates == sorted(dates)


async def test_price_history_intraday_equity(layer: Layer) -> None:
    """The same intraday contract on an exchange-traded name, where sessions have gaps."""
    try:
        history = await layer.call("get_price_history", ticker=AAPL, period="5d", interval="5m")
    except layer.error_type as exc:  # a closed-session window can legitimately be empty
        pytest.skip(f"no intraday AAPL bars available right now: {exc}")

    dates = [b.date for b in history.bars]
    assert len(set(dates)) == len(dates)
    for date in dates:
        parsed = datetime.datetime.fromisoformat(date)
        assert parsed.tzinfo is not None
        # Intraday bars fall inside the trading session, never at the midnight index that
        # daily bars use.
        assert (parsed.hour, parsed.minute) != (0, 0), f"midnight intraday bar: {date}"
