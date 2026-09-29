"""Live contract: get_quote and get_price_history."""

import datetime

import pytest

from tests.live.conftest import (
    AAPL,
    BTC,
    UNKNOWN,
    Layer,
    assert_distinct_intraday_timestamps,
    require_present,
)

#: Every quote field Yahoo populates for a large US listing.
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
    # Widened by 10%: a session setting a new 52-week extreme can run ahead of the
    # year_high/year_low Yahoo reports, and that lag is not a contract violation.
    assert quote.year_low * 0.9 <= quote.price <= quote.year_high * 1.1, (
        f"price {quote.price} is outside the 52-week range "
        f"[{quote.year_low}, {quote.year_high}] by more than the tolerated lag"
    )
    assert quote.currency == "USD"
    assert quote.market_cap is not None and quote.market_cap > 1e11
    assert quote.volume is not None and quote.volume > 0

    # change/change_percent are ours, not Yahoo's: this checks our arithmetic on live
    # inputs, and that the result is a percent rather than a fraction.
    assert quote.change == pytest.approx(quote.price - quote.previous_close)
    assert quote.change_percent == pytest.approx(
        (quote.price - quote.previous_close) / quote.previous_close * 100.0
    )
    assert -50 < quote.change_percent < 50


async def test_quote_crypto_shape_and_units(layer: Layer) -> None:
    """A non-equity instrument still has to satisfy the quote contract."""
    result = await layer.call("get_quote", tickers=[BTC])

    assert result.errors == []
    quote = result.quotes[0]
    assert quote.symbol == BTC
    # Not market_cap: Yahoo reports none for crypto, which is why the field is optional.
    require_present(quote, ("price", "currency", "previous_close", "day_high", "day_low", "volume"))
    assert quote.price > 0
    assert quote.currency == "USD"
    assert quote.volume is not None and quote.volume > 0


async def test_quote_batch_is_partial_and_keyed_by_symbol(layer: Layer) -> None:
    """One bad ticker reports itself in errors without discarding the others.

    Read by symbol, never by position: a failed ticker is absent from `quotes`, so every
    later position shifts. Both the tool description and the glossary warn about this.
    """
    result = await layer.call("get_quote", tickers=[AAPL, UNKNOWN, BTC])

    by_symbol = {q.symbol: q for q in result.quotes}
    assert set(by_symbol) == {AAPL, BTC}, f"expected AAPL and BTC-USD, got {list(by_symbol)}"
    assert by_symbol[AAPL].price > 0
    assert by_symbol[BTC].price > 0

    assert len(result.errors) == 1
    failure = result.errors[0]
    assert failure.symbol == UNKNOWN
    # The message must tell the model that retrying will not help.
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
    # The cap is configurable, so assert the relationship rather than the number.
    assert history.summary.bars > len(history.bars)
    assert history.summary.start_date < history.bars[0].date, (
        "the summary must start before the first returned bar when truncated"
    )
    assert history.summary.end_date == history.bars[-1].date


async def test_price_history_intraday_timestamps_are_distinct(layer: Layer) -> None:
    """BTC-USD trades 24/7, so this holds overnight, at a weekend and on a holiday."""
    history = await layer.call("get_price_history", ticker=BTC, period="5d", interval="5m")

    assert len(history.bars) > 1
    assert_distinct_intraday_timestamps([b.date for b in history.bars])


async def test_price_history_intraday_equity(layer: Layer) -> None:
    """The same contract on an exchange-traded name, whose sessions have gaps."""
    try:
        history = await layer.call("get_price_history", ticker=AAPL, period="5d", interval="5m")
    except layer.error_type as exc:  # a closed-session window can legitimately be empty
        pytest.skip(f"no intraday AAPL bars available right now: {exc}")

    dates = [b.date for b in history.bars]
    assert_distinct_intraday_timestamps(dates)
    for date in dates:
        # Intraday bars sit inside the session, never at the midnight index daily bars use.
        parsed = datetime.datetime.fromisoformat(date)
        assert (parsed.hour, parsed.minute) != (0, 0), f"midnight intraday bar: {date}"
