"""YFinanceClient.get_price_history: bar parsing, truncation and bar timestamps."""

import math
from typing import get_args

import pytest

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    HistoryInterval,
)
from finance_mcp.data.yfinance_client import (
    _INTRADAY_INTERVALS,
    YFinanceClient,
)
from tests.fakes import (
    FakeClock,
    fake_ticker_factory,
    make_client,
    make_history_df,
    make_intraday_df,
)


def test_get_price_history_parses_bars_and_summary() -> None:
    df = make_history_df([100.0, 101.0, 102.0, 103.0])
    client = make_client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.bars == 4
    assert hist.summary.start_close == 100.0
    assert hist.summary.end_close == 103.0
    assert hist.summary.total_return_percent == pytest.approx(3.0)
    assert hist.summary.period_high == 104.0
    assert hist.bars[-1].close == 103.0
    assert hist.truncated is False


def test_get_price_history_empty_raises_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df([])))
    with pytest.raises(SymbolNotFound):
        client.get_price_history("BADSYM", period="1mo", interval="1d")


def test_get_price_history_truncates_to_max_bars() -> None:
    df = make_history_df([float(i) for i in range(1, 11)])
    client = YFinanceClient(
        ticker_factory=fake_ticker_factory(history_df=df),
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        max_bars=5,
    )
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert len(hist.bars) == 5
    assert hist.truncated is True
    assert hist.summary.bars == 10


def test_get_price_history_drops_nan_rows() -> None:
    df = make_history_df([100.0, 101.0, 102.0])
    df.loc[df.index[1], "Close"] = float("nan")
    client = make_client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.bars == 2
    assert all(math.isfinite(b.close) for b in hist.bars)


def test_get_price_history_all_nan_raises() -> None:
    df = make_history_df([100.0])
    df.loc[df.index[0], "Close"] = float("nan")
    client = make_client(factory=fake_ticker_factory(history_df=df))
    with pytest.raises(SymbolNotFound):
        client.get_price_history("AAPL", period="1mo", interval="1d")


def test_get_price_history_zero_start_close_no_crash() -> None:
    df = make_history_df([0.0, 5.0])
    client = make_client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.total_return_percent == 0.0


def test_get_price_history_drops_inf_rows() -> None:
    df = make_history_df([100.0, 101.0, 102.0])
    df.loc[df.index[1], "Close"] = float("inf")
    client = make_client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.bars == 2


def test_get_price_history_all_inf_raises() -> None:
    df = make_history_df([100.0])
    df.loc[df.index[0], "Close"] = float("inf")
    client = make_client(factory=fake_ticker_factory(history_df=df))
    with pytest.raises(SymbolNotFound):
        client.get_price_history("AAPL", period="1mo", interval="1d")


def test_get_price_history_parse_error_becomes_data_unavailable() -> None:
    df = make_history_df([100.0, 101.0]).drop(columns=["Volume"])
    client = make_client(factory=fake_ticker_factory(history_df=df))
    with pytest.raises(DataUnavailable) as exc:
        client.get_price_history("AAPL", period="1mo", interval="1d")
    assert "AAPL" in str(exc.value)


def test_get_price_history_caches_and_keys_on_interval() -> None:
    calls = {"n": 0}
    df = make_history_df([100.0, 101.0])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(history_df=df)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_price_history("AAPL", "1mo", "1d")
    client.get_price_history("AAPL", "1mo", "1d")
    assert calls["n"] == 1
    client.get_price_history("AAPL", "1mo", "1wk")  # different interval -> distinct key
    assert calls["n"] == 2
    clock.advance(301.0)
    client.get_price_history("AAPL", "1mo", "1d")  # expired -> refetch
    assert calls["n"] == 3


def test_get_price_history_single_bar() -> None:
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df([100.0])))
    h = client.get_price_history("AAPL", "1d", "1d")
    assert h.summary.bars == 1 and h.summary.total_return_percent == 0.0
    assert h.summary.start_date == h.summary.end_date and h.truncated is False


# --- intraday bars keep their time (item 1) ---


def test_intraday_bars_carry_a_full_timestamp_with_utc_offset() -> None:
    df = make_intraday_df([100.0, 101.0, 102.0])
    client = make_client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1d", interval="5m")
    assert [b.date for b in hist.bars] == [
        "2026-09-25T09:30:00-04:00",
        "2026-09-25T09:35:00-04:00",
        "2026-09-25T09:40:00-04:00",
    ]
    assert hist.summary.start_date == "2026-09-25T09:30:00-04:00"
    assert hist.summary.end_date == "2026-09-25T09:40:00-04:00"


@pytest.mark.parametrize("interval", ["1m", "5m", "15m", "30m", "1h"])
def test_every_intraday_interval_emits_distinct_timestamps(interval: str) -> None:
    df = make_intraday_df([100.0, 101.0])
    client = make_client(factory=fake_ticker_factory(history_df=df))
    dates = [b.date for b in client.get_price_history("AAPL", "1d", interval).bars]
    assert len(set(dates)) == 2
    assert all("T" in d and d.endswith("-04:00") for d in dates)


@pytest.mark.parametrize("interval", ["1d", "1wk", "1mo"])
def test_daily_and_longer_bars_stay_date_only(interval: str) -> None:
    # Yahoo's daily index is midnight in the EXCHANGE's timezone; emitting a timestamp (or
    # converting to UTC) would either lie about the time or shift the calendar date.
    df = make_history_df([100.0, 101.0], start="2026-09-24", tz="America/New_York")
    client = make_client(factory=fake_ticker_factory(history_df=df))
    dates = [b.date for b in client.get_price_history("AAPL", "1mo", interval).bars]
    assert dates == ["2026-09-24", "2026-09-25"]


def test_intraday_intervals_are_the_non_daily_history_intervals() -> None:
    # Pins the two sets against HistoryInterval so a newly supported interval cannot
    # silently default to date-only formatting.
    assert set(get_args(HistoryInterval)) - {"1d", "1wk", "1mo"} == _INTRADAY_INTERVALS
