"""YFinanceClient.analyze_performance, and the bars cache it shares with get_price_history."""

import math
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from finance_mcp.data import analytics
from finance_mcp.data.errors import DataUnavailable, InvalidInput, SymbolNotFound
from finance_mcp.data.models import (
    PerformanceStats,
)
from finance_mcp.data.yfinance_client import (
    MAX_CACHEABLE_BARS,
)
from tests.fakes import (
    FakeClock,
    counting,
    fake_ticker_factory,
    make_client,
    make_history_df,
)


def test_analyze_performance_computes_stats() -> None:
    closes = [100.0 + i for i in range(120)]  # 120 calendar days -> past the annualization gate
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    p = client.analyze_performance("AAPL", "6mo")
    assert isinstance(p, PerformanceStats)
    assert p.symbol == "AAPL" and p.period == "6mo" and p.bars == 120
    assert p.total_return_percent == pytest.approx(analytics.total_return(closes))
    assert p.periods_per_year is not None
    assert p.annualized_volatility_percent == pytest.approx(
        analytics.annualized_volatility(closes, p.periods_per_year)
    )
    assert p.max_drawdown_percent == pytest.approx(analytics.max_drawdown(closes))
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))
    assert p.sma_200 is None  # < 200 bars
    assert p.start_date == "2024-01-01" and p.end_date == "2024-04-29"


def test_analyze_performance_reports_risk_adjusted_stats() -> None:
    closes = [100.0 + i for i in range(120)]  # 120 calendar days: past the annualization gate
    p = make_client(
        factory=fake_ticker_factory(history_df=make_history_df(closes))
    ).analyze_performance("AAPL", "6mo", risk_free_rate=0.0)
    assert p.periods_per_year is not None
    assert p.risk_free_rate == 0.0
    assert p.sharpe_ratio == pytest.approx(analytics.sharpe_ratio(closes, p.periods_per_year, 0.0))
    assert p.sortino_ratio is None  # a monotonic rise never fell below the target
    assert p.downside_deviation_percent == pytest.approx(0.0)
    assert p.calmar_ratio is None  # ... and never drew down, so Calmar is undefined


def test_analyze_performance_risk_free_rate_is_echoed_and_applied() -> None:
    closes = [100.0 + (i % 7) - (i % 3) + i * 0.2 for i in range(120)]
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    raw = client.analyze_performance("AAPL", "6mo", risk_free_rate=0.0)
    excess = client.analyze_performance("AAPL", "6mo", risk_free_rate=0.05)
    assert excess.risk_free_rate == 0.05
    assert raw.sharpe_ratio is not None and excess.sharpe_ratio is not None
    assert excess.sharpe_ratio < raw.sharpe_ratio  # a positive hurdle lowers the ratio


def test_analyze_performance_computes_calmar_from_its_own_figures() -> None:
    closes = [100.0, 130.0, 90.0] + [100.0 + i for i in range(117)]
    p = make_client(
        factory=fake_ticker_factory(history_df=make_history_df(closes))
    ).analyze_performance("AAPL", "6mo")
    assert p.annualized_return_percent is not None
    assert p.calmar_ratio == pytest.approx(
        p.annualized_return_percent / abs(p.max_drawdown_percent)
    )


def test_analyze_performance_nulls_risk_adjusted_stats_on_a_short_window() -> None:
    p = make_client(
        factory=fake_ticker_factory(history_df=make_history_df([100.0, 101.0, 99.0, 103.0]))
    ).analyze_performance("AAPL", "5d", risk_free_rate=0.05)
    assert p.risk_free_rate == 0.05  # echoed even when unused, so the caller can see it
    assert p.sharpe_ratio is None
    assert p.sortino_ratio is None
    assert p.downside_deviation_percent is None
    assert p.calmar_ratio is None
    assert p.total_return_percent == pytest.approx(3.0)


def test_analyze_performance_rejects_a_risk_free_rate_of_minus_one() -> None:
    # The tool bounds this, but the data layer is reachable directly.
    client = make_client(
        factory=fake_ticker_factory(history_df=make_history_df([100.0 + i for i in range(120)]))
    )
    with pytest.raises(InvalidInput):
        client.analyze_performance("AAPL", "6mo", risk_free_rate=-1.0)


def _perf_stats(closes: list[float], **df_kw: Any) -> PerformanceStats:
    df = make_history_df(closes, **df_kw)
    return make_client(factory=fake_ticker_factory(history_df=df)).analyze_performance("X", "1y")


def test_analyze_performance_one_year_annualized_equals_total_return() -> None:
    # 366 daily bars, as for BTC-USD. The tolerance is not slop: 365 elapsed days is
    # 0.99932 years, so the figures differ in the 4th significant digit. Exact equality at
    # years=1.0 is pinned in test_analytics_logic.py.
    closes = [100.0 * (0.7702 ** (i / 365)) for i in range(366)]
    p = _perf_stats(closes)
    assert p.bars == 366
    assert p.start_date == "2024-01-01" and p.end_date == "2024-12-31"
    assert p.total_return_percent == pytest.approx(-22.98, rel=1e-3)
    assert p.annualized_return_percent is not None
    assert p.annualized_return_percent == pytest.approx(p.total_return_percent, rel=2e-3)


def test_analyze_performance_infers_seven_day_calendar() -> None:
    p = _perf_stats([100.0 + i for i in range(366)])
    assert p.periods_per_year == pytest.approx(365.0, rel=0.01)


def test_analyze_performance_infers_weekday_calendar() -> None:
    # A weekday-only series infers 5/7 of the calendar = ~261/yr. Real exchanges print ~252
    # because of ~9 market holidays, which pandas freq="B" does not model -- the point of the
    # assertion is that the inferred rate tracks the weekday calendar and is nowhere near 365.
    p = _perf_stats([100.0 + i for i in range(261)], freq="B")
    assert p.periods_per_year == pytest.approx(365.25 * 5 / 7, rel=0.02)
    assert p.periods_per_year is not None and p.periods_per_year < 300.0


def test_analyze_performance_annualized_return_ignores_bar_count() -> None:
    # The same calendar move over the same year, printed at two different bar rates.
    seven_day = _perf_stats([100.0] * 365 + [80.0])
    weekday = _perf_stats([100.0] * 260 + [80.0], freq="B")
    assert seven_day.annualized_return_percent is not None
    assert weekday.annualized_return_percent is not None
    assert seven_day.annualized_return_percent == pytest.approx(
        weekday.annualized_return_percent, rel=0.02
    )


def test_analyze_performance_crypto_volatility_is_not_understated() -> None:
    # The same closes on a 24/7 and a weekday calendar: identical per-observation dispersion,
    # but the 24/7 series observes more returns a year, so it must annualize higher by the
    # square root of the ratio of the two inferred rates.
    closes = [100.0 + (5.0 if i % 2 else 0.0) for i in range(261)]
    seven_day = _perf_stats(closes)
    weekday = _perf_stats(closes, freq="B")
    assert seven_day.annualized_volatility_percent is not None
    assert weekday.annualized_volatility_percent is not None
    assert seven_day.periods_per_year is not None and weekday.periods_per_year is not None
    assert seven_day.periods_per_year > weekday.periods_per_year
    ratio = seven_day.annualized_volatility_percent / weekday.annualized_volatility_percent
    expected = math.sqrt(seven_day.periods_per_year / weekday.periods_per_year)
    assert ratio == pytest.approx(expected, rel=1e-6)
    assert ratio > 1.0  # the 24/7 instrument is not flattened to the equity convention


def test_analyze_performance_short_window_nulls_annualized_fields() -> None:
    # A few days' move must not become a yearly figure.
    p = _perf_stats([100.0, 101.0, 102.0, 103.0, 104.0])
    assert p.annualized_return_percent is None
    assert p.annualized_volatility_percent is None
    assert p.periods_per_year is None
    # Everything that does not annualize still reports.
    assert p.total_return_percent == pytest.approx(4.0)
    assert p.max_drawdown_percent == pytest.approx(0.0)


def test_analyze_performance_annualizes_at_the_threshold() -> None:
    # 86 consecutive daily bars => exactly 85 elapsed days => on the gate, so it annualizes.
    p = _perf_stats([100.0 + i for i in range(86)])
    assert p.annualized_return_percent is not None
    assert p.annualized_volatility_percent is not None
    assert p.periods_per_year is not None


def test_analyze_performance_does_not_annualize_below_the_threshold() -> None:
    # 85 bars => 84 elapsed days => one day short of the gate.
    p = _perf_stats([100.0 + i for i in range(85)])
    assert p.annualized_return_percent is None
    assert p.annualized_volatility_percent is None
    assert p.periods_per_year is None


def test_analyze_performance_annualizes_the_shortest_three_month_window() -> None:
    # period="3mo" spans 87-95 elapsed days depending on the call date; 88 daily bars is
    # the shortest.
    p = _perf_stats([100.0 + i for i in range(88)])
    assert p.annualized_return_percent is not None
    assert p.annualized_volatility_percent is not None
    assert p.periods_per_year is not None


def test_analyze_performance_still_nulls_a_one_month_window() -> None:
    p = _perf_stats([100.0 + i for i in range(31)])
    assert p.annualized_return_percent is None
    assert p.periods_per_year is None


def _counting_history(closes: list[float]) -> tuple[Callable[[str], Any], list[str]]:
    return counting(fake_ticker_factory(history_df=make_history_df(closes)))


def test_oversized_bar_lists_are_not_retained_in_the_cache() -> None:
    # An entry-count LRU does not bound bytes: a period="max" daily history is ~11.5k bars at
    # ~787 B each (~9 MB), so 256 such entries would retain gigabytes. Lists past the limit
    # are still returned in full -- they are just not kept.
    closes = [100.0 + i for i in range(MAX_CACHEABLE_BARS + 1)]
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    bars = client._all_bars("AAPL", "max", "1d")
    assert len(bars) == MAX_CACHEABLE_BARS + 1  # returned whole
    assert ("bars", "AAPL", "max", "1d") not in client._cache


def test_bar_lists_at_the_limit_are_retained() -> None:
    closes = [100.0 + i for i in range(MAX_CACHEABLE_BARS)]
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    client._all_bars("AAPL", "10y", "1d")
    assert ("bars", "AAPL", "10y", "1d") in client._cache


def test_repeated_analysis_of_an_oversized_history_does_not_refetch() -> None:
    # Bars past the limit are not cached, so without a cached PerformanceStats every call
    # would go back to the network. The derived result is tiny; cache that instead.
    factory, calls = _counting_history([100.0 + i for i in range(MAX_CACHEABLE_BARS + 1)])
    client = make_client(factory)
    # An explicit rate keeps the count about the asset's bars, not the T-bill history.
    first = client.analyze_performance("AAPL", "max", 0.0)
    second = client.analyze_performance("AAPL", "max", 0.0)
    assert len(calls) == 1
    assert first.total_return_percent == second.total_return_percent


def test_oversized_price_history_still_caches_its_derived_view() -> None:
    factory, calls = _counting_history([100.0 + i for i in range(MAX_CACHEABLE_BARS + 1)])
    client = make_client(factory)
    client.get_price_history("AAPL", period="max", interval="1d")
    client.get_price_history("AAPL", period="max", interval="1d")
    assert len(calls) == 1


@pytest.mark.parametrize("history_first", [True, False])
def test_analyze_performance_and_get_price_history_share_one_fetch(history_first: bool) -> None:
    factory, calls = _counting_history([100.0 + i for i in range(120)])
    client = make_client(factory)
    if history_first:
        client.get_price_history("AAPL", period="6mo", interval="1d")
    client.analyze_performance("AAPL", "6mo", 0.0)
    client.get_price_history("AAPL", period="6mo", interval="1d")
    assert len(calls) == 1


def test_analyze_performance_smas_survive_a_short_window() -> None:
    # SMAs do not annualize, so the annualization gate does not blank them.
    closes = [100.0 + i for i in range(60)]  # 59 elapsed days, under the gate
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    p = client.analyze_performance("AAPL", "3mo")
    assert p.annualized_return_percent is None
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))
    assert p.sma_200 is None


def test_analyze_performance_too_few_bars_raises() -> None:
    client = make_client(factory=fake_ticker_factory(history_df=make_history_df([100.0])))
    with pytest.raises(DataUnavailable):
        client.analyze_performance("AAPL", "1d")


def test_analyze_performance_invalid_symbol_raises() -> None:
    client = make_client(factory=fake_ticker_factory(history_df=pd.DataFrame()))
    with pytest.raises(SymbolNotFound):
        client.analyze_performance("BAD", "1y")


def test_analyze_performance_caches_within_ttl_and_keys_on_period() -> None:
    factory, calls = _counting_history([100.0, 110.0, 99.0])
    clock = FakeClock()
    client = make_client(factory, clock=clock, history_ttl=300.0)
    client.analyze_performance("AAPL", "1mo", 0.0)
    client.analyze_performance("AAPL", "1mo", 0.0)
    assert len(calls) == 1
    client.analyze_performance("AAPL", "1y", 0.0)
    assert len(calls) == 2
    clock.advance(301.0)
    client.analyze_performance("AAPL", "1mo", 0.0)
    assert len(calls) == 3


def test_analyze_performance_cache_keys_on_the_risk_free_rate() -> None:
    df = make_history_df([100.0 + i for i in range(300)])
    client = make_client(factory=fake_ticker_factory(history_df=df))

    raw = client.analyze_performance("AAPL", "1y", 0.0)
    excess = client.analyze_performance("AAPL", "1y", 0.05)

    assert raw.risk_free_rate == 0.0
    assert excess.risk_free_rate == 0.05
    assert raw.sharpe_ratio is not None and excess.sharpe_ratio is not None
    assert excess.sharpe_ratio < raw.sharpe_ratio
    # And the first rate is still served from cache rather than recomputed differently.
    assert client.analyze_performance("AAPL", "1y", 0.0).sharpe_ratio == raw.sharpe_ratio


def test_analyze_performance_sma_200_populated() -> None:
    closes = [100.0 + i for i in range(250)]
    p = make_client(
        factory=fake_ticker_factory(history_df=make_history_df(closes))
    ).analyze_performance("AAPL", "1y")
    assert p.sma_200 == pytest.approx(analytics.sma(closes, 200)) and p.sma_200 is not None
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))
