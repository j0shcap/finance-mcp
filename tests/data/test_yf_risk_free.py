"""The risk-free rate the analytics use when the caller passes none: the 13-week T-bill.

When risk_free_rate is omitted, analyze_performance, compare_to_benchmark and
compare_tickers average Yahoo's ^IRX over the dates actually measured, converted from its
bank-discount quote to an effective annual rate. When that cannot be done, the rate is
reported unavailable and the figures that need it are null - never silently computed at 0.
"""

import math
import statistics
import threading
from typing import Any

import pandas as pd
import pytest
from yfinance.exceptions import YFException

from finance_mcp.data import analytics
from finance_mcp.data.yfinance_client import TREASURY_BILL_SYMBOL
from tests.fakes import counting, fake_multi_ticker_factory, make_client, make_history_df

# 200 weekdays from Monday 2024-01-01 to Friday 2024-10-04: past the annualization gate.
DAYS = 200
START = "2024-01-01"


def _walk(seed: float, step: float, wobble: float, n: int = DAYS) -> list[float]:
    return [seed + step * i + wobble * math.sin(i) for i in range(n)]


def _weekdays(closes: list[float], start: str = START) -> pd.DataFrame:
    return make_history_df(closes, start=start, freq="B")


def _bills(percent: float | list[float], start: str = START, n: int = DAYS) -> pd.DataFrame:
    """^IRX closes: a percent discount yield per weekday."""
    closes = percent if isinstance(percent, list) else [percent] * n
    return _weekdays(closes, start=start)


ASSET = _walk(100.0, 0.30, 2.0)
BENCH = _walk(400.0, 0.80, 4.0)
RATE_4_03 = analytics.treasury_bill_effective_rate(4.03)


def _factory(**per_symbol: dict[str, Any]) -> Any:
    base: dict[str, dict[str, Any]] = {
        "AAPL": {"history_df": _weekdays(ASSET)},
        TREASURY_BILL_SYMBOL: {"history_df": _bills(4.03)},
    }
    base.update(per_symbol)
    return fake_multi_ticker_factory(base)


def _bill_override(**kwargs: Any) -> dict[str, dict[str, Any]]:
    return {TREASURY_BILL_SYMBOL: kwargs}


# --- analyze_performance -------------------------------------------------------------


def test_omitted_rate_is_the_treasury_bill_yield_over_the_window() -> None:
    p = make_client(_factory()).analyze_performance("AAPL", "1y")
    assert p.risk_free_rate == pytest.approx(RATE_4_03)
    assert p.risk_free_rate_source == "treasury_bill"
    assert p.risk_free_rate_note is None
    assert p.periods_per_year is not None
    assert p.sharpe_ratio == pytest.approx(
        analytics.sharpe_ratio(ASSET, p.periods_per_year, RATE_4_03)
    )
    assert p.sortino_ratio == pytest.approx(
        analytics.sortino_ratio(ASSET, p.periods_per_year, RATE_4_03)
    )


def test_the_average_covers_only_the_measured_dates() -> None:
    # Ten weekdays at 9% before the asset's first bar must not leak into its rate.
    before = _weekdays([9.0] * 10, start="2023-12-18")
    inside = [2.0 + 0.01 * i for i in range(DAYS)]
    bills = pd.concat([before, _bills(inside)])
    p = make_client(_factory(**_bill_override(history_df=bills))).analyze_performance("AAPL", "1y")
    expected = statistics.fmean(analytics.treasury_bill_effective_rate(v) for v in inside)
    assert p.risk_free_rate == pytest.approx(expected)


def test_a_bond_market_holiday_at_either_edge_is_not_a_coverage_gap() -> None:
    # Bills starting three weekdays late and ending three early still cover the window.
    bills = _bills(4.03, start="2024-01-04", n=DAYS - 6)
    p = make_client(_factory(**_bill_override(history_df=bills))).analyze_performance("AAPL", "1y")
    assert p.risk_free_rate_source == "treasury_bill"


@pytest.mark.parametrize(
    ("bills", "named_date"),
    [
        (_bills(4.03, start="2024-03-01", n=150), "2024-03-01"),  # starts too late
        (_bills(4.03, n=60), "2024-03-22"),  # ends too early
        (_bills(4.03, start="2025-01-06", n=20), "2025-01-06"),  # no bill inside the window
    ],
)
def test_bills_that_do_not_cover_the_window_leave_the_rate_unavailable(
    bills: pd.DataFrame, named_date: str
) -> None:
    p = make_client(_factory(**_bill_override(history_df=bills))).analyze_performance("AAPL", "1y")
    assert p.risk_free_rate is None
    assert p.risk_free_rate_source == "unavailable"
    assert p.risk_free_rate_note is not None
    assert named_date in p.risk_free_rate_note and "2024-01-01" in p.risk_free_rate_note
    assert "risk_free_rate" in p.risk_free_rate_note  # says how to proceed


def test_without_a_rate_the_rate_dependent_figures_are_null_and_the_rest_stand() -> None:
    factory = _factory(**_bill_override(history_error=YFException("bills down")))
    p = make_client(factory).analyze_performance("AAPL", "1y")
    assert p.risk_free_rate_source == "unavailable"
    assert p.risk_free_rate_note is not None and "bills down" in p.risk_free_rate_note
    assert p.sharpe_ratio is None
    assert p.sortino_ratio is None
    assert p.downside_deviation_percent is None
    # None of these depend on the rate.
    assert p.annualized_return_percent is not None
    assert p.annualized_volatility_percent is not None
    assert p.calmar_ratio is not None
    assert p.total_return_percent == pytest.approx(analytics.total_return(ASSET))


def test_bill_data_implying_a_non_positive_price_is_unavailable_not_an_error() -> None:
    factory = _factory(**_bill_override(history_df=_bills(500.0)))
    p = make_client(factory).analyze_performance("AAPL", "1y")
    assert p.risk_free_rate_source == "unavailable"
    assert p.risk_free_rate_note is not None and "^IRX" in p.risk_free_rate_note


def test_an_unavailable_rate_is_not_cached() -> None:
    factory, calls = counting(_factory(**_bill_override(history_error=YFException("down"))))
    client = make_client(factory)
    client.analyze_performance("AAPL", "1y")
    client.analyze_performance("AAPL", "1y")
    assert calls.count(TREASURY_BILL_SYMBOL) == 2


def test_a_resolved_default_rate_is_cached() -> None:
    factory, calls = counting(_factory())
    client = make_client(factory)
    first = client.analyze_performance("AAPL", "1y")
    second = client.analyze_performance("AAPL", "1y")
    assert calls.count(TREASURY_BILL_SYMBOL) == 1
    assert second.risk_free_rate == first.risk_free_rate


@pytest.mark.parametrize("rate", [0.0, 0.05])
def test_a_given_rate_is_used_as_is_and_skips_the_bill_fetch(rate: float) -> None:
    factory, calls = counting(_factory())
    p = make_client(factory).analyze_performance("AAPL", "1y", risk_free_rate=rate)
    assert p.risk_free_rate == rate
    assert p.risk_free_rate_source == "caller"
    assert p.risk_free_rate_note is None
    assert TREASURY_BILL_SYMBOL not in calls


def test_the_default_and_a_given_rate_are_cached_apart() -> None:
    client = make_client(_factory())
    default = client.analyze_performance("AAPL", "1y")
    raw = client.analyze_performance("AAPL", "1y", risk_free_rate=0.0)
    assert default.risk_free_rate_source == "treasury_bill"
    assert raw.risk_free_rate == 0.0
    assert default.sharpe_ratio is not None and raw.sharpe_ratio is not None
    assert default.sharpe_ratio < raw.sharpe_ratio


def test_the_asset_and_the_bills_are_fetched_concurrently() -> None:
    gate = threading.Barrier(2, timeout=10)
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": _weekdays(ASSET)},
            TREASURY_BILL_SYMBOL: {"history_df": _bills(4.03)},
        },
        gate=gate,
    )
    assert make_client(factory).analyze_performance("AAPL", "1y").risk_free_rate is not None


# --- compare_to_benchmark ------------------------------------------------------------


def _benchmark_factory(**per_symbol: dict[str, Any]) -> Any:
    return _factory(SPY={"history_df": _weekdays(BENCH)}, **per_symbol)


def test_benchmark_alpha_uses_the_treasury_bill_rate_over_the_overlap() -> None:
    result = make_client(_benchmark_factory()).compare_to_benchmark("AAPL", "SPY", "1y")
    assert result.risk_free_rate == pytest.approx(RATE_4_03)
    assert result.risk_free_rate_source == "treasury_bill"
    assert result.beta is not None
    assert result.annualized_return_percent is not None
    assert result.benchmark_annualized_return_percent is not None
    assert result.alpha_percent == pytest.approx(
        analytics.jensen_alpha(
            result.annualized_return_percent,
            result.benchmark_annualized_return_percent,
            result.beta,
            RATE_4_03,
        )
    )


def test_benchmark_rate_is_averaged_over_the_overlap_not_the_asset_window() -> None:
    # The benchmark starts 20 weekdays late, so the overlap does too; the 9% bills before
    # it fall outside the measured dates.
    bills = _bills([9.0] * 20 + [3.0] * (DAYS - 20))
    late_bench = _weekdays(BENCH[: DAYS - 20], start="2024-01-29")
    factory = _factory(SPY={"history_df": late_bench}, **_bill_override(history_df=bills))
    result = make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y")
    assert result.start_date == "2024-01-29"
    assert result.risk_free_rate == pytest.approx(analytics.treasury_bill_effective_rate(3.0))


def test_benchmark_without_a_rate_keeps_every_figure_but_alpha() -> None:
    factory = _benchmark_factory(**_bill_override(history_error=YFException("bills down")))
    result = make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y")
    assert result.risk_free_rate is None
    assert result.risk_free_rate_source == "unavailable"
    assert result.risk_free_rate_note is not None
    assert result.alpha_percent is None
    assert result.beta is not None
    assert result.tracking_error_percent is not None
    assert result.information_ratio is not None


def test_benchmark_with_a_given_rate_skips_the_bill_fetch() -> None:
    factory, calls = counting(_benchmark_factory())
    result = make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y", 0.02)
    assert result.risk_free_rate == 0.02
    assert result.risk_free_rate_source == "caller"
    assert TREASURY_BILL_SYMBOL not in calls


def test_benchmark_legs_and_bills_are_fetched_concurrently() -> None:
    gate = threading.Barrier(3, timeout=10)
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": _weekdays(ASSET)},
            "SPY": {"history_df": _weekdays(BENCH)},
            TREASURY_BILL_SYMBOL: {"history_df": _bills(4.03)},
        },
        gate=gate,
    )
    result = make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y")
    assert result.risk_free_rate_source == "treasury_bill"


# --- compare_tickers -----------------------------------------------------------------

METRICS_INFO = {"longName": "Apple Inc.", "currency": "USD", "trailingPE": 30.0}


def _rows_factory(**per_symbol: dict[str, Any]) -> Any:
    return _factory(
        AAPL={"history_df": _weekdays(ASSET), "info": METRICS_INFO},
        MSFT={"history_df": _weekdays(_walk(200.0, 0.5, 3.0)), "info": METRICS_INFO},
        **per_symbol,
    )


def test_compare_tickers_gives_each_row_the_treasury_bill_rate() -> None:
    table = make_client(_rows_factory()).compare_tickers(["AAPL", "MSFT"], "1y")
    assert table.risk_free_rate is None  # the caller passed none
    assert table.risk_free_rate_source == "treasury_bill"
    for row in table.rows:
        assert row.risk_free_rate == pytest.approx(RATE_4_03)
        assert row.risk_free_rate_source == "treasury_bill"
        assert row.sharpe_ratio is not None


def test_compare_tickers_fetches_the_bills_once_for_every_row() -> None:
    factory, calls = counting(_rows_factory())
    make_client(factory).compare_tickers(["AAPL", "MSFT"], "1y")
    assert calls.count(TREASURY_BILL_SYMBOL) == 1


def test_compare_tickers_averages_each_row_over_its_own_dates() -> None:
    # A listing 100 weekdays younger is measured against the bills of its own window.
    bills = _bills([1.0] * 100 + [5.0] * 100)
    young = _weekdays(_walk(50.0, 0.2, 1.0, n=100), start="2024-05-20")
    factory = _rows_factory(
        YOUNG={"history_df": young, "info": METRICS_INFO}, **_bill_override(history_df=bills)
    )
    table = make_client(factory).compare_tickers(["AAPL", "YOUNG"], "1y")
    by_symbol = {row.symbol: row for row in table.rows}
    assert by_symbol["YOUNG"].risk_free_rate == pytest.approx(
        analytics.treasury_bill_effective_rate(5.0)
    )
    assert by_symbol["AAPL"].risk_free_rate == pytest.approx(
        statistics.fmean(
            [analytics.treasury_bill_effective_rate(1.0)] * 100
            + [analytics.treasury_bill_effective_rate(5.0)] * 100
        )
    )


def test_compare_tickers_rows_without_a_rate_still_rank_on_everything_else() -> None:
    factory = _rows_factory(**_bill_override(history_error=YFException("bills down")))
    table = make_client(factory).compare_tickers(["AAPL", "MSFT"], "1y")
    assert table.errors == []
    for row in table.rows:
        assert row.risk_free_rate_source == "unavailable"
        assert row.risk_free_rate_note is not None
        assert row.sharpe_ratio is None and row.sortino_ratio is None
        assert row.calmar_ratio is not None and row.trailing_pe == 30.0


def test_compare_tickers_with_a_given_rate_uses_it_everywhere_without_fetching() -> None:
    factory, calls = counting(_rows_factory())
    table = make_client(factory).compare_tickers(["AAPL", "MSFT"], "1y", risk_free_rate=0.03)
    assert table.risk_free_rate == 0.03
    assert table.risk_free_rate_source == "caller"
    assert all(row.risk_free_rate == 0.03 for row in table.rows)
    assert all(row.risk_free_rate_source == "caller" for row in table.rows)
    assert TREASURY_BILL_SYMBOL not in calls


def test_compare_tickers_with_nothing_to_fetch_does_not_fetch_bills() -> None:
    factory, calls = counting(_rows_factory())
    table = make_client(factory).compare_tickers(["  "], "1y")
    assert table.rows == []
    assert calls == []
