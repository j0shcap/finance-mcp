"""Mocked-client tests for the benchmark comparison and the multi-ticker table.

Kept out of test_yfinance_client.py, which is already past 2 000 lines.
"""

import math
import threading
from typing import Any

import pytest

from finance_mcp.data import analytics
from finance_mcp.data.errors import DataUnavailable, InvalidInput
from finance_mcp.data.models import BenchmarkComparison
from finance_mcp.data.yfinance_client import YFinanceClient
from tests.conftest import FakeClock, fake_multi_ticker_factory, make_history_df

# 200 consecutive calendar days from 2024-01-01, so both legs clear the 90-day gate.
DAYS = 200


def _client(**kwargs: Any) -> YFinanceClient:
    return YFinanceClient(time_fn=FakeClock(), **kwargs)


def _walk(seed: float, step: float, wobble: float, n: int = DAYS) -> list[float]:
    """A deterministic, non-monotonic series: a drift plus a repeating wobble."""
    return [seed + step * i + wobble * math.sin(i) for i in range(n)]


def _compare(
    asset_closes: list[float],
    bench_closes: list[float],
    *,
    asset_freq: str = "D",
    bench_freq: str = "D",
    risk_free_rate: float = 0.0,
) -> BenchmarkComparison:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(asset_closes, freq=asset_freq)},
            "SPY": {"history_df": make_history_df(bench_closes, freq=bench_freq)},
        }
    )
    return _client(ticker_factory=factory).compare_to_benchmark("AAPL", "SPY", "1y", risk_free_rate)


def test_compare_to_benchmark_reports_the_relative_statistics() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0)
    result = _compare(asset, bench)
    assert result.symbol == "AAPL" and result.benchmark == "SPY" and result.period == "1y"
    assert result.overlapping_observations == DAYS
    assert result.start_date == "2024-01-01"
    assert result.periods_per_year is not None
    assert result.beta == pytest.approx(analytics.beta(asset, bench))
    assert result.correlation == pytest.approx(analytics.correlation(asset, bench))
    assert result.tracking_error_percent == pytest.approx(
        analytics.tracking_error(asset, bench, result.periods_per_year)
    )
    assert result.information_ratio == pytest.approx(
        analytics.information_ratio(asset, bench, result.periods_per_year)
    )


def test_compare_to_benchmark_reports_excess_return_as_a_difference() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0)
    result = _compare(asset, bench)
    assert result.excess_return_percent == pytest.approx(
        result.total_return_percent - result.benchmark_total_return_percent
    )


def test_compare_to_benchmark_alpha_is_computed_from_the_aligned_cagrs() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0)
    result = _compare(asset, bench, risk_free_rate=0.04)
    assert result.risk_free_rate == 0.04
    assert result.beta is not None
    assert result.annualized_return_percent is not None
    assert result.benchmark_annualized_return_percent is not None
    assert result.alpha_percent == pytest.approx(
        analytics.jensen_alpha(
            result.annualized_return_percent,
            result.benchmark_annualized_return_percent,
            result.beta,
            0.04,
        )
    )


def test_compare_to_benchmark_defaults_the_risk_free_rate_to_zero() -> None:
    result = _compare(_walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0))
    assert result.risk_free_rate == 0.0


def test_compare_to_benchmark_aligns_a_seven_day_asset_to_a_weekday_benchmark() -> None:
    """BTC-USD vs SPY: the overlap is the benchmark's weekdays, so periods_per_year is read
    off a 5-day calendar -- not the 365 the crypto leg alone implies.

    The expected figure is ~261, not the ~252 real market data gives: pandas' "B" frequency
    is every Mon-Fri, and it is the ~9 annual market holidays that take a real exchange from
    261 down to 252. The point of the assertion is the calendar the factor came from.
    """
    asset = _walk(100.0, 0.30, 2.0)  # freq="D": every calendar day
    bench = _walk(400.0, 0.80, 4.0)  # freq="B": weekdays only
    result = _compare(asset, bench, bench_freq="B")
    assert result.overlapping_observations < DAYS
    assert result.periods_per_year is not None
    # 365.25 / 7 * 5 = 260.9 weekdays a year, nowhere near the crypto leg's 365.
    assert result.periods_per_year == pytest.approx(261.0, rel=0.02)
    # The reported dates are the OVERLAP, not either input window.
    assert result.end_date <= "2024-07-18"


def test_compare_to_benchmark_needs_two_overlapping_dates() -> None:
    """A one-day overlap has no return to compare, so it must report the overlap count
    rather than indexing into an empty return list."""
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df([100.0, 101.0, 102.0], start="2024-01-01")},
            "SPY": {"history_df": make_history_df([50.0, 51.0, 52.0], start="2024-01-03")},
        }
    )
    with pytest.raises(DataUnavailable, match="overlapping"):
        _client(ticker_factory=factory).compare_to_benchmark("AAPL", "SPY", "5d", 0.0)


def test_compare_to_benchmark_with_no_shared_dates_names_both_symbols() -> None:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df([100.0, 101.0], start="2024-01-01")},
            "SPY": {"history_df": make_history_df([50.0, 51.0], start="2025-06-01")},
        }
    )
    with pytest.raises(DataUnavailable) as excinfo:
        _client(ticker_factory=factory).compare_to_benchmark("AAPL", "SPY", "1y", 0.0)
    assert "AAPL" in str(excinfo.value) and "SPY" in str(excinfo.value)


def test_compare_to_benchmark_nulls_annualized_figures_on_a_short_overlap() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0, n=30), _walk(400.0, 0.80, 4.0, n=30)
    result = _compare(asset, bench)  # 30 calendar days: under the 90-day gate
    assert result.periods_per_year is None
    assert result.annualized_return_percent is None
    assert result.benchmark_annualized_return_percent is None
    assert result.alpha_percent is None
    assert result.tracking_error_percent is None
    assert result.information_ratio is None
    # Beta and correlation need no annualization, so the gate must not blank them.
    assert result.beta is not None
    assert result.correlation is not None
    # Nor do the total returns.
    assert result.excess_return_percent is not None


def test_compare_to_benchmark_flat_benchmark_gives_no_beta_or_alpha() -> None:
    """A benchmark that never moves has zero variance."""
    result = _compare(_walk(100.0, 0.30, 2.0), [400.0] * DAYS)
    assert result.beta is None
    assert result.correlation is None
    assert result.alpha_percent is None  # alpha is undefined without a beta
    assert result.tracking_error_percent is not None  # active return still varies


def test_compare_to_benchmark_rejects_comparing_a_symbol_to_itself() -> None:
    factory = fake_multi_ticker_factory(
        {"SPY": {"history_df": make_history_df(_walk(400.0, 0.8, 4.0))}}
    )
    with pytest.raises(InvalidInput, match="two different"):
        _client(ticker_factory=factory).compare_to_benchmark("SPY", "spy", "1y", 0.0)


def test_compare_to_benchmark_propagates_an_unknown_benchmark() -> None:
    factory = fake_multi_ticker_factory(
        {"AAPL": {"history_df": make_history_df(_walk(100.0, 0.3, 2.0))}}
    )
    with pytest.raises(DataUnavailable):
        _client(ticker_factory=factory).compare_to_benchmark("AAPL", "NOPE", "1y", 0.0)


def test_compare_to_benchmark_normalizes_both_symbols() -> None:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(_walk(100.0, 0.30, 2.0))},
            "SPY": {"history_df": make_history_df(_walk(400.0, 0.80, 4.0))},
        }
    )
    result = _client(ticker_factory=factory).compare_to_benchmark(" aapl ", "spy", "1y", 0.0)
    assert (result.symbol, result.benchmark) == ("AAPL", "SPY")


def test_compare_to_benchmark_fetches_both_legs_concurrently() -> None:
    # Both fetches wait on a 2-party barrier, so a sequential implementation deadlocks
    # until the timeout rather than returning.
    gate = threading.Barrier(2, timeout=10)
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(_walk(100.0, 0.30, 2.0))},
            "SPY": {"history_df": make_history_df(_walk(400.0, 0.80, 4.0))},
        },
        gate=gate,
    )
    result = _client(ticker_factory=factory).compare_to_benchmark("AAPL", "SPY", "1y", 0.0)
    assert result.overlapping_observations == DAYS
