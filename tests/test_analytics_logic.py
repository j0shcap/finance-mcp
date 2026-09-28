"""Unit tests for the pure analytics math (no MCP/network)."""

import math
import statistics

import pytest

from finance_mcp.data.analytics import (
    annualized_return,
    annualized_volatility,
    infer_periods_per_year,
    max_drawdown,
    sma,
    total_return,
)
from finance_mcp.data.errors import InvalidInput

SERIES = [100.0, 110.0, 99.0]  # returns: +0.10, then 99/110-1 = -0.10


def test_total_return() -> None:
    assert total_return(SERIES) == pytest.approx(-1.0)  # 99/100 - 1


def test_total_return_needs_two() -> None:
    with pytest.raises(InvalidInput):
        total_return([100.0])


def test_annualized_return_equals_total_return_over_one_year() -> None:
    # The defining property: over exactly one calendar year, CAGR IS the total return.
    closes = [100.0 + i for i in range(253)]  # 100 .. 352
    assert annualized_return(closes, years=1.0) == pytest.approx(total_return(closes), rel=1e-12)


def test_annualized_return_is_independent_of_bar_count() -> None:
    # The reported BTC-USD bug in pure-math form: the same calendar move over the same
    # elapsed time must annualize identically whether the instrument printed 253 weekday
    # bars or 366 seven-day-a-week bars. The old bar-count exponent gave two answers.
    weekday = [100.0] + [100.0] * 251 + [77.02]
    seven_day = [100.0] + [100.0] * 364 + [77.02]
    assert annualized_return(weekday, years=1.0) == pytest.approx(
        annualized_return(seven_day, years=1.0)
    )


def test_annualized_return_half_year_compounds_up() -> None:
    # years=0.5 -> exponent 2 -> 1.21**2 - 1 = 0.4641 -> 46.41%
    assert annualized_return([100.0, 121.0], years=0.5) == pytest.approx(46.41, rel=1e-4)


def test_annualized_return_multi_year_takes_the_root() -> None:
    # Doubling over two years annualizes to 2**0.5 - 1 = 41.42%, not 100%.
    assert annualized_return([100.0, 200.0], years=2.0) == pytest.approx(41.4214, rel=1e-4)


def test_annualized_return_declining() -> None:
    # A 10% loss over a quarter-year annualizes to 0.9**4 - 1 = -34.39%.
    assert annualized_return([100.0, 90.0], years=0.25) == pytest.approx((0.9**4 - 1) * 100)


def test_annualized_return_needs_two() -> None:
    with pytest.raises(InvalidInput):
        annualized_return([100.0], years=1.0)


@pytest.mark.parametrize("years", [0.0, -1.0])
def test_annualized_return_rejects_nonpositive_years(years: float) -> None:
    with pytest.raises(InvalidInput):
        annualized_return([100.0, 110.0], years=years)


def test_annualized_return_overflow_becomes_invalid_input() -> None:
    # A doubling over a microscopic window: 2.0 ** 1_000_000 overflows the float range.
    with pytest.raises(InvalidInput):
        annualized_return([100.0, 200.0], years=1e-6)


def test_infer_periods_per_year_seven_day_market() -> None:
    # 366 daily bars across one year -> 365 intervals/year, the crypto shape.
    assert infer_periods_per_year(366, 1.0) == pytest.approx(365.0)


def test_infer_periods_per_year_weekday_market() -> None:
    # 253 weekday bars across one year -> 252 intervals/year, the equity shape.
    assert infer_periods_per_year(253, 1.0) == pytest.approx(252.0)


def test_infer_periods_per_year_needs_two_bars() -> None:
    with pytest.raises(InvalidInput):
        infer_periods_per_year(1, 1.0)


@pytest.mark.parametrize("years", [0.0, -1.0])
def test_infer_periods_per_year_rejects_nonpositive_years(years: float) -> None:
    with pytest.raises(InvalidInput):
        infer_periods_per_year(253, years)


def test_annualized_volatility() -> None:
    # stdev([0.10, -0.10], ddof=1) = sqrt(0.02) = 0.1414214; * sqrt(252) * 100 ~= 224.499
    assert annualized_volatility(SERIES, 252.0) == pytest.approx(224.499, rel=1e-4)


def test_annualized_volatility_scales_with_the_factor() -> None:
    # A 24/7 instrument observes ~365 returns/year, so the same daily dispersion
    # annualizes sqrt(365/252) ~= 1.20x higher than the equity convention.
    equity = annualized_volatility(SERIES, 252.0)
    crypto = annualized_volatility(SERIES, 365.0)
    assert crypto == pytest.approx(equity * math.sqrt(365.0 / 252.0))


def test_annualized_volatility_too_few_returns_is_zero() -> None:
    assert annualized_volatility([100.0, 110.0], 252.0) == 0.0  # only 1 return


def test_max_drawdown() -> None:
    assert max_drawdown(SERIES) == pytest.approx(-10.0)  # 99/110 - 1


def test_max_drawdown_monotonic_up_is_zero() -> None:
    assert max_drawdown([100.0, 101.0, 102.0]) == pytest.approx(0.0)


def test_max_drawdown_single_element_is_zero() -> None:
    assert max_drawdown([100.0]) == pytest.approx(0.0)


def test_max_drawdown_needs_one() -> None:
    with pytest.raises(InvalidInput):
        max_drawdown([])


def test_sma() -> None:
    assert sma(SERIES, 2) == pytest.approx(104.5)  # (110+99)/2
    assert sma(SERIES, 5) is None


def test_max_drawdown_uses_running_peak() -> None:
    # Peak 120 (bar 1), trough 90 (bar 2), recovers to 130 (last bar):
    # running-peak gives -25% (120->90); a naive first-to-min would wrongly give -10%.
    assert max_drawdown([100.0, 120.0, 90.0, 130.0]) == pytest.approx(-25.0)


def test_annualized_volatility_multiple_returns() -> None:
    closes = [100.0, 110.0, 99.0, 108.9]
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    expected = statistics.stdev(rets) * math.sqrt(252.0) * 100
    assert annualized_volatility(closes, 252.0) == pytest.approx(expected)


def test_sma_at_window_boundary() -> None:
    assert sma([100.0, 110.0, 99.0], 3) == pytest.approx(103.0)  # mean of all three, not None


def test_sma_nonpositive_window_raises() -> None:
    with pytest.raises(InvalidInput):
        sma([100.0, 110.0, 99.0], 0)
