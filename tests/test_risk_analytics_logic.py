"""Unit tests for the risk-adjusted and benchmark-relative math (no MCP/network).

Every expected value here is hand-computed from a short series; the arithmetic is shown
in the comment above each assertion so a reader can check it without running the code.
"""

import math

import pytest

from finance_mcp.data.analytics import (
    align_closes,
    annualized_volatility,
    beta,
    calmar_ratio,
    correlation,
    downside_deviation,
    information_ratio,
    jensen_alpha,
    periodic_risk_free,
    sharpe_ratio,
    simple_returns,
    sortino_ratio,
    tracking_error,
)
from finance_mcp.data.errors import InvalidInput

# returns: +0.10 then 104.5/110 - 1 = -0.05.
# mean = 0.025; stdev(ddof=1) = 0.15 / sqrt(2) = 0.106066017
MIXED = [100.0, 110.0, 104.5]
FLAT_UP = [100.0, 110.0, 121.0]  # returns +0.10, +0.10 -> zero dispersion


def test_simple_returns() -> None:
    assert simple_returns(MIXED) == pytest.approx([0.10, -0.05])


def test_simple_returns_single_close_is_empty() -> None:
    assert simple_returns([100.0]) == []


def test_periodic_risk_free_de_annualizes_geometrically() -> None:
    # 1.01 ** 4 = 1.04060401, so a 4.060401% annual rate is exactly 1%/quarter.
    assert periodic_risk_free(0.04060401, 4.0) == pytest.approx(0.01, rel=1e-9)


def test_periodic_risk_free_zero_is_zero() -> None:
    assert periodic_risk_free(0.0, 252.0) == 0.0


@pytest.mark.parametrize("rate", [-1.0, -1.5])
def test_periodic_risk_free_rejects_rates_at_or_below_minus_one(rate: float) -> None:
    with pytest.raises(InvalidInput):
        periodic_risk_free(rate, 252.0)


@pytest.mark.parametrize("ppy", [0.0, -252.0])
def test_periodic_risk_free_rejects_nonpositive_periods(ppy: float) -> None:
    with pytest.raises(InvalidInput):
        periodic_risk_free(0.05, ppy)


def test_sharpe_ratio_zero_risk_free() -> None:
    # 0.025 / 0.106066017 * sqrt(252) = 0.2357023 * 15.8745079 = 3.7416574
    assert sharpe_ratio(MIXED, 252.0) == pytest.approx(3.7416574, rel=1e-6)


def test_sharpe_ratio_defaults_to_a_zero_risk_free_rate() -> None:
    assert sharpe_ratio(MIXED, 252.0) == sharpe_ratio(MIXED, 252.0, 0.0)


def test_sharpe_ratio_subtracts_the_risk_free_rate() -> None:
    # ppy=4 and rf=4.060401%/yr -> exactly 1%/period. Excess = [0.09, -0.06]:
    # mean 0.015, stdev 0.15/sqrt(2) = 0.106066017 (unchanged by a constant shift)
    # 0.015 / 0.106066017 * sqrt(4) = 0.1414214 * 2 = 0.2828427
    assert sharpe_ratio(MIXED, 4.0, 0.04060401) == pytest.approx(0.2828427, rel=1e-6)


def test_sharpe_ratio_falls_when_the_risk_free_rate_rises() -> None:
    zero, hurdle = sharpe_ratio(MIXED, 252.0, 0.0), sharpe_ratio(MIXED, 252.0, 0.05)
    assert zero is not None and hurdle is not None
    assert hurdle < zero


def test_sharpe_ratio_zero_dispersion_is_none() -> None:
    # A constant return has no dispersion, so return-per-unit-of-risk is undefined.
    assert sharpe_ratio(FLAT_UP, 252.0) is None


def test_sharpe_ratio_needs_two_returns() -> None:
    assert sharpe_ratio([100.0, 110.0], 252.0) is None


def test_downside_deviation_ignores_upside() -> None:
    # shortfalls [0, -0.05] -> mean square 0.0025/2 = 0.00125 -> sqrt = 0.035355339
    # annualized percent: 0.035355339 * sqrt(252) * 100 = 56.1248608
    assert downside_deviation(MIXED, 252.0) == pytest.approx(56.1248608, rel=1e-6)


def test_downside_deviation_is_zero_with_no_downside() -> None:
    assert downside_deviation(FLAT_UP, 252.0) == pytest.approx(0.0)


def test_downside_deviation_needs_two_returns() -> None:
    assert downside_deviation([100.0, 110.0], 252.0) is None


def test_downside_deviation_is_below_total_volatility() -> None:
    # Only half the dispersion is on the downside here, so it must be the smaller figure.
    measured = downside_deviation(MIXED, 252.0)
    assert measured is not None
    assert measured < annualized_volatility(MIXED, 252.0)


def test_sortino_ratio_uses_downside_dispersion_only() -> None:
    # 0.025 / 0.035355339 * sqrt(252) = 0.7071068 * 15.8745079 = 11.2249722
    assert sortino_ratio(MIXED, 252.0) == pytest.approx(11.2249722, rel=1e-6)


def test_sortino_ratio_exceeds_sharpe_when_downside_is_the_smaller_half() -> None:
    sortino, sharpe = sortino_ratio(MIXED, 252.0), sharpe_ratio(MIXED, 252.0)
    assert sortino is not None and sharpe is not None
    assert sortino > sharpe


def test_sortino_ratio_no_downside_is_none() -> None:
    # Nothing ever fell below the target, so downside risk is zero and the ratio undefined.
    assert sortino_ratio(FLAT_UP, 252.0) is None


def test_sortino_ratio_needs_two_returns() -> None:
    assert sortino_ratio([100.0, 110.0], 252.0) is None


def test_sortino_ratio_subtracts_the_risk_free_rate() -> None:
    # Excess [0.09, -0.06]: mean 0.015, downside sqrt(0.0036/2) = 0.0424264
    # 0.015 / 0.0424264 * 2 = 0.7071068
    assert sortino_ratio(MIXED, 4.0, 0.04060401) == pytest.approx(0.7071068, rel=1e-6)


def test_calmar_ratio() -> None:
    # 4.5% CAGR against a 5% peak-to-trough fall -> 0.9 units of return per unit of pain.
    assert calmar_ratio(4.5, -5.0) == pytest.approx(0.9)


def test_calmar_ratio_uses_the_drawdown_magnitude_not_its_sign() -> None:
    assert calmar_ratio(4.5, -5.0) == calmar_ratio(4.5, 5.0)


def test_calmar_ratio_negative_cagr_stays_negative() -> None:
    assert calmar_ratio(-10.0, -20.0) == pytest.approx(-0.5)


def test_calmar_ratio_without_a_drawdown_is_none() -> None:
    # A monotonically rising series never drew down; dividing by zero pain is undefined.
    assert calmar_ratio(12.0, 0.0) is None


def test_sharpe_ratio_annualization_factor_scales_the_ratio() -> None:
    quarterly, daily = sharpe_ratio(MIXED, 4.0), sharpe_ratio(MIXED, 252.0)
    assert quarterly is not None and daily is not None
    assert daily == pytest.approx(quarterly * math.sqrt(252.0 / 4.0))


# --- benchmark-relative -------------------------------------------------------------

# asset returns [+0.10, -0.05, +0.02]; benchmark returns [+0.05, -0.02, +0.01]
ASSET = [100.0, 110.0, 104.5, 106.59]
BENCH = [100.0, 105.0, 102.9, 103.929]
# A series whose returns are exactly twice the benchmark's: beta 2, correlation 1.
DOUBLE = [100.0, 110.0, 99.0]
HALF_BENCH = [100.0, 105.0, 99.75]
FLAT = [100.0, 100.0, 100.0, 100.0]


def test_align_closes_inner_joins_on_date() -> None:
    asset = [("2024-01-01", 10.0), ("2024-01-02", 11.0), ("2024-01-03", 12.0)]
    bench = [("2024-01-02", 20.0), ("2024-01-03", 21.0), ("2024-01-04", 22.0)]
    dates, a, b = align_closes(asset, bench)
    assert dates == ["2024-01-02", "2024-01-03"]
    assert a == [11.0, 12.0]
    assert b == [20.0, 21.0]


def test_align_closes_drops_the_weekend_bars_of_a_seven_day_instrument() -> None:
    # The crypto-vs-equity case: Sat/Sun exist for the asset and not for the benchmark,
    # so the overlap is the five weekdays. Forward-filling the equity instead would
    # invent two closes that never traded.
    crypto = [(f"2024-01-0{d}", 100.0 + d) for d in range(1, 8)]  # Mon 1st .. Sun 7th
    equity = [(f"2024-01-0{d}", 50.0 + d) for d in (1, 2, 3, 4, 5)]
    dates, a, b = align_closes(crypto, equity)
    assert len(dates) == 5
    assert dates[-1] == "2024-01-05"
    assert a == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert b == [51.0, 52.0, 53.0, 54.0, 55.0]


def test_align_closes_with_no_shared_dates_is_empty() -> None:
    dates, a, b = align_closes([("2024-01-01", 1.0)], [("2025-01-01", 1.0)])
    assert (dates, a, b) == ([], [], [])


def test_align_closes_orders_oldest_first_regardless_of_input_order() -> None:
    asset = [("2024-01-03", 12.0), ("2024-01-01", 10.0)]
    bench = [("2024-01-01", 20.0), ("2024-01-03", 21.0)]
    dates, _, _ = align_closes(asset, bench)
    assert dates == ["2024-01-01", "2024-01-03"]


def test_beta_of_a_double_leveraged_series_is_two() -> None:
    assert beta(DOUBLE, HALF_BENCH) == pytest.approx(2.0)


def test_beta_hand_computed() -> None:
    # cov(ddof=1) = 0.00263333, var(ddof=1) = 0.00123333 -> 2.1351351
    assert beta(ASSET, BENCH) == pytest.approx(2.1351351, rel=1e-6)


def test_beta_against_itself_is_one() -> None:
    assert beta(BENCH, BENCH) == pytest.approx(1.0)


def test_beta_of_a_flat_benchmark_is_none() -> None:
    # A benchmark that never moves has zero variance: sensitivity to it is undefined.
    assert beta(ASSET, FLAT) is None


def test_beta_needs_two_returns() -> None:
    assert beta([100.0, 110.0], [100.0, 105.0]) is None


def test_beta_rejects_unaligned_series() -> None:
    with pytest.raises(InvalidInput):
        beta(ASSET, BENCH[:-1])


def test_correlation_of_a_scaled_series_is_one() -> None:
    assert correlation(DOUBLE, HALF_BENCH) == pytest.approx(1.0)


def test_correlation_hand_computed() -> None:
    # 0.00263333 / (0.0750555 * 0.0351188) = 0.9990400
    assert correlation(ASSET, BENCH) == pytest.approx(0.9990400, rel=1e-6)


def test_correlation_with_a_flat_series_is_none() -> None:
    assert correlation(ASSET, FLAT) is None
    assert correlation(FLAT, BENCH) is None


def test_correlation_needs_two_returns() -> None:
    assert correlation([100.0, 110.0], [100.0, 105.0]) is None


def test_correlation_rejects_unaligned_series() -> None:
    with pytest.raises(InvalidInput):
        correlation(ASSET, BENCH[:-1])


def test_jensen_alpha_is_zero_when_the_return_is_exactly_what_beta_predicts() -> None:
    # rf 2%, benchmark 10% -> a beta-1.5 asset is "supposed" to make 2 + 1.5*8 = 14%.
    assert jensen_alpha(14.0, 10.0, 1.5, 0.02) == pytest.approx(0.0)


def test_jensen_alpha_is_the_outperformance_over_the_predicted_return() -> None:
    assert jensen_alpha(17.0, 10.0, 1.5, 0.02) == pytest.approx(3.0)


def test_jensen_alpha_is_negative_when_beta_alone_would_have_done_better() -> None:
    assert jensen_alpha(11.0, 10.0, 1.5, 0.02) == pytest.approx(-3.0)


def test_jensen_alpha_treats_the_risk_free_rate_as_a_decimal() -> None:
    # rf is 0.02 = 2%, NOT 2. A beta of 1 makes the rate cancel out entirely.
    assert jensen_alpha(12.0, 10.0, 1.0, 0.02) == pytest.approx(2.0)


def test_tracking_error_of_a_perfect_tracker_is_zero() -> None:
    assert tracking_error(BENCH, BENCH, 252.0) == pytest.approx(0.0)


def test_tracking_error_hand_computed() -> None:
    # active returns [0.05, -0.03, 0.01]; stdev(ddof=1) = 0.04 -> 0.04*sqrt(252)*100
    assert tracking_error(ASSET, BENCH, 252.0) == pytest.approx(63.4980315, rel=1e-6)


def test_tracking_error_needs_two_returns() -> None:
    assert tracking_error([100.0, 110.0], [100.0, 105.0], 252.0) is None


def test_tracking_error_rejects_unaligned_series() -> None:
    with pytest.raises(InvalidInput):
        tracking_error(ASSET, BENCH[:-1], 252.0)


def test_information_ratio_hand_computed() -> None:
    # mean active 0.01 / stdev 0.04 * sqrt(252) = 0.25 * 15.8745079 = 3.9686270
    assert information_ratio(ASSET, BENCH, 252.0) == pytest.approx(3.9686270, rel=1e-6)


def test_information_ratio_of_a_perfect_tracker_is_none() -> None:
    # Zero tracking error: active return per unit of active risk is undefined.
    assert information_ratio(BENCH, BENCH, 252.0) is None


def test_information_ratio_needs_two_returns() -> None:
    assert information_ratio([100.0, 110.0], [100.0, 105.0], 252.0) is None


def test_information_ratio_rejects_unaligned_series() -> None:
    with pytest.raises(InvalidInput):
        information_ratio(ASSET, BENCH[:-1], 252.0)
