"""Date-based bond pricing: coupon schedule, day counts, and street-convention analytics.

Reference values are cited inline. The Microsoft examples are the ones shipped in the
Excel function documentation; the Treasury examples are the worked examples in the
Uniform Offering Circular's formula appendix (31 CFR part 356, appendix B).
"""

import datetime

import pytest

from finance_mcp.data.calculators import (
    MAX_BOND_SPAN_YEARS,
    _add_months,
    _bond_metrics,
    _coupon_schedule,
    _days_30_360_us,
    bond_price,
    bond_price_dated,
    bond_ytm,
    bond_ytm_dated,
)
from finance_mcp.data.errors import InvalidInput
from finance_mcp.data.models import (
    BondDatedAnalytics,
    BondDayCount,
    FirstPeriodDiscount,
)
from finance_mcp.tools._inputs import MAX_BOND_YEARS


def d(iso: str) -> datetime.date:
    return datetime.date.fromisoformat(iso)


# --------------------------------------------------------------------------------------
# _add_months: end-of-month preserving month arithmetic
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        # A month-end date steps to the target month's end, not to the same day number.
        # This is what makes a 31 March maturity generate 30 September coupons.
        ("1992-03-31", -6, "1991-09-30"),
        ("1990-09-30", -6, "1990-03-31"),
        ("2024-01-31", 1, "2024-02-29"),  # leap February
        ("2023-01-31", 1, "2023-02-28"),  # non-leap February
        ("2023-02-28", 6, "2023-08-31"),  # month-end in, month-end out
        ("2024-02-29", 12, "2025-02-28"),
        # A mid-month day is preserved exactly.
        ("2017-11-15", -6, "2017-05-15"),
        ("2008-02-15", -120, "1998-02-15"),
        ("2007-11-15", 3, "2008-02-15"),
        ("2007-11-15", 14, "2009-01-15"),
    ],
)
def test_add_months_preserves_month_end(start: str, months: int, expected: str) -> None:
    assert _add_months(d(start), months) == d(expected)


def test_add_months_zero_is_identity() -> None:
    assert _add_months(d("2024-03-15"), 0) == d("2024-03-15")


# --------------------------------------------------------------------------------------
# _days_30_360_us: the NASD / Excel `basis=0` day count
# --------------------------------------------------------------------------------------


def test_days_30_360_matches_excel_accrint_example() -> None:
    """Excel ACCRINT docs: issue 2008-03-01, settlement 2008-05-01, 10%, par 1000,
    basis 0 (US 30/360) accrues 16.666667. That is 60/360 of a year, so the day count
    between those dates must be exactly 60.

    https://support.microsoft.com/en-us/office/accrint-function-fe45d089-6722-4fb3-9379-e1f911d8dc74
    """
    days = _days_30_360_us(d("2008-03-01"), d("2008-05-01"))
    assert days == 60
    assert 1000.0 * 0.10 * days / 360.0 == pytest.approx(16.666667, abs=1e-6)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2007-11-15", "2008-02-15", 90),  # a plain quarter: the Excel PRICE example's A
        ("2024-01-01", "2025-01-01", 360),  # a whole year is 360 days
        ("2024-01-15", "2024-02-15", 30),
        # Rule: D1 == 31 -> 30.
        ("2024-01-31", "2024-02-15", 15),
        ("2024-03-31", "2024-04-15", 15),
        # Rule: D2 == 31 and D1 >= 30 -> D2 = 30. With D1 = 30 the month is exactly 30
        # days; without the rule 2024-01-30 -> 2024-03-31 would count 61, not 60.
        ("2024-01-30", "2024-03-31", 60),
        ("2024-01-31", "2024-03-31", 60),
        # D2 == 31 but D1 < 30, so D2 is left alone: 15 Jan -> 31 Mar is 30 + 16.
        ("2024-01-15", "2024-03-31", 76),
        # Rule: D1 is the last day of February -> D1 = 30.
        ("2023-02-28", "2023-03-31", 30),
        ("2024-02-29", "2024-03-31", 30),
        # Rule: both dates are the last day of February -> D2 = 30 (and D1 = 30), so a
        # Feb-end to Feb-end span is a whole number of 30-day months.
        ("2023-02-28", "2024-02-29", 360),
        # A February that is not month-end gets no special treatment.
        ("2024-02-15", "2024-03-15", 30),
    ],
)
def test_days_30_360_us_nasd_rules(start: str, end: str, expected: int) -> None:
    assert _days_30_360_us(d(start), d(end)) == expected


# --------------------------------------------------------------------------------------
# _coupon_schedule: generated backward from maturity
# --------------------------------------------------------------------------------------


def test_coupon_schedule_matches_treasury_month_end_example() -> None:
    """31 CFR 356 appendix B, example I.B: an 8.5% 2-year note issued 1990-04-02, due
    1992-03-31, with interest payments on September 30 and March 31. The appendix states
    r = 181 days (settlement 1990-04-02 to the next payment 1990-09-30) and s = 183 days
    (1990-03-31 to 1990-09-30), which pins the surrounding coupon dates.

    https://www.govinfo.gov/content/pkg/CFR-2025-title31-vol2/pdf/CFR-2025-title31-vol2-part356-appB.pdf
    """
    previous, next_coupon, periods = _coupon_schedule(d("1990-04-02"), d("1992-03-31"), 2)
    assert previous == d("1990-03-31")
    assert next_coupon == d("1990-09-30")
    assert (next_coupon - previous).days == 183
    assert (next_coupon - d("1990-04-02")).days == 181
    assert periods == 4


def test_coupon_schedule_for_the_excel_price_example() -> None:
    """Excel PRICE docs example: settlement 2008-02-15, maturity 2017-11-15, semiannual.
    Coupons fall on 15 May and 15 November, so settlement sits in the 2007-11-15 ->
    2008-05-15 period with 20 coupons left to run.
    """
    previous, next_coupon, periods = _coupon_schedule(d("2008-02-15"), d("2017-11-15"), 2)
    assert (previous, next_coupon) == (d("2007-11-15"), d("2008-05-15"))
    assert periods == 20


def test_coupon_schedule_on_a_coupon_date_starts_a_fresh_period() -> None:
    """Settlement exactly on a coupon date: that date is the *previous* coupon (zero
    accrued), and a whole period runs to the next one."""
    previous, next_coupon, periods = _coupon_schedule(d("2007-11-15"), d("2017-11-15"), 2)
    assert (previous, next_coupon) == (d("2007-11-15"), d("2008-05-15"))
    assert periods == 20


def test_coupon_schedule_one_day_before_maturity_has_a_single_period() -> None:
    previous, next_coupon, periods = _coupon_schedule(d("2017-11-14"), d("2017-11-15"), 2)
    assert (previous, next_coupon) == (d("2017-05-15"), d("2017-11-15"))
    assert periods == 1


@pytest.mark.parametrize(
    ("frequency", "expected_previous", "expected_next", "expected_periods"),
    [
        (1, "2018-11-15", "2019-11-15", 9),
        (2, "2018-11-15", "2019-05-15", 18),
        (3, "2018-11-15", "2019-03-15", 27),
        (4, "2018-11-15", "2019-02-15", 36),
        (6, "2018-11-15", "2019-01-15", 54),
        (12, "2018-12-15", "2019-01-15", 107),
    ],
)
def test_coupon_schedule_honours_frequency(
    frequency: int, expected_previous: str, expected_next: str, expected_periods: int
) -> None:
    """Every whole-month coupon frequency anchors on the maturity day-of-month."""
    previous, next_coupon, periods = _coupon_schedule(d("2018-12-20"), d("2027-11-15"), frequency)
    assert previous == d(expected_previous)
    assert next_coupon == d(expected_next)
    assert periods == expected_periods


# --------------------------------------------------------------------------------------
# _bond_metrics: the shared core. At first_fraction == 1.0 the fractional-period formulas
# collapse to the on-coupon-date ones, which is what lets bond_price delegate to it.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("coupon_rate", "years", "ytm", "frequency"),
    [
        (0.05, 10.0, 0.06, 2),  # discount
        (0.06, 10.0, 0.06, 2),  # par
        (0.07, 10.0, 0.06, 2),  # premium
        (0.0, 5.0, 0.04, 1),  # zero coupon
        (0.05, 2.5, 0.06, 2),  # half-year maturity
        (0.09, 30.0, 0.0884, 2),  # long bond
    ],
)
def test_bond_metrics_on_a_coupon_date_reproduces_bond_price_exactly(
    coupon_rate: float, years: float, ytm: float, frequency: int
) -> None:
    """The delegation contract: identical results, not merely close ones. Anything less
    would silently move the existing tool's numbers."""
    expected = bond_price(
        face=1000.0,
        coupon_rate=coupon_rate,
        years_to_maturity=years,
        ytm=ytm,
        frequency=frequency,
    )
    dirty, macaulay, modified, convexity = _bond_metrics(
        face=1000.0,
        coupon_rate=coupon_rate,
        frequency=frequency,
        y=ytm / frequency,
        n=round(years * frequency),
        first_fraction=1.0,
    )
    assert dirty == expected.price
    assert macaulay == expected.macaulay_duration
    assert modified == expected.modified_duration
    assert convexity == expected.convexity


# --------------------------------------------------------------------------------------
# bond_price_dated: settlement between coupon dates, against published references
# --------------------------------------------------------------------------------------


def test_price_matches_the_excel_price_documentation_example() -> None:
    """Microsoft's PRICE example: settlement 2008-02-15, maturity 2017-11-15, coupon
    5.75%, yield 6.50%, redemption 100, frequency 2, basis 0 (US 30/360). The docs display
    the rounded $94.63; Excel's unrounded result is 94.634362.

    Settlement sits mid-period, so this is the case the on-coupon calculator rejects:
    90 of the 180 day-count days have elapsed, accruing exactly half a coupon.

    https://support.microsoft.com/en-us/office/price-function-3ea9deac-8dfa-436f-a7c8-17ea02c21b0a
    """
    result = bond_price_dated(
        settlement=d("2008-02-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.0575,
        ytm=0.065,
        face=100.0,
        frequency=2,
        day_count="30/360",
    )
    assert result.clean_price == pytest.approx(94.634362, abs=1e-6)
    assert round(result.clean_price, 2) == 94.63  # the figure the docs print
    assert result.accrued_interest == pytest.approx(1.4375, abs=1e-9)
    assert result.dirty_price == pytest.approx(96.071862, abs=1e-6)
    assert (result.accrued_days, result.period_days) == (90.0, 180.0)
    assert result.accrued_fraction == pytest.approx(0.5, abs=1e-12)
    assert result.periods_remaining == 20
    assert result.previous_coupon_date == d("2007-11-15")
    assert result.next_coupon_date == d("2008-05-15")
    assert result.day_count == "30/360"


def test_price_matches_the_excel_yield_documentation_example_price() -> None:
    """Microsoft's YIELD example prices the same 5.75% bond, maturing 2016-11-15, at
    pr = 95.04287 for a yield of exactly 6.5%. Pricing it at 6.5% must return that price,
    which is the other half of the YIELD round trip asserted below.

    https://support.microsoft.com/en-us/office/yield-function-f5f5ca43-c4bd-434f-8bd2-ed3c9727a4fe
    """
    result = bond_price_dated(
        settlement=d("2008-02-15"),
        maturity=d("2016-11-15"),
        coupon_rate=0.0575,
        ytm=0.065,
        day_count="30/360",
    )
    assert round(result.clean_price, 5) == 95.04287
    assert result.clean_price == pytest.approx(95.0428744, abs=1e-6)
    assert result.periods_remaining == 18


def test_price_matches_the_libreoffice_price_documentation_example() -> None:
    """LibreOffice's PRICE help documents PRICE(1999-02-15; 2007-11-15; 0.0575; 0.065;
    100; 2; 0) = 95.04287 -- an independent implementation of the same basis-0 street
    formula, on a different pair of dates than the Microsoft examples.
    """
    result = bond_price_dated(
        settlement=d("1999-02-15"),
        maturity=d("2007-11-15"),
        coupon_rate=0.0575,
        ytm=0.065,
        day_count="30/360",
    )
    assert round(result.clean_price, 5) == 95.04287


def test_price_matches_the_treasury_appendix_b_example() -> None:
    """31 CFR 356 appendix B, example I.A: a 10-3/4%-era long bond with C = 8.75,
    i = 0.0884, r = s = 184 and n = 59 prices at P = 99.057893 per 100.

    r == s means settlement falls exactly on a coupon date, one full period before the
    next payment, with 60 coupons left to run. That is the convention-neutral case: with
    no part period, the Treasury's simple-interest stub factor and the street compound
    factor are the same number, so this reference pins the dated path regardless of which
    first-period convention is in force. It also has to agree with the on-coupon
    calculator, since they are pricing the same cashflows.

    https://www.govinfo.gov/content/pkg/CFR-2025-title31-vol2/pdf/CFR-2025-title31-vol2-part356-appB.pdf
    """
    result = bond_price_dated(
        settlement=d("1985-11-15"),
        maturity=d("2015-11-15"),
        coupon_rate=0.0875,
        ytm=0.0884,
        face=100.0,
        frequency=2,
    )
    assert result.periods_remaining == 60
    assert result.clean_price == pytest.approx(99.057893, abs=1e-6)
    assert result.accrued_interest == 0.0
    assert result.dirty_price == result.clean_price
    assert result.accrued_fraction == 0.0
    on_coupon = bond_price(
        face=100.0, coupon_rate=0.0875, years_to_maturity=30.0, ytm=0.0884, frequency=2
    )
    assert result.clean_price == pytest.approx(on_coupon.price, rel=1e-12)


@pytest.mark.parametrize("day_count", ["actual/actual", "30/360"])
def test_settlement_on_a_coupon_date_equals_the_on_coupon_calculator(
    day_count: BondDayCount,
) -> None:
    """With nothing accrued there is no part period, so both day counts must reproduce the
    existing on-coupon tool exactly -- including its duration and convexity."""
    result = bond_price_dated(
        settlement=d("2007-11-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.05,
        ytm=0.06,
        face=1000.0,
        frequency=2,
        day_count=day_count,
    )
    expected = bond_price(
        face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2
    )
    assert result.accrued_interest == 0.0
    assert result.clean_price == expected.price
    assert result.dirty_price == expected.price
    assert result.macaulay_duration == expected.macaulay_duration
    assert result.modified_duration == expected.modified_duration
    assert result.convexity == expected.convexity
    assert result.current_yield == expected.current_yield


def test_actual_actual_icma_uses_the_real_days_in_the_coupon_period() -> None:
    """Actual/Actual (ICMA) counts real days on both sides of the ratio, so the accrual
    denominator is the length of *this* coupon period (182 days from 2007-11-15 to
    2008-05-15), not a nominal 180. The 30/360 count of the same span is 90/180, so the
    two conventions must disagree -- that difference is the reason day_count exists.
    """

    def price(day_count: BondDayCount) -> BondDatedAnalytics:
        return bond_price_dated(
            settlement=d("2008-02-15"),
            maturity=d("2017-11-15"),
            coupon_rate=0.0575,
            ytm=0.065,
            face=100.0,
            frequency=2,
            day_count=day_count,
        )

    icma = price("actual/actual")
    assert (icma.accrued_days, icma.period_days) == (92.0, 182.0)
    assert icma.accrued_interest == pytest.approx(2.875 * 92.0 / 182.0, abs=1e-12)
    assert icma.clean_price == pytest.approx(94.635449, abs=1e-6)
    assert icma.clean_price != price("30/360").clean_price


def test_dirty_price_is_clean_plus_accrued() -> None:
    result = bond_price_dated(
        settlement=d("2024-03-07"),
        maturity=d("2031-09-30"),
        coupon_rate=0.0425,
        ytm=0.0391,
        face=1000.0,
    )
    assert result.dirty_price == pytest.approx(result.clean_price + result.accrued_interest, 1e-12)
    assert result.accrued_fraction == pytest.approx(
        result.accrued_days / result.period_days, abs=1e-15
    )


def test_prices_scale_with_face_and_the_per_100_fields_are_the_quote() -> None:
    """Per-face and per-100 are the same number rescaled: the market quotes per 100, but a
    caller pricing 1,000,000 of face wants the cash amount."""
    per_100 = bond_price_dated(
        settlement=d("2008-02-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.0575,
        ytm=0.065,
        face=100.0,
        day_count="30/360",
    )
    per_million = bond_price_dated(
        settlement=d("2008-02-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.0575,
        ytm=0.065,
        face=1_000_000.0,
        day_count="30/360",
    )
    assert per_million.clean_price == pytest.approx(per_100.clean_price * 10_000.0, rel=1e-12)
    assert per_million.clean_price_per_100 == pytest.approx(per_100.clean_price, rel=1e-12)
    assert per_million.dirty_price_per_100 == pytest.approx(per_100.dirty_price, rel=1e-12)
    assert per_million.accrued_interest_per_100 == pytest.approx(
        per_100.accrued_interest, rel=1e-12
    )
    # The rate-risk metrics are per-unit, so they do not scale with face at all (only
    # the last bit moves, from summing present values at a different magnitude).
    assert per_million.macaulay_duration == pytest.approx(per_100.macaulay_duration, rel=1e-12)
    assert per_million.convexity == pytest.approx(per_100.convexity, rel=1e-12)


def test_zero_coupon_bond_accrues_nothing_and_prices_as_one_discounted_redemption() -> None:
    result = bond_price_dated(
        settlement=d("2024-03-07"),
        maturity=d("2029-05-15"),
        coupon_rate=0.0,
        ytm=0.05,
        face=100.0,
        frequency=2,
    )
    assert result.accrued_interest == 0.0
    assert result.current_yield == 0.0
    assert result.clean_price == result.dirty_price
    # Eleven periods remain; the part period runs from 2023-11-15 to 2024-05-15.
    periods = 10.0 + (1.0 - result.accrued_fraction)
    assert result.clean_price == pytest.approx(100.0 / 1.025**periods, rel=1e-12)


def test_current_yield_is_the_coupon_over_the_clean_price() -> None:
    result = bond_price_dated(
        settlement=d("2008-02-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.0575,
        ytm=0.065,
        face=100.0,
        day_count="30/360",
    )
    assert result.current_yield == pytest.approx(5.75 / result.clean_price, rel=1e-12)


# --------------------------------------------------------------------------------------
# Risk metrics over a fractional first period. No published reference quotes these to 1e-6,
# so they are checked against central finite differences of the price function actually
# returned -- which is the property that matters: the analytic derivatives must be the
# derivatives of THIS price, and the on-coupon case alone would not exercise w_k = k - 1 + f.
# --------------------------------------------------------------------------------------

DERIVATIVE_CASES = [
    # (settlement, maturity, coupon_rate, ytm, frequency, day_count)
    ("2008-02-15", "2017-11-15", 0.0575, 0.065, 2, "30/360"),
    ("2008-02-15", "2017-11-15", 0.0575, 0.065, 2, "actual/actual"),
    ("2024-03-07", "2031-09-30", 0.0425, 0.0391, 2, "actual/actual"),
    ("2024-03-07", "2054-05-15", 0.0475, 0.0450, 2, "actual/actual"),  # 30-year
    ("2024-03-07", "2029-05-15", 0.0, 0.05, 2, "actual/actual"),  # zero coupon
    ("2024-07-01", "2034-01-01", 0.03, 0.02, 1, "30/360"),  # annual
    ("2024-02-20", "2027-11-15", 0.06, 0.055, 4, "actual/actual"),  # quarterly
    ("2024-02-20", "2026-11-15", 0.06, 0.055, 12, "30/360"),  # monthly
    ("2024-03-07", "2031-09-30", 0.0425, -0.004, 2, "actual/actual"),  # negative yield
]


@pytest.mark.parametrize(
    ("settlement", "maturity", "coupon_rate", "ytm", "frequency", "day_count"),
    DERIVATIVE_CASES,
)
def test_duration_and_convexity_are_the_derivatives_of_the_dirty_price(
    settlement: str,
    maturity: str,
    coupon_rate: float,
    ytm: float,
    frequency: int,
    day_count: BondDayCount,
) -> None:
    """modified duration = -(1/P) dP/dY and convexity = (1/P) d2P/dY2, on the DIRTY price
    P and the ANNUAL yield Y. Macaulay is modified times (1 + Y/frequency)."""

    def dirty(rate: float) -> float:
        return bond_price_dated(
            settlement=d(settlement),
            maturity=d(maturity),
            coupon_rate=coupon_rate,
            ytm=rate,
            face=100.0,
            frequency=frequency,
            day_count=day_count,
        ).dirty_price

    result = bond_price_dated(
        settlement=d(settlement),
        maturity=d(maturity),
        coupon_rate=coupon_rate,
        ytm=ytm,
        face=100.0,
        frequency=frequency,
        day_count=day_count,
    )
    # A part period really is in play for every case here except the on-coupon ones.
    assert 0.0 < result.accrued_fraction < 1.0

    h = 1e-5
    price, up, down = dirty(ytm), dirty(ytm + h), dirty(ytm - h)
    assert result.modified_duration == pytest.approx(-(up - down) / (2.0 * h * price), rel=1e-6)
    assert result.convexity == pytest.approx((up - 2.0 * price + down) / (h * h * price), rel=1e-5)
    assert result.macaulay_duration == pytest.approx(
        result.modified_duration * (1.0 + ytm / frequency), rel=1e-12
    )


def test_duration_and_convexity_together_predict_a_yield_move() -> None:
    """The practical reading of the two metrics: the second-order price estimate

        dP/P ~= -modified_duration * dY + convexity * dY**2 / 2

    On a 30-year bond over 1 basis point, duration alone is already ~0.1% short; adding
    the convexity term recovers the actual move to within a few parts per million.
    """

    def dirty(rate: float) -> float:
        return bond_price_dated(
            settlement=d("2024-03-07"),
            maturity=d("2054-05-15"),
            coupon_rate=0.0475,
            ytm=rate,
            face=100.0,
        ).dirty_price

    base = bond_price_dated(
        settlement=d("2024-03-07"),
        maturity=d("2054-05-15"),
        coupon_rate=0.0475,
        ytm=0.045,
        face=100.0,
    )
    shift = 0.0001
    actual = dirty(0.045 + shift) - base.dirty_price
    first_order = -base.modified_duration * shift * base.dirty_price
    second_order = 0.5 * base.convexity * shift * shift * base.dirty_price
    assert first_order == pytest.approx(-0.16909, abs=1e-5)
    assert actual == pytest.approx(first_order + second_order, rel=1e-5)
    # Duration alone is measurably short, which is why convexity is reported at all.
    assert abs(actual - first_order) > abs(actual - (first_order + second_order))


# --------------------------------------------------------------------------------------
# bond_ytm_dated: solve the yield from a CLEAN price
# --------------------------------------------------------------------------------------


def test_ytm_matches_the_excel_yield_documentation_example() -> None:
    """Microsoft's YIELD example: settlement 2008-02-15, maturity 2016-11-15, coupon
    5.75%, pr = 95.04287, redemption 100, frequency 2, basis 0 -> 6.5%.

    The documented price is given to five decimals, which bounds how exactly the yield can
    come back: 1e-6 on the yield is two orders of magnitude finer than that rounding.

    https://support.microsoft.com/en-us/office/yield-function-f5f5ca43-c4bd-434f-8bd2-ed3c9727a4fe
    """
    result = bond_ytm_dated(
        settlement=d("2008-02-15"),
        maturity=d("2016-11-15"),
        coupon_rate=0.0575,
        clean_price=95.04287,
        face=100.0,
        frequency=2,
        day_count="30/360",
    )
    assert result.yield_to_maturity == pytest.approx(0.065, abs=1e-6)
    assert result.clean_price == 95.04287
    assert result.accrued_interest == pytest.approx(1.4375, abs=1e-9)
    assert result.dirty_price == pytest.approx(95.04287 + 1.4375, abs=1e-9)


@pytest.mark.parametrize("day_count", ["actual/actual", "30/360"])
@pytest.mark.parametrize("ytm", [-0.005, 0.0, 0.0125, 0.065, 0.19])
def test_price_and_yield_round_trip(day_count: BondDayCount, ytm: float) -> None:
    priced = bond_price_dated(
        settlement=d("2024-03-07"),
        maturity=d("2041-09-30"),
        coupon_rate=0.0425,
        ytm=ytm,
        face=1000.0,
        frequency=2,
        day_count=day_count,
    )
    solved = bond_ytm_dated(
        settlement=d("2024-03-07"),
        maturity=d("2041-09-30"),
        coupon_rate=0.0425,
        clean_price=priced.clean_price,
        face=1000.0,
        frequency=2,
        day_count=day_count,
    )
    assert solved.yield_to_maturity == pytest.approx(ytm, abs=1e-9)
    assert solved.accrued_interest == pytest.approx(priced.accrued_interest, rel=1e-12)
    assert solved.dirty_price == pytest.approx(priced.dirty_price, rel=1e-12)


def test_ytm_on_a_coupon_date_equals_the_on_coupon_solver() -> None:
    dated = bond_ytm_dated(
        settlement=d("2007-11-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.05,
        clean_price=925.61,
        face=1000.0,
        frequency=2,
    )
    on_coupon = bond_ytm(
        face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, price=925.61, frequency=2
    )
    assert dated.yield_to_maturity == pytest.approx(on_coupon.yield_to_maturity, abs=1e-9)
    assert dated.accrued_interest == 0.0
    assert dated.dirty_price == dated.clean_price


def test_a_bond_priced_at_par_on_a_coupon_date_yields_its_coupon() -> None:
    result = bond_ytm_dated(
        settlement=d("2007-11-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.06,
        clean_price=100.0,
        face=100.0,
        frequency=2,
    )
    assert result.yield_to_maturity == pytest.approx(0.06, abs=1e-9)


# --------------------------------------------------------------------------------------
# Input validation. Every message is written for the model that will read it.
# --------------------------------------------------------------------------------------


def test_the_dated_span_bound_matches_the_tool_boundary_bound() -> None:
    """The on-coupon tools bound their coupon loop with a years_to_maturity Field; the
    dated ones cannot, because the loop length is a relationship between two arguments.
    The two bounds must still be the same number, so pin them together.
    """
    assert MAX_BOND_SPAN_YEARS == MAX_BOND_YEARS


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"settlement": d("2017-11-15")}, "strictly before maturity"),
        ({"settlement": d("2017-11-16")}, "strictly before maturity"),
        ({"face": 0.0}, "face must be positive"),
        ({"face": -100.0}, "face must be positive"),
        ({"frequency": 5}, "divide 12 evenly"),
        ({"frequency": 7}, "divide 12 evenly"),
        ({"frequency": 0}, "divide 12 evenly"),
        ({"frequency": -2}, "divide 12 evenly"),
        ({"settlement": d("1900-01-01")}, "at most 100 years"),
    ],
)
def test_invalid_dated_inputs_are_rejected(kwargs: dict[str, object], message: str) -> None:
    defaults: dict[str, object] = {
        "settlement": d("2008-02-15"),
        "maturity": d("2017-11-15"),
        "coupon_rate": 0.05,
        "face": 100.0,
        "frequency": 2,
    }
    with pytest.raises(InvalidInput, match=message):
        bond_price_dated(ytm=0.06, **{**defaults, **kwargs})  # type: ignore[arg-type]
    with pytest.raises(InvalidInput, match=message):
        bond_ytm_dated(clean_price=95.0, **{**defaults, **kwargs})  # type: ignore[arg-type]


def test_a_span_exactly_at_the_bound_is_accepted() -> None:
    """100 years is allowed; the bound rejects only spans longer than it."""
    result = bond_price_dated(
        settlement=d("1917-11-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.05,
        ytm=0.06,
        face=100.0,
        frequency=2,
    )
    assert result.periods_remaining == 200


@pytest.mark.parametrize(("ytm", "frequency"), [(-2.0, 2), (-3.0, 2), (-1.0, 1), (-12.0, 12)])
def test_a_yield_at_or_below_minus_frequency_is_rejected(ytm: float, frequency: int) -> None:
    """The binding constraint is on the discount base 1 + ytm/frequency, exactly as for
    bond_price -- not ytm > -1."""
    with pytest.raises(InvalidInput, match="greater than -frequency"):
        bond_price_dated(
            settlement=d("2008-02-15"),
            maturity=d("2017-11-15"),
            coupon_rate=0.05,
            ytm=ytm,
            face=100.0,
            frequency=frequency,
        )


def test_a_deeply_negative_yield_above_minus_frequency_still_prices() -> None:
    result = bond_price_dated(
        settlement=d("2008-02-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.05,
        ytm=-1.5,
        face=100.0,
        frequency=2,
    )
    assert result.dirty_price > 0.0


def test_a_yield_that_would_leave_a_non_positive_clean_price_is_rejected() -> None:
    """Accrued interest is a fixed cash amount, but the discounted cashflows shrink with the
    yield, so a high enough yield drives clean = dirty - accrued to zero and then negative.
    bond_ytm_dated rejects such a quote, so bond_price_dated must not emit one: otherwise the
    two tools stop being inverses and current_yield comes back negative (or divides by zero).
    """
    for ytm in (12.0, 20.0):
        with pytest.raises(InvalidInput, match="non-positive clean price"):
            bond_price_dated(
                settlement=d("2024-04-01"),
                maturity=d("2044-01-15"),
                coupon_rate=0.05,
                ytm=ytm,
            )


@pytest.mark.parametrize("clean_price", [0.0, -1.0])
def test_a_non_positive_clean_price_is_rejected(clean_price: float) -> None:
    with pytest.raises(InvalidInput, match="clean_price must be positive"):
        bond_ytm_dated(
            settlement=d("2008-02-15"),
            maturity=d("2017-11-15"),
            coupon_rate=0.05,
            clean_price=clean_price,
        )


# --------------------------------------------------------------------------------------
# first_period_discount: the street convention compounds across the part period, while the
# US Treasury's own regulation uses SIMPLE interest over it. The two are not interchangeable
# -- on 31 CFR 356 appendix B example I.D they differ by 0.0077 per 100 -- so the Treasury
# examples with a real part period can only be reproduced by asking for their convention.
# --------------------------------------------------------------------------------------


def test_treasury_simple_stub_reproduces_appendix_b_example_i_d() -> None:
    """31 CFR 356 appendix B, example I.D: a 9-1/2% 10-year note accruing from 1985-11-15,
    issued 1985-11-29, due 1995-11-15, coupons on May 15 and November 15, at a yield of
    9.54%. The appendix gives r = 167, s = 181, n = 19, A = 0.367403 and P = 99.730918.

    Its formula divides by [1 + (r/s)(i/2)] -- simple interest over the part period.

    https://www.govinfo.gov/content/pkg/CFR-2025-title31-vol2/pdf/CFR-2025-title31-vol2-part356-appB.pdf
    """
    result = bond_price_dated(
        settlement=d("1985-11-29"),
        maturity=d("1995-11-15"),
        coupon_rate=0.095,
        ytm=0.0954,
        face=100.0,
        frequency=2,
        first_period_discount="simple",
    )
    assert (result.accrued_days, result.period_days) == (14.0, 181.0)
    assert result.periods_remaining == 20
    assert result.accrued_interest == pytest.approx(0.367403, abs=1e-6)
    assert result.clean_price == pytest.approx(99.730918, abs=1e-6)
    assert result.first_period_discount == "simple"


def test_an_odd_short_first_coupon_period_is_out_of_scope() -> None:
    """31 CFR 356 appendix B, example I.B is an 8-1/2% 2-year note ISSUED 1990-04-02 and due
    1992-03-31, whose first interest payment period is SHORT: it runs from the issue date to
    1990-09-30, not a full six months, so its first coupon pays only (C/2)(r/s) of a full
    coupon. The appendix prices it at 99.838183.

    This module generates a regular schedule and pays a full coupon on every date, so it
    prices a *different* bond -- a whole first coupon rather than a stub one -- and comes out
    slightly higher. That gap is the documented limitation, asserted here so it cannot be
    mistaken for a rounding difference and so the scope boundary is pinned by a test.

    The schedule itself is right: the same example's published r = 181 / s = 183 day counts
    are what test_coupon_schedule_matches_treasury_month_end_example checks. Only the odd
    first coupon AMOUNT is unsupported.
    """
    result = bond_price_dated(
        settlement=d("1990-04-02"),
        maturity=d("1992-03-31"),
        coupon_rate=0.085,
        ytm=0.0859,
        face=100.0,
        frequency=2,
        first_period_discount="simple",
    )
    assert (result.accrued_days, result.period_days) == (2.0, 183.0)
    assert result.periods_remaining == 4
    appendix_price = 99.838183
    assert result.clean_price == pytest.approx(99.836290, abs=1e-6)
    assert result.clean_price < appendix_price
    # Small in absolute terms, but ~19x the 1e-6 tolerance the regular-period references
    # are held to -- a real modelling difference, not noise.
    assert appendix_price - result.clean_price == pytest.approx(0.001893, abs=1e-6)


def test_the_two_first_period_conventions_disagree_by_a_material_amount() -> None:
    """The reason this is an explicit choice and not an implementation detail: on the
    appendix's own example the conventions differ in the third decimal of the price."""

    def price(discount: FirstPeriodDiscount) -> float:
        return bond_price_dated(
            settlement=d("1985-11-29"),
            maturity=d("1995-11-15"),
            coupon_rate=0.095,
            ytm=0.0954,
            face=100.0,
            frequency=2,
            first_period_discount=discount,
        ).clean_price

    assert price("simple") == pytest.approx(99.730918, abs=1e-6)
    assert price("compound") == pytest.approx(99.738573, abs=1e-6)
    assert price("compound") - price("simple") == pytest.approx(0.007655, abs=1e-6)


def test_in_the_final_coupon_period_simple_is_what_matches_excel() -> None:
    """Excel's PRICE/YIELD documentation gives a SEPARATE formula for "one coupon period or
    less to redemption" that discounts the stub with simple interest:

        P = (redemption + coupon) / (1 + (DSR/E) * yield/frequency) - (A/E) * coupon

    (Microsoft, "YIELD function" -- https://support.microsoft.com/en-us/office/
    yield-function-f5f5ca43-c4bd-434f-8bd2-ed3c9727a4fe.) That is exactly this module's
    'simple' branch, so it is 'simple', not the default 'compound', that reproduces Excel
    inside the last period. Pinned here because every other Excel/Treasury reference in this
    file has n >= 18, leaving the n == 1 boundary otherwise untested.
    """

    def price(ytm: float, discount: FirstPeriodDiscount) -> BondDatedAnalytics:
        # A 6% semiannual bond maturing 2024-07-15, settling 2024-04-01: one coupon left.
        return bond_price_dated(
            settlement=d("2024-04-01"),
            maturity=d("2024-07-15"),
            coupon_rate=0.06,
            ytm=ytm,
            frequency=2,
            day_count="30/360",
            first_period_discount=discount,
        )

    for ytm, excel_price, compound_price in (
        (0.065, 99.834871, 99.847465),
        (0.20, 96.107283, 96.214664),
    ):
        simple = price(ytm, "simple")
        compound = price(ytm, "compound")
        assert simple.periods_remaining == 1
        # The Excel closed form, evaluated from the schedule this module resolved.
        coupon, a, e = 100 * 0.06 / 2, simple.accrued_days, simple.period_days
        closed_form = (100 + coupon) / (1 + (e - a) / e * ytm / 2) - a / e * coupon
        assert closed_form == pytest.approx(excel_price, abs=1e-6)
        assert simple.clean_price_per_100 == pytest.approx(excel_price, abs=1e-6)
        # The default compounds throughout instead, and is measurably higher. Documented, not
        # a defect: 'compound' is one uniform formula for every n, which is what the street
        # and LibreOffice do -- but it is why the Excel-equivalence claims name this exception.
        assert compound.clean_price_per_100 == pytest.approx(compound_price, abs=1e-6)
        assert compound.clean_price_per_100 > simple.clean_price_per_100


def test_compound_is_the_default() -> None:
    """The task's 'standard street convention', and what Excel's PRICE implements while more
    than one coupon period remains."""
    explicit = bond_price_dated(
        settlement=d("1985-11-29"),
        maturity=d("1995-11-15"),
        coupon_rate=0.095,
        ytm=0.0954,
        first_period_discount="compound",
    )
    default = bond_price_dated(
        settlement=d("1985-11-29"),
        maturity=d("1995-11-15"),
        coupon_rate=0.095,
        ytm=0.0954,
    )
    assert default.first_period_discount == "compound"
    assert default.clean_price == explicit.clean_price


@pytest.mark.parametrize("first_period_discount", ["compound", "simple"])
def test_the_conventions_agree_on_a_coupon_date(
    first_period_discount: FirstPeriodDiscount,
) -> None:
    """With no part period the stub factor is (1 + y) either way, so both conventions must
    collapse to the same price -- and to the on-coupon calculator."""
    result = bond_price_dated(
        settlement=d("2007-11-15"),
        maturity=d("2017-11-15"),
        coupon_rate=0.05,
        ytm=0.06,
        face=1000.0,
        frequency=2,
        first_period_discount=first_period_discount,
    )
    expected = bond_price(
        face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2
    )
    assert result.clean_price == pytest.approx(expected.price, rel=1e-12)
    assert result.macaulay_duration == pytest.approx(expected.macaulay_duration, rel=1e-12)
    assert result.modified_duration == pytest.approx(expected.modified_duration, rel=1e-12)
    assert result.convexity == pytest.approx(expected.convexity, rel=1e-12)


@pytest.mark.parametrize(
    ("settlement", "maturity", "coupon_rate", "ytm", "frequency", "day_count"),
    DERIVATIVE_CASES,
)
def test_simple_stub_duration_and_convexity_are_also_derivatives(
    settlement: str,
    maturity: str,
    coupon_rate: float,
    ytm: float,
    frequency: int,
    day_count: BondDayCount,
) -> None:
    """The simple-stub branch has its own analytic derivatives (the stub factor depends on
    the yield too), so they need the same finite-difference check as the compound ones."""

    def dirty(rate: float) -> float:
        return bond_price_dated(
            settlement=d(settlement),
            maturity=d(maturity),
            coupon_rate=coupon_rate,
            ytm=rate,
            face=100.0,
            frequency=frequency,
            day_count=day_count,
            first_period_discount="simple",
        ).dirty_price

    result = bond_price_dated(
        settlement=d(settlement),
        maturity=d(maturity),
        coupon_rate=coupon_rate,
        ytm=ytm,
        face=100.0,
        frequency=frequency,
        day_count=day_count,
        first_period_discount="simple",
    )
    h = 1e-5
    price, up, down = dirty(ytm), dirty(ytm + h), dirty(ytm - h)
    assert result.modified_duration == pytest.approx(-(up - down) / (2.0 * h * price), rel=1e-6)
    assert result.convexity == pytest.approx((up - 2.0 * price + down) / (h * h * price), rel=1e-5)
    assert result.macaulay_duration == pytest.approx(
        result.modified_duration * (1.0 + ytm / frequency), rel=1e-12
    )


@pytest.mark.parametrize("ytm", [0.0, 0.0425, 0.11])
def test_simple_stub_price_and_yield_round_trip(ytm: float) -> None:
    priced = bond_price_dated(
        settlement=d("1985-11-29"),
        maturity=d("1995-11-15"),
        coupon_rate=0.095,
        ytm=ytm,
        face=100.0,
        first_period_discount="simple",
    )
    solved = bond_ytm_dated(
        settlement=d("1985-11-29"),
        maturity=d("1995-11-15"),
        coupon_rate=0.095,
        clean_price=priced.clean_price,
        face=100.0,
        first_period_discount="simple",
    )
    assert solved.yield_to_maturity == pytest.approx(ytm, abs=1e-9)
    assert solved.first_period_discount == "simple"
