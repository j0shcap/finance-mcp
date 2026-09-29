"""Date-based bond pricing: coupon schedule, day counts, and street-convention analytics.

Reference values are cited inline. The Microsoft examples are the ones shipped in the
Excel function documentation; the Treasury examples are the worked examples in the
Uniform Offering Circular's formula appendix (31 CFR part 356, appendix B).
"""

import datetime

import pytest

from finance_mcp.data.calculators import (
    _add_months,
    _bond_metrics,
    _coupon_schedule,
    _days_30_360_us,
    bond_price,
    bond_price_dated,
)
from finance_mcp.data.models import BondDatedAnalytics, BondDayCount


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
