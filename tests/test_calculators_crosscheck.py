"""Property-based cross-checks of the calculators against independent implementations.

The unit tests pin chosen examples; these sweep the input space. Each property either
compares against code this project did not write (numpy-financial's TVM/NPV/IRR/MIRR, a
scipy root-finder, a closed-form numpy bond price), or checks an identity that holds only
if the implementation is right (solve for pv, recompute fv, get the pv back; NPV at every
reported IRR is zero; a loan amortizes to exactly nothing).

Domains are bounded to what the tools are for - rates a few tens of percent either way,
terms up to a few hundred periods - so a failure means a wrong answer, not float noise at
an extreme nobody prices. The run is derandomized (see tests/conftest.py), so `make check`
explores the same examples every time.
"""

import datetime
import math

import numpy as np
import numpy_financial as npf
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from scipy.optimize import brentq

from finance_mcp.data import calculators
from finance_mcp.data.errors import InvalidInput
from finance_mcp.data.models import DatedCashflow

# numpy-financial evaluates both arms of its np.where(rate == 0, ...), so a zero rate makes
# it warn about the division in the arm it then discards.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")

#: Zero, or a rate clear of zero. numpy-financial computes (1+r)**n - 1 naively, which
#: carries a relative error of ~1e-16/|rate| - 1e-8 at a rate of 1e-8, and the whole annuity
#: term (or a nan) below ~1e-16 - so near zero it is no oracle. The calculators handle that
#: limit exactly; test_calculators_logic.py pins it against the zero-rate answer instead.
RATES = st.one_of(
    st.just(0.0), st.floats(min_value=-0.1, max_value=0.3).filter(lambda r: abs(r) > 1e-6)
)
POSITIVE_RATES = st.floats(min_value=0.001, max_value=0.3)
PERIODS = st.integers(min_value=1, max_value=360)
AMOUNTS = st.floats(min_value=-1e6, max_value=1e6)
PAYMENTS = st.floats(min_value=-1e4, max_value=1e4)
WHEN = st.sampled_from(["end", "begin"])


def close(actual: float, expected: float, scale: float = 1.0) -> bool:
    """Relative agreement with an absolute floor proportional to the problem's size.

    The floor is what makes a sum of large terms that cancels to ~0 comparable: its
    rounding error is set by the terms, not by the (tiny) result.
    """
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9 * max(scale, 1.0))


def tvm(solve_for: str, **known: float | str) -> float:
    return calculators.time_value_of_money(solve_for=solve_for, **known).solved_value  # type: ignore[arg-type]


# --- time value of money --------------------------------------------------------------


@given(rate=RATES, nper=PERIODS, pmt=PAYMENTS, pv=AMOUNTS, when=WHEN)
def test_fv_matches_numpy_financial(
    rate: float, nper: int, pmt: float, pv: float, when: str
) -> None:
    expected = float(npf.fv(rate, nper, pmt, pv, when=when))
    actual = tvm("fv", rate=rate, nper=nper, pmt=pmt, pv=pv, when=when)
    assert close(actual, expected, scale=abs(expected) + abs(pv) + abs(pmt) * nper)


@given(rate=RATES, nper=PERIODS, pmt=PAYMENTS, fv=AMOUNTS, when=WHEN)
def test_pv_matches_numpy_financial(
    rate: float, nper: int, pmt: float, fv: float, when: str
) -> None:
    expected = float(npf.pv(rate, nper, pmt, fv, when=when))
    actual = tvm("pv", rate=rate, nper=nper, pmt=pmt, fv=fv, when=when)
    assert close(actual, expected, scale=abs(expected) + abs(fv) + abs(pmt) * nper)


@given(rate=RATES, nper=PERIODS, pv=AMOUNTS, fv=AMOUNTS, when=WHEN)
def test_pmt_matches_numpy_financial(
    rate: float, nper: int, pv: float, fv: float, when: str
) -> None:
    expected = float(npf.pmt(rate, nper, pv, fv, when=when))
    actual = tvm("pmt", rate=rate, nper=nper, pv=pv, fv=fv, when=when)
    assert close(actual, expected, scale=abs(expected) + (abs(pv) + abs(fv)) / nper)


@given(rate=RATES, nper=PERIODS, pmt=PAYMENTS, pv=AMOUNTS, when=WHEN)
def test_pv_to_fv_to_pv_round_trips(
    rate: float, nper: int, pmt: float, pv: float, when: str
) -> None:
    fv = tvm("fv", rate=rate, nper=nper, pmt=pmt, pv=pv, when=when)
    back = tvm("pv", rate=rate, nper=nper, pmt=pmt, fv=fv, when=when)
    growth = (1 + rate) ** nper
    # Undoing the growth factor divides the fv's rounding error back down by it.
    assert close(back, pv, scale=abs(pv) + abs(pmt) * nper + abs(fv) / growth)


@given(rate=POSITIVE_RATES, nper=PERIODS, pmt=st.floats(-1e4, -1.0), pv=st.floats(-1e6, 0.0))
def test_solved_nper_reproduces_the_term(rate: float, nper: int, pmt: float, pv: float) -> None:
    """An investor paying in (pv, pmt <= 0) has one term that reaches a given balance."""
    fv = float(npf.fv(rate, nper, pmt, pv))
    assert tvm("nper", rate=rate, pmt=pmt, pv=pv, fv=fv) == pytest.approx(nper, rel=1e-7)


@given(
    rate=POSITIVE_RATES,
    nper=st.integers(2, 360),
    pmt=st.floats(-1e4, 0.0),
    pv=st.floats(-1e6, -1.0),
)
def test_solved_rate_reproduces_the_rate(rate: float, nper: int, pmt: float, pv: float) -> None:
    """Paying in only, the future value rises monotonically in the rate: one root."""
    fv = float(npf.fv(rate, nper, pmt, pv))
    assert tvm("rate", nper=nper, pmt=pmt, pv=pv, fv=fv) == pytest.approx(rate, rel=1e-7)


# --- npv / irr / mirr -----------------------------------------------------------------

CASHFLOWS = st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=40)


@given(rate=RATES, cashflows=CASHFLOWS)
def test_npv_matches_numpy_financial(rate: float, cashflows: list[float]) -> None:
    expected = float(npf.npv(rate, cashflows))
    actual = calculators.npv(rate, cashflows).npv
    assert close(actual, expected, scale=sum(abs(c) for c in cashflows))


@st.composite
def conventional_cashflows(draw: st.DrawFn) -> list[float]:
    """An outlay followed by inflows summing to more than it: one IRR, and it is positive.

    Inflows are capped at 100x the outlay, which keeps the IRR inside the range irr()
    documents it searches (up to 1,000,000% per period).
    """
    outlay = draw(st.floats(min_value=1.0, max_value=1e6))
    inflows = draw(
        st.lists(st.floats(min_value=0.0, max_value=outlay * 100), min_size=1, max_size=30)
    )
    assume(outlay * 1.001 < sum(inflows) <= outlay * 100)
    return [-outlay, *inflows]


@given(cashflows=conventional_cashflows())
def test_conventional_irr_matches_numpy_financial_and_is_unique(cashflows: list[float]) -> None:
    result = calculators.irr(cashflows)
    assert result.is_unique
    assert result.irr == pytest.approx(float(npf.irr(cashflows)), rel=1e-6, abs=1e-9)


@given(cashflows=st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=20))
def test_npv_at_every_reported_irr_is_zero(cashflows: list[float]) -> None:
    """Holds for non-conventional flows too, where there may be several roots or none."""
    assume(any(c < 0 for c in cashflows) and any(c > 0 for c in cashflows))
    try:
        result = calculators.irr(cashflows)
    except InvalidInput:
        return  # a sign change with no real root in the searched range is a clean refusal
    scale = sum(abs(c) for c in cashflows)
    for root in result.all_irrs:
        # Near rate = -1 the NPV is so steep that a root accurate to 1e-12 can still leave
        # a visible residual, so a root is either a sign change within a hair of it - ten
        # times the 1e-12 width the bisection promises - or, for a tangent root, a
        # near-zero NPV.
        hair = 1e-11
        below = calculators.npv(root - hair, cashflows).npv
        above = calculators.npv(root + hair, cashflows).npv
        residual = calculators.npv(root, cashflows).npv
        assert below * above <= 0.0 or abs(residual) <= 1e-6 * scale, (
            f"{root} is not an IRR: NPV {residual}, {below} just below, {above} just above"
        )


@given(
    cashflows=conventional_cashflows(),
    finance_rate=st.floats(0.0, 0.3),
    reinvest_rate=st.floats(0.0, 0.3),
)
def test_mirr_matches_numpy_financial(
    cashflows: list[float], finance_rate: float, reinvest_rate: float
) -> None:
    expected = float(npf.mirr(cashflows, finance_rate, reinvest_rate))
    actual = calculators.mirr(cashflows, finance_rate, reinvest_rate).mirr
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-12)


# --- dated cashflows ------------------------------------------------------------------

BASE_DATE = datetime.date(2020, 1, 1)


@st.composite
def dated_cashflows(draw: st.DrawFn) -> list[DatedCashflow]:
    """A conventional investment on irregular dates (strictly after the outlay)."""
    flows = draw(conventional_cashflows())
    offsets = sorted(
        draw(st.sets(st.integers(1, 3650), min_size=len(flows) - 1, max_size=len(flows) - 1))
    )
    days = [0, *offsets]
    return [
        DatedCashflow(date=BASE_DATE + datetime.timedelta(days=d), amount=a)
        for d, a in zip(days, flows, strict=True)
    ]


def _reference_xnpv(rate: float, cashflows: list[DatedCashflow]) -> float:
    """Excel's XNPV, written directly: Actual/365 from the earliest date."""
    start = min(cf.date for cf in cashflows)
    years = np.array([(cf.date - start).days / 365.0 for cf in cashflows])
    amounts = np.array([cf.amount for cf in cashflows])
    return float(np.sum(amounts / (1.0 + rate) ** years))


@given(rate=RATES, cashflows=dated_cashflows())
def test_xnpv_matches_the_excel_formula(rate: float, cashflows: list[DatedCashflow]) -> None:
    expected = _reference_xnpv(rate, cashflows)
    actual = calculators.xnpv(rate, cashflows).npv
    assert close(actual, expected, scale=sum(abs(cf.amount) for cf in cashflows))


@given(cashflows=dated_cashflows())
def test_xirr_matches_a_scipy_root_of_the_excel_formula(cashflows: list[DatedCashflow]) -> None:
    """Conventional flows with a positive net: XNPV falls monotonically through one root > 0.

    Only where that root is inside xirr's documented search range: 100x returned a day
    after the outlay is an annual IRR with hundreds of digits.
    """
    assume(_reference_xnpv(1e4, cashflows) < 0.0)
    expected = brentq(lambda r: _reference_xnpv(r, cashflows), 0.0, 1e4, xtol=1e-14)
    actual = calculators.xirr(cashflows).irr
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9)


# --- bonds ----------------------------------------------------------------------------

FREQUENCIES = st.sampled_from([1, 2, 4, 12])
COUPONS = st.floats(min_value=0.0, max_value=0.15)
YIELDS = st.floats(min_value=0.0, max_value=0.2)


def _reference_bond_price(
    face: float, coupon_rate: float, years: int, ytm: float, frequency: int
) -> float:
    """Discount every coupon and the redemption at the per-period yield, term by term."""
    n = years * frequency
    y = ytm / frequency
    t = np.arange(1, n + 1)
    flows = np.full(n, face * coupon_rate / frequency)
    flows[-1] += face
    return float(np.sum(flows / (1.0 + y) ** t))


@given(
    face=st.floats(100.0, 1e6),
    coupon_rate=COUPONS,
    years=st.integers(1, 30),
    ytm=YIELDS,
    frequency=FREQUENCIES,
)
def test_bond_price_matches_the_discounted_cashflows(
    face: float, coupon_rate: float, years: int, ytm: float, frequency: int
) -> None:
    expected = _reference_bond_price(face, coupon_rate, years, ytm, frequency)
    actual = calculators.bond_price(face, coupon_rate, years, ytm, frequency).price
    assert actual == pytest.approx(expected, rel=1e-10)


@given(coupon_rate=COUPONS, years=st.integers(1, 30), ytm=YIELDS, frequency=FREQUENCIES)
def test_bond_price_to_ytm_round_trips(
    coupon_rate: float, years: int, ytm: float, frequency: int
) -> None:
    price = calculators.bond_price(1000.0, coupon_rate, years, ytm, frequency).price
    back = calculators.bond_ytm(1000.0, coupon_rate, years, price, frequency).yield_to_maturity
    assert back == pytest.approx(ytm, abs=1e-8)


@given(
    days_to_maturity=st.integers(30, 30 * 365),
    settle_offset=st.integers(0, 3650),
    coupon_rate=COUPONS,
    ytm=YIELDS,
    frequency=st.sampled_from([1, 2, 4, 12]),
    day_count=st.sampled_from(["actual/actual", "30/360"]),
    first_period_discount=st.sampled_from(["compound", "simple"]),
)
def test_dated_bond_price_to_ytm_round_trips(
    days_to_maturity: int,
    settle_offset: int,
    coupon_rate: float,
    ytm: float,
    frequency: int,
    day_count: str,
    first_period_discount: str,
) -> None:
    """Settlement anywhere in a coupon period, under every day count and stub convention."""
    settlement = datetime.date(2020, 1, 1) + datetime.timedelta(days=settle_offset)
    maturity = settlement + datetime.timedelta(days=days_to_maturity)
    conventions = {
        "frequency": frequency,
        "day_count": day_count,
        "first_period_discount": first_period_discount,
    }
    priced = calculators.bond_price_dated(
        settlement=settlement,
        maturity=maturity,
        coupon_rate=coupon_rate,
        ytm=ytm,
        **conventions,  # type: ignore[arg-type]
    )
    solved = calculators.bond_ytm_dated(
        settlement=settlement,
        maturity=maturity,
        coupon_rate=coupon_rate,
        clean_price=priced.clean_price,
        **conventions,  # type: ignore[arg-type]
    )
    assert solved.yield_to_maturity == pytest.approx(ytm, abs=1e-8)
    assert solved.accrued_interest == pytest.approx(priced.accrued_interest, rel=1e-12)


# --- loans and rates ------------------------------------------------------------------


@given(
    principal=st.floats(1_000.0, 5e6),
    # Clear of zero for the same reason as RATES: there numpy-financial divides by zero.
    annual_rate=st.one_of(st.just(0.0), st.floats(1e-6, 0.25)),
    term_months=st.integers(1, 480),
)
def test_loan_schedule_amortizes_to_zero_and_matches_numpy_financial(
    principal: float, annual_rate: float, term_months: int
) -> None:
    loan = calculators.loan_schedule(principal, annual_rate, term_months, include_schedule=True)
    rows = loan.schedule
    assert rows is not None and len(rows) == term_months == loan.n_payments

    # Each row is rounded to cents (documented on loan_schedule), and each total once more
    # at the end: half a cent per rounding, plus float noise at an exact half cent.
    cent = 0.005 + 1e-9
    rows_slack = cent * (term_months + 1)
    expected_payment = -float(npf.pmt(annual_rate / 12, term_months, principal))
    assert abs(loan.monthly_payment - expected_payment) <= cent
    assert abs(sum(r.principal for r in rows) - principal) <= rows_slack
    assert abs(sum(r.interest for r in rows) - loan.total_interest) <= rows_slack
    assert abs(loan.total_paid - (principal + loan.total_interest)) <= 2 * cent
    assert rows[-1].balance == 0.0


@given(
    rate=st.floats(0.0, 0.5),
    periods_per_year=st.integers(1, 365),
    compounding=st.sampled_from(["discrete", "continuous"]),
)
def test_rate_conversion_round_trips(rate: float, periods_per_year: int, compounding: str) -> None:
    effective = calculators.convert_rate(
        rate,
        periods_per_year,
        "nominal_to_effective",
        compounding,  # type: ignore[arg-type]
    ).converted_rate
    nominal = calculators.convert_rate(
        effective,
        periods_per_year,
        "effective_to_nominal",
        compounding,  # type: ignore[arg-type]
    ).converted_rate
    assert nominal == pytest.approx(rate, rel=1e-9, abs=1e-12)
