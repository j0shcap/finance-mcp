import datetime
import math

import pytest

from finance_mcp.data.calculators import (
    _bisect_bracket,
    _find_all_roots,
    _minimize,
    bond_price,
    bond_ytm,
    convert_rate,
    irr,
    loan_schedule,
    mirr,
    npv,
    time_value_of_money,
    xirr,
    xnpv,
)
from finance_mcp.data.errors import InvalidInput
from finance_mcp.data.models import DatedCashflow


def test_future_value_compound_interest() -> None:
    # $1000 invested (outflow, so pv negative) at 5%/yr for 10 yrs, no payments.
    result = time_value_of_money(solve_for="fv", pv=-1000.0, pmt=0.0, rate=0.05, nper=10.0)
    assert result.solved_for == "fv"
    assert result.solved_value == pytest.approx(1628.894627, rel=1e-6)


def test_present_value() -> None:
    # What deposit grows to $1628.894627 at 5% over 10 yrs? -> 1000 (outflow).
    result = time_value_of_money(solve_for="pv", fv=1628.894627, pmt=0.0, rate=0.05, nper=10.0)
    assert result.solved_value == pytest.approx(-1000.0, rel=1e-6)


def test_payment_annuity() -> None:
    # Loan of 200000 (inflow) at 0.5%/mo for 360 mo, fv 0 -> payment ~ -1199.10 (outflow).
    result = time_value_of_money(solve_for="pmt", pv=200000.0, fv=0.0, rate=0.005, nper=360.0)
    assert result.solved_value == pytest.approx(-1199.101, rel=1e-5)


def test_rate_is_cagr_when_no_payments() -> None:
    # Solve rate: 1000 -> 2000 over 10 yrs, no payments -> CAGR 7.1773%.
    result = time_value_of_money(solve_for="rate", pv=-1000.0, fv=2000.0, pmt=0.0, nper=10.0)
    assert result.solved_value == pytest.approx(0.0717734625, rel=1e-6)


def test_nper() -> None:
    # Periods to grow 1000 -> 2000 at 5%: ln(2)/ln(1.05) ~ 14.2067.
    result = time_value_of_money(solve_for="nper", pv=-1000.0, fv=2000.0, pmt=0.0, rate=0.05)
    assert result.solved_value == pytest.approx(14.2067, rel=1e-4)


def test_zero_rate_payment() -> None:
    # r = 0: pv + pmt*n + fv = 0. pv=-1200, n=12, fv=0 -> pmt = 100.
    result = time_value_of_money(solve_for="pmt", pv=-1200.0, fv=0.0, rate=0.0, nper=12.0)
    assert result.solved_value == pytest.approx(100.0, rel=1e-9)


def test_missing_required_input_raises() -> None:
    # Solving for fv requires pv, pmt, rate, nper; omit rate.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="fv", pv=-1000.0, pmt=0.0, nper=10.0)


def test_result_echoes_all_fields() -> None:
    result = time_value_of_money(solve_for="fv", pv=-1000.0, pmt=0.0, rate=0.05, nper=10.0)
    assert result.rate == 0.05
    assert result.nper == 10.0
    assert math.isclose(result.fv, result.solved_value)


def test_loan_payment_30yr() -> None:
    # 200k at 6% annual, 360 months -> payment ~ 1199.10.
    result = loan_schedule(
        principal=200000.0, annual_rate=0.06, term_months=360, include_schedule=True
    )
    assert result.monthly_payment == pytest.approx(1199.101, rel=1e-5)
    assert result.n_payments == 360
    assert result.total_interest == pytest.approx(result.total_paid - 200000.0, rel=1e-9)
    assert result.schedule[-1].balance == pytest.approx(0.0, abs=1e-2)


def test_loan_zero_interest() -> None:
    # 12000 at 0% over 12 months -> 1000/mo, no interest.
    result = loan_schedule(principal=12000.0, annual_rate=0.0, term_months=12)
    assert result.monthly_payment == pytest.approx(1000.0, rel=1e-9)
    assert result.total_interest == pytest.approx(0.0, abs=1e-6)
    assert result.n_payments == 12


def test_loan_extra_payment_shortens_term() -> None:
    base = loan_schedule(principal=200000.0, annual_rate=0.06, term_months=360)
    faster = loan_schedule(
        principal=200000.0, annual_rate=0.06, term_months=360, extra_payment=200.0
    )
    assert faster.n_payments < base.n_payments
    assert faster.total_interest < base.total_interest


def test_loan_invalid_term_raises() -> None:
    with pytest.raises(InvalidInput):
        loan_schedule(principal=1000.0, annual_rate=0.05, term_months=0)


def test_fv_zero_rate_with_payments() -> None:
    # r = 0: fv = -(pv + pmt*n). pv=-1000, pmt=-50, n=12 -> 1600.
    result = time_value_of_money(solve_for="fv", pv=-1000.0, pmt=-50.0, rate=0.0, nper=12.0)
    assert result.solved_value == pytest.approx(1600.0, rel=1e-9)


def test_pv_zero_rate_with_payments() -> None:
    result = time_value_of_money(solve_for="pv", fv=1600.0, pmt=-50.0, rate=0.0, nper=12.0)
    assert result.solved_value == pytest.approx(-1000.0, rel=1e-9)


def test_nper_zero_rate_with_payments() -> None:
    # r = 0: nper = -(pv + fv)/pmt. pv=-1200, fv=0, pmt=100 -> 12.
    result = time_value_of_money(solve_for="nper", pv=-1200.0, fv=0.0, pmt=100.0, rate=0.0)
    assert result.solved_value == pytest.approx(12.0, rel=1e-9)


def test_nper_general_with_payments() -> None:
    # 12-month annuity at 1%/mo, payment that amortizes 1000 -> nper ~ 12.
    result = time_value_of_money(solve_for="nper", pv=1000.0, fv=0.0, pmt=-88.84879, rate=0.01)
    assert result.solved_value == pytest.approx(12.0, rel=1e-3)


def test_rate_general_with_payments() -> None:
    # Solve the periodic rate of a 12-month annuity amortizing 1000 -> ~1%/mo.
    result = time_value_of_money(solve_for="rate", pv=1000.0, fv=0.0, pmt=-88.84879, nper=12.0)
    assert result.solved_value == pytest.approx(0.01, rel=1e-4)


def test_nper_zero_rate_and_zero_pmt_raises() -> None:
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="nper", pv=-1000.0, fv=2000.0, pmt=0.0, rate=0.0)


def test_rate_zero_pv_and_pmt_raises() -> None:
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="rate", pv=0.0, fv=100.0, pmt=0.0, nper=10.0)


def test_rate_no_real_solution_raises() -> None:
    # pv and fv same sign with no payments -> no real positive growth factor.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="rate", pv=-1000.0, fv=-500.0, pmt=0.0, nper=10.0)


def test_loan_invalid_principal_raises() -> None:
    with pytest.raises(InvalidInput):
        loan_schedule(principal=0.0, annual_rate=0.05, term_months=12)


def test_loan_negative_rate_raises() -> None:
    with pytest.raises(InvalidInput):
        loan_schedule(principal=1000.0, annual_rate=-0.01, term_months=12)


def test_loan_schedule_summary_only_by_default() -> None:
    result = loan_schedule(principal=200000.0, annual_rate=0.06, term_months=360)
    assert result.monthly_payment == pytest.approx(1199.101, rel=1e-5)
    assert result.n_payments == 360
    assert result.total_interest == pytest.approx(result.total_paid - 200000.0, rel=1e-9)
    assert result.schedule == []  # full rows omitted by default


def test_loan_schedule_includes_rows_when_requested() -> None:
    result = loan_schedule(
        principal=200000.0, annual_rate=0.06, term_months=360, include_schedule=True
    )
    assert len(result.schedule) == 360
    assert result.schedule[-1].balance == pytest.approx(0.0, abs=1e-2)


def test_loan_negative_extra_payment_raises() -> None:
    with pytest.raises(InvalidInput):
        loan_schedule(principal=1000.0, annual_rate=0.05, term_months=12, extra_payment=-10.0)


def test_npv_basic() -> None:
    assert npv(0.10, [-1000.0, 500.0, 500.0, 500.0]).npv == pytest.approx(243.426, rel=1e-4)


def test_npv_zero_rate_is_sum() -> None:
    assert npv(0.0, [-1000.0, 600.0, 600.0]).npv == pytest.approx(200.0, rel=1e-9)


def test_npv_empty_raises() -> None:
    with pytest.raises(InvalidInput):
        npv(0.1, [])


def test_npv_invalid_rate_raises() -> None:
    with pytest.raises(InvalidInput):
        npv(-1.0, [-100.0, 110.0])


def test_irr_simple_exact() -> None:
    assert irr([-100.0, 110.0]).irr == pytest.approx(0.10, rel=1e-9)


def test_irr_roundtrips_through_npv() -> None:
    cashflows = [-1000.0, 500.0, 500.0, 500.0]
    r = irr(cashflows).irr
    assert npv(r, cashflows).npv == pytest.approx(0.0, abs=1e-6)


def test_irr_no_sign_change_raises() -> None:
    with pytest.raises(InvalidInput):
        irr([100.0, 200.0, 300.0])


def test_irr_single_cashflow_raises() -> None:
    with pytest.raises(InvalidInput):
        irr([-100.0])


def _cf(year: int, amount: float) -> DatedCashflow:
    return DatedCashflow(date=datetime.date(year, 1, 1), amount=amount)


def test_xnpv_zero_at_irr_rate() -> None:
    flows = [_cf(2021, -1000.0), _cf(2022, 1100.0)]  # 365-day span (non-leap)
    assert xnpv(0.10, flows).npv == pytest.approx(0.0, abs=1e-6)


def test_xirr_simple_annual() -> None:
    flows = [_cf(2021, -1000.0), _cf(2022, 1100.0)]
    assert xirr(flows).irr == pytest.approx(0.10, rel=1e-6)


def test_xirr_order_independent() -> None:
    flows = [_cf(2022, 1100.0), _cf(2021, -1000.0)]  # reversed input
    assert xirr(flows).irr == pytest.approx(0.10, rel=1e-6)


def test_xnpv_empty_raises() -> None:
    with pytest.raises(InvalidInput):
        xnpv(0.1, [])


def test_xnpv_invalid_rate_raises() -> None:
    with pytest.raises(InvalidInput):
        xnpv(-1.0, [_cf(2021, -100.0), _cf(2022, 110.0)])


def test_xirr_no_sign_change_raises() -> None:
    with pytest.raises(InvalidInput):
        xirr([_cf(2021, 100.0), _cf(2022, 200.0)])


def test_xirr_single_cashflow_raises() -> None:
    with pytest.raises(InvalidInput):
        xirr([_cf(2021, -100.0)])


def test_nominal_to_effective_monthly() -> None:
    r = convert_rate(0.12, periods_per_year=12, direction="nominal_to_effective")
    assert r.converted_rate == pytest.approx(0.12682503, rel=1e-7)


def test_nominal_to_effective_quarterly() -> None:
    r = convert_rate(0.12, periods_per_year=4, direction="nominal_to_effective")
    assert r.converted_rate == pytest.approx(0.12550881, rel=1e-7)


def test_effective_to_nominal_roundtrip() -> None:
    ear = convert_rate(0.12, periods_per_year=12, direction="nominal_to_effective").converted_rate
    back = convert_rate(ear, periods_per_year=12, direction="effective_to_nominal").converted_rate
    assert back == pytest.approx(0.12, rel=1e-9)


def test_convert_rate_invalid_periods_raises() -> None:
    with pytest.raises(InvalidInput):
        convert_rate(0.12, periods_per_year=0, direction="nominal_to_effective")


def test_convert_rate_invalid_nominal_raises() -> None:
    # 1 + nominal/m <= 0
    with pytest.raises(InvalidInput):
        convert_rate(-20.0, periods_per_year=12, direction="nominal_to_effective")


def test_convert_rate_invalid_effective_raises() -> None:
    # 1 + effective <= 0
    with pytest.raises(InvalidInput):
        convert_rate(-2.0, periods_per_year=12, direction="effective_to_nominal")


def test_fv_annuity_due_is_ordinary_times_one_plus_r() -> None:
    ordinary = time_value_of_money(
        solve_for="fv", pv=0.0, pmt=-100.0, rate=0.05, nper=10.0
    ).solved_value
    due = time_value_of_money(
        solve_for="fv", pv=0.0, pmt=-100.0, rate=0.05, nper=10.0, when="begin"
    ).solved_value
    assert due == pytest.approx(ordinary * 1.05, rel=1e-9)
    assert due == pytest.approx(1320.679, rel=1e-5)


def test_pmt_annuity_due_smaller_than_ordinary() -> None:
    # To hit the same FV, begin-of-period payments are smaller (they compound longer).
    ordinary = time_value_of_money(
        solve_for="pmt", pv=0.0, fv=10000.0, rate=0.05, nper=10.0
    ).solved_value
    due = time_value_of_money(
        solve_for="pmt", pv=0.0, fv=10000.0, rate=0.05, nper=10.0, when="begin"
    ).solved_value
    assert abs(due) == pytest.approx(abs(ordinary) / 1.05, rel=1e-9)


def test_nper_annuity_due_consistent() -> None:
    # Round-trip: nper that produces a known due-FV should recover ~10.
    fv = time_value_of_money(
        solve_for="fv", pv=0.0, pmt=-100.0, rate=0.05, nper=10.0, when="begin"
    ).solved_value
    n = time_value_of_money(
        solve_for="nper", pv=0.0, fv=fv, pmt=-100.0, rate=0.05, when="begin"
    ).solved_value
    assert n == pytest.approx(10.0, rel=1e-6)


def test_rate_annuity_due_consistent() -> None:
    fv = time_value_of_money(
        solve_for="fv", pv=0.0, pmt=-100.0, rate=0.05, nper=10.0, when="begin"
    ).solved_value
    r = time_value_of_money(
        solve_for="rate", pv=0.0, fv=fv, pmt=-100.0, nper=10.0, when="begin"
    ).solved_value
    assert r == pytest.approx(0.05, rel=1e-6)


def test_zero_rate_when_begin_equals_end() -> None:
    end = time_value_of_money(solve_for="fv", pv=0.0, pmt=-100.0, rate=0.0, nper=10.0).solved_value
    begin = time_value_of_money(
        solve_for="fv", pv=0.0, pmt=-100.0, rate=0.0, nper=10.0, when="begin"
    ).solved_value
    assert end == pytest.approx(begin, rel=1e-12)


def test_bond_price_at_par() -> None:
    r = bond_price(face=1000.0, coupon_rate=0.06, years_to_maturity=10.0, ytm=0.06, frequency=2)
    assert r.price == pytest.approx(1000.0, rel=1e-9)


def test_bond_price_discount() -> None:
    r = bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2)
    assert r.price == pytest.approx(925.61, rel=1e-4)


def test_zero_coupon_duration_equals_maturity() -> None:
    r = bond_price(face=1000.0, coupon_rate=0.0, years_to_maturity=5.0, ytm=0.04, frequency=1)
    assert r.price == pytest.approx(1000.0 / 1.04**5, rel=1e-9)
    assert r.macaulay_duration == pytest.approx(5.0, rel=1e-9)
    assert r.modified_duration == pytest.approx(5.0 / 1.04, rel=1e-9)
    assert r.convexity == pytest.approx(5.0 * 6.0 / 1.04**2, rel=1e-9)


def test_bond_current_yield() -> None:
    # current_yield = annual coupon / price = 50 / 925.61 = 0.054016 (price pinned
    # independently in test_bond_price_discount). Asserting against a literal, not 50/r.price
    # (which would equal the implementation by construction and could never fail).
    r = bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2)
    assert r.current_yield == pytest.approx(0.054016, rel=1e-4)


def test_bond_price_invalid_frequency_raises() -> None:
    with pytest.raises(InvalidInput):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=0)


def test_bond_price_invalid_face_raises() -> None:
    with pytest.raises(InvalidInput):
        bond_price(face=0.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2)


def test_bond_ytm_roundtrip() -> None:
    price = bond_price(
        face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2
    ).price
    r = bond_ytm(face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, price=price, frequency=2)
    assert r.yield_to_maturity == pytest.approx(0.06, rel=1e-6)


def test_bond_ytm_par_equals_coupon() -> None:
    r = bond_ytm(face=1000.0, coupon_rate=0.06, years_to_maturity=10.0, price=1000.0, frequency=2)
    assert r.yield_to_maturity == pytest.approx(0.06, rel=1e-6)


def test_bond_ytm_invalid_price_raises() -> None:
    with pytest.raises(InvalidInput):
        bond_ytm(face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, price=0.0, frequency=2)


def test_nominal_to_effective_continuous() -> None:
    r = convert_rate(
        0.12, periods_per_year=1, direction="nominal_to_effective", compounding="continuous"
    )
    assert r.converted_rate == pytest.approx(0.12749685, rel=1e-7)
    assert r.compounding == "continuous"


def test_effective_to_nominal_continuous() -> None:
    r = convert_rate(
        0.12, periods_per_year=1, direction="effective_to_nominal", compounding="continuous"
    )
    assert r.converted_rate == pytest.approx(0.11332869, rel=1e-7)


def test_continuous_roundtrip() -> None:
    ear = convert_rate(
        0.12, periods_per_year=1, direction="nominal_to_effective", compounding="continuous"
    ).converted_rate
    back = convert_rate(
        ear, periods_per_year=1, direction="effective_to_nominal", compounding="continuous"
    ).converted_rate
    assert back == pytest.approx(0.12, rel=1e-9)


def test_continuous_effective_invalid_raises() -> None:
    with pytest.raises(InvalidInput):
        convert_rate(
            -2.0, periods_per_year=1, direction="effective_to_nominal", compounding="continuous"
        )


def test_convert_rate_default_is_discrete() -> None:
    r = convert_rate(0.12, periods_per_year=12, direction="nominal_to_effective")
    assert r.converted_rate == pytest.approx(0.12682503, rel=1e-7)
    assert r.compounding == "discrete"


def test_bond_price_non_integer_periods_raises() -> None:
    # 2.5 years annual -> 2.5 periods, not a whole coupon count.
    with pytest.raises(InvalidInput):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=2.5, ytm=0.06, frequency=1)


def test_bond_price_fractional_periods_raises() -> None:
    with pytest.raises(InvalidInput):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=9.99, ytm=0.06, frequency=2)


def test_bond_price_half_year_semiannual_ok() -> None:
    # 2.5 years semiannual -> 5 whole periods, prices fine.
    r = bond_price(face=1000.0, coupon_rate=0.06, years_to_maturity=2.5, ytm=0.06, frequency=2)
    assert r.price == pytest.approx(1000.0, rel=1e-9)


def test_bond_ytm_non_integer_periods_raises() -> None:
    with pytest.raises(InvalidInput):
        bond_ytm(face=1000.0, coupon_rate=0.05, years_to_maturity=2.5, price=950.0, frequency=1)


def test_irr_multi_root_reports_both() -> None:
    # Non-conventional flow with two sign changes -> two IRRs (10% and 20%).
    result = irr([-100.0, 230.0, -132.0])
    assert result.is_unique is False
    assert len(result.all_irrs) == 2
    assert result.all_irrs[0] == pytest.approx(0.10, rel=1e-6)
    assert result.all_irrs[1] == pytest.approx(0.20, rel=1e-6)
    assert result.irr == pytest.approx(0.10, rel=1e-6)  # smallest non-negative tie-break


def test_irr_multi_root_each_zeros_npv() -> None:
    cashflows = [-100.0, 230.0, -132.0]
    for r in irr(cashflows).all_irrs:
        assert npv(r, cashflows).npv == pytest.approx(0.0, abs=1e-6)


def test_irr_unique_sets_flag() -> None:
    result = irr([-100.0, 110.0])
    assert result.is_unique is True
    assert result.all_irrs == pytest.approx([0.10])
    assert result.irr == pytest.approx(0.10, rel=1e-9)


def test_irr_conventional_multi_period_unique() -> None:
    result = irr([-1000.0, 500.0, 500.0, 500.0])
    assert result.is_unique is True
    assert result.irr == pytest.approx(0.23375, rel=1e-4)


def test_xirr_unique_sets_flag() -> None:
    flows = [_cf(2021, -1000.0), _cf(2022, 1100.0)]
    result = xirr(flows)
    assert result.is_unique is True
    assert result.irr == pytest.approx(0.10, rel=1e-6)


def test_mirr_known_value() -> None:
    result = mirr([-1000.0, 500.0, 400.0, 300.0, 100.0], finance_rate=0.10, reinvest_rate=0.12)
    assert result.mirr == pytest.approx(0.13168560, rel=1e-6)
    assert result.finance_rate == 0.10
    assert result.reinvest_rate == 0.12


def test_mirr_single_value_for_multi_irr_flow() -> None:
    # The flow that has two IRRs (10%, 20%) yields one deterministic MIRR.
    result = mirr([-100.0, 230.0, -132.0], finance_rate=0.10, reinvest_rate=0.10)
    # fv_pos = 230*1.1; pv_neg = -100 - 132/1.1^2; n = 2.
    fv_pos = 230.0 * 1.10
    pv_neg = 100.0 + 132.0 / 1.10**2
    expected = (fv_pos / pv_neg) ** 0.5 - 1.0
    assert result.mirr == pytest.approx(expected, rel=1e-9)


def test_mirr_no_positive_raises() -> None:
    with pytest.raises(InvalidInput):
        mirr([-100.0, -50.0], finance_rate=0.1, reinvest_rate=0.1)


def test_mirr_no_negative_raises() -> None:
    with pytest.raises(InvalidInput):
        mirr([100.0, 50.0], finance_rate=0.1, reinvest_rate=0.1)


def test_mirr_invalid_rate_raises() -> None:
    with pytest.raises(InvalidInput):
        mirr([-100.0, 200.0], finance_rate=-1.0, reinvest_rate=0.1)


# --- guards and edge cases ---


def test_rate_solve_large_nper_no_overflow() -> None:
    # The rate search must not overflow on (1 + r)**360. 200k @ 1199.101/mo -> 0.5%/mo.
    result = time_value_of_money(solve_for="rate", pv=200000.0, fv=0.0, pmt=-1199.101, nper=360.0)
    assert result.solved_value == pytest.approx(0.005, rel=1e-3)


def test_rate_no_root_small_nper_raises() -> None:
    # All-outflow, fv=0: f(r) never crosses zero; expansion exhausts -> InvalidInput.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="rate", pv=-100.0, fv=0.0, pmt=-100.0, nper=2.0)


def test_rate_no_root_overflow_in_expansion_raises() -> None:
    # No root, large nper: expansion overflows mid-loop -> treated as no bracket.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="rate", pv=-100.0, fv=0.0, pmt=-100.0, nper=200.0)


def test_rate_initial_overflow_raises() -> None:
    # nper so large that f(high) overflows on the first evaluation -> InvalidInput, not raw.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="rate", pv=-100.0, fv=0.0, pmt=-100.0, nper=5000.0)


def test_nper_pmt_zero_bad_signs_raises() -> None:
    # pmt=0, pv and fv same sign -> ratio <= 0 -> no real solution.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="nper", pv=-1000.0, fv=-2000.0, pmt=0.0, rate=0.05)


def test_nper_general_no_solution_raises() -> None:
    # pmt!=0 with inputs making (k-fv)/(k+pv) <= 0 -> no real solution.
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="nper", pv=3000.0, fv=1000.0, pmt=-100.0, rate=0.05)


def test_irr_within_extended_range() -> None:
    # IRR = 999/period (99,900%) lies in the log-spaced tail of the search grid.
    assert irr([-1.0, 1000.0]).irr == pytest.approx(999.0, rel=1e-9)


def test_irr_no_root_in_range_raises() -> None:
    # IRR = 99,999/period (~1e7%) is past the 1,000,000% upper bound -> sign change but
    # no root in the searched range.
    with pytest.raises(InvalidInput):
        irr([-1.0, 100000.0])


def test_xirr_no_root_in_range_raises() -> None:
    with pytest.raises(InvalidInput):
        xirr([_cf(2021, -1.0), _cf(2022, 100000.0)])


def test_mirr_too_few_cashflows_raises() -> None:
    with pytest.raises(InvalidInput):
        mirr([-100.0], finance_rate=0.1, reinvest_rate=0.1)


def test_mirr_ignores_zero_cashflow() -> None:
    # A 0.0 flow is neither financed nor reinvested. n=3; fv_pos = 500*1.12 + 700 = 1260,
    # pv_neg = 1000 -> MIRR = (1260/1000)**(1/3) - 1 = 0.080083 (computed by hand, so a
    # mishandled zero flow that still happened to be positive would be caught).
    result = mirr([-1000.0, 0.0, 500.0, 700.0], finance_rate=0.10, reinvest_rate=0.12)
    assert result.mirr == pytest.approx(0.080083, rel=1e-4)


def test_bond_price_nonpositive_maturity_raises() -> None:
    with pytest.raises(InvalidInput):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=0.0, ytm=0.06, frequency=2)


def test_bond_price_zero_periods_raises() -> None:
    # Tiny maturity rounds to 0 whole periods (within tolerance) -> n < 1.
    with pytest.raises(InvalidInput):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=1e-10, ytm=0.06, frequency=1)


def test_bisect_bracket_returns_after_iteration_cap() -> None:
    # An enormous bracket cannot converge within 500 halvings -> hits the fallback return.
    result = _bisect_bracket(lambda x: x, -1e200, 1e200)
    assert math.isfinite(result)


def test_find_all_roots_captures_exact_grid_zero() -> None:
    # A root that lands exactly on a grid abscissa is recorded as an exact zero.
    low, high, grid_points = -0.999999, 10.0, 1100
    step = (high - low) / (grid_points - 1)
    gp = low + 50 * step
    roots = _find_all_roots(lambda x: x - gp, low=low, high=high, grid_points=grid_points)
    assert any(abs(r - gp) < 1e-9 for r in roots)


def test_find_all_roots_dedups_close_roots() -> None:
    # With a wide dedup tolerance, two distinct roots collapse to one.
    roots = _find_all_roots(lambda x: (x - 0.1) * (x - 0.2), dedup_tol=1.0)
    assert len(roots) == 1


def test_npv_long_series_extreme_rate_yields_signed_infinity() -> None:
    # Near rate == -1 the discount factor (1+rate)**period underflows to 0.0 for long
    # series; npv must degrade to a signed infinity, not raise ZeroDivisionError. The
    # largest-exponent non-zero flow (+800 at period 360) dictates the sign -> +inf.
    cashflows = [-100000.0] + [800.0] * 360
    assert npv(-0.999999, cashflows).npv == math.inf


def test_npv_extreme_rate_ignores_zero_cashflows_in_underflow_region() -> None:
    # Zero cashflows in the underflow region (factor == 0.0, cash == 0.0) are skipped;
    # the largest-exponent non-zero flow dictates the signed infinity.
    result = npv(-0.999999, [-100.0] + [0.0] * 55 + [200.0])
    assert result.npv == math.inf


def test_irr_long_monthly_series_matches_annuity_solution() -> None:
    # The root finder evaluates npv near rate == -1, where (1 + r)**360 underflows. Expected
    # value from the annuity equation 100000 = 800 * (1 - (1+r)^-360)/r  ->  r = 0.00744641.
    cashflows = [-100000.0] + [800.0] * 360
    result = irr(cashflows)
    assert result.irr == pytest.approx(0.00744641, rel=1e-5)
    assert result.is_unique  # one sign change -> a single IRR


def test_xirr_long_calendar_span_matches_closed_form() -> None:
    # A >51-year span pushes (1+rate)**years to underflow at the bracket low end. Two flows
    # have a closed-form XIRR: 5000/1000 = (1+r)^years, years = 23376/365 (Actual/365) ->
    # r = 5**(365/23376) - 1 = 0.02544868.
    flows = [_cf(1960, -1000.0), _cf(2024, 5000.0)]
    assert xirr(flows).irr == pytest.approx(0.02544868, rel=1e-5)


def test_rate_zero_nper_raises() -> None:
    with pytest.raises(InvalidInput):
        time_value_of_money(solve_for="rate", pv=-100.0, fv=200.0, pmt=0.0, nper=0.0)


def test_pmt_zero_nper_raises() -> None:
    # nper == 0 divides by zero in both the r == 0 and the annuity branch.
    with pytest.raises(InvalidInput, match="nper"):
        time_value_of_money(solve_for="pmt", pv=1000.0, fv=0.0, rate=0.05, nper=0.0)


def test_pmt_zero_nper_zero_rate_raises() -> None:
    with pytest.raises(InvalidInput, match="nper"):
        time_value_of_money(solve_for="pmt", pv=1000.0, fv=0.0, rate=0.0, nper=0.0)


def test_pv_rate_at_minus_one_raises() -> None:
    # rate == -1 makes the growth factor 0, which _pv divides by.
    with pytest.raises(InvalidInput, match="rate"):
        time_value_of_money(solve_for="pv", fv=100.0, pmt=0.0, rate=-1.0, nper=5.0)


def test_fv_rate_below_minus_one_raises() -> None:
    # A negative base with fractional nper is a complex number, which the result
    # model cannot hold; reject the rate instead of leaking a ValidationError.
    with pytest.raises(InvalidInput, match="rate"):
        time_value_of_money(solve_for="fv", pv=-1000.0, pmt=0.0, rate=-1.5, nper=2.5)


def test_pmt_rate_below_minus_one_raises() -> None:
    with pytest.raises(InvalidInput, match="rate"):
        time_value_of_money(solve_for="pmt", pv=1000.0, fv=0.0, rate=-1.5, nper=10.0)


def test_nper_rate_at_minus_one_raises() -> None:
    # log(1 + rate) is log(0) -> math domain error without the guard.
    with pytest.raises(InvalidInput, match="rate"):
        time_value_of_money(solve_for="nper", pv=-1000.0, fv=2000.0, pmt=0.0, rate=-1.0)


def test_nper_zero_pv_and_zero_pmt_raises() -> None:
    # With no payments the equation reduces to (1+r)^n = -fv/pv; pv == 0 divides by zero.
    with pytest.raises(InvalidInput, match="pv"):
        time_value_of_money(solve_for="nper", pv=0.0, fv=100.0, pmt=0.0, rate=0.05)


def test_pv_growth_factor_underflow_raises() -> None:
    # (1 + -0.9999)**1e5 underflows to exactly 0.0; _pv divides by it.
    with pytest.raises(InvalidInput, match="underflow"):
        time_value_of_money(solve_for="pv", fv=100.0, pmt=0.0, rate=-0.9999, nper=1e5)


def test_loan_final_payment_clears_balance_large_principal() -> None:
    # Float drift accumulated over 360 periods left principal unpaid at the end of the
    # term, so the last schedule row reported a balance still outstanding (0.14 at this
    # size). The final payment must absorb the residual and end at a zero balance. The
    # principal has to be this large for the drift to survive rounding to cents.
    result = loan_schedule(principal=1e13, annual_rate=0.07, term_months=360, include_schedule=True)
    assert result.n_payments == 360
    assert result.schedule[-1].balance == 0.0


def test_loan_final_payment_clears_balance_zero_rate() -> None:
    # At 0% the payment is principal/term exactly and the drift is far larger: the loan
    # ended 6.88 short of paying off, so total_paid understated the principal.
    principal = 1e15
    result = loan_schedule(principal=principal, annual_rate=0.0, term_months=360)
    assert result.n_payments == 360
    assert result.total_interest == pytest.approx(0.0, abs=1e-6)
    assert result.total_paid == pytest.approx(principal, abs=0.01)


def test_loan_rate_too_large_raises() -> None:
    # (1 + 1e4/12)**360 overflows; that must surface as InvalidInput, not OverflowError.
    with pytest.raises(InvalidInput, match="annual_rate"):
        loan_schedule(principal=1000.0, annual_rate=1e4, term_months=360)


def test_loan_negligible_rate_behaves_as_zero_rate() -> None:
    # (1 + 1e-18/12)**12 == 1.0 in floating point, so the annuity formula would divide
    # by growth - 1 == 0. Fall back to the straight-line payment.
    result = loan_schedule(principal=1200.0, annual_rate=1e-18, term_months=12)
    assert result.monthly_payment == pytest.approx(100.0, rel=1e-9)
    assert result.n_payments == 12
    assert result.total_interest == pytest.approx(0.0, abs=1e-6)


def test_convert_rate_continuous_overflow_raises() -> None:
    # exp(1000) overflows; the tool must see InvalidInput, not OverflowError.
    with pytest.raises(InvalidInput, match="too large"):
        convert_rate(1000.0, 1, "nominal_to_effective", "continuous")


def test_convert_rate_discrete_overflow_raises() -> None:
    # (1 + 1e300/12)**12 overflows too, so the discrete path needs the same guard.
    with pytest.raises(InvalidInput, match="too large"):
        convert_rate(1e300, 12, "nominal_to_effective")


def test_bond_price_accepts_yield_below_minus_one_when_base_positive() -> None:
    # frequency=2 makes the periodic yield -0.75, so the discount base is 0.25 > 0:
    # price = 25/0.25 + 1025/0.25**2 = 100 + 16400 = 16500.
    result = bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=1.0, ytm=-1.5, frequency=2)
    assert result.price == pytest.approx(16500.0, rel=1e-12)


def test_bond_price_yield_at_negative_frequency_raises() -> None:
    # ytm == -frequency makes 1 + ytm/frequency exactly 0 -> still invalid.
    with pytest.raises(InvalidInput, match="ytm"):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=1.0, ytm=-2.0, frequency=2)


def test_bond_price_annual_frequency_still_rejects_minus_one() -> None:
    # With frequency=1 the old and new constraints coincide.
    with pytest.raises(InvalidInput, match="ytm"):
        bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=1.0, ytm=-1.0, frequency=1)


def test_irr_tangent_root_local_maximum() -> None:
    # npv(r) = -(1 - 1/(1+r))**2 touches zero at r = 0 from below: a double root with
    # no sign change, so the grid scan alone reports "no IRR".
    result = irr([-1.0, 2.0, -1.0])
    assert result.irr == pytest.approx(0.0, abs=1e-6)
    assert result.is_unique is True
    assert len(result.all_irrs) == 1


def test_irr_tangent_root_local_minimum() -> None:
    # The same flow negated: npv touches zero at r = 0 from above.
    result = irr([1.0, -2.0, 1.0])
    assert result.irr == pytest.approx(0.0, abs=1e-6)
    assert result.is_unique is True


def test_irr_tangent_root_away_from_zero() -> None:
    # A double root at 15%: (1 - 1.15/(1+r))**2 scaled.
    result = irr([-1.0, 2.3, -1.3225])
    assert result.irr == pytest.approx(0.15, rel=1e-6)
    assert result.is_unique is True


def test_irr_tangent_root_scales_with_cashflow_magnitude() -> None:
    # The tangent tolerance is relative to the local |f|, so a 1e9-sized flow with the
    # same shape must also be recognised.
    result = irr([-1e9, 2e9, -1e9])
    assert result.irr == pytest.approx(0.0, abs=1e-6)


def test_irr_above_one_thousand_percent() -> None:
    # 1 -> 12 in one period is an IRR of 1100%, past the old uniform grid's upper bound.
    result = irr([-1.0, 12.0])
    assert result.irr == pytest.approx(11.0, rel=1e-9)
    assert result.is_unique is True


def test_irr_two_roots_closer_than_a_grid_step() -> None:
    # Roots at 10% and 10.01% are ~1e-4 apart, far inside the ~1e-2 uniform grid step.
    # Built as (x - 1/1.10)(x - 1/1.1001) in x = 1/(1+r).
    cashflows = [1.0 / (1.10 * 1.1001), -(1.0 / 1.10 + 1.0 / 1.1001), 1.0]
    result = irr(cashflows)
    assert len(result.all_irrs) == 2
    assert result.all_irrs[0] == pytest.approx(0.10, rel=1e-6)
    assert result.all_irrs[1] == pytest.approx(0.1001, rel=1e-6)
    assert result.is_unique is False
    for root in result.all_irrs:
        assert npv(root, cashflows).npv == pytest.approx(0.0, abs=1e-9)


def test_irr_near_tangent_but_not_a_root_is_not_reported() -> None:
    # min|npv| is 1e-9 with cashflows of order 1 — well above the floating-point
    # cancellation floor, so this is genuinely rootless and must stay rootless.
    with pytest.raises(InvalidInput):
        irr([0.01 + 1e-9, -0.2, 1.0])


def test_find_all_roots_without_log_tail() -> None:
    # log_high == high disables the geometric tail, so a root past `high` is missed.
    roots = _find_all_roots(lambda x: x - 11.0, log_high=10.0)
    assert roots == []


def test_find_all_roots_skips_flat_sampled_regions() -> None:
    # A step function has zero first differences almost everywhere; the turning-point
    # scan must skip those instead of minimizing over a flat bracket.
    assert _find_all_roots(lambda x: 1.0 if x < 5.0 else 2.0) == []


def test_minimize_finds_interior_minimum() -> None:
    assert _minimize(lambda x: (x - 0.25) ** 2, -1.0, 1.0) == pytest.approx(0.25, abs=1e-9)
