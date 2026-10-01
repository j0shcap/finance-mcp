import datetime
import math
from collections.abc import Callable
from typing import Any

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
from finance_mcp.data.models import Compounding, DatedCashflow, RateDirection


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # Solving for fv requires pv, pmt, rate and nper; rate is omitted.
        pytest.param(
            {"solve_for": "fv", "pv": -1000.0, "pmt": 0.0, "nper": 10.0}, "rate", id="missing"
        ),
        pytest.param(
            {"solve_for": "nper", "pv": -1000.0, "fv": 2000.0, "pmt": 0.0, "rate": 0.0},
            "rate and pmt are zero",
            id="nper-zero-rate-and-pmt",
        ),
        # With no payments the equation reduces to (1+r)^n = -fv/pv; pv == 0 divides by zero.
        pytest.param(
            {"solve_for": "nper", "pv": 0.0, "fv": 100.0, "pmt": 0.0, "rate": 0.05},
            "pv",
            id="nper-zero-pv-and-pmt",
        ),
        # pmt=0 with pv and fv the same sign -> ratio <= 0 -> no real solution.
        pytest.param(
            {"solve_for": "nper", "pv": -1000.0, "fv": -2000.0, "pmt": 0.0, "rate": 0.05},
            "pv/fv signs",
            id="nper-pmt-zero-bad-signs",
        ),
        # pmt != 0 with inputs making (k-fv)/(k+pv) <= 0 -> no real solution.
        pytest.param(
            {"solve_for": "nper", "pv": 3000.0, "fv": 1000.0, "pmt": -100.0, "rate": 0.05},
            "No real solution",
            id="nper-general-no-solution",
        ),
        # log(1 + rate) is log(0) -> math domain error without the guard.
        pytest.param(
            {"solve_for": "nper", "pv": -1000.0, "fv": 2000.0, "pmt": 0.0, "rate": -1.0},
            "rate",
            id="nper-rate-at-minus-one",
        ),
        pytest.param(
            {"solve_for": "rate", "pv": 0.0, "fv": 100.0, "pmt": 0.0, "nper": 10.0},
            "pv and pmt are both zero",
            id="rate-zero-pv-and-pmt",
        ),
        # pv and fv the same sign with no payments -> no real positive growth factor.
        pytest.param(
            {"solve_for": "rate", "pv": -1000.0, "fv": -500.0, "pmt": 0.0, "nper": 10.0},
            "pv/fv signs",
            id="rate-no-real-solution",
        ),
        pytest.param(
            {"solve_for": "rate", "pv": -100.0, "fv": 200.0, "pmt": 0.0, "nper": 0.0},
            "nper",
            id="rate-zero-nper",
        ),
        # All outflows with fv=0: f(r) never crosses zero, so the bracket expansion runs out.
        pytest.param(
            {"solve_for": "rate", "pv": -100.0, "fv": 0.0, "pmt": -100.0, "nper": 2.0},
            "bracket",
            id="rate-no-root",
        ),
        # No root and a large nper: the expansion overflows mid-loop -> treated as no bracket.
        pytest.param(
            {"solve_for": "rate", "pv": -100.0, "fv": 0.0, "pmt": -100.0, "nper": 200.0},
            "bracket",
            id="rate-no-root-overflow-in-expansion",
        ),
        # nper so large that f(high) overflows on the first evaluation.
        pytest.param(
            {"solve_for": "rate", "pv": -100.0, "fv": 0.0, "pmt": -100.0, "nper": 5000.0},
            "bracket",
            id="rate-initial-overflow",
        ),
        # nper == 0 divides by zero in both the r == 0 and the annuity branch.
        pytest.param(
            {"solve_for": "pmt", "pv": 1000.0, "fv": 0.0, "rate": 0.05, "nper": 0.0},
            "nper",
            id="pmt-zero-nper",
        ),
        pytest.param(
            {"solve_for": "pmt", "pv": 1000.0, "fv": 0.0, "rate": 0.0, "nper": 0.0},
            "nper",
            id="pmt-zero-nper-zero-rate",
        ),
        # A negative base with fractional nper is a complex number, which the result model
        # cannot hold; the rate is rejected instead of leaking a ValidationError.
        pytest.param(
            {"solve_for": "pmt", "pv": 1000.0, "fv": 0.0, "rate": -1.5, "nper": 10.0},
            "rate",
            id="pmt-rate-below-minus-one",
        ),
        pytest.param(
            {"solve_for": "fv", "pv": -1000.0, "pmt": 0.0, "rate": -1.5, "nper": 2.5},
            "rate",
            id="fv-rate-below-minus-one",
        ),
        # rate == -1 makes the growth factor 0, which _pv divides by.
        pytest.param(
            {"solve_for": "pv", "fv": 100.0, "pmt": 0.0, "rate": -1.0, "nper": 5.0},
            "rate",
            id="pv-rate-at-minus-one",
        ),
        # (1 + -0.9999)**1e5 underflows to exactly 0.0, which _pv divides by.
        pytest.param(
            {"solve_for": "pv", "fv": 100.0, "pmt": 0.0, "rate": -0.9999, "nper": 1e5},
            "underflow",
            id="pv-growth-factor-underflow",
        ),
    ],
)
def test_tvm_rejects_unsolvable_inputs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(InvalidInput, match=match):
        time_value_of_money(**kwargs)


def test_omitted_fv_defaults_to_zero() -> None:
    # Excel's PMT/PV/NPER/RATE default fv to 0, so a plain loan payment needs no fv.
    result = time_value_of_money(solve_for="pmt", pv=400000.0, rate=0.005, nper=360.0)
    assert result.solved_value == pytest.approx(-2398.2021006, rel=1e-9)
    assert result.fv == 0.0


def test_failure_with_defaulted_fv_says_fv_was_omitted() -> None:
    # A CAGR solve that forgot fv: the defaulted 0 is the likely mistake, so name it.
    with pytest.raises(InvalidInput, match="fv was omitted and defaulted to 0"):
        time_value_of_money(solve_for="rate", pv=-1000.0, nper=10.0)


def test_failure_with_explicit_fv_does_not_mention_the_default() -> None:
    with pytest.raises(InvalidInput) as excinfo:
        time_value_of_money(solve_for="rate", pv=-1000.0, fv=0.0, nper=10.0)
    assert "omitted" not in str(excinfo.value)


def test_result_echoes_all_fields() -> None:
    result = time_value_of_money(solve_for="fv", pv=-1000.0, pmt=0.0, rate=0.05, nper=10.0)
    assert result.rate == 0.05
    assert result.nper == 10.0
    assert math.isclose(result.fv, result.solved_value)


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


def test_loan_extra_payment_reports_what_it_saves() -> None:
    # 400k at 6.5% over 30 years: 360 payments and 510,177.95 of interest without the
    # extra 500/month, 233 payments and 304,620.81 with it.
    faster = loan_schedule(
        principal=400000.0, annual_rate=0.065, term_months=360, extra_payment=500.0
    )
    assert faster.n_payments == 233
    assert faster.payments_saved == 127
    assert faster.interest_saved == 205557.14


def test_loan_savings_are_measured_against_the_same_loan_without_the_extra() -> None:
    base = loan_schedule(principal=250000.0, annual_rate=0.0725, term_months=300)
    faster = loan_schedule(
        principal=250000.0, annual_rate=0.0725, term_months=300, extra_payment=137.5
    )
    assert faster.payments_saved == base.n_payments - faster.n_payments
    # Saved interest is rounded once from the unrounded totals, so it can differ by a cent
    # from subtracting the two rounded totals.
    assert faster.interest_saved == pytest.approx(
        base.total_interest - faster.total_interest, abs=0.01
    )


def test_loan_without_extra_payment_saves_nothing() -> None:
    result = loan_schedule(principal=200000.0, annual_rate=0.06, term_months=360)
    assert result.payments_saved == 0
    assert result.interest_saved == 0.0


def test_loan_savings_at_zero_rate_are_payments_only() -> None:
    result = loan_schedule(principal=1200.0, annual_rate=0.0, term_months=12, extra_payment=100.0)
    assert result.n_payments == 6
    assert result.payments_saved == 6
    assert result.interest_saved == 0.0


def test_loan_extra_payment_clearing_the_loan_at_once() -> None:
    # One payment retires the loan: the saving is every later payment and all the interest
    # after the first month.
    base = loan_schedule(principal=10000.0, annual_rate=0.12, term_months=24)
    result = loan_schedule(
        principal=10000.0, annual_rate=0.12, term_months=24, extra_payment=20000.0
    )
    assert result.n_payments == 1
    assert result.payments_saved == 23
    assert result.total_interest == 100.0
    assert result.interest_saved == pytest.approx(base.total_interest - 100.0, abs=0.01)


def test_loan_savings_do_not_depend_on_returning_the_rows() -> None:
    summary = loan_schedule(300000.0, 0.055, 360, extra_payment=250.0)
    with_rows = loan_schedule(300000.0, 0.055, 360, extra_payment=250.0, include_schedule=True)
    assert (summary.interest_saved, summary.payments_saved) == (
        with_rows.interest_saved,
        with_rows.payments_saved,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"principal": 0.0}, "principal"),
        ({"term_months": 0}, "term_months"),
        ({"annual_rate": -0.01}, "annual_rate"),
        ({"extra_payment": -10.0}, "extra_payment"),
        # (1 + 1e4/12)**360 overflows; that must surface as InvalidInput, not OverflowError.
        ({"annual_rate": 1e4, "term_months": 360}, "annual_rate"),
    ],
)
def test_loan_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    loan = {"principal": 1000.0, "annual_rate": 0.05, "term_months": 12, **kwargs}
    with pytest.raises(InvalidInput, match=match):
        loan_schedule(**loan)


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


def test_loan_schedule_summary_only_by_default() -> None:
    # 200k at 6% annual, 360 months -> payment ~ 1199.10.
    result = loan_schedule(principal=200000.0, annual_rate=0.06, term_months=360)
    assert result.monthly_payment == pytest.approx(1199.101, rel=1e-5)
    assert result.n_payments == 360
    assert result.total_interest == pytest.approx(result.total_paid - 200000.0, rel=1e-9)
    assert result.schedule == []


def test_loan_schedule_includes_rows_when_requested() -> None:
    result = loan_schedule(
        principal=200000.0, annual_rate=0.06, term_months=360, include_schedule=True
    )
    assert len(result.schedule) == 360
    assert result.schedule[-1].balance == pytest.approx(0.0, abs=1e-2)


def test_npv_basic() -> None:
    assert npv(0.10, [-1000.0, 500.0, 500.0, 500.0]).npv == pytest.approx(243.426, rel=1e-4)


def test_npv_zero_rate_is_sum() -> None:
    assert npv(0.0, [-1000.0, 600.0, 600.0]).npv == pytest.approx(200.0, rel=1e-9)


def _cf(year: int, amount: float) -> DatedCashflow:
    return DatedCashflow(date=datetime.date(year, 1, 1), amount=amount)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(lambda: npv(0.1, []), "empty", id="npv-empty"),
        pytest.param(lambda: npv(-1.0, [-100.0, 110.0]), "rate", id="npv-rate"),
        pytest.param(lambda: irr([-100.0]), "two cashflows", id="irr-single"),
        pytest.param(lambda: irr([100.0, 200.0, 300.0]), "sign change", id="irr-no-sign-change"),
        # IRR = 99,999/period (~1e7%) is past the 1,000,000% upper bound: a sign change but
        # no root in the searched range.
        pytest.param(lambda: irr([-1.0, 100000.0]), "No internal rate", id="irr-out-of-range"),
        pytest.param(lambda: xnpv(0.1, []), "empty", id="xnpv-empty"),
        pytest.param(
            lambda: xnpv(-1.0, [_cf(2021, -100.0), _cf(2022, 110.0)]), "rate", id="xnpv-rate"
        ),
        pytest.param(lambda: xirr([_cf(2021, -100.0)]), "two cashflows", id="xirr-single"),
        pytest.param(
            lambda: xirr([_cf(2021, 100.0), _cf(2022, 200.0)]),
            "sign change",
            id="xirr-no-sign-change",
        ),
        pytest.param(
            lambda: xirr([_cf(2021, -1.0), _cf(2022, 100000.0)]),
            "No internal rate",
            id="xirr-out-of-range",
        ),
        pytest.param(
            lambda: mirr([-100.0], finance_rate=0.1, reinvest_rate=0.1),
            "two cashflows",
            id="mirr-single",
        ),
        pytest.param(
            lambda: mirr([-100.0, -50.0], finance_rate=0.1, reinvest_rate=0.1),
            "one negative and one positive",
            id="mirr-no-positive",
        ),
        pytest.param(
            lambda: mirr([100.0, 50.0], finance_rate=0.1, reinvest_rate=0.1),
            "one negative and one positive",
            id="mirr-no-negative",
        ),
        pytest.param(
            lambda: mirr([-100.0, 200.0], finance_rate=-1.0, reinvest_rate=0.1),
            "greater than -1",
            id="mirr-rate",
        ),
    ],
)
def test_cashflow_calculators_reject_invalid_inputs(call: Callable[[], object], match: str) -> None:
    with pytest.raises(InvalidInput, match=match):
        call()


def test_irr_roundtrips_through_npv() -> None:
    cashflows = [-1000.0, 500.0, 500.0, 500.0]
    r = irr(cashflows).irr
    assert npv(r, cashflows).npv == pytest.approx(0.0, abs=1e-6)


def test_xnpv_zero_at_irr_rate() -> None:
    flows = [_cf(2021, -1000.0), _cf(2022, 1100.0)]  # 365-day span (non-leap)
    assert xnpv(0.10, flows).npv == pytest.approx(0.0, abs=1e-6)


def test_xirr_order_independent() -> None:
    flows = [_cf(2022, 1100.0), _cf(2021, -1000.0)]  # reversed input
    assert xirr(flows).irr == pytest.approx(0.10, rel=1e-6)


@pytest.mark.parametrize(
    ("rate", "periods_per_year", "direction", "compounding", "expected"),
    [
        (0.12, 12, "nominal_to_effective", "discrete", 0.12682503),
        (0.12, 4, "nominal_to_effective", "discrete", 0.12550881),
        (0.12, 1, "nominal_to_effective", "continuous", 0.12749685),
        (0.12, 1, "effective_to_nominal", "continuous", 0.11332869),
    ],
)
def test_convert_rate_known_values(
    rate: float,
    periods_per_year: int,
    direction: RateDirection,
    compounding: Compounding,
    expected: float,
) -> None:
    r = convert_rate(rate, periods_per_year, direction, compounding)
    assert r.converted_rate == pytest.approx(expected, rel=1e-7)
    assert r.compounding == compounding


@pytest.mark.parametrize(("periods_per_year", "compounding"), [(12, "discrete"), (1, "continuous")])
def test_convert_rate_round_trips(periods_per_year: int, compounding: Compounding) -> None:
    ear = convert_rate(0.12, periods_per_year, "nominal_to_effective", compounding)
    back = convert_rate(ear.converted_rate, periods_per_year, "effective_to_nominal", compounding)
    assert back.converted_rate == pytest.approx(0.12, rel=1e-9)


def test_convert_rate_default_is_discrete() -> None:
    r = convert_rate(0.12, periods_per_year=12, direction="nominal_to_effective")
    assert r.compounding == "discrete"


@pytest.mark.parametrize(
    ("rate", "periods_per_year", "direction", "compounding", "match"),
    [
        (0.12, 0, "nominal_to_effective", "discrete", "periods_per_year"),
        # 1 + nominal/m <= 0
        (-20.0, 12, "nominal_to_effective", "discrete", "nominal rate"),
        # 1 + effective <= 0
        (-2.0, 12, "effective_to_nominal", "discrete", "Effective rate"),
        (-2.0, 1, "effective_to_nominal", "continuous", "Effective rate"),
        # exp(1000) and (1 + 1e300/12)**12 overflow; the tool must see InvalidInput.
        (1000.0, 1, "nominal_to_effective", "continuous", "too large"),
        (1e300, 12, "nominal_to_effective", "discrete", "too large"),
    ],
)
def test_convert_rate_rejects_invalid_inputs(
    rate: float,
    periods_per_year: int,
    direction: RateDirection,
    compounding: Compounding,
    match: str,
) -> None:
    with pytest.raises(InvalidInput, match=match):
        convert_rate(rate, periods_per_year, direction, compounding)


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frequency": 0}, "frequency"),
        ({"face": 0.0}, "face"),
        ({"years_to_maturity": 0.0}, "years_to_maturity must be positive"),
        # 2.5 years annual -> 2.5 periods, not a whole coupon count.
        ({"years_to_maturity": 2.5, "frequency": 1}, "whole number"),
        ({"years_to_maturity": 9.99}, "whole number"),
        # A tiny maturity rounds to 0 whole periods (within tolerance) -> n < 1.
        ({"years_to_maturity": 1e-10, "frequency": 1}, "at least one period"),
        # ytm == -frequency makes 1 + ytm/frequency exactly 0.
        ({"years_to_maturity": 1.0, "ytm": -2.0}, "ytm"),
        ({"years_to_maturity": 1.0, "ytm": -1.0, "frequency": 1}, "ytm"),
    ],
)
def test_bond_price_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    bond = {
        "face": 1000.0,
        "coupon_rate": 0.05,
        "years_to_maturity": 10.0,
        "ytm": 0.06,
        "frequency": 2,
        **kwargs,
    }
    with pytest.raises(InvalidInput, match=match):
        bond_price(**bond)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"price": 0.0}, "price must be positive"),
        ({"years_to_maturity": 2.5, "frequency": 1}, "whole number"),
    ],
)
def test_bond_ytm_rejects_invalid_inputs(kwargs: dict[str, Any], match: str) -> None:
    bond = {
        "face": 1000.0,
        "coupon_rate": 0.05,
        "years_to_maturity": 10.0,
        "price": 950.0,
        "frequency": 2,
        **kwargs,
    }
    with pytest.raises(InvalidInput, match=match):
        bond_ytm(**bond)


def test_bond_ytm_roundtrip() -> None:
    price = bond_price(
        face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, ytm=0.06, frequency=2
    ).price
    r = bond_ytm(face=1000.0, coupon_rate=0.05, years_to_maturity=10.0, price=price, frequency=2)
    assert r.yield_to_maturity == pytest.approx(0.06, rel=1e-6)


def test_bond_ytm_par_equals_coupon() -> None:
    r = bond_ytm(face=1000.0, coupon_rate=0.06, years_to_maturity=10.0, price=1000.0, frequency=2)
    assert r.yield_to_maturity == pytest.approx(0.06, rel=1e-6)


def test_bond_price_half_year_semiannual_ok() -> None:
    # 2.5 years semiannual -> 5 whole periods, prices fine.
    r = bond_price(face=1000.0, coupon_rate=0.06, years_to_maturity=2.5, ytm=0.06, frequency=2)
    assert r.price == pytest.approx(1000.0, rel=1e-9)


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


# --- guards and edge cases ---


def test_rate_solve_large_nper_no_overflow() -> None:
    # The rate search must not overflow on (1 + r)**360. 200k @ 1199.101/mo -> 0.5%/mo.
    result = time_value_of_money(solve_for="rate", pv=200000.0, fv=0.0, pmt=-1199.101, nper=360.0)
    assert result.solved_value == pytest.approx(0.005, rel=1e-3)


@pytest.mark.parametrize(
    ("cashflows", "expected"),
    [
        # 1 -> 12 in one period is an IRR of 1100%.
        ([-1.0, 12.0], 11.0),
        # IRR = 999/period (99,900%) lies in the log-spaced tail of the search grid.
        ([-1.0, 1000.0], 999.0),
    ],
)
def test_irr_finds_very_large_rates(cashflows: list[float], expected: float) -> None:
    result = irr(cashflows)
    assert result.irr == pytest.approx(expected, rel=1e-9)
    assert result.is_unique is True


def test_mirr_ignores_zero_cashflow() -> None:
    # A 0.0 flow is neither financed nor reinvested. n=3; fv_pos = 500*1.12 + 700 = 1260,
    # pv_neg = 1000 -> MIRR = (1260/1000)**(1/3) - 1 = 0.080083 (computed by hand, so a
    # mishandled zero flow that still happened to be positive would be caught).
    result = mirr([-1000.0, 0.0, 500.0, 700.0], finance_rate=0.10, reinvest_rate=0.12)
    assert result.mirr == pytest.approx(0.080083, rel=1e-4)


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


def test_loan_final_payment_clears_balance_large_principal() -> None:
    # Float drift over 360 periods would leave principal unpaid at the end of the term (0.14
    # at this size, which survives rounding to cents); the final payment absorbs it.
    result = loan_schedule(principal=1e13, annual_rate=0.07, term_months=360, include_schedule=True)
    assert result.n_payments == 360
    assert result.schedule[-1].balance == 0.0


def test_loan_final_payment_clears_balance_zero_rate() -> None:
    # At 0% the payment is principal/term exactly and the drift is far larger: without the
    # final-payment adjustment the loan ends 6.88 short and total_paid understates principal.
    principal = 1e15
    result = loan_schedule(principal=principal, annual_rate=0.0, term_months=360)
    assert result.n_payments == 360
    assert result.total_interest == pytest.approx(0.0, abs=1e-6)
    assert result.total_paid == pytest.approx(principal, abs=0.01)


def test_loan_negligible_rate_behaves_as_zero_rate() -> None:
    # (1 + 1e-18/12)**12 == 1.0 in floating point, so the annuity formula would divide
    # by growth - 1 == 0. Fall back to the straight-line payment.
    result = loan_schedule(principal=1200.0, annual_rate=1e-18, term_months=12)
    assert result.monthly_payment == pytest.approx(100.0, rel=1e-9)
    assert result.n_payments == 12
    assert result.total_interest == pytest.approx(0.0, abs=1e-6)


def test_bond_price_accepts_yield_below_minus_one_when_base_positive() -> None:
    # frequency=2 makes the periodic yield -0.75, so the discount base is 0.25 > 0:
    # price = 25/0.25 + 1025/0.25**2 = 100 + 16400 = 16500.
    result = bond_price(face=1000.0, coupon_rate=0.05, years_to_maturity=1.0, ytm=-1.5, frequency=2)
    assert result.price == pytest.approx(16500.0, rel=1e-12)


@pytest.mark.parametrize(
    ("cashflows", "expected"),
    [
        # npv(r) = -(1 - 1/(1+r))**2 touches zero at r = 0 from below: a double root with no
        # sign change, which the grid scan alone would report as "no IRR".
        ([-1.0, 2.0, -1.0], 0.0),
        # The same flow negated: npv touches zero at r = 0 from above.
        ([1.0, -2.0, 1.0], 0.0),
        # A double root at 15%: (1 - 1.15/(1+r))**2 scaled.
        ([-1.0, 2.3, -1.3225], 0.15),
        # The tangent tolerance is relative to the local |f|, so the same shape at 1e9 is
        # also recognised.
        ([-1e9, 2e9, -1e9], 0.0),
    ],
)
def test_irr_finds_tangent_roots(cashflows: list[float], expected: float) -> None:
    result = irr(cashflows)
    assert result.irr == pytest.approx(expected, abs=1e-6)
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


# Below ~1e-16, (1 + rate)**nper - 1 cancels to exactly 0, so a naive annuity factor makes
# the annuity term vanish (fv) or divides by zero (pmt). Found by the property cross-checks.
@pytest.mark.parametrize("rate", [1e-18, 6e-132, 2.2e-309])
def test_tvm_with_a_vanishingly_small_rate_matches_the_zero_rate_answer(rate: float) -> None:
    fv = time_value_of_money(solve_for="fv", pv=-100, pmt=-10, rate=rate, nper=12)
    assert fv.solved_value == pytest.approx(220.0)
    pv = time_value_of_money(solve_for="pv", fv=220, pmt=-10, rate=rate, nper=12)
    assert pv.solved_value == pytest.approx(-100.0)
    pmt = time_value_of_money(solve_for="pmt", pv=-100, fv=220, rate=rate, nper=12)
    assert pmt.solved_value == pytest.approx(-10.0)
    nper = time_value_of_money(solve_for="nper", pv=-100, fv=220, pmt=-10, rate=rate)
    assert nper.solved_value == pytest.approx(12.0)


def test_irr_does_not_report_a_turning_point_next_to_a_sign_change_as_a_root() -> None:
    # [0, -150, 1] has one IRR, -99.33% (1/(1+r) = 150). Near r = -1 the first grid step
    # both changes sign and turns; bisecting the half that does not straddle zero would
    # report its endpoint (-98%, NPV -4997) as the headline IRR.
    result = irr([0, -150, 1])
    assert result.all_irrs == [pytest.approx(-149 / 150)]
    assert result.is_unique


def test_irr_does_not_report_a_minimum_next_to_the_pole_as_a_tangent_root() -> None:
    # [0, 0, -117, 1] has one IRR, 1/117 - 1. Its NPV has a local minimum near -98.7% (NPV
    # -237276) that a tangent tolerance scaled by the neighbouring sample next to the
    # rate == -1 pole (NPV ~1e15) would accept as a double root.
    result = irr([0, 0, -117, 1])
    assert result.all_irrs == [pytest.approx(1 / 117 - 1)]
