"""Pure financial calculators. No MCP imports, no I/O — deterministic math only.

Time-value-of-money uses the standard end-of-period (ordinary annuity) equation
with the Excel cash-flow sign convention (money received is positive, money paid
is negative):

    PV*(1+r)^n + PMT*((1+r)^n - 1)/r + FV = 0      (r != 0)
    PV + PMT*n + FV = 0                            (r == 0)
"""

import math
from collections.abc import Callable, Iterable
from typing import Literal

from finance_mcp.data.errors import InvalidInput
from finance_mcp.data.models import (
    AmortizationRow,
    BondAnalytics,
    BondYTM,
    Compounding,
    DatedCashflow,
    IRRResult,
    LoanSchedule,
    MIRRResult,
    NPVResult,
    RateConversionResult,
    RateDirection,
    TVMResult,
    TVMVariable,
)


def _bisect_bracket(f: Callable[[float], float], low: float, high: float) -> float:
    """Bisect for a root of ``f`` in a bracket [low, high] known to straddle zero.

    Uses interval-width convergence (scale-independent), not a raw ``|f|`` tolerance.
    """
    f_low = f(low)
    for _ in range(500):
        mid = (low + high) / 2.0
        if (high - low) / 2.0 < 1e-12:
            return mid
        f_mid = f(mid)
        if f_low * f_mid <= 0.0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def _bisect(f: Callable[[float], float], low: float = -0.999999, high: float = 10.0) -> float:
    """Find a single root of ``f``, expanding ``high`` to bracket a sign change.

    For monotonic functions (TVM rate, bond yield). Raises InvalidInput if no sign
    change can be bracketed within the search range — including when ``f`` overflows
    for large arguments (e.g. ``(1+r)**nper`` with very large ``nper``) before a
    bracket is found, rather than letting a raw OverflowError escape.
    """
    f_low = f(low)
    try:
        f_high = f(high)
    except OverflowError as exc:
        raise InvalidInput("Could not bracket a root in the searched range.") from exc
    attempts = 0
    while f_low * f_high > 0.0 and attempts < 100:
        high *= 2.0
        try:
            f_high = f(high)
        except OverflowError:
            break  # f exploded past the bracket; treat as no bracketable root
        attempts += 1
    if f_low * f_high > 0.0:
        raise InvalidInput("Could not bracket a root in the searched range.")
    return _bisect_bracket(f, low, high)


def _find_all_roots(
    f: Callable[[float], float],
    low: float = -0.999999,
    high: float = 10.0,
    grid_points: int = 1100,
    dedup_tol: float = 1e-7,
) -> list[float]:
    """Find every real root of ``f`` on (low, high] by scanning a fixed uniform grid.

    Deterministic: fixed bounds, resolution, and tolerances. Records exact grid-point
    zeros and bisects every sign-change bracket, then returns sorted, de-duplicated
    roots. Used for IRR/XIRR, where non-conventional cashflows can have several roots.
    """
    step = (high - low) / (grid_points - 1)
    prev_x = low
    prev_f = f(low)
    roots: list[float] = [low] if prev_f == 0.0 else []
    for i in range(1, grid_points):
        x = low + i * step
        fx = f(x)
        if fx == 0.0:
            roots.append(x)
        elif prev_f * fx < 0.0:
            roots.append(_bisect_bracket(f, prev_x, x))
        prev_x, prev_f = x, fx
    roots.sort()
    deduped: list[float] = []
    for root in roots:
        if not deduped or abs(root - deduped[-1]) > dedup_tol:
            deduped.append(root)
    return deduped


def _irr_result(roots: list[float]) -> IRRResult:
    """Build an IRRResult, choosing a deterministic representative scalar root."""
    non_negative = [r for r in roots if r >= 0.0]
    primary = min(non_negative) if non_negative else max(roots)
    return IRRResult(irr=primary, all_irrs=roots, is_unique=len(roots) == 1)


def _require(name: str, value: float | None) -> float:
    if value is None:
        raise InvalidInput(f"'{name}' is required when it is not the variable being solved for.")
    return value


def _require_rate(rate: float) -> float:
    """Reject per-period rates at or below -100%, which the TVM equation cannot express.

    At rate == -1 the growth factor (1+rate)**nper is exactly 0 — ``_pv`` divides by it
    and ``_nper`` takes log(1+rate) = log(0). Below -1 the base is negative, so a
    fractional ``nper`` produces a complex number that the result model cannot hold.
    Every rate-taking calculator here (npv, xnpv, mirr, convert_rate) already requires
    rate > -1; this keeps TVM consistent with them.
    """
    if rate <= -1.0:
        raise InvalidInput("rate must be greater than -1 (-100%) per period.")
    return rate


def _fv(pv: float, pmt: float, rate: float, nper: float, due: bool = False) -> float:
    if rate == 0.0:
        return -(pv + pmt * nper)
    growth: float = (1.0 + rate) ** nper
    mult = (1.0 + rate) if due else 1.0
    return -(pv * growth + pmt * mult * (growth - 1.0) / rate)


def _pv(fv: float, pmt: float, rate: float, nper: float, due: bool = False) -> float:
    if rate == 0.0:
        return -(fv + pmt * nper)
    growth: float = (1.0 + rate) ** nper
    if growth == 0.0:
        # (1+rate)**nper underflowed to exactly 0.0 (rate near -1 with a large nper);
        # the present value it implies is not representable as a float.
        raise InvalidInput(
            "The growth factor (1+rate)**nper underflowed to zero; pv is not "
            "representable for this rate and nper."
        )
    mult = (1.0 + rate) if due else 1.0
    return -(fv + pmt * mult * (growth - 1.0) / rate) / growth


def _pmt(pv: float, fv: float, rate: float, nper: float, due: bool = False) -> float:
    if nper == 0.0:
        raise InvalidInput("Cannot solve for pmt over zero periods (nper must be non-zero).")
    if rate == 0.0:
        return -(pv + fv) / nper
    growth: float = (1.0 + rate) ** nper
    mult = (1.0 + rate) if due else 1.0
    return -(pv * growth + fv) * rate / (mult * (growth - 1.0))


def _nper(pv: float, fv: float, pmt: float, rate: float, due: bool = False) -> float:
    if rate == 0.0:
        if pmt == 0.0:
            raise InvalidInput("Cannot solve for nper when both rate and pmt are zero.")
        return -(pv + fv) / pmt
    if pmt == 0.0:
        if pv == 0.0:
            raise InvalidInput("Cannot solve for nper when pv and pmt are both zero.")
        # pv*(1+r)^n + fv = 0  ->  (1+r)^n = -fv/pv
        ratio = -fv / pv
        if ratio <= 0.0:
            raise InvalidInput("No real solution for nper with the given pv/fv signs.")
        return math.log(ratio) / math.log(1.0 + rate)
    mult = (1.0 + rate) if due else 1.0
    k = pmt * mult / rate
    numerator = k - fv
    denominator = k + pv
    if denominator == 0.0 or numerator / denominator <= 0.0:
        raise InvalidInput("No real solution for nper with the given inputs.")
    return math.log(numerator / denominator) / math.log(1.0 + rate)


def _rate(pv: float, fv: float, pmt: float, nper: float, due: bool = False) -> float:
    if nper == 0.0:
        raise InvalidInput("Cannot solve for rate over zero periods (nper must be non-zero).")
    # Closed form when there are no periodic payments (CAGR); timing is irrelevant.
    if pmt == 0.0:
        if pv == 0.0:
            raise InvalidInput("Cannot solve for rate when pv and pmt are both zero.")
        ratio = -fv / pv
        if ratio <= 0.0:
            raise InvalidInput("No real rate solution for the given pv/fv signs.")
        growth_factor: float = ratio ** (1.0 / nper)
        return growth_factor - 1.0

    # General case: solve f(r) = fv(pv, pmt, r, nper) - fv_target = 0 numerically.
    # Search from high=1.0 (100%/period): covers every realistic per-period rate while
    # keeping (1+r)**nper finite for large nper (high=10 overflows for, e.g., nper=360).
    return _bisect(lambda r: _fv(pv, pmt, r, nper, due) - fv, high=1.0)


def time_value_of_money(
    solve_for: TVMVariable,
    pv: float | None = None,
    fv: float | None = None,
    pmt: float | None = None,
    rate: float | None = None,
    nper: float | None = None,
    when: Literal["end", "begin"] = "end",
) -> TVMResult:
    """Solve a time-value-of-money problem for one unknown variable.

    Provide every variable except the one named by ``solve_for``. ``pmt`` defaults
    to 0 when omitted and not being solved for. Covers compound interest, present/
    future value, annuity payments, period count, and CAGR (solve for ``rate`` with
    ``pmt=0``). ``when`` selects end- or begin-of-period payments (begin = annuity-due).
    """
    due = when == "begin"
    pmt_known = 0.0 if (pmt is None and solve_for != "pmt") else pmt

    if solve_for == "fv":
        value = _fv(
            _require("pv", pv),
            _require("pmt", pmt_known),
            _require_rate(_require("rate", rate)),
            _require("nper", nper),
            due,
        )
    elif solve_for == "pv":
        value = _pv(
            _require("fv", fv),
            _require("pmt", pmt_known),
            _require_rate(_require("rate", rate)),
            _require("nper", nper),
            due,
        )
    elif solve_for == "pmt":
        value = _pmt(
            _require("pv", pv),
            _require("fv", fv),
            _require_rate(_require("rate", rate)),
            _require("nper", nper),
            due,
        )
    elif solve_for == "nper":
        value = _nper(
            _require("pv", pv),
            _require("fv", fv),
            _require("pmt", pmt_known),
            _require_rate(_require("rate", rate)),
            due,
        )
    else:  # rate
        value = _rate(
            _require("pv", pv),
            _require("fv", fv),
            _require("pmt", pmt_known),
            _require("nper", nper),
            due,
        )

    resolved: dict[str, float | None] = {
        "pv": pv,
        "fv": fv,
        "pmt": pmt_known,
        "rate": rate,
        "nper": nper,
    }
    resolved[solve_for] = value
    return TVMResult(
        solved_for=solve_for,
        solved_value=value,
        pv=resolved["pv"] if resolved["pv"] is not None else 0.0,
        fv=resolved["fv"] if resolved["fv"] is not None else 0.0,
        pmt=resolved["pmt"] if resolved["pmt"] is not None else 0.0,
        rate=resolved["rate"] if resolved["rate"] is not None else 0.0,
        nper=resolved["nper"] if resolved["nper"] is not None else 0.0,
    )


def loan_schedule(
    principal: float,
    annual_rate: float,
    term_months: int,
    extra_payment: float = 0.0,
    include_schedule: bool = False,
) -> LoanSchedule:
    """Build an amortization summary for a fixed-rate loan or mortgage.

    ``annual_rate`` is a nominal APR compounded monthly: the periodic rate is
    ``annual_rate / 12`` (not derived from an effective annual rate), and payments
    are monthly. To use an effective annual rate, convert it first with
    ``convert_rate(rate, 12, "effective_to_nominal")``. ``extra_payment`` is an
    additional amount applied to principal each month; it shortens the term.
    The summary (payment, totals, payoff count) is always computed; the full
    per-period rows are returned only when ``include_schedule`` is True.
    """
    if principal <= 0.0:
        raise InvalidInput("principal must be positive.")
    if term_months <= 0:
        raise InvalidInput("term_months must be a positive integer.")
    if annual_rate < 0.0:
        raise InvalidInput("annual_rate cannot be negative.")
    if extra_payment < 0.0:
        raise InvalidInput("extra_payment cannot be negative.")

    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0.0:
        payment = principal / term_months
    else:
        growth: float = (1.0 + monthly_rate) ** term_months
        payment = principal * monthly_rate * growth / (growth - 1.0)

    rows: list[AmortizationRow] = []
    balance = principal
    total_paid = 0.0
    total_interest = 0.0
    period = 0
    # Guard against non-terminating loops; term_months is the natural upper bound.
    while balance > 1e-9 and period < term_months:
        period += 1
        interest = balance * monthly_rate
        scheduled = payment + extra_payment
        principal_paid = scheduled - interest
        if principal_paid >= balance:  # final (partial) payment
            principal_paid = balance
            scheduled = principal_paid + interest
        balance -= principal_paid
        total_paid += scheduled
        total_interest += interest
        if include_schedule:
            rows.append(
                AmortizationRow(
                    period=period,
                    payment=round(scheduled, 2),
                    principal=round(principal_paid, 2),
                    interest=round(interest, 2),
                    balance=round(max(balance, 0.0), 2),
                )
            )

    return LoanSchedule(
        monthly_payment=round(payment, 2),
        n_payments=period,
        total_paid=round(total_paid, 2),
        total_interest=round(total_interest, 2),
        schedule=rows,
    )


def _discount_sum(rate: float, terms: Iterable[tuple[float, float]]) -> float:
    """Sum ``cash / (1 + rate)**exp`` over ``(cash, exp)`` terms, robust near rate == -1.

    The discount factor ``(1 + rate)**exp`` overflows for large ``exp`` (the term then
    decays toward 0) and underflows to ``0.0`` as ``rate`` approaches -1 (the term is then
    infinite). The IRR/XIRR root-finders evaluate present value at the bracket low end
    (rate ~ -1), so a naive ``cash / 0.0`` raised ZeroDivisionError on otherwise-valid long
    cashflow series. Here an overflowed term contributes 0 and an underflowed term
    contributes a signed infinity (the largest-exponent term dominates the limit), so the
    present value stays well-signed and the root scan converges instead of crashing.

    ``rate`` must be > -1, so ``base`` is positive and the power is always real.
    """
    base = 1.0 + rate
    total = 0.0
    dominant_sign = 0
    dominant_exp = -math.inf
    for cash, exp in terms:
        try:
            factor = base**exp
        except OverflowError:
            continue  # denominator overflowed: term is negligible (-> 0)
        if factor == 0.0:  # denominator underflowed: term is infinite
            if cash != 0.0 and exp > dominant_exp:
                dominant_exp = exp
                dominant_sign = 1 if cash > 0.0 else -1
            continue
        total += cash / factor
    return math.copysign(math.inf, dominant_sign) if dominant_sign != 0 else total


def npv(rate: float, cashflows: list[float]) -> NPVResult:
    """Net present value of equally-spaced cashflows, with cashflows[0] at t=0 (undiscounted).

    NPV = sum(cashflows[t] / (1 + rate)**t for t in 0..n). Note this differs from
    Excel's NPV(), which assumes the first cashflow is one period in the future.
    """
    if not cashflows:
        raise InvalidInput("cashflows must not be empty.")
    if rate <= -1.0:
        raise InvalidInput("rate must be greater than -1 (-100%).")
    total = _discount_sum(rate, ((cash, float(period)) for period, cash in enumerate(cashflows)))
    return NPVResult(rate=rate, npv=total)


def _has_sign_change(values: list[float]) -> bool:
    signs = {value > 0.0 for value in values if value != 0.0}
    return len(signs) > 1


def irr(cashflows: list[float]) -> IRRResult:
    """Internal rate of return of equally-spaced cashflows; needs >=1 sign change.

    Non-conventional cashflows can have multiple IRRs; all real roots found in
    (-100%, 1000%] are returned (see IRRResult.all_irrs / is_unique). For a single
    unambiguous figure use ``mirr``.
    """
    if len(cashflows) < 2:
        raise InvalidInput("irr needs at least two cashflows.")
    if not _has_sign_change(cashflows):
        raise InvalidInput("irr needs at least one sign change in the cashflows.")
    roots = _find_all_roots(lambda r: npv(r, cashflows).npv)
    if not roots:
        raise InvalidInput("No internal rate of return exists in (-100%, 1000%]; consider mirr().")
    return _irr_result(roots)


def mirr(cashflows: list[float], finance_rate: float, reinvest_rate: float) -> MIRRResult:
    """Modified internal rate of return; always unique given the two rates.

    Equally-spaced periods, cashflows[0] at t=0. Negative flows are financed at
    ``finance_rate``; positive flows are reinvested at ``reinvest_rate``:

        MIRR = ( FV(positives @ reinvest_rate) / -PV(negatives @ finance_rate) )**(1/n) - 1

    Unlike ``irr`` this is single-valued, so it is the preferred figure for
    non-conventional cashflows (more than one sign change).
    """
    if len(cashflows) < 2:
        raise InvalidInput("mirr needs at least two cashflows.")
    if finance_rate <= -1.0 or reinvest_rate <= -1.0:
        raise InvalidInput("finance_rate and reinvest_rate must be greater than -1 (-100%).")
    if not any(c > 0.0 for c in cashflows) or not any(c < 0.0 for c in cashflows):
        raise InvalidInput("mirr needs at least one negative and one positive cashflow.")
    n = len(cashflows) - 1
    fv_pos = 0.0
    pv_neg = 0.0
    for t, cash in enumerate(cashflows):
        if cash > 0.0:
            fv_pos += cash * (1.0 + reinvest_rate) ** (n - t)
        elif cash < 0.0:
            pv_neg += cash / (1.0 + finance_rate) ** t
    ratio: float = fv_pos / -pv_neg
    result: float = ratio ** (1.0 / n) - 1.0
    return MIRRResult(mirr=result, finance_rate=finance_rate, reinvest_rate=reinvest_rate)


def xnpv(rate: float, cashflows: list[DatedCashflow]) -> NPVResult:
    """Net present value of dated cashflows; base date is the earliest, 365-day basis.

    XNPV = sum(amount / (1 + rate)**((date - base_date).days / 365)). The annual
    ``rate`` discounts by actual elapsed days, so irregular spacing is handled.
    Day count is Actual/365 fixed (matches Excel XNPV; leap years still divide by
    365). The base date is the earliest cashflow, so the result is independent of
    input order (Excel instead uses the first-listed date; with ordered input the
    two agree).
    """
    if not cashflows:
        raise InvalidInput("cashflows must not be empty.")
    if rate <= -1.0:
        raise InvalidInput("rate must be greater than -1 (-100%).")
    base = min(cf.date for cf in cashflows)
    total = _discount_sum(rate, ((cf.amount, (cf.date - base).days / 365.0) for cf in cashflows))
    return NPVResult(rate=rate, npv=total)


def xirr(cashflows: list[DatedCashflow]) -> IRRResult:
    """Annualized internal rate of return of dated cashflows; needs >=1 sign change.

    Uses the same Actual/365 day count and earliest-date base as ``xnpv``.
    """
    if len(cashflows) < 2:
        raise InvalidInput("xirr needs at least two cashflows.")
    if not _has_sign_change([cf.amount for cf in cashflows]):
        raise InvalidInput("xirr needs at least one sign change in the cashflows.")
    roots = _find_all_roots(lambda r: xnpv(r, cashflows).npv)
    if not roots:
        raise InvalidInput("No internal rate of return exists in (-100%, 1000%]; consider mirr().")
    return _irr_result(roots)


def convert_rate(
    rate: float,
    periods_per_year: int,
    direction: RateDirection,
    compounding: Compounding = "discrete",
) -> RateConversionResult:
    """Convert between a nominal annual rate and an effective annual rate (EAR).

    Discrete (m = ``periods_per_year`` compounding periods):
      nominal_to_effective: EAR = (1 + nominal/m)**m - 1
      effective_to_nominal: nominal = m * ((1 + EAR)**(1/m) - 1)
    Continuous (``periods_per_year`` is ignored):
      nominal_to_effective: EAR = exp(nominal) - 1
      effective_to_nominal: nominal = ln(1 + EAR)
    """
    if periods_per_year < 1:
        raise InvalidInput("periods_per_year must be at least 1.")
    if compounding == "continuous":
        if direction == "nominal_to_effective":
            converted: float = math.exp(rate) - 1.0
        else:
            if 1.0 + rate <= 0.0:
                raise InvalidInput("Effective rate must be greater than -1 (-100%).")
            converted = math.log(1.0 + rate)
    elif direction == "nominal_to_effective":
        if 1.0 + rate / periods_per_year <= 0.0:
            raise InvalidInput("Invalid nominal rate for the given compounding frequency.")
        converted = (1.0 + rate / periods_per_year) ** periods_per_year - 1.0
    else:
        if 1.0 + rate <= 0.0:
            raise InvalidInput("Effective rate must be greater than -1 (-100%).")
        converted = periods_per_year * ((1.0 + rate) ** (1.0 / periods_per_year) - 1.0)
    return RateConversionResult(
        input_rate=rate,
        periods_per_year=periods_per_year,
        direction=direction,
        compounding=compounding,
        converted_rate=converted,
    )


def bond_price(
    face: float,
    coupon_rate: float,
    years_to_maturity: float,
    ytm: float,
    frequency: int = 2,
) -> BondAnalytics:
    """Price a fixed-coupon bond and its duration/convexity at a given yield (ytm).

    ``coupon_rate`` and ``ytm`` are annual decimals; ``frequency`` is coupons per year
    (2 = semiannual). Returns the price plus Macaulay/modified duration (years) and
    convexity (years^2).

    Prices the bond AS OF A COUPON DATE: there is no accrued interest and no fractional
    first period (on a coupon date the clean and dirty prices coincide). Therefore
    ``years_to_maturity * frequency`` must be a whole number of coupon periods.
    """
    if face <= 0.0:
        raise InvalidInput("face must be positive.")
    if frequency < 1:
        raise InvalidInput("frequency must be at least 1.")
    if years_to_maturity <= 0.0:
        raise InvalidInput("years_to_maturity must be positive.")
    if ytm <= -1.0:
        raise InvalidInput("ytm must be greater than -1 (-100%).")

    periods = years_to_maturity * frequency
    n = round(periods)
    if abs(periods - n) > 1e-9:
        raise InvalidInput(
            "years_to_maturity * frequency must be a whole number of coupon periods "
            f"(got {periods}); this calculator prices on a coupon date only. Choose a "
            "maturity that lands on a coupon date (a multiple of 1/frequency)."
        )
    if n < 1:
        raise InvalidInput("years_to_maturity * frequency must be at least one period.")
    periodic_coupon = face * coupon_rate / frequency
    y = ytm / frequency

    price = 0.0
    weighted_time = 0.0
    convexity_sum = 0.0
    for k in range(1, n + 1):
        cash = periodic_coupon + (face if k == n else 0.0)
        pv = cash / (1.0 + y) ** k
        price += pv
        weighted_time += k * pv
        convexity_sum += cash * k * (k + 1) / (1.0 + y) ** (k + 2)

    macaulay = (weighted_time / price) / frequency
    modified = macaulay / (1.0 + y)
    convexity = (convexity_sum / price) / (frequency**2)
    return BondAnalytics(
        price=price,
        current_yield=face * coupon_rate / price,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
    )


def bond_ytm(
    face: float,
    coupon_rate: float,
    years_to_maturity: float,
    price: float,
    frequency: int = 2,
) -> BondYTM:
    """Solve the annual yield to maturity that prices the bond at ``price``."""
    if price <= 0.0:
        raise InvalidInput("price must be positive.")
    rate = _bisect(
        lambda y: bond_price(face, coupon_rate, years_to_maturity, y, frequency).price - price
    )
    return BondYTM(yield_to_maturity=rate)
