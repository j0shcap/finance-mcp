"""Pure financial calculators. No MCP imports, no I/O — deterministic math only.

Time-value-of-money uses the standard end-of-period (ordinary annuity) equation
with the Excel cash-flow sign convention (money received is positive, money paid
is negative):

    PV*(1+r)^n + PMT*((1+r)^n - 1)/r + FV = 0      (r != 0)
    PV + PMT*n + FV = 0                            (r == 0)
"""

import calendar
import datetime
import math
from collections.abc import Callable, Iterable
from typing import Literal

from finance_mcp.data.errors import InvalidInput
from finance_mcp.data.models import (
    AmortizationRow,
    BondAnalytics,
    BondDatedAnalytics,
    BondDatedYTM,
    BondDayCount,
    BondYTM,
    Compounding,
    DatedCashflow,
    FirstPeriodDiscount,
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


_GOLDEN_INV = (math.sqrt(5.0) - 1.0) / 2.0


def _minimize(f: Callable[[float], float], low: float, high: float) -> float:
    """Return the minimizer of a unimodal ``f`` on [low, high] by golden-section search.

    Deterministic: a fixed 100 iterations with no early exit, which shrinks the bracket
    by 0.618**100 (~1e-21) — far below the precision any caller needs — and keeps the
    number of ``f`` evaluations, and therefore the result, identical on every run.
    """
    a, b = low, high
    c = b - _GOLDEN_INV * (b - a)
    d = a + _GOLDEN_INV * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(100):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN_INV * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN_INV * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def _find_all_roots(
    f: Callable[[float], float],
    low: float = -0.999999,
    high: float = 10.0,
    grid_points: int = 1100,
    log_high: float = 1e4,
    log_points: int = 300,
    dedup_tol: float = 1e-7,
    tangent_tol: float = 1e-9,
) -> list[float]:
    """Find every real root of ``f`` on (low, log_high] by scanning a fixed grid.

    Deterministic: fixed abscissae, iteration counts, and tolerances. The abscissae are
    ``grid_points`` uniform samples on (low, high] followed by ``log_points`` geometric
    samples on (high, log_high] — log spacing keeps the very large rates affordable, so
    an IRR of several hundred times the principal is still found.

    Three kinds of root are recorded:

    * an exact zero landing on an abscissa;
    * a sign change between adjacent abscissae, refined by bisection;
    * a turning point of the sampled sequence, refined by golden-section search. This is
      what catches a tangent (double) root, where ``f`` touches zero without changing
      sign, and a pair of roots closer together than one grid step, where both lie
      inside a single interval and the endpoints share a sign. A refined turning point
      counts as a root when ``|f|`` there has collapsed to the floating-point
      cancellation floor, measured relative to the bracket endpoints (``tangent_tol``);
      if instead it has crossed zero, the two halves each hold a root and are bisected
      separately.

    Used for IRR/XIRR, where non-conventional cashflows can have several roots.
    """
    step = (high - low) / (grid_points - 1)
    abscissae = [low + i * step for i in range(grid_points)]
    if log_high > high:
        ratio = (log_high / high) ** (1.0 / log_points)
        x = high
        for _ in range(log_points):
            x *= ratio
            abscissae.append(x)
    values = [f(x) for x in abscissae]

    roots = [x for x, fx in zip(abscissae, values, strict=True) if fx == 0.0]
    for i in range(1, len(abscissae)):
        if values[i - 1] * values[i] < 0.0:
            roots.append(_bisect_bracket(f, abscissae[i - 1], abscissae[i]))
    for i in range(1, len(abscissae) - 1):
        rise, fall = values[i] - values[i - 1], values[i + 1] - values[i]
        if rise == 0.0 or fall == 0.0 or (rise > 0.0) == (fall > 0.0):
            continue  # flat or monotone here: no turning point to refine
        bracket_low, bracket_high = abscissae[i - 1], abscissae[i + 1]
        if rise < 0.0:  # local minimum of f
            turning = _minimize(f, bracket_low, bracket_high)
        else:  # local maximum of f: minimize -f
            turning = _minimize(lambda t: -f(t), bracket_low, bracket_high)
        f_turning = f(turning)
        tolerance = max(abs(values[i - 1]), abs(values[i + 1])) * tangent_tol
        if abs(f_turning) <= tolerance:
            roots.append(turning)  # tangent (double) root
        elif f_turning * values[i - 1] < 0.0:
            # The extremum overshot zero: a root on each side, both inside one step.
            roots.append(_bisect_bracket(f, bracket_low, turning))
            roots.append(_bisect_bracket(f, turning, bracket_high))

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

    Rounding: ``monthly_payment`` and the per-row ``payment``/``principal``/``interest``/
    ``balance`` amounts are rounded to cents for presentation, while ``total_paid`` and
    ``total_interest`` accumulate the unrounded values and are rounded only at the end.
    Summing the rounded rows can therefore differ from the reported totals by a few
    cents. The last period's payment is adjusted to clear the remaining balance exactly,
    so the schedule always ends at a zero balance and the principal is fully amortized.
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
        try:
            growth: float = (1.0 + monthly_rate) ** term_months
        except OverflowError as exc:
            raise InvalidInput(
                "annual_rate is too large for this term: the compounding factor "
                "(1 + annual_rate/12)**term_months overflowed."
            ) from exc
        if growth == 1.0:
            # A rate so small that compounding it over the whole term is a no-op in
            # floating point. The annuity formula divides by growth - 1, so use the
            # straight-line payment instead of dividing by zero.
            payment = principal / term_months
        else:
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
        if principal_paid >= balance or period == term_months:
            # Final payment. The period check matters even when the scheduled payment
            # would not otherwise finish the loan: accumulated float error can leave a
            # tiny residual balance after the last scheduled period, which would
            # otherwise go unpaid. By construction the payment was solved from this
            # principal, rate, and term, so the residual absorbed here is only noise.
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
    (-100%, 1,000,000%] are returned (see IRRResult.all_irrs / is_unique). For a single
    unambiguous figure use ``mirr``.
    """
    if len(cashflows) < 2:
        raise InvalidInput("irr needs at least two cashflows.")
    if not _has_sign_change(cashflows):
        raise InvalidInput("irr needs at least one sign change in the cashflows.")
    roots = _find_all_roots(lambda r: npv(r, cashflows).npv)
    if not roots:
        raise InvalidInput(
            "No internal rate of return exists in (-100%, 1,000,000%]; consider mirr()."
        )
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
        raise InvalidInput(
            "No internal rate of return exists in (-100%, 1,000,000%]; consider mirr()."
        )
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
            try:
                converted: float = math.exp(rate) - 1.0
            except OverflowError as exc:
                raise InvalidInput("rate is too large to convert: exp(rate) overflowed.") from exc
        else:
            if 1.0 + rate <= 0.0:
                raise InvalidInput("Effective rate must be greater than -1 (-100%).")
            converted = math.log(1.0 + rate)
    elif direction == "nominal_to_effective":
        if 1.0 + rate / periods_per_year <= 0.0:
            raise InvalidInput("Invalid nominal rate for the given compounding frequency.")
        try:
            converted = (1.0 + rate / periods_per_year) ** periods_per_year - 1.0
        except OverflowError as exc:
            raise InvalidInput(
                "rate is too large for the given compounding frequency: "
                "(1 + rate/periods_per_year)**periods_per_year overflowed."
            ) from exc
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


#: Longest settlement-to-maturity span the dated bond calculators will price. Mirrors
#: ``tools/_inputs.MAX_BOND_YEARS``, which bounds the on-coupon tools' ``years_to_maturity``
#: field; the dated span cannot be a static Field bound because it spans two arguments.
#: ``tests/test_bond_dated_logic.py`` pins the two to the same value so they cannot drift.
MAX_BOND_SPAN_YEARS = 100

#: Average calendar year, used only to turn a settlement-to-maturity span into years for
#: the bound above. Nothing priced depends on it; the day counts use exact dates.
_DAYS_PER_YEAR = 365.25


def _is_month_end(d: datetime.date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def _add_months(d: datetime.date, months: int) -> datetime.date:
    """Shift ``d`` by whole ``months``, preserving month-end and clamping short months.

    Bond schedules roll by month, not by day, so the arithmetic has to answer two
    questions the calendar leaves open. A month-end date stays at month-end (a 31 March
    maturity pays on 30 September, which is what the Treasury schedules do), and a day
    number the target month does not have is clamped to its last day (31 January + 1
    month is 28 or 29 February, never 3 March).
    """
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    last = calendar.monthrange(year, month)[1]
    day = last if _is_month_end(d) else min(d.day, last)
    return datetime.date(year, month, day)


def _days_30_360_us(start: datetime.date, end: datetime.date) -> int:
    """Days between two dates on the US (NASD) 30/360 day count -- Excel's ``basis=0``.

    Every month counts as 30 days and every year as 360, after four adjustments applied
    in this order (the order matters: the February rules feed the day-31 rules):

    1. both dates are the last day of February -> the end day becomes 30;
    2. the start date is the last day of February -> the start day becomes 30;
    3. the end day is 31 and the start day is 30 or 31 -> the end day becomes 30;
    4. the start day is 31 -> the start day becomes 30.
    """
    start_day, end_day = start.day, end.day
    if start.month == 2 and _is_month_end(start):
        if end.month == 2 and _is_month_end(end):
            end_day = 30
        start_day = 30
    if end_day == 31 and start_day >= 30:
        end_day = 30
    if start_day == 31:
        start_day = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (end_day - start_day)


def _coupon_schedule(
    settlement: datetime.date, maturity: datetime.date, frequency: int
) -> tuple[datetime.date, datetime.date, int]:
    """Resolve the coupon period containing ``settlement``, working back from maturity.

    Returns ``(previous_coupon, next_coupon, periods_remaining)``. Coupon dates are
    generated backward from ``maturity`` in steps of ``12 // frequency`` months, so the
    schedule is anchored on the maturity day-of-month -- the market convention, and the
    only end that is always a real payment date. ``periods_remaining`` counts the coupons
    still to be paid, ``next_coupon`` through ``maturity`` inclusive.

    Settlement exactly on a coupon date yields that date as ``previous_coupon``: a fresh
    period has just begun, so nothing has accrued yet.

    Each date is recomputed from ``maturity`` rather than from the previous step, so the
    month-end rule cannot ratchet a schedule off its anchor day (stepping 31 March back
    one month at a time would reach 28 February and stay there).

    Assumes a regular schedule: every period is a whole ``12 // frequency`` months. Odd
    (long or short) first or last coupon periods are out of scope. Requires
    ``settlement < maturity``; callers validate that first.
    """
    step = 12 // frequency
    periods = 1
    next_coupon = maturity
    previous = _add_months(maturity, -step)
    while previous > settlement:
        periods += 1
        next_coupon = previous
        previous = _add_months(maturity, -step * periods)
    return previous, next_coupon, periods


def _metrics_compound(
    face: float, coupon: float, y: float, n: int, f: float
) -> tuple[float, float, float]:
    """Street convention: the part period is COMPOUNDED, so cashflow k is discounted over
    ``w_k = (k - 1) + f`` periods. Returns ``(dirty, macaulay_periods, convexity_periods)``.
    """
    base = 1.0 + y
    price = 0.0
    weighted_time = 0.0
    convexity_sum = 0.0
    for k in range(1, n + 1):
        cash = coupon + (face if k == n else 0.0)
        w = (k - 1) + f
        pv = cash / base**w
        price += pv
        weighted_time += w * pv
        convexity_sum += cash * w * (w + 1.0) / base ** (w + 2.0)
    return price, weighted_time / price, convexity_sum / price


def _metrics_simple(
    face: float, coupon: float, y: float, n: int, f: float
) -> tuple[float, float, float]:
    """US Treasury convention (31 CFR 356 appendix B): the part period earns SIMPLE interest.

    The cashflows are discounted over whole periods and the whole present value is then
    divided by a stub factor ``1 + f*y`` instead of ``(1+y)**f``:

        X     = sum CF_k / (1+y)**(k-1)
        dirty = X / (1 + f*y)

    The stub factor depends on the yield, so duration and convexity are not the compound
    formulas with a different exponent -- they need their own derivatives. Writing
    ``u = 1/(1 + f*y)`` and using ``X' = -x_weighted/(1+y)``:

        macaulay_periods  = x_weighted/X + (1+y)*f*u
        convexity_periods = x_second/X + 2*f*u*x_weighted/((1+y)*X) + 2*(f*u)**2

    At ``f == 1`` the stub factor is exactly ``1 + y``, so both branches reduce to the same
    on-coupon-date numbers.
    """
    base = 1.0 + y
    x = 0.0  # X       = sum CF_k v**(k-1)
    x_weighted = 0.0  # sum (k-1) CF_k v**(k-1) = -(1+y) X'
    x_second = 0.0  # X''     = sum (k-1) k CF_k v**(k+1)
    for k in range(1, n + 1):
        cash = coupon + (face if k == n else 0.0)
        pv = cash / base ** (k - 1)
        x += pv
        x_weighted += (k - 1) * pv
        x_second += (k - 1) * k * cash / base ** (k + 1)
    u = 1.0 / (1.0 + f * y)
    macaulay = x_weighted / x + base * f * u
    convexity = x_second / x + 2.0 * f * u * x_weighted / (base * x) + 2.0 * (f * u) ** 2
    return x * u, macaulay, convexity


def _bond_metrics(
    face: float,
    coupon_rate: float,
    frequency: int,
    y: float,
    n: int,
    first_fraction: float = 1.0,
    first_period_discount: FirstPeriodDiscount = "compound",
) -> tuple[float, float, float, float]:
    """Price a coupon stream and its risk metrics, allowing a fractional first period.

    Returns ``(dirty_price, macaulay_years, modified_years, convexity_years_squared)``.

    ``y`` is the PERIODIC yield (annual / frequency) and ``n`` the number of coupons still
    to be paid. ``first_fraction`` is how much of the first coupon period is still to run:
    1.0 on a coupon date, and ``1 - accrued/period`` when settlement falls inside a period.
    ``first_period_discount`` selects how that part period is discounted -- see
    ``_metrics_compound`` (street/Excel) and ``_metrics_simple`` (US Treasury).

    The price returned is the DIRTY price: the present value of every remaining cashflow,
    which is the cash a buyer pays. On a coupon date nothing has accrued and these collapse
    exactly to the on-coupon formulas -- which is why ``bond_price`` can delegate here
    without moving any of its numbers.

    Macaulay duration is ``-(1+y)/P * dP/dy`` expressed in years and modified duration is
    ``-(1/P) * dP/dY`` for the annual yield ``Y``, which is why ``modified = macaulay/(1+y)``
    holds for both conventions.
    """
    coupon = face * coupon_rate / frequency
    branch = _metrics_simple if first_period_discount == "simple" else _metrics_compound
    dirty, macaulay_periods, convexity_periods = branch(face, coupon, y, n, first_fraction)
    macaulay = macaulay_periods / frequency
    return dirty, macaulay, macaulay / (1.0 + y), convexity_periods / frequency**2


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
    if 1.0 + ytm / frequency <= 0.0:
        # The pricing loop only needs a positive discount base (1 + ytm/frequency);
        # the binding constraint is ytm > -frequency, not ytm > -1.
        raise InvalidInput(
            "ytm must be greater than -frequency so that 1 + ytm/frequency is positive "
            f"(got ytm={ytm} with frequency={frequency})."
        )

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
    price, macaulay, modified, convexity = _bond_metrics(
        face=face, coupon_rate=coupon_rate, frequency=frequency, y=ytm / frequency, n=n
    )
    return BondAnalytics(
        price=price,
        current_yield=face * coupon_rate / price,
        macaulay_duration=macaulay,
        modified_duration=modified,
        convexity=convexity,
    )


def _accrual(
    day_count: BondDayCount,
    previous: datetime.date,
    settlement: datetime.date,
    next_coupon: datetime.date,
    frequency: int,
) -> tuple[float, float]:
    """Days accrued and days in the coupon period, on ``day_count``: market ``(A, E)``.

    * ``"30/360"`` -- the US (NASD) count Excel calls ``basis=0``. ``E`` is the NOMINAL
      ``360/frequency``, not a measured span: that is what Excel's PRICE uses for the
      30/360 bases, and it is what makes accrued interest exactly half a coupon at the
      mid-point of a semiannual period.
    * ``"actual/actual"`` -- ICMA (the convention for US Treasuries and most sovereigns).
      Both sides are real elapsed days, so ``E`` is the true length of THIS coupon period
      and a coupon always accrues to exactly its full amount by the next coupon date.
    """
    if day_count == "30/360":
        return float(_days_30_360_us(previous, settlement)), 360.0 / frequency
    return float((settlement - previous).days), float((next_coupon - previous).days)


def _dated_terms(
    settlement: datetime.date,
    maturity: datetime.date,
    face: float,
    frequency: int,
    day_count: BondDayCount,
) -> tuple[datetime.date, datetime.date, int, float, float]:
    """Validate dated-bond inputs and resolve the schedule; shared by price and yield.

    Returns ``(previous_coupon, next_coupon, periods_remaining, accrued_days, period_days)``.
    """
    if face <= 0.0:
        raise InvalidInput("face must be positive.")
    if frequency < 1 or 12 % frequency != 0:
        raise InvalidInput(
            "frequency must divide 12 evenly (1, 2, 3, 4, 6 or 12) so that coupon dates fall a "
            f"whole number of months apart; got {frequency}."
        )
    if settlement >= maturity:
        raise InvalidInput(
            f"settlement ({settlement}) must be strictly before maturity ({maturity})."
        )
    if (maturity - settlement).days / _DAYS_PER_YEAR > MAX_BOND_SPAN_YEARS:
        raise InvalidInput(
            f"settlement to maturity must span at most {MAX_BOND_SPAN_YEARS} years "
            f"(got {settlement} to {maturity})."
        )
    previous, next_coupon, periods = _coupon_schedule(settlement, maturity, frequency)
    accrued_days, period_days = _accrual(day_count, previous, settlement, next_coupon, frequency)
    return previous, next_coupon, periods, accrued_days, period_days


def _require_bond_yield(ytm: float, frequency: int) -> float:
    """Reject yields the periodic discount base cannot express (same rule as bond_price)."""
    if 1.0 + ytm / frequency <= 0.0:
        raise InvalidInput(
            "ytm must be greater than -frequency so that 1 + ytm/frequency is positive "
            f"(got ytm={ytm} with frequency={frequency})."
        )
    return ytm


def bond_price_dated(
    settlement: datetime.date,
    maturity: datetime.date,
    coupon_rate: float,
    ytm: float,
    face: float = 100.0,
    frequency: int = 2,
    day_count: BondDayCount = "actual/actual",
    first_period_discount: FirstPeriodDiscount = "compound",
) -> BondDatedAnalytics:
    """Price a fixed-coupon bond for a settlement date, which may fall between coupons.

    The dated counterpart to ``bond_price``, which prices on a coupon date only. Coupon
    dates are generated backward from ``maturity`` every ``12 / frequency`` months, so the
    schedule is anchored on the maturity day-of-month and month-ends are preserved (a 31
    March maturity pays on 30 September).

    ``coupon_rate`` and ``ytm`` are annual decimals. ``day_count`` measures the elapsed
    part of the current coupon period and defaults to Actual/Actual ICMA -- the convention
    for US Treasuries and most sovereigns. Pass ``"30/360"`` for the US corporate/municipal
    convention, which is also Excel's default (``basis=0``) and reproduces its PRICE.

    Returns the clean and dirty prices (per ``face`` and per 100), the accrued interest,
    and duration/convexity computed with the fractional first period under the standard
    street convention -- the part period is compounded, ``(1+y)**(DSC/E)``.

    Assumes a regular schedule: every coupon period is a whole ``12 / frequency`` months.
    Bonds with an odd (long or short) first or last coupon period are out of scope, and
    pricing one here would silently use the wrong first period.
    """
    previous, next_coupon, periods, accrued_days, period_days = _dated_terms(
        settlement, maturity, face, frequency, day_count
    )
    _require_bond_yield(ytm, frequency)

    fraction = accrued_days / period_days
    dirty, macaulay, modified, convexity = _bond_metrics(
        face=face,
        coupon_rate=coupon_rate,
        frequency=frequency,
        y=ytm / frequency,
        n=periods,
        first_fraction=1.0 - fraction,
        first_period_discount=first_period_discount,
    )
    accrued = face * coupon_rate / frequency * fraction
    clean = dirty - accrued
    per_100 = 100.0 / face
    return BondDatedAnalytics(
        settlement=settlement,
        maturity=maturity,
        previous_coupon_date=previous,
        next_coupon_date=next_coupon,
        periods_remaining=periods,
        frequency=frequency,
        day_count=day_count,
        first_period_discount=first_period_discount,
        accrued_days=accrued_days,
        period_days=period_days,
        accrued_fraction=fraction,
        accrued_interest=accrued,
        accrued_interest_per_100=accrued * per_100,
        clean_price=clean,
        dirty_price=dirty,
        clean_price_per_100=clean * per_100,
        dirty_price_per_100=dirty * per_100,
        current_yield=face * coupon_rate / clean,
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
    """Solve the annual yield to maturity that prices the bond at ``price``.

    The search starts just above -100%, so this finds yields > -1 only — narrower than
    the range ``bond_price`` can price (ytm > -frequency). Yields that deeply negative
    have no market interpretation, and restricting the bracket keeps the solve robust.
    """
    if price <= 0.0:
        raise InvalidInput("price must be positive.")
    rate = _bisect(
        lambda y: bond_price(face, coupon_rate, years_to_maturity, y, frequency).price - price
    )
    return BondYTM(yield_to_maturity=rate)


def bond_ytm_dated(
    settlement: datetime.date,
    maturity: datetime.date,
    coupon_rate: float,
    clean_price: float,
    face: float = 100.0,
    frequency: int = 2,
    day_count: BondDayCount = "actual/actual",
    first_period_discount: FirstPeriodDiscount = "compound",
) -> BondDatedYTM:
    """Solve the annual yield to maturity from a bond's CLEAN price at a settlement date.

    The dated counterpart to ``bond_ytm``, and the inverse of ``bond_price_dated``. The
    price is the clean (quoted) one, which is how bonds are quoted; the accrued interest
    implied by the coupon schedule is returned alongside, so the caller also sees the dirty
    price -- the cash actually paid.

    As with ``bond_ytm``, the search starts just above -100%, so this finds yields > -1 only
    -- narrower than the range ``bond_price_dated`` can price (ytm > -frequency). Yields
    that deeply negative have no market interpretation, and restricting the bracket keeps
    the solve robust.
    """
    if clean_price <= 0.0:
        raise InvalidInput("clean_price must be positive.")
    _, _, periods, accrued_days, period_days = _dated_terms(
        settlement, maturity, face, frequency, day_count
    )
    fraction = accrued_days / period_days
    accrued = face * coupon_rate / frequency * fraction
    first_fraction = 1.0 - fraction

    def clean_at(ytm: float) -> float:
        dirty, _, _, _ = _bond_metrics(
            face=face,
            coupon_rate=coupon_rate,
            frequency=frequency,
            y=ytm / frequency,
            n=periods,
            first_fraction=first_fraction,
            first_period_discount=first_period_discount,
        )
        return dirty - accrued

    rate = _bisect(lambda ytm: clean_at(ytm) - clean_price)
    return BondDatedYTM(
        yield_to_maturity=rate,
        clean_price=clean_price,
        accrued_interest=accrued,
        dirty_price=clean_price + accrued,
        first_period_discount=first_period_discount,
    )
