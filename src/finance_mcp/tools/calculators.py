"""MCP tool wrappers for the pure financial calculators. Thin: validate + translate."""

from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.data import calculators
from finance_mcp.data.models import (
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
from finance_mcp.tools._dispatch import run_calc


def register(mcp: FastMCP) -> None:
    """Register calculator tools on the given server instance."""

    @mcp.tool
    def time_value_of_money(
        solve_for: Annotated[
            TVMVariable,
            Field(description="Which unknown to solve for. Provide all other variables."),
        ],
        pv: Annotated[
            float | None,
            Field(
                description=(
                    "Present value. Sign convention: cash received is positive, "
                    "cash paid is negative."
                )
            ),
        ] = None,
        fv: Annotated[float | None, Field(description="Future value.")] = None,
        pmt: Annotated[
            float | None,
            Field(description="Payment per period (defaults to 0 if omitted)."),
        ] = None,
        rate: Annotated[
            float | None,
            Field(description="Interest rate per period as a decimal, e.g. 0.05 for 5%."),
        ] = None,
        nper: Annotated[
            float | None,
            Field(description="Number of periods."),
        ] = None,
        when: Annotated[
            Literal["end", "begin"],
            Field(description="Payment timing: 'end' (ordinary) or 'begin' (annuity-due)."),
        ] = "end",
    ) -> TVMResult:
        """Solve compound interest / present & future value / annuity payment / period count / CAGR.

        Covers most personal-finance math via one equation. Examples: future value of a
        deposit (solve_for='fv'), a loan/mortgage payment (solve_for='pmt'), or a CAGR
        between two values with no interim payments (solve_for='rate', pmt=0). Use
        when='begin' for begin-of-period (annuity-due) payments.
        """
        return run_calc(
            lambda: calculators.time_value_of_money(
                solve_for=solve_for, pv=pv, fv=fv, pmt=pmt, rate=rate, nper=nper, when=when
            )
        )

    @mcp.tool
    def bond_price(
        face: Annotated[float, Field(gt=0, description="Face (par) value of the bond.")],
        coupon_rate: Annotated[
            float,
            Field(description="Annual coupon rate as a decimal, e.g. 0.05 for 5%."),
        ],
        years_to_maturity: Annotated[float, Field(gt=0, description="Years until maturity.")],
        ytm: Annotated[
            float,
            Field(description="Annual yield to maturity as a decimal, e.g. 0.06 for 6%."),
        ],
        frequency: Annotated[
            int, Field(gt=0, description="Coupon payments per year, e.g. 2 for semiannual.")
        ] = 2,
    ) -> BondAnalytics:
        """Price a fixed-coupon bond at a given yield, with duration and convexity."""
        return run_calc(
            lambda: calculators.bond_price(
                face=face,
                coupon_rate=coupon_rate,
                years_to_maturity=years_to_maturity,
                ytm=ytm,
                frequency=frequency,
            )
        )

    @mcp.tool
    def bond_ytm(
        face: Annotated[float, Field(gt=0, description="Face (par) value of the bond.")],
        coupon_rate: Annotated[
            float,
            Field(description="Annual coupon rate as a decimal, e.g. 0.05 for 5%."),
        ],
        years_to_maturity: Annotated[float, Field(gt=0, description="Years until maturity.")],
        price: Annotated[float, Field(gt=0, description="Current market price of the bond.")],
        frequency: Annotated[
            int, Field(gt=0, description="Coupon payments per year, e.g. 2 for semiannual.")
        ] = 2,
    ) -> BondYTM:
        """Solve the annual yield to maturity that prices the bond at the given market price."""
        return run_calc(
            lambda: calculators.bond_ytm(
                face=face,
                coupon_rate=coupon_rate,
                years_to_maturity=years_to_maturity,
                price=price,
                frequency=frequency,
            )
        )

    @mcp.tool
    def loan_schedule(
        principal: Annotated[float, Field(gt=0, description="Loan amount borrowed.")],
        annual_rate: Annotated[
            float,
            Field(ge=0, description="Annual interest rate as a decimal, e.g. 0.06 for 6%."),
        ],
        term_months: Annotated[
            int,
            Field(gt=0, description="Loan term in months, e.g. 360 for 30 years."),
        ],
        extra_payment: Annotated[
            float,
            Field(ge=0, description="Extra principal paid each month; shortens the term."),
        ] = 0.0,
        include_schedule: Annotated[
            bool,
            Field(description="Return the full per-period amortization rows (can be large)."),
        ] = False,
    ) -> LoanSchedule:
        """Compute the monthly payment, total interest, and (optionally) the full schedule.

        annual_rate is a nominal APR compounded monthly (periodic rate = annual_rate/12),
        with monthly payments. By default returns just the summary; set
        include_schedule=True for every row.
        """
        return run_calc(
            lambda: calculators.loan_schedule(
                principal=principal,
                annual_rate=annual_rate,
                term_months=term_months,
                extra_payment=extra_payment,
                include_schedule=include_schedule,
            )
        )

    @mcp.tool
    def npv(
        rate: Annotated[
            float,
            Field(description="Discount rate per period as a decimal, e.g. 0.10 for 10%."),
        ],
        cashflows: Annotated[
            list[float],
            Field(
                description="Cashflows by period; cashflows[0] is at t=0 (now), outflows negative."
            ),
        ],
    ) -> NPVResult:
        """Net present value of equally-spaced cashflows (cashflows[0] is at t=0, undiscounted)."""
        return run_calc(lambda: calculators.npv(rate, cashflows))

    @mcp.tool
    def irr(
        cashflows: Annotated[
            list[float],
            Field(description="Cashflows by period; needs >=1 sign change. Outflows negative."),
        ],
    ) -> IRRResult:
        """Internal rate of return (per period) of equally-spaced cashflows.

        Non-conventional flows can have multiple IRRs (see all_irrs/is_unique); use
        mirr for a single unambiguous figure.
        """
        return run_calc(lambda: calculators.irr(cashflows))

    @mcp.tool
    def mirr(
        cashflows: Annotated[
            list[float],
            Field(description="Cashflows by period; needs >=1 negative and >=1 positive."),
        ],
        finance_rate: Annotated[
            float, Field(description="Rate to finance (discount) negative cashflows, as a decimal.")
        ],
        reinvest_rate: Annotated[
            float,
            Field(description="Rate to reinvest (compound) positive cashflows, as a decimal."),
        ],
    ) -> MIRRResult:
        """Modified internal rate of return: single-valued, unlike irr.

        The preferred figure for non-conventional cashflows (more than one sign change),
        since it has exactly one solution given the finance and reinvestment rates.
        """
        return run_calc(lambda: calculators.mirr(cashflows, finance_rate, reinvest_rate))

    @mcp.tool
    def xnpv(
        rate: Annotated[float, Field(description="Annual discount rate as a decimal.")],
        cashflows: Annotated[
            list[DatedCashflow],
            Field(description="Dated cashflows; discounted by actual days from the earliest date."),
        ],
    ) -> NPVResult:
        """Net present value of cashflows on actual calendar dates (irregular spacing allowed).

        Actual/365 day count (matches Excel XNPV); base date is the earliest cashflow.
        """
        return run_calc(lambda: calculators.xnpv(rate, cashflows))

    @mcp.tool
    def xirr(
        cashflows: Annotated[
            list[DatedCashflow],
            Field(
                description="Dated cashflows; needs at least one sign change. Outflows negative."
            ),
        ],
    ) -> IRRResult:
        """Annualized internal rate of return of cashflows on actual calendar dates.

        Actual/365 day count (matches Excel XIRR); base date is the earliest cashflow.
        """
        return run_calc(lambda: calculators.xirr(cashflows))

    @mcp.tool
    def convert_rate(
        rate: Annotated[float, Field(description="The rate to convert, as a decimal.")],
        periods_per_year: Annotated[
            int,
            Field(gt=0, description="Compounding periods per year, e.g. 12 for monthly."),
        ],
        direction: Annotated[
            RateDirection,
            Field(description="Which way to convert."),
        ],
        compounding: Annotated[
            Compounding,
            Field(
                description="Compounding: 'discrete' uses periods_per_year; 'continuous' uses e^r."
            ),
        ] = "discrete",
    ) -> RateConversionResult:
        """Convert between a nominal annual rate (APR) and an effective annual rate (APY/EAR).

        Discrete uses periods_per_year (e.g. 12 = monthly); continuous ignores it
        (EAR = e^nominal - 1; nominal = ln(1 + EAR)).
        """
        return run_calc(
            lambda: calculators.convert_rate(rate, periods_per_year, direction, compounding)
        )
