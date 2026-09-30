"""MCP tool wrappers for the pure financial calculators. Thin: validate + translate."""

import datetime
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.data import calculators
from finance_mcp.data.models import (
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
from finance_mcp.tools._annotations import calculator
from finance_mcp.tools._dispatch import run_calc
from finance_mcp.tools._inputs import (
    MAX_BOND_YEARS,
    MAX_CASHFLOWS,
    MAX_COUPON_FREQUENCY,
    MAX_LOAN_TERM_MONTHS,
    MAX_PERIODS_PER_YEAR,
)

# Parameter types shared by the bond tools, so each description is written once.
CouponRate = Annotated[
    float,
    Field(description="Annual coupon rate as a decimal, e.g. 0.05 for 5%."),
]
BondFace = Annotated[float, Field(gt=0, description="Face (par) value of the bond.")]
YearsToMaturity = Annotated[
    float, Field(gt=0, le=MAX_BOND_YEARS, description="Years until maturity.")
]
CouponFrequency = Annotated[
    int,
    Field(
        gt=0,
        le=MAX_COUPON_FREQUENCY,
        description="Coupon payments per year, e.g. 2 for semiannual.",
    ),
]
Settlement = Annotated[
    datetime.date,
    Field(
        description="Settlement date (ISO 8601, e.g. 2024-03-07): the date the trade "
        "settles and the buyer pays. May fall between coupon dates."
    ),
]
Maturity = Annotated[
    datetime.date,
    Field(
        description="Maturity (redemption) date (ISO 8601). Coupon dates are generated "
        "backward from here, so the schedule sits on this day-of-month."
    ),
]
DatedFace = Annotated[
    float,
    Field(
        gt=0,
        description="Face (par) value. Defaults to 100, so prices come back per 100 "
        "of face -- the market quoting convention. Pass the real position size to get "
        "cash amounts.",
    ),
]
DatedFrequency = Annotated[
    int,
    Field(
        gt=0,
        le=MAX_COUPON_FREQUENCY,
        description="Coupon payments per year, e.g. 2 for semiannual. Must divide 12 "
        "evenly (1, 2, 3, 4, 6 or 12) so coupon dates fall a whole number of months "
        "apart.",
    ),
]
DayCount = Annotated[
    BondDayCount,
    Field(
        description="Day count for the accrued part of the coupon period. "
        "'actual/actual' (the default) is Actual/Actual ICMA, used by US Treasuries and "
        "most sovereigns. '30/360' is the US (NASD) convention for corporates and munis, "
        "and is what Excel uses by default (basis=0).",
    ),
]
PartPeriodDiscount = Annotated[
    FirstPeriodDiscount,
    Field(
        description="How to discount the part period before the next coupon. "
        "'compound' (the default) is the street convention, (1+y)**stub, and matches "
        "Excel's PRICE/YIELD while more than one coupon remains. 'simple' is 1 + stub*y "
        "-- the US Treasury convention in 31 CFR 356 appendix B, and also what Excel "
        "switches to in the FINAL coupon period, where 'compound' is ~0.01 per 100 "
        "higher. They differ by a few thousandths per 100 elsewhere, so use 'simple' to "
        "match Treasury's published figures or Excel inside the last period.",
    ),
]


def register(mcp: FastMCP) -> None:
    """Register calculator tools on the given server instance."""

    @mcp.tool(annotations=calculator("Time Value of Money"))
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
            Field(
                gt=-1,
                description="Interest rate per period as a decimal, e.g. 0.05 for 5%.",
            ),
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

    @mcp.tool(annotations=calculator("Bond Price, Duration & Convexity"))
    def bond_price(
        face: BondFace,
        coupon_rate: CouponRate,
        years_to_maturity: YearsToMaturity,
        ytm: Annotated[
            float,
            Field(description="Annual yield to maturity as a decimal, e.g. 0.06 for 6%."),
        ],
        frequency: CouponFrequency = 2,
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

    @mcp.tool(annotations=calculator("Bond Yield to Maturity"))
    def bond_ytm(
        face: BondFace,
        coupon_rate: CouponRate,
        years_to_maturity: YearsToMaturity,
        price: Annotated[float, Field(gt=0, description="Current market price of the bond.")],
        frequency: CouponFrequency = 2,
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

    @mcp.tool(annotations=calculator("Bond Price by Settlement Date"))
    def bond_price_dated(
        settlement: Settlement,
        maturity: Maturity,
        coupon_rate: CouponRate,
        ytm: Annotated[
            float,
            Field(description="Annual yield to maturity as a decimal, e.g. 0.065 for 6.5%."),
        ],
        face: DatedFace = 100.0,
        frequency: DatedFrequency = 2,
        day_count: DayCount = "actual/actual",
        first_period_discount: PartPeriodDiscount = "compound",
    ) -> BondDatedAnalytics:
        """Price a bond for a settlement date that may fall BETWEEN coupon dates.

        Use this for a real bond quoted by its maturity date; use bond_price only when
        settlement lands exactly on a coupon date. Returns the CLEAN price (quoted, excludes
        accrued interest), the DIRTY price (clean + accrued = the cash the buyer pays), the
        accrued interest, and duration/convexity computed with the fractional first period
        under the standard street convention. Each price is given per 'face' and per 100 of
        face.

        Day count defaults to Actual/Actual ICMA (US Treasuries and most sovereigns); pass
        day_count='30/360' for the US corporate/municipal convention, which reproduces
        Excel's PRICE with basis=0 -- except in the final coupon period, where Excel uses
        simple interest over the stub: pass first_period_discount='simple' to match it there.

        Assumes a regular schedule -- every coupon period a whole 12/frequency months. Bonds
        with an odd (long or short) first or last coupon period are not supported.
        """
        return run_calc(
            lambda: calculators.bond_price_dated(
                settlement=settlement,
                maturity=maturity,
                coupon_rate=coupon_rate,
                ytm=ytm,
                face=face,
                frequency=frequency,
                day_count=day_count,
                first_period_discount=first_period_discount,
            )
        )

    @mcp.tool(annotations=calculator("Bond Yield by Settlement Date"))
    def bond_ytm_dated(
        settlement: Settlement,
        maturity: Maturity,
        coupon_rate: CouponRate,
        clean_price: Annotated[
            float,
            Field(
                gt=0,
                description="The CLEAN market price per 'face' -- the quoted price, EXCLUDING "
                "accrued interest. If you have the dirty/invoice price, subtract accrued "
                "interest first.",
            ),
        ],
        face: DatedFace = 100.0,
        frequency: DatedFrequency = 2,
        day_count: DayCount = "actual/actual",
        first_period_discount: PartPeriodDiscount = "compound",
    ) -> BondDatedYTM:
        """Solve the annual yield to maturity from a bond's CLEAN price at a settlement date.

        The dated counterpart to bond_ytm, and the inverse of bond_price_dated: use it when
        settlement may fall between coupon dates. Also returns the accrued interest and the
        dirty price, so a clean-price quote still tells you the cash amount. Day count
        defaults to Actual/Actual ICMA; '30/360' reproduces Excel's YIELD with basis=0, except
        in the final coupon period -- pass first_period_discount='simple' to match Excel there.
        """
        return run_calc(
            lambda: calculators.bond_ytm_dated(
                settlement=settlement,
                maturity=maturity,
                coupon_rate=coupon_rate,
                clean_price=clean_price,
                face=face,
                frequency=frequency,
                day_count=day_count,
                first_period_discount=first_period_discount,
            )
        )

    @mcp.tool(annotations=calculator("Loan Payment & Amortization Schedule"))
    def loan_schedule(
        principal: Annotated[float, Field(gt=0, description="Loan amount borrowed.")],
        annual_rate: Annotated[
            float,
            Field(ge=0, description="Annual interest rate as a decimal, e.g. 0.06 for 6%."),
        ],
        term_months: Annotated[
            int,
            Field(
                gt=0,
                le=MAX_LOAN_TERM_MONTHS,
                description="Loan term in months, e.g. 360 for 30 years.",
            ),
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

    @mcp.tool(annotations=calculator("Net Present Value"))
    def npv(
        rate: Annotated[
            float,
            Field(gt=-1, description="Discount rate per period as a decimal, e.g. 0.10 for 10%."),
        ],
        cashflows: Annotated[
            list[float],
            Field(
                min_length=1,
                max_length=MAX_CASHFLOWS,
                description="Cashflows by period; cashflows[0] is at t=0 (now), outflows negative.",
            ),
        ],
    ) -> NPVResult:
        """Net present value of equally-spaced cashflows (cashflows[0] is at t=0, undiscounted)."""
        return run_calc(lambda: calculators.npv(rate, cashflows))

    @mcp.tool(annotations=calculator("Internal Rate of Return"))
    def irr(
        cashflows: Annotated[
            list[float],
            Field(
                min_length=2,
                max_length=MAX_CASHFLOWS,
                description="Cashflows by period; needs >=1 sign change. Outflows negative.",
            ),
        ],
    ) -> IRRResult:
        """Internal rate of return (per period) of equally-spaced cashflows.

        Non-conventional flows can have multiple IRRs (see all_irrs/is_unique); use
        mirr for a single unambiguous figure.
        """
        return run_calc(lambda: calculators.irr(cashflows))

    @mcp.tool(annotations=calculator("Modified Internal Rate of Return"))
    def mirr(
        cashflows: Annotated[
            list[float],
            Field(
                min_length=2,
                max_length=MAX_CASHFLOWS,
                description="Cashflows by period; needs >=1 negative and >=1 positive.",
            ),
        ],
        finance_rate: Annotated[
            float,
            Field(
                gt=-1,
                description="Rate to finance (discount) negative cashflows, as a decimal.",
            ),
        ],
        reinvest_rate: Annotated[
            float,
            Field(
                gt=-1,
                description="Rate to reinvest (compound) positive cashflows, as a decimal.",
            ),
        ],
    ) -> MIRRResult:
        """Modified internal rate of return: single-valued, unlike irr.

        The preferred figure for non-conventional cashflows (more than one sign change),
        since it has exactly one solution given the finance and reinvestment rates.
        """
        return run_calc(lambda: calculators.mirr(cashflows, finance_rate, reinvest_rate))

    @mcp.tool(annotations=calculator("Net Present Value (Dated Cashflows)"))
    def xnpv(
        rate: Annotated[float, Field(gt=-1, description="Annual discount rate as a decimal.")],
        cashflows: Annotated[
            list[DatedCashflow],
            Field(
                min_length=1,
                max_length=MAX_CASHFLOWS,
                description="Dated cashflows; discounted by actual days from the earliest date.",
            ),
        ],
    ) -> NPVResult:
        """Net present value of cashflows on actual calendar dates (irregular spacing allowed).

        Actual/365 day count (matches Excel XNPV); base date is the earliest cashflow.
        """
        return run_calc(lambda: calculators.xnpv(rate, cashflows))

    @mcp.tool(annotations=calculator("Internal Rate of Return (Dated Cashflows)"))
    def xirr(
        cashflows: Annotated[
            list[DatedCashflow],
            Field(
                min_length=2,
                max_length=MAX_CASHFLOWS,
                description="Dated cashflows; needs at least one sign change. Outflows negative.",
            ),
        ],
    ) -> IRRResult:
        """Annualized internal rate of return of cashflows on actual calendar dates.

        Actual/365 day count (matches Excel XIRR); base date is the earliest cashflow.
        """
        return run_calc(lambda: calculators.xirr(cashflows))

    @mcp.tool(annotations=calculator("Nominal / Effective Rate Conversion"))
    def convert_rate(
        rate: Annotated[float, Field(description="The rate to convert, as a decimal.")],
        periods_per_year: Annotated[
            int,
            Field(
                gt=0,
                le=MAX_PERIODS_PER_YEAR,
                description="Compounding periods per year, e.g. 12 for monthly (365 = daily is "
                "the finest discrete step; use compounding='continuous' for the limit).",
            ),
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
