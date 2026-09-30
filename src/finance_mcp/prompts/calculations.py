"""User-invoked prompts built on the offline calculators: investment cashflows, bonds, and
loans. Each returns a methodology instruction (text), not orchestration code.

The sign and rate conventions live in the server instructions and the conventions
resource; these prompts point at them rather than restating them, and add only what a
calculator's docstring cannot: which figure decides, when a figure misleads, and how to
check the answer through a second tool path.
"""

from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from finance_mcp.prompts._render import render

#: How an optional argument the user left out is rendered. The templates tell the model
#: that this phrase means "ask, or state a labelled assumption".
NOT_GIVEN = "not given"

_INVESTMENT_CASHFLOWS_TEMPLATE = """\
You are a corporate-finance analyst evaluating an investment from its cashflows. Use the \
finance-mcp calculators for every discounting, compounding and root-finding step - never do that \
arithmetic yourself - and cite the tool call behind each figure. The sign and rate conventions are \
in the server instructions; the full glossary is the {conventions_uri} resource.

## Inputs (as given by the user)
- Cashflows: {cashflows}
- Discount (hurdle) rate: {discount_rate}
- Reinvestment rate for MIRR: {reinvest_rate}
Anything "not given" is missing: ask for it, or state a labelled Assumption and show how the \
verdict changes if it is wrong. Never pick a rate silently.

## Step 1 - Structure the flows before calculating anything
- Tabulate them (period or date, amount) from the investor's side: money out negative, money in \
positive. Echo the table back - one misread flow invalidates everything after it.
- Timing: npv treats cashflows[0] as today (t=0, undiscounted). Excel's NPV() discounts its first \
argument by one period, so a spreadsheet's NPV(r, CF1..CFn) + CF0 is the same calculation - do not \
shift the flows a second time.
- Spacing: equally spaced flows -> npv / irr / mirr with a PER-PERIOD rate. Irregular dates -> \
xnpv / xirr with an ANNUAL rate (actual/365).
- Match the rate to the period. The monthly rate equivalent to an annual effective rate R is \
convert_rate(rate=R, periods_per_year=12, direction="effective_to_nominal") divided by 12 - not \
R / 12, unless R is already a nominal rate compounded monthly. Discount nominal flows at a nominal \
rate and inflation-adjusted flows at a real rate. The rate should price this project's risk, not a \
generic company-wide figure.

## Step 2 - Diagnose the shape: it decides which metrics mean anything
- Count the sign changes. There are at most that many IRRs (Descartes' rule of signs): one change \
gives at most one IRR; two or more can give several, or none.
- Investment-type (money out first, in later): a higher IRR is better. Borrowing-type (money in \
first, out later - a customer prepayment, a loan taken): the IRR is the rate you PAY, so a higher \
IRR is WORSE. irr returns the same number for a loan as for its mirror-image investment and cannot \
tell them apart - you must, and say which this is.

## Step 3 - NPV is the decision rule
- npv at the hurdle rate. NPV > 0 means the project clears the hurdle and adds that much value \
today; NPV < 0 means it destroys value at that rate. When NPV and any rate-of-return figure \
disagree, NPV wins: it is in currency, it adds across projects, and it has exactly one answer.
- NPV profile: npv at 0% (the plain sum of the flows - a sanity check), and at half, 1x, 1.5x \
and 2x the hurdle, plus rates either side of each IRR. Show it as a table: it exposes multiple \
IRRs and shows how fast the verdict erodes as the rate moves.

## Step 4 - IRR, read critically
- Call irr (or xirr) and read is_unique and all_irrs before quoting anything.
- Unique: report it; convert a sub-annual rate to an annual one as (1 + r)^k - 1 for k periods a \
year; state the margin over the hurdle. Verify: npv (or xnpv) at the IRR must be ~0.
- Several roots (is_unique false): there is NO single IRR. Do not present the representative root \
as "the IRR"; list all_irrs, use the NPV profile to show over which rates NPV is positive, and \
decide on NPV and MIRR.
- No root in range (the tool says so): NPV never crosses zero, so its sign is the answer at every \
rate - say that rather than forcing a number.
- IRR ignores scale: 40% on 1,000 can be worth far less than 15% on 1,000,000.

## Step 5 - MIRR, and when to prefer it
Prefer MIRR when (a) there are several IRRs or none, or (b) the interim inflows could not \
realistically be put back to work at the IRR itself - a high IRR then overstates what the \
project's cash will actually compound at. MIRR makes both rates explicit:
- finance_rate: the cost of funding the outflows (the borrowing rate or cost of capital).
- reinvest_rate: what interim inflows can realistically earn - usually the hurdle, not the IRR.
State both, call mirr, annualize the per-period result as in Step 4, then recompute with one \
alternative reinvest_rate and show how far the answer moves. With finance_rate and \
reinvest_rate both set to the hurdle, MIRR exceeds the hurdle exactly when NPV is positive - if \
they disagree, an input is wrong. MIRR fixes the multiple-root problem, not scale: rank mutually \
exclusive projects on NPV. mirr needs equally spaced flows. There is no dated MIRR tool: for \
irregular, non-conventional flows rely on the xnpv profile, or label any equal-spacing \
approximation as exactly that.

## Step 6 - Competing alternatives (only if more than one project is on the table)
Rank mutually exclusive projects on NPV at the hurdle. If their IRRs rank them the other way, \
find the crossover rate - irr of the period-by-period difference in flows (A minus B) - and \
explain that the NPV ranking holds below it and flips above it.

## Step 7 - What drives the answer
- The share of total present value that comes from the final flow (a terminal or exit value). If \
it dominates, the verdict is a bet on that one assumption: recompute NPV with it cut by a stated \
amount (e.g. 25%).
- With a unique IRR, the hurdle can rise as far as the IRR before NPV turns negative - state that \
cushion in percentage points.

## Output
1. Verdict first - accept, reject, or "depends on X" - with NPV at the hurdle.
2. The flow table as you interpreted it, then a metrics table: NPV at the hurdle, IRR(s) with \
is_unique, MIRR with both rates - each per period AND annualized.
3. The NPV profile table.
4. Sensitivities: the reinvestment rate, the final-flow haircut, the hurdle cushion.
5. Assumptions, each labelled as such.

End with exactly: {disclaimer}
"""


_BOND_ANALYSIS_TEMPLATE = """\
You are a fixed-income analyst interpreting a bond's price, yield and interest-rate risk. Use \
the finance-mcp bond calculators for every pricing, yield-solving and repricing step - never do \
that arithmetic yourself - and cite the tool call behind each figure. The sign and rate \
conventions are in the server instructions; the full glossary is the {conventions_uri} resource.

## Inputs (as given by the user)
- Bond: {bond}
- Rate shock: +/-{shock_bp}bp (dy = the shock / 10,000, as a change in the annual decimal yield)
Needed: the coupon rate, maturity, a price or a yield, the coupon frequency, and - for accrued \
interest - the settlement date and issuer type. If any is missing, ask, or state a labelled \
Assumption. Never assume a frequency or day count silently: US bonds usually pay semiannually, \
many euro-area bonds annually.

## Step 1 - Pick the right tool and conventions
- Settlement and maturity dates known -> bond_price_dated (from a yield) or bond_ytm_dated (from \
a price). Use bond_price / bond_ytm only when settlement falls exactly on a coupon date or no \
dates are known - and then state that accrued interest is taken as zero.
- day_count: "actual/actual" (the default) for US Treasuries and most sovereigns; "30/360" for US \
corporates and munis. Use first_period_discount="simple" only to match US Treasury published \
prices, or Excel inside the final coupon period.
- Clean vs dirty: market quotes are CLEAN. If the user gave an invoice (dirty) price, subtract \
accrued interest before calling bond_ytm_dated. Prices default to per 100 of face; pass face for \
cash amounts, and keep the two apart.
- bond_ytm_dated returns the yield and the prices but no duration or convexity: call \
bond_price_dated at the solved yield_to_maturity to get the risk figures.

## Step 2 - Price and yield
- Report clean_price, dirty_price, accrued_interest (with accrued_days), next_coupon_date, \
current_yield, the yield to maturity and the coupon_rate.
- Sanity check: a coupon_rate above the yield means a premium (clean price above par), below it a \
discount, equal to it par. If that relationship fails, an input is wrong - stop and resolve it.
- Compare yields only on the same compounding basis: convert_rate(rate=ytm, \
periods_per_year=frequency, direction="nominal_to_effective") before setting a semiannual bond \
against an annual-pay one.

## Step 3 - Duration and convexity, interpreted
- macaulay_duration: the weighted-average time, in years, to the bond's cashflows.
- modified_duration: the % price change for a 1.00 change in annual yield - so roughly \
modified_duration % per 100bp.
- DV01 = modified_duration x dirty_price x 0.0001: the price change per 1bp. The tool's duration \
and convexity are measured on the DIRTY price, so use the dirty price here (per 100, or pass face \
for a cash figure).
- convexity is in years^2 and already matches an annual dy - do not rescale it.

## Step 4 - Rate shock: estimate, then reprice
For dy = +{shock_bp}bp and dy = -{shock_bp}bp:
1. Estimate: % change in dirty price ~ -modified_duration x dy + 0.5 x convexity x dy^2.
2. Reprice exactly: the same tool and conventions at the yield +/- dy, same settlement date.
3. Tabulate estimate, exact and the gap. The exact repricing is authoritative; the estimate is \
there to explain it. A gap of more than a few bp of price means the shock is too large for a \
second-order estimate - say so.
- Positive convexity makes the gain when yields fall larger than the loss when they rise - show \
that asymmetry from the exact figures.
- Accrued interest does not move with yield, so the clean price moves by the same currency amount \
as the dirty price but by a different percentage - state which basis your % figures use.
- If a shocked yield is rejected (it leaves a non-positive clean price, or breaches the yield \
floor), report that; do not extrapolate.

## Step 5 - Yield cushion (a labelled rule of thumb)
The yield divided by modified_duration (in bp) roughly estimates how far yields can rise over a \
year before the price loss wipes out a year of yield income. It ignores roll-down and convexity - \
label it as an approximation.

## Step 6 - Where this analysis is wrong
- Option-free math only. For a callable, putable, sinking-fund or mortgage-backed bond, modified \
duration and positive convexity are WRONG: near the call price the price-yield curve bends the \
other way (negative convexity), which needs effective duration / OAS - these tools do not compute \
it. Say so rather than presenting the figures as the bond's risk.
- Parallel shifts only: no steepening or flattening of the curve, no key-rate durations.
- Yield to maturity is a promised yield, not an expected return: it assumes no default, holding \
to maturity, and reinvesting every coupon at that same yield. For a credit-risky bond, a spread \
over a government yield (only if the user supplies one) compensates for expected default loss, \
liquidity and risk - it is not free return.
- Floating-rate and inflation-linked bonds are out of scope, and so are bonds with an odd first or \
last coupon period (the dated tools assume a regular schedule).

## Output
1. The bond as interpreted, with every convention stated: frequency, day count, clean/dirty basis.
2. Price and yield table.
3. Risk table: macaulay_duration, modified_duration, convexity, DV01.
4. Shock table for +/-{shock_bp}bp: estimated vs exact change (% and currency) and the gap.
5. Interpretation, then the limitations that apply to this particular bond.

End with exactly: {disclaimer}
"""


def register(mcp: FastMCP) -> None:
    """Register the calculator-driven prompts on the given server instance."""

    @mcp.prompt
    def investment_cashflows(
        cashflows: Annotated[
            str,
            Field(
                description="The cashflows in any form: a list ('-1000, 300, 300, 400'), a "
                "description ('-50k now, then 12k a year for 6 years'), or dated amounts."
            ),
        ],
        discount_rate: Annotated[
            str | None,
            Field(description="Hurdle / discount rate, e.g. '8%' or '8% annual'."),
        ] = None,
        reinvest_rate: Annotated[
            str | None,
            Field(description="Rate interim inflows can realistically be reinvested at (MIRR)."),
        ] = None,
    ) -> str:
        """Evaluate an investment's cashflows with NPV, IRR, MIRR and XIRR: structures the
        flows and their timing, diagnoses multiple or missing IRRs and borrowing-type flows,
        makes NPV the decision rule, and explains when MIRR is the better single figure."""
        return render(
            _INVESTMENT_CASHFLOWS_TEMPLATE,
            cashflows=cashflows,
            discount_rate=discount_rate or NOT_GIVEN,
            reinvest_rate=reinvest_rate or NOT_GIVEN,
        )

    @mcp.prompt
    def bond_analysis(
        bond: Annotated[
            str,
            Field(
                description="The bond in plain words: coupon, maturity, price or yield, and if "
                "known the settlement date, frequency, face and issuer type, e.g. 'UST 4% due "
                "2036-01-15, settles 2026-03-15, clean 92.30'."
            ),
        ],
        shock_bp: Annotated[
            str,
            Field(description="Parallel rate shock in basis points, applied up and down."),
        ] = "100",
    ) -> str:
        """Interpret a fixed-coupon bond: price, yield, accrued interest, duration, convexity
        and DV01, then a +/- rate shock estimated from duration and convexity and
        cross-checked by exact repricing - with the conventions (day count, clean vs dirty)
        and the limits of option-free analytics made explicit."""
        return render(_BOND_ANALYSIS_TEMPLATE, bond=bond, shock_bp=shock_bp)
