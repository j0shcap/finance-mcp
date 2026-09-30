"""User-invoked prompts built on the offline calculators: investment cashflows, bonds, and
loans. Each returns a methodology instruction (text), not orchestration code.

The sign and rate conventions live in the server instructions and the conventions
resource; these prompts point at them rather than restating them, and add only what a
calculator's docstring cannot: which figure decides, when a figure misleads, and how to
check the answer through a second tool path.
"""

import math
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import PromptError
from pydantic import Field

from finance_mcp.prompts._render import render

#: How an optional argument the user left out is rendered. The templates tell the model
#: that this phrase means "ask, or state a labelled assumption".
NOT_GIVEN = "not given"


def parse_shock_bp(raw: str) -> str:
    """Normalize a rate shock ("100", "25bp", " 12.5 ") to a plain number of basis points.

    Raises PromptError (which reaches the client despite error masking) for anything else,
    including a value below 1bp: "1%" or "0.01" is almost certainly a percent or a decimal,
    and would otherwise render as garbled text or a shock 100x too small.
    """
    try:
        value = float(raw.strip().lower().removesuffix("bp"))
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value >= 1):
        raise PromptError(
            f"shock_bp must be a number of basis points >= 1, e.g. '100'; got {raw!r}."
        )
    return f"{value:g}"


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
- Unique: report it; annualize a per-period rate r with k periods a year through \
convert_rate(rate=r x k, periods_per_year=k, direction="nominal_to_effective"), which is \
(1 + r)^k - 1; state the margin over the hurdle. Verify: npv (or xnpv) at the IRR must be ~0.
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
- With a unique IRR, the NPV verdict flips where the hurdle crosses the IRR - state that cushion \
in percentage points, in the right direction: investment-type NPV turns negative as the hurdle \
RISES to the IRR, borrowing-type NPV as the hurdle FALLS to it.

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
accrued interest before calling bond_ytm_dated. Accrued interest does not depend on the yield, so \
read accrued_interest from bond_price_dated at any ytm (e.g. the coupon rate) with the same dates \
and conventions. Prices default to per 100 of face; pass face for \
cash amounts, and keep the two apart.
- bond_ytm_dated returns the yield and the prices but no duration or convexity: call \
bond_price_dated at the solved yield_to_maturity to get the risk figures.

## Step 2 - Price and yield
- Report clean_price, dirty_price, accrued_interest (with accrued_days), next_coupon_date, \
current_yield, the yield to maturity and the coupon_rate.
- Sanity check: a coupon_rate above the yield means a premium (clean price above par), below it a \
discount. Near par that is only approximate: between coupon dates a bond whose coupon equals its \
yield prices a few thousandths below 100 clean, which is expected. A clear violation (a coupon \
well above the yield but a price well below par) means an input is wrong - stop and resolve it.
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


_LOAN_PLANNER_TEMPLATE = """\
You are a fee-only financial planner explaining a fixed-rate loan or mortgage. Use the \
finance-mcp calculators for every payment, balance, rate-conversion and discounting step - never \
do that arithmetic yourself - and cite the tool call behind each figure. The sign and rate \
conventions are in the server instructions; the full glossary is the {conventions_uri} resource.

## Inputs (as given by the user)
- Principal: {principal}
- Annual rate: {annual_rate}
- Term: {term_months} months
- Extra monthly principal payment: {extra_payment}
Normalize them first and echo the normalized values back: the rate as a decimal (6.5% -> 0.065), \
the term in months, amounts as plain numbers. If a figure is ambiguous (is "0.5" 0.5% or 50%?), \
confirm it rather than guess.

## Step 1 - Establish which rate this is: it changes the answer
- loan_schedule amortizes at the NOTE rate: a nominal annual rate compounded monthly (the \
periodic rate is annual_rate / 12). A disclosed US "APR" (Truth in Lending) folds points and fees \
into the rate and is NOT the rate to amortize at - use it only to compare offers with each other. \
If the user quoted an APR, ask for the note rate.
- Compounding: US loans compound monthly, but Canadian fixed-rate mortgages compound semiannually. \
Convert with convert_rate(rate=R, periods_per_year=2, direction="nominal_to_effective"), then \
convert_rate(rate=that result, periods_per_year=12, direction="effective_to_nominal"), and pass \
the result as annual_rate.
- The tools assume one fixed rate for the whole term. For an adjustable-rate loan, model only up \
to the first reset and say everything after it is unknown. All figures are principal and \
interest only: property tax, insurance, PMI and HOA dues are excluded - say so.

## Step 2 - Baseline
- loan_schedule(principal, annual_rate, term_months): monthly_payment, total_interest, and total \
interest as a share of the principal.
- Verify through a second path: time_value_of_money(solve_for="pmt", pv=principal, \
rate=annual_rate / 12, nper=term_months, fv=0) must match monthly_payment to the cent (it comes \
back negative: cash you pay).
- APR vs EAR: convert_rate(rate=annual_rate, periods_per_year=12, \
direction="nominal_to_effective") gives the effective annual rate - the true annual cost of the \
note rate once monthly compounding is counted. It still excludes fees, so never present it as the \
disclosed APR.

## Step 3 - How the loan front-loads interest
- The first payment's interest is principal x annual_rate / 12; show what share of the first \
payment that is.
- Remaining balance after 5 and 10 years: time_value_of_money(solve_for="fv", pv=principal, \
pmt=-monthly_payment, rate=annual_rate / 12, nper=60 or 120). The balance is the NEGATIVE of the \
fv it returns; it can differ from a schedule row by cents because monthly_payment is rounded to \
the cent. This avoids pulling a full schedule (include_schedule=True returns every row).

## Step 4 - Extra payments
Run loan_schedule with extra_payment (the user's figure, or a labelled illustrative one if none \
was given). Report the months saved (from n_payments) and the total interest saved. Then frame it \
honestly:
- "Interest saved" is an undiscounted sum of future dollars, so it overstates the benefit. The \
economic return on each prepaid dollar is exactly the loan's note rate - guaranteed and \
risk-free, and reduced by tax only if the interest is deductible and the user itemizes.
- Weigh it against the alternatives: paying down higher-rate debt first, an employer retirement \
match, an emergency fund - prepaid home equity cannot be spent without borrowing it back. \
Expected market returns are not like-for-like: they are risky, the prepayment return is not.
- Extra payments do not lower the required monthly payment (only a recast does), and some loans \
carry prepayment penalties - ask.
- Biweekly plans: half the payment every two weeks adds roughly one payment a year, approximately \
extra_payment = monthly_payment / 12 - label that as an approximation.

## Step 5 - Refinancing (only if the user raises it)
Ask for the current balance and months remaining, the new rate and term, closing costs and points \
(and whether they are rolled into the new loan), and how many months h the user expects to keep \
the loan before selling or refinancing again.
- The naive break-even - closing costs divided by the drop in monthly payment - is wrong when the \
term resets: spreading the remaining balance over a fresh 30 years lowers the payment without \
saving anything. Never report it alone.
- Instead compare the two loans at month h: the payments made over h months plus the balance still \
owed at h (from time_value_of_money, as in Step 3), plus closing costs on the new loan. The \
cheaper total wins at that horizon; find the h where the two cross.
- Then discount it: treat the refinance as a cashflow series - closing costs paid now as a \
negative first flow (none if rolled in, but then the new balance is higher), each month's payment \
saving (old minus new) as a positive flow, and at month h the difference in balances owed (old \
minus new) added to the last flow - and take its npv at the user's opportunity rate, converted \
to a monthly rate. A positive NPV means refinancing pays at that horizon.
- Also run the refinance on the same remaining term, to separate the rate effect from the term \
effect.

## Output
1. The normalized inputs, and the rate type and compounding assumed.
2. Summary table: monthly_payment, total_interest, the effective annual rate, and the balances \
at 5 and 10 years.
3. An extra-payment table (and a refinance table, if relevant): months and interest saved, the \
break-even horizon, the NPV.
4. The tradeoffs in plain words - liquidity, risk, taxes, penalties - and what would change the \
conclusion. This is decision support, not a recommendation.

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
            Field(
                description="Parallel rate shock as a number of basis points, applied up and "
                "down, e.g. '100' (not '1%')."
            ),
        ] = "100",
    ) -> str:
        """Interpret a fixed-coupon bond: price, yield, accrued interest, duration, convexity
        and DV01, then a +/- rate shock estimated from duration and convexity and
        cross-checked by exact repricing - with the conventions (day count, clean vs dirty)
        and the limits of option-free analytics made explicit."""
        return render(_BOND_ANALYSIS_TEMPLATE, bond=bond, shock_bp=parse_shock_bp(shock_bp))

    @mcp.prompt
    def loan_planner(
        principal: Annotated[str, Field(description="Amount borrowed, e.g. '400000'.")],
        annual_rate: Annotated[
            str,
            Field(description="Annual note rate, e.g. '6.5%' (the rate, not a fee-loaded APR)."),
        ],
        term_months: Annotated[
            str, Field(description="Loan term in months, e.g. '360' for 30 years.")
        ],
        extra_payment: Annotated[
            str | None,
            Field(description="Optional extra principal paid each month, e.g. '250'."),
        ] = None,
    ) -> str:
        """Plan a fixed-rate loan or mortgage: payment, total interest, APR vs effective
        annual rate, how extra payments shorten the loan (and what that is really worth),
        and a refinance break-even that does not mistake a longer term for a saving."""
        return render(
            _LOAN_PLANNER_TEMPLATE,
            principal=principal,
            annual_rate=annual_rate,
            term_months=term_months,
            extra_payment=extra_payment or NOT_GIVEN,
        )
