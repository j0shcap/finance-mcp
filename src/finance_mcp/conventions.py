"""The units/sign conventions this server's numbers follow, written once.

Yahoo's units are inconsistent (some ratios are fractions, others are already percents)
and the cashflow tools follow Excel's sign convention, so a model that guesses gets the
magnitudes wrong by 100x or the sign backwards. That guidance therefore has to reach the
model three ways: in the server ``instructions`` (always in context), as the
``finance://conventions`` resource (readable on demand), and inside the analyze_stock
prompt. All three render from the constants here so they can never drift apart.
"""

from fastmcp import FastMCP

CONVENTIONS_URI = "finance://conventions"

#: The two tool families, named for the server instructions. Tested against the registry
#: (see tests/test_tool_metadata.py), so a tool added or renamed cannot leave the model
#: with a stale map of the server.
MARKET_DATA_TOOLS = (
    "search_symbols",
    "get_quote",
    "get_price_history",
    "get_financials",
    "get_company_profile",
    "get_key_metrics",
    "get_analyst_data",
    "analyze_performance",
    "compare_to_benchmark",
    "compare_tickers",
    "get_news",
)
CALCULATOR_TOOLS = (
    "time_value_of_money",
    "loan_schedule",
    "npv",
    "irr",
    "mirr",
    "xnpv",
    "xirr",
    "bond_price",
    "bond_ytm",
    "bond_price_dated",
    "bond_ytm_dated",
    "convert_rate",
)

#: Per-field units. Shared verbatim by the resource and the analyze_stock prompt.
UNITS_GLOSSARY = """\
- return_on_equity, return_on_assets, gross_margins, operating_margins, profit_margins, and \
ebitda_margins are FRACTIONS (0.27 = 27%, 1.41 = 141%) - multiply by 100 for display.
- debt_to_equity is ALREADY A PERCENT (79.5 means 79.5% ~ 0.80x) - it is NOT 79.5x.
- dividend_yield (profile) is ALREADY A PERCENT (0.35 = 0.35%, 5.92 = 5.92%) - not a fraction.
- recommendation_mean is INVERTED: 1 = strong buy ... 5 = strong sell (lower = more bullish).
- P/E, forward P/E, P/B, P/S, PEG, EV/EBITDA, EV/Revenue, current/quick ratio are plain ratios; \
EV, total debt/cash, FCF, EBITDA are absolute amounts; EPS and book value are per-share.
- Absolute amounts are not all in one currency: get_key_metrics reports total debt/cash, FCF, \
EBITDA, revenue per share and book value in financial_currency, while enterprise_value and the EPS \
fields are in currency (the quote currency). get_financials values are in the statement's currency \
field. For most US names these are the same; for ADRs and other cross-listings they are not.
- analyze_performance runs on auto-adjusted prices, so its returns already include reinvested \
dividends (~ total return) - do not add the dividend yield on top.
- analyze_performance annualizes over calendar time, so annualized_return_percent equals \
total_return_percent on a one-year window. For windows under ~3 months it returns null for \
annualized_return_percent, annualized_volatility_percent and periods_per_year - quote the total \
return for that window and never annualize it yourself.
- get_quote returns one entry per ticker in quotes plus a per-ticker errors list. Use its price as \
the single headline price if sources disagree, and read it from the entry whose symbol matches the \
ticker you are pricing, never by position: any ticker that failed is in errors instead, so \
positions shift. A ticker in errors was not fetched at all - say so rather than substituting \
another source's price.
- risk_free_rate (analyze_performance, compare_to_benchmark, compare_tickers) is an ANNUAL \
DECIMAL: 0.045 = 4.5%. It defaults to 0, which makes sharpe_ratio and sortino_ratio RAW return \
per unit of risk rather than excess-over-cash figures - the rate used is echoed in every result, \
so read it before calling a Sharpe "excess". Pass a current T-bill yield when the comparison is \
against cash.
- sharpe_ratio, sortino_ratio, calmar_ratio, beta, correlation and information_ratio are \
DIMENSIONLESS ratios - never percents. downside_deviation_percent, tracking_error_percent, \
alpha_percent and excess_return_percent are PERCENTS; the last two are percentage POINTS of \
difference (alpha_percent 3.0 = 3 points of annualized return beyond what beta predicted), not \
multiples. On a POSITIVE sharpe_ratio, a sortino_ratio above it means the dispersion was mostly \
upside; when the Sharpe is negative that comparison inverts, so do not read the gap as a quality \
signal there.
- Every annualized figure, and every ratio that depends on one, is null when the window spans \
under ~3 months: sharpe_ratio, sortino_ratio, downside_deviation_percent and calmar_ratio on \
analyze_performance, and alpha_percent, tracking_error_percent and information_ratio on \
compare_to_benchmark. beta, correlation and excess_return_percent need no annualization, so they \
survive a short window. A null ratio means "not computable", never "zero".
- compare_to_benchmark INNER-JOINS the two daily close series on date, so a 24/7 instrument \
compared with an equity benchmark contributes only its weekday closes and its weekend move lands \
in the next session's return. overlapping_observations is how many dates were actually used - read \
it first, because a thin overlap makes beta and alpha noise. Returns stay in each instrument's own \
quote currency, so a cross-currency pair mixes an FX move into every figure.
- compare_tickers returns partial results like get_quote: a ticker with no usable price history is \
in errors with no row, while a row whose valuation metrics failed is present with those fields \
null and metrics_error set. Rows flagged currency_differs (and the table-level mixed_currencies) \
are not denominated in base_currency - rank those on ratios, not absolute amounts. Each row \
carries its own periods_per_year, so a 24/7 instrument in the table was annualized on a different \
calendar than the equities beside it - check it before ranking volatility or sharpe_ratio across \
rows."""

#: Sign and rate conventions for the calculators. Shared by the resource and instructions.
CALCULATOR_CONVENTIONS = """\
- Signs follow Excel: cash received is positive, cash paid is negative. A deposit or a loan \
principal you pay out is negative pv; the balance you get back is positive fv. Get this wrong and \
the sign of the answer flips (or a rate solve has no solution).
- Rates are decimals, never percents: 0.05 means 5%.
- npv, irr, mirr and time_value_of_money take a PER-PERIOD rate matching the cashflow spacing \
(monthly flows -> monthly rate). xnpv, xirr, bond_price, bond_ytm, loan_schedule and convert_rate \
take ANNUAL rates; loan_schedule's annual_rate is a nominal APR compounded monthly. So do \
bond_price_dated and bond_ytm_dated.
- irr and xirr return a per-period and an annualized rate respectively; both can have several \
roots for non-conventional flows (see all_irrs/is_unique) - prefer mirr then.
- convert_rate moves between a nominal annual rate (APR) and an effective annual rate (APY/EAR).
- Bond prices come in two flavours and mixing them up misstates the cash by up to a full \
coupon. The CLEAN price is what the market quotes; the DIRTY (or full/invoice) price is \
clean + accrued interest, and is what the buyer actually pays. bond_price_dated reports both, \
per face and per 100 of face; bond_ytm_dated solves from the CLEAN price, so subtract accrued \
interest first if you were given a dirty one. On a coupon date nothing has accrued and the two \
coincide, which is why bond_price/bond_ytm report a single price.
- bond_price_dated and bond_ytm_dated default to the Actual/Actual ICMA day count (US \
Treasuries and most sovereigns), NOT Excel's default of 30/360 - pass day_count='30/360' to \
match Excel's PRICE/YIELD or to price a US corporate or municipal bond."""

CONVENTIONS_DOC = f"""\
# finance-mcp conventions

## Calculators: signs and rates
{CALCULATOR_CONVENTIONS}

## Market data: units per field (Yahoo is inconsistent - read before doing arithmetic)
{UNITS_GLOSSARY}
"""

SERVER_INSTRUCTIONS = f"""\
Finance tools over Yahoo Finance market data plus offline financial calculators.

Two tool families:
- Market data (open world, live, one Yahoo call each): {", ".join(MARKET_DATA_TOOLS)}.
  Resolve a name to a ticker with search_symbols first; get_quote prices up to 25 tickers in
  one call. Tickers are Yahoo symbols, case-insensitive, with the usual prefixes and
  suffixes: BRK-B, ^GSPC, RY.TO, BTC-USD, EURUSD=X.
- Calculators (pure, deterministic, no network): {", ".join(CALCULATOR_TOOLS)}.

Conventions that change the answer:
{CALCULATOR_CONVENTIONS}

Yahoo's market-data units are inconsistent: margins and ROE/ROA are fractions (0.27 = 27%) but
debt_to_equity and dividend_yield are ALREADY PERCENTS (79.5 = 79.5%, 5.92 = 5.92%), and
recommendation_mean is inverted (1 = strong buy, 5 = strong sell). Absolute amounts are not all
in one currency - check each result's currency/financial_currency. Read the {CONVENTIONS_URI}
resource for the full per-field glossary before converting or comparing figures.

Every tool is read-only. Unavailable data is reported as an error or a null field; never fill a
gap with a guess.
"""


def register(mcp: FastMCP) -> None:
    """Expose the conventions glossary as a resource on the given server."""

    @mcp.resource(
        CONVENTIONS_URI,
        name="finance_conventions",
        title="Finance data conventions",
        mime_type="text/markdown",
    )
    def conventions() -> str:
        """Units, sign and rate conventions for every finance-mcp tool result.

        Read this before converting, comparing, or doing arithmetic on figures from the
        market-data tools: the source reports some ratios as fractions and others as
        percents, and absolute amounts are not all in one currency.
        """
        return CONVENTIONS_DOC
