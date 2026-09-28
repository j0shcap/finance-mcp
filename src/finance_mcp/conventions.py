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
another source's price."""

#: Sign and rate conventions for the calculators. Shared by the resource and instructions.
CALCULATOR_CONVENTIONS = """\
- Signs follow Excel: cash received is positive, cash paid is negative. A deposit or a loan \
principal you pay out is negative pv; the balance you get back is positive fv. Get this wrong and \
the sign of the answer flips (or a rate solve has no solution).
- Rates are decimals, never percents: 0.05 means 5%.
- npv, irr, mirr and time_value_of_money take a PER-PERIOD rate matching the cashflow spacing \
(monthly flows -> monthly rate). xnpv, xirr, bond_price, bond_ytm, loan_schedule and convert_rate \
take ANNUAL rates; loan_schedule's annual_rate is a nominal APR compounded monthly.
- irr and xirr return a per-period and an annualized rate respectively; both can have several \
roots for non-conventional flows (see all_irrs/is_unique) - prefer mirr then.
- convert_rate moves between a nominal annual rate (APR) and an effective annual rate (APY/EAR)."""

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
- Market data (open world, live, one Yahoo call each): get_quote (batch, up to 25 tickers),
  get_price_history, get_financials, get_company_profile, get_analyst_data, get_news,
  search_symbols, get_key_metrics, analyze_performance. Use search_symbols to resolve a name
  to a ticker first. Tickers are Yahoo symbols, case-insensitive, with the usual prefixes and
  suffixes: BRK-B, ^GSPC, RY.TO, BTC-USD, EURUSD=X.
- Calculators (pure, deterministic, no network): time_value_of_money, loan_schedule, npv, irr,
  mirr, xnpv, xirr, bond_price, bond_ytm, convert_rate.

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
