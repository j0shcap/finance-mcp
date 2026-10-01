"""Pydantic return models. Every tool returns one of these (never a bare dict)."""

import datetime
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, field_serializer

TVMVariable = Literal["pv", "fv", "pmt", "rate", "nper"]
Statement = Literal["income", "balance", "cashflow"]
NewsSource = Literal["ticker", "search"]
RelevanceCheck = Literal["applied", "not_an_equity", "no_company_name", "unavailable"]
RiskFreeSource = Literal["caller", "treasury_bill", "unavailable"]
StatementPeriod = Literal["annual", "quarterly"]
HistoryPeriod = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
HistoryInterval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
RateDirection = Literal["nominal_to_effective", "effective_to_nominal"]
Compounding = Literal["discrete", "continuous"]
BondDayCount = Literal["actual/actual", "30/360"]
FirstPeriodDiscount = Literal["compound", "simple"]


#: Significant digits kept in market-data output. Yahoo's prices are float32 values
#: widened to float64 (float32 holds ~7.2 digits), so the rest is conversion noise:
#: float32(316.98) arrives as 316.9800109863281.
MARKET_DATA_SIGNIFICANT_DIGITS = 7


def round_significant(value: float, digits: int = MARKET_DATA_SIGNIFICANT_DIGITS) -> float:
    """Round ``value`` to ``digits`` significant digits, never rounding away integer digits.

    316.9800109863281 -> 316.98 and 1.133529782295227 -> 1.13353, but a volume of
    49875295 keeps every digit: past ``digits`` integer digits the value is rounded to
    a whole number instead. Zero, NaN and infinities pass through unchanged.
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, max(0, digits - 1 - math.floor(math.log10(abs(value)))))


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round_significant(value)
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    return value


class MarketData(BaseModel):
    """Base for results built from Yahoo data: floats are rounded on output only.

    Rounding happens in serialization, so attributes keep full precision for the
    analytics that read them, and the published schema is unchanged. Calculator results
    do not use this base: they are exact math that callers check against Excel.
    """

    # Deliberately no return annotation: pydantic takes a serializer's return type as the
    # field's published output schema, so "-> Any" would erase every field's type. Left
    # unannotated, each field keeps its own schema.
    @field_serializer("*", mode="wrap")
    def _round_market_floats(self, value: Any, handler: SerializerFunctionWrapHandler):  # type: ignore[no-untyped-def]
        return _round_floats(handler(value))


#: Shared by every result that echoes a risk-free rate, so the three stay word-for-word alike.
_RISK_FREE_SOURCE_DESCRIPTION = (
    "Where risk_free_rate came from. 'caller': the rate passed in. 'treasury_bill': none was "
    "passed, so it is the 13-week US T-bill yield (Yahoo ^IRX) averaged over the measured dates "
    "and converted from its discount quote to an effective annual rate. 'unavailable': none was "
    "passed and that average could not be formed - see risk_free_rate_note; the figures that "
    "need a rate are null."
)
_RISK_FREE_NOTE_DESCRIPTION = (
    "Why the rate is unavailable and how to proceed (pass risk_free_rate explicitly); null "
    "otherwise."
)


class TVMResult(BaseModel):
    """Result of a time-value-of-money computation."""

    solved_for: TVMVariable = Field(description="Which variable was solved for.")
    solved_value: float = Field(description="The computed value of the solved_for variable.")
    pv: float = Field(description="Present value (cash received positive, cash paid negative).")
    fv: float = Field(description="Future value (same sign convention as pv).")
    pmt: float = Field(description="Payment per period (same sign convention as pv).")
    rate: float = Field(description="Interest rate per period (decimal, e.g. 0.05).")
    nper: float = Field(description="Number of periods.")


class AmortizationRow(BaseModel):
    """One period of a loan amortization schedule."""

    period: int = Field(description="1-based period index.")
    payment: float = Field(description="Total payment made this period.")
    principal: float = Field(description="Portion applied to principal.")
    interest: float = Field(description="Portion applied to interest.")
    balance: float = Field(description="Remaining principal after this period.")


class LoanSchedule(BaseModel):
    """Amortization summary and full schedule for a loan or mortgage."""

    monthly_payment: float = Field(description="Scheduled monthly payment (excl. extra).")
    n_payments: int = Field(description="Number of payments until payoff.")
    total_paid: float = Field(description="Sum of all payments made.")
    total_interest: float = Field(description="Total interest paid over the loan.")
    interest_saved: float = Field(
        description="Interest the extra payment saves versus the same loan without it; "
        "0 when extra_payment is 0."
    )
    payments_saved: int = Field(
        description="Monthly payments the extra payment saves (months off the term) versus "
        "the same loan without it; 0 when extra_payment is 0."
    )
    schedule: list[AmortizationRow] = Field(description="Per-period amortization rows.")


class NPVResult(BaseModel):
    """Net present value at a given discount rate."""

    rate: float = Field(
        description="Discount rate used, as a decimal: per period for npv, annual for xnpv."
    )
    npv: float = Field(description="Net present value.")


class IRRResult(BaseModel):
    """Internal rate of return of a cashflow series.

    Non-conventional cashflows (more than one sign change) can have several IRRs;
    in that case ``is_unique`` is False and ``irr`` is only a representative value.
    """

    irr: float = Field(
        description="Representative IRR as a decimal: per period for irr, annual for xirr. "
        "When not unique this is the smallest non-negative root (or the root nearest zero if "
        "all are negative); inspect all_irrs and is_unique, or use mirr() for a single value."
    )
    all_irrs: list[float] = Field(
        default_factory=list,
        description="Every real IRR found in (-100%, 1,000,000%], as decimals, ascending.",
    )
    is_unique: bool = Field(
        default=True,
        description="True when exactly one IRR exists; when False, irr is one of several.",
    )


class MIRRResult(BaseModel):
    """Modified internal rate of return (always unique given the two rates)."""

    mirr: float = Field(
        description="Modified IRR per period, as a decimal (annualize externally if needed)."
    )
    finance_rate: float = Field(
        description="Per-period rate used to discount negative cashflows, as a decimal."
    )
    reinvest_rate: float = Field(
        description="Per-period rate used to compound positive cashflows, as a decimal."
    )


class DatedCashflow(BaseModel):
    """A single cashflow on a calendar date (for XNPV/XIRR)."""

    date: datetime.date = Field(description="Cashflow date (ISO 8601, e.g. 2024-01-31).")
    amount: float = Field(description="Cashflow amount; outflows negative, inflows positive.")


class RateConversionResult(BaseModel):
    """Result of a nominal<->effective annual rate conversion."""

    input_rate: float = Field(description="The rate provided, as a decimal.")
    periods_per_year: int = Field(description="Compounding periods per year used.")
    direction: RateDirection = Field(description="Conversion performed.")
    compounding: Compounding = Field(
        default="discrete", description="Compounding convention used for the conversion."
    )
    converted_rate: float = Field(description="The resulting rate, as a decimal.")


class BondAnalytics(BaseModel):
    """Price and interest-rate risk metrics for a fixed-coupon bond."""

    price: float = Field(
        description="Present value of the bond, in the same units as face. Priced on a coupon "
        "date, so clean and dirty price coincide."
    )
    current_yield: float = Field(description="Annual coupon divided by price, as a decimal.")
    macaulay_duration: float = Field(description="Macaulay duration in years.")
    modified_duration: float = Field(description="Modified duration in years (price sensitivity).")
    convexity: float = Field(description="Convexity in years^2.")


class BondYTM(BaseModel):
    """Yield to maturity solved from a bond's market price."""

    yield_to_maturity: float = Field(description="Annual yield to maturity, as a decimal.")


class BondDatedAnalytics(BaseModel):
    """A bond priced for a settlement date, which may fall between two coupon dates.

    Prices come in two flavours and both are reported, because confusing them misstates
    the cash by up to one coupon: the CLEAN price is what the market quotes, the DIRTY
    price is what a buyer actually pays. Each is given per ``face`` and per 100 of face
    (the quoting convention). Duration and convexity use the same part-period convention
    as the price (``first_period_discount``) and are measured against the dirty price.
    """

    settlement: datetime.date = Field(description="Settlement date the bond was priced for.")
    maturity: datetime.date = Field(description="Maturity (redemption) date.")
    previous_coupon_date: datetime.date = Field(
        description="Coupon date at or immediately before settlement; interest accrues from here. "
        "Equals settlement when settlement is itself a coupon date."
    )
    next_coupon_date: datetime.date = Field(
        description="Next coupon date after settlement; the first cashflow the buyer receives."
    )
    periods_remaining: int = Field(
        description="Coupons still to be paid, counting next_coupon_date through maturity."
    )
    frequency: int = Field(description="Coupon payments per year used (2 = semiannual).")
    day_count: BondDayCount = Field(
        description="Day-count convention used to measure the accrued part of the coupon period."
    )
    first_period_discount: FirstPeriodDiscount = Field(
        description="How the part-period stub was discounted: 'compound' (the street/Excel "
        "convention, (1+y)**(stub)) or 'simple' (the US Treasury convention in 31 CFR 356 "
        "appendix B, 1 + stub*y). They differ by a few thousandths per 100 on a part period."
    )
    accrued_days: float = Field(
        description="Days from previous_coupon_date to settlement, on this day count "
        "('A' in the market formulas). Zero on a coupon date."
    )
    period_days: float = Field(
        description="Days in the current coupon period, on this day count ('E'): the actual "
        "days between the surrounding coupon dates for actual/actual, or a nominal "
        "360/frequency for 30/360."
    )
    accrued_fraction: float = Field(
        description="Elapsed fraction of the current coupon period, accrued_days / period_days. "
        "0.0 on a coupon date; the remaining 1 - this is the fractional first period the "
        "cashflows are discounted over."
    )
    accrued_interest: float = Field(
        description="Interest accrued since previous_coupon_date, in currency per 'face'. The "
        "buyer pays this to the seller on top of the quoted (clean) price."
    )
    accrued_interest_per_100: float = Field(
        description="Accrued interest per 100 of face: the market convention, and what Excel's "
        "ACCRINT reports."
    )
    clean_price: float = Field(
        description="CLEAN price per 'face': present value of the remaining cashflows EXCLUDING "
        "accrued interest. This is the quoted/traded price, and what a yield is solved from."
    )
    dirty_price: float = Field(
        description="DIRTY price per 'face' (also 'full' or 'invoice' price): clean_price + "
        "accrued_interest, i.e. the present value of every remaining cashflow. This is the cash "
        "the buyer actually pays at settlement."
    )
    clean_price_per_100: float = Field(
        description="Clean price per 100 of face, the market quoting convention. Equals Excel's "
        "PRICE when day_count='30/360', except in the final coupon period, where Excel uses "
        "simple interest over the stub (first_period_discount='simple')."
    )
    dirty_price_per_100: float = Field(
        description="Dirty price per 100 of face: clean_price_per_100 + accrued_interest_per_100."
    )
    current_yield: float = Field(
        description="Annual coupon divided by the CLEAN price (a simple income measure; it "
        "ignores the pull to par, so it is not a yield to maturity)."
    )
    macaulay_duration: float = Field(
        description="Macaulay duration in years: the cashflow-weighted average time to payment, "
        "weighted by present value against the DIRTY price."
    )
    modified_duration: float = Field(
        description="Modified duration in years: the approximate percentage fall in the DIRTY "
        "price per 1.00 (100 percentage points) rise in annual yield, so a 1 basis point move "
        "is this figure times 0.0001."
    )
    convexity: float = Field(
        description="Convexity in years^2, against the DIRTY price: the second-order correction "
        "to the duration estimate of a price change."
    )


class BondDatedYTM(BaseModel):
    """Yield to maturity solved from a bond's clean price for a given settlement date."""

    yield_to_maturity: float = Field(
        description="Annual yield to maturity as a decimal (0.065 = 6.5%), compounded at "
        "'frequency' per year."
    )
    clean_price: float = Field(
        description="The CLEAN price that was solved from, per 'face' (echoed back as given)."
    )
    accrued_interest: float = Field(
        description="Interest accrued to settlement, per 'face', implied by the coupon schedule."
    )
    dirty_price: float = Field(
        description="clean_price + accrued_interest per 'face': the cash the buyer pays."
    )
    first_period_discount: FirstPeriodDiscount = Field(
        description="Part-period discounting convention the yield was solved under: 'compound' "
        "(street/Excel) or 'simple' (US Treasury)."
    )


class Quote(MarketData):
    """A current price snapshot for one ticker."""

    symbol: str = Field(description="Ticker symbol.")
    currency: str | None = Field(default=None, description="Quote currency (ISO 4217, e.g. 'USD').")
    price: float = Field(description="Latest price, in quote currency.")
    previous_close: float | None = Field(
        default=None, description="Previous session close, in quote currency."
    )
    change: float | None = Field(
        default=None, description="Price change vs previous close, in quote currency."
    )
    change_percent: float | None = Field(
        default=None, description="Change vs previous close, as a PERCENT (1.5 = 1.5%)."
    )
    day_high: float | None = Field(default=None, description="Intraday high, in quote currency.")
    day_low: float | None = Field(default=None, description="Intraday low, in quote currency.")
    year_high: float | None = Field(default=None, description="52-week high, in quote currency.")
    year_low: float | None = Field(default=None, description="52-week low, in quote currency.")
    market_cap: float | None = Field(
        default=None, description="Market capitalization in quote currency (absolute units)."
    )
    volume: float | None = Field(
        default=None, description="Volume of the latest session, in shares."
    )


class QuoteError(MarketData):
    """Why one ticker in a get_quote batch could not be fetched."""

    symbol: str = Field(
        description="The ticker that failed, normalized (or exactly as given if it could not be)."
    )
    error: str = Field(
        description="Why it failed: an invalid/delisted symbol, or a source/network failure. "
        "Read it before retrying - retrying an invalid symbol will not help."
    )


class QuoteResult(MarketData):
    """Quotes for a batch of tickers: the ones that worked, plus per-ticker failures."""

    quotes: list[Quote] = Field(
        description="Successful quotes in the order their tickers were requested "
        "(duplicate/equivalent spellings collapse to one entry)."
    )
    errors: list[QuoteError] = Field(
        default_factory=list,
        description="One entry per ticker that could not be fetched; empty when all succeeded. "
        "A failed ticker does NOT invalidate the quotes that are present.",
    )


class PriceBar(MarketData):
    """One OHLCV bar. Prices are auto-adjusted for splits and dividends."""

    date: str = Field(
        description="Bar timestamp (ISO 8601). Date-only ('2026-09-25') for the daily and "
        "longer intervals (1d/1wk/1mo), which cover whole sessions; a full timestamp with the "
        "exchange's UTC offset ('2026-09-25T09:35:00-04:00') for intraday intervals (1m-1h)."
    )
    open: float = Field(description="Adjusted open, in quote currency.")
    high: float = Field(description="Adjusted high, in quote currency.")
    low: float = Field(description="Adjusted low, in quote currency.")
    close: float = Field(description="Adjusted close, in quote currency.")
    volume: float = Field(description="Volume, in shares.")


class PriceSummary(MarketData):
    """Compact summary over the requested history window."""

    start_date: str = Field(description="First bar date/timestamp (see PriceBar.date).")
    end_date: str = Field(description="Last bar date/timestamp (see PriceBar.date).")
    start_close: float = Field(description="Adjusted close of the first bar, in quote currency.")
    end_close: float = Field(description="Adjusted close of the last bar, in quote currency.")
    total_return_percent: float = Field(
        description="Change from first to last adjusted close, as a PERCENT (5.0 = 5%)."
    )
    period_high: float = Field(description="Highest high over the window, in quote currency.")
    period_low: float = Field(description="Lowest low over the window, in quote currency.")
    bars: int = Field(description="Number of bars in the full window.")


class PriceHistory(MarketData):
    """OHLCV history plus a computed summary."""

    symbol: str = Field(description="Ticker symbol.")
    period: str = Field(description="Requested period, e.g. '1mo'.")
    interval: str = Field(description="Requested interval, e.g. '1d'.")
    bars: list[PriceBar] = Field(description="OHLCV bars (most recent, may be truncated).")
    summary: PriceSummary = Field(description="Summary computed over the full window.")
    truncated: bool = Field(
        description="True if bars were capped; summary still covers the full window."
    )


class FinancialStatement(MarketData):
    """A financial statement (income/balance/cashflow) as a label -> per-period values table."""

    symbol: str = Field(description="Ticker symbol.")
    statement: Statement = Field(description="Which statement.")
    period: StatementPeriod = Field(description="Reporting period granularity.")
    currency: str | None = Field(
        default=None,
        description="Currency (ISO 4217) the values are reported in: Yahoo's financialCurrency, "
        "falling back to the quote currency. This can differ from the currency the shares trade "
        "in (SAP reports in EUR while its US listing quotes in USD), so never compare absolute "
        "figures across companies without checking it. Null if Yahoo does not report it.",
    )
    period_ends: list[str] = Field(
        description="Period-end dates (ISO 8601), most recent first; values align to this order."
    )
    line_items: dict[str, list[float | None]] = Field(
        description="Line item label -> values aligned to period_ends, in the company's reporting "
        "currency in absolute units (e.g. 416161000000 = 416.161B); null if not reported."
    )
    missing_line_items: list[str] = Field(
        default_factory=list,
        description="Requested line-item labels that this statement does not contain. Labels are "
        "matched ignoring case and extra whitespace, so anything listed here is genuinely a "
        "different label - check line_item_suggestions and available_line_items and retry.",
    )
    available_line_items: list[str] = Field(
        default_factory=list,
        description="Every line-item label the statement contains. Populated only when some "
        "requested label was missing (otherwise it would just repeat line_items' keys).",
    )
    line_item_suggestions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="For each missing label, the closest actual labels (e.g. 'Revenue' -> "
        "'Total Revenue'). Omits labels with no close match.",
    )


class DividendEvent(MarketData):
    """A single cash dividend."""

    date: str = Field(description="Ex-dividend date (ISO 8601).")
    amount: float = Field(description="Cash dividend per share, in quote currency.")


class SplitEvent(MarketData):
    """A single stock split."""

    date: str = Field(description="Split date (ISO 8601).")
    ratio: float = Field(
        description="Shares after the split per share before (e.g. 4.0 = a 4-for-1 split)."
    )


class CompanyProfile(MarketData):
    """Company profile and key stats, with recent corporate actions."""

    symbol: str = Field(description="Ticker symbol.")
    name: str | None = Field(default=None, description="Company name.")
    sector: str | None = Field(default=None, description="Sector.")
    industry: str | None = Field(default=None, description="Industry.")
    country: str | None = Field(default=None, description="Country.")
    website: str | None = Field(default=None, description="Website URL.")
    employees: int | None = Field(default=None, description="Number of full-time employees.")
    summary: str | None = Field(default=None, description="Business description (free text).")
    currency: str | None = Field(
        default=None,
        description="Quote currency (ISO 4217, e.g. 'USD'). Financial statements may be reported "
        "in a different currency.",
    )
    market_cap: float | None = Field(
        default=None, description="Market capitalization in quote currency (absolute units)."
    )
    trailing_pe: float | None = Field(
        default=None, description="Trailing twelve-month price/earnings ratio."
    )
    forward_pe: float | None = Field(
        default=None, description="Forward price/earnings ratio (next-year estimate)."
    )
    dividend_yield: float | None = Field(
        default=None,
        description="Trailing dividend yield as a PERCENT (5.92 = 5.92%, not 0.0592).",
    )
    beta: float | None = Field(
        default=None, description="Beta vs the market over ~5 years (1.0 = moves with the market)."
    )
    recent_dividends: list[DividendEvent] = Field(
        default_factory=list, description="Most recent dividends (newest last)."
    )
    splits: list[SplitEvent] = Field(default_factory=list, description="Stock split history.")


class KeyMetrics(MarketData):
    """Valuation / profitability / leverage ratios as reported by Yahoo. Units vary by field.

    Absolute amounts are NOT all in one currency: the financialData figures (total debt/cash,
    free cash flow, EBITDA) are in ``financial_currency``, the EPS fields in ``currency``. When
    the two differ (ADRs and other cross-listings) Yahoo computes the price multiples and
    enterprise value across both currencies, so those are unreliable; the cross-listing rule in
    the finance://conventions resource says what such a company can be compared on.
    """

    symbol: str = Field(description="Ticker symbol.")
    currency: str | None = Field(
        default=None,
        description="Quote currency (ISO 4217, e.g. 'USD') the shares trade in; the unit for "
        "the EPS fields.",
    )
    financial_currency: str | None = Field(
        default=None,
        description="Currency (ISO 4217) the company reports its financials in (Yahoo's "
        "financialCurrency); the unit for total_debt, total_cash, free_cashflow, ebitda, "
        "revenue_per_share and book_value. May differ from `currency`, e.g. SAP reports in EUR "
        "while its US listing quotes in USD.",
    )
    trailing_pe: float | None = Field(default=None, description="Trailing P/E ratio.")
    forward_pe: float | None = Field(default=None, description="Forward P/E ratio.")
    price_to_book: float | None = Field(
        default=None,
        description="Price/book ratio. Unreliable for a cross-listing (financial_currency != "
        "currency): Yahoo divides the listed price by book value in another currency.",
    )
    price_to_sales: float | None = Field(
        default=None,
        description="Price/sales (TTM) ratio. Unreliable for a cross-listing: Yahoo divides "
        "market cap in `currency` by revenue in `financial_currency`.",
    )
    peg_ratio: float | None = Field(default=None, description="P/E-to-growth ratio.")
    enterprise_value: float | None = Field(
        default=None,
        description="Enterprise value (absolute units), as Yahoo derives it from market cap "
        "and the balance sheet. In `currency` for a domestic listing; for a cross-listing it "
        "mixes both currencies and is badly wrong - recompute it from market cap, total_debt "
        "and total_cash at a quoted FX rate.",
    )
    ev_to_ebitda: float | None = Field(
        default=None,
        description="Enterprise value / EBITDA ratio. Unreliable wherever enterprise_value is.",
    )
    ev_to_revenue: float | None = Field(
        default=None,
        description="Enterprise value / revenue ratio. Unreliable wherever enterprise_value is.",
    )
    return_on_equity: float | None = Field(
        default=None, description="Return on equity as a FRACTION (0.27 = 27%)."
    )
    return_on_assets: float | None = Field(
        default=None, description="Return on assets as a FRACTION (0.27 = 27%)."
    )
    gross_margins: float | None = Field(
        default=None, description="Gross margin as a FRACTION (0.27 = 27%)."
    )
    operating_margins: float | None = Field(
        default=None, description="Operating margin as a FRACTION (0.27 = 27%)."
    )
    profit_margins: float | None = Field(
        default=None, description="Net profit margin as a FRACTION (0.27 = 27%)."
    )
    ebitda_margins: float | None = Field(
        default=None, description="EBITDA margin as a FRACTION (0.27 = 27%)."
    )
    debt_to_equity: float | None = Field(
        default=None, description="Debt-to-equity as a PERCENT (79.5 = 79.5%), not a multiple."
    )
    current_ratio: float | None = Field(default=None, description="Current ratio.")
    quick_ratio: float | None = Field(default=None, description="Quick ratio.")
    total_debt: float | None = Field(
        default=None, description="Total debt in `financial_currency` (absolute units)."
    )
    total_cash: float | None = Field(
        default=None, description="Total cash in `financial_currency` (absolute units)."
    )
    free_cashflow: float | None = Field(
        default=None, description="Free cash flow in `financial_currency` (absolute units)."
    )
    ebitda: float | None = Field(
        default=None, description="EBITDA in `financial_currency` (absolute units)."
    )
    trailing_eps: float | None = Field(
        default=None, description="Trailing EPS, per share in `currency` (matches trailing_pe)."
    )
    forward_eps: float | None = Field(
        default=None, description="Forward EPS, per share in `currency` (matches forward_pe)."
    )
    revenue_per_share: float | None = Field(
        default=None, description="Revenue per share, in `financial_currency`."
    )
    book_value: float | None = Field(
        default=None, description="Book value per share, in `financial_currency`."
    )


class RecommendationPeriod(MarketData):
    """Analyst recommendation counts for one period bucket."""

    period: str = Field(
        description="Period bucket relative to the current month: '0m' = current month, "
        "'-1m' = one month ago, '-2m' = two months ago, '-3m' = three months ago."
    )
    strong_buy: int = Field(description="Number of analysts with a Strong Buy rating.")
    buy: int = Field(description="Number of analysts with a Buy rating.")
    hold: int = Field(description="Number of analysts with a Hold rating.")
    sell: int = Field(description="Number of analysts with a Sell rating.")
    strong_sell: int = Field(description="Number of analysts with a Strong Sell rating.")


class AnalystData(MarketData):
    """Sell-side analyst consensus and price targets as reported by Yahoo."""

    symbol: str = Field(description="Ticker symbol.")
    currency: str | None = Field(
        default=None,
        description="Currency (ISO 4217, e.g. 'USD') for all *_price fields.",
    )
    current_price: float | None = Field(
        default=None, description="Latest traded price, in the result's currency."
    )
    recommendation_key: str | None = Field(
        default=None,
        description="Consensus recommendation string, e.g. 'buy', 'hold', 'strong_buy'.",
    )
    recommendation_mean: float | None = Field(
        default=None,
        description="Mean analyst recommendation on a 1-5 scale: 1.0 = Strong Buy, "
        "2.0 = Buy, 3.0 = Hold, 4.0 = Sell, 5.0 = Strong Sell.",
    )
    number_of_analysts: int | None = Field(
        default=None,
        description="Number of analysts contributing to the consensus estimates.",
    )
    target_mean_price: float | None = Field(
        default=None, description="Mean analyst 12-month price target, in the result's currency."
    )
    target_median_price: float | None = Field(
        default=None,
        description="Median analyst 12-month price target, in the result's currency.",
    )
    target_high_price: float | None = Field(
        default=None, description="Highest analyst 12-month price target, in the result's currency."
    )
    target_low_price: float | None = Field(
        default=None, description="Lowest analyst 12-month price target, in the result's currency."
    )
    recommendation_trend: list[RecommendationPeriod] = Field(
        default_factory=list,
        description="Per-period recommendation counts, newest first (up to 4 months). "
        "Empty list if no analyst coverage data is available.",
    )


class NewsArticle(MarketData):
    """A single recent news item about a symbol, as surfaced by Yahoo Finance."""

    title: str = Field(description="Headline text.")
    publisher: str | None = Field(
        default=None, description="Publisher display name, e.g. 'Yahoo Finance'; may be null."
    )
    link: str | None = Field(
        default=None, description="Canonical article URL (else a click-through URL); may be null."
    )
    published: str | None = Field(
        default=None,
        description="Publish time as an ISO 8601 UTC timestamp, e.g. '2026-05-31T11:44:34Z'; "
        "may be null.",
    )
    summary: str | None = Field(
        default=None, description="Short blurb summarizing the article; may be empty or null."
    )
    mentions_company: bool | None = Field(
        default=None,
        description="True if the title or summary names the company or its ticker; False if "
        "neither does, which usually means a market-wide story Yahoo filed under the ticker. "
        "It is a whole-word text match that leans toward False: brand and executive names "
        "('Google' for Alphabet, 'Musk' for Tesla) and bare one- or two-letter tickers ('HD' "
        "without '$' or parentheses) do not count, so read a False article's title before "
        "discarding it. A company named by a common word can still match a Title-Case "
        "headline ('Price Target'). Null when not assessed - see the result's relevance_check.",
    )


class NewsResult(MarketData):
    """Recent news for a symbol, newest first."""

    symbol: str = Field(description="Ticker symbol.")
    articles: list[NewsArticle] = Field(
        description="Recent news articles, newest first; empty if no news is available."
    )
    source: NewsSource = Field(
        default="ticker",
        description=(
            "Which Yahoo endpoint served these articles. 'ticker' is the per-symbol news "
            "stream, which carries a summary for each article. 'search' is the fallback used "
            "when that stream returns nothing: it still carries title, publisher, link and "
            "publish time, but NO summary, so every summary is null for a reason unrelated to "
            "the articles themselves - do not read that as the stories being contentless."
        ),
    )
    relevance_check: RelevanceCheck = Field(
        default="unavailable",
        description=(
            "Whether each article's mentions_company was assessed. 'applied': it was. "
            "'not_an_equity': the symbol is an ETF, index, fund, coin or currency pair, where "
            "market-wide news is relevant, so every flag is null. 'no_company_name': Yahoo has "
            "no name for this symbol (often an unknown symbol), so every flag is null. "
            "'unavailable': fetching the name failed (see relevance_note) - every flag is null "
            "and a retry may succeed. The articles themselves are unaffected in every case."
        ),
    )
    relevance_note: str | None = Field(
        default=None,
        description="Why relevance_check is 'no_company_name' or 'unavailable'; null otherwise.",
    )


class SymbolMatch(MarketData):
    """One search hit resolving a name/query to a tradable symbol."""

    symbol: str = Field(description="Ticker symbol, e.g. 'AAPL'.")
    name: str | None = Field(
        default=None, description="Company or instrument long name, or short name as a fallback."
    )
    quote_type: str | None = Field(
        default=None,
        description="Instrument type returned by Yahoo: EQUITY, ETF, CRYPTOCURRENCY, "
        "FUTURE, INDEX, MUTUALFUND, or similar.",
    )
    exchange: str | None = Field(
        default=None,
        description="Exchange display name where the instrument trades, e.g. 'NASDAQ'.",
    )
    sector: str | None = Field(default=None, description="Sector classification, if available.")
    industry: str | None = Field(default=None, description="Industry classification, if available.")
    score: float | None = Field(
        default=None,
        description="Yahoo relevance score for this result (higher = better match to the query).",
    )


class SymbolSearchResult(MarketData):
    """Search results for a query, best-match first."""

    query: str = Field(description="The search query that produced these results.")
    matches: list[SymbolMatch] = Field(
        description="Matching symbols ordered by relevance (best match first)."
    )


class PerformanceStats(MarketData):
    """Return and risk statistics computed from daily closes over the requested window."""

    symbol: str = Field(description="Ticker symbol.")
    period: str = Field(description="Look-back window requested, e.g. '1y'.")
    bars: int = Field(description="Number of daily closes used.")
    start_date: str = Field(description="First close date (ISO 8601).")
    end_date: str = Field(description="Last close date (ISO 8601).")
    total_return_percent: float = Field(
        description="Total return over the window, as a PERCENT (12.3 = 12.3%)."
    )
    annualized_return_percent: float | None = Field(
        default=None,
        description=(
            "Annualized return (CAGR) over the actual calendar span between start_date and "
            "end_date, percent. Over a one-year window this equals total_return_percent. "
            "Null when the span is under 85 days (just under three months), because "
            "annualizing a sub-quarter move extrapolates short-run noise into a yearly "
            "figure - use total_return_percent for such windows and do not annualize it "
            "yourself."
        ),
    )
    annualized_volatility_percent: float | None = Field(
        default=None,
        description=(
            "Annualized volatility of daily returns, percent, scaled by periods_per_year. "
            "Null when the window is under 85 days, for the same reason as "
            "annualized_return_percent."
        ),
    )
    periods_per_year: float | None = Field(
        default=None,
        description=(
            "Observations per year inferred from the data and used to scale "
            "annualized_volatility_percent: roughly 252 for a weekday-traded equity and 365 "
            "for a 24/7 instrument such as crypto, and lower for one that was halted or "
            "thinly traded. Null when the window is under 85 days."
        ),
    )
    max_drawdown_percent: float = Field(
        description="Largest peak-to-trough decline, as a negative percent (-23.4 = -23.4%)."
    )
    risk_free_rate: float | None = Field(
        description=(
            "Annual risk-free rate the Sharpe, Sortino and downside figures are measured "
            "against, as an effective annual DECIMAL (0.042 = 4.2%). By default the 13-week "
            "T-bill yield over the window, so those figures are excess-over-cash; a caller's "
            "rate of 0 makes them RAW return per unit of risk. Null when no rate was passed "
            "and the T-bill average could not be formed. Echoed even when the window is too "
            "short to use it."
        ),
    )
    risk_free_rate_source: RiskFreeSource = Field(description=_RISK_FREE_SOURCE_DESCRIPTION)
    risk_free_rate_note: str | None = Field(default=None, description=_RISK_FREE_NOTE_DESCRIPTION)
    sharpe_ratio: float | None = Field(
        default=None,
        description=(
            "Annualized Sharpe ratio: mean return in excess of risk_free_rate divided by the "
            "standard deviation of those excess returns, scaled by periods_per_year. "
            "Dimensionless, so it is comparable across instruments. Null when the window is "
            "under 85 days (no periods_per_year to scale by), the returns never varied, or "
            "risk_free_rate is null."
        ),
    )
    sortino_ratio: float | None = Field(
        default=None,
        description=(
            "Annualized Sortino ratio: the same numerator as sharpe_ratio, but divided by "
            "downside_deviation instead of total volatility, so upside swings are not "
            "penalized. When the numerator is POSITIVE it sits above the Sharpe if the "
            "dispersion was mostly upside; when the numerator is negative the comparison "
            "inverts, so only read the gap between the two on a positive Sharpe. Null "
            "when the window is under 85 days, risk_free_rate is null, or nothing fell below "
            "the risk-free target - null there means 'no downside observed', not 'bad'."
        ),
    )
    downside_deviation_percent: float | None = Field(
        default=None,
        description=(
            "Annualized dispersion of returns BELOW risk_free_rate, percent - the "
            "denominator of sortino_ratio. Measured as a root-mean-square shortfall below "
            "the target rather than a spread around the mean, so it can EXCEED "
            "annualized_volatility_percent when most returns fell short. Null when the "
            "window is under 85 days or risk_free_rate is null; 0.0 means no return fell "
            "below the target."
        ),
    )
    calmar_ratio: float | None = Field(
        default=None,
        description=(
            "annualized_return_percent divided by the magnitude of max_drawdown_percent: "
            "compound return per unit of worst peak-to-trough pain. Dimensionless. Null when "
            "the window is under 85 days or the series never drew down."
        ),
    )
    sma_50: float | None = Field(
        default=None, description="50-day simple moving average; null if < 50 bars."
    )
    sma_200: float | None = Field(
        default=None, description="200-day simple moving average; null if < 200 bars."
    )


class BenchmarkComparison(MarketData):
    """How one instrument performed relative to a benchmark over the dates they share.

    Both series are daily auto-adjusted closes, INNER-JOINED on date: a 24/7 instrument's
    weekend closes are dropped because the benchmark has none, so the asset's weekend move
    lands in the following session's return. ``overlapping_observations`` is the number of
    shared dates actually used - read it before trusting any figure here, since a thin
    overlap (a recent listing, a long halt) makes every statistic noisy.

    Returns are in each instrument's own quote currency. Comparing a non-USD listing against
    a USD benchmark therefore mixes an FX move into beta, alpha and excess return; compare
    like-for-like listings, or treat a cross-currency figure as indicative only.
    """

    symbol: str = Field(description="Ticker symbol analyzed.")
    benchmark: str = Field(description="Benchmark ticker compared against.")
    period: str = Field(description="Look-back window requested, e.g. '1y'.")
    overlapping_observations: int = Field(
        description="Daily closes the two instruments share over the window, after the "
        "inner join on date. Fewer than the asset's own bar count whenever the calendars "
        "differ (a 7-day crypto series against a 5-day equity benchmark)."
    )
    start_date: str = Field(description="First shared close date (ISO 8601).")
    end_date: str = Field(description="Last shared close date (ISO 8601).")
    periods_per_year: float | None = Field(
        default=None,
        description="Observations per year inferred from the OVERLAPPING dates - roughly "
        "252 when either leg trades weekdays only, even if the other trades every day. "
        "Null when the overlap spans under 85 days.",
    )
    risk_free_rate: float | None = Field(
        description="Annual risk-free rate used for alpha, as an effective annual decimal "
        "(0.042 = 4.2%). By default the 13-week T-bill yield averaged over the overlapping "
        "dates; with a beta of exactly 1 it cancels out of alpha entirely. Null when no rate "
        "was passed and the T-bill average could not be formed, and alpha is then null too.",
    )
    risk_free_rate_source: RiskFreeSource = Field(description=_RISK_FREE_SOURCE_DESCRIPTION)
    risk_free_rate_note: str | None = Field(default=None, description=_RISK_FREE_NOTE_DESCRIPTION)
    total_return_percent: float = Field(
        description="The asset's total return over the shared dates, as a PERCENT (12.3 = 12.3%)."
    )
    benchmark_total_return_percent: float = Field(
        description="The benchmark's total return over the same shared dates, as a PERCENT."
    )
    excess_return_percent: float = Field(
        description="total_return_percent minus benchmark_total_return_percent, in "
        "percentage POINTS. A simple difference, not a ratio and not beta-adjusted - for "
        "the beta-adjusted version read alpha_percent."
    )
    annualized_return_percent: float | None = Field(
        default=None,
        description="The asset's CAGR over the shared dates, percent. Null when the "
        "overlap spans under 85 days.",
    )
    benchmark_annualized_return_percent: float | None = Field(
        default=None,
        description="The benchmark's CAGR over the same shared dates, percent. Null when "
        "the overlap spans under 85 days.",
    )
    beta: float | None = Field(
        default=None,
        description="Sensitivity to the benchmark: 1.0 moves with it, above 1 amplifies "
        "it, negative moves against it. Null when the overlap has under two returns or "
        "the benchmark never moved.",
    )
    correlation: float | None = Field(
        default=None,
        description="Pearson correlation of the daily returns, -1 to 1. Read it alongside "
        "beta: a large beta at a low correlation means the moves are big but unrelated, so "
        "the beta explains little. Null when either series never moved.",
    )
    alpha_percent: float | None = Field(
        default=None,
        description="Annualized Jensen's alpha in percentage POINTS: (Ra - Rf) - beta * "
        "(Rb - Rf), the return earned beyond what the beta exposure predicted. Null when "
        "beta, either annualized return, or risk_free_rate is null.",
    )
    tracking_error_percent: float | None = Field(
        default=None,
        description="Annualized standard deviation of the daily active return (asset minus "
        "benchmark), percent. 0 for a perfect tracker. Null when the overlap spans under "
        "85 days.",
    )
    information_ratio: float | None = Field(
        default=None,
        description="Mean active return per unit of tracking error, annualized and "
        "dimensionless: how reliably the asset out- or under-performed rather than by how "
        "much. Null when the overlap spans under 85 days or tracking error is zero.",
    )


class ComparisonError(MarketData):
    """Why one ticker in a compare_tickers batch has no row."""

    symbol: str = Field(
        description="The ticker that failed, normalized (or exactly as given if it could not be)."
    )
    error: str = Field(
        description="Why no row could be built: an invalid/delisted symbol, too little price "
        "history, or a source failure. Read it before retrying - retrying an invalid symbol "
        "will not help."
    )


class TickerComparisonRow(MarketData):
    """One ticker's row in a side-by-side comparison: performance plus key valuation.

    Performance figures come from daily auto-adjusted closes over the requested window
    (returns therefore already include reinvested dividends); valuation figures are Yahoo's
    as-reported ratios, with the units the get_key_metrics glossary describes. A row is
    present whenever its price history was fetched, so the valuation fields can be null with
    ``metrics_error`` explaining why.
    """

    symbol: str = Field(description="Ticker symbol.")
    currency: str | None = Field(
        default=None,
        description="Quote currency (ISO 4217) this row's returns are denominated in.",
    )
    financial_currency: str | None = Field(
        default=None,
        description="Currency the company reports financials in. Differs from `currency` for "
        "ADRs and other cross-listings, whose price_to_sales, price_to_book and EV multiples "
        "Yahoo computes across both currencies - rank such a row on P/E, PEG and the margins.",
    )
    currency_differs: bool = Field(
        default=False,
        description="True when this row's `currency` is known and differs from the table's "
        "base_currency: its returns include an FX component the other rows do not, so do not "
        "rank absolute amounts against them. False when the currency is unknown - a null "
        "currency is unlabelled, not proven different.",
    )
    bars: int = Field(description="Daily closes used for this row's performance figures.")
    start_date: str = Field(description="First close date (ISO 8601).")
    end_date: str = Field(description="Last close date (ISO 8601).")
    total_return_percent: float = Field(
        description="Total return over the window, as a PERCENT (12.3 = 12.3%)."
    )
    annualized_return_percent: float | None = Field(
        default=None, description="CAGR over the window, percent; null under 85 days."
    )
    annualized_volatility_percent: float | None = Field(
        default=None, description="Annualized volatility, percent; null under 85 days."
    )
    max_drawdown_percent: float = Field(
        description="Largest peak-to-trough decline, as a negative percent."
    )
    risk_free_rate: float | None = Field(
        description="Annual risk-free rate this row's Sharpe and Sortino are measured against, "
        "as an effective annual decimal: the caller's rate, or by default the 13-week T-bill "
        "average over THIS row's dates (a younger listing covers fewer of them). Null when "
        "the T-bill average could not be formed."
    )
    risk_free_rate_source: RiskFreeSource = Field(description=_RISK_FREE_SOURCE_DESCRIPTION)
    risk_free_rate_note: str | None = Field(default=None, description=_RISK_FREE_NOTE_DESCRIPTION)
    sharpe_ratio: float | None = Field(
        default=None,
        description="Annualized Sharpe against this row's risk_free_rate; null under 85 days, "
        "when returns never varied, or when the row has no rate.",
    )
    sortino_ratio: float | None = Field(
        default=None,
        description="Annualized Sortino against this row's risk_free_rate; null under 85 "
        "days, with no downside, or when the row has no rate.",
    )
    calmar_ratio: float | None = Field(
        default=None,
        description="CAGR per unit of max drawdown; null under 85 days or with no drawdown.",
    )
    periods_per_year: float | None = Field(
        default=None,
        description="Trading periods per year inferred for THIS row (~252 for a weekday-traded "
        "equity, ~365 for a 24/7 instrument), which scales its annualized and risk-adjusted "
        "figures. Rows with different values were annualized on different calendars - say so "
        "before ranking their volatility or Sharpe against each other. Null under 85 days.",
    )
    trailing_pe: float | None = Field(default=None, description="Trailing P/E ratio.")
    forward_pe: float | None = Field(default=None, description="Forward P/E ratio.")
    price_to_book: float | None = Field(
        default=None,
        description="Price/book ratio. Unreliable for a cross-listing (financial_currency != "
        "currency): Yahoo divides the listed price by book value in another currency.",
    )
    price_to_sales: float | None = Field(
        default=None,
        description="Price/sales (TTM) ratio. Unreliable for a cross-listing: Yahoo divides "
        "market cap in `currency` by revenue in `financial_currency`.",
    )
    peg_ratio: float | None = Field(
        default=None,
        description="P/E-to-growth ratio - the growth-adjusted multiple to rank on, rather "
        "than raw P/E.",
    )
    ev_to_ebitda: float | None = Field(
        default=None,
        description="Enterprise value / EBITDA. Unreliable for a cross-listing: Yahoo mixes "
        "both currencies into the enterprise value.",
    )
    profit_margins: float | None = Field(
        default=None, description="Net profit margin as a FRACTION (0.27 = 27%)."
    )
    return_on_equity: float | None = Field(
        default=None, description="Return on equity as a FRACTION (0.27 = 27%)."
    )
    debt_to_equity: float | None = Field(
        default=None, description="Debt-to-equity as a PERCENT (79.5 = 79.5%), not a multiple."
    )
    metrics_error: str | None = Field(
        default=None,
        description="Set when the valuation metrics could not be fetched for this ticker, "
        "leaving every valuation field above null. The performance figures are unaffected.",
    )


class TickerComparison(MarketData):
    """Side-by-side performance and valuation for a small set of tickers.

    Results are partial, like get_quote: a ticker whose price history could not be fetched
    appears in ``errors`` with no row, and one row's failure never discards the others. Rows
    follow the order the tickers were requested (duplicate spellings collapse).
    """

    period: str = Field(description="Look-back window used for every row, e.g. '1y'.")
    risk_free_rate: float | None = Field(
        description="The annual risk-free rate the caller passed, applied to every row, as a "
        "decimal (0.045 = 4.5%). Null when none was passed: each row then carries its own "
        "13-week T-bill average over its own dates - read the rows' risk_free_rate.",
    )
    risk_free_rate_source: RiskFreeSource = Field(
        description="'caller': every row uses risk_free_rate. 'treasury_bill': each row uses "
        "the T-bill yield over its own dates (a row's own risk_free_rate_source is "
        "'unavailable' where its window is not covered). 'unavailable': the T-bill history "
        "could not be fetched, so no row has a rate - see risk_free_rate_note."
    )
    risk_free_rate_note: str | None = Field(
        default=None,
        description="Why the table-level source is 'unavailable', and how to proceed; null "
        "otherwise.",
    )
    base_currency: str | None = Field(
        default=None,
        description="The `currency` of the first row that reported one; the reference for each "
        "row's currency_differs flag. Null when no row reported a currency.",
    )
    mixed_currencies: bool = Field(
        default=False,
        description="True when at least one row's currency differs from base_currency. The "
        "returns in this table are then not in one unit: an FX move is mixed into the rows "
        "that differ, so compare them on ratios and say so.",
    )
    rows: list[TickerComparisonRow] = Field(
        description="One row per ticker whose price history was fetched, in request order."
    )
    errors: list[ComparisonError] = Field(
        default_factory=list,
        description="One entry per ticker that produced no row; empty when all succeeded.",
    )
