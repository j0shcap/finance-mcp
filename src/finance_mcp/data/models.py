"""Pydantic return models. Every tool returns one of these (never a bare dict)."""

import datetime
from typing import Literal

from pydantic import BaseModel, Field

TVMVariable = Literal["pv", "fv", "pmt", "rate", "nper"]
Statement = Literal["income", "balance", "cashflow"]
StatementPeriod = Literal["annual", "quarterly"]
HistoryPeriod = Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
HistoryInterval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]
RateDirection = Literal["nominal_to_effective", "effective_to_nominal"]
Compounding = Literal["discrete", "continuous"]


class TVMResult(BaseModel):
    """Result of a time-value-of-money computation."""

    solved_for: TVMVariable = Field(description="Which variable was solved for.")
    solved_value: float = Field(description="The computed value of the solved-for variable.")
    pv: float = Field(description="Present value (cash convention: inflow positive).")
    fv: float = Field(description="Future value.")
    pmt: float = Field(description="Payment per period.")
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
    schedule: list[AmortizationRow] = Field(description="Per-period amortization rows.")


class NPVResult(BaseModel):
    """Net present value at a given discount rate."""

    rate: float = Field(description="Discount rate used (per period, or annual for dated).")
    npv: float = Field(description="Net present value.")


class IRRResult(BaseModel):
    """Internal rate of return of a cashflow series.

    Non-conventional cashflows (more than one sign change) can have several IRRs;
    in that case ``is_unique`` is False and ``irr`` is only a representative value.
    """

    irr: float = Field(
        description="Representative IRR (per period, or annual for dated). When not unique "
        "this is the smallest non-negative root (or the root nearest zero if all are "
        "negative); inspect all_irrs and is_unique, or use mirr() for a single value."
    )
    all_irrs: list[float] = Field(
        default_factory=list,
        description="Every real IRR found in (-100%, 1,000,000%], ascending.",
    )
    is_unique: bool = Field(
        default=True,
        description="True when exactly one IRR exists; when False, irr is one of several.",
    )


class MIRRResult(BaseModel):
    """Modified internal rate of return (always unique given the two rates)."""

    mirr: float = Field(description="Modified IRR per period (annualize externally if needed).")
    finance_rate: float = Field(description="Rate used to discount negative cashflows.")
    reinvest_rate: float = Field(description="Rate used to compound positive cashflows.")


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

    price: float = Field(description="Present value (clean price) of the bond.")
    current_yield: float = Field(description="Annual coupon divided by price.")
    macaulay_duration: float = Field(description="Macaulay duration in years.")
    modified_duration: float = Field(description="Modified duration in years (price sensitivity).")
    convexity: float = Field(description="Convexity in years^2.")


class BondYTM(BaseModel):
    """Yield to maturity solved from a bond's market price."""

    yield_to_maturity: float = Field(description="Annual yield to maturity, as a decimal.")


class Quote(BaseModel):
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
        default=None, description="Percent change vs previous close (e.g. 1.5 means 1.5%)."
    )
    day_high: float | None = Field(default=None, description="Intraday high, in quote currency.")
    day_low: float | None = Field(default=None, description="Intraday low, in quote currency.")
    year_high: float | None = Field(default=None, description="52-week high, in quote currency.")
    year_low: float | None = Field(default=None, description="52-week low, in quote currency.")
    market_cap: float | None = Field(
        default=None, description="Market capitalization in quote currency (absolute units)."
    )
    volume: float | None = Field(default=None, description="Last trade volume, in shares.")


class PriceBar(BaseModel):
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


class PriceSummary(BaseModel):
    """Compact summary over the requested history window."""

    start_date: str = Field(description="First bar date/timestamp (see PriceBar.date).")
    end_date: str = Field(description="Last bar date/timestamp (see PriceBar.date).")
    start_close: float = Field(description="Adjusted close of the first bar, in quote currency.")
    end_close: float = Field(description="Adjusted close of the last bar, in quote currency.")
    total_return_percent: float = Field(
        description="Percent change from first to last adjusted close (e.g. 5.0 means 5%)."
    )
    period_high: float = Field(description="Highest high over the window, in quote currency.")
    period_low: float = Field(description="Lowest low over the window, in quote currency.")
    bars: int = Field(description="Number of bars in the full window.")


class PriceHistory(BaseModel):
    """OHLCV history plus a computed summary."""

    symbol: str = Field(description="Ticker symbol.")
    period: str = Field(description="Requested period, e.g. '1mo'.")
    interval: str = Field(description="Requested interval, e.g. '1d'.")
    bars: list[PriceBar] = Field(description="OHLCV bars (most recent, may be truncated).")
    summary: PriceSummary = Field(description="Summary computed over the full window.")
    truncated: bool = Field(
        description="True if bars were capped; summary still covers the full window."
    )


class FinancialStatement(BaseModel):
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


class DividendEvent(BaseModel):
    """A single cash dividend."""

    date: str = Field(description="Ex-dividend date (ISO 8601).")
    amount: float = Field(description="Cash dividend per share, in the trading currency.")


class SplitEvent(BaseModel):
    """A single stock split."""

    date: str = Field(description="Split date (ISO 8601).")
    ratio: float = Field(
        description="Shares after the split per share before (e.g. 4.0 = a 4-for-1 split)."
    )


class CompanyProfile(BaseModel):
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
        description="Trailing dividend yield as a PERCENT, as reported by Yahoo "
        "(e.g. 5.92 means 5.92%, not 0.0592).",
    )
    beta: float | None = Field(
        default=None, description="Beta vs the market over ~5 years (1.0 = moves with the market)."
    )
    recent_dividends: list[DividendEvent] = Field(
        default_factory=list, description="Most recent dividends (newest last)."
    )
    splits: list[SplitEvent] = Field(default_factory=list, description="Stock split history.")


class KeyMetrics(BaseModel):
    """Valuation / profitability / leverage ratios as reported by Yahoo. Units vary by field.

    Absolute amounts are NOT all in one currency: the financialData figures (total debt/cash,
    free cash flow, EBITDA) are in ``financial_currency``, while the market-derived enterprise
    value is in ``currency``. When the two differ (ADRs and other cross-listings) Yahoo's
    derived per-share figures can also be internally inconsistent, so compare such companies
    on the ratios rather than on absolute amounts.
    """

    symbol: str = Field(description="Ticker symbol.")
    currency: str | None = Field(
        default=None,
        description="Quote currency (ISO 4217, e.g. 'USD') the shares trade in; the unit for "
        "enterprise_value and the EPS fields.",
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
    price_to_book: float | None = Field(default=None, description="Price/book ratio.")
    price_to_sales: float | None = Field(default=None, description="Price/sales (TTM) ratio.")
    peg_ratio: float | None = Field(default=None, description="P/E-to-growth ratio.")
    enterprise_value: float | None = Field(
        default=None,
        description="Enterprise value in `currency` (absolute units). Yahoo derives it from "
        "market cap, so for cross-listings whose share count and quote currency disagree it can "
        "be badly wrong - sanity-check it against market_cap + total_debt - total_cash.",
    )
    ev_to_ebitda: float | None = Field(default=None, description="Enterprise value / EBITDA ratio.")
    ev_to_revenue: float | None = Field(
        default=None, description="Enterprise value / revenue ratio."
    )
    return_on_equity: float | None = Field(
        default=None, description="Return on equity, as a fraction (0.27 = 27%)."
    )
    return_on_assets: float | None = Field(
        default=None, description="Return on assets, as a fraction (0.27 = 27%)."
    )
    gross_margins: float | None = Field(default=None, description="Gross margin, as a fraction.")
    operating_margins: float | None = Field(
        default=None, description="Operating margin, as a fraction."
    )
    profit_margins: float | None = Field(
        default=None, description="Net profit margin, as a fraction."
    )
    ebitda_margins: float | None = Field(default=None, description="EBITDA margin, as a fraction.")
    debt_to_equity: float | None = Field(
        default=None, description="Debt-to-equity, as a PERCENT (79.5 = 79.5%)."
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


class RecommendationPeriod(BaseModel):
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


class AnalystData(BaseModel):
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


class NewsArticle(BaseModel):
    """A single recent news item about a symbol, as surfaced by Yahoo Finance."""

    title: str = Field(description="Headline text.")
    publisher: str | None = Field(
        default=None, description="Publisher display name, e.g. 'Yahoo Finance'; may be None."
    )
    link: str | None = Field(
        default=None, description="Canonical article URL (else a click-through URL); may be None."
    )
    published: str | None = Field(
        default=None,
        description="Publish time as an ISO8601 UTC timestamp, e.g. '2026-05-31T11:44:34Z'; "
        "may be None.",
    )
    summary: str | None = Field(
        default=None, description="Short blurb summarizing the article; may be empty or None."
    )


class NewsResult(BaseModel):
    """Recent news for a symbol, newest first."""

    symbol: str = Field(description="Ticker symbol.")
    articles: list[NewsArticle] = Field(
        description="Recent news articles, newest first; empty if no news is available."
    )


class SymbolMatch(BaseModel):
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


class SymbolSearchResult(BaseModel):
    """Search results for a query, best-match first."""

    query: str = Field(description="The search query that produced these results.")
    matches: list[SymbolMatch] = Field(
        description="Matching symbols ordered by relevance (best match first)."
    )


class PerformanceStats(BaseModel):
    """Return and risk statistics computed from daily closes over the requested window."""

    symbol: str = Field(description="Ticker symbol.")
    period: str = Field(description="Look-back window requested, e.g. '1y'.")
    bars: int = Field(description="Number of daily closes used.")
    start_date: str = Field(description="First close date (ISO 8601).")
    end_date: str = Field(description="Last close date (ISO 8601).")
    total_return_percent: float = Field(
        description="Total return over the window (e.g. 12.3 = 12.3%)."
    )
    annualized_return_percent: float | None = Field(
        default=None,
        description=(
            "Annualized return (CAGR) over the actual calendar span between start_date and "
            "end_date, percent. Over a one-year window this equals total_return_percent. "
            "Null when the span is under 90 days, because annualizing a sub-quarter move "
            "extrapolates short-run noise into a yearly figure - use total_return_percent "
            "for such windows and do not annualize it yourself."
        ),
    )
    annualized_volatility_percent: float | None = Field(
        default=None,
        description=(
            "Annualized volatility of daily returns, percent, scaled by periods_per_year. "
            "Null when the window is under 90 days, for the same reason as "
            "annualized_return_percent."
        ),
    )
    periods_per_year: float | None = Field(
        default=None,
        description=(
            "Observations per year inferred from the data and used to scale "
            "annualized_volatility_percent: roughly 252 for a weekday-traded equity and 365 "
            "for a 24/7 instrument such as crypto, and lower for one that was halted or "
            "thinly traded. Null when the window is under 90 days."
        ),
    )
    max_drawdown_percent: float = Field(
        description="Largest peak-to-trough decline, as a negative percent (e.g. -23.4 = -23.4%)."
    )
    sma_50: float | None = Field(
        default=None, description="50-day simple moving average; null if < 50 bars."
    )
    sma_200: float | None = Field(
        default=None, description="200-day simple moving average; null if < 200 bars."
    )
