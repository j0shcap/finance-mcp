"""Live contract: get_financials, get_company_profile, get_key_metrics, get_analyst_data.

These are the tools the conventions glossary makes unit promises about, so most of the
unit assertions live here.
"""

import datetime

import pandas as pd
import pytest
import yfinance as yf

from finance_mcp.data.providers.yahoo import corporate_actions
from tests.live.conftest import (
    AAPL,
    SAP,
    SPY,
    Layer,
    iso_dates_descending,
    require_present,
    retry_throttled,
)

#: Fields Yahoo populates for any large listing. peg_ratio is included deliberately: the
#: client reads info["pegRatio"] and Yahoo has shipped a "trailingPegRatio" spelling, which
#: makes it the likeliest key to drift out from under us.
METRICS_FIELDS = (
    "currency",
    "financial_currency",
    "trailing_pe",
    "forward_pe",
    "price_to_book",
    "price_to_sales",
    "peg_ratio",
    "enterprise_value",
    "ev_to_ebitda",
    "ev_to_revenue",
    "return_on_equity",
    "return_on_assets",
    "gross_margins",
    "operating_margins",
    "profit_margins",
    "ebitda_margins",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "total_debt",
    "total_cash",
    "free_cashflow",
    "ebitda",
    "trailing_eps",
    "forward_eps",
    "revenue_per_share",
    "book_value",
)

#: Ratios and margins reported as FRACTIONS (0.27 = 27%), per the conventions glossary.
FRACTION_FIELDS = (
    "return_on_equity",
    "return_on_assets",
    "gross_margins",
    "operating_margins",
    "profit_margins",
    "ebitda_margins",
)

PROFILE_FIELDS = (
    "name",
    "sector",
    "industry",
    "country",
    "website",
    "employees",
    "summary",
    "currency",
    "market_cap",
    "trailing_pe",
    "forward_pe",
    "dividend_yield",
    "beta",
)


async def test_financials_income_annual_shape(layer: Layer) -> None:
    statement = await layer.call("get_financials", ticker=AAPL, statement="income", period="annual")

    assert statement.symbol == AAPL
    assert statement.statement == "income"
    assert statement.period == "annual"
    require_present(statement, ("currency",))
    assert statement.currency == "USD"

    assert statement.period_ends, "an annual income statement must report period ends"
    for end in statement.period_ends:
        datetime.date.fromisoformat(end)
    assert iso_dates_descending(statement.period_ends), (
        f"period_ends must be strictly most-recent-first, got {statement.period_ends}"
    )

    # The analyze_stock prompt relies on both labels by name.
    for label in ("Total Revenue", "Net Income"):
        assert label in statement.line_items, (
            f"{label!r} is missing; available labels: {sorted(statement.line_items)}"
        )
        values = statement.line_items[label]
        assert len(values) == len(statement.period_ends), (
            f"{label} has {len(values)} values for {len(statement.period_ends)} periods"
        )

    revenue = statement.line_items["Total Revenue"][0]
    # Absolute units, not millions: 416161000000, not 416161. A scale change would shrink
    # every figure the model reports by six orders of magnitude.
    assert revenue is not None and revenue > 1e11, (
        f"AAPL annual revenue should be hundreds of billions in absolute units, got {revenue}"
    )

    assert statement.missing_line_items == []
    assert statement.available_line_items == [], (
        "available_line_items stays empty when nothing was missing; otherwise it would just "
        "repeat line_items' keys"
    )


async def test_financials_quarterly_shape(layer: Layer) -> None:
    statement = await layer.call(
        "get_financials", ticker=AAPL, statement="balance", period="quarterly"
    )

    assert statement.period == "quarterly"
    assert len(statement.period_ends) >= 2
    assert iso_dates_descending(statement.period_ends)
    assert statement.line_items, "a quarterly balance sheet must report line items"
    # ~3 months apart is what distinguishes a quarterly statement from an annual one.
    newest = datetime.date.fromisoformat(statement.period_ends[0])
    second = datetime.date.fromisoformat(statement.period_ends[1])
    assert 60 <= (newest - second).days <= 125, (
        f"consecutive quarterly period ends should be ~one quarter apart, got "
        f"{statement.period_ends[0]} and {statement.period_ends[1]}"
    )


async def test_financials_missing_label_reports_suggestions(layer: Layer) -> None:
    """A label that does not exist comes back with the real labels to retry from."""
    statement = await layer.call(
        "get_financials",
        ticker=AAPL,
        statement="income",
        period="annual",
        line_items=["Revenue"],
    )

    assert statement.missing_line_items == ["Revenue"]
    assert statement.available_line_items, (
        "available_line_items must be populated when a requested label was missing"
    )
    assert "Total Revenue" in statement.available_line_items
    # The suggestion turns a wrong guess into a working retry.
    assert "Total Revenue" in statement.line_item_suggestions.get("Revenue", []), (
        f"expected 'Revenue' to suggest 'Total Revenue', got {statement.line_item_suggestions}"
    )


async def test_financials_non_usd_reporter_is_labelled(layer: Layer) -> None:
    """SAP reports in EUR while its US listing quotes in USD.

    FinancialStatement.currency names this case explicitly. A flip to USD would have the
    model comparing EUR revenue against USD peers without noticing.
    """
    statement = await layer.call("get_financials", ticker=SAP, statement="income", period="annual")

    require_present(statement, ("currency",))
    assert statement.currency == "EUR", (
        f"SAP reports its financials in EUR, got {statement.currency!r}"
    )
    assert statement.line_items


async def test_key_metrics_cross_listing_currencies_differ(layer: Layer) -> None:
    """The two currency fields are not interchangeable; SAP is where that shows."""
    metrics = await layer.call("get_key_metrics", ticker=SAP)

    require_present(metrics, ("currency", "financial_currency"))
    assert metrics.currency == "USD", "the SAP ADR quotes in USD"
    assert metrics.financial_currency == "EUR", "SAP reports its financials in EUR"
    assert metrics.currency != metrics.financial_currency


async def test_company_profile_shape(layer: Layer) -> None:
    profile = await layer.call("get_company_profile", ticker=AAPL)

    assert profile.symbol == AAPL
    require_present(profile, PROFILE_FIELDS)

    assert profile.currency == "USD"
    assert profile.market_cap is not None and profile.market_cap > 1e11
    assert profile.trailing_pe is not None and profile.trailing_pe > 0
    assert profile.employees is not None and profile.employees > 1000
    assert profile.beta is not None and 0 < profile.beta < 4
    assert profile.website is not None and profile.website.startswith("http")

    assert profile.recent_dividends, "AAPL pays a dividend, so recent_dividends must be populated"
    dividend_dates = [d.date for d in profile.recent_dividends]
    # Newest LAST - the opposite order from period_ends. Backwards, "the latest dividend"
    # would be the oldest one.
    assert dividend_dates == sorted(dividend_dates), (
        f"recent_dividends must be oldest-first (newest last), got {dividend_dates}"
    )
    for dividend in profile.recent_dividends:
        datetime.date.fromisoformat(dividend.date)
        assert dividend.amount > 0

    assert profile.splits, "AAPL has split before, so splits must be populated"
    for split in profile.splits:
        datetime.date.fromisoformat(split.date)
        assert split.ratio > 0


def test_profile_events_from_weekly_history_match_the_public_daily_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """corporate_actions' weekly-history shortcut returns exactly what .dividends/.splits do.

    KO's events reach back before 1970, so a source that drops old events can't pass. The
    private cache is asserted present: without it corporate_actions quietly takes the slow
    public path, and this test is where that shows up.
    """
    monkeypatch.setattr(yf.config.debug, "hide_exceptions", False)  # as YahooProvider sets it
    ticker = yf.Ticker("KO")

    def read() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        assert hasattr(ticker._lazy_load_price_history(), "_get_history_cache")
        return (*corporate_actions(ticker), ticker.dividends, ticker.splits)

    dividends, splits, daily_dividends, daily_splits = retry_throttled(read, "the KO history")
    assert dividends.index.min().year < 1970 and splits.index.min().year < 1970
    assert dividends.equals(daily_dividends) and splits.equals(daily_splits)


async def test_company_profile_dividend_yield_is_a_percent(layer: Layer) -> None:
    """dividend_yield is a percent (5.92 = 5.92%), cross-checked against the dividends.

    A range assertion cannot catch this flip: for any yield y, both y and y/100 fall inside
    a plausible 0-25 band. Recomputing the trailing yield from the dividends and price is
    the only real check. The factor-of-5 tolerance covers the differing trailing windows; a
    units flip is a factor of 100.
    """
    profile = await layer.call("get_company_profile", ticker=AAPL)
    quote = (await layer.call("get_quote", tickers=[AAPL])).quotes[0]

    require_present(profile, ("dividend_yield",))
    assert profile.dividend_yield is not None
    assert 0 <= profile.dividend_yield < 25, (
        f"a dividend yield of {profile.dividend_yield} is not a plausible percent"
    )

    cutoff = datetime.date.today() - datetime.timedelta(days=370)
    trailing = sum(
        d.amount for d in profile.recent_dividends if datetime.date.fromisoformat(d.date) >= cutoff
    )
    assert trailing > 0, "expected at least one AAPL dividend in the trailing year"
    computed_percent = trailing / quote.price * 100.0

    ratio = profile.dividend_yield / computed_percent
    assert 0.2 < ratio < 5, (
        f"dividend_yield {profile.dividend_yield} does not agree with the trailing "
        f"dividends/price yield of {computed_percent:.4f}% (ratio {ratio:.4f}). A ratio near "
        f"0.01 means Yahoo switched this field to a fraction and the docs, the conventions "
        f"glossary and the analyze_stock prompt are now all 100x wrong."
    )


async def test_key_metrics_shape_and_units(layer: Layer) -> None:
    metrics = await layer.call("get_key_metrics", ticker=AAPL)

    assert metrics.symbol == AAPL
    require_present(metrics, METRICS_FIELDS)

    assert metrics.currency == "USD"
    assert metrics.financial_currency == "USD"

    for field in FRACTION_FIELDS:
        value = getattr(metrics, field)
        assert -5 < value < 5, (
            f"{field} is {value}, which is not a fraction - a value near {value / 100:.4f} "
            f"would be. Yahoo switching this to a percent makes every margin the model "
            f"reports 100x too large."
        )

    # debt_to_equity is ALREADY a percent (79.5 = 79.5%, not 79.5x). The >5 bound separates
    # a percent from a silently-converted ratio; AAPL's sits well above it.
    assert metrics.debt_to_equity is not None
    assert 5 < metrics.debt_to_equity < 2000, (
        f"debt_to_equity is {metrics.debt_to_equity}; as a percent it should be well above "
        f"5, and a value near {metrics.debt_to_equity * 100} would mean a fraction"
    )

    for field in ("trailing_pe", "forward_pe", "price_to_book", "price_to_sales", "ev_to_ebitda"):
        value = getattr(metrics, field)
        assert 0 < value < 500, f"{field} is {value}, outside any plausible multiple"

    for field in ("enterprise_value", "total_debt", "total_cash", "free_cashflow", "ebitda"):
        value = getattr(metrics, field)
        assert value > 1e9, f"{field} is {value}, too small to be absolute units for AAPL"

    for field in ("trailing_eps", "forward_eps", "revenue_per_share", "book_value"):
        value = getattr(metrics, field)
        assert 0 < value < 1000, f"{field} is {value}, not a plausible per-share figure"

    # EBITDA and its margin must agree in sign, or one of them is not what it claims.
    assert (metrics.ebitda > 0) == (metrics.ebitda_margins > 0)


async def test_analyst_data_shape_and_units(layer: Layer) -> None:
    analyst = await layer.call("get_analyst_data", ticker=AAPL)

    assert analyst.symbol == AAPL
    require_present(
        analyst,
        (
            "currency",
            "current_price",
            "recommendation_key",
            "recommendation_mean",
            "number_of_analysts",
            "target_mean_price",
            "target_median_price",
            "target_high_price",
            "target_low_price",
        ),
    )

    assert analyst.currency == "USD"
    assert analyst.current_price is not None and analyst.current_price > 0

    # The 1-5 scale is INVERTED (1 = strong buy), per the tool description and glossary.
    assert analyst.recommendation_mean is not None
    assert 1.0 <= analyst.recommendation_mean <= 5.0, (
        f"recommendation_mean {analyst.recommendation_mean} is off the documented 1-5 scale"
    )
    assert analyst.recommendation_key is not None
    assert analyst.recommendation_key == analyst.recommendation_key.lower()

    assert analyst.number_of_analysts is not None and analyst.number_of_analysts >= 1

    low, mean, median, high = (
        analyst.target_low_price,
        analyst.target_mean_price,
        analyst.target_median_price,
        analyst.target_high_price,
    )
    assert low is not None and mean is not None and median is not None and high is not None
    assert 0 < low <= mean <= high, f"targets out of order: low={low} mean={mean} high={high}"
    assert low <= median <= high, f"median {median} outside [{low}, {high}]"

    assert analyst.recommendation_trend, "AAPL is widely covered, so the trend must be populated"
    periods = [p.period for p in analyst.recommendation_trend]
    assert set(periods) <= {"0m", "-1m", "-2m", "-3m"}, (
        f"recommendation_trend periods must be month offsets from the current month, got {periods}"
    )
    assert len(set(periods)) == len(periods), f"duplicate trend periods: {periods}"
    for period in analyst.recommendation_trend:
        counts = (
            period.strong_buy,
            period.buy,
            period.hold,
            period.sell,
            period.strong_sell,
        )
        assert all(c >= 0 for c in counts), f"negative analyst counts in {period.period}: {counts}"
        assert sum(counts) > 0, f"no analyst counts at all for {period.period}"


async def test_analyst_data_on_an_etf_is_a_clear_error(layer: Layer) -> None:
    """An ETF has no sell-side coverage, and the error has to say so.

    The tool description and the analyze_stock prompt both promise this. If Yahoo starts
    publishing ETF targets, this goes red and one of the three has to change.
    """
    async with layer.expect_error("No analyst coverage"):
        await layer.call("get_analyst_data", ticker=SPY)
