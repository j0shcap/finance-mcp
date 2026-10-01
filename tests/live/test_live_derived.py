"""Live contract: analyze_performance, get_news, search_symbols.

analyze_performance is ours, computed from live bars, so its assertions check our analytics
against real market data. get_news and search_symbols read payload shapes yfinance has
restructured before.
"""

import datetime

import pytest

from tests.live.conftest import AAPL, BTC, SPY, UNKNOWN, Layer, require_present

ANNUALIZED_FIELDS = (
    "annualized_return_percent",
    "annualized_volatility_percent",
    "periods_per_year",
)


async def test_analyze_performance_equity_shape_and_units(layer: Layer) -> None:
    stats = await layer.call("analyze_performance", ticker=AAPL, period="1y")

    assert stats.symbol == AAPL
    assert stats.period == "1y"
    assert stats.bars > 200, f"a one-year daily window should hold ~252 bars, got {stats.bars}"
    datetime.date.fromisoformat(stats.start_date)
    datetime.date.fromisoformat(stats.end_date)
    assert stats.start_date < stats.end_date

    require_present(stats, (*ANNUALIZED_FIELDS, "sma_50", "sma_200"))

    # Annualization runs off the calendar span, so over a one-year window the annualized
    # return equals the total return - the invariant the tool description states.
    assert stats.annualized_return_percent == pytest.approx(stats.total_return_percent, rel=0.05), (
        f"over a 1y window annualized ({stats.annualized_return_percent}) must equal total "
        f"({stats.total_return_percent}); a bar-count-based factor would diverge here"
    )

    # ~252 weekday sessions a year for an exchange-traded equity.
    assert stats.periods_per_year is not None
    assert 220 < stats.periods_per_year < 275, (
        f"periods_per_year {stats.periods_per_year} is not a weekday trading calendar"
    )

    assert stats.annualized_volatility_percent is not None
    assert 0 < stats.annualized_volatility_percent < 200, (
        f"annualized volatility {stats.annualized_volatility_percent} is not a plausible percent"
    )
    assert stats.max_drawdown_percent <= 0, (
        f"max_drawdown_percent must be negative or zero, got {stats.max_drawdown_percent}"
    )
    assert stats.max_drawdown_percent > -100

    assert stats.sma_50 is not None and stats.sma_50 > 0
    assert stats.sma_200 is not None and stats.sma_200 > 0
    # A moving average of the same series cannot be an order of magnitude away from it.
    assert 0.3 < stats.sma_50 / stats.sma_200 < 3


async def test_analyze_performance_crypto_infers_a_247_calendar(layer: Layer) -> None:
    """A 24/7 instrument yields ~365 observations a year, not ~252.

    periods_per_year is inferred from the data, and the prompt has the model cite it when
    comparing volatility across asset classes, so the two calendars must come out different.
    """
    stats = await layer.call("analyze_performance", ticker=BTC, period="1y")

    require_present(stats, ANNUALIZED_FIELDS)
    assert stats.periods_per_year is not None
    assert 330 < stats.periods_per_year < 375, (
        f"BTC-USD trades every day, so periods_per_year should be ~365, got "
        f"{stats.periods_per_year}"
    )
    assert stats.annualized_return_percent == pytest.approx(stats.total_return_percent, rel=0.05)


async def test_analyze_performance_suppresses_annualization_on_short_windows(
    layer: Layer,
) -> None:
    """Under ~90 days the annualized figures are null rather than extrapolated noise."""
    stats = await layer.call("analyze_performance", ticker=AAPL, period="1mo")

    for field in ANNUALIZED_FIELDS:
        assert getattr(stats, field) is None, (
            f"{field} must be null on a sub-quarter window - annualizing a one-month move "
            f"reports short-run noise as a yearly rate"
        )
    # The unannualized figures are still reported: they are what the model should quote.
    assert stats.bars > 5
    assert stats.max_drawdown_percent <= 0


async def test_analyze_performance_defaults_to_the_treasury_bill_rate(layer: Layer) -> None:
    """^IRX is a percent discount quote; the default must come back as a plausible decimal.

    A unit slip (percent read as a decimal) would give ~4.0 instead of ~0.04.
    """
    stats = await layer.call("analyze_performance", ticker=AAPL, period="1y")

    assert stats.risk_free_rate_source == "treasury_bill", stats.risk_free_rate_note
    assert stats.risk_free_rate is not None
    assert 0 <= stats.risk_free_rate < 0.2, (
        f"a one-year T-bill average of {stats.risk_free_rate} is not a plausible annual decimal"
    )
    require_present(stats, ("sharpe_ratio", "downside_deviation_percent"))


async def test_a_window_older_than_the_treasury_bill_history_has_no_default_rate(
    layer: Layer,
) -> None:
    """^GSPC's max history starts in 1927, ^IRX's in 1960: the rate is unavailable, not 0."""
    stats = await layer.call("analyze_performance", ticker="^GSPC", period="max")

    assert stats.start_date < "1960-01-01", f"^GSPC max now starts {stats.start_date}"
    assert stats.risk_free_rate is None
    assert stats.risk_free_rate_source == "unavailable"
    assert stats.risk_free_rate_note is not None and "^IRX" in stats.risk_free_rate_note
    assert stats.sharpe_ratio is None
    require_present(stats, ("annualized_return_percent", "calmar_ratio"))


async def test_news_flags_company_mentions_for_a_stock_only(layer: Layer) -> None:
    stock = await layer.call("get_news", ticker=AAPL, count=10)
    fund = await layer.call("get_news", ticker=SPY, count=5)

    assert stock.relevance_check == "applied"
    assert all(isinstance(a.mentions_company, bool) for a in stock.articles)
    assert any(a.mentions_company for a in stock.articles), (
        "no AAPL article names Apple or AAPL; the name extraction or matching has broken"
    )
    assert fund.relevance_check == "not_an_equity"
    assert all(a.mentions_company is None for a in fund.articles)


async def test_news_shape(layer: Layer) -> None:
    result = await layer.call("get_news", ticker=AAPL, count=5)

    assert result.symbol == AAPL
    assert result.articles, "AAPL always has recent news"
    assert len(result.articles) <= 5, "count must cap the article list"

    for article in result.articles:
        assert article.title and article.title.strip(), "every article needs a headline"

    # These come from nested keys (content.provider.displayName, content.canonicalUrl.url,
    # content.summary) that yfinance has restructured before. One article may be missing
    # one; a payload change nulls them across every article at once.
    #
    # summary only when the per-symbol stream served the result: the search fallback has no
    # summary field at all, so asserting one there would fail on a working result. Asserting
    # it is null on that path keeps the check meaningful rather than merely skipped.
    expected = ("publisher", "link", "summary") if result.source == "ticker" else ("publisher",)
    for field in expected:
        assert any(getattr(a, field) for a in result.articles), (
            f"no article has a {field}; the nested news payload shape has most likely changed"
        )
    if result.source == "search":
        assert all(a.summary is None for a in result.articles), (
            "the search fallback carries no summary; a value here means the payload gained one"
        )

    for article in result.articles:
        if article.link is not None:
            assert article.link.startswith("http"), f"not a URL: {article.link}"
        if article.published is not None:
            published = datetime.datetime.fromisoformat(article.published)
            assert published.tzinfo is not None, (
                f"published must be an ISO8601 UTC timestamp, got {article.published}"
            )
            age = datetime.datetime.now(tz=datetime.UTC) - published
            assert datetime.timedelta(days=-2) < age < datetime.timedelta(days=365), (
                f"'recent news' should not be {age.days} days old: {article.published}"
            )


async def test_news_for_an_unknown_symbol_is_empty_not_an_error(layer: Layer) -> None:
    """The tool description promises an empty list, not an error, for an unknown symbol."""
    result = await layer.call("get_news", ticker=UNKNOWN, count=5)

    assert result.articles == []


async def test_search_resolves_a_company_name(layer: Layer) -> None:
    result = await layer.call("search_symbols", query="Apple", max_results=8)

    assert result.query == "Apple"
    assert result.matches, "'Apple' must resolve to something"
    assert len(result.matches) <= 8, "max_results must cap the match list"

    by_symbol = {m.symbol: m for m in result.matches}
    # This is how the model turns a name into a ticker before every other call, so the
    # canonical name resolving to the canonical symbol is the whole contract.
    assert AAPL in by_symbol, f"'Apple' must resolve to AAPL, got {sorted(by_symbol)}"

    apple = by_symbol[AAPL]
    require_present(apple, ("name", "quote_type", "exchange", "score"))
    assert apple.quote_type == "EQUITY"
    assert apple.score is not None and apple.score > 0

    scores = [m.score for m in result.matches if m.score is not None]
    assert scores == sorted(scores, reverse=True), (
        f"matches must be ordered best-first, got scores {scores}"
    )


async def test_search_returns_non_equity_types(layer: Layer) -> None:
    """quote_type is how the model tells a coin or an ETF from a stock, so it must vary."""
    result = await layer.call("search_symbols", query="Bitcoin", max_results=8)

    assert result.matches
    types = {m.quote_type for m in result.matches}
    assert "CRYPTOCURRENCY" in types, (
        f"a search for Bitcoin should surface a cryptocurrency, got types {types}"
    )


async def test_search_for_nonsense_is_empty_not_an_error(layer: Layer) -> None:
    """An unmatched query returns an empty match list, as the tool description promises."""
    result = await layer.call("search_symbols", query="zzzqqqxxnotacompany", max_results=8)

    assert result.matches == []
