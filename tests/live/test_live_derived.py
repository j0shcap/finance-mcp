"""Live contract: analyze_performance, get_news, search_symbols.

analyze_performance is computed by us from live bars, so its assertions check our
analytics against real market data; get_news and search_symbols read payload shapes
yfinance has changed before (the nested news ``content``, ``yf.Search``).
"""

import datetime

import pytest

from tests.live.conftest import AAPL, BTC, UNKNOWN, Layer, require_present

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

    # Annualization runs off the calendar span, so on a one-year window the annualized
    # return equals the total return. This is the invariant the tool description states and
    # the prompt repeats; asserting it on live data is the point of this test.
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

    periods_per_year is inferred from the data rather than hardcoded, and the prompt tells
    the model to state it when comparing volatility across asset classes - so the two
    calendars have to actually come out different.
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
    # The unannualized figures are still reported, since they are what the model should quote.
    assert stats.bars > 5
    assert stats.max_drawdown_percent <= 0


async def test_news_shape(layer: Layer) -> None:
    result = await layer.call("get_news", ticker=AAPL, count=5)

    assert result.symbol == AAPL
    assert result.articles, "AAPL always has recent news"
    assert len(result.articles) <= 5, "count must cap the article list"

    for article in result.articles:
        assert article.title and article.title.strip(), "every article needs a headline"

    # These three come from nested keys (content.provider.displayName,
    # content.canonicalUrl.url, content.summary) that yfinance has restructured before. Any
    # single article may be missing one, but a payload change nulls them all at once.
    for field in ("publisher", "link", "summary"):
        assert any(getattr(a, field) for a in result.articles), (
            f"no article has a {field}; the nested news payload shape has most likely changed"
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
    # search_symbols is how the model turns a name into a ticker before every other call,
    # so the canonical name resolving to the canonical symbol is the whole contract.
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
    """quote_type is how the model tells an ETF or a coin from a stock, so it must vary."""
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
