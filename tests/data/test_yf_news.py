"""YFinanceClient.get_news and its search fallback."""

from typing import Any

import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.models import (
    NewsResult,
)
from finance_mcp.data.yfinance_client import (
    YFinanceClient,
)
from tests.fakes import (
    FakeClock,
    fake_search_factory,
    fake_ticker_factory,
    make_client,
    make_news_item,
    make_search_news_item,
)

NEWS_ITEMS = [
    make_news_item(
        "Apple hits record high",
        publisher="Yahoo Finance",
        link="https://finance.yahoo.com/news/a",
        published="2026-05-31T11:44:34Z",
        summary="Shares rally.",
    ),
    make_news_item(
        "Analysts upgrade Apple",
        publisher="Reuters",
        link="https://finance.yahoo.com/news/b",
        published="2026-05-30T09:00:00Z",
        summary="Upgrade to buy.",
    ),
    make_news_item(
        "Apple supplier news",
        publisher="Bloomberg",
        link="https://finance.yahoo.com/news/c",
        published="2026-05-29T08:00:00Z",
        summary="Supplier ramps output.",
    ),
]


def test_get_news_happy_path_maps_fields_newest_first() -> None:
    client = make_client(factory=fake_ticker_factory(news=NEWS_ITEMS))
    result = client.get_news("AAPL")
    assert isinstance(result, NewsResult)
    assert result.symbol == "AAPL"
    assert [a.title for a in result.articles] == [
        "Apple hits record high",
        "Analysts upgrade Apple",
        "Apple supplier news",
    ]
    first = result.articles[0]
    assert first.publisher == "Yahoo Finance"
    assert first.link == "https://finance.yahoo.com/news/a"
    assert first.published == "2026-05-31T11:44:34Z"
    assert first.summary == "Shares rally."


def test_get_news_empty_returns_empty_no_raise() -> None:
    client = make_client(factory=fake_ticker_factory(news=[]))
    result = client.get_news("ZZZZ")
    assert result.symbol == "ZZZZ" and result.articles == []


def test_get_news_falls_back_to_search_when_the_ticker_stream_is_empty() -> None:
    """An empty ticker stream is cross-checked, because Yahoo returns one for an outage.

    yfinance parses a 500 from the news endpoint into an empty list, so a server error and
    a symbol with genuinely no coverage are indistinguishable at the call site. If the
    search endpoint has articles for the symbol, the empty stream was a failure - reporting
    "no recent news" there tells the model a company had no catalysts when it did.
    """
    search = fake_search_factory(
        news=[
            make_search_news_item("Apple beats", "Reuters", "https://x/a", 1790647283),
            make_search_news_item("Apple ships", "AP", "https://x/b", 1790647000),
        ]
    )
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    result = client.get_news("AAPL")

    assert [a.title for a in result.articles] == ["Apple beats", "Apple ships"]
    assert result.source == "search", "the result must say which endpoint served it"
    assert search.calls, "the empty stream must be cross-checked against search"  # type: ignore[attr-defined]


def test_get_news_search_fallback_maps_the_flat_payload_shape() -> None:
    """The fallback's payload is flat with a unix timestamp, not the nested content shape."""
    search = fake_search_factory(
        news=[make_search_news_item("Apple beats", "Reuters", "https://x/a", 1790647283)]
    )
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    article = client.get_news("AAPL").articles[0]

    assert article.title == "Apple beats"
    assert article.publisher == "Reuters"
    assert article.link == "https://x/a"
    assert article.published == "2026-09-29T02:01:23+00:00"
    # The search endpoint carries no summary; null is honest, a fabricated one would not be.
    assert article.summary is None


def test_get_news_does_not_call_search_when_the_ticker_stream_has_news() -> None:
    """The fallback costs a request, so it must only run when the primary came back empty."""
    search = fake_search_factory(news=[make_search_news_item("should not be used")])
    client = make_client(factory=fake_ticker_factory(news=NEWS_ITEMS), search_factory=search)

    result = client.get_news("AAPL")

    assert search.calls == [], "search must not be called when the ticker stream has news"  # type: ignore[attr-defined]
    assert result.source == "ticker"
    assert all(a.title != "should not be used" for a in result.articles)


def test_get_news_empty_from_both_sources_is_still_empty() -> None:
    """A symbol with no coverage anywhere reports no news, not an error."""
    search = fake_search_factory(news=[])
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    result = client.get_news("ZZZZ")

    assert result.symbol == "ZZZZ" and result.articles == []


def test_get_news_a_failing_search_fallback_leaves_the_empty_result_intact() -> None:
    """The primary succeeded with "no news"; a broken cross-check must not make it an error."""
    search = fake_search_factory(error=YFException("search down"))
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    assert client.get_news("ZZZZ").articles == []


def test_get_news_search_fallback_caps_at_count_and_drops_untitled_items() -> None:
    search = fake_search_factory(
        news=[
            make_search_news_item(None, "Reuters", "https://x/a", 1790647283),
            make_search_news_item("kept", "AP", "https://x/b", 1790647000),
            make_search_news_item("dropped by count", "AP", "https://x/c", 1790646000),
        ]
    )
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    assert [a.title for a in client.get_news("AAPL", count=1).articles] == ["kept"]


def test_get_news_typed_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(news_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_news("AAPL")
    assert "rate limited" in str(exc.value)
    assert type(exc.value) is DataUnavailable  # never SymbolNotFound


def test_get_news_raw_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(news_error=RuntimeError("boom")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_news("AAPL")
    assert "boom" in str(exc.value)
    assert type(exc.value) is DataUnavailable


def test_get_news_skips_item_without_title() -> None:
    items = [
        NEWS_ITEMS[0],
        make_news_item(None, publisher="Reuters", link="https://x"),
        NEWS_ITEMS[1],
    ]
    client = make_client(factory=fake_ticker_factory(news=items))
    result = client.get_news("AAPL")
    assert [a.title for a in result.articles] == [
        "Apple hits record high",
        "Analysts upgrade Apple",
    ]


def test_get_news_null_nested_keys_yield_none() -> None:
    item = make_news_item(
        "Title only",
        published="2026-05-31T00:00:00Z",
        summary="",
        omit_provider=True,
        omit_canonical=True,
    )
    client = make_client(factory=fake_ticker_factory(news=[item]))
    [article] = client.get_news("AAPL").articles
    assert article.title == "Title only"
    assert article.publisher is None
    assert article.link is None
    assert article.summary is None  # "" coerced to None
    assert article.published == "2026-05-31T00:00:00Z"


def test_get_news_none_provider_and_canonical_yield_none() -> None:
    # provider/canonicalUrl present but explicitly None (a shape yfinance can return).
    item = make_news_item("Title", publisher=None, link=None)
    client = make_client(factory=fake_ticker_factory(news=[item]))
    [article] = client.get_news("AAPL").articles
    assert article.publisher is None and article.link is None


def test_get_news_click_through_fallback_link() -> None:
    item = make_news_item(
        "Title",
        published="2026-05-31T00:00:00Z",
        omit_canonical=True,
        click_through="https://fallback.example/x",
    )
    client = make_client(factory=fake_ticker_factory(news=[item]))
    [article] = client.get_news("AAPL").articles
    assert article.link == "https://fallback.example/x"


def test_get_news_clamps_to_count_and_passes_args_to_source() -> None:
    # The fake returns ALL 3 items regardless of count; the client must clamp to 2.
    factory = fake_ticker_factory(news=NEWS_ITEMS)
    client = make_client(factory=factory)
    result = client.get_news("AAPL", count=2)
    assert len(result.articles) == 2  # client-side clamp, not the fake
    assert factory.captured_news_count["count"] == 2  # type: ignore[attr-defined]
    assert factory.captured_news_count["tab"] == "news"  # type: ignore[attr-defined]


def test_get_news_parse_error_is_data_unavailable() -> None:
    # A truthy but non-string title survives the title guard yet fails NewsArticle
    # validation (title: str), forcing the parse-stage error path.
    item = {"id": "x", "content": {"title": 123}}
    client = make_client(factory=fake_ticker_factory(news=[item]))
    with pytest.raises(DataUnavailable) as exc:
        client.get_news("AAPL")
    assert "Failed to parse news for 'AAPL'" in str(exc.value)


def test_get_news_caches_within_ttl_and_expires() -> None:
    calls = {"n": 0}

    def counting(symbol: str) -> Any:
        calls["n"] += 1
        return fake_ticker_factory(news=NEWS_ITEMS)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_news("AAPL")
    client.get_news("AAPL")
    assert calls["n"] == 1  # cached within history_ttl
    clock.advance(301.0)
    client.get_news("AAPL")
    assert calls["n"] == 2  # expired after history_ttl
