"""YFinanceClient.get_news and its search fallback."""

import threading

import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.models import (
    NewsResult,
)
from tests.fakes import (
    FakeClock,
    FakeSearch,
    counting,
    fake_multi_ticker_factory,
    fake_ticker_factory,
    make_client,
    make_news_item,
    make_search_news_item,
)

APPLE_INFO = {"quoteType": "EQUITY", "longName": "Apple Inc.", "shortName": "Apple Inc."}

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


def test_get_news_falls_back_to_search_when_the_ticker_stream_is_empty() -> None:
    # yfinance parses a 500 from the news endpoint into an empty list, so an outage and a
    # symbol with no coverage look the same until search is asked.
    search = FakeSearch(
        news=[
            make_search_news_item("Apple beats", "Reuters", "https://x/a", 1790647283),
            make_search_news_item("Apple ships", "AP", "https://x/b", 1790647000),
        ]
    )
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    result = client.get_news("AAPL")

    assert [a.title for a in result.articles] == ["Apple beats", "Apple ships"]
    assert result.source == "search"


def test_get_news_search_fallback_maps_the_flat_payload_shape() -> None:
    search = FakeSearch(
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
    search = FakeSearch(news=[make_search_news_item("should not be used")])
    client = make_client(factory=fake_ticker_factory(news=NEWS_ITEMS), search_factory=search)

    result = client.get_news("AAPL")

    assert search.calls == []
    assert result.source == "ticker"
    assert all(a.title != "should not be used" for a in result.articles)


def test_get_news_empty_from_both_sources_is_still_empty() -> None:
    search = FakeSearch(news=[])
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    result = client.get_news("ZZZZ")

    assert result.symbol == "ZZZZ" and result.articles == []


def test_get_news_a_failing_search_fallback_leaves_the_empty_result_intact() -> None:
    search = FakeSearch(error=YFException("search down"))
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    assert client.get_news("ZZZZ").articles == []


def test_get_news_search_fallback_caps_at_count_and_drops_untitled_items() -> None:
    search = FakeSearch(
        news=[
            make_search_news_item(None, "Reuters", "https://x/a", 1790647283),
            make_search_news_item("kept", "AP", "https://x/b", 1790647000),
            make_search_news_item("dropped by count", "AP", "https://x/c", 1790646000),
        ]
    )
    client = make_client(factory=fake_ticker_factory(news=[]), search_factory=search)

    assert [a.title for a in client.get_news("AAPL", count=1).articles] == ["kept"]


@pytest.mark.parametrize("error", [YFException("rate limited"), RuntimeError("boom")])
def test_get_news_fetch_failure_is_data_unavailable(error: Exception) -> None:
    client = make_client(factory=fake_ticker_factory(news_error=error))
    with pytest.raises(DataUnavailable) as exc:
        client.get_news("AAPL")
    assert type(exc.value) is DataUnavailable  # never SymbolNotFound
    assert str(exc.value) == f"Failed to fetch news for 'AAPL': {error}"


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
    # The fake ignores count and returns all three items.
    factory = fake_ticker_factory(news=NEWS_ITEMS)
    client = make_client(factory=factory)
    result = client.get_news("AAPL", count=2)
    assert len(result.articles) == 2
    assert factory.captured_news_count["count"] == 2  # type: ignore[attr-defined]
    assert factory.captured_news_count["tab"] == "news"  # type: ignore[attr-defined]


def test_get_news_parse_error_is_data_unavailable() -> None:
    # A truthy non-string title passes the title check but fails NewsArticle validation.
    item = {"id": "x", "content": {"title": 123}}
    client = make_client(factory=fake_ticker_factory(news=[item]))
    with pytest.raises(DataUnavailable) as exc:
        client.get_news("AAPL")
    assert "Failed to parse news for 'AAPL'" in str(exc.value)


def test_get_news_caches_within_ttl_and_expires() -> None:
    # Each fetch is the news stream plus the company identity behind the relevance flags;
    # the identity is a fundamentals entry, so it outlives the news TTL.
    factory, calls = counting(fake_ticker_factory(news=NEWS_ITEMS, info=APPLE_INFO))
    clock = FakeClock()
    client = make_client(factory, clock=clock, history_ttl=300.0, fundamentals_ttl=3600.0)
    client.get_news("AAPL")
    client.get_news("AAPL")
    assert len(calls) == 2
    clock.advance(301.0)
    client.get_news("AAPL")
    assert len(calls) == 3


# --- mentions_company --------------------------------------------------------------

MIXED_NEWS = [
    make_news_item("Apple unveils a new iPhone", summary="The launch event ran long."),
    make_news_item("S&P 500 dips after inflation data", summary="Stocks slipped broadly."),
    make_news_item("Chipmakers rally", summary="AAPL suppliers led the gains."),
]


def _flags(result: NewsResult) -> list[bool | None]:
    return [a.mentions_company for a in result.articles]


def test_get_news_flags_which_articles_name_the_company() -> None:
    client = make_client(fake_ticker_factory(news=MIXED_NEWS, info=APPLE_INFO))
    result = client.get_news("AAPL")
    # By name in the title; by neither; by ticker in the summary.
    assert _flags(result) == [True, False, True]
    assert result.relevance_check == "applied"


def test_get_news_keeps_every_article_in_order_whatever_its_flag() -> None:
    client = make_client(fake_ticker_factory(news=MIXED_NEWS, info=APPLE_INFO))
    titles = [a.title for a in client.get_news("AAPL").articles]
    assert titles == [item["content"]["title"] for item in MIXED_NEWS]


def test_get_news_matches_the_ticker_root_of_a_share_class() -> None:
    info = {"quoteType": "EQUITY", "longName": "Berkshire Hathaway Inc."}
    news = [make_news_item("BRK.B slips after the annual letter"), make_news_item("Fed holds")]
    client = make_client(fake_ticker_factory(news=news, info=info))
    assert _flags(client.get_news("BRK-B")) == [True, False]


@pytest.mark.parametrize("quote_type", ["ETF", "INDEX", "CRYPTOCURRENCY", "CURRENCY"])
def test_get_news_does_not_assess_a_non_equity(quote_type: str) -> None:
    # Market-wide news IS relevant to an index, ETF, coin or currency pair.
    info = {"quoteType": quote_type, "longName": "SPDR S&P 500 ETF Trust"}
    client = make_client(fake_ticker_factory(news=MIXED_NEWS, info=info))
    result = client.get_news("SPY")
    assert _flags(result) == [None, None, None]
    assert result.relevance_check == "not_an_equity"


def test_get_news_when_the_name_cannot_be_fetched_still_returns_the_news() -> None:
    client = make_client(
        fake_ticker_factory(news=MIXED_NEWS, info_error=YFException("info endpoint down"))
    )
    result = client.get_news("AAPL")
    assert len(result.articles) == 3
    assert _flags(result) == [None, None, None]
    assert result.relevance_check == "unavailable"
    assert result.relevance_note is not None and "info endpoint down" in result.relevance_note


def test_get_news_for_a_symbol_yahoo_has_no_name_for() -> None:
    # Yahoo's empty info is a lasting answer about the symbol, not an outage.
    client = make_client(fake_ticker_factory(news=MIXED_NEWS, info={}))
    result = client.get_news("AAPL")
    assert len(result.articles) == 3
    assert _flags(result) == [None, None, None]
    assert result.relevance_check == "no_company_name"
    assert result.relevance_note is not None and "AAPL" in result.relevance_note


def test_get_news_caches_a_symbol_with_no_name_like_any_other() -> None:
    # An unknown symbol: no stream, no search hits, no name. All three answers are lasting,
    # so a repeat call must not go back to Yahoo three times.
    factory, calls = counting(fake_ticker_factory(news=[], info={}))
    search = FakeSearch(news=[])
    client = make_client(factory, search_factory=search)
    client.get_news("ZZZZ")
    client.get_news("ZZZZ")
    assert len(calls) == 2  # news + identity, once
    assert len(search.calls) == 1


def test_get_news_flags_are_assessed_with_no_note() -> None:
    client = make_client(fake_ticker_factory(news=MIXED_NEWS, info=APPLE_INFO))
    assert client.get_news("AAPL").relevance_note is None


def test_get_news_does_not_cache_a_result_whose_flags_could_not_be_assessed() -> None:
    # A transient info failure must not pin unflagged news for the whole TTL.
    factory, calls = counting(
        fake_ticker_factory(news=MIXED_NEWS, info_error=YFException("info endpoint down"))
    )
    client = make_client(factory)
    client.get_news("AAPL")
    client.get_news("AAPL")
    assert len(calls) == 4  # news + identity, twice


def test_get_news_flags_search_fallback_articles_too() -> None:
    search = FakeSearch(
        news=[make_search_news_item("Apple beats"), make_search_news_item("Oil climbs")]
    )
    client = make_client(fake_ticker_factory(news=[], info=APPLE_INFO), search_factory=search)
    result = client.get_news("AAPL")
    assert result.source == "search"
    assert _flags(result) == [True, False]


def test_get_news_fetches_the_stream_and_the_identity_concurrently() -> None:
    # The barrier only opens when both ticker lookups are in flight at once.
    gate = threading.Barrier(2, timeout=10)
    factory = fake_multi_ticker_factory(
        {"AAPL": {"news": MIXED_NEWS, "info": APPLE_INFO}}, gate=gate
    )
    assert _flags(make_client(factory).get_news("AAPL")) == [True, False, True]


def test_get_news_reuses_the_identity_across_counts() -> None:
    factory, calls = counting(fake_ticker_factory(news=MIXED_NEWS, info=APPLE_INFO))
    client = make_client(factory)
    client.get_news("AAPL", count=3)
    client.get_news("AAPL", count=2)
    assert len(calls) == 3  # identity once, the stream once per count
