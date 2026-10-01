"""Symbol normalization, which happens before any cache lookup or fetch."""

import pytest

from finance_mcp.data.errors import SymbolNotFound
from tests.fakes import (
    INCOME_WITH_NAN,
    QUOTE_FI,
    counting,
    fake_ticker_factory,
    make_client,
    make_financials_df,
    make_history_df,
    make_news_item,
)


def test_symbols_are_normalized_before_caching_and_echoed_normalized() -> None:
    factory, calls = counting(fake_ticker_factory(fast_info=QUOTE_FI))
    client = make_client(factory)
    lower = client.get_quote(["aapl"]).quotes
    padded = client.get_quote([" AAPL "]).quotes
    assert calls == ["AAPL"]  # one fetch, with the normalized symbol
    assert [q.symbol for q in lower] == ["AAPL"]
    assert [q.symbol for q in padded] == ["AAPL"]


def test_price_history_normalizes_symbol() -> None:
    factory, calls = counting(fake_ticker_factory(history_df=make_history_df([100.0, 101.0])))
    client = make_client(factory)
    first = client.get_price_history(" aapl", period="1mo", interval="1d")
    client.get_price_history("AAPL", period="1mo", interval="1d")
    assert calls == ["AAPL"]
    assert first.symbol == "AAPL"


def test_profile_metrics_analyst_news_and_performance_normalize_symbol() -> None:
    info = {"longName": "Apple Inc.", "currency": "USD", "targetMeanPrice": 250.0}
    df = make_history_df([100.0, 101.0, 102.0])
    factory = fake_ticker_factory(
        info=info,
        history_df=df,
        news=[make_news_item("Hi")],
        financials={
            "income_stmt": make_financials_df(INCOME_WITH_NAN, ["2024-09-30", "2023-09-30"])
        },
    )
    client = make_client(factory=factory)
    assert client.get_company_profile(" aapl ").symbol == "AAPL"
    assert client.get_key_metrics(" aapl ").symbol == "AAPL"
    assert client.get_analyst_data(" aapl ").symbol == "AAPL"
    assert client.get_news(" aapl ").symbol == "AAPL"
    assert client.analyze_performance(" aapl ", "1y").symbol == "AAPL"
    assert client.get_financials(" aapl ", "income", "annual").symbol == "AAPL"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_symbol_raises_symbol_not_found_without_fetching(blank: str) -> None:
    factory, calls = counting(fake_ticker_factory(fast_info=QUOTE_FI))
    client = make_client(factory)
    with pytest.raises(SymbolNotFound, match="Empty ticker symbol"):
        client.get_company_profile(blank)
    assert calls == []
