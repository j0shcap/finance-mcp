import math
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, Literal, get_args

import pandas as pd
import pytest
from yfinance.exceptions import (
    YFException,
    YFPricesMissingError,
    YFRateLimitError,
    YFTickerMissingError,
    YFTzMissingError,
)

from finance_mcp.data import analytics
from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    DividendEvent,
    FinancialStatement,
    HistoryInterval,
    KeyMetrics,
    NewsResult,
    PerformanceStats,
    PriceBar,
    PriceHistory,
    PriceSummary,
    Quote,
    QuoteResult,
    SplitEvent,
    SymbolSearchResult,
)
from finance_mcp.data.yfinance_client import (
    _INTRADAY_INTERVALS,
    DEFAULT_CACHE_MAX_ENTRIES,
    QUOTE_MAX_WORKERS,
    YFinanceClient,
    _recommendation_trend,
)
from tests.conftest import (
    FakeClock,
    fake_search_factory,
    fake_symbol_ticker_factory,
    fake_ticker_factory,
    make_client,
    make_financials_df,
    make_history_df,
    make_intraday_df,
    make_news_item,
    make_recommendations_df,
    make_series,
)


def test_models_and_errors_exist() -> None:
    q = Quote(
        symbol="AAPL",
        currency="USD",
        price=190.0,
        previous_close=188.0,
        change=2.0,
        change_percent=1.06,
        day_high=191.0,
        day_low=187.0,
        year_high=200.0,
        year_low=150.0,
        market_cap=3.0e12,
        volume=50_000_000,
    )
    assert q.symbol == "AAPL"
    bar = PriceBar(date="2024-01-02", open=1.0, high=2.0, low=0.5, close=1.5, volume=100)
    summary = PriceSummary(
        start_date="2024-01-02",
        end_date="2024-01-03",
        start_close=1.5,
        end_close=1.6,
        total_return_percent=6.67,
        period_high=2.0,
        period_low=0.5,
        bars=2,
    )
    hist = PriceHistory(
        symbol="AAPL", period="1mo", interval="1d", bars=[bar], summary=summary, truncated=False
    )
    assert hist.summary.bars == 2
    assert issubclass(SymbolNotFound, DataUnavailable)
    assert str(DataUnavailable("boom")) == "boom"


def test_fundamentals_and_profile_models() -> None:
    stmt = FinancialStatement(
        symbol="AAPL",
        statement="income",
        period="annual",
        period_ends=["2024-09-30", "2023-09-30"],
        line_items={"Total Revenue": [391_035.0, None]},
    )
    assert stmt.statement == "income"
    assert stmt.period == "annual"
    assert stmt.period_ends[0] == "2024-09-30"
    assert stmt.line_items["Total Revenue"] == [391_035.0, None]

    profile = CompanyProfile(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        market_cap=3.0e12,
        recent_dividends=[DividendEvent(date="2024-08-12", amount=0.25)],
        splits=[SplitEvent(date="2020-08-31", ratio=4.0)],
    )
    assert profile.symbol == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.recent_dividends[0].amount == 0.25
    assert profile.splits[0].ratio == 4.0

    empty = CompanyProfile(symbol="MSFT")
    assert empty.recent_dividends == []
    assert empty.splits == []
    assert empty.name is None


QUOTE_FI = {
    "last_price": 190.0,
    "previous_close": 188.0,
    "day_high": 191.0,
    "day_low": 187.0,
    "year_high": 200.0,
    "year_low": 150.0,
    "market_cap": 3.0e12,
    "currency": "USD",
    "last_volume": 50_000_000,
}


def _client(**kw: Any) -> YFinanceClient:
    clock = kw.pop("clock", FakeClock())
    factory = kw.pop("factory", fake_ticker_factory(fast_info=QUOTE_FI))
    return YFinanceClient(ticker_factory=factory, time_fn=clock, quote_ttl=30.0, history_ttl=300.0)


def test_get_quote_parses_and_computes_change() -> None:
    [q] = _client().get_quote(["AAPL"]).quotes
    assert q.symbol == "AAPL"
    assert q.price == 190.0
    assert q.change == pytest.approx(2.0)
    assert q.change_percent == pytest.approx(2.0 / 188.0 * 100.0)
    assert q.currency == "USD"


def test_get_quote_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting_factory(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting_factory, time_fn=clock, quote_ttl=30.0, history_ttl=300.0
    )
    client.get_quote(["AAPL"])
    client.get_quote(["AAPL"])
    assert calls["n"] == 1
    clock.advance(31.0)
    client.get_quote(["AAPL"])
    assert calls["n"] == 2


def test_get_quote_cache_expires_exactly_at_ttl() -> None:
    # The cache hit test is `now - hit[0] < ttl`, so a sample taken exactly `ttl` later
    # is a MISS. Pins the strict inequality (a flip to `<=` would extend staleness).
    calls = {"n": 0}

    def counting_factory(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    clock = FakeClock()
    client = YFinanceClient(ticker_factory=counting_factory, time_fn=clock, quote_ttl=30.0)
    client.get_quote(["AAPL"])
    clock.advance(30.0)  # exactly at the TTL boundary -> not strictly within -> refetch
    client.get_quote(["AAPL"])
    assert calls["n"] == 2


def test_get_quote_missing_price_raises_symbol_not_found() -> None:
    client = _client(factory=fake_ticker_factory(fast_info={"last_price": None}))
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("BADSYM")


def test_get_quote_surfaces_yfinance_error_message() -> None:
    client = _client(
        factory=fake_ticker_factory(fast_info_error=YFException("yahoo says: rate limited"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client._fetch_quote("AAPL")
    assert "yahoo says: rate limited" in str(exc.value)


def test_get_quote_invalid_symbol_returns_clean_symbol_not_found() -> None:
    client = _client(factory=fake_ticker_factory(fast_info_error=KeyError("exchangeTimezoneName")))
    with pytest.raises(SymbolNotFound) as exc:
        client._fetch_quote("BAD")
    assert "No quote data for 'BAD'" in str(exc.value)
    assert "exchangeTimezoneName" not in str(exc.value)


def test_get_quote_none_price_is_symbol_not_found() -> None:
    client = _client(factory=fake_ticker_factory(fast_info={"last_price": None}))
    with pytest.raises(SymbolNotFound) as exc:
        client._fetch_quote("BAD")
    assert "No quote data for" in str(exc.value)


def test_get_quote_no_second_network_call_on_failure() -> None:
    calls = {"history": 0}

    class _Ticker:
        @property
        def fast_info(self) -> Any:
            raise KeyError("x")

        def history(self, **_kwargs: Any) -> Any:
            calls["history"] += 1
            return None

    def factory(_symbol: str) -> Any:
        return _Ticker()

    client = _client(factory=factory)
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("BAD")
    assert calls["history"] == 0


def test_get_price_history_parses_bars_and_summary() -> None:
    df = make_history_df([100.0, 101.0, 102.0, 103.0])
    client = _client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.bars == 4
    assert hist.summary.start_close == 100.0
    assert hist.summary.end_close == 103.0
    assert hist.summary.total_return_percent == pytest.approx(3.0)
    assert hist.summary.period_high == 104.0
    assert hist.bars[-1].close == 103.0
    assert hist.truncated is False


def test_get_price_history_empty_raises_symbol_not_found() -> None:
    client = _client(factory=fake_ticker_factory(history_df=make_history_df([])))
    with pytest.raises(SymbolNotFound):
        client.get_price_history("BADSYM", period="1mo", interval="1d")


def test_get_price_history_truncates_to_max_bars() -> None:
    df = make_history_df([float(i) for i in range(1, 11)])
    client = YFinanceClient(
        ticker_factory=fake_ticker_factory(history_df=df),
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        max_bars=5,
    )
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert len(hist.bars) == 5
    assert hist.truncated is True
    assert hist.summary.bars == 10


def test_get_quote_nan_price_raises_symbol_not_found() -> None:
    client = _client(
        factory=fake_ticker_factory(fast_info={"last_price": float("nan"), "previous_close": 188.0})
    )
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("AAPL")


def test_get_quote_nan_previous_close_yields_none_change() -> None:
    fi = {**QUOTE_FI, "previous_close": float("nan")}
    client = _client(factory=fake_ticker_factory(fast_info=fi))
    [q] = client.get_quote(["AAPL"]).quotes
    assert q.price == 190.0
    assert q.change is None
    assert q.change_percent is None


def test_get_price_history_drops_nan_rows() -> None:
    df = make_history_df([100.0, 101.0, 102.0])
    df.loc[df.index[1], "Close"] = float("nan")
    client = _client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.bars == 2
    assert all(math.isfinite(b.close) for b in hist.bars)


def test_get_price_history_all_nan_raises() -> None:
    df = make_history_df([100.0])
    df.loc[df.index[0], "Close"] = float("nan")
    client = _client(factory=fake_ticker_factory(history_df=df))
    with pytest.raises(SymbolNotFound):
        client.get_price_history("AAPL", period="1mo", interval="1d")


def test_get_price_history_zero_start_close_no_crash() -> None:
    df = make_history_df([0.0, 5.0])
    client = _client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.total_return_percent == 0.0


def test_get_quote_inf_price_raises_symbol_not_found() -> None:
    client = _client(
        factory=fake_ticker_factory(fast_info={"last_price": float("inf"), "previous_close": 188.0})
    )
    with pytest.raises(SymbolNotFound):
        client._fetch_quote("X")


def test_get_price_history_drops_inf_rows() -> None:
    df = make_history_df([100.0, 101.0, 102.0])
    df.loc[df.index[1], "Close"] = float("inf")
    client = _client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1mo", interval="1d")
    assert hist.summary.bars == 2


def test_get_price_history_all_inf_raises() -> None:
    df = make_history_df([100.0])
    df.loc[df.index[0], "Close"] = float("inf")
    client = _client(factory=fake_ticker_factory(history_df=df))
    with pytest.raises(SymbolNotFound):
        client.get_price_history("AAPL", period="1mo", interval="1d")


class _RaisingCurrencyFastInfo:
    """fast_info stub whose `currency` property raises, last_price is fine."""

    last_price = 190.0
    previous_close = 188.0

    @property
    def currency(self) -> str:
        raise YFException("boom")


def test_get_quote_fast_info_attr_error_becomes_data_unavailable() -> None:
    df = make_history_df([100.0])

    def factory(_symbol: str) -> Any:
        return SimpleNamespace(fast_info=_RaisingCurrencyFastInfo(), history=lambda **_k: df)

    client = _client(factory=factory)
    with pytest.raises(DataUnavailable) as exc:
        client._fetch_quote("X")
    assert "boom" in str(exc.value)


def test_get_price_history_parse_error_becomes_data_unavailable() -> None:
    df = make_history_df([100.0, 101.0]).drop(columns=["Volume"])
    client = _client(factory=fake_ticker_factory(history_df=df))
    with pytest.raises(DataUnavailable) as exc:
        client.get_price_history("AAPL", period="1mo", interval="1d")
    assert "AAPL" in str(exc.value)


def test_get_quote_distinct_symbols_cached_independently() -> None:
    calls = {"n": 0}

    def counting_factory(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    client = YFinanceClient(
        ticker_factory=counting_factory, time_fn=FakeClock(), quote_ttl=30.0, history_ttl=300.0
    )
    results = client.get_quote(["AAPL", "MSFT"]).quotes
    assert [r.symbol for r in results] == ["AAPL", "MSFT"]
    assert calls["n"] == 2


INCOME = {  # rows: label -> [most-recent, prior]
    "Total Revenue": [400.0, 380.0],
    "Net Income": [100.0, float("nan")],
}


def _fin_client(**kw: Any) -> YFinanceClient:
    factory = kw.pop("factory")
    return YFinanceClient(
        ticker_factory=factory,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )


def test_get_financials_parses_periods_and_line_items() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials("AAPL", "income", "annual")
    assert fs.symbol == "AAPL" and fs.statement == "income" and fs.period == "annual"
    assert fs.period_ends == ["2024-09-30", "2023-09-30"]
    assert fs.line_items["Total Revenue"] == [400.0, 380.0]
    assert fs.line_items["Net Income"] == [100.0, None]  # NaN -> None


def test_get_financials_line_items_filter() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials("AAPL", "income", "annual", line_items=["Total Revenue", "Nope"])
    assert list(fs.line_items.keys()) == ["Total Revenue"]  # only matching labels, "Nope" dropped


def test_get_financials_quarterly_attr() -> None:
    df = make_financials_df({"Total Revenue": [100.0]}, ["2025-03-31"])
    client = _fin_client(factory=fake_ticker_factory(financials={"quarterly_balance_sheet": df}))
    fs = client.get_financials("AAPL", "balance", "quarterly")
    assert fs.period_ends == ["2025-03-31"]


def test_get_financials_empty_raises_symbol_not_found() -> None:
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": pd.DataFrame()}))
    with pytest.raises(SymbolNotFound):
        client.get_financials("BAD", "income", "annual")


def test_get_financials_fetch_error_is_data_unavailable() -> None:
    client = _fin_client(factory=fake_ticker_factory(financials_error=RuntimeError("yahoo down")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_financials("AAPL", "income", "annual")
    assert "yahoo down" in str(exc.value)


def test_get_financials_parse_error_is_data_unavailable() -> None:
    # Non-datetime columns make `col.date()` raise inside the parse block.
    df = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["Total Revenue"])
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    with pytest.raises(DataUnavailable) as exc:
        client.get_financials("AAPL", "income", "annual")
    assert "AAPL" in str(exc.value)


def test_get_financials_cached_within_ttl() -> None:
    calls = {"n": 0}
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])

    def counting(symbol: str) -> Any:
        calls["n"] += 1
        return fake_ticker_factory(financials={"income_stmt": df})(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_financials("AAPL", "income", "annual")
    client.get_financials("AAPL", "income", "annual")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.get_financials("AAPL", "income", "annual")
    assert calls["n"] == 2


FULL_INFO = {
    "longName": "Apple Inc.",
    "shortName": "Apple",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "website": "https://www.apple.com",
    "fullTimeEmployees": 166000,
    "longBusinessSummary": "Apple designs phones.",
    "currency": "USD",
    "marketCap": 4.5e12,
    "trailingPE": 37.7,
    "forwardPE": 32.5,
    "dividendYield": 0.35,
    "beta": 1.06,
}


def _profile_client(**kw: Any) -> YFinanceClient:
    factory = kw.pop("factory")
    return YFinanceClient(
        ticker_factory=factory,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )


def test_get_company_profile_maps_fields() -> None:
    client = _profile_client(factory=fake_ticker_factory(info=FULL_INFO))
    p = client.get_company_profile("AAPL")
    assert p.symbol == "AAPL"
    assert p.name == "Apple Inc." and p.sector == "Technology"
    assert p.industry == "Consumer Electronics" and p.country == "United States"
    assert p.employees == 166000 and p.currency == "USD"
    assert p.market_cap == 4.5e12 and p.trailing_pe == 37.7 and p.beta == 1.06
    assert p.summary == "Apple designs phones."


def test_get_company_profile_name_falls_back_to_short_name() -> None:
    info = {"shortName": "Apple", "sector": "Tech"}
    client = _profile_client(factory=fake_ticker_factory(info=info))
    assert client.get_company_profile("AAPL").name == "Apple"


def test_get_company_profile_recent_dividends_capped_at_8() -> None:
    dates = [f"2023-{m:02d}-01" for m in range(1, 13)]  # 12 dividends
    div = make_series(dates, [0.20 + i * 0.01 for i in range(12)])
    client = _profile_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=div))
    p = client.get_company_profile("AAPL")
    assert len(p.recent_dividends) == 8  # only most recent 8
    assert p.recent_dividends[-1].date == "2023-12-01"  # newest last
    assert p.recent_dividends[0].date == "2023-05-01"


def test_get_company_profile_all_splits() -> None:
    spl = make_series(["1987-06-16", "2000-06-21", "2020-08-31"], [2.0, 2.0, 4.0])
    client = _profile_client(factory=fake_ticker_factory(info=FULL_INFO, splits=spl))
    p = client.get_company_profile("AAPL")
    assert len(p.splits) == 3 and p.splits[-1].ratio == 4.0 and p.splits[-1].date == "2020-08-31"


def test_get_company_profile_missing_fields_are_none_and_empty() -> None:
    client = _profile_client(factory=fake_ticker_factory(info={"longName": "X Corp"}))
    p = client.get_company_profile("X")
    assert p.name == "X Corp" and p.sector is None and p.market_cap is None
    assert p.recent_dividends == [] and p.splits == []


def test_get_company_profile_no_name_raises_symbol_not_found() -> None:
    client = _profile_client(factory=fake_ticker_factory(info={"trailingPegRatio": None}))
    with pytest.raises(SymbolNotFound):
        client.get_company_profile("BAD")


def test_get_company_profile_typed_error_is_data_unavailable() -> None:
    client = _profile_client(factory=fake_ticker_factory(info_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert "rate limited" in str(exc.value)


def test_get_company_profile_raw_error_is_symbol_not_found() -> None:
    client = _profile_client(factory=fake_ticker_factory(info_error=KeyError("boom")))
    with pytest.raises(SymbolNotFound):
        client.get_company_profile("AAPL")


def test_get_company_profile_skips_non_finite_dividends_and_splits() -> None:
    div = make_series(["2023-01-01", "2023-06-01"], [0.20, float("nan")])
    spl = make_series(["2000-06-21", "2020-08-31"], [float("inf"), 4.0])
    client = _profile_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=div, splits=spl))
    p = client.get_company_profile("AAPL")
    assert [d.date for d in p.recent_dividends] == ["2023-01-01"]  # NaN dropped
    assert [s.date for s in p.splits] == ["2020-08-31"]  # inf dropped


def test_get_company_profile_parse_error_is_data_unavailable() -> None:
    # A non-datetime index makes `ts.date()` raise inside the parse block.
    bad_div = pd.Series([0.25], index=["not-a-date"], dtype=float)
    client = _profile_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=bad_div))
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert "AAPL" in str(exc.value)


def test_get_company_profile_dividends_read_error_is_data_unavailable() -> None:
    # .info already identified the instrument, so a failing dividends READ is a data
    # availability issue (DataUnavailable), NOT SymbolNotFound. Guards the refactor that
    # moved this read out of the SymbolNotFound try block.
    client = _profile_client(
        factory=fake_ticker_factory(info=FULL_INFO, dividends_error=YFException("divs down"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert type(exc.value) is DataUnavailable  # not the SymbolNotFound subclass
    assert "divs down" in str(exc.value)


def test_get_company_profile_splits_read_error_is_data_unavailable() -> None:
    client = _profile_client(
        factory=fake_ticker_factory(info=FULL_INFO, splits_error=YFException("splits down"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert type(exc.value) is DataUnavailable
    assert "splits down" in str(exc.value)


def test_get_financials_filter_reuses_cached_fetch() -> None:
    calls = {"n": 0}
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(financials={"income_stmt": df})(symbol)

    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    full = client.get_financials("AAPL", "income", "annual")
    f1 = client.get_financials("AAPL", "income", "annual", line_items=["Total Revenue"])
    f2 = client.get_financials("AAPL", "income", "annual", line_items=["Net Income"])
    assert calls["n"] == 1  # one fetch; both filters reuse the cached statement
    assert list(f1.line_items) == ["Total Revenue"]
    assert list(f2.line_items) == ["Net Income"]
    assert set(full.line_items) == {"Total Revenue", "Net Income"}  # cached object un-mutated


def test_get_company_profile_nan_employees_nulled_not_fatal() -> None:
    info = {**FULL_INFO, "fullTimeEmployees": float("nan")}
    client = _profile_client(factory=fake_ticker_factory(info=info))
    p = client.get_company_profile("AAPL")
    assert p.employees is None  # junk field nulls; profile still returned
    assert p.name == "Apple Inc."


def test_get_company_profile_float_employees_coerced_to_int() -> None:
    info = {**FULL_INFO, "fullTimeEmployees": 166000.0}
    client = _profile_client(factory=fake_ticker_factory(info=info))
    assert client.get_company_profile("AAPL").employees == 166000


@pytest.mark.parametrize(
    ("statement", "period", "attr"),
    [
        ("income", "annual", "income_stmt"),
        ("income", "quarterly", "quarterly_income_stmt"),
        ("balance", "annual", "balance_sheet"),
        ("balance", "quarterly", "quarterly_balance_sheet"),
        ("cashflow", "annual", "cashflow"),
        ("cashflow", "quarterly", "quarterly_cashflow"),
    ],
)
def test_get_financials_all_statement_period_combos(
    statement: Literal["income", "balance", "cashflow"],
    period: Literal["annual", "quarterly"],
    attr: str,
) -> None:
    df = make_financials_df({"X": [1.0]}, ["2024-12-31"])
    client = _fin_client(factory=fake_ticker_factory(financials={attr: df}))
    fs = client.get_financials("AAPL", statement, period)
    assert fs.statement == statement and fs.period == period
    assert fs.period_ends == ["2024-12-31"] and fs.line_items["X"] == [1.0]


def test_get_quote_zero_previous_close_change_pct_none() -> None:
    fi = {**QUOTE_FI, "previous_close": 0.0}
    [q] = _client(factory=fake_ticker_factory(fast_info=fi)).get_quote(["AAPL"]).quotes
    assert q.change == pytest.approx(190.0) and q.change_percent is None and q.previous_close == 0.0


def test_get_price_history_caches_and_keys_on_interval() -> None:
    calls = {"n": 0}
    df = make_history_df([100.0, 101.0])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(history_df=df)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_price_history("AAPL", "1mo", "1d")
    client.get_price_history("AAPL", "1mo", "1d")
    assert calls["n"] == 1
    client.get_price_history("AAPL", "1mo", "1wk")  # different interval -> distinct key
    assert calls["n"] == 2
    clock.advance(301.0)
    client.get_price_history("AAPL", "1mo", "1d")  # expired -> refetch
    assert calls["n"] == 3


def test_get_company_profile_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(info=FULL_INFO)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_company_profile("AAPL")
    client.get_company_profile("AAPL")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.get_company_profile("AAPL")
    assert calls["n"] == 2


def test_get_quote_batch_keeps_good_tickers_when_one_is_missing() -> None:
    def factory(symbol: str) -> object:
        if symbol == "AAPL":
            return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)
        return fake_ticker_factory(fast_info_error=KeyError("exchangeTimezoneName"))(symbol)

    result = _client(factory=factory).get_quote(["AAPL", "MSFT"])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert [e.symbol for e in result.errors] == ["MSFT"]


def test_get_price_history_single_bar() -> None:
    client = _client(factory=fake_ticker_factory(history_df=make_history_df([100.0])))
    h = client.get_price_history("AAPL", "1d", "1d")
    assert h.summary.bars == 1 and h.summary.total_return_percent == 0.0
    assert h.summary.start_date == h.summary.end_date and h.truncated is False


def test_get_quote_non_price_nan_fields_nulled() -> None:
    fi = {**QUOTE_FI, "market_cap": float("nan"), "last_volume": float("nan")}
    [q] = _client(factory=fake_ticker_factory(fast_info=fi)).get_quote(["AAPL"]).quotes
    assert q.price == 190.0 and q.market_cap is None and q.volume is None


def test_get_company_profile_empty_long_name_uses_short_name() -> None:
    factory = fake_ticker_factory(info={"longName": "", "shortName": "Apple"})
    client = _profile_client(factory=factory)
    assert client.get_company_profile("AAPL").name == "Apple"


def test_get_financials_line_items_preserve_order() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials(
        "AAPL", "income", "annual", line_items=["Net Income", "Total Revenue"]
    )
    assert list(fs.line_items) == ["Net Income", "Total Revenue"]


def test_get_financials_line_items_all_miss_empty() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials("AAPL", "income", "annual", line_items=["Nonexistent"])
    assert fs.line_items == {}


def test_get_company_profile_dividends_below_cap_returns_all() -> None:
    div = make_series(["2023-02-01", "2023-05-01", "2023-08-01"], [0.23, 0.24, 0.25])
    client = _profile_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=div))
    p = client.get_company_profile("AAPL")
    assert [d.date for d in p.recent_dividends] == ["2023-02-01", "2023-05-01", "2023-08-01"]


METRICS_INFO = {
    "longName": "Apple Inc.",
    "trailingPE": 37.73,
    "forwardPE": 32.48,
    "priceToBook": 42.98,
    "priceToSalesTrailing12Months": 10.15,
    "pegRatio": 2.72,
    "enterpriseValue": 4599540350976,
    "enterpriseToEbitda": 28.75,
    "enterpriseToRevenue": 10.19,
    "returnOnEquity": 1.41,
    "returnOnAssets": 0.26,
    "grossMargins": 0.478,
    "operatingMargins": 0.322,
    "profitMargins": 0.271,
    "ebitdaMargins": 0.354,
    "debtToEquity": 79.55,
    "currentRatio": 1.07,
    "quickRatio": 0.906,
    "totalDebt": 84710998016,
    "totalCash": 68507000832,
    "freeCashflow": 101090746368,
    "ebitda": 159975997440,
    "trailingEps": 8.27,
    "forwardEps": 9.61,
    "revenuePerShare": 30.53,
    "bookValue": 7.26,
}


def _metrics_client(**kw: Any) -> YFinanceClient:
    factory = kw.pop("factory")
    return YFinanceClient(
        ticker_factory=factory,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )


def test_get_key_metrics_maps_fields() -> None:
    m = _metrics_client(factory=fake_ticker_factory(info=METRICS_INFO)).get_key_metrics("AAPL")
    assert isinstance(m, KeyMetrics)
    assert m.symbol == "AAPL"
    assert m.trailing_pe == 37.73 and m.forward_pe == 32.48 and m.price_to_book == 42.98
    assert m.price_to_sales == 10.15 and m.peg_ratio == 2.72
    assert m.enterprise_value == 4599540350976 and m.ev_to_ebitda == 28.75
    assert m.ev_to_revenue == 10.19
    assert m.return_on_equity == 1.41 and m.return_on_assets == 0.26
    assert m.gross_margins == 0.478 and m.operating_margins == 0.322
    assert m.profit_margins == 0.271 and m.ebitda_margins == 0.354
    assert m.debt_to_equity == 79.55 and m.current_ratio == 1.07 and m.quick_ratio == 0.906
    assert m.total_debt == 84710998016 and m.total_cash == 68507000832
    assert m.free_cashflow == 101090746368 and m.ebitda == 159975997440
    assert m.trailing_eps == 8.27 and m.forward_eps == 9.61
    assert m.revenue_per_share == 30.53 and m.book_value == 7.26


def test_get_key_metrics_missing_and_nan_are_none() -> None:
    info = {"longName": "X Corp", "trailingPE": float("nan")}
    m = _metrics_client(factory=fake_ticker_factory(info=info)).get_key_metrics("X")
    assert m.symbol == "X" and m.trailing_pe is None
    assert m.ebitda is None and m.profit_margins is None


def test_get_key_metrics_no_name_raises_symbol_not_found() -> None:
    client = _metrics_client(factory=fake_ticker_factory(info={"trailingPegRatio": None}))
    with pytest.raises(SymbolNotFound):
        client.get_key_metrics("BAD")


def test_get_key_metrics_typed_error_is_data_unavailable() -> None:
    client = _metrics_client(factory=fake_ticker_factory(info_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_key_metrics("AAPL")
    assert "rate limited" in str(exc.value)


def test_get_key_metrics_raw_error_is_symbol_not_found() -> None:
    client = _metrics_client(factory=fake_ticker_factory(info_error=KeyError("boom")))
    with pytest.raises(SymbolNotFound):
        client.get_key_metrics("AAPL")


def test_get_key_metrics_mapping_failure_is_data_unavailable() -> None:
    # A value that survives the name check but fails float() coercion in mapping.
    info = {"longName": "Apple Inc.", "trailingPE": object()}
    client = _metrics_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_key_metrics("AAPL")
    assert "Failed to parse metrics for 'AAPL'" in str(exc.value)


def test_get_key_metrics_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(info=METRICS_INFO)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_key_metrics("AAPL")
    client.get_key_metrics("AAPL")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.get_key_metrics("AAPL")
    assert calls["n"] == 2


def _perf_client(**kw: Any) -> YFinanceClient:
    factory = kw.pop("factory")
    return YFinanceClient(
        ticker_factory=factory,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )


def test_analyze_performance_computes_stats() -> None:
    closes = [100.0 + i for i in range(120)]  # 120 calendar days -> past the annualization gate
    client = _perf_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    p = client.analyze_performance("AAPL", "6mo")
    assert isinstance(p, PerformanceStats)
    assert p.symbol == "AAPL" and p.period == "6mo" and p.bars == 120
    assert p.total_return_percent == pytest.approx(analytics.total_return(closes))
    assert p.periods_per_year is not None
    assert p.annualized_volatility_percent == pytest.approx(
        analytics.annualized_volatility(closes, p.periods_per_year)
    )
    assert p.max_drawdown_percent == pytest.approx(analytics.max_drawdown(closes))
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))
    assert p.sma_200 is None  # < 200 bars
    assert p.start_date == "2024-01-01" and p.end_date == "2024-04-29"


def _perf_stats(closes: list[float], **df_kw: Any) -> PerformanceStats:
    df = make_history_df(closes, **df_kw)
    return _perf_client(factory=fake_ticker_factory(history_df=df)).analyze_performance("X", "1y")


def test_analyze_performance_one_year_annualized_equals_total_return() -> None:
    """The acceptance case: 366 seven-day-a-week bars spanning one calendar year.

    This is the reported BTC-USD shape. The old code applied an exponent of 252/365 and
    reported -16.49% for a real -22.98% year; the two must agree over a one-year window.
    The residual tolerance is exact, not slop: 365 elapsed days is 365/365.25 = 0.99932
    years, so the CAGR exponent is 1.000685 and the figures differ in the 4th significant
    digit. Exact equality at years=1.0 is pinned in test_analytics_logic.py.
    """
    closes = [100.0 * (0.7702 ** (i / 365)) for i in range(366)]
    p = _perf_stats(closes)
    assert p.bars == 366
    assert p.start_date == "2024-01-01" and p.end_date == "2024-12-31"
    assert p.total_return_percent == pytest.approx(-22.98, rel=1e-3)
    assert p.annualized_return_percent is not None
    assert p.annualized_return_percent == pytest.approx(p.total_return_percent, rel=2e-3)


def test_analyze_performance_infers_seven_day_calendar() -> None:
    p = _perf_stats([100.0 + i for i in range(366)])
    assert p.periods_per_year == pytest.approx(365.0, rel=0.01)


def test_analyze_performance_infers_weekday_calendar() -> None:
    # A weekday-only series infers 5/7 of the calendar = ~261/yr. Real exchanges print ~252
    # because of ~9 market holidays, which pandas freq="B" does not model -- the point of the
    # assertion is that the inferred rate tracks the weekday calendar and is nowhere near 365.
    p = _perf_stats([100.0 + i for i in range(261)], freq="B")
    assert p.periods_per_year == pytest.approx(365.25 * 5 / 7, rel=0.02)
    assert p.periods_per_year is not None and p.periods_per_year < 300.0


def test_analyze_performance_annualized_return_ignores_bar_count() -> None:
    # The same calendar move over the same year, printed at two different bar rates.
    # The old bar-count exponent gave two different answers; calendar time gives one.
    seven_day = _perf_stats([100.0] * 365 + [80.0])
    weekday = _perf_stats([100.0] * 260 + [80.0], freq="B")
    assert seven_day.annualized_return_percent is not None
    assert weekday.annualized_return_percent is not None
    assert seven_day.annualized_return_percent == pytest.approx(
        weekday.annualized_return_percent, rel=0.02
    )


def test_analyze_performance_crypto_volatility_is_not_understated() -> None:
    # Identical daily dispersion, 24/7 vs weekday. The 24/7 series observes more returns per
    # year, so the same dispersion must annualize higher by sqrt of the ratio of the two
    # inferred factors. Under the old constant 252 both series annualized identically.
    # The SAME closes on both calendars, so per-observation dispersion is identical and the
    # only thing that can move the annualized figure is the inferred observation rate.
    closes = [100.0 + (5.0 if i % 2 else 0.0) for i in range(261)]
    seven_day = _perf_stats(closes)
    weekday = _perf_stats(closes, freq="B")
    assert seven_day.annualized_volatility_percent is not None
    assert weekday.annualized_volatility_percent is not None
    assert seven_day.periods_per_year is not None and weekday.periods_per_year is not None
    assert seven_day.periods_per_year > weekday.periods_per_year
    ratio = seven_day.annualized_volatility_percent / weekday.annualized_volatility_percent
    expected = math.sqrt(seven_day.periods_per_year / weekday.periods_per_year)
    assert ratio == pytest.approx(expected, rel=1e-6)
    assert ratio > 1.0  # the 24/7 instrument is not flattened to the equity convention


def test_analyze_performance_short_window_nulls_annualized_fields() -> None:
    # The reported AAPL period=5d case: a few days' move must not become a yearly figure.
    p = _perf_stats([100.0, 101.0, 102.0, 103.0, 104.0])
    assert p.annualized_return_percent is None
    assert p.annualized_volatility_percent is None
    assert p.periods_per_year is None
    # Everything that does not annualize still reports.
    assert p.total_return_percent == pytest.approx(4.0)
    assert p.max_drawdown_percent == pytest.approx(0.0)


def test_analyze_performance_annualizes_at_the_threshold() -> None:
    # 91 consecutive daily bars => exactly 90 elapsed days => on the gate, so it annualizes.
    p = _perf_stats([100.0 + i for i in range(91)])
    assert p.annualized_return_percent is not None
    assert p.annualized_volatility_percent is not None
    assert p.periods_per_year is not None


def test_analyze_performance_does_not_annualize_below_the_threshold() -> None:
    # 90 bars => 89 elapsed days => one day short of the gate.
    p = _perf_stats([100.0 + i for i in range(90)])
    assert p.annualized_return_percent is None
    assert p.annualized_volatility_percent is None
    assert p.periods_per_year is None


def test_analyze_performance_shares_the_bars_cache_with_get_price_history() -> None:
    calls = {"n": 0}
    df = make_history_df([100.0 + i for i in range(120)])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(history_df=df)(symbol)

    client = _perf_client(factory=counting)
    client.get_price_history("AAPL", period="6mo", interval="1d")
    client.analyze_performance("AAPL", "6mo")
    assert calls["n"] == 1  # one fetch feeds both derived views


def test_get_price_history_reuses_bars_fetched_by_analyze_performance() -> None:
    calls = {"n": 0}
    df = make_history_df([100.0 + i for i in range(120)])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(history_df=df)(symbol)

    client = _perf_client(factory=counting)
    client.analyze_performance("AAPL", "6mo")
    client.get_price_history("AAPL", period="6mo", interval="1d")
    assert calls["n"] == 1  # the dedupe works in either order


def test_analyze_performance_sma_when_enough_bars() -> None:
    closes = [100.0 + i for i in range(60)]  # 60 daily bars
    client = _perf_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    p = client.analyze_performance("AAPL", "3mo")
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))
    assert p.sma_200 is None  # still < 200


def test_analyze_performance_smas_survive_a_short_window() -> None:
    # SMAs do not annualize, so the 90-day gate must not blank them.
    closes = [100.0 + i for i in range(60)]  # 59 elapsed days, under the gate
    client = _perf_client(factory=fake_ticker_factory(history_df=make_history_df(closes)))
    p = client.analyze_performance("AAPL", "3mo")
    assert p.annualized_return_percent is None
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))


def test_analyze_performance_too_few_bars_raises() -> None:
    client = _perf_client(factory=fake_ticker_factory(history_df=make_history_df([100.0])))
    with pytest.raises(DataUnavailable):
        client.analyze_performance("AAPL", "1d")


def test_analyze_performance_invalid_symbol_raises() -> None:
    client = _perf_client(factory=fake_ticker_factory(history_df=pd.DataFrame()))
    with pytest.raises(SymbolNotFound):
        client.analyze_performance("BAD", "1y")


def test_analyze_performance_caches_within_ttl() -> None:
    calls = {"n": 0}
    df = make_history_df([100.0, 110.0, 99.0])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(history_df=df)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.analyze_performance("AAPL", "1mo")
    client.analyze_performance("AAPL", "1mo")
    assert calls["n"] == 1
    clock.advance(301.0)
    client.analyze_performance("AAPL", "1mo")
    assert calls["n"] == 2


def test_analyze_performance_cache_keys_on_period() -> None:
    calls = {"n": 0}
    df = make_history_df([100.0, 110.0, 99.0])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(history_df=df)(symbol)

    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.analyze_performance("AAPL", "1mo")
    client.analyze_performance("AAPL", "1mo")
    assert calls["n"] == 1
    client.analyze_performance("AAPL", "1y")  # distinct period -> distinct key
    assert calls["n"] == 2


def test_analyze_performance_period_propagates() -> None:
    df = make_history_df([100.0, 110.0, 99.0])
    p = _perf_client(factory=fake_ticker_factory(history_df=df)).analyze_performance("AAPL", "5y")
    assert p.period == "5y"


def test_analyze_performance_sma_200_populated() -> None:
    closes = [100.0 + i for i in range(250)]
    p = _perf_client(
        factory=fake_ticker_factory(history_df=make_history_df(closes))
    ).analyze_performance("AAPL", "1y")
    assert p.sma_200 == pytest.approx(analytics.sma(closes, 200)) and p.sma_200 is not None
    assert p.sma_50 == pytest.approx(analytics.sma(closes, 50))


def test_profile_and_metrics_caches_do_not_collide() -> None:
    calls = {"n": 0}

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(info=METRICS_INFO)(symbol)

    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    prof = client.get_company_profile("AAPL")
    metr = client.get_key_metrics("AAPL")
    assert calls["n"] == 2  # distinct cache keys -> two fetches
    assert prof.symbol == "AAPL" and metr.symbol == "AAPL"


ANALYST_INFO = {
    "longName": "Apple Inc.",
    "shortName": "Apple",
    "currency": "USD",
    "currentPrice": 190.0,
    "recommendationKey": "buy",
    "recommendationMean": 1.9,
    "numberOfAnalystOpinions": 40,
    "targetMeanPrice": 220.0,
    "targetMedianPrice": 218.0,
    "targetHighPrice": 300.0,
    "targetLowPrice": 150.0,
}

ANALYST_TREND = [
    ("0m", 12, 20, 6, 1, 0),
    ("-1m", 11, 19, 7, 1, 0),
    ("-2m", 10, 18, 8, 2, 1),
    ("-3m", 9, 17, 9, 2, 1),
]


def test_get_analyst_data_happy_path() -> None:
    df = make_recommendations_df(ANALYST_TREND)
    client = make_client(factory=fake_ticker_factory(info=ANALYST_INFO, recommendations=df))
    a = client.get_analyst_data("AAPL")
    assert isinstance(a, AnalystData)
    assert a.symbol == "AAPL" and a.currency == "USD"
    assert a.current_price == 190.0
    assert a.recommendation_key == "buy" and a.recommendation_mean == 1.9
    assert a.number_of_analysts == 40
    assert a.target_mean_price == 220.0 and a.target_median_price == 218.0
    assert a.target_high_price == 300.0 and a.target_low_price == 150.0
    assert len(a.recommendation_trend) == 4
    first = a.recommendation_trend[0]
    assert first.period == "0m"
    assert (first.strong_buy, first.buy, first.hold, first.sell, first.strong_sell) == (
        12,
        20,
        6,
        1,
        0,
    )
    assert [p.period for p in a.recommendation_trend] == ["0m", "-1m", "-2m", "-3m"]
    last = a.recommendation_trend[-1]
    assert (last.strong_buy, last.buy, last.hold, last.sell, last.strong_sell) == (9, 17, 9, 2, 1)


def test_get_analyst_data_no_coverage_raises_data_unavailable() -> None:
    info = {"longName": "SPDR S&P 500 ETF", "currency": "USD"}
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("SPY")
    assert "No analyst coverage for 'SPY'" in str(exc.value)


def test_get_analyst_data_no_coverage_is_not_symbol_not_found() -> None:
    # SymbolNotFound subclasses DataUnavailable; assert it is the base class, not the subclass.
    info = {"longName": "SPDR S&P 500 ETF", "currency": "USD"}
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc_info:
        client.get_analyst_data("SPY")
    assert type(exc_info.value) is DataUnavailable


def test_get_analyst_data_raw_error_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=KeyError("boom")))
    with pytest.raises(SymbolNotFound) as exc:
        client.get_analyst_data("AAPL")
    assert "No analyst data for 'AAPL'" in str(exc.value)


def test_get_analyst_data_empty_info_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info={}))
    with pytest.raises(SymbolNotFound):
        client.get_analyst_data("BAD")


def test_get_analyst_data_typed_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "rate limited" in str(exc.value)


def test_get_analyst_data_nan_and_missing_numerics_are_none() -> None:
    info = {
        "longName": "Apple Inc.",
        "currency": "USD",
        "numberOfAnalystOpinions": 40,  # ensures coverage
        "recommendationMean": float("nan"),
        # target prices all missing
    }
    client = make_client(factory=fake_ticker_factory(info=info))
    a = client.get_analyst_data("AAPL")
    assert a.number_of_analysts == 40
    assert a.recommendation_mean is None
    assert a.target_mean_price is None and a.target_median_price is None
    assert a.target_high_price is None and a.target_low_price is None


def test_get_analyst_data_coverage_via_targets_only_empty_trend() -> None:
    info = {"longName": "Apple Inc.", "currency": "USD", "targetMeanPrice": 220.0}
    client = make_client(factory=fake_ticker_factory(info=info))
    a = client.get_analyst_data("AAPL")
    assert a.target_mean_price == 220.0
    assert a.recommendation_trend == []  # empty recommendations frame


def test_recommendation_trend_none_df_returns_empty() -> None:
    # yfinance can return None for recommendations; the helper must tolerate it.
    assert _recommendation_trend(None) == []


def test_get_analyst_data_parse_error_is_data_unavailable() -> None:
    # Coverage exists (numberOfAnalystOpinions), but a non-coercible recommendation
    # count makes RecommendationPeriod construction fail in the parse stage.
    info = {"longName": "Apple Inc.", "currency": "USD", "numberOfAnalystOpinions": 40}
    bad_df = pd.DataFrame(
        [{"period": "0m", "strongBuy": object(), "buy": 1, "hold": 1, "sell": 0, "strongSell": 0}]
    )
    client = make_client(factory=fake_ticker_factory(info=info, recommendations=bad_df))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "Failed to parse analyst data for 'AAPL'" in str(exc.value)


def test_get_analyst_data_recommendations_read_error_is_data_unavailable() -> None:
    # A failing recommendations READ (after .info identified coverage) surfaces as
    # DataUnavailable, not SymbolNotFound — guards the refactor that moved this read.
    info = {"longName": "Apple Inc.", "currency": "USD", "numberOfAnalystOpinions": 40}
    client = make_client(
        factory=fake_ticker_factory(info=info, recommendations_error=YFException("recs down"))
    )
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert type(exc.value) is DataUnavailable
    assert "recs down" in str(exc.value)


def test_get_analyst_data_non_numeric_recommendation_mean_raises_data_unavailable() -> None:
    # Defensive: in live data recommendationMean is always a float, but if the source
    # ever returns a non-numeric value for it, float() raises ValueError — that must
    # surface as DataUnavailable, not a raw traceback (matches _fetch_metrics).
    info = {
        "longName": "Apple Inc.",
        "currency": "USD",
        "recommendationMean": "n/a",  # non-numeric — float() will raise ValueError
        "numberOfAnalystOpinions": 40,
    }
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "Failed to parse analyst data for 'AAPL'" in str(exc.value)


def test_get_analyst_data_non_numeric_target_price_raises_data_unavailable() -> None:
    # Same defensive guard for a target price: a non-numeric value must surface as
    # DataUnavailable, not a raw ValueError/TypeError.
    info = {
        "longName": "Apple Inc.",
        "currency": "USD",
        "numberOfAnalystOpinions": 40,
        "targetMeanPrice": "n/a",  # non-numeric — float() will raise ValueError
    }
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_analyst_data("AAPL")
    assert "Failed to parse analyst data for 'AAPL'" in str(exc.value)


def test_get_analyst_data_caches_within_ttl() -> None:
    calls = {"n": 0}
    df = make_recommendations_df(ANALYST_TREND)

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(info=ANALYST_INFO, recommendations=df)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_analyst_data("AAPL")
    client.get_analyst_data("AAPL")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.get_analyst_data("AAPL")
    assert calls["n"] == 2


SEARCH_QUOTES: list[dict[str, Any]] = [
    {
        "symbol": "AAPL",
        "longname": "Apple Inc.",
        "shortname": "Apple",
        "quoteType": "EQUITY",
        "exchDisp": "NASDAQ",
        "sectorDisp": "Technology",
        "industryDisp": "Consumer Electronics",
        "score": 12345.6,
    },
    {
        "symbol": "APLE",
        "shortname": "Apple Hospitality REIT",  # only shortname
        "quoteType": "EQUITY",
        "exchDisp": "NYSE",
    },
]


def test_search_symbols_happy_path_maps_fields() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=SEARCH_QUOTES),
    )
    result = client.search_symbols("apple")
    assert isinstance(result, SymbolSearchResult)
    assert result.query == "apple"
    assert [m.symbol for m in result.matches] == ["AAPL", "APLE"]
    first = result.matches[0]
    assert first.name == "Apple Inc." and first.quote_type == "EQUITY"
    assert first.exchange == "NASDAQ" and first.sector == "Technology"
    assert first.industry == "Consumer Electronics" and first.score == 12345.6
    second = result.matches[1]
    assert second.name == "Apple Hospitality REIT"  # longname missing -> shortname
    assert second.sector is None and second.score is None


def test_search_symbols_empty_quotes_returns_empty_no_raise() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=[]),
    )
    result = client.search_symbols("zzzznope")
    assert result.query == "zzzznope" and result.matches == []


def test_search_symbols_typed_error_is_data_unavailable() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(error=YFException("search rate limited")),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "search rate limited" in str(exc.value)


def test_search_symbols_raw_error_is_data_unavailable() -> None:
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(error=RuntimeError("boom")),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "boom" in str(exc.value)


def test_search_symbols_passes_max_results() -> None:
    captured: dict[str, Any] = {}

    def search(query: str, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return SimpleNamespace(quotes=SEARCH_QUOTES)

    client = make_client(factory=fake_ticker_factory(), search_factory=search)
    client.search_symbols("apple", max_results=3)
    assert captured["max_results"] == 3
    assert captured["news_count"] == 0 and captured["lists_count"] == 0


def test_search_symbols_skips_quote_without_symbol() -> None:
    quotes: list[dict[str, Any]] = [
        {"shortname": "No Symbol Co"},
        {"symbol": "AAPL", "longname": "Apple Inc."},
    ]
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=quotes),
    )
    result = client.search_symbols("apple")
    assert [m.symbol for m in result.matches] == ["AAPL"]


def test_search_symbols_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting_search(query: str, **kwargs: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(quotes=SEARCH_QUOTES)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=fake_ticker_factory(),
        search_factory=counting_search,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.search_symbols("apple")
    client.search_symbols("apple")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.search_symbols("apple")
    assert calls["n"] == 2


def test_search_symbols_parse_error_is_data_unavailable() -> None:
    # A quote whose score is a non-coercible object survives mapping until SymbolMatch
    # construction; force a parse failure via a bad value type for a typed field.
    quotes: list[dict[str, Any]] = [{"symbol": "AAPL", "score": object()}]
    client = make_client(
        factory=fake_ticker_factory(),
        search_factory=fake_search_factory(quotes=quotes),
    )
    with pytest.raises(DataUnavailable) as exc:
        client.search_symbols("apple")
    assert "Failed to parse search results for 'apple'" in str(exc.value)


def _news_client(**kw: Any) -> YFinanceClient:
    factory = kw.pop("factory")
    return YFinanceClient(
        ticker_factory=factory,
        time_fn=kw.pop("clock", FakeClock()),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
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
    client = _news_client(factory=fake_ticker_factory(news=NEWS_ITEMS))
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
    client = _news_client(factory=fake_ticker_factory(news=[]))
    result = client.get_news("ZZZZ")
    assert result.symbol == "ZZZZ" and result.articles == []


def test_get_news_typed_error_is_data_unavailable() -> None:
    client = _news_client(factory=fake_ticker_factory(news_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_news("AAPL")
    assert "rate limited" in str(exc.value)
    assert type(exc.value) is DataUnavailable  # never SymbolNotFound


def test_get_news_raw_error_is_data_unavailable() -> None:
    client = _news_client(factory=fake_ticker_factory(news_error=RuntimeError("boom")))
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
    client = _news_client(factory=fake_ticker_factory(news=items))
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
    client = _news_client(factory=fake_ticker_factory(news=[item]))
    [article] = client.get_news("AAPL").articles
    assert article.title == "Title only"
    assert article.publisher is None
    assert article.link is None
    assert article.summary is None  # "" coerced to None
    assert article.published == "2026-05-31T00:00:00Z"


def test_get_news_none_provider_and_canonical_yield_none() -> None:
    # provider/canonicalUrl present but explicitly None (a shape yfinance can return).
    item = make_news_item("Title", publisher=None, link=None)
    client = _news_client(factory=fake_ticker_factory(news=[item]))
    [article] = client.get_news("AAPL").articles
    assert article.publisher is None and article.link is None


def test_get_news_click_through_fallback_link() -> None:
    item = make_news_item(
        "Title",
        published="2026-05-31T00:00:00Z",
        omit_canonical=True,
        click_through="https://fallback.example/x",
    )
    client = _news_client(factory=fake_ticker_factory(news=[item]))
    [article] = client.get_news("AAPL").articles
    assert article.link == "https://fallback.example/x"


def test_get_news_clamps_to_count_and_passes_args_to_source() -> None:
    # The fake returns ALL 3 items regardless of count; the client must clamp to 2.
    factory = fake_ticker_factory(news=NEWS_ITEMS)
    client = _news_client(factory=factory)
    result = client.get_news("AAPL", count=2)
    assert len(result.articles) == 2  # client-side clamp, not the fake
    assert factory.captured_news_count["count"] == 2  # type: ignore[attr-defined]
    assert factory.captured_news_count["tab"] == "news"  # type: ignore[attr-defined]


def test_get_news_parse_error_is_data_unavailable() -> None:
    # A truthy but non-string title survives the title guard yet fails NewsArticle
    # validation (title: str), forcing the parse-stage error path.
    item = {"id": "x", "content": {"title": 123}}
    client = _news_client(factory=fake_ticker_factory(news=[item]))
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


# --- symbol normalization (strip + upper) ---


def test_symbols_are_normalized_before_caching_and_echoed_normalized() -> None:
    """'aapl' and ' AAPL ' name the same instrument, so they must share one cache entry."""
    calls: list[str] = []

    def counting_factory(symbol: str) -> Any:
        calls.append(symbol)
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    client = _client(factory=counting_factory)
    lower = client.get_quote(["aapl"]).quotes
    padded = client.get_quote([" AAPL "]).quotes
    assert calls == ["AAPL"]  # one fetch, with the normalized symbol
    assert [q.symbol for q in lower] == ["AAPL"]
    assert [q.symbol for q in padded] == ["AAPL"]


def test_price_history_normalizes_symbol() -> None:
    df = make_history_df([100.0, 101.0])
    calls: list[str] = []

    def counting_factory(symbol: str) -> Any:
        calls.append(symbol)
        return fake_ticker_factory(history_df=df)(symbol)

    client = _client(factory=counting_factory)
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
        financials={"income_stmt": make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])},
    )
    client = _client(factory=factory)
    assert client.get_company_profile(" aapl ").symbol == "AAPL"
    assert client.get_key_metrics(" aapl ").symbol == "AAPL"
    assert client.get_analyst_data(" aapl ").symbol == "AAPL"
    assert client.get_news(" aapl ").symbol == "AAPL"
    assert client.analyze_performance(" aapl ", "1y").symbol == "AAPL"
    assert client.get_financials(" aapl ", "income", "annual").symbol == "AAPL"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_symbol_raises_symbol_not_found_without_fetching(blank: str) -> None:
    calls: list[str] = []

    def counting_factory(symbol: str) -> Any:
        calls.append(symbol)
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    client = _client(factory=counting_factory)
    with pytest.raises(SymbolNotFound, match="Empty ticker symbol"):
        client.get_company_profile(blank)
    assert calls == []


# --- bounded LRU cache (item 7) ---


def _counting_quote_factory(calls: list[str]) -> Callable[[str], Any]:
    def factory(symbol: str) -> Any:
        calls.append(symbol)
        return fake_ticker_factory(fast_info=QUOTE_FI)(symbol)

    return factory


def test_cache_evicts_least_recently_used_entry_over_max() -> None:
    calls: list[str] = []
    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=_counting_quote_factory(calls),
        time_fn=clock,
        quote_ttl=30.0,
        cache_max_entries=2,
    )
    client.get_quote(["AAA"])
    client.get_quote(["BBB"])
    client.get_quote(["AAA"])  # cache hit -> AAA becomes the most recently USED entry
    client.get_quote(["CCC"])  # inserting a third entry evicts BBB, not AAA
    assert len(client._cache) == 2
    calls.clear()
    client.get_quote(["AAA"])
    assert calls == []  # still cached
    client.get_quote(["BBB"])
    assert calls == ["BBB"]  # was evicted, so refetched


def test_cache_purges_expired_entries_on_insert() -> None:
    calls: list[str] = []
    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=_counting_quote_factory(calls), time_fn=clock, quote_ttl=30.0
    )
    client.get_quote(["AAA"])
    clock.advance(31.0)
    client.get_quote(["BBB"])
    # AAA is past its own TTL, so it is dropped rather than squatting on the bound.
    assert len(client._cache) == 1
    assert ("quote", "BBB") in client._cache


def test_cache_never_exceeds_max_entries() -> None:
    calls: list[str] = []
    client = YFinanceClient(
        ticker_factory=_counting_quote_factory(calls),
        time_fn=FakeClock(),
        quote_ttl=30.0,
        cache_max_entries=3,
    )
    for i in range(20):
        client.get_quote([f"SYM{i}"])
        assert len(client._cache) <= 3


def test_cache_default_max_entries_is_bounded() -> None:
    client = YFinanceClient(ticker_factory=fake_ticker_factory(fast_info=QUOTE_FI))
    assert client._cache_max_entries == DEFAULT_CACHE_MAX_ENTRIES
    assert DEFAULT_CACHE_MAX_ENTRIES > 0


# --- error classification: transport failure vs. missing symbol (item 3) ---


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeHTTPError(Exception):
    """Shaped like the curl_cffi/requests HTTPError yfinance lets escape raise_for_status()."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = _FakeResponse(status_code)


TRANSPORT_ERRORS = [
    ConnectionError("Connection reset by peer"),
    TimeoutError("timed out after 30s"),
    OSError("network is unreachable"),
    _FakeHTTPError("500 Server Error: Internal Server Error for url: ...", 500),
    _FakeHTTPError("503 Server Error: Service Unavailable for url: ...", 503),
    ValueError("Expecting value: line 1 column 1 (char 0)"),
]


@pytest.mark.parametrize("exc", TRANSPORT_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_quote_transport_failure_is_data_unavailable_not_symbol_not_found(exc: Exception) -> None:
    client = _client(factory=fake_ticker_factory(fast_info_error=exc))
    with pytest.raises(DataUnavailable) as raised:
        client._fetch_quote("AAPL")
    assert not isinstance(raised.value, SymbolNotFound)
    assert str(exc) in str(raised.value)  # underlying message preserved
    assert "may be invalid or delisted" not in str(raised.value)


@pytest.mark.parametrize("exc", TRANSPORT_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_info_transport_failure_is_data_unavailable_not_symbol_not_found(exc: Exception) -> None:
    client = _client(factory=fake_ticker_factory(info_error=exc))
    with pytest.raises(DataUnavailable) as raised:
        client.get_company_profile("AAPL")
    assert not isinstance(raised.value, SymbolNotFound)
    assert str(exc) in str(raised.value)


NO_DATA_ERRORS = [
    KeyError("exchangeTimezoneName"),
    YFTickerMissingError("NOPE", "possibly delisted; no price data found"),
    YFTzMissingError("NOPE"),
    YFPricesMissingError("NOPE", ""),
    _FakeHTTPError("404 Client Error: Not Found for url: ...", 404),
    Exception("Quote not found for ticker symbol: NOPE"),
]


@pytest.mark.parametrize("exc", NO_DATA_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_quote_no_data_signals_are_symbol_not_found(exc: Exception) -> None:
    client = _client(factory=fake_ticker_factory(fast_info_error=exc))
    with pytest.raises(SymbolNotFound) as raised:
        client._fetch_quote("NOPE")
    assert str(raised.value) == "No quote data for 'NOPE'. The symbol may be invalid or delisted."


@pytest.mark.parametrize("exc", NO_DATA_ERRORS, ids=lambda e: type(e).__name__ + str(e)[:12])
def test_info_no_data_signals_are_symbol_not_found(exc: Exception) -> None:
    client = _client(factory=fake_ticker_factory(info_error=exc))
    with pytest.raises(SymbolNotFound) as raised:
        client.get_key_metrics("NOPE")
    assert "No metrics data for 'NOPE'" in str(raised.value)


def test_rate_limit_error_stays_data_unavailable() -> None:
    client = _client(factory=fake_ticker_factory(fast_info_error=YFRateLimitError()))
    with pytest.raises(DataUnavailable) as raised:
        client._fetch_quote("AAPL")
    assert not isinstance(raised.value, SymbolNotFound)
    assert "Rate limited" in str(raised.value)


# --- intraday bars keep their time (item 1) ---


def test_intraday_bars_carry_a_full_timestamp_with_utc_offset() -> None:
    df = make_intraday_df([100.0, 101.0, 102.0])
    client = _client(factory=fake_ticker_factory(history_df=df))
    hist = client.get_price_history("AAPL", period="1d", interval="5m")
    assert [b.date for b in hist.bars] == [
        "2026-09-25T09:30:00-04:00",
        "2026-09-25T09:35:00-04:00",
        "2026-09-25T09:40:00-04:00",
    ]
    assert hist.summary.start_date == "2026-09-25T09:30:00-04:00"
    assert hist.summary.end_date == "2026-09-25T09:40:00-04:00"


@pytest.mark.parametrize("interval", ["1m", "5m", "15m", "30m", "1h"])
def test_every_intraday_interval_emits_distinct_timestamps(interval: str) -> None:
    df = make_intraday_df([100.0, 101.0])
    client = _client(factory=fake_ticker_factory(history_df=df))
    dates = [b.date for b in client.get_price_history("AAPL", "1d", interval).bars]
    assert len(set(dates)) == 2
    assert all("T" in d and d.endswith("-04:00") for d in dates)


@pytest.mark.parametrize("interval", ["1d", "1wk", "1mo"])
def test_daily_and_longer_bars_stay_date_only(interval: str) -> None:
    # Yahoo's daily index is midnight in the EXCHANGE's timezone; emitting a timestamp (or
    # converting to UTC) would either lie about the time or shift the calendar date.
    df = make_intraday_df([100.0, 101.0], start="2026-09-24 00:00", freq="1D")
    client = _client(factory=fake_ticker_factory(history_df=df))
    dates = [b.date for b in client.get_price_history("AAPL", "1mo", interval).bars]
    assert dates == ["2026-09-24", "2026-09-25"]


def test_intraday_intervals_are_the_non_daily_history_intervals() -> None:
    # Pins the two sets against HistoryInterval so a newly supported interval cannot
    # silently default to date-only formatting.
    assert set(get_args(HistoryInterval)) - {"1d", "1wk", "1mo"} == _INTRADAY_INTERVALS


# --- currency labelling for cross-currency comparisons (item 2) ---

SAP_INFO = {  # SAP's US listing quotes in USD while it reports its financials in EUR
    "longName": "SAP SE",
    "currency": "USD",
    "financialCurrency": "EUR",
    "enterpriseValue": 3.42e12,
    "totalDebt": 9.94e9,
    "totalCash": 1.16e10,
    "freeCashflow": 9.09e9,
    "ebitda": 1.18e10,
}


def test_financial_statement_is_labelled_with_the_reporting_currency() -> None:
    df = make_financials_df(INCOME, ["2024-12-31", "2023-12-31"])
    client = _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}, info=SAP_INFO))
    assert client.get_financials("SAP", "income", "annual").currency == "EUR"


def test_financial_statement_currency_falls_back_to_quote_currency() -> None:
    df = make_financials_df(INCOME, ["2024-12-31", "2023-12-31"])
    client = _fin_client(
        factory=fake_ticker_factory(
            financials={"income_stmt": df}, info={"longName": "Apple Inc.", "currency": "USD"}
        )
    )
    assert client.get_financials("AAPL", "income", "annual").currency == "USD"


def test_financial_statement_currency_is_none_when_info_is_unusable() -> None:
    df = make_financials_df(INCOME, ["2024-12-31", "2023-12-31"])
    for factory in (
        fake_ticker_factory(financials={"income_stmt": df}, info={}),
        fake_ticker_factory(financials={"income_stmt": df}, info_error=OSError("no network")),
    ):
        client = _fin_client(factory=factory)
        # An unlabelled statement beats a failed one: the values are still correct.
        fs = client.get_financials("AAPL", "income", "annual")
        assert fs.currency is None
        assert fs.line_items["Total Revenue"] == [400.0, 380.0]


def test_key_metrics_carry_both_quote_and_financial_currency() -> None:
    client = _metrics_client(factory=fake_ticker_factory(info=SAP_INFO))
    metrics = client.get_key_metrics("SAP")
    # EBITDA/debt/cash/FCF come from Yahoo's financialData (EUR); EV is derived from
    # market cap and is in the quote currency (USD). One currency field would misreport half.
    assert metrics.currency == "USD"
    assert metrics.financial_currency == "EUR"
    assert metrics.ebitda == 1.18e10
    assert metrics.enterprise_value == 3.42e12


def test_key_metrics_currencies_are_none_when_absent() -> None:
    client = _metrics_client(factory=fake_ticker_factory(info={"longName": "X"}))
    metrics = client.get_key_metrics("X")
    assert metrics.currency is None and metrics.financial_currency is None


# --- unknown line-item labels are surfaced, not silently dropped (item 4) ---


def _income_client() -> YFinanceClient:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    return _fin_client(factory=fake_ticker_factory(financials={"income_stmt": df}))


def test_unknown_line_items_are_reported_with_suggestions() -> None:
    fs = _income_client().get_financials(
        "AAPL", "income", "annual", line_items=["Total Revenue", "Revenue"]
    )
    assert list(fs.line_items) == ["Total Revenue"]
    assert fs.missing_line_items == ["Revenue"]
    assert fs.available_line_items == ["Total Revenue", "Net Income"]
    assert fs.line_item_suggestions["Revenue"] == ["Total Revenue"]


def test_matching_line_items_report_no_misses() -> None:
    fs = _income_client().get_financials("AAPL", "income", "annual", line_items=["Net Income"])
    assert list(fs.line_items) == ["Net Income"]
    assert fs.missing_line_items == []
    # Redundant with line_items when nothing is missing, so it stays empty.
    assert fs.available_line_items == []
    assert fs.line_item_suggestions == {}


def test_line_items_match_case_and_whitespace_insensitively() -> None:
    fs = _income_client().get_financials(
        "AAPL", "income", "annual", line_items=["total revenue", "  Net   Income  "]
    )
    # Keyed by the statement's canonical label, whatever spelling was requested.
    assert list(fs.line_items) == ["Total Revenue", "Net Income"]
    assert fs.missing_line_items == []


def test_repeated_line_items_collapse() -> None:
    fs = _income_client().get_financials(
        "AAPL", "income", "annual", line_items=["Total Revenue", "TOTAL REVENUE", "Nope", "Nope"]
    )
    assert list(fs.line_items) == ["Total Revenue"]
    assert fs.missing_line_items == ["Nope"]


def test_line_item_with_no_close_match_gets_no_suggestion() -> None:
    fs = _income_client().get_financials("AAPL", "income", "annual", line_items=["zzzzzzzz"])
    assert fs.line_items == {}
    assert fs.missing_line_items == ["zzzzzzzz"]
    assert fs.available_line_items == ["Total Revenue", "Net Income"]
    assert "zzzzzzzz" not in fs.line_item_suggestions


def test_unfiltered_statement_reports_no_misses() -> None:
    fs = _income_client().get_financials("AAPL", "income", "annual")
    assert fs.missing_line_items == [] and fs.available_line_items == []


# --- get_quote: concurrent fetch with partial results (item 5) ---


def test_get_quote_returns_partial_results_instead_of_failing_the_batch() -> None:
    client = _client(
        factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI, "MSFT": QUOTE_FI})
    )
    result = client.get_quote(["AAPL", "BADSYM", "MSFT"])
    assert [q.symbol for q in result.quotes] == ["AAPL", "MSFT"]  # request order preserved
    assert [e.symbol for e in result.errors] == ["BADSYM"]
    assert result.errors[0].error == (
        "No quote data for 'BADSYM'. The symbol may be invalid or delisted."
    )


def test_get_quote_error_entry_carries_a_transport_failure_message() -> None:
    client = _client(
        factory=fake_symbol_ticker_factory(
            fast_info={"AAPL": QUOTE_FI}, errors={"MSFT": ConnectionError("connection reset")}
        )
    )
    result = client.get_quote(["AAPL", "MSFT"])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert "connection reset" in result.errors[0].error


def test_get_quote_all_failing_returns_no_quotes_and_all_errors() -> None:
    client = _client(factory=fake_symbol_ticker_factory())
    result = client.get_quote(["NOPE1", "NOPE2"])
    assert result.quotes == []
    assert [e.symbol for e in result.errors] == ["NOPE1", "NOPE2"]


def test_get_quote_deduplicates_equivalent_symbols() -> None:
    calls: list[str] = []
    client = _client(factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI}, calls=calls))
    result = client.get_quote(["AAPL", "aapl", " AAPL "])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert calls == ["AAPL"]


def test_get_quote_blank_symbol_becomes_an_error_entry_not_an_exception() -> None:
    client = _client(factory=fake_symbol_ticker_factory(fast_info={"AAPL": QUOTE_FI}))
    result = client.get_quote(["AAPL", "  "])
    assert [q.symbol for q in result.quotes] == ["AAPL"]
    assert result.errors[0].symbol == "  "
    assert "Empty ticker symbol" in result.errors[0].error


def test_get_quote_empty_list_returns_empty_result() -> None:
    assert _client(factory=fake_symbol_ticker_factory()).get_quote([]) == QuoteResult(
        quotes=[], errors=[]
    )


def test_get_quote_fetches_tickers_concurrently() -> None:
    # Every fetch waits on a 4-party barrier, so this only completes if all four symbols
    # are in flight at once; a sequential fetcher would block until the timeout.
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    gate = threading.Barrier(len(symbols), timeout=10)
    client = _client(
        factory=fake_symbol_ticker_factory(fast_info=dict.fromkeys(symbols, QUOTE_FI), gate=gate)
    )
    result = client.get_quote(symbols)
    assert [q.symbol for q in result.quotes] == symbols
    assert result.errors == []


def test_get_quote_concurrency_is_bounded() -> None:
    assert QUOTE_MAX_WORKERS > 1
    symbols = [f"SYM{i}" for i in range(QUOTE_MAX_WORKERS + 5)]
    client = _client(factory=fake_symbol_ticker_factory(fast_info=dict.fromkeys(symbols, QUOTE_FI)))
    # More tickers than workers still completes: the pool queues the overflow.
    assert len(client.get_quote(symbols).quotes) == len(symbols)


# --- the statement currency is read once per symbol, and failures are not cached ---


def _statement_currency_factory(
    info_reads: list[str], info: dict[str, Any], fail_first: int = 0
) -> Callable[[str], Any]:
    """A ticker factory that records every ``.info`` read and can fail the first N of them."""
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    state = {"failures_left": fail_first}

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def info(self) -> Any:
            info_reads.append(self._symbol)
            if state["failures_left"] > 0:
                state["failures_left"] -= 1
                raise YFRateLimitError()
            return info

        def __getattr__(self, name: str) -> Any:
            return df

    return _Ticker


def test_statement_currency_is_fetched_once_per_symbol() -> None:
    info_reads: list[str] = []
    client = _fin_client(factory=_statement_currency_factory(info_reads, SAP_INFO))
    for statement in ("income", "balance", "cashflow"):
        for period in ("annual", "quarterly"):
            fs = client.get_financials("SAP", statement, period)
            assert fs.currency == "EUR"
    # All six statement/period combinations share one currency, so one .info request.
    assert info_reads == ["SAP"]


def test_failed_statement_currency_read_is_not_cached() -> None:
    info_reads: list[str] = []
    client = _fin_client(factory=_statement_currency_factory(info_reads, SAP_INFO, fail_first=1))
    first = client.get_financials("SAP", "income", "annual")
    assert first.currency is None  # unlabelled beats failing the statement
    # A rate limit says nothing about the reporting currency, so the next call retries
    # instead of asserting "Yahoo does not report it" for the whole fundamentals TTL.
    second = client.get_financials("SAP", "balance", "annual")
    assert second.currency == "EUR"
    assert info_reads == ["SAP", "SAP"]


def test_absent_statement_currency_is_cached() -> None:
    info_reads: list[str] = []
    client = _fin_client(factory=_statement_currency_factory(info_reads, {}))
    assert client.get_financials("X", "income", "annual").currency is None
    assert client.get_financials("X", "balance", "annual").currency is None
    # A genuine "Yahoo reports no currency" IS cacheable - no repeat request.
    assert info_reads == ["X"]


# --- cache entries are timestamped when the fetch completes ---


def test_cache_entry_is_timestamped_after_the_fetch_completes() -> None:
    clock = FakeClock()
    calls: list[str] = []

    def slow_factory(symbol: str) -> Any:
        class _Ticker:
            @property
            def fast_info(self) -> Any:
                calls.append(symbol)
                clock.advance(35.0)  # the fetch itself outlasts the 30s quote TTL
                return SimpleNamespace(**QUOTE_FI)

        return _Ticker()

    client = YFinanceClient(ticker_factory=slow_factory, time_fn=clock, quote_ttl=30.0)
    client.get_quote(["AAPL"])
    client.get_quote(["AAPL"])
    # Timestamping at the start would insert the entry already expired, making the cache
    # a no-op during exactly the slowdown it exists to absorb.
    assert calls == ["AAPL"]


def test_refreshing_a_present_key_makes_it_most_recently_used() -> None:
    client = YFinanceClient(
        ticker_factory=fake_ticker_factory(fast_info=QUOTE_FI),
        time_fn=FakeClock(),
        cache_max_entries=2,
    )

    def racing_fetch() -> str:
        # What concurrent get_quote calls do: another thread inserts this key while this
        # fetch is in flight (so the write below lands on a key already present), and a
        # third key is cached after it.
        client._cache[("k", "A")] = (client._now(), 30.0, "stale")
        client._cache[("k", "B")] = (client._now(), 30.0, "B")
        return "fresh"

    assert client._cached(("k", "A"), 30.0, racing_fetch) == "fresh"
    client._cached(("k", "C"), 30.0, lambda: "C")  # over the bound: evict the LRU entry
    # Refreshing A must make it most recently used; leaving it in the racing thread's
    # older slot would evict the entry that was just written.
    assert client._cache[("k", "A")][2] == "fresh"
    assert ("k", "B") not in client._cache


def test_empty_line_items_filter_returns_the_whole_statement() -> None:
    # An empty filter cannot mean "return nothing useful": that was the one silent-drop
    # case left. Library callers get the full statement; the tool rejects [] outright.
    fs = _income_client().get_financials("AAPL", "income", "annual", line_items=[])
    assert list(fs.line_items) == ["Total Revenue", "Net Income"]
    assert fs.missing_line_items == []
