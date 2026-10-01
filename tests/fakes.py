"""Fakes for yfinance and builders for the frames it returns, shared across tests."""

import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pandas as pd
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from finance_mcp.data.yahoo import FINANCIALS_ATTR
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.server import create_server

#: fast_info for a healthy quote.
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


class FakeClock:
    """Mutable monotonic clock for deterministic TTL tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


#: The exchange timezone fake histories and corporate actions carry, as yfinance's do.
EXCHANGE_TZ = "America/New_York"


def make_history_df(
    closes: list[float], *, start: str = "2024-01-01", freq: str = "D", tz: str = EXCHANGE_TZ
) -> pd.DataFrame:
    """Build an OHLCV frame shaped like yfinance's (tests/shapes/yahoo.json).

    ``freq`` picks the trading calendar: "D" gives consecutive calendar days (a 24/7
    instrument such as crypto), "B" gives weekdays only (an equity). Like yfinance's, the
    index is tz-aware at second resolution and the frame carries the corporate-action
    columns.
    """
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz=tz).as_unit("s")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000 * (i + 1) for i in range(len(closes))],
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=idx,
    )


def make_financials_df(rows: dict[str, list[float]], period_ends: list[str]) -> pd.DataFrame:
    """rows = {line_item_label: [values most-recent-first]}; columns are the period-end dates."""
    columns = pd.DatetimeIndex(pd.to_datetime(period_ends)).as_unit("s")
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def make_series(dates: list[str], values: list[float]) -> pd.Series:
    """A dividends or splits series: yfinance stamps each at 09:30 exchange time."""
    index = pd.DatetimeIndex(pd.to_datetime(dates) + pd.Timedelta(hours=9, minutes=30))
    return pd.Series(values, index=index.tz_localize(EXCHANGE_TZ).as_unit("s"), dtype=float)


class FakeHTTPError(Exception):
    """What yfinance raises for a symbol Yahoo doesn't know: an HTTP error carrying a 404."""

    def __init__(self, status_code: int = 404) -> None:
        super().__init__(f"HTTP Error {status_code}: ")
        self.response = SimpleNamespace(status_code=status_code)


def fake_ticker_factory(
    fast_info: dict[str, Any] | None = None,
    history_df: pd.DataFrame | None = None,
    error: Exception | None = None,
    fast_info_error: Exception | None = None,
    history_error: Exception | None = None,
    financials: dict[str, Any] | None = None,
    financials_error: Exception | None = None,
    info: dict[str, Any] | None = None,
    dividends: pd.Series | None = None,
    splits: pd.Series | None = None,
    info_error: Exception | None = None,
    recommendations: pd.DataFrame | None = None,
    news: list[dict[str, Any]] | None = None,
    news_error: Exception | None = None,
    actions_error: Exception | None = None,
    recommendations_error: Exception | None = None,
) -> Callable[[str], Any]:
    """Build a ticker factory returning a stub Ticker for any symbol.

    ``error`` makes BOTH ``.fast_info`` access and ``.history()`` raise it.
    ``fast_info_error`` makes only ``.fast_info`` access raise; ``history_error`` makes
    only ``.history()`` raise. These compose so combined scenarios can be expressed.

    ``financials`` maps a yfinance financials attribute name (e.g. ``"income_stmt"``,
    ``"quarterly_balance_sheet"``) to the DataFrame the stub returns for that attribute.
    ``financials_error`` makes access to any of those financials attributes raise it.
    """
    fi_exc = fast_info_error or error
    hist_exc = history_error or error
    financials_map = financials or {}
    # The ``count`` and ``tab`` the last get_news call received.
    captured_news_call: dict[str, int | str] = {}
    # The (period, interval) of every dividends-and-splits read, in order.
    captured_actions_calls: list[tuple[str, str]] = []
    # Every attribute the client may read a statement from, so the stub cannot drift.
    statement_attrs = frozenset(FINANCIALS_ATTR.values())

    def history_cache(period: str, interval: str) -> dict[str, Any]:
        # yfinance's private price-history cache, which corporate_actions reads.
        if actions_error is not None:
            raise actions_error
        captured_actions_calls.append((period, interval))
        empty = pd.Series(dtype=float)
        return {
            "dividends": dividends if dividends is not None else empty,
            "splits": splits if splits is not None else empty,
        }

    class _Ticker:
        @property
        def fast_info(self) -> Any:
            if fi_exc is not None:
                raise fi_exc
            return SimpleNamespace(**(fast_info or {}))

        def history(self, **_kwargs: Any) -> Any:
            if hist_exc is not None:
                raise hist_exc
            return history_df if history_df is not None else pd.DataFrame()

        @property
        def info(self) -> Any:
            if info_error is not None:
                raise info_error
            return info if info is not None else {}

        def _lazy_load_price_history(self) -> Any:
            return SimpleNamespace(_get_history_cache=history_cache)

        @property
        def recommendations(self) -> Any:
            if recommendations_error is not None:
                raise recommendations_error
            return recommendations if recommendations is not None else pd.DataFrame()

        def get_news(self, count: int = 10, tab: str = "news") -> list[dict[str, Any]]:
            if news_error is not None:
                raise news_error
            captured_news_call["count"] = count
            captured_news_call["tab"] = tab
            return news or []

        def __getattr__(self, name: str) -> Any:
            if name in statement_attrs:
                if financials_error is not None:
                    raise financials_error
                return financials_map.get(name, pd.DataFrame())
            raise AttributeError(name)

    def factory(_symbol: str) -> Any:
        return _Ticker()

    factory.captured_news_call = captured_news_call  # type: ignore[attr-defined]
    factory.captured_actions_calls = captured_actions_calls  # type: ignore[attr-defined]
    return factory


def fake_multi_ticker_factory(
    per_symbol: dict[str, dict[str, Any]],
    gate: threading.Barrier | None = None,
) -> Callable[[str], Any]:
    """A ticker factory whose stub differs BY SYMBOL, for multi-symbol scenarios.

    ``per_symbol`` maps a symbol to the keyword arguments ``fake_ticker_factory`` would take
    for it (``history_df``, ``info``, ``history_error``, ...), so each leg of a comparison
    can succeed or fail independently. A symbol that is absent raises KeyError on every
    access, which is what yfinance leaks for an unknown symbol. ``gate`` is waited on at
    construction, so a batch only completes when the symbols are fetched concurrently.
    """
    stubs = {symbol: fake_ticker_factory(**kwargs) for symbol, kwargs in per_symbol.items()}
    missing = fake_ticker_factory(error=FakeHTTPError(404))

    def factory(symbol: str) -> Any:
        if gate is not None:
            gate.wait()
        return stubs.get(symbol, missing)(symbol)

    return factory


def make_recommendations_df(
    rows: list[tuple[str, int, int, int, int, int]],
) -> pd.DataFrame:
    """Build a yfinance-shaped recommendations frame.

    Each row is (period, strongBuy, buy, hold, sell, strongSell).
    """
    return pd.DataFrame(rows, columns=["period", "strongBuy", "buy", "hold", "sell", "strongSell"])


class FakeSearch:
    """Stands in for ``yf.Search``; ``calls`` records each call's query and keyword args."""

    def __init__(
        self,
        quotes: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        news: list[dict[str, Any]] | None = None,
    ) -> None:
        self.quotes = quotes or []
        self.news = news or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, query: str, **kwargs: Any) -> Any:
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(quotes=self.quotes, news=self.news)


def make_search_news_item(
    title: str | None,
    publisher: str | None = None,
    link: str | None = None,
    published: int | None = None,
) -> dict[str, Any]:
    """Build a news item in the FLAT shape ``yf.Search().news`` returns.

    Unlike ``make_news_item``'s nested ``content`` payload, the search endpoint returns
    top-level keys and a unix ``providerPublishTime``, and carries no summary at all.
    """
    return {
        "uuid": "uuid-" + (title or "none"),
        "title": title,
        "publisher": publisher,
        "link": link,
        "providerPublishTime": published,
    }


def make_news_item(
    title: str | None,
    publisher: str | None = None,
    link: str | None = None,
    published: str | None = None,
    summary: str | None = None,
    *,
    click_through: str | None = None,
    omit_provider: bool = False,
    omit_canonical: bool = False,
) -> dict[str, Any]:
    """Build a yfinance-shaped news item: ``{"id": ..., "content": {...}}``.

    Mirrors the nested shape ``Ticker.get_news`` returns. ``omit_provider`` /
    ``omit_canonical`` drop those nested keys entirely (to exercise the ``None``
    guards); ``click_through`` provides the fallback link source.
    """
    content: dict[str, Any] = {"title": title, "summary": summary, "pubDate": published}
    if not omit_provider:
        content["provider"] = {"displayName": publisher} if publisher is not None else None
    if not omit_canonical:
        content["canonicalUrl"] = {"url": link} if link is not None else None
    if click_through is not None:
        content["clickThroughUrl"] = {"url": click_through}
    return {"id": "id-" + (title or "none"), "content": content}


def make_intraday_df(
    closes: list[float],
    *,
    start: str = "2026-09-25 09:30",
    freq: str = "5min",
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """An intraday OHLCV frame: a tz-aware index at sub-daily bars, as yfinance returns."""
    return make_history_df(closes, start=start, freq=freq, tz=tz)


def fake_symbol_ticker_factory(
    fast_info: dict[str, dict[str, Any]] | None = None,
    errors: dict[str, Exception] | None = None,
    gate: threading.Barrier | None = None,
) -> Callable[[str], Any]:
    """Shorthand for fake_multi_ticker_factory when only quotes matter.

    ``fast_info`` maps symbol -> that symbol's fast_info dict; ``errors`` maps symbol -> an
    exception raised on ``.fast_info`` access. Any other symbol raises KeyError, as for an
    unknown symbol.
    """
    per_symbol: dict[str, dict[str, Any]] = {
        symbol: {"fast_info": quote} for symbol, quote in (fast_info or {}).items()
    }
    per_symbol.update({symbol: {"fast_info_error": exc} for symbol, exc in (errors or {}).items()})
    return fake_multi_ticker_factory(per_symbol, gate=gate)


def make_client(
    factory: Callable[[str], Any] | None = None,
    *,
    clock: FakeClock | None = None,
    search_factory: Callable[..., Any] | None = None,
    **options: Any,
) -> YFinanceClient:
    """A YFinanceClient wired to fakes, so no test reaches Yahoo.

    ``factory`` defaults to a stub with a healthy quote and ``search_factory`` to an empty
    search (the news fallback calls it). ``options`` pass through, e.g. ``quote_ttl``.
    """
    return YFinanceClient(
        ticker_factory=factory or fake_ticker_factory(fast_info=QUOTE_FI),
        search_factory=search_factory or FakeSearch(),
        time_fn=clock if clock is not None else FakeClock(),
        **options,
    )


@asynccontextmanager
async def connect(
    factory: Callable[[str], Any] | None = None, **options: Any
) -> AsyncIterator[Client[FastMCPTransport]]:
    """An in-memory MCP client on a server whose data layer is ``make_client(factory, ...)``."""
    async with Client(create_server(yf_client=make_client(factory, **options))) as client:
        yield client


def counting(factory: Callable[[str], Any]) -> tuple[Callable[[str], Any], list[str]]:
    """Wrap a ticker factory; the returned list records each symbol it is asked for."""
    calls: list[str] = []

    def wrapped(symbol: str) -> Any:
        calls.append(symbol)
        return factory(symbol)

    return wrapped, calls


INCOME_WITH_NAN = {  # rows: label -> [most-recent, prior]
    "Total Revenue": [400.0, 380.0],
    "Net Income": [100.0, float("nan")],
}


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
