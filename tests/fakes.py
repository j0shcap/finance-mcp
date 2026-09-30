"""Fakes for yfinance and builders for the frames it returns, shared across tests."""

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pandas as pd

from finance_mcp.data.yfinance_client import _FINANCIALS_ATTR, YFinanceClient

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


def make_history_df(
    closes: list[float], *, start: str = "2024-01-01", freq: str = "D", tz: str | None = None
) -> pd.DataFrame:
    """Build a yfinance-shaped OHLCV frame.

    ``freq`` picks the trading calendar: "D" gives consecutive calendar days (a 24/7
    instrument such as crypto), "B" gives weekdays only (an equity). ``tz`` makes the
    index tz-aware, as yfinance's really is; see ``make_intraday_df``.
    """
    idx = pd.to_datetime(pd.date_range(start, periods=len(closes), freq=freq, tz=tz))
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1000 * (i + 1) for i in range(len(closes))],
        },
        index=idx,
    )


def make_financials_df(rows: dict[str, list[float]], period_ends: list[str]) -> pd.DataFrame:
    """rows = {line_item_label: [values most-recent-first]}; columns are the period-end dates."""
    return pd.DataFrame.from_dict(rows, orient="index", columns=pd.to_datetime(period_ends))


def make_series(dates: list[str], values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.to_datetime(dates), dtype=float)


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
    dividends_error: Exception | None = None,
    splits_error: Exception | None = None,
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
    captured_news_count: dict[str, int | str] = {}
    # Every attribute the client may read a statement from, so the stub cannot drift.
    statement_attrs = frozenset(_FINANCIALS_ATTR.values())

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

        @property
        def dividends(self) -> Any:
            if dividends_error is not None:
                raise dividends_error
            return dividends if dividends is not None else pd.Series(dtype=float)

        @property
        def splits(self) -> Any:
            if splits_error is not None:
                raise splits_error
            return splits if splits is not None else pd.Series(dtype=float)

        @property
        def recommendations(self) -> Any:
            if recommendations_error is not None:
                raise recommendations_error
            return recommendations if recommendations is not None else pd.DataFrame()

        def get_news(self, count: int = 10, tab: str = "news") -> list[dict[str, Any]]:
            if news_error is not None:
                raise news_error
            captured_news_count["count"] = count
            captured_news_count["tab"] = tab
            return news or []

        def __getattr__(self, name: str) -> Any:
            if name in statement_attrs:
                if financials_error is not None:
                    raise financials_error
                return financials_map.get(name, pd.DataFrame())
            raise AttributeError(name)

    def factory(_symbol: str) -> Any:
        return _Ticker()

    factory.captured_news_count = captured_news_count  # type: ignore[attr-defined]
    return factory


def fake_multi_ticker_factory(
    per_symbol: dict[str, dict[str, Any]],
    calls: list[str] | None = None,
    gate: threading.Barrier | None = None,
) -> Callable[[str], Any]:
    """A ticker factory whose stub differs BY SYMBOL, for multi-symbol scenarios.

    ``per_symbol`` maps a symbol to the keyword arguments ``fake_ticker_factory`` would take
    for it (``history_df``, ``info``, ``history_error``, ...), so each leg of a comparison
    can succeed or fail independently. A symbol that is absent raises KeyError on every
    access, which is what yfinance leaks for an unknown symbol. ``calls`` records each
    symbol constructed; ``gate`` is waited on at construction, so a batch only completes
    when the symbols are fetched concurrently.
    """
    stubs = {symbol: fake_ticker_factory(**kwargs) for symbol, kwargs in per_symbol.items()}
    missing = fake_ticker_factory(error=KeyError("exchangeTimezoneName"))

    def factory(symbol: str) -> Any:
        if calls is not None:
            calls.append(symbol)
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


def fake_search_factory(
    quotes: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
    news: list[dict[str, Any]] | None = None,
) -> Callable[[str], Any]:
    """Return a callable that mimics ``yf.Search``.

    The returned callable accepts ``(query, **kwargs)``; if ``error`` is set it
    raises it, otherwise it returns an object whose ``.quotes`` and ``.news``
    attributes are ``quotes or []`` and ``news or []``. Calls are recorded on the
    callable's ``calls`` list so a test can assert the fallback was or was not taken.
    """
    calls: list[dict[str, Any]] = []

    def _search(query: str, **kwargs: Any) -> Any:
        calls.append({"query": query, **kwargs})
        if error is not None:
            raise error
        return SimpleNamespace(quotes=quotes or [], news=news or [])

    _search.calls = calls  # type: ignore[attr-defined]
    return _search


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
    calls: list[str] | None = None,
    gate: threading.Barrier | None = None,
) -> Callable[[str], Any]:
    """A ticker factory whose behaviour varies BY SYMBOL, for partial-batch scenarios.

    ``fast_info`` maps symbol -> that symbol's fast_info dict; ``errors`` maps symbol -> an
    exception raised on ``.fast_info`` access. A symbol in neither raises KeyError, which is
    what yfinance leaks for an unknown symbol. ``calls`` records the symbols fetched.
    ``gate`` is waited on before each fast_info read, so a batch only completes if the
    symbols are fetched concurrently (a sequential fetcher deadlocks the barrier).
    """
    quotes = fast_info or {}
    failures = errors or {}

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def fast_info(self) -> Any:
            if calls is not None:
                calls.append(self._symbol)
            if gate is not None:
                gate.wait()
            if self._symbol in failures:
                raise failures[self._symbol]
            if self._symbol not in quotes:
                raise KeyError("exchangeTimezoneName")
            return SimpleNamespace(**quotes[self._symbol])

    return _Ticker


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
        search_factory=search_factory or fake_search_factory(),
        time_fn=clock if clock is not None else FakeClock(),
        **options,
    )


def counting(factory: Callable[[str], Any]) -> tuple[Callable[[str], Any], list[str]]:
    """Wrap a ticker factory; the returned list records each symbol it is asked for."""
    calls: list[str] = []

    def wrapped(symbol: str) -> Any:
        calls.append(symbol)
        return factory(symbol)

    return wrapped, calls


INCOME = {  # rows: label -> [most-recent, prior]
    "Total Revenue": [400.0, 380.0],
    "Net Income": [100.0, float("nan")],
}


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
