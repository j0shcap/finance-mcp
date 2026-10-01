"""Yahoo Finance access through yfinance: every fetch and parse, nothing cached.

Network access is isolated here. yfinance errors and empty results become
DataUnavailable/SymbolNotFound whose message is surfaced to the caller verbatim, because we
cannot enumerate every Yahoo failure mode. SymbolNotFound is reserved for signals that
really mean "no data for this symbol" (see _is_no_data_error): everything else, transport
failures included, stays a plain DataUnavailable.

YahooSource takes symbols already normalized by its caller and never caches: caching and
the choice of what to fetch together belong to YFinanceClient.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import yfinance as yf
from yfinance.exceptions import YFTickerMissingError

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    DividendEvent,
    FinancialStatement,
    KeyMetrics,
    NewsArticle,
    NewsSource,
    PriceBar,
    Quote,
    RecommendationPeriod,
    SplitEvent,
    Statement,
    StatementPeriod,
    SymbolMatch,
    SymbolSearchResult,
)
from finance_mcp.data.relevance import Identity

#: Signals that genuinely mean "Yahoo has no data for this symbol". KeyError is what
#: fast_info leaks for an unknown symbol; YFTickerMissingError covers yfinance's own
#: missing-ticker/timezone/prices errors (YFTzMissingError and YFPricesMissingError
#: subclass it). Anything outside this set is treated as a source/transport failure.
_NO_DATA_ERRORS = (KeyError, YFTickerMissingError)
#: Intervals whose bars are points in time rather than whole sessions. Kept in sync with
#: HistoryInterval (a test pins it): everything that is not a daily-or-longer interval.
INTRADAY_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h"})
#: The yfinance Ticker attribute holding each (statement, period).
FINANCIALS_ATTR = {
    ("income", "annual"): "income_stmt",
    ("income", "quarterly"): "quarterly_income_stmt",
    ("balance", "annual"): "balance_sheet",
    ("balance", "quarterly"): "quarterly_balance_sheet",
    ("cashflow", "annual"): "cashflow",
    ("cashflow", "quarterly"): "quarterly_cashflow",
}


class YahooSource:
    """Fetches and parses one kind of Yahoo data per method, into this package's models."""

    def __init__(
        self,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        search_factory: Callable[[str], Any] = yf.Search,
    ) -> None:
        self._ticker = ticker_factory
        # Widened so the keyword arguments yf.Search is called with typecheck.
        self._search: Callable[..., Any] = search_factory
        # By default yfinance's price and statement fetches swallow a transport failure and
        # return an empty frame, which reads exactly like an unknown symbol; every
        # classification here needs failures as exceptions. Process-wide, but this server
        # is the only yfinance user in its process.
        yf.config.debug.hide_exceptions = False

    def _ticker_with_info(
        self, symbol: str, fetch_label: str, kind: str
    ) -> tuple[Any, dict[str, Any]]:
        """Fetch a ticker and its ``.info``, asserting the symbol names a real instrument.

        Access errors are classified by _data_error; an ``info`` dict that is empty or has
        no longName/shortName (Yahoo's tell for an unknown symbol) becomes SymbolNotFound.
        """
        ticker = self._ticker(symbol)
        try:
            info = ticker.info
        except Exception as exc:
            raise _data_error(exc, fetch_label, kind, symbol) from exc
        if not info or not (info.get("longName") or info.get("shortName")):
            raise SymbolNotFound(_no_data_msg(kind, symbol))
        return ticker, info

    def quote(self, symbol: str) -> Quote:
        try:
            fi = self._ticker(symbol).fast_info
            price = _opt(getattr(fi, "last_price", None))
            prev = _opt(getattr(fi, "previous_close", None))
            currency = getattr(fi, "currency", None)
            day_high = _opt(getattr(fi, "day_high", None))
            day_low = _opt(getattr(fi, "day_low", None))
            year_high = _opt(getattr(fi, "year_high", None))
            year_low = _opt(getattr(fi, "year_low", None))
            market_cap = _opt(getattr(fi, "market_cap", None))
            volume = _opt(getattr(fi, "last_volume", None))
        except Exception as exc:
            raise _data_error(exc, "quote", "quote", symbol) from exc
        if price is None:
            raise SymbolNotFound(_no_data_msg("quote", symbol))
        change = (price - prev) if prev is not None else None
        change_pct = (change / prev * 100.0) if (change is not None and prev) else None
        return Quote(
            symbol=symbol,
            currency=currency,
            price=price,
            previous_close=prev,
            change=change,
            change_percent=change_pct,
            day_high=day_high,
            day_low=day_low,
            year_high=year_high,
            year_low=year_low,
            market_cap=market_cap,
            volume=volume,
        )

    def bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]:
        """Fetch and parse the FULL (untruncated) OHLCV bars, dropping non-finite rows."""
        intraday = interval in INTRADAY_INTERVALS
        no_history = f"No price history for '{symbol}'. Check the symbol/period/interval."
        try:
            df = self._ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:
            # Unhidden, an unknown symbol raises (a 404, YFPricesMissingError) rather than
            # returning an empty frame; the no-data signals keep the history wording.
            if _is_no_data_error(exc):
                raise SymbolNotFound(no_history) from exc
            raise DataUnavailable(f"Failed to fetch history for '{symbol}': {exc}") from exc
        if df is None or df.empty:
            raise SymbolNotFound(no_history)
        all_bars: list[PriceBar] = []
        with _unavailable_on_error(f"Failed to parse history for '{symbol}'"):
            columns = [df[name] for name in ("Open", "High", "Low", "Close", "Volume")]
            for idx, *values in zip(df.index, *columns, strict=True):
                o, h, low, c, v = (float(x) for x in values)
                if any(not math.isfinite(x) for x in (o, h, low, c, v)):
                    continue
                all_bars.append(
                    PriceBar(
                        date=_bar_date(idx, intraday), open=o, high=h, low=low, close=c, volume=v
                    )
                )
        if not all_bars:
            raise SymbolNotFound(no_history)
        return all_bars

    def financial_statement(
        self, symbol: str, statement: Statement, period: StatementPeriod
    ) -> FinancialStatement:
        """The parsed statement, without its currency (that comes from another endpoint)."""
        attr = FINANCIALS_ATTR[(statement, period)]
        ticker = self._ticker(symbol)
        with _unavailable_on_error(f"Failed to fetch {statement} statement for '{symbol}'"):
            df = getattr(ticker, attr)
        if df is None or df.empty:
            raise SymbolNotFound(
                f"No {statement} statement available for '{symbol}'. It may be an ETF, index, or "
                "other instrument without financial statements, or an invalid symbol."
            )
        with _unavailable_on_error(f"Failed to parse {statement} statement for '{symbol}'"):
            period_ends = [col.date().isoformat() for col in df.columns]
            line_items: dict[str, list[float | None]] = {
                str(idx): [_opt(v) for v in values] for idx, *values in df.itertuples(name=None)
            }
            return FinancialStatement(
                symbol=symbol,
                statement=statement,
                period=period,
                currency=None,
                period_ends=period_ends,
                line_items=line_items,
            )

    def profile(self, symbol: str) -> CompanyProfile:
        ticker, info = self._ticker_with_info(symbol, "profile", "profile")
        # Also covers the dividends/splits reads, which are separate Yahoo requests.
        with _unavailable_on_error(f"Failed to parse profile for '{symbol}'"):
            return CompanyProfile(
                symbol=symbol,
                name=info.get("longName") or info.get("shortName"),
                sector=info.get("sector"),
                industry=info.get("industry"),
                country=info.get("country"),
                website=info.get("website"),
                employees=_opt_int(info.get("fullTimeEmployees")),
                summary=info.get("longBusinessSummary"),
                currency=info.get("currency"),
                market_cap=_opt(info.get("marketCap")),
                trailing_pe=_opt(info.get("trailingPE")),
                forward_pe=_opt(info.get("forwardPE")),
                dividend_yield=_opt(info.get("dividendYield")),
                beta=_opt(info.get("beta")),
                recent_dividends=_dividend_events(ticker.dividends, limit=8),
                splits=_split_events(ticker.splits),
            )

    def key_metrics(self, symbol: str) -> KeyMetrics:
        _, info = self._ticker_with_info(symbol, "metrics", "metrics")
        with _unavailable_on_error(f"Failed to parse metrics for '{symbol}'"):
            return KeyMetrics(
                symbol=symbol,
                currency=info.get("currency"),
                financial_currency=info.get("financialCurrency"),
                trailing_pe=_opt(info.get("trailingPE")),
                forward_pe=_opt(info.get("forwardPE")),
                price_to_book=_opt(info.get("priceToBook")),
                price_to_sales=_opt(info.get("priceToSalesTrailing12Months")),
                peg_ratio=_opt(info.get("pegRatio")),
                enterprise_value=_opt(info.get("enterpriseValue")),
                ev_to_ebitda=_opt(info.get("enterpriseToEbitda")),
                ev_to_revenue=_opt(info.get("enterpriseToRevenue")),
                return_on_equity=_opt(info.get("returnOnEquity")),
                return_on_assets=_opt(info.get("returnOnAssets")),
                gross_margins=_opt(info.get("grossMargins")),
                operating_margins=_opt(info.get("operatingMargins")),
                profit_margins=_opt(info.get("profitMargins")),
                ebitda_margins=_opt(info.get("ebitdaMargins")),
                debt_to_equity=_opt(info.get("debtToEquity")),
                current_ratio=_opt(info.get("currentRatio")),
                quick_ratio=_opt(info.get("quickRatio")),
                total_debt=_opt(info.get("totalDebt")),
                total_cash=_opt(info.get("totalCash")),
                free_cashflow=_opt(info.get("freeCashflow")),
                ebitda=_opt(info.get("ebitda")),
                trailing_eps=_opt(info.get("trailingEps")),
                forward_eps=_opt(info.get("forwardEps")),
                revenue_per_share=_opt(info.get("revenuePerShare")),
                book_value=_opt(info.get("bookValue")),
            )

    def analyst_data(self, symbol: str) -> AnalystData:
        ticker, info = self._ticker_with_info(symbol, "analyst data", "analyst")
        parse_failed = f"Failed to parse analyst data for '{symbol}'"
        with _unavailable_on_error(parse_failed):
            mean = _opt(info.get("recommendationMean"))
            analysts = _opt_int(info.get("numberOfAnalystOpinions"))
            target_mean = _opt(info.get("targetMeanPrice"))
            target_median = _opt(info.get("targetMedianPrice"))
            target_high = _opt(info.get("targetHighPrice"))
            target_low = _opt(info.get("targetLowPrice"))
        targets = (target_mean, target_median, target_high, target_low)
        if mean is None and analysts is None and all(t is None for t in targets):
            raise DataUnavailable(
                f"No analyst coverage for '{symbol}'. It may be an ETF, index, or other "
                "instrument without sell-side analyst data."
            )
        # Also covers the recommendations read, which is a separate Yahoo request.
        with _unavailable_on_error(parse_failed):
            return AnalystData(
                symbol=symbol,
                currency=info.get("currency"),
                current_price=_opt(info.get("currentPrice")),
                recommendation_key=info.get("recommendationKey"),
                recommendation_mean=mean,
                number_of_analysts=analysts,
                target_mean_price=target_mean,
                target_median_price=target_median,
                target_high_price=target_high,
                target_low_price=target_low,
                recommendation_trend=_recommendation_trend(ticker.recommendations),
            )

    def news(self, symbol: str, count: int) -> tuple[list[NewsArticle], NewsSource]:
        with _unavailable_on_error(f"Failed to fetch news for '{symbol}'"):
            items = self._ticker(symbol).get_news(count=count, tab="news")
        if not items:
            # Ambiguous: yfinance turns a 500 from the news endpoint into an empty list.
            # Cross-check search rather than report "no recent news", which the model
            # would read as "no catalysts".
            return self._search_news(symbol, count), "search"
        with _unavailable_on_error(f"Failed to parse news for '{symbol}'"):
            articles = [a for a in (_news_article(it) for it in items) if a is not None][:count]
        return articles, "ticker"

    def identity(self, symbol: str) -> Identity:
        _, info = self._ticker_with_info(symbol, "company identity", "identity")
        return Identity(
            quote_type=info.get("quoteType"),
            long_name=info.get("longName"),
            short_name=info.get("shortName"),
        )

    def _search_news(self, symbol: str, count: int) -> list[NewsArticle]:
        """News for ``symbol`` from the search endpoint, or none if it cannot supply any.

        A failure is swallowed deliberately: the ticker stream already answered "no news",
        so raising would turn a symbol with genuinely no coverage into an error.
        """
        try:
            found = self._search(symbol, max_results=1, news_count=count, lists_count=0).news
        except Exception:
            return []
        articles = (_search_news_article(item) for item in found or [])
        return [a for a in articles if a is not None][:count]

    def search(self, query: str, max_results: int) -> SymbolSearchResult:
        with _unavailable_on_error(f"Search failed for '{query}'"):
            result = self._search(query, max_results=max_results, news_count=0, lists_count=0)
            quotes = result.quotes
        if not quotes:
            return SymbolSearchResult(query=query, matches=[])
        with _unavailable_on_error(f"Failed to parse search results for '{query}'"):
            matches = [_symbol_match(q) for q in quotes if q.get("symbol")]
            return SymbolSearchResult(query=query, matches=matches)

    def statement_currency(self, symbol: str) -> str | None:
        """The currency a symbol's statements are reported in, from ``.info``."""
        return _read_statement_currency(self._ticker(symbol))


def _read_statement_currency(ticker: Any) -> str | None:
    """financialCurrency, else the quote currency; None when Yahoo reports neither."""
    info = ticker.info or {}
    currency = info.get("financialCurrency") or info.get("currency")
    return str(currency) if currency else None


def _bar_date(idx: Any, intraday: bool) -> str:
    """Format a bar's index value.

    Intraday bars keep the clock time and the exchange's UTC offset. Daily and longer bars
    are sessions indexed at exchange-local midnight, so they stay date-only: a timestamp
    would imply a trade time, and converting it to UTC would shift the calendar date.
    """
    return str(idx.isoformat() if intraday else idx.date().isoformat())


def _is_no_data_error(exc: Exception) -> bool:
    """True only for "this symbol has no data" signals — never for transport failures."""
    if isinstance(exc, _NO_DATA_ERRORS):
        return True
    # yfinance calls response.raise_for_status(), so Yahoo's 404 for an unknown symbol
    # escapes as a raw HTTP error from its HTTP client rather than a YFException.
    if getattr(getattr(exc, "response", None), "status_code", None) == 404:
        return True
    return "quote not found" in str(exc).lower()


def _data_error(exc: Exception, fetch_label: str, kind: str, symbol: str) -> DataUnavailable:
    """Classify a raw fetch failure as a missing symbol or an unavailable source.

    Only the no-data signals become SymbolNotFound: a transport failure, rate limit or
    malformed payload says nothing about the symbol.
    """
    if _is_no_data_error(exc):
        return SymbolNotFound(_no_data_msg(kind, symbol))
    return DataUnavailable(f"Failed to fetch {fetch_label} for '{symbol}': {exc}")


@contextmanager
def _unavailable_on_error(message: str) -> Iterator[None]:
    """Re-raise any failure in the block as DataUnavailable("<message>: <error>")."""
    try:
        yield
    except Exception as exc:
        raise DataUnavailable(f"{message}: {exc}") from exc


def _no_data_msg(kind: str, symbol: str) -> str:
    """The message for a symbol with no usable data."""
    return f"No {kind} data for '{symbol}'. The symbol may be invalid or delisted."


def _recommendation_trend(df: Any) -> list[RecommendationPeriod]:
    if df is None or len(df) == 0:
        return []
    return [
        RecommendationPeriod(
            period=str(row.get("period", "")),
            strong_buy=_opt_int(row.get("strongBuy")) or 0,
            buy=_opt_int(row.get("buy")) or 0,
            hold=_opt_int(row.get("hold")) or 0,
            sell=_opt_int(row.get("sell")) or 0,
            strong_sell=_opt_int(row.get("strongSell")) or 0,
        )
        for row in df.to_dict("records")
    ]


def _symbol_match(q: dict[str, Any]) -> SymbolMatch:
    return SymbolMatch(
        symbol=str(q["symbol"]),
        name=q.get("longname") or q.get("shortname"),
        quote_type=q.get("quoteType"),
        exchange=q.get("exchDisp"),
        sector=q.get("sectorDisp"),
        industry=q.get("industryDisp"),
        score=_opt(q.get("score")),
    )


def _search_news_article(item: dict[str, Any]) -> NewsArticle | None:
    """Parse the flat news shape the search endpoint returns.

    Unlike :func:`_news_article`'s, the fields sit at the top level, the timestamp is unix
    seconds, and there is no summary (reported as null, not filled from the title).
    """
    title = item.get("title")
    if not title:
        return None
    published = item.get("providerPublishTime")
    return NewsArticle(
        title=title,
        publisher=item.get("publisher"),
        link=item.get("link"),
        published=datetime.fromtimestamp(published, tz=UTC).isoformat() if published else None,
        summary=None,
    )


def _news_article(item: dict[str, Any]) -> NewsArticle | None:
    content = item.get("content") or {}
    title = content.get("title")
    if not title:
        return None
    return NewsArticle(
        title=title,
        publisher=(content.get("provider") or {}).get("displayName"),
        link=(content.get("canonicalUrl") or {}).get("url")
        or (content.get("clickThroughUrl") or {}).get("url"),
        published=content.get("pubDate"),
        summary=content.get("summary") or None,
    )


def _dividend_events(series: Any, limit: int) -> list[DividendEvent]:
    if series is None or len(series) == 0:
        return []
    events: list[DividendEvent] = []
    for ts, value in series.tail(limit).items():
        amount = _opt(value)
        if amount is not None:
            events.append(DividendEvent(date=ts.date().isoformat(), amount=amount))
    return events


def _split_events(series: Any) -> list[SplitEvent]:
    if series is None or len(series) == 0:
        return []
    events: list[SplitEvent] = []
    for ts, value in series.items():
        ratio = _opt(value)
        if ratio is not None:
            events.append(SplitEvent(date=ts.date().isoformat(), ratio=ratio))
    return events


def _opt(value: Any) -> float | None:
    if value is None:
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _opt_int(value: Any) -> int | None:
    f = _opt(value)
    return int(f) if f is not None else None
