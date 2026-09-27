"""Cached, error-surfacing wrapper over yfinance. No MCP imports.

Network access is isolated here. yfinance errors and empty results become
DataUnavailable/SymbolNotFound whose message is surfaced to the caller verbatim,
because we cannot enumerate every Yahoo failure mode.
"""

import math
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import yfinance as yf
from yfinance.exceptions import YFException

from finance_mcp.data import analytics
from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    DividendEvent,
    FinancialStatement,
    KeyMetrics,
    NewsArticle,
    NewsResult,
    PerformanceStats,
    PriceBar,
    PriceHistory,
    PriceSummary,
    Quote,
    RecommendationPeriod,
    SplitEvent,
    Statement,
    StatementPeriod,
    SymbolMatch,
    SymbolSearchResult,
)

DEFAULT_MAX_BARS = 260
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
# Below roughly a quarter of calendar time, annualizing compounds short-run noise into a
# yearly figure that reads as a forecast (a 4-day AAPL move once reported as +47.3%/yr).
MIN_ANNUALIZATION_DAYS = 90

_FINANCIALS_ATTR = {
    ("income", "annual"): "income_stmt",
    ("income", "quarterly"): "quarterly_income_stmt",
    ("balance", "annual"): "balance_sheet",
    ("balance", "quarterly"): "quarterly_balance_sheet",
    ("cashflow", "annual"): "cashflow",
    ("cashflow", "quarterly"): "quarterly_cashflow",
}


class YFinanceClient:
    """Thin yfinance facade with a per-key TTL cache."""

    def __init__(
        self,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        search_factory: Callable[[str], Any] = yf.Search,
        time_fn: Callable[[], float] = time.monotonic,
        quote_ttl: float = 30.0,
        history_ttl: float = 300.0,
        fundamentals_ttl: float = 3600.0,
        max_bars: int = DEFAULT_MAX_BARS,
    ) -> None:
        self._ticker = ticker_factory
        # yf.Search is called with keyword args (max_results/news_count/lists_count);
        # widen to Callable[..., Any] so those kwargs typecheck.
        self._search: Callable[..., Any] = search_factory
        self._now = time_fn
        self._quote_ttl = quote_ttl
        self._history_ttl = history_ttl
        self._fundamentals_ttl = fundamentals_ttl
        self._max_bars = max_bars
        self._cache: dict[tuple[str, ...], tuple[float, Any]] = {}

    def _cached[T](self, key: tuple[str, ...], ttl: float, fetch: Callable[[], T]) -> T:
        hit = self._cache.get(key)
        now = self._now()
        if hit is not None and now - hit[0] < ttl:
            # The cache is heterogeneous (Any value); the key space guarantees each key
            # always maps to the same T, so this single cast is the only one needed.
            return cast(T, hit[1])
        value = fetch()
        self._cache[key] = (now, value)
        return value

    def _ticker_with_info(
        self, symbol: str, fetch_label: str, kind: str
    ) -> tuple[Any, dict[str, Any]]:
        """Fetch a ticker and its ``.info``, asserting the symbol names a real instrument.

        Shared by the profile/metrics/analyst fetchers. A yfinance-typed error becomes
        DataUnavailable; any other access error, or an ``info`` dict with no longName/
        shortName (Yahoo's tell for an unknown symbol), becomes SymbolNotFound.
        """
        ticker = self._ticker(symbol)
        try:
            info = ticker.info
        except YFException as exc:
            raise DataUnavailable(f"Failed to fetch {fetch_label} for '{symbol}': {exc}") from exc
        except Exception as exc:  # raw leak for symbols with no data
            raise SymbolNotFound(_no_data_msg(kind, symbol)) from exc
        if not info or not (info.get("longName") or info.get("shortName")):
            raise SymbolNotFound(_no_data_msg(kind, symbol))
        return ticker, info

    def get_quote(self, symbols: list[str]) -> list[Quote]:
        def fetch(sym: str) -> Callable[[], Quote]:
            return lambda: self._fetch_quote(sym)

        norm = [_norm(s) for s in symbols]
        return [self._cached(("quote", s), self._quote_ttl, fetch(s)) for s in norm]

    def _fetch_quote(self, symbol: str) -> Quote:
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
        except YFException as exc:
            # yfinance's own typed errors (rate limit, etc.) — surface verbatim.
            raise DataUnavailable(f"Failed to fetch quote for '{symbol}': {exc}") from exc
        except Exception as exc:
            # fast_info leaks raw errors (e.g. KeyError) for symbols with no data.
            raise SymbolNotFound(_no_data_msg("quote", symbol)) from exc
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

    def get_price_history(self, symbol: str, period: str, interval: str) -> PriceHistory:
        symbol = _norm(symbol)
        return self._cached(
            ("history", symbol, period, interval),
            self._history_ttl,
            lambda: self._fetch_history(symbol, period, interval),
        )

    def _all_bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]:
        """Parsed bars for one (symbol, period, interval), cached once for every consumer.

        get_price_history and analyze_performance are two views of the same fetch; keying the
        raw bars separately from the derived models keeps them on a single network round-trip.
        """
        return self._cached(
            ("bars", symbol, period, interval),
            self._history_ttl,
            lambda: self._fetch_all_bars(symbol, period, interval),
        )

    def _fetch_all_bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]:
        """Fetch and parse the FULL (untruncated) OHLCV bars, dropping non-finite rows."""
        try:
            df = self._ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
        except Exception as exc:  # surface any yfinance failure verbatim
            raise DataUnavailable(f"Failed to fetch history for '{symbol}': {exc}") from exc
        if df is None or df.empty:
            raise SymbolNotFound(
                f"No price history for '{symbol}'. Check the symbol/period/interval."
            )
        try:
            all_bars: list[PriceBar] = []
            for idx, row in df.iterrows():
                o, h, low, c, v = (
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    float(row["Volume"]),
                )
                if any(not math.isfinite(x) for x in (o, h, low, c, v)):
                    continue
                all_bars.append(
                    PriceBar(
                        date=idx.date().isoformat(), open=o, high=h, low=low, close=c, volume=v
                    )
                )
            if not all_bars:
                raise SymbolNotFound(
                    f"No price history for '{symbol}'. Check the symbol/period/interval."
                )
            return all_bars
        except SymbolNotFound:
            raise
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(f"Failed to parse history for '{symbol}': {exc}") from exc

    def _fetch_history(self, symbol: str, period: str, interval: str) -> PriceHistory:
        all_bars = self._all_bars(symbol, period, interval)
        start_close = all_bars[0].close
        total_return = ((all_bars[-1].close / start_close - 1.0) * 100.0) if start_close else 0.0
        summary = PriceSummary(
            start_date=all_bars[0].date,
            end_date=all_bars[-1].date,
            start_close=start_close,
            end_close=all_bars[-1].close,
            total_return_percent=total_return,
            period_high=max(b.high for b in all_bars),
            period_low=min(b.low for b in all_bars),
            bars=len(all_bars),
        )
        truncated = len(all_bars) > self._max_bars
        bars = all_bars[-self._max_bars :] if truncated else all_bars
        return PriceHistory(
            symbol=symbol,
            period=period,
            interval=interval,
            bars=bars,
            summary=summary,
            truncated=truncated,
        )

    def analyze_performance(self, symbol: str, period: str) -> PerformanceStats:
        symbol = _norm(symbol)
        # No cache entry of its own: the underlying bars are cached by _all_bars, and the
        # stats are cheap to recompute from them.
        return self._compute_performance(symbol, period)

    def _compute_performance(self, symbol: str, period: str) -> PerformanceStats:
        bars = self._all_bars(symbol, period, "1d")
        if len(bars) < 2:
            raise DataUnavailable(
                f"Not enough price history to compute performance for '{symbol}'."
            )
        closes = [b.close for b in bars]
        # Annualize off wall-clock time, not the bar count: how many bars a year holds is a
        # property of the instrument's trading calendar (~252 weekday, ~365 for crypto), so
        # both the CAGR exponent and the volatility factor are read from the dates.
        elapsed_days = _elapsed_days(bars[0].date, bars[-1].date)
        annualized_return: float | None = None
        annualized_volatility: float | None = None
        periods_per_year: float | None = None
        if elapsed_days >= MIN_ANNUALIZATION_DAYS:
            years = elapsed_days / analytics.DAYS_PER_YEAR
            periods_per_year = analytics.infer_periods_per_year(len(bars), years)
            annualized_return = analytics.annualized_return(closes, years)
            annualized_volatility = analytics.annualized_volatility(closes, periods_per_year)
        return PerformanceStats(
            symbol=symbol,
            period=period,
            bars=len(bars),
            start_date=bars[0].date,
            end_date=bars[-1].date,
            total_return_percent=analytics.total_return(closes),
            annualized_return_percent=annualized_return,
            annualized_volatility_percent=annualized_volatility,
            periods_per_year=periods_per_year,
            max_drawdown_percent=analytics.max_drawdown(closes),
            sma_50=analytics.sma(closes, SMA_SHORT_WINDOW),
            sma_200=analytics.sma(closes, SMA_LONG_WINDOW),
        )

    def get_financials(
        self,
        symbol: str,
        statement: Statement,
        period: StatementPeriod,
        line_items: list[str] | None = None,
    ) -> FinancialStatement:
        symbol = _norm(symbol)
        full = self._cached(
            ("financials", symbol, statement, period),
            self._fundamentals_ttl,
            lambda: self._fetch_financials(symbol, statement, period),
        )
        if line_items is None:
            return full
        wanted = {li: full.line_items[li] for li in line_items if li in full.line_items}
        return full.model_copy(update={"line_items": wanted})

    def _fetch_financials(
        self,
        symbol: str,
        statement: Statement,
        period: StatementPeriod,
    ) -> FinancialStatement:
        attr = _FINANCIALS_ATTR[(statement, period)]
        try:
            df = getattr(self._ticker(symbol), attr)
        except Exception as exc:  # surface any yfinance failure verbatim
            raise DataUnavailable(
                f"Failed to fetch {statement} statement for '{symbol}': {exc}"
            ) from exc
        if df is None or df.empty:
            raise SymbolNotFound(
                f"No {statement} statement available for '{symbol}'. It may be an ETF, index, or "
                "other instrument without financial statements, or an invalid symbol."
            )
        try:
            period_ends = [col.date().isoformat() for col in df.columns]
            line_items: dict[str, list[float | None]] = {
                str(idx): [_opt(v) for v in row] for idx, row in df.iterrows()
            }
            return FinancialStatement(
                symbol=symbol,
                statement=statement,
                period=period,
                period_ends=period_ends,
                line_items=line_items,
            )
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(
                f"Failed to parse {statement} statement for '{symbol}': {exc}"
            ) from exc

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        symbol = _norm(symbol)
        return self._cached(
            ("profile", symbol), self._fundamentals_ttl, lambda: self._fetch_profile(symbol)
        )

    def _fetch_profile(self, symbol: str) -> CompanyProfile:
        ticker, info = self._ticker_with_info(symbol, "profile", "profile")
        try:
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
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(f"Failed to parse profile for '{symbol}': {exc}") from exc

    def get_key_metrics(self, symbol: str) -> KeyMetrics:
        symbol = _norm(symbol)
        return self._cached(
            ("metrics", symbol), self._fundamentals_ttl, lambda: self._fetch_metrics(symbol)
        )

    def _fetch_metrics(self, symbol: str) -> KeyMetrics:
        _, info = self._ticker_with_info(symbol, "metrics", "metrics")
        try:
            return KeyMetrics(
                symbol=symbol,
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
        except Exception as exc:  # surface any mapping failure verbatim
            raise DataUnavailable(f"Failed to parse metrics for '{symbol}': {exc}") from exc

    def get_analyst_data(self, symbol: str) -> AnalystData:
        symbol = _norm(symbol)
        return self._cached(
            ("analyst", symbol),
            self._fundamentals_ttl,
            lambda: self._fetch_analyst(symbol),
        )

    def _fetch_analyst(self, symbol: str) -> AnalystData:
        ticker, info = self._ticker_with_info(symbol, "analyst data", "analyst")
        try:
            mean = _opt(info.get("recommendationMean"))
            analysts = _opt_int(info.get("numberOfAnalystOpinions"))
            target_mean = _opt(info.get("targetMeanPrice"))
            target_median = _opt(info.get("targetMedianPrice"))
            target_high = _opt(info.get("targetHighPrice"))
            target_low = _opt(info.get("targetLowPrice"))
        except Exception as exc:  # non-numeric values from the source
            raise DataUnavailable(f"Failed to parse analyst data for '{symbol}': {exc}") from exc
        targets = (target_mean, target_median, target_high, target_low)
        if mean is None and analysts is None and all(t is None for t in targets):
            raise DataUnavailable(
                f"No analyst coverage for '{symbol}'. It may be an ETF, index, or other "
                "instrument without sell-side analyst data."
            )
        try:
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
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(f"Failed to parse analyst data for '{symbol}': {exc}") from exc

    def get_news(self, symbol: str, count: int = 10) -> NewsResult:
        symbol = _norm(symbol)
        return self._cached(
            ("news", symbol, str(count)),
            self._history_ttl,
            lambda: self._fetch_news(symbol, count),
        )

    def _fetch_news(self, symbol: str, count: int) -> NewsResult:
        try:
            items = self._ticker(symbol).get_news(count=count, tab="news")
        except YFException as exc:
            raise DataUnavailable(f"Failed to fetch news for '{symbol}': {exc}") from exc
        except Exception as exc:  # a failed news fetch is a data issue, not a missing symbol
            raise DataUnavailable(f"Failed to fetch news for '{symbol}': {exc}") from exc
        if not items:
            return NewsResult(symbol=symbol, articles=[])
        try:
            articles = [a for a in (_news_article(it) for it in items) if a is not None][:count]
            return NewsResult(symbol=symbol, articles=articles)
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(f"Failed to parse news for '{symbol}': {exc}") from exc

    def search_symbols(self, query: str, max_results: int = 8) -> SymbolSearchResult:
        return self._cached(
            ("search", query, str(max_results)),
            self._fundamentals_ttl,
            lambda: self._fetch_search(query, max_results),
        )

    def _fetch_search(self, query: str, max_results: int) -> SymbolSearchResult:
        try:
            result = self._search(query, max_results=max_results, news_count=0, lists_count=0)
            quotes = result.quotes
        except Exception as exc:  # a failed search is a data issue, not a missing symbol
            raise DataUnavailable(f"Search failed for '{query}': {exc}") from exc
        if not quotes:
            return SymbolSearchResult(query=query, matches=[])
        try:
            matches = [_symbol_match(q) for q in quotes if q.get("symbol")]
            return SymbolSearchResult(query=query, matches=matches)
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(f"Failed to parse search results for '{query}': {exc}") from exc


def _elapsed_days(start: str, end: str) -> int:
    """Calendar days between two PriceBar dates.

    Parses via ``datetime.fromisoformat`` rather than ``date.fromisoformat`` so it accepts
    both a bare date ("2024-01-01") and a full intraday timestamp with offset.
    """
    return (datetime.fromisoformat(end).date() - datetime.fromisoformat(start).date()).days


def _norm(symbol: str) -> str:
    """Normalize a ticker so equivalent spellings share one cache entry and one fetch.

    Yahoo symbols are upper-case; callers routinely pass 'aapl' or 'AAPL '. Normalizing
    here (before the cache key is built) is also what makes the echoed symbol canonical.
    """
    normalized = symbol.strip().upper()
    if not normalized:
        raise SymbolNotFound("Empty ticker symbol.")
    return normalized


def _no_data_msg(kind: str, symbol: str) -> str:
    """Single source of truth for the 'symbol has no usable data' message."""
    return f"No {kind} data for '{symbol}'. The symbol may be invalid or delisted."


def _recommendation_trend(df: Any) -> list[RecommendationPeriod]:
    if df is None or len(df) == 0:
        return []
    periods: list[RecommendationPeriod] = []
    for row in df.to_dict("records"):
        periods.append(
            RecommendationPeriod(
                period=str(row.get("period", "")),
                strong_buy=_opt_int(row.get("strongBuy")) or 0,
                buy=_opt_int(row.get("buy")) or 0,
                hold=_opt_int(row.get("hold")) or 0,
                sell=_opt_int(row.get("sell")) or 0,
                strong_sell=_opt_int(row.get("strongSell")) or 0,
            )
        )
    return periods


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
