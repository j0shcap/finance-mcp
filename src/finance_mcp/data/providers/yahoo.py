"""Yahoo Finance access through yfinance: every fetch and parse, nothing cached.

Network access is isolated here. yfinance errors and empty results become
DataUnavailable/SymbolNotFound whose message is surfaced to the caller verbatim, because we
cannot enumerate every Yahoo failure mode. SymbolNotFound is reserved for signals that
really mean "no data for this symbol" (see _is_no_data_error): everything else, transport
failures included, stays a plain DataUnavailable.

YahooProvider implements every port in providers/ports.py. It takes symbols already
normalized by its caller and never caches: caching and the choice of what to fetch together
belong to DataService.
"""

import math
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import yfinance as yf
from yfinance.exceptions import YFRateLimitError, YFTickerMissingError

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    DividendEvent,
    Earnings,
    EarningsPeriod,
    EstimateRange,
    FinancialStatement,
    Identity,
    KeyMetrics,
    NewsArticle,
    NewsSource,
    NextEarnings,
    PeriodEstimate,
    PriceBar,
    Quote,
    RecommendationPeriod,
    ReportedQuarter,
    SplitEvent,
    Statement,
    StatementPeriod,
    SymbolMatch,
    SymbolSearchResult,
)

#: Yahoo's 13-week US Treasury bill yield: percent, on a bank-discount basis.
TREASURY_BILL_SYMBOL = "^IRX"

#: The quoteSummary modules behind earnings. quoteType carries the exchange timezone, and is
#: all Yahoo returns for an instrument that doesn't report earnings (an ETF, index, coin).
_EARNINGS_DATA_MODULES = ("calendarEvents", "earningsHistory", "earningsTrend")
EARNINGS_MODULES = (*_EARNINGS_DATA_MODULES, "quoteType")
_EARNINGS_PERIODS: dict[str, EarningsPeriod] = {
    "0q": "reporting_quarter",
    "+1q": "following_quarter",
    "0y": "reporting_fiscal_year",
    "+1y": "following_fiscal_year",
}

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


#: Most Yahoo requests in flight at once, and retries of a transient failure, by default.
DEFAULT_MAX_CONCURRENT_REQUESTS = 8
DEFAULT_REQUEST_RETRIES = 2
#: First backoff before a retry; each further one is three times longer (2s, 6s, ...), each
#: jittered by +-50% so throttled requests don't retry in lockstep.
RETRY_BASE_DELAY_SECONDS = 2.0
#: A retry is only started if it can begin within this long of the first attempt. yfinance's
#: own request timeouts are 10s (price history) and 30s (everything else), so this keeps a
#: retried call well inside an MCP client's.
RETRY_BUDGET_SECONDS = 15.0
#: How the transport fails fast and transiently, matched by class name anywhere in an
#: exception's hierarchy (curl_cffi's classes, without importing it): a dropped or refused
#: connection or a DNS blip (ConnectionError, DNSError), and a truncated body
#: (IncompleteRead). SSL and proxy errors subclass or sit beside them but don't heal, so
#: they are excluded first; a Timeout has already spent yfinance's 10-30s and isn't retried.
_TRANSIENT_ERROR_NAMES = frozenset({"ConnectionError", "IncompleteRead"})
_PERMANENT_ERROR_NAMES = frozenset({"SSLError", "ProxyError"})
#: curl error codes curl_cffi raises as a bare HTTPError: HTTP2 (16), HTTP2_STREAM (92) -
#: "stream was not closed cleanly", the classic transient failure against Yahoo.
_TRANSIENT_CURL_CODES = frozenset({16, 92})

# Indirections the tests replace, so backoff is instant and jitter-free under test.
_sleep: Callable[[float], None] = time.sleep


def _jitter(delay: float) -> float:
    return delay * random.uniform(0.5, 1.5)  # nosec B311 - spreads retries, not security


_gate_holder = threading.local()


def is_transient(exc: BaseException) -> bool:
    """Throttling or a fast network failure: a later attempt can succeed."""
    if isinstance(exc, YFRateLimitError):
        return True
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _PERMANENT_ERROR_NAMES:
        return False
    if names & _TRANSIENT_ERROR_NAMES:
        return True
    if "CurlError" in names and getattr(exc, "code", None) in _TRANSIENT_CURL_CODES:
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


class YahooProvider:
    """Fetches and parses one kind of Yahoo data per method, into this package's models.

    Every Yahoo request goes through ``_request``: at most ``max_concurrent_requests`` run at
    once, and a throttled or dropped request is retried with backoff.
    """

    def __init__(
        self,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        search_factory: Callable[[str], Any] = yf.Search,
        *,
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
    ) -> None:
        self._ticker = ticker_factory
        # Widened so the keyword arguments yf.Search is called with typecheck.
        self._search: Callable[..., Any] = search_factory
        self._gate = threading.BoundedSemaphore(max_concurrent_requests)
        self._retry_delays = tuple(
            RETRY_BASE_DELAY_SECONDS * 3**attempt for attempt in range(request_retries)
        )
        # By default yfinance's price and statement fetches swallow a transport failure and
        # return an empty frame, which reads exactly like an unknown symbol; every
        # classification here needs failures as exceptions. Process-wide, but this server
        # is the only yfinance user in its process.
        yf.config.debug.hide_exceptions = False

    def _request[T](self, fetch: Callable[[], T], *, retry: bool = True) -> T:
        """Run one Yahoo request under the gate, retrying transient failures.

        The gate is held only while ``fetch`` runs, never during a backoff, and is not
        reentrant: ``fetch`` must be a leaf request, which a nested call enforces by raising.
        The last failure is re-raised unchanged, so callers classify it as before.
        """
        if getattr(_gate_holder, "held", False):
            raise RuntimeError("nested Yahoo request: wrap only the request itself in _request")
        started = time.monotonic()
        # Each wait precedes the next attempt; None marks the last attempt.
        for wait in (*map(_jitter, self._retry_delays if retry else ()), None):
            with self._gate:
                _gate_holder.held = True
                try:
                    return fetch()
                except Exception as exc:
                    out_of_budget = (
                        wait is not None
                        and time.monotonic() - started + wait > RETRY_BUDGET_SECONDS
                    )
                    if wait is None or out_of_budget or not is_transient(exc):
                        raise
                finally:
                    _gate_holder.held = False
            _sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover - the last attempt returns or raises

    def _ticker_with_info(
        self, symbol: str, fetch_label: str, kind: str, *, retry: bool = True
    ) -> tuple[Any, dict[str, Any]]:
        """Fetch a ticker and its ``.info``, asserting the symbol names a real instrument.

        Access errors are classified by _data_error; an ``info`` dict that is empty or has
        no longName/shortName (Yahoo's tell for an unknown symbol) becomes SymbolNotFound.
        """
        try:
            ticker, info = self._request(lambda: _with_info(self._ticker(symbol)), retry=retry)
        except Exception as exc:
            raise _data_error(exc, fetch_label, kind, symbol) from exc
        if not info or not (info.get("longName") or info.get("shortName")):
            raise SymbolNotFound(_no_data_msg(kind, symbol))
        return ticker, info

    def quote(self, symbol: str) -> Quote:
        try:
            # One request block: fast_info's attributes fetch lazily, several requests deep.
            fi = self._request(lambda: _read_fast_info(self._ticker(symbol)))
            price = _opt(fi["last_price"])
            prev = _opt(fi["previous_close"])
            currency = fi["currency"]
            day_high = _opt(fi["day_high"])
            day_low = _opt(fi["day_low"])
            year_high = _opt(fi["year_high"])
            year_low = _opt(fi["year_low"])
            market_cap = _opt(fi["market_cap"])
            volume = _opt(fi["last_volume"])
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
            df = self._request(
                lambda: self._ticker(symbol).history(
                    period=period, interval=interval, auto_adjust=True
                )
            )
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

    def treasury_bill_yields(self, period: str) -> list[PriceBar]:
        return self.bars(TREASURY_BILL_SYMBOL, period, "1d")

    def financial_statement(
        self, symbol: str, statement: Statement, period: StatementPeriod
    ) -> FinancialStatement:
        """The parsed statement, without its currency (that comes from another endpoint)."""
        attr = FINANCIALS_ATTR[(statement, period)]
        with _unavailable_on_error(f"Failed to fetch {statement} statement for '{symbol}'"):
            df = self._request(lambda: getattr(self._ticker(symbol), attr))
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
        # Also covers corporate_actions, a separate Yahoo request.
        with _unavailable_on_error(f"Failed to parse profile for '{symbol}'"):
            dividends, splits = self._request(lambda: corporate_actions(ticker))
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
                recent_dividends=_dividend_events(dividends, limit=8),
                splits=_split_events(splits),
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
            recommendations = self._request(lambda: ticker.recommendations)
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
                recommendation_trend=_recommendation_trend(recommendations),
            )

    def news(self, symbol: str, count: int) -> tuple[list[NewsArticle], NewsSource]:
        with _unavailable_on_error(f"Failed to fetch news for '{symbol}'"):
            items = self._request(lambda: self._ticker(symbol).get_news(count=count, tab="news"))
        if not items:
            # Ambiguous: yfinance turns a 500 from the news endpoint into an empty list.
            # Cross-check search rather than report "no recent news", which the model
            # would read as "no catalysts".
            return self._search_news(symbol, count), "search"
        with _unavailable_on_error(f"Failed to parse news for '{symbol}'"):
            articles = [a for a in (_news_article(it) for it in items) if a is not None][:count]
        return articles, "ticker"

    def identity(self, symbol: str) -> Identity:
        # Best-effort (it only labels news), so a throttled lookup isn't retried.
        _, info = self._ticker_with_info(symbol, "company identity", "identity", retry=False)
        return Identity(
            is_company=info.get("quoteType") == "EQUITY",
            long_name=info.get("longName"),
            short_name=info.get("shortName"),
        )

    def _search_news(self, symbol: str, count: int) -> list[NewsArticle]:
        """News for ``symbol`` from the search endpoint, or none if it cannot supply any.

        A failure is swallowed deliberately: the ticker stream already answered "no news",
        so raising would turn a symbol with genuinely no coverage into an error.
        """
        try:
            found = self._request(
                lambda: self._search(symbol, max_results=1, news_count=count, lists_count=0).news,
                retry=False,
            )
        except Exception:
            return []
        articles = (_search_news_article(item) for item in found or [])
        return [a for a in articles if a is not None][:count]

    def earnings(self, symbol: str) -> Earnings:
        try:
            modules = self._request(lambda: quote_summary(self._ticker(symbol), EARNINGS_MODULES))
        except DataUnavailable:
            raise
        except Exception as exc:
            raise _data_error(exc, "earnings", "earnings", symbol) from exc
        quote_type = modules.get("quoteType") or {}
        # Yahoo answers an instrument without earnings with quoteType alone, not a 404.
        if quote_type.get("quoteType") != "EQUITY" and not any(
            name in modules for name in _EARNINGS_DATA_MODULES
        ):
            raise DataUnavailable(
                f"No earnings for '{symbol}': it is not a company (an ETF, fund, index, "
                "currency or crypto), and only companies report earnings."
            )
        with _unavailable_on_error(f"Failed to parse earnings for '{symbol}'"):
            exchange_tz = ZoneInfo(quote_type.get("timeZoneFullName") or "UTC")
            history = sorted(
                (
                    earnings_quarter
                    for earnings_quarter in (modules.get("earningsHistory") or {}).get("history")
                    or []
                    if earnings_quarter.get("quarter")
                ),
                key=lambda earnings_quarter: _raw(earnings_quarter["quarter"]),
            )
            trend = (modules.get("earningsTrend") or {}).get("trend") or []
            # A period without coverage has no end date and nothing else worth reporting.
            estimates = [
                _period_estimate(trend_item)
                for trend_item in trend
                if trend_item.get("period") in _EARNINGS_PERIODS and trend_item.get("endDate")
            ]
            order = list(_EARNINGS_PERIODS.values())
            return Earnings(
                symbol=symbol,
                next_report=_next_earnings(modules.get("calendarEvents") or {}, exchange_tz),
                estimates=sorted(
                    _drop_contradicted_revenue_year_ago(estimates),
                    key=lambda estimate: order.index(estimate.period),
                ),
                history=[_reported_quarter(earnings_quarter) for earnings_quarter in history],
                history_currency=next(
                    (
                        earnings_quarter.get("currency")
                        for earnings_quarter in history
                        if earnings_quarter.get("currency")
                    ),
                    None,
                ),
            )

    def search(self, query: str, max_results: int) -> SymbolSearchResult:
        with _unavailable_on_error(f"Search failed for '{query}'"):
            quotes = self._request(
                lambda: (
                    self._search(query, max_results=max_results, news_count=0, lists_count=0).quotes
                )
            )
        if not quotes:
            return SymbolSearchResult(query=query, matches=[])
        with _unavailable_on_error(f"Failed to parse search results for '{query}'"):
            matches = [_symbol_match(q) for q in quotes if q.get("symbol")]
            return SymbolSearchResult(query=query, matches=matches)

    def statement_currency(self, symbol: str) -> str | None:
        """The currency a symbol's statements are reported in, from ``.info``.

        Best-effort (it only labels a statement), so a throttled read isn't retried.
        """
        with _unavailable_on_error(f"Failed to fetch the statement currency for '{symbol}'"):
            return self._request(
                lambda: _read_statement_currency(self._ticker(symbol)), retry=False
            )


def _with_info(ticker: Any) -> tuple[Any, dict[str, Any]]:
    return ticker, ticker.info


#: The fast_info attributes a quote reads. Each may fetch lazily, so they are all read inside
#: one gated request.
_FAST_INFO_FIELDS = (
    "last_price",
    "previous_close",
    "currency",
    "day_high",
    "day_low",
    "year_high",
    "year_low",
    "market_cap",
    "last_volume",
)


def corporate_actions(ticker: Any) -> tuple[Any, Any]:
    """A ticker's (dividends, splits) over its whole history, with exact ex-dates.

    Read from yfinance's private price-history cache at a weekly interval: the same events
    the public .dividends/.splits get from daily history, in about a quarter of the data.
    Falls back to those public reads if the cache is gone. Run it as one request: it may
    also fetch the exchange timezone.
    """
    load = getattr(ticker, "_lazy_load_price_history", None)
    price_history = load() if load is not None else None
    history_cache = getattr(price_history, "_get_history_cache", None)
    if history_cache is None:
        return ticker.dividends, ticker.splits
    actions = history_cache(period="max", interval="1wk")
    return actions["dividends"], actions["splits"]


def quote_summary(ticker: Any, modules: tuple[str, ...]) -> dict[str, Any]:
    """Yahoo quoteSummary modules for ``ticker``, by name, in one request.

    Through yfinance's private quote scraper, which carries its authenticated session: the
    public properties make a request each and drop fields the answer needs (whether the next
    date is confirmed, each estimate period's end date, the history's currency).
    """
    fetch = getattr(getattr(ticker, "_quote", None), "_fetch", None)
    if fetch is None:
        raise DataUnavailable("This version of yfinance can't fetch Yahoo's quoteSummary.")
    response = fetch(modules=list(modules))
    try:
        result: dict[str, Any] = response["quoteSummary"]["result"][0]
    except (KeyError, IndexError, TypeError) as exc:
        # Not a KeyError: that is this adapter's "no such symbol" signal, and a malformed
        # reply says nothing about the symbol.
        raise DataUnavailable(f"Unexpected quoteSummary response: {response!r:.200}") from exc
    return result


def _raw(value: Any) -> Any:
    """A quoteSummary value: some modules wrap numbers as {"raw": ..., "fmt": ...}."""
    return value.get("raw") if isinstance(value, dict) else value


def _opt_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _percent(value: Any) -> float | None:
    fraction = _opt(_raw(value))
    return fraction * 100.0 if fraction is not None else None


# Receivers below are named for the payload part they read: scripts/yahoo_shapes.py records
# the shape of every literal key read from each, by name.
def _next_earnings(calendar: dict[str, Any], exchange_tz: ZoneInfo) -> NextEarnings | None:
    calendar_earnings = calendar.get("earnings") or {}
    dates = sorted(calendar_earnings.get("earningsDate") or [])
    if not dates:
        return None
    at = [datetime.fromtimestamp(ts, exchange_tz).isoformat() for ts in dates]
    return NextEarnings(
        date=at[0],
        date_is_estimate=_opt_bool(calendar_earnings.get("isEarningsDateEstimate")),
        window_end=at[-1] if len(at) > 1 else None,
    )


def _estimate_range(estimate: dict[str, Any], year_ago: Any) -> EstimateRange:
    return EstimateRange(
        average=_opt(_raw(estimate.get("avg"))),
        low=_opt(_raw(estimate.get("low"))),
        high=_opt(_raw(estimate.get("high"))),
        analysts=_opt_int(_raw(estimate.get("numberOfAnalysts"))),
        year_ago=_opt(_raw(year_ago)),
        growth_percent=_percent(estimate.get("growth")),
    )


def _period_estimate(trend_item: dict[str, Any]) -> PeriodEstimate:
    eps_estimate = trend_item.get("earningsEstimate") or {}
    revenue_estimate = trend_item.get("revenueEstimate") or {}
    return PeriodEstimate(
        period=_EARNINGS_PERIODS[trend_item["period"]],
        fiscal_period_end=trend_item["endDate"],
        eps=_estimate_range(eps_estimate, eps_estimate.get("yearAgoEps")),
        eps_currency=eps_estimate.get("earningsCurrency"),
        revenue=_estimate_range(revenue_estimate, revenue_estimate.get("yearAgoRevenue")),
        revenue_currency=revenue_estimate.get("revenueCurrency"),
    )


#: Yahoo understates the year-ago revenue of many March fiscal years (Toyota's by 64%, Sony's
#: by 94%), which inflates the year's growth AND leaves its year-ago short of what its quarters
#: imply. Either alone happens legitimately - a hyper-grower's year-ago is short, and earlier
#: quarters can outgrow the two still estimated - so a value is dropped only on both. Growth is
#: compared as multiples (1 + growth): observed bad values reach 1.17x and a 0.74 shortfall.
_YEAR_GROWTH_ABOVE_QUARTERS = 1.1
_YEAR_AGO_SHORTFALL = 0.8


def _drop_contradicted_revenue_year_ago(estimates: list[PeriodEstimate]) -> list[PeriodEstimate]:
    """Null a fiscal year's revenue year_ago and growth where its quarters contradict them.

    Needs two distinct quarters inside the year: with one, correct values of fast growers
    trip both tests. Only revenue: EPS year-agos match what companies report.
    """
    # Quarter end -> (growth as a multiple, year-ago revenue); a repeated end counts once.
    quarters: dict[date, tuple[float, float]] = {}
    for estimate in estimates:
        revenue = estimate.revenue
        if (
            estimate.period in ("reporting_quarter", "following_quarter")
            and revenue.growth_percent is not None
            and revenue.growth_percent > -100
            and revenue.year_ago is not None
            and revenue.year_ago > 0
        ):
            quarters[date.fromisoformat(estimate.fiscal_period_end)] = (
                1 + revenue.growth_percent / 100,
                revenue.year_ago,
            )
    checked = []
    for estimate in estimates:
        year = estimate.revenue
        if (
            estimate.period.endswith("fiscal_year")
            and year.growth_percent is not None
            and year.year_ago is not None
        ):
            end = date.fromisoformat(estimate.fiscal_period_end)
            inside = [q for q_end, q in quarters.items() if _one_year_before(end) < q_end <= end]
            if (
                len(inside) >= 2
                and 1 + year.growth_percent / 100
                > _YEAR_GROWTH_ABOVE_QUARTERS * max(growth for growth, _ in inside)
                and year.year_ago
                < _YEAR_AGO_SHORTFALL * 4 / len(inside) * sum(ago for _, ago in inside)
            ):
                cleared = year.model_copy(update={"year_ago": None, "growth_percent": None})
                estimate = estimate.model_copy(update={"revenue": cleared})
        checked.append(estimate)
    return checked


def _one_year_before(day: date) -> date:
    # Fiscal periods end on month-ends; February 29 has no counterpart a year earlier.
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)


def _reported_quarter(earnings_quarter: dict[str, Any]) -> ReportedQuarter:
    # Quarter ends are UTC midnight: in a western exchange's timezone they'd be the day before.
    end = datetime.fromtimestamp(_raw(earnings_quarter["quarter"]), UTC).date()
    return ReportedQuarter(
        fiscal_quarter_end=end.isoformat(),
        eps_estimate=_opt(_raw(earnings_quarter.get("epsEstimate"))),
        eps_actual=_opt(_raw(earnings_quarter.get("epsActual"))),
        surprise_percent=_percent(earnings_quarter.get("surprisePercent")),
    )


def _read_fast_info(ticker: Any) -> dict[str, Any]:
    fast_info = ticker.fast_info
    return {name: getattr(fast_info, name, None) for name in _FAST_INFO_FIELDS}


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
