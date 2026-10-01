"""Cached, error-surfacing wrapper over yfinance. No MCP imports.

Network access is isolated here. yfinance errors and empty results become
DataUnavailable/SymbolNotFound whose message is surfaced to the caller verbatim,
because we cannot enumerate every Yahoo failure mode. SymbolNotFound is reserved for
signals that really mean "no data for this symbol" (see _is_no_data_error): everything
else, transport failures included, stays a plain DataUnavailable.
"""

import difflib
import math
import statistics
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any, NamedTuple, cast

import yfinance as yf
from yfinance.exceptions import YFTickerMissingError

from finance_mcp.data import analytics, relevance
from finance_mcp.data.errors import DataUnavailable, InvalidInput, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    BenchmarkComparison,
    CompanyProfile,
    ComparisonError,
    DividendEvent,
    FinancialStatement,
    KeyMetrics,
    NewsArticle,
    NewsResult,
    NewsSource,
    PerformanceStats,
    PriceBar,
    PriceHistory,
    PriceSummary,
    Quote,
    QuoteError,
    QuoteResult,
    RecommendationPeriod,
    RelevanceCheck,
    RiskFreeSource,
    SplitEvent,
    Statement,
    StatementPeriod,
    SymbolMatch,
    SymbolSearchResult,
    TickerComparison,
    TickerComparisonRow,
)

DEFAULT_MAX_BARS = 260
DEFAULT_CACHE_MAX_ENTRIES = 256
# The LRU bounds how many entries are held, not how large they are: a period="max" daily
# history is ~11.5k bars (~9 MB). Longer bar lists are returned in full but not kept;
# ~2000 daily bars is about eight years, so every ordinary window stays cached.
MAX_CACHEABLE_BARS = 2000
# Keeps a large quote batch from opening a connection per ticker at once.
QUOTE_MAX_WORKERS = 8
# The fewest shared closes that yield a single return to compare.
MIN_OVERLAP_OBSERVATIONS = 2
# Each comparison row costs two Yahoo calls (history + info), hence a lower bound than
# quotes for the same ceiling on concurrent connections.
COMPARE_MAX_WORKERS = 5
# Intervals whose bars are points in time rather than whole sessions. Kept in sync with
# HistoryInterval (a test pins it): everything that is not a daily-or-longer interval.
_INTRADAY_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h"})
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
# Below about a quarter, annualizing compounds short-run noise into a yearly figure that
# reads as a forecast. Just under three months because a period="3mo" window spans 87-95
# elapsed days depending on the call date, and the same request should not gain and lose
# its annualized fields from one day to the next.
MIN_ANNUALIZATION_DAYS = 85
#: Yahoo's 13-week US Treasury bill yield, the default risk-free rate. Quoted in percent on
#: a bank-discount basis; see analytics.treasury_bill_effective_rate.
TREASURY_BILL_SYMBOL = "^IRX"
#: How far inside a measured window the T-bill history may start or end and still count as
#: covering it, so a bond-market holiday at either edge is not a gap.
RISK_FREE_EDGE_TOLERANCE_DAYS = 7

_FINANCIALS_ATTR = {
    ("income", "annual"): "income_stmt",
    ("income", "quarterly"): "quarterly_income_stmt",
    ("balance", "annual"): "balance_sheet",
    ("balance", "quarterly"): "quarterly_balance_sheet",
    ("cashflow", "annual"): "cashflow",
    ("cashflow", "quarterly"): "quarterly_cashflow",
}


class _RiskFree(NamedTuple):
    """The risk-free rate one computation used, and how it was chosen."""

    rate: float | None
    source: RiskFreeSource
    note: str | None = None
    #: True only when the T-bill FETCH failed, so a retry could resolve the rate. A window
    #: the history does not cover, or an implausible quote, gives the same answer each time.
    retryable: bool = False


#: The T-bill history a computation draws its default rate from: the bars, or why they
#: could not be fetched. None when the caller passed a rate, so nothing was fetched.
_Bills = list[PriceBar] | str | None


class _Identity(NamedTuple):
    """What the relevance flags need to know about a symbol, from Yahoo's ``info``."""

    quote_type: str | None
    long_name: str | None
    short_name: str | None


class _IdentityGap(NamedTuple):
    """Why a symbol's identity is missing, and whether retrying could change that."""

    reason: str
    lasting: bool


class YFinanceClient:
    """Thin yfinance facade with a per-key TTL cache (bounded, least-recently-used)."""

    def __init__(
        self,
        ticker_factory: Callable[[str], Any] = yf.Ticker,
        search_factory: Callable[[str], Any] = yf.Search,
        time_fn: Callable[[], float] = time.monotonic,
        quote_ttl: float = 30.0,
        history_ttl: float = 300.0,
        fundamentals_ttl: float = 3600.0,
        max_bars: int = DEFAULT_MAX_BARS,
        cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        self._ticker = ticker_factory
        # Widened so the keyword arguments yf.Search is called with typecheck.
        self._search: Callable[..., Any] = search_factory
        self._now = time_fn
        self._quote_ttl = quote_ttl
        self._history_ttl = history_ttl
        self._fundamentals_ttl = fundamentals_ttl
        self._max_bars = max_bars
        self._cache_max_entries = cache_max_entries
        # key -> (stored_at, ttl, value), in LRU order (oldest use first). The ttl is kept
        # per entry so the purge pass can judge expiry without knowing who wrote it.
        self._cache: OrderedDict[tuple[str, ...], tuple[float, float, Any]] = OrderedDict()
        # get_quote fetches concurrently, so cache bookkeeping is guarded. Fetches run
        # OUTSIDE the lock: two threads racing on one uncached key just fetch it twice.
        self._cache_lock = threading.Lock()
        # By default yfinance's price and statement fetches swallow a transport failure and
        # return an empty frame, which reads exactly like an unknown symbol; every
        # classification below needs failures as exceptions. Process-wide, but this server
        # is the only yfinance user in its process.
        yf.config.debug.hide_exceptions = False

    def _cached[T](
        self,
        key: tuple[str, ...],
        ttl: float,
        fetch: Callable[[], T],
        cacheable: Callable[[T], bool] | None = None,
    ) -> T:
        """Return ``fetch()``, reusing a live entry and storing the result under ``key``.

        ``cacheable`` is consulted after the fetch to decide whether the value is worth
        keeping; ``None`` means always keep it. It bounds an entry by size, which the
        entry-count LRU cannot do.
        """
        now = self._now()
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                if now - hit[0] < ttl:
                    self._cache.move_to_end(key)
                    # Values are Any, but each key prefix always stores the same type.
                    return cast(T, hit[2])
                del self._cache[key]
        value = fetch()
        if cacheable is not None and not cacheable(value):
            return value
        stored_at = self._now()  # read AFTER the fetch: a slow fetch must not age its entry
        with self._cache_lock:
            self._purge_expired(stored_at)
            self._cache[key] = (stored_at, ttl, value)
            # Assigning a key that is still present (a concurrent fetch of the same key
            # got there first) leaves it in its old position, so order it explicitly.
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max_entries:
                self._cache.popitem(last=False)
        return value

    def _purge_expired(self, now: float) -> None:
        """Drop every entry past its own TTL, so stale keys cannot squat on the bound."""
        for key in [k for k, (stored, ttl, _) in self._cache.items() if now - stored >= ttl]:
            del self._cache[key]

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

    def get_quote(self, symbols: list[str]) -> QuoteResult:
        """Fetch quotes for a batch of symbols concurrently, with partial results.

        A batch is a set of independent lookups, so one unknown or unreachable ticker
        reports itself in ``errors`` instead of discarding the quotes that did work.
        """
        pending, failures = _normalize_batch(symbols)
        errors = [QuoteError(symbol=raw, error=reason) for raw, reason in failures]
        fetched = _fetch_concurrently(pending, self._quote_or_error, QUOTE_MAX_WORKERS)
        errors.extend(r for r in fetched if isinstance(r, QuoteError))
        return QuoteResult(quotes=[r for r in fetched if isinstance(r, Quote)], errors=errors)

    def _quote_or_error(self, symbol: str) -> Quote | QuoteError:
        """One symbol's cached quote, or the reason it could not be fetched."""
        try:
            return self._cached(
                ("quote", symbol), self._quote_ttl, lambda: self._fetch_quote(symbol)
            )
        except DataUnavailable as exc:
            return QuoteError(symbol=symbol, error=str(exc))

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

    def get_price_history(self, symbol: str, period: str, interval: str) -> PriceHistory:
        symbol = _norm(symbol)
        return self._cached(
            ("history", symbol, period, interval),
            self._history_ttl,
            lambda: self._fetch_history(symbol, period, interval),
        )

    def _all_bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]:
        """Parsed bars for one (symbol, period, interval), shared by every view of them.

        Histories longer than MAX_CACHEABLE_BARS are not retained, so a very long window
        costs one fetch per view; each view caches its own small result, so repeat calls
        still avoid the network.
        """
        return self._cached(
            ("bars", symbol, period, interval),
            self._history_ttl,
            lambda: self._fetch_all_bars(symbol, period, interval),
            cacheable=lambda bars: len(bars) <= MAX_CACHEABLE_BARS,
        )

    def _fetch_all_bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]:
        """Fetch and parse the FULL (untruncated) OHLCV bars, dropping non-finite rows."""
        intraday = interval in _INTRADAY_INTERVALS
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

    def analyze_performance(
        self, symbol: str, period: str, risk_free_rate: float | None = None
    ) -> PerformanceStats:
        """Return and risk statistics over ``period``.

        With no ``risk_free_rate``, the Sharpe, Sortino and downside figures are measured
        against the 13-week T-bill yield averaged over the same dates.
        """
        symbol = _norm(symbol)
        # Cached in its own right because bars past MAX_CACHEABLE_BARS are not. The rate is
        # in the key since the rate-dependent figures are computed from it. A result whose
        # T-bill fetch failed is not kept, so the next call retries it; a lasting gap is.
        stats, _ = self._cached(
            ("performance", symbol, period, _rate_key(risk_free_rate)),
            self._history_ttl,
            lambda: self._fetch_performance(symbol, period, risk_free_rate),
            cacheable=lambda fetched: not fetched[1].retryable,
        )
        return stats

    def _fetch_performance(
        self, symbol: str, period: str, risk_free_rate: float | None
    ) -> tuple[PerformanceStats, _RiskFree]:
        bars, bills = _in_parallel(
            lambda: self._all_bars(symbol, period, "1d"),
            lambda: self._bills_for(period, risk_free_rate),
        )
        risk_free = _risk_free_over(risk_free_rate, bills, bars[0].date, bars[-1].date)
        return _performance(symbol, period, bars, risk_free), risk_free

    def _bills_for(self, period: str, risk_free_rate: float | None) -> _Bills:
        """The T-bill history for a default rate; nothing when the caller gave a rate.

        A failed fetch is returned as its message rather than raised: only the rate-
        dependent figures need it, so it must not fail the whole computation.
        """
        if risk_free_rate is not None:
            return None
        try:
            return self._all_bars(TREASURY_BILL_SYMBOL, period, "1d")
        except DataUnavailable as exc:
            return str(exc)

    def compare_to_benchmark(
        self, symbol: str, benchmark: str, period: str, risk_free_rate: float | None = None
    ) -> BenchmarkComparison:
        """Benchmark-relative statistics over the dates the two instruments share.

        The two histories and, for a default rate, the T-bill history are fetched in
        parallel.
        """
        symbol, bench = _norm(symbol), _norm(benchmark)
        if symbol == bench:
            raise InvalidInput(
                f"A benchmark comparison needs two different symbols; '{symbol}' was given "
                "for both. Use analyze_performance for a single instrument."
            )
        (asset_bars, bench_bars), bills = _in_parallel(
            lambda: _in_parallel(
                lambda: self._all_bars(symbol, period, "1d"),
                lambda: self._all_bars(bench, period, "1d"),
            ),
            lambda: self._bills_for(period, risk_free_rate),
        )
        dates, asset_closes, bench_closes = analytics.align_closes(
            [(b.date, b.close) for b in asset_bars], [(b.date, b.close) for b in bench_bars]
        )
        if len(dates) < MIN_OVERLAP_OBSERVATIONS:
            raise DataUnavailable(
                f"'{symbol}' and '{bench}' have only {len(dates)} overlapping daily close(s) "
                f"over '{period}', so there is nothing to compare. Try a longer period, or "
                "check that both symbols traded over this window."
            )
        # Alpha's risk-free leg spans the dates actually compared, not either full history.
        risk_free = _risk_free_over(risk_free_rate, bills, dates[0], dates[-1])
        elapsed_days = _elapsed_days(dates[0], dates[-1])
        periods_per_year: float | None = None
        asset_cagr: float | None = None
        bench_cagr: float | None = None
        tracking: float | None = None
        info_ratio: float | None = None
        # Beta and correlation are unit-free and need no calendar, so they are computed
        # unconditionally; everything annualized shares analyze_performance's gate.
        asset_beta = analytics.beta(asset_closes, bench_closes)
        if elapsed_days >= MIN_ANNUALIZATION_DAYS:
            years = elapsed_days / analytics.DAYS_PER_YEAR
            periods_per_year = analytics.infer_periods_per_year(len(dates), years)
            asset_cagr = analytics.annualized_return(asset_closes, years)
            bench_cagr = analytics.annualized_return(bench_closes, years)
            tracking = analytics.tracking_error(asset_closes, bench_closes, periods_per_year)
            info_ratio = analytics.information_ratio(asset_closes, bench_closes, periods_per_year)
        alpha: float | None = None
        if (
            asset_beta is not None
            and asset_cagr is not None
            and bench_cagr is not None
            and risk_free.rate is not None
        ):
            alpha = analytics.jensen_alpha(asset_cagr, bench_cagr, asset_beta, risk_free.rate)
        asset_total = analytics.total_return(asset_closes)
        bench_total = analytics.total_return(bench_closes)
        return BenchmarkComparison(
            symbol=symbol,
            benchmark=bench,
            period=period,
            overlapping_observations=len(dates),
            start_date=dates[0],
            end_date=dates[-1],
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free.rate,
            risk_free_rate_source=risk_free.source,
            risk_free_rate_note=risk_free.note,
            total_return_percent=asset_total,
            benchmark_total_return_percent=bench_total,
            excess_return_percent=asset_total - bench_total,
            annualized_return_percent=asset_cagr,
            benchmark_annualized_return_percent=bench_cagr,
            beta=asset_beta,
            correlation=analytics.correlation(asset_closes, bench_closes),
            alpha_percent=alpha,
            tracking_error_percent=tracking,
            information_ratio=info_ratio,
        )

    def compare_tickers(
        self, symbols: list[str], period: str, risk_free_rate: float | None = None
    ) -> TickerComparison:
        """Side-by-side performance and valuation for a small batch, fetched concurrently.

        One ticker's failure reports itself in ``errors`` instead of discarding the rows
        that worked. For a default rate the T-bill history is fetched once, alongside the
        rows, and each row averages it over its own dates: a period="max" history is too
        long to cache, so fetching it per row would cost a round trip per ticker.
        """
        pending, failures = _normalize_batch(symbols)
        errors = [ComparisonError(symbol=raw, error=reason) for raw, reason in failures]
        bills: _Bills = None
        built: list[TickerComparisonRow | ComparisonError] = []
        if pending:
            # Started alongside the rows; each row waits for it only after its own bars.
            with ThreadPoolExecutor(max_workers=1) as bill_pool:
                bills_future = bill_pool.submit(self._bills_for, period, risk_free_rate)
                built = _fetch_concurrently(
                    pending,
                    lambda s: self._comparison_row(s, period, risk_free_rate, bills_future),
                    COMPARE_MAX_WORKERS,
                )
                bills = bills_future.result()
        rows = [r for r in built if isinstance(r, TickerComparisonRow)]
        errors.extend(r for r in built if isinstance(r, ComparisonError))
        base_currency = next((row.currency for row in rows if row.currency), None)
        for row in rows:
            row.currency_differs = row.currency is not None and row.currency != base_currency
        table_rate = _table_risk_free(risk_free_rate, bills)
        return TickerComparison(
            period=period,
            risk_free_rate=risk_free_rate,
            risk_free_rate_source=table_rate.source,
            risk_free_rate_note=table_rate.note,
            base_currency=base_currency,
            mixed_currencies=any(row.currency_differs for row in rows),
            rows=rows,
            errors=errors,
        )

    def _comparison_row(
        self,
        symbol: str,
        period: str,
        risk_free_rate: float | None,
        bills: Future[_Bills],
    ) -> TickerComparisonRow | ComparisonError:
        """One ticker's row, or the reason it has none.

        A history failure becomes a ComparisonError, since without performance there is
        nothing to compare. Valuation metrics are supplementary: a metrics failure keeps
        the row, with those fields null and the reason in metrics_error.
        """
        try:
            bars = self._all_bars(symbol, period, "1d")
            risk_free = _risk_free_over(risk_free_rate, bills.result(), bars[0].date, bars[-1].date)
            perf = _performance(symbol, period, bars, risk_free)
        except DataUnavailable as exc:
            return ComparisonError(symbol=symbol, error=str(exc))
        metrics: KeyMetrics | None = None
        metrics_error: str | None = None
        try:
            metrics = self.get_key_metrics(symbol)
        except DataUnavailable as exc:
            metrics_error = str(exc)
        return TickerComparisonRow(
            symbol=symbol,
            currency=metrics.currency if metrics else None,
            financial_currency=metrics.financial_currency if metrics else None,
            bars=perf.bars,
            start_date=perf.start_date,
            end_date=perf.end_date,
            total_return_percent=perf.total_return_percent,
            annualized_return_percent=perf.annualized_return_percent,
            annualized_volatility_percent=perf.annualized_volatility_percent,
            max_drawdown_percent=perf.max_drawdown_percent,
            sharpe_ratio=perf.sharpe_ratio,
            sortino_ratio=perf.sortino_ratio,
            calmar_ratio=perf.calmar_ratio,
            periods_per_year=perf.periods_per_year,
            risk_free_rate=perf.risk_free_rate,
            risk_free_rate_source=perf.risk_free_rate_source,
            risk_free_rate_note=perf.risk_free_rate_note,
            trailing_pe=metrics.trailing_pe if metrics else None,
            forward_pe=metrics.forward_pe if metrics else None,
            price_to_book=metrics.price_to_book if metrics else None,
            price_to_sales=metrics.price_to_sales if metrics else None,
            peg_ratio=metrics.peg_ratio if metrics else None,
            ev_to_ebitda=metrics.ev_to_ebitda if metrics else None,
            profit_margins=metrics.profit_margins if metrics else None,
            return_on_equity=metrics.return_on_equity if metrics else None,
            debt_to_equity=metrics.debt_to_equity if metrics else None,
            metrics_error=metrics_error,
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
        if not line_items:  # None or [] - an empty filter means the whole statement
            return full
        return _filter_line_items(full, line_items)

    def _fetch_financials(
        self, symbol: str, statement: Statement, period: StatementPeriod
    ) -> FinancialStatement:
        """The statement, labelled with its reporting currency once it has parsed."""
        parsed = self._fetch_statement(symbol, statement, period)
        return parsed.model_copy(update={"currency": self._statement_currency(symbol)})

    def _fetch_statement(
        self, symbol: str, statement: Statement, period: StatementPeriod
    ) -> FinancialStatement:
        """The parsed statement, without its currency (that comes from another endpoint)."""
        attr = _FINANCIALS_ATTR[(statement, period)]
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

    def _statement_currency(self, symbol: str) -> str | None:
        """The currency a statement is reported in, cached per symbol.

        It comes from ``.info``, a separate Yahoo endpoint shared by all six statements, so
        it has its own cache key. Best-effort: a failed read leaves the statement
        unlabelled rather than failing it, and is not cached.
        """
        try:
            return self._cached(
                ("statement_currency", symbol),
                self._fundamentals_ttl,
                lambda: _read_statement_currency(self._ticker(symbol)),
            )
        except Exception:  # labelling is best-effort, never fatal
            return None

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        symbol = _norm(symbol)
        return self._cached(
            ("profile", symbol), self._fundamentals_ttl, lambda: self._fetch_profile(symbol)
        )

    def _fetch_profile(self, symbol: str) -> CompanyProfile:
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

    def get_key_metrics(self, symbol: str) -> KeyMetrics:
        symbol = _norm(symbol)
        return self._cached(
            ("metrics", symbol), self._fundamentals_ttl, lambda: self._fetch_metrics(symbol)
        )

    def _fetch_metrics(self, symbol: str) -> KeyMetrics:
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

    def get_analyst_data(self, symbol: str) -> AnalystData:
        symbol = _norm(symbol)
        return self._cached(
            ("analyst", symbol),
            self._fundamentals_ttl,
            lambda: self._fetch_analyst(symbol),
        )

    def _fetch_analyst(self, symbol: str) -> AnalystData:
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

    def get_news(self, symbol: str, count: int = 10) -> NewsResult:
        symbol = _norm(symbol)
        return self._cached(
            ("news", symbol, str(count)),
            self._history_ttl,
            lambda: self._fetch_news(symbol, count),
            # Unflagged news from an identity OUTAGE is retried, not pinned; a symbol Yahoo
            # simply has no name for is a lasting answer and is cached like any other.
            cacheable=lambda result: result.relevance_check != "unavailable",
        )

    def _fetch_news(self, symbol: str, count: int) -> NewsResult:
        # The identity behind the relevance flags is an independent request.
        (articles, source), identity = _in_parallel(
            lambda: self._news_articles(symbol, count),
            lambda: self._identity_or_gap(symbol),
        )
        check, note = _flag_mentions(articles, symbol, identity)
        return NewsResult(
            symbol=symbol,
            articles=articles,
            source=source,
            relevance_check=check,
            relevance_note=note,
        )

    def _news_articles(self, symbol: str, count: int) -> tuple[list[NewsArticle], NewsSource]:
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

    def _identity_or_gap(self, symbol: str) -> _Identity | _IdentityGap:
        """The symbol's instrument type and names, or why Yahoo could not supply them.

        Only the relevance flags depend on this, so a failure degrades them to null rather
        than failing the news call. SymbolNotFound (no info, or no name in it) is a lasting
        answer about the symbol; any other DataUnavailable is an outage worth retrying.
        """
        try:
            return self._cached(
                ("identity", symbol), self._fundamentals_ttl, lambda: self._fetch_identity(symbol)
            )
        except SymbolNotFound as exc:
            return _IdentityGap(str(exc), lasting=True)
        except DataUnavailable as exc:
            return _IdentityGap(str(exc), lasting=False)

    def _fetch_identity(self, symbol: str) -> _Identity:
        _, info = self._ticker_with_info(symbol, "company identity", "identity")
        return _Identity(
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

    def search_symbols(self, query: str, max_results: int = 8) -> SymbolSearchResult:
        return self._cached(
            ("search", query, str(max_results)),
            self._fundamentals_ttl,
            lambda: self._fetch_search(query, max_results),
        )

    def _fetch_search(self, query: str, max_results: int) -> SymbolSearchResult:
        with _unavailable_on_error(f"Search failed for '{query}'"):
            result = self._search(query, max_results=max_results, news_count=0, lists_count=0)
            quotes = result.quotes
        if not quotes:
            return SymbolSearchResult(query=query, matches=[])
        with _unavailable_on_error(f"Failed to parse search results for '{query}'"):
            matches = [_symbol_match(q) for q in quotes if q.get("symbol")]
            return SymbolSearchResult(query=query, matches=matches)


def _normalize_batch(symbols: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Normalize and de-duplicate a batch in request order.

    Returns the symbols to fetch and a (raw, reason) pair for each one that could not be
    normalized, which therefore never reaches the network.
    """
    pending: list[str] = []
    failures: list[tuple[str, str]] = []
    for raw in symbols:
        try:
            symbol = _norm(raw)
        except DataUnavailable as exc:
            failures.append((raw, str(exc)))
            continue
        if symbol not in pending:
            pending.append(symbol)
    return pending, failures


def _fetch_concurrently[T](
    items: list[str], fetch: Callable[[str], T], max_workers: int
) -> list[T]:
    """Run ``fetch`` over ``items`` in parallel, returning results in input order."""
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fetch, items))


def _performance(
    symbol: str, period: str, bars: list[PriceBar], risk_free: _RiskFree
) -> PerformanceStats:
    """Return and risk statistics for ``bars``, with the rate-dependent set at ``risk_free``."""
    if len(bars) < 2:
        raise DataUnavailable(f"Not enough price history to compute performance for '{symbol}'.")
    closes = [b.close for b in bars]
    # Annualize off calendar time, not the bar count: bars per year depends on the
    # trading calendar (~252 weekday, ~365 for crypto).
    elapsed_days = _elapsed_days(bars[0].date, bars[-1].date)
    annualized_return: float | None = None
    annualized_volatility: float | None = None
    periods_per_year: float | None = None
    # The risk-adjusted figures need periods_per_year (Calmar the CAGR), so they share
    # the annualization gate; Sharpe, Sortino and downside also need a rate.
    sharpe: float | None = None
    sortino: float | None = None
    downside: float | None = None
    calmar: float | None = None
    max_drawdown = analytics.max_drawdown(closes)
    if elapsed_days >= MIN_ANNUALIZATION_DAYS:
        years = elapsed_days / analytics.DAYS_PER_YEAR
        periods_per_year = analytics.infer_periods_per_year(len(bars), years)
        annualized_return = analytics.annualized_return(closes, years)
        annualized_volatility = analytics.annualized_volatility(closes, periods_per_year)
        if risk_free.rate is not None:
            sharpe = analytics.sharpe_ratio(closes, periods_per_year, risk_free.rate)
            sortino = analytics.sortino_ratio(closes, periods_per_year, risk_free.rate)
            downside = analytics.downside_deviation(closes, periods_per_year, risk_free.rate)
        calmar = analytics.calmar_ratio(annualized_return, max_drawdown)
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
        max_drawdown_percent=max_drawdown,
        risk_free_rate=risk_free.rate,
        risk_free_rate_source=risk_free.source,
        risk_free_rate_note=risk_free.note,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        downside_deviation_percent=downside,
        calmar_ratio=calmar,
        sma_50=analytics.sma(closes, SMA_SHORT_WINDOW),
        sma_200=analytics.sma(closes, SMA_LONG_WINDOW),
    )


_HOW_TO_PROCEED = (
    "Pass risk_free_rate explicitly to get the rate-dependent figures (0 gives raw "
    "return per unit of risk)."
)


def _bills_fetch_failed(reason: str) -> _RiskFree:
    """No default rate because the T-bill history could not be fetched: worth a retry."""
    return _RiskFree(
        None,
        "unavailable",
        f"The 13-week T-bill yield ({TREASURY_BILL_SYMBOL}) could not be fetched: {reason}. "
        f"{_HOW_TO_PROCEED}",
        retryable=True,
    )


def _table_risk_free(risk_free_rate: float | None, bills: _Bills) -> _RiskFree:
    """compare_tickers' header: the caller's rate, or whether the one T-bill fetch worked.

    Window coverage is judged per row, so a header over a successful fetch says
    "treasury_bill" and leaves any row-level gap to that row's own source and note.
    """
    if risk_free_rate is not None:
        return _RiskFree(risk_free_rate, "caller")
    if isinstance(bills, str):
        return _bills_fetch_failed(bills)
    return _RiskFree(None, "treasury_bill")


def _rate_key(risk_free_rate: float | None) -> str:
    """Cache-key component: the caller's rate, or a marker for the T-bill default."""
    return "treasury_bill" if risk_free_rate is None else str(risk_free_rate)


def _risk_free_over(risk_free_rate: float | None, bills: _Bills, start: str, end: str) -> _RiskFree:
    """The caller's rate, else the mean effective T-bill rate over ``start``..``end``.

    The window average, not today's yield: a Sharpe over 2021-2026 measured against a 4%
    hurdle would charge the years when bills paid nothing as if they had paid 4%. The
    history must reach both ends of the window (within a holiday's tolerance), since
    averaging only part of it would misstate the cash return foregone.
    """
    if risk_free_rate is not None:
        return _RiskFree(risk_free_rate, "caller")
    # With no rate given, _bills_for always fetched: bills is the history or its error.
    if not isinstance(bills, list):
        return _bills_fetch_failed(str(bills))
    first_day, last_day = _day(start), _day(end)
    inside = [b for b in bills if first_day <= _day(b.date) <= last_day]
    tolerance = RISK_FREE_EDGE_TOLERANCE_DAYS
    if (
        not inside
        or (_day(inside[0].date) - first_day).days > tolerance
        or (last_day - _day(inside[-1].date)).days > tolerance
    ):
        return _RiskFree(
            None,
            "unavailable",
            f"The 13-week T-bill history ({TREASURY_BILL_SYMBOL}) covers {bills[0].date} to "
            f"{bills[-1].date}, which does not span the measured window {start} to {end}. "
            f"{_HOW_TO_PROCEED}",
        )
    try:
        rates = [analytics.treasury_bill_effective_rate(b.close) for b in inside]
    except InvalidInput as exc:
        return _RiskFree(
            None,
            "unavailable",
            f"The 13-week T-bill history ({TREASURY_BILL_SYMBOL}) has an implausible "
            f"quote: {exc} {_HOW_TO_PROCEED}",
        )
    return _RiskFree(statistics.fmean(rates), "treasury_bill")


def _day(timestamp: str) -> date:
    """The calendar date of a PriceBar date or intraday timestamp.

    ``datetime.fromisoformat`` rather than ``date.fromisoformat``: only it accepts both a
    bare date ("2024-01-01") and a timestamp with an offset.
    """
    return datetime.fromisoformat(timestamp).date()


def _in_parallel[A, B](first: Callable[[], A], second: Callable[[], B]) -> tuple[A, B]:
    """Run two independent blocking fetches at once; either one's exception propagates.

    If both fail, the first error is raised with the second attached as a note, so
    neither is lost.
    """
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first)
        second_future = pool.submit(second)
        try:
            first_result = first_future.result()
        except Exception as exc:
            second_error = second_future.exception()
            if second_error is not None:
                exc.add_note(f"The fetch run alongside it also failed: {second_error!r}")
            raise
        return first_result, second_future.result()


def _flag_mentions(
    articles: list[NewsArticle], symbol: str, identity: _Identity | _IdentityGap
) -> tuple[RelevanceCheck, str | None]:
    """Set each article's mentions_company; say whether that could be assessed, and why not."""
    if isinstance(identity, _IdentityGap):
        return ("no_company_name" if identity.lasting else "unavailable"), identity.reason
    # Only a company has a name a headline can omit; for an ETF, index, fund, coin or
    # currency pair the market-wide stories are the relevant ones.
    if identity.quote_type != "EQUITY":
        return "not_an_equity", None
    names = relevance.company_aliases(identity.long_name, identity.short_name)
    symbols = relevance.symbol_aliases(symbol)
    for article in articles:
        text = f"{article.title} {article.summary or ''}"
        article.mentions_company = relevance.mentions_company(text, names, symbols)
    return "applied", None


def _elapsed_days(start: str, end: str) -> int:
    """Calendar days between two PriceBar dates."""
    return (_day(end) - _day(start)).days


def _label_key(label: str) -> str:
    """Comparison key for a line-item label: case- and whitespace-insensitive."""
    return " ".join(label.split()).casefold()


def _filter_line_items(full: FinancialStatement, requested: list[str]) -> FinancialStatement:
    """Narrow a statement to the requested labels, reporting whatever did not match.

    Labels match case- and whitespace-insensitively ('total revenue' finds 'Total
    Revenue'). Unmatched ones are reported in missing_line_items with close-match
    suggestions and the full label list, rather than dropped silently.
    """
    available = list(full.line_items)
    by_key = {_label_key(label): label for label in available}
    wanted: dict[str, list[float | None]] = {}
    missing: list[str] = []
    for label in requested:
        match = by_key.get(_label_key(label))
        if match is None:
            if label not in missing:
                missing.append(label)
        else:
            wanted.setdefault(match, full.line_items[match])
    suggestions = {}
    for label in missing:
        close = difflib.get_close_matches(_label_key(label), list(by_key), n=3, cutoff=0.6)
        if close:
            suggestions[label] = [by_key[key] for key in close]
    return full.model_copy(
        update={
            "line_items": wanted,
            "missing_line_items": missing,
            "available_line_items": available if missing else [],
            "line_item_suggestions": suggestions,
        }
    )


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


def _norm(symbol: str) -> str:
    """Upper-case and strip a ticker, so equivalent spellings share one cache entry."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise SymbolNotFound("Empty ticker symbol.")
    return normalized


# Signals that genuinely mean "Yahoo has no data for this symbol". KeyError is what
# fast_info leaks for an unknown symbol; YFTickerMissingError covers yfinance's own
# missing-ticker/timezone/prices errors (YFTzMissingError and YFPricesMissingError
# subclass it). Anything outside this set is treated as a source/transport failure.
_NO_DATA_ERRORS = (KeyError, YFTickerMissingError)


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
