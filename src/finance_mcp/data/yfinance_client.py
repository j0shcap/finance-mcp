"""Cached, error-surfacing wrapper over yfinance. No MCP imports.

Network access is isolated here. yfinance errors and empty results become
DataUnavailable/SymbolNotFound whose message is surfaced to the caller verbatim,
because we cannot enumerate every Yahoo failure mode. SymbolNotFound is reserved for
signals that really mean "no data for this symbol" (see _is_no_data_error): everything
else, transport failures included, stays a plain DataUnavailable.
"""

import difflib
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, cast

import yfinance as yf
from yfinance.exceptions import YFTickerMissingError

from finance_mcp.data import analytics
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
    PerformanceStats,
    PriceBar,
    PriceHistory,
    PriceSummary,
    Quote,
    QuoteError,
    QuoteResult,
    RecommendationPeriod,
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
# The LRU bounds how many entries are held, not how large they are. A period="max" daily
# history is ~11.5k PriceBar models (~9 MB), so 256 of those would retain gigabytes. Bar
# lists longer than this are still returned in full; they are just not kept. ~2000 daily
# bars is about eight years, so every ordinary window stays cached.
MAX_CACHEABLE_BARS = 2000
# Quotes in a batch are independent single requests, so they are fetched in parallel; the
# bound keeps a large batch from opening a connection per ticker at once.
QUOTE_MAX_WORKERS = 8
# A benchmark comparison is two independent history fetches; both go through
# _fetch_concurrently under this bound.
BENCHMARK_MAX_WORKERS = 2
# The fewest shared closes that yield a single return to compare.
MIN_OVERLAP_OBSERVATIONS = 2
# Each comparison row costs two Yahoo calls (history + info), so the worker bound is lower
# than the quote bound for the same ceiling on concurrent connections.
COMPARE_MAX_WORKERS = 5
# Intervals whose bars are points in time rather than whole sessions. Kept in sync with
# HistoryInterval (a test pins it): everything that is not a daily-or-longer interval.
_INTRADAY_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h"})
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
# Below roughly a quarter of calendar time, annualizing compounds short-run noise into a
# yearly figure that reads as a forecast (a 4-day AAPL move once reported as +47.3%/yr).
# Set just under three months rather than at it: a period="3mo" window spans 87-95 elapsed
# days depending on the call date, so a 90-day gate cut through that range and made the
# annualized fields blink in and out for the same request.
MIN_ANNUALIZATION_DAYS = 85

_FINANCIALS_ATTR = {
    ("income", "annual"): "income_stmt",
    ("income", "quarterly"): "quarterly_income_stmt",
    ("balance", "annual"): "balance_sheet",
    ("balance", "quarterly"): "quarterly_balance_sheet",
    ("cashflow", "annual"): "cashflow",
    ("cashflow", "quarterly"): "quarterly_cashflow",
}


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
        # yf.Search is called with keyword args (max_results/news_count/lists_count);
        # widen to Callable[..., Any] so those kwargs typecheck.
        self._search: Callable[..., Any] = search_factory
        self._now = time_fn
        self._quote_ttl = quote_ttl
        self._history_ttl = history_ttl
        self._fundamentals_ttl = fundamentals_ttl
        self._max_bars = max_bars
        self._cache_max_entries = cache_max_entries
        # key -> (stored_at, ttl, value). Insertion order is LRU order (oldest use first);
        # the per-entry ttl is stored so the purge pass can judge expiry without knowing
        # which caller wrote the entry.
        self._cache: OrderedDict[tuple[str, ...], tuple[float, float, Any]] = OrderedDict()
        # get_quote fetches concurrently, so cache bookkeeping is guarded. Fetches run
        # OUTSIDE the lock: two threads racing on one uncached key just fetch it twice.
        self._cache_lock = threading.Lock()

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
                    self._cache.move_to_end(key)  # most recently used
                    # The cache is heterogeneous (Any value); the key space guarantees each
                    # key always maps to the same T, so this single cast is the only one needed.
                    return cast(T, hit[2])
                del self._cache[key]  # stale: drop before refetching
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
                self._cache.popitem(last=False)  # evict the least recently used entry
        return value

    def _purge_expired(self, now: float) -> None:
        """Drop every entry past its own TTL, so stale keys cannot squat on the bound."""
        for key in [k for k, (stored, ttl, _) in self._cache.items() if now - stored >= ttl]:
            del self._cache[key]

    def _ticker_with_info(
        self, symbol: str, fetch_label: str, kind: str
    ) -> tuple[Any, dict[str, Any]]:
        """Fetch a ticker and its ``.info``, asserting the symbol names a real instrument.

        Shared by the profile/metrics/analyst fetchers. Access errors are classified by
        _data_error; an ``info`` dict that is empty or has no longName/shortName (Yahoo's
        tell for an unknown symbol) becomes SymbolNotFound.
        """
        ticker = self._ticker(symbol)
        try:
            info = ticker.info
        except Exception as exc:
            raise _data_error(exc, fetch_label, kind, symbol) from exc
        if not info or not (info.get("longName") or info.get("shortName")):
            raise SymbolNotFound(_no_data_msg(kind, symbol))
        return ticker, info

    def _normalize_batch(self, symbols: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Normalize, de-duplicate and order a batch; returns (pending, (raw, reason) pairs).

        Shared by get_quote and compare_tickers, which wrap the failures in their own error
        models. A symbol that cannot even be normalized never reaches the network.
        """
        pending: list[str] = []  # normalized, de-duped, in request order
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

    def get_quote(self, symbols: list[str]) -> QuoteResult:
        """Fetch quotes for a batch of symbols concurrently, with partial results.

        A batch is a set of independent lookups, so one unknown or unreachable ticker
        reports itself in ``errors`` instead of discarding the quotes that did work.
        """
        pending, failures = self._normalize_batch(symbols)
        errors: list[QuoteError] = [
            QuoteError(symbol=raw, error=reason) for raw, reason in failures
        ]
        if not pending:
            return QuoteResult(quotes=[], errors=errors)
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
        """Parsed bars for one (symbol, period, interval), cached once for every consumer.

        get_price_history and analyze_performance are two views of the same fetch; keying the
        raw bars separately from the derived models keeps them on a single network round-trip.

        Histories longer than MAX_CACHEABLE_BARS are not retained, so a very long window
        costs one fetch per view instead of retaining megabytes of bars. Both views cache
        their own small derived result, so repeat calls still avoid the network.
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
                        date=_bar_date(idx, intraday), open=o, high=h, low=low, close=c, volume=v
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

    def analyze_performance(
        self, symbol: str, period: str, risk_free_rate: float = 0.0
    ) -> PerformanceStats:
        symbol = _norm(symbol)
        # The bars this reads are usually cached by _all_bars, but a history past
        # MAX_CACHEABLE_BARS is not retained -- without an entry here every call to a long
        # window would go back to the network. PerformanceStats is a few hundred bytes.
        # risk_free_rate is part of the key because the Sharpe, Sortino and downside figures
        # are computed from it: keying on (symbol, period) alone would serve the first
        # caller's rate to every later one.
        return self._cached(
            ("performance", symbol, period, str(risk_free_rate)),
            self._history_ttl,
            lambda: self._compute_performance(symbol, period, risk_free_rate),
        )

    def _compute_performance(
        self, symbol: str, period: str, risk_free_rate: float = 0.0
    ) -> PerformanceStats:
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
        # Every risk-adjusted figure is scaled by periods_per_year (Calmar needs the CAGR),
        # so they all sit behind the same gate rather than falling back on a fixed 252.
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
            sharpe = analytics.sharpe_ratio(closes, periods_per_year, risk_free_rate)
            sortino = analytics.sortino_ratio(closes, periods_per_year, risk_free_rate)
            downside = analytics.downside_deviation(closes, periods_per_year, risk_free_rate)
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
            risk_free_rate=risk_free_rate,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            downside_deviation_percent=downside,
            calmar_ratio=calmar,
            sma_50=analytics.sma(closes, SMA_SHORT_WINDOW),
            sma_200=analytics.sma(closes, SMA_LONG_WINDOW),
        )

    def compare_to_benchmark(
        self, symbol: str, benchmark: str, period: str, risk_free_rate: float = 0.0
    ) -> BenchmarkComparison:
        """Benchmark-relative statistics over the dates the two instruments share.

        The two histories are independent fetches, so they run in parallel; both are the
        same cached ``_all_bars`` entries the other analytics tools use.
        """
        symbol, bench = _norm(symbol), _norm(benchmark)
        if symbol == bench:
            raise InvalidInput(
                f"A benchmark comparison needs two different symbols; '{symbol}' was given "
                "for both. Use analyze_performance for a single instrument."
            )
        asset_bars, bench_bars = _fetch_concurrently(
            [symbol, bench],
            lambda s: self._all_bars(s, period, "1d"),
            BENCHMARK_MAX_WORKERS,
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
        elapsed_days = _elapsed_days(dates[0], dates[-1])
        periods_per_year: float | None = None
        asset_cagr: float | None = None
        bench_cagr: float | None = None
        tracking: float | None = None
        info_ratio: float | None = None
        # Beta and correlation are unit-free and need no calendar, so they are computed
        # unconditionally; everything annualized sits behind the same 90-day gate
        # analyze_performance uses.
        asset_beta = analytics.beta(asset_closes, bench_closes)
        if elapsed_days >= MIN_ANNUALIZATION_DAYS:
            years = elapsed_days / analytics.DAYS_PER_YEAR
            periods_per_year = analytics.infer_periods_per_year(len(dates), years)
            asset_cagr = analytics.annualized_return(asset_closes, years)
            bench_cagr = analytics.annualized_return(bench_closes, years)
            tracking = analytics.tracking_error(asset_closes, bench_closes, periods_per_year)
            info_ratio = analytics.information_ratio(asset_closes, bench_closes, periods_per_year)
        alpha: float | None = None
        if asset_beta is not None and asset_cagr is not None and bench_cagr is not None:
            alpha = analytics.jensen_alpha(asset_cagr, bench_cagr, asset_beta, risk_free_rate)
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
            risk_free_rate=risk_free_rate,
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
        self, symbols: list[str], period: str, risk_free_rate: float = 0.0
    ) -> TickerComparison:
        """Side-by-side performance and valuation for a small batch, fetched concurrently.

        Each row is two independent lookups over the cached fetchers the single-ticker tools
        already use, so a batch costs no more than calling them one at a time -- and one
        ticker's failure reports itself instead of discarding the rows that worked.
        """
        pending, failures = self._normalize_batch(symbols)
        errors = [ComparisonError(symbol=raw, error=reason) for raw, reason in failures]
        built = _fetch_concurrently(
            pending,
            lambda s: self._comparison_row(s, period, risk_free_rate),
            COMPARE_MAX_WORKERS,
        )
        rows = [r for r in built if isinstance(r, TickerComparisonRow)]
        errors.extend(r for r in built if isinstance(r, ComparisonError))
        base_currency = next((row.currency for row in rows if row.currency), None)
        for row in rows:
            row.currency_differs = row.currency is not None and row.currency != base_currency
        return TickerComparison(
            period=period,
            risk_free_rate=risk_free_rate,
            base_currency=base_currency,
            mixed_currencies=any(row.currency_differs for row in rows),
            rows=rows,
            errors=errors,
        )

    def _comparison_row(
        self, symbol: str, period: str, risk_free_rate: float
    ) -> TickerComparisonRow | ComparisonError:
        """One ticker's row, or the reason it has none.

        Performance is the row's backbone: without it there is nothing to compare, so a
        history failure becomes a ComparisonError. Valuation metrics are supplementary, so a
        metrics failure leaves the row in place with those fields null and the reason in
        metrics_error -- dropping a whole row because Yahoo's ``info`` blipped would lose the
        return figures that did arrive.
        """
        try:
            perf = self._compute_performance(symbol, period, risk_free_rate)
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
        self,
        symbol: str,
        statement: Statement,
        period: StatementPeriod,
    ) -> FinancialStatement:
        attr = _FINANCIALS_ATTR[(statement, period)]
        ticker = self._ticker(symbol)
        try:
            df = getattr(ticker, attr)
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
                currency=self._statement_currency(symbol, ticker),
                period_ends=period_ends,
                line_items=line_items,
            )
        except Exception as exc:  # surface any parsing failure verbatim
            raise DataUnavailable(
                f"Failed to parse {statement} statement for '{symbol}': {exc}"
            ) from exc

    def _statement_currency(self, symbol: str, ticker: Any) -> str | None:
        """The currency a statement is reported in, cached per symbol.

        It comes from ``.info``, a different Yahoo endpoint than the statement itself, and
        is the same for all six statement/period combinations — so it is cached under its
        own key rather than re-requested per statement against a rate-limited source.
        Best-effort: a failed read leaves the statement unlabelled instead of failing it,
        and is NOT cached, because a rate limit says nothing about the reporting currency.
        """
        try:
            return self._cached(
                ("statement_currency", symbol),
                self._fundamentals_ttl,
                lambda: _read_statement_currency(ticker),
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


def _fetch_concurrently[T](
    items: list[str], fetch: Callable[[str], T], max_workers: int
) -> list[T]:
    """Run ``fetch`` over ``items`` in parallel, preserving input order.

    Independent single-symbol lookups have no reason to serialize. ``pool.map`` keeps the
    results in request order so callers can pair them back up positionally.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as pool:
        return list(pool.map(fetch, items))


def _elapsed_days(start: str, end: str) -> int:
    """Calendar days between two PriceBar dates.

    Parses via ``datetime.fromisoformat`` rather than ``date.fromisoformat`` so it accepts
    both a bare date ("2024-01-01") and a full intraday timestamp with offset.
    """
    return (datetime.fromisoformat(end).date() - datetime.fromisoformat(start).date()).days


def _label_key(label: str) -> str:
    """Comparison key for a line-item label: case- and whitespace-insensitive."""
    return " ".join(label.split()).casefold()


def _filter_line_items(full: FinancialStatement, requested: list[str]) -> FinancialStatement:
    """Narrow a statement to the requested labels, reporting whatever did not match.

    Labels are matched on _label_key, so 'total revenue' finds 'Total Revenue' (the exact
    Yahoo spelling is easy to get almost right). Anything still unmatched is reported in
    missing_line_items with close-match suggestions and the full label list, instead of
    being dropped silently and leaving the caller to wonder why the statement is short.
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

    Intraday bars are moments, so they keep the clock time and the exchange's UTC offset
    (2026-09-25T09:35:00-04:00). Daily and longer bars are whole sessions indexed at
    midnight in the exchange's timezone, so they stay date-only — emitting the timestamp
    would imply a trade time, and normalizing it to UTC would shift the calendar date.
    """
    return str(idx.isoformat() if intraday else idx.date().isoformat())


def _norm(symbol: str) -> str:
    """Normalize a ticker so equivalent spellings share one cache entry and one fetch.

    Yahoo symbols are upper-case; callers routinely pass 'aapl' or 'AAPL '. Normalizing
    here (before the cache key is built) is also what makes the echoed symbol canonical.
    """
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

    A connection reset, DNS failure, timeout, HTTP 5xx, rate limit or malformed payload
    says nothing about the symbol, so it stays a DataUnavailable carrying the underlying
    message; only the no-data signals become SymbolNotFound.
    """
    if _is_no_data_error(exc):
        return SymbolNotFound(_no_data_msg(kind, symbol))
    return DataUnavailable(f"Failed to fetch {fetch_label} for '{symbol}': {exc}")


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
