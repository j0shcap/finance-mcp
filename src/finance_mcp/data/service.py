"""The logic layer the tools call: caching and orchestration over the provider ports.

It decides what is fetched together (an asset with its benchmark and the T-bill history),
what is cached and for how long, and assembles results from the pure modules
(performance, risk_free, relevance). It knows providers only through providers/ports.py,
so nothing here is provider-specific; server.py chooses the providers.
"""

import difflib
import time
from collections.abc import Callable
from concurrent.futures import Future

from finance_mcp.data.cache import CacheKey, TTLCache
from finance_mcp.data.concurrency import in_background, in_parallel, map_concurrently
from finance_mcp.data.errors import DataUnavailable, InvalidInput, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    BenchmarkComparison,
    CompanyProfile,
    ComparisonError,
    Earnings,
    FinancialStatement,
    Identity,
    KeyMetrics,
    NewsResult,
    PerformanceStats,
    PriceBar,
    PriceHistory,
    PriceSummary,
    Quote,
    QuoteError,
    QuoteResult,
    Statement,
    StatementPeriod,
    SymbolSearchResult,
    TickerComparison,
    TickerComparisonRow,
)
from finance_mcp.data.performance import benchmark_comparison, comparison_row, performance
from finance_mcp.data.providers.ports import EarningsProvider, MarketDataProvider, NewsProvider
from finance_mcp.data.relevance import IdentityGap, flag_mentions
from finance_mcp.data.risk_free import (
    Bills,
    RiskFree,
    rate_key,
    risk_free_over,
    table_risk_free,
)

DEFAULT_MAX_BARS = 260
DEFAULT_CACHE_MAX_ENTRIES = 256
# The LRU bounds how many entries are held, not how large they are: a period="max" daily
# history is ~11.5k bars (~9 MB). Longer bar lists are returned in full but not kept;
# ~2000 daily bars is about eight years, so every ordinary window stays cached.
MAX_CACHEABLE_BARS = 2000
# Threads per batch. A provider may also bound its own requests across all concurrent tool
# calls (see server.py).
QUOTE_MAX_WORKERS = 8
# Each comparison row makes two provider calls (history + metrics), so fewer run at once.
COMPARE_MAX_WORKERS = 5


class DataService:
    """Market data and news over provider ports, with a per-key TTL cache (bounded, LRU)."""

    def __init__(
        self,
        market: MarketDataProvider,
        news: NewsProvider,
        earnings: EarningsProvider,
        *,
        time_fn: Callable[[], float] = time.monotonic,
        quote_ttl: float = 30.0,
        history_ttl: float = 300.0,
        fundamentals_ttl: float = 3600.0,
        max_bars: int = DEFAULT_MAX_BARS,
        cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
    ) -> None:
        self._market = market
        self._news = news
        self._earnings = earnings
        self._quote_ttl = quote_ttl
        self._history_ttl = history_ttl
        self._fundamentals_ttl = fundamentals_ttl
        self._max_bars = max_bars
        self._cache = TTLCache(time_fn, cache_max_entries)

    def _cached[T](
        self,
        key: CacheKey,
        ttl: float,
        fetch: Callable[[], T],
        cacheable: Callable[[T], bool] | None = None,
    ) -> T:
        return self._cache.get_or_fetch(key, ttl, fetch, cacheable)

    def get_quote(self, symbols: list[str]) -> QuoteResult:
        """Fetch quotes for a batch of symbols concurrently, with partial results.

        A batch is a set of independent lookups, so one unknown or unreachable ticker
        reports itself in ``errors`` instead of discarding the quotes that did work.
        """
        pending, failures = _normalize_batch(symbols)
        errors = [QuoteError(symbol=raw, error=reason) for raw, reason in failures]
        fetched = map_concurrently(pending, self._quote_or_error, QUOTE_MAX_WORKERS)
        errors.extend(r for r in fetched if isinstance(r, QuoteError))
        return QuoteResult(quotes=[r for r in fetched if isinstance(r, Quote)], errors=errors)

    def _quote_or_error(self, symbol: str) -> Quote | QuoteError:
        """One symbol's cached quote, or the reason it could not be fetched."""
        try:
            return self._cached(
                ("quote", symbol), self._quote_ttl, lambda: self._market.quote(symbol)
            )
        except DataUnavailable as exc:
            return QuoteError(symbol=symbol, error=str(exc))

    def get_price_history(self, symbol: str, period: str, interval: str) -> PriceHistory:
        symbol = _norm(symbol)
        return self._cached(
            ("history", symbol, period, interval),
            self._history_ttl,
            lambda: self._fetch_history(symbol, period, interval),
        )

    def _treasury_bills(self, period: str) -> list[PriceBar]:
        """The T-bill yields behind a default risk-free rate; long histories, as in _all_bars,
        are not retained."""
        return self._cached(
            ("treasury_bills", period),
            self._history_ttl,
            lambda: self._market.treasury_bill_yields(period),
            cacheable=_cacheable_bars,
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
            lambda: self._market.bars(symbol, period, interval),
            cacheable=_cacheable_bars,
        )

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
            ("performance", symbol, period, rate_key(risk_free_rate)),
            self._history_ttl,
            lambda: self._fetch_performance(symbol, period, risk_free_rate),
            cacheable=lambda fetched: not fetched[1].retryable,
        )
        return stats

    def _fetch_performance(
        self, symbol: str, period: str, risk_free_rate: float | None
    ) -> tuple[PerformanceStats, RiskFree]:
        bars, bills = in_parallel(
            lambda: self._all_bars(symbol, period, "1d"),
            lambda: self._bills_for(period, risk_free_rate),
        )
        risk_free = risk_free_over(risk_free_rate, bills, bars[0].date, bars[-1].date)
        return performance(symbol, period, bars, risk_free), risk_free

    def _bills_for(self, period: str, risk_free_rate: float | None) -> Bills:
        """The T-bill history for a default rate; nothing when the caller gave a rate.

        A failed fetch is returned as its message rather than raised: only the rate-
        dependent figures need it, so it must not fail the whole computation.
        """
        if risk_free_rate is not None:
            return None
        try:
            return self._treasury_bills(period)
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
        (asset_bars, bench_bars), bills = in_parallel(
            lambda: in_parallel(
                lambda: self._all_bars(symbol, period, "1d"),
                lambda: self._all_bars(bench, period, "1d"),
            ),
            lambda: self._bills_for(period, risk_free_rate),
        )
        return benchmark_comparison(
            symbol, bench, period, asset_bars, bench_bars, risk_free_rate, bills
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
        bills: Bills = None
        built: list[TickerComparisonRow | ComparisonError] = []
        if pending:
            # Started alongside the rows; each row waits for it only after its own bars.
            with in_background(lambda: self._bills_for(period, risk_free_rate)) as bills_future:
                built = map_concurrently(
                    pending,
                    lambda s: self._ticker_row(s, period, risk_free_rate, bills_future),
                    COMPARE_MAX_WORKERS,
                )
                bills = bills_future.result()
        rows = [r for r in built if isinstance(r, TickerComparisonRow)]
        errors.extend(r for r in built if isinstance(r, ComparisonError))
        base_currency = next((row.currency for row in rows if row.currency), None)
        for row in rows:
            row.currency_differs = row.currency is not None and row.currency != base_currency
        table_rate = table_risk_free(risk_free_rate, bills)
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

    def _ticker_row(
        self,
        symbol: str,
        period: str,
        risk_free_rate: float | None,
        bills: Future[Bills],
    ) -> TickerComparisonRow | ComparisonError:
        """One ticker's row, or the reason it has none.

        A history failure becomes a ComparisonError, since without performance there is
        nothing to compare. Valuation metrics are supplementary: a metrics failure keeps
        the row, with those fields null and the reason in metrics_error.
        """
        try:
            bars = self._all_bars(symbol, period, "1d")
            risk_free = risk_free_over(risk_free_rate, bills.result(), bars[0].date, bars[-1].date)
            perf = performance(symbol, period, bars, risk_free)
        except DataUnavailable as exc:
            return ComparisonError(symbol=symbol, error=str(exc))
        metrics: KeyMetrics | None = None
        metrics_error: str | None = None
        try:
            metrics = self.get_key_metrics(symbol)
        except DataUnavailable as exc:
            metrics_error = str(exc)
        return comparison_row(perf, metrics, metrics_error)

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
        parsed = self._market.financial_statement(symbol, statement, period)
        return parsed.model_copy(update={"currency": self._statement_currency(symbol)})

    def _statement_currency(self, symbol: str) -> str | None:
        """The currency a statement is reported in, cached per symbol.

        It may come from a separate provider request shared by all six statements, so it
        has its own cache key. Best-effort: a failed read leaves the statement unlabelled
        rather than failing it, and is not cached.
        """
        try:
            return self._cached(
                ("statement_currency", symbol),
                self._fundamentals_ttl,
                lambda: self._market.statement_currency(symbol),
            )
        except DataUnavailable:  # labelling is best-effort, never fatal
            return None

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        symbol = _norm(symbol)
        return self._cached(
            ("profile", symbol), self._fundamentals_ttl, lambda: self._market.profile(symbol)
        )

    def get_key_metrics(self, symbol: str) -> KeyMetrics:
        symbol = _norm(symbol)
        return self._cached(
            ("metrics", symbol), self._fundamentals_ttl, lambda: self._market.key_metrics(symbol)
        )

    def get_analyst_data(self, symbol: str) -> AnalystData:
        symbol = _norm(symbol)
        return self._cached(
            ("analyst", symbol),
            self._fundamentals_ttl,
            lambda: self._market.analyst_data(symbol),
        )

    def get_earnings(self, symbol: str) -> Earnings:
        symbol = _norm(symbol)
        return self._cached(
            ("earnings", symbol),
            self._fundamentals_ttl,
            lambda: self._earnings.earnings(symbol),
        )

    def get_news(self, symbol: str, count: int = 10) -> NewsResult:
        symbol = _norm(symbol)
        return self._cached(
            ("news", symbol, str(count)),
            self._history_ttl,
            lambda: self._fetch_news(symbol, count),
            # Unflagged news from an identity OUTAGE is retried, not pinned; a symbol the
            # provider simply has no name for is a lasting answer and is cached like any other.
            cacheable=lambda result: result.relevance_check != "unavailable",
        )

    def _fetch_news(self, symbol: str, count: int) -> NewsResult:
        # The identity behind the relevance flags is an independent request.
        (articles, source), identity = in_parallel(
            lambda: self._news.news(symbol, count),
            lambda: self._identity_or_gap(symbol),
        )
        check, note = flag_mentions(articles, symbol, identity)
        return NewsResult(
            symbol=symbol,
            articles=articles,
            source=source,
            relevance_check=check,
            relevance_note=note,
        )

    def _identity_or_gap(self, symbol: str) -> Identity | IdentityGap:
        """The symbol's instrument type and names, or why the provider could not supply them.

        Only the relevance flags depend on this, so a failure degrades them to null rather
        than failing the news call. SymbolNotFound (no info, or no name in it) is a lasting
        answer about the symbol; any other DataUnavailable is an outage worth retrying.
        """
        try:
            return self._cached(
                ("identity", symbol), self._fundamentals_ttl, lambda: self._market.identity(symbol)
            )
        except SymbolNotFound as exc:
            return IdentityGap(str(exc), lasting=True)
        except DataUnavailable as exc:
            return IdentityGap(str(exc), lasting=False)

    def search_symbols(self, query: str, max_results: int = 8) -> SymbolSearchResult:
        return self._cached(
            ("search", query, str(max_results)),
            self._fundamentals_ttl,
            lambda: self._market.search(query, max_results),
            # No matches may be a source failure that looks like an answer: ask again.
            cacheable=lambda result: bool(result.matches),
        )


def _cacheable_bars(bars: list[PriceBar]) -> bool:
    return len(bars) <= MAX_CACHEABLE_BARS


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


def _norm(symbol: str) -> str:
    """Upper-case and strip a ticker, so equivalent spellings share one cache entry."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise SymbolNotFound("Empty ticker symbol.")
    return normalized
