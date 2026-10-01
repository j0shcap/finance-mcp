"""Return, risk and benchmark-relative statistics assembled into result models.

Pure: every function takes bars (and a resolved or resolvable risk-free rate) that the client
has already fetched, and returns a model.
"""

from finance_mcp.data import analytics
from finance_mcp.data.errors import DataUnavailable
from finance_mcp.data.models import (
    BenchmarkComparison,
    KeyMetrics,
    PerformanceStats,
    PriceBar,
    TickerComparisonRow,
)
from finance_mcp.data.risk_free import Bills, RiskFree, day, risk_free_over

# The fewest shared closes that yield a single return to compare.
MIN_OVERLAP_OBSERVATIONS = 2
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
# Below about a quarter, annualizing compounds short-run noise into a yearly figure that
# reads as a forecast. Just under three months because a period="3mo" window spans 87-95
# elapsed days depending on the call date, and the same request should not gain and lose
# its annualized fields from one day to the next.
MIN_ANNUALIZATION_DAYS = 85


def performance(
    symbol: str, period: str, bars: list[PriceBar], risk_free: RiskFree
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


def benchmark_comparison(
    symbol: str,
    bench: str,
    period: str,
    asset_bars: list[PriceBar],
    bench_bars: list[PriceBar],
    risk_free_rate: float | None,
    bills: Bills,
) -> BenchmarkComparison:
    """Benchmark-relative statistics over the dates the two bar series share."""
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
    risk_free = risk_free_over(risk_free_rate, bills, dates[0], dates[-1])
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


def comparison_row(
    perf: PerformanceStats, metrics: KeyMetrics | None, metrics_error: str | None
) -> TickerComparisonRow:
    """One comparison row: the performance figures plus whatever valuation metrics arrived."""
    return TickerComparisonRow(
        symbol=perf.symbol,
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


def _elapsed_days(start: str, end: str) -> int:
    """Calendar days between two PriceBar dates."""
    return (day(end) - day(start)).days
