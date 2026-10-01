"""Pure return/risk math over a series of closing prices. No MCP/network imports.

Assumes strictly positive closes (real adjusted prices); a zero close would divide by zero.

Annualization is expressed in calendar terms supplied by the caller -- ``years`` for
compounding, ``periods_per_year`` for scaling dispersion. Nothing here assumes a trading
calendar: how much time a series spans depends on what the instrument trades, which only
the data source knows.

The risk-adjusted and benchmark-relative statistics return None when a figure is not
computable (too few returns, or a zero denominator) rather than 0.0, which would assert a
result they have not measured.
"""

import math
import statistics

from finance_mcp.data.errors import InvalidInput

DAYS_PER_YEAR = 365.25
"""Mean calendar days per year, including the leap-day quarter."""


def total_return(closes: list[float]) -> float:
    """Total percent return from the first to the last close (e.g. 12.3 = 12.3%)."""
    if len(closes) < 2:
        raise InvalidInput("need at least two closes")
    return (closes[-1] / closes[0] - 1) * 100


def annualized_return(closes: list[float], years: float) -> float:
    """Annualized return (CAGR) in percent over ``years`` of elapsed calendar time.

    ``years`` is required rather than defaulted: the compounding exponent depends on how
    much wall-clock time the series covers, which only the caller knows. With ``years=1.0``
    this returns exactly :func:`total_return`.
    """
    if len(closes) < 2:
        raise InvalidInput("need at least two closes")
    if years <= 0:
        raise InvalidInput("years must be positive")
    try:
        growth: float = (closes[-1] / closes[0]) ** (1.0 / years)
    except OverflowError as exc:
        raise InvalidInput(
            "Window is too short to annualize: the growth factor overflowed."
        ) from exc
    return (growth - 1) * 100


def infer_periods_per_year(bars: int, years: float) -> float:
    """Observations per year implied by ``bars`` samples spanning ``years`` of calendar time.

    Reads the trading calendar off the data instead of assuming one: ~252 for a weekday
    market, ~365 for a 24/7 instrument such as crypto, and a correspondingly lower figure
    for an instrument that was halted or thinly traded over the window.
    """
    if bars < 2:
        raise InvalidInput("need at least two bars")
    if years <= 0:
        raise InvalidInput("years must be positive")
    return (bars - 1) / years


def annualized_volatility(closes: list[float], periods_per_year: float) -> float:
    """Annualized volatility of daily simple returns, in percent; 0.0 with fewer than 2 returns.

    ``periods_per_year`` scales the per-observation dispersion up to a yearly figure; derive
    it from the data with :func:`infer_periods_per_year`.
    """
    returns = simple_returns(closes)
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * math.sqrt(periods_per_year) * 100


def max_drawdown(closes: list[float]) -> float:
    """Largest peak-to-trough decline, as a non-positive percent (e.g. -23.4 = -23.4%)."""
    if len(closes) < 1:
        raise InvalidInput("need at least one close")
    peak = closes[0]
    worst = 0.0
    for close in closes:
        if close > peak:
            peak = close
        drawdown = (close / peak - 1) * 100
        if drawdown < worst:
            worst = drawdown
    return worst


def sma(closes: list[float], window: int) -> float | None:
    """Simple moving average of the last ``window`` closes; None if fewer than ``window`` closes."""
    if window <= 0:
        raise InvalidInput("window must be a positive integer.")
    if len(closes) < window:
        return None
    return statistics.fmean(closes[-window:])


# --- risk-adjusted statistics ---------------------------------------------------------


def simple_returns(closes: list[float]) -> list[float]:
    """Period-over-period simple returns as decimals (0.01 = +1%); [] for a single close."""
    return [closes[t] / closes[t - 1] - 1 for t in range(1, len(closes))]


def periodic_risk_free(annual_rate: float, periods_per_year: float) -> float:
    """De-annualize an annual risk-free rate to one observation period, geometrically.

    Compounding, not dividing: at 252 observations a year, dividing overstates the
    per-period rate and so understates every excess return computed against it.
    """
    if annual_rate <= -1.0:
        raise InvalidInput("risk_free_rate must be greater than -1 (i.e. above -100%).")
    if periods_per_year <= 0:
        raise InvalidInput("periods_per_year must be positive")
    # float ** float is typed Any, since it can produce a complex.
    growth: float = (1.0 + annual_rate) ** (1.0 / periods_per_year)
    return growth - 1.0


TREASURY_BILL_DAYS = 91
"""Term of a 13-week Treasury bill, the instrument ^IRX quotes."""


def treasury_bill_effective_rate(discount_yield_percent: float) -> float:
    """Convert a 13-week T-bill's quoted discount yield to an effective annual rate.

    Bills are quoted on a bank-discount basis: the percent of FACE a buyer is discounted,
    scaled to a 360-day year, so a 4.03% quote means paying 1 - 0.0403 * 91/360 per 1 of
    face. The analytics de-annualize the risk-free rate geometrically
    (:func:`periodic_risk_free`), so this returns the rate that compounds to the bill's
    91-day return over a DAYS_PER_YEAR year -- the same year the returns are annualized
    on. 4.03% quoted is 4.195% effective (Treasury's simple-interest "investment rate"
    for the same bill is 4.128%).
    """
    price = 1.0 - discount_yield_percent / 100.0 * TREASURY_BILL_DAYS / 360.0
    if price <= 0.0:
        raise InvalidInput(
            f"A {discount_yield_percent}% discount yield implies a non-positive bill price."
        )
    # float ** float is typed Any, since it can produce a complex.
    growth: float = price ** (-DAYS_PER_YEAR / TREASURY_BILL_DAYS)
    return growth - 1.0


def _excess_returns(
    closes: list[float], periods_per_year: float, risk_free_rate: float
) -> list[float]:
    """Per-observation returns net of the de-annualized risk-free rate."""
    per_period = periodic_risk_free(risk_free_rate, periods_per_year)
    return [r - per_period for r in simple_returns(closes)]


def sharpe_ratio(
    closes: list[float], periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    """Annualized Sharpe ratio: mean excess return over its standard deviation.

    ``risk_free_rate`` is an annual decimal (0.045 = 4.5%); with the default 0.0 this is
    raw return per unit of risk, not an excess-return figure. None when there are fewer
    than two returns or the returns never vary.
    """
    excess = _excess_returns(closes, periods_per_year, risk_free_rate)
    if len(excess) < 2:
        return None
    dispersion = statistics.stdev(excess)
    if dispersion == 0.0:
        return None
    return statistics.fmean(excess) / dispersion * math.sqrt(periods_per_year)


def _downside_dispersion(excess: list[float]) -> float:
    """Root-mean-square SHORTFALL below the target, over every observation.

    Averaging over all observations (not only the negative ones) is the standard downside
    deviation: an instrument that rarely falls should score as low downside risk, which
    averaging over the few negatives alone would hide.
    """
    return math.sqrt(statistics.fmean([min(0.0, e) ** 2 for e in excess]))


def downside_deviation(
    closes: list[float], periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    """Annualized downside deviation in percent; 0.0 when nothing fell below the target.

    None with fewer than two returns, matching :func:`sharpe_ratio`.
    """
    excess = _excess_returns(closes, periods_per_year, risk_free_rate)
    if len(excess) < 2:
        return None
    return _downside_dispersion(excess) * math.sqrt(periods_per_year) * 100


def sortino_ratio(
    closes: list[float], periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    """Annualized Sortino ratio: mean excess return over DOWNSIDE deviation only.

    None with fewer than two returns, or when nothing fell below the risk-free target
    (zero downside risk makes the ratio undefined rather than infinitely good).
    """
    excess = _excess_returns(closes, periods_per_year, risk_free_rate)
    if len(excess) < 2:
        return None
    dispersion = _downside_dispersion(excess)
    if dispersion == 0.0:
        return None
    return statistics.fmean(excess) / dispersion * math.sqrt(periods_per_year)


def calmar_ratio(annualized_return_percent: float, max_drawdown_percent: float) -> float | None:
    """CAGR per unit of worst peak-to-trough decline; None when there was no drawdown.

    Both arguments are percents and the ratio is dimensionless. The drawdown is taken by
    magnitude, so the caller may pass it with either sign.
    """
    if max_drawdown_percent == 0.0:
        return None
    return annualized_return_percent / abs(max_drawdown_percent)


# --- benchmark-relative statistics -----------------------------------------------------


def align_closes(
    asset: list[tuple[str, float]], benchmark: list[tuple[str, float]]
) -> tuple[list[str], list[float], list[float]]:
    """Inner-join two dated close series on their dates; returns (dates, asset, benchmark).

    Only shared dates are kept. Carrying a benchmark close forward over days it did not
    trade (a 24/7 asset's weekends, say) would add flat returns that deflate its volatility
    and every beta computed against it; instead the asset's weekend move lands in Monday's
    return.

    Dates are compared as the strings the data layer produced (ISO 8601, so lexical order
    is chronological order) and returned oldest-first.
    """
    asset_by_date = dict(asset)
    benchmark_by_date = dict(benchmark)
    dates = sorted(asset_by_date.keys() & benchmark_by_date.keys())
    return dates, [asset_by_date[d] for d in dates], [benchmark_by_date[d] for d in dates]


def _paired_returns(
    asset_closes: list[float], benchmark_closes: list[float]
) -> tuple[list[float], list[float]]:
    """Simple returns for two already-aligned close series, asserting they line up.

    A length mismatch means the caller skipped :func:`align_closes`; silently zipping to
    the shorter one would pair each asset return with the wrong day's benchmark return and
    report a plausible, wrong beta.
    """
    if len(asset_closes) != len(benchmark_closes):
        raise InvalidInput(
            "asset and benchmark series must cover the same dates; align them first."
        )
    return simple_returns(asset_closes), simple_returns(benchmark_closes)


def _active_returns(asset_closes: list[float], benchmark_closes: list[float]) -> list[float]:
    """Per-observation return of the asset net of the benchmark's."""
    a, b = _paired_returns(asset_closes, benchmark_closes)
    return [x - y for x, y in zip(a, b, strict=True)]


def beta(asset_closes: list[float], benchmark_closes: list[float]) -> float | None:
    """Sensitivity of the asset's returns to the benchmark's: cov(a, b) / var(b).

    None with fewer than two paired returns, or when the benchmark never moved.
    """
    a, b = _paired_returns(asset_closes, benchmark_closes)
    if len(a) < 2:
        return None
    benchmark_variance = statistics.variance(b)
    if benchmark_variance == 0.0:
        return None
    return statistics.covariance(a, b) / benchmark_variance


def correlation(asset_closes: list[float], benchmark_closes: list[float]) -> float | None:
    """Pearson correlation of the two return series, in [-1, 1].

    None with fewer than two paired returns, or when either series never moved (a constant
    series has no correlation with anything).
    """
    a, b = _paired_returns(asset_closes, benchmark_closes)
    if len(a) < 2:
        return None
    if statistics.stdev(a) == 0.0 or statistics.stdev(b) == 0.0:
        return None
    return statistics.correlation(a, b)


def jensen_alpha(
    asset_annualized_return_percent: float,
    benchmark_annualized_return_percent: float,
    beta_value: float,
    risk_free_rate: float,
) -> float:
    """Annualized Jensen's alpha in percent: return earned beyond what beta predicted.

    (Ra - Rf) - beta * (Rb - Rf), with the returns in percent and ``risk_free_rate`` an
    annual decimal (0.045 = 4.5%), converted here.
    """
    risk_free_percent = risk_free_rate * 100
    return (asset_annualized_return_percent - risk_free_percent) - beta_value * (
        benchmark_annualized_return_percent - risk_free_percent
    )


def tracking_error(
    asset_closes: list[float], benchmark_closes: list[float], periods_per_year: float
) -> float | None:
    """Annualized standard deviation of the active (asset minus benchmark) return, percent.

    None with fewer than two paired returns.
    """
    active = _active_returns(asset_closes, benchmark_closes)
    if len(active) < 2:
        return None
    return statistics.stdev(active) * math.sqrt(periods_per_year) * 100


def information_ratio(
    asset_closes: list[float], benchmark_closes: list[float], periods_per_year: float
) -> float | None:
    """Annualized mean active return per unit of tracking error.

    None with fewer than two paired returns, or when the asset tracked the benchmark
    exactly (zero active risk makes the ratio undefined).
    """
    active = _active_returns(asset_closes, benchmark_closes)
    if len(active) < 2:
        return None
    dispersion = statistics.stdev(active)
    if dispersion == 0.0:
        return None
    return statistics.fmean(active) / dispersion * math.sqrt(periods_per_year)
