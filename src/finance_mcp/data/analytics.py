"""Pure return/risk math over a series of closing prices. No MCP/network imports.

Assumes strictly positive closes (real adjusted prices); a zero close would divide by zero.

Annualization is expressed in calendar terms supplied by the caller -- ``years`` for
compounding, ``periods_per_year`` for scaling dispersion. Nothing here infers a trading
calendar from the bar count: how much wall-clock time a series spans is a property of the
data source, not of this module, and baking in a constant (252) silently misstates any
instrument that does not trade on the US equity calendar.
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
    it from the data with :func:`infer_periods_per_year` rather than assuming 252.
    """
    returns = [closes[t] / closes[t - 1] - 1 for t in range(1, len(closes))]
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
