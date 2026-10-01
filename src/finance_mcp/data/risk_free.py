"""The risk-free rate behind the Sharpe, Sortino and alpha figures.

The caller's rate when one is given; otherwise the 13-week US T-bill yield averaged over the
dates actually measured, from history the caller has already fetched. Pure: nothing here
fetches.
"""

import statistics
from datetime import date, datetime
from typing import NamedTuple

from finance_mcp.data import analytics
from finance_mcp.data.errors import InvalidInput
from finance_mcp.data.models import PriceBar, RiskFreeSource

#: Yahoo's 13-week US Treasury bill yield, the default risk-free rate. Quoted in percent on
#: a bank-discount basis; see analytics.treasury_bill_effective_rate.
TREASURY_BILL_SYMBOL = "^IRX"
#: How far inside a measured window the T-bill history may start or end and still count as
#: covering it, so a bond-market holiday at either edge is not a gap.
RISK_FREE_EDGE_TOLERANCE_DAYS = 7


class RiskFree(NamedTuple):
    """The risk-free rate one computation used, and how it was chosen."""

    rate: float | None
    source: RiskFreeSource
    note: str | None = None
    #: True only when the T-bill FETCH failed, so a retry could resolve the rate. A window
    #: the history does not cover, or an implausible quote, gives the same answer each time.
    retryable: bool = False


#: The T-bill history a computation draws its default rate from: the bars, or why they
#: could not be fetched. None when the caller passed a rate, so nothing was fetched.
Bills = list[PriceBar] | str | None


_HOW_TO_PROCEED = (
    "Pass risk_free_rate explicitly to get the rate-dependent figures (0 gives raw "
    "return per unit of risk)."
)


def bills_fetch_failed(reason: str) -> RiskFree:
    """No default rate because the T-bill history could not be fetched: worth a retry."""
    return RiskFree(
        None,
        "unavailable",
        f"The 13-week T-bill yield ({TREASURY_BILL_SYMBOL}) could not be fetched: {reason}. "
        f"{_HOW_TO_PROCEED}",
        retryable=True,
    )


def table_risk_free(risk_free_rate: float | None, bills: Bills) -> RiskFree:
    """compare_tickers' header: the caller's rate, or whether the one T-bill fetch worked.

    Window coverage is judged per row, so a header over a successful fetch says
    "treasury_bill" and leaves any row-level gap to that row's own source and note.
    """
    if risk_free_rate is not None:
        return RiskFree(risk_free_rate, "caller")
    if isinstance(bills, str):
        return bills_fetch_failed(bills)
    return RiskFree(None, "treasury_bill")


def rate_key(risk_free_rate: float | None) -> str:
    """Cache-key component: the caller's rate, or a marker for the T-bill default."""
    return "treasury_bill" if risk_free_rate is None else str(risk_free_rate)


def risk_free_over(risk_free_rate: float | None, bills: Bills, start: str, end: str) -> RiskFree:
    """The caller's rate, else the mean effective T-bill rate over ``start``..``end``.

    The window average, not today's yield: a Sharpe over 2021-2026 measured against a 4%
    hurdle would charge the years when bills paid nothing as if they had paid 4%. The
    history must reach both ends of the window (within a holiday's tolerance), since
    averaging only part of it would misstate the cash return foregone.
    """
    if risk_free_rate is not None:
        return RiskFree(risk_free_rate, "caller")
    # With no rate given, _bills_for always fetched: bills is the history or its error.
    if not isinstance(bills, list):
        return bills_fetch_failed(str(bills))
    first_day, last_day = day(start), day(end)
    inside = [b for b in bills if first_day <= day(b.date) <= last_day]
    tolerance = RISK_FREE_EDGE_TOLERANCE_DAYS
    if (
        not inside
        or (day(inside[0].date) - first_day).days > tolerance
        or (last_day - day(inside[-1].date)).days > tolerance
    ):
        return RiskFree(
            None,
            "unavailable",
            f"The 13-week T-bill history ({TREASURY_BILL_SYMBOL}) covers {bills[0].date} to "
            f"{bills[-1].date}, which does not span the measured window {start} to {end}. "
            f"{_HOW_TO_PROCEED}",
        )
    try:
        rates = [analytics.treasury_bill_effective_rate(b.close) for b in inside]
    except InvalidInput as exc:
        return RiskFree(
            None,
            "unavailable",
            f"The 13-week T-bill history ({TREASURY_BILL_SYMBOL}) has an implausible "
            f"quote: {exc} {_HOW_TO_PROCEED}",
        )
    return RiskFree(statistics.fmean(rates), "treasury_bill")


def day(timestamp: str) -> date:
    """The calendar date of a PriceBar date or intraday timestamp.

    ``datetime.fromisoformat`` rather than ``date.fromisoformat``: only it accepts both a
    bare date ("2024-01-01") and a timestamp with an offset.
    """
    return datetime.fromisoformat(timestamp).date()
