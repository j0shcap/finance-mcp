"""Live contract: get_earnings through the configured provider.

Each part of the result is asserted separately (next report, estimates, history), so a
provider that stops supplying one of them fails here rather than returning it empty.
"""

import datetime

import pytest

from tests.live.conftest import AAPL, SAP, SPY, Layer

TOYOTA = "7203.T"  # a fiscal year ending in March, whose year-ago revenue the source understates


async def test_earnings_shape(layer: Layer) -> None:
    earnings = await layer.call("get_earnings", ticker=AAPL)

    # The next report: dated in the exchange's own offset, near today.
    assert earnings.next_report is not None, "AAPL always has a scheduled next report"
    when = datetime.datetime.fromisoformat(earnings.next_report.date)
    assert when.utcoffset() in (datetime.timedelta(hours=-4), datetime.timedelta(hours=-5))
    days_away = (when.date() - datetime.date.today()).days
    assert -7 <= days_away <= 120, f"next report {when} is {days_away} days away"

    # Estimates: the four periods, each anchored to a fiscal period end.
    estimates = {e.period: e for e in earnings.estimates}
    assert list(estimates) == [
        "reporting_quarter",
        "following_quarter",
        "reporting_fiscal_year",
        "following_fiscal_year",
    ]
    ends = {
        period: datetime.date.fromisoformat(e.fiscal_period_end) for period, e in estimates.items()
    }
    assert ends["reporting_quarter"] < ends["following_quarter"]
    assert ends["reporting_fiscal_year"] < ends["following_fiscal_year"]
    for estimate in earnings.estimates:
        assert estimate.eps.average is not None and estimate.eps.average > 0
        assert estimate.revenue.average is not None and estimate.revenue.average > 1e9
        assert estimate.eps.analysts is not None and estimate.eps.analysts > 0
        assert estimate.eps_currency == estimate.revenue_currency == "USD"

    # History: four quarters, oldest first, with the surprise as a percent.
    assert len(earnings.history) == 4
    quarter_ends = [q.fiscal_quarter_end for q in earnings.history]
    assert quarter_ends == sorted(quarter_ends)
    assert earnings.history_currency == "USD"
    for quarter in earnings.history:
        assert quarter.eps_actual is not None and quarter.eps_estimate is not None
        assert quarter.surprise_percent is not None
        expected = (quarter.eps_actual - quarter.eps_estimate) / abs(quarter.eps_estimate) * 100
        assert quarter.surprise_percent == pytest.approx(expected, abs=0.1), (
            "surprise_percent must be a percent of the estimate, not a fraction"
        )


async def test_earnings_currencies_are_reported_separately(layer: Layer) -> None:
    """An ADR's EPS and revenue need not share a currency (SAP: USD EPS, EUR revenue)."""
    earnings = await layer.call("get_earnings", ticker=SAP)
    for estimate in earnings.estimates:
        assert estimate.eps_currency is not None and estimate.revenue_currency is not None
    assert earnings.history_currency is not None


async def test_earnings_for_an_etf_is_a_clear_error(layer: Layer) -> None:
    """An ETF is a real symbol without earnings: the tool must say so, not report it unknown."""
    async with layer.expect_error("only companies report earnings"):
        await layer.call("get_earnings", ticker=SPY)


async def test_a_fiscal_year_growth_agrees_with_its_quarters(layer: Layer) -> None:
    """Toyota's reporting-year revenue growth: null (dropped as inconsistent) or plausible.

    Judged only while two quarters of that fiscal year are still estimated; nearer its end
    there is one, too few to tell a wrong year-ago from fast growth, and it passes through.
    """
    earnings = await layer.call("get_earnings", ticker=TOYOTA)
    by_period = {e.period: e for e in earnings.estimates}
    year = by_period["reporting_fiscal_year"]
    year_end = datetime.date.fromisoformat(year.fiscal_period_end)
    quarters_inside = [
        e
        for period, e in by_period.items()
        if period.endswith("quarter")
        and (year_end.year - 1, year_end.month, year_end.day)
        < _ymd(datetime.date.fromisoformat(e.fiscal_period_end))
        <= _ymd(year_end)
    ]
    if len(quarters_inside) < 2:
        pytest.skip(f"only {len(quarters_inside)} quarter of the fiscal year left to compare")
    growth = year.revenue.growth_percent
    assert growth is None or growth < 100, f"implausible fiscal-year revenue growth: {growth}%"


def _ymd(day: datetime.date) -> tuple[int, int, int]:
    return day.year, day.month, day.day
