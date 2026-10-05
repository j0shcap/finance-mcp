"""get_earnings: YahooProvider's parsing of quoteSummary, and DataService's caching of it."""

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastmcp.exceptions import ToolError
from yfinance.exceptions import YFException

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.providers.yahoo import EARNINGS_MODULES, _one_year_before
from tests.fakes import (
    FakeHTTPError,
    connect,
    counting,
    fake_ticker_factory,
    make_client,
    make_earnings_summary,
)


def _earnings(summary: dict[str, Any], symbol: str = "AAPL") -> Any:
    return make_client(fake_ticker_factory(quote_summary=summary)).get_earnings(symbol)


def test_maps_the_next_report_estimates_and_history() -> None:
    e = _earnings(make_earnings_summary())

    assert e.symbol == "AAPL"
    assert e.next_report is not None
    assert e.next_report.date == "2026-10-29T16:00:00-04:00"  # after the close, local time
    assert e.next_report.date_is_estimate is False
    assert e.next_report.window_end is None

    assert [(p.period, p.fiscal_period_end) for p in e.estimates] == [
        ("reporting_quarter", "2026-09-30"),
        ("following_quarter", "2026-12-31"),
        ("reporting_fiscal_year", "2026-09-30"),
        ("following_fiscal_year", "2027-09-30"),
    ]
    quarter = e.estimates[0]
    assert (quarter.eps.average, quarter.eps.low, quarter.eps.high) == (1.98, 1.93, 2.07)
    assert quarter.eps.analysts == 27 and quarter.eps.year_ago == 1.85
    assert quarter.eps.growth_percent == pytest.approx(6.95)  # Yahoo's fraction, as a percent
    assert quarter.revenue.average == 113.6e9 and quarter.revenue.year_ago == 102.5e9
    assert quarter.revenue.growth_percent == pytest.approx(10.89)

    # Oldest first, and quarter ends are UTC dates: in New York time they'd be a day early.
    assert [q.fiscal_quarter_end for q in e.history] == [
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
        "2025-12-31",
    ]
    assert (e.history[0].eps_estimate, e.history[0].eps_actual) == (1.77, 1.85)
    assert e.history[0].surprise_percent == pytest.approx(4.52)
    assert e.history_currency == "USD"


def test_fetches_every_module_in_one_request() -> None:
    factory = fake_ticker_factory(quote_summary=make_earnings_summary())
    make_client(factory).get_earnings("AAPL")
    assert factory.captured_quote_summary_modules == [list(EARNINGS_MODULES)]  # type: ignore[attr-defined]


def test_each_figure_keeps_its_own_currency() -> None:
    summary = make_earnings_summary(
        eps_currency="USD", revenue_currency="EUR", history_currency="USD"
    )
    e = _earnings(summary, "SAP")
    assert (e.estimates[0].eps_currency, e.estimates[0].revenue_currency) == ("USD", "EUR")
    assert e.history_currency == "USD"


def test_an_estimated_window_reports_both_ends_and_the_estimate_flag() -> None:
    # Across the US daylight-saving change on November 1, so the offsets differ.
    window_end = int(datetime(2026, 11, 3, 16, 0, tzinfo=ZoneInfo("America/New_York")).timestamp())
    e = _earnings(make_earnings_summary(dates=[window_end, 1793304000], date_is_estimate=True))
    assert e.next_report is not None
    assert e.next_report.date == "2026-10-29T16:00:00-04:00"
    assert e.next_report.window_end == "2026-11-03T16:00:00-05:00"
    assert e.next_report.date_is_estimate is True


def test_the_next_report_is_dated_in_the_exchange_timezone() -> None:
    late_utc = int(datetime(2026, 11, 4, 23, 0, tzinfo=UTC).timestamp())
    e = _earnings(make_earnings_summary(dates=[late_utc], timezone="Asia/Tokyo"), "7203.T")
    assert e.next_report is not None
    assert e.next_report.date == "2026-11-05T08:00:00+09:00"


def test_missing_parts_are_null_or_empty_not_errors() -> None:
    summary = make_earnings_summary(dates=[], trend_periods=(), history_quarters=())
    del summary["quoteType"]["timeZoneFullName"]
    e = _earnings(summary, "NESN.SW")
    assert e.next_report is None
    assert e.estimates == [] and e.history == []
    assert e.history_currency is None


def test_missing_estimate_fields_are_null() -> None:
    summary = make_earnings_summary(trend_periods=("0q",))
    summary["earningsTrend"]["trend"][0]["earningsEstimate"] = {}
    estimate = _earnings(summary).estimates[0]
    assert estimate.eps.average is None and estimate.eps.growth_percent is None
    assert estimate.eps_currency is None
    assert estimate.revenue.average == 113.6e9


def test_periods_beyond_the_four_it_reports_are_ignored() -> None:
    e = _earnings(make_earnings_summary(trend_periods=("-1q", "0q", "+5y")))
    assert [p.period for p in e.estimates] == ["reporting_quarter"]


def test_estimates_come_in_period_order_whatever_order_they_arrive_in() -> None:
    e = _earnings(make_earnings_summary(trend_periods=("+1y", "0q", "0y", "+1q")))
    assert [p.period for p in e.estimates] == [
        "reporting_quarter",
        "following_quarter",
        "reporting_fiscal_year",
        "following_fiscal_year",
    ]


def test_a_period_without_coverage_is_left_out_not_fatal() -> None:
    # VOD.L: no quarterly coverage (null end dates), but fiscal-year estimates.
    summary = make_earnings_summary()
    for item in summary["earningsTrend"]["trend"][:2]:
        item["endDate"] = None
    e = _earnings(summary, "VOD.L")
    assert [p.period for p in e.estimates] == ["reporting_fiscal_year", "following_fiscal_year"]


def test_a_history_entry_without_a_quarter_is_skipped() -> None:
    summary = make_earnings_summary()
    del summary["earningsHistory"]["history"][0]["quarter"]
    assert len(_earnings(summary).history) == 3


def test_an_unstated_confirmation_is_unknown_not_confirmed() -> None:
    summary = make_earnings_summary()
    del summary["calendarEvents"]["earnings"]["isEarningsDateEstimate"]
    e = _earnings(summary)
    assert e.next_report is not None and e.next_report.date_is_estimate is None


def test_a_company_without_earnings_coverage_gets_an_empty_result() -> None:
    summary = {"quoteType": {"quoteType": "EQUITY", "timeZoneFullName": "America/New_York"}}
    e = _earnings(summary, "SIEB")
    assert e.next_report is None and e.estimates == [] and e.history == []


@pytest.mark.parametrize("quote_type", ["ETF", "INDEX", "CRYPTOCURRENCY", "MUTUALFUND"])
def test_an_instrument_without_earnings_is_unavailable_not_unknown(quote_type: str) -> None:
    # Yahoo answers 200 with quoteType alone, rather than a 404.
    summary = {"quoteType": {"quoteType": quote_type, "timeZoneFullName": "America/New_York"}}
    with pytest.raises(DataUnavailable) as exc:
        _earnings(summary, "SPY")
    assert type(exc.value) is DataUnavailable
    assert "'SPY': it is not a company" in str(exc.value)
    assert "quote type" not in str(exc.value)  # the provider's own vocabulary stays inside


def test_an_unknown_symbol_is_symbol_not_found() -> None:
    client = make_client(fake_ticker_factory(quote_summary_error=FakeHTTPError(404)))
    with pytest.raises(SymbolNotFound, match="No earnings data for 'ZZZZQQ'"):
        client.get_earnings("ZZZZQQ")


def test_a_failed_request_is_data_unavailable() -> None:
    client = make_client(fake_ticker_factory(quote_summary_error=YFException("Yahoo down")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_earnings("AAPL")
    assert type(exc.value) is DataUnavailable
    assert "Failed to fetch earnings for 'AAPL'" in str(exc.value)


def test_a_yfinance_without_the_quote_scraper_says_so() -> None:
    client = make_client(lambda _symbol: SimpleNamespace())
    with pytest.raises(DataUnavailable, match=r"^This version of yfinance can't fetch"):
        client.get_earnings("AAPL")


@pytest.mark.parametrize(
    "response",
    [{"finance": {"error": "Internal"}}, {"quoteSummary": {"result": []}}, None],
    ids=["no-envelope", "empty-result", "no-body"],
)
def test_a_malformed_reply_is_unavailable_not_an_unknown_symbol(response: Any) -> None:
    def ticker(_symbol: str) -> Any:
        return SimpleNamespace(_quote=SimpleNamespace(_fetch=lambda modules: response))

    with pytest.raises(DataUnavailable) as exc:
        make_client(ticker).get_earnings("AAPL")
    assert type(exc.value) is DataUnavailable
    assert "Unexpected quoteSummary response" in str(exc.value)


def test_a_malformed_payload_is_a_parse_failure() -> None:
    summary = make_earnings_summary(dates=["next Thursday"])  # type: ignore[list-item]
    with pytest.raises(DataUnavailable, match="Failed to parse earnings for 'AAPL'"):
        _earnings(summary)


def test_earnings_are_cached_per_normalized_symbol() -> None:
    factory, calls = counting(fake_ticker_factory(quote_summary=make_earnings_summary()))
    client = make_client(factory)
    assert client.get_earnings(" aapl ").symbol == "AAPL"
    client.get_earnings("AAPL")
    assert calls == ["AAPL"]


async def test_get_earnings_tool() -> None:
    factory = fake_ticker_factory(quote_summary=make_earnings_summary())
    async with connect(factory) as client:
        result = await client.call_tool("get_earnings", {"ticker": "AAPL"})
    assert result.data.next_report.date == "2026-10-29T16:00:00-04:00"
    assert result.data.history[-1].fiscal_quarter_end == "2025-12-31"


async def test_get_earnings_tool_surfaces_the_reason() -> None:
    summary = {"quoteType": {"quoteType": "ETF", "timeZoneFullName": "America/New_York"}}
    async with connect(fake_ticker_factory(quote_summary=summary)) as client:
        with pytest.raises(ToolError, match="only companies report earnings"):
            await client.call_tool("get_earnings", {"ticker": "SPY"})


def _revenue(summary: dict[str, Any], period: str, average: float, year_ago: float) -> None:
    """Set one trend period's revenue average, year-ago and the growth between them."""
    item = next(i for i in summary["earningsTrend"]["trend"] if i["period"] == period)
    item["revenueEstimate"].update(
        avg={"raw": average},
        yearAgoRevenue={"raw": year_ago},
        growth={"raw": average / year_ago - 1},
    )


def _toyota_shaped(year_ago: float = 18.26e12, quarter_growth: float = 0.07) -> dict[str, Any]:
    # Fiscal year ending March: both quarters (Sep, Dec) fall inside it.
    summary = make_earnings_summary()
    ends = {"0q": "2026-09-30", "+1q": "2026-12-31", "0y": "2027-03-31", "+1y": "2028-03-31"}
    for item in summary["earningsTrend"]["trend"]:
        item["endDate"] = ends[item["period"]]
    _revenue(summary, "0q", 12.38e12 * (1 + quarter_growth), 12.38e12)
    _revenue(summary, "+1q", 13.46e12 * (1 + quarter_growth), 13.46e12)
    _revenue(summary, "0y", 53.25e12, year_ago)
    _revenue(summary, "+1y", 55.28e12, 53.25e12)
    return summary


def _fiscal_year(e: Any, period: str = "reporting_fiscal_year") -> Any:
    return next(estimate for estimate in e.estimates if estimate.period == period)


def test_a_year_ago_contradicting_its_quarters_is_dropped_not_reported() -> None:
    year = _fiscal_year(_earnings(_toyota_shaped(), "7203.T"))
    assert year.revenue.year_ago is None and year.revenue.growth_percent is None
    assert year.revenue.average == 53.25e12  # the estimate itself stands
    assert year.revenue.analysts == 27 and year.eps.year_ago == 1.85  # EPS untouched


def test_a_consistent_year_ago_is_kept() -> None:
    year = _fiscal_year(_earnings(_toyota_shaped(year_ago=50.68e12), "7203.T"))
    assert year.revenue.year_ago == 50.68e12
    assert year.revenue.growth_percent == pytest.approx(5.07, abs=0.01)


def test_a_hyper_growers_short_year_ago_is_kept_when_its_quarters_grew_as_fast() -> None:
    # NBIS: the year-ago is small next to its quarters, but so is every quarter's.
    year = _fiscal_year(_earnings(_toyota_shaped(year_ago=9.0e12, quarter_growth=5.0), "NBIS"))
    assert year.revenue.year_ago == 9.0e12


def test_growth_well_above_the_quarters_alone_is_not_enough() -> None:
    # A year-ago in line with its quarters: earlier quarters simply grew faster.
    summary = _toyota_shaped(year_ago=45e12, quarter_growth=0.01)
    year = _fiscal_year(_earnings(summary, "7203.T"))
    assert year.revenue.year_ago == 45e12


def test_one_quarter_inside_the_year_is_not_enough_to_judge() -> None:
    summary = _toyota_shaped()
    next(i for i in summary["earningsTrend"]["trend"] if i["period"] == "+1q")["endDate"] = (
        "2027-06-30"  # the following quarter belongs to the next fiscal year
    )
    assert _fiscal_year(_earnings(summary, "7203.T")).revenue.year_ago == 18.26e12


def test_a_repeated_quarter_end_counts_once() -> None:
    summary = _toyota_shaped()
    next(i for i in summary["earningsTrend"]["trend"] if i["period"] == "+1q")["endDate"] = (
        "2026-09-30"
    )
    assert _fiscal_year(_earnings(summary, "7203.T")).revenue.year_ago == 18.26e12


def test_the_following_fiscal_year_is_not_judged_on_quarters_outside_it() -> None:
    summary = _toyota_shaped()
    _revenue(summary, "+1y", 55.28e12, 18e12)  # would fail both tests if the quarters counted
    year = _fiscal_year(_earnings(summary, "7203.T"), "following_fiscal_year")
    assert year.revenue.year_ago == 18e12


def _move_quarter(summary: dict[str, Any], period: str, end: str) -> dict[str, Any]:
    next(i for i in summary["earningsTrend"]["trend"] if i["period"] == period)["endDate"] = end
    return summary


def test_a_quarter_ending_on_the_fiscal_year_end_is_inside_it() -> None:
    summary = _move_quarter(_toyota_shaped(), "+1q", "2027-03-31")
    assert _fiscal_year(_earnings(summary, "7203.T")).revenue.year_ago is None


def test_a_quarter_ending_exactly_a_year_before_the_fiscal_year_end_is_outside_it() -> None:
    summary = _move_quarter(_toyota_shaped(), "0q", "2026-03-31")
    assert _fiscal_year(_earnings(summary, "7203.T")).revenue.year_ago == 18.26e12


def test_a_quarter_with_degenerate_figures_is_not_used_as_evidence() -> None:
    summary = _toyota_shaped()
    _revenue(summary, "+1q", 0.0, 13.46e12)  # growth of -100%: a multiple of zero
    assert _fiscal_year(_earnings(summary, "7203.T")).revenue.year_ago == 18.26e12


def test_a_fiscal_year_ending_on_february_29_has_a_year_earlier() -> None:
    assert _one_year_before(date(2028, 2, 29)) == date(2027, 2, 28)
