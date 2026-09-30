"""YFinanceClient.get_company_profile."""

import pandas as pd
import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from tests.fakes import (
    FakeClock,
    counting,
    fake_ticker_factory,
    make_client,
    make_series,
)

FULL_INFO = {
    "longName": "Apple Inc.",
    "shortName": "Apple",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "country": "United States",
    "website": "https://www.apple.com",
    "fullTimeEmployees": 166000,
    "longBusinessSummary": "Apple designs phones.",
    "currency": "USD",
    "marketCap": 4.5e12,
    "trailingPE": 37.7,
    "forwardPE": 32.5,
    "dividendYield": 0.35,
    "beta": 1.06,
}


def test_get_company_profile_maps_fields() -> None:
    client = make_client(factory=fake_ticker_factory(info=FULL_INFO))
    p = client.get_company_profile("AAPL")
    assert p.symbol == "AAPL"
    assert p.name == "Apple Inc." and p.sector == "Technology"
    assert p.industry == "Consumer Electronics" and p.country == "United States"
    assert p.employees == 166000 and p.currency == "USD"
    assert p.market_cap == 4.5e12 and p.trailing_pe == 37.7 and p.beta == 1.06
    assert p.summary == "Apple designs phones."


@pytest.mark.parametrize(
    "info",
    [{"shortName": "Apple"}, {"longName": "", "shortName": "Apple"}],
    ids=["absent", "empty"],
)
def test_get_company_profile_name_falls_back_to_short_name(info: dict[str, str]) -> None:
    client = make_client(factory=fake_ticker_factory(info=info))
    assert client.get_company_profile("AAPL").name == "Apple"


def test_get_company_profile_recent_dividends_capped_at_8() -> None:
    dates = [f"2023-{m:02d}-01" for m in range(1, 13)]  # 12 dividends
    div = make_series(dates, [0.20 + i * 0.01 for i in range(12)])
    client = make_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=div))
    p = client.get_company_profile("AAPL")
    assert len(p.recent_dividends) == 8  # only most recent 8
    assert p.recent_dividends[-1].date == "2023-12-01"  # newest last
    assert p.recent_dividends[0].date == "2023-05-01"


def test_get_company_profile_all_splits() -> None:
    spl = make_series(["1987-06-16", "2000-06-21", "2020-08-31"], [2.0, 2.0, 4.0])
    client = make_client(factory=fake_ticker_factory(info=FULL_INFO, splits=spl))
    p = client.get_company_profile("AAPL")
    assert len(p.splits) == 3 and p.splits[-1].ratio == 4.0 and p.splits[-1].date == "2020-08-31"


def test_get_company_profile_missing_fields_are_none_and_empty() -> None:
    client = make_client(factory=fake_ticker_factory(info={"longName": "X Corp"}))
    p = client.get_company_profile("X")
    assert p.name == "X Corp" and p.sector is None and p.market_cap is None
    assert p.recent_dividends == [] and p.splits == []


def test_get_company_profile_no_name_raises_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info={"trailingPegRatio": None}))
    with pytest.raises(SymbolNotFound):
        client.get_company_profile("BAD")


def test_get_company_profile_source_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert type(exc.value) is DataUnavailable
    assert "rate limited" in str(exc.value)


def test_get_company_profile_unknown_symbol_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=KeyError("boom")))
    with pytest.raises(SymbolNotFound):
        client.get_company_profile("AAPL")


def test_get_company_profile_skips_non_finite_dividends_and_splits() -> None:
    div = make_series(["2023-01-01", "2023-06-01"], [0.20, float("nan")])
    spl = make_series(["2000-06-21", "2020-08-31"], [float("inf"), 4.0])
    client = make_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=div, splits=spl))
    p = client.get_company_profile("AAPL")
    assert [d.date for d in p.recent_dividends] == ["2023-01-01"]  # NaN dropped
    assert [s.date for s in p.splits] == ["2020-08-31"]  # inf dropped


def test_get_company_profile_parse_error_is_data_unavailable() -> None:
    # A non-datetime index makes `ts.date()` raise inside the parse block.
    bad_div = pd.Series([0.25], index=["not-a-date"], dtype=float)
    client = make_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=bad_div))
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert "AAPL" in str(exc.value)


@pytest.mark.parametrize("read", ["dividends", "splits"])
def test_get_company_profile_event_read_error_is_data_unavailable(read: str) -> None:
    # .info has already identified the instrument, so this is not SymbolNotFound.
    client = make_client(
        factory=fake_ticker_factory(
            info=FULL_INFO, **{f"{read}_error": YFException(f"{read} down")}
        )
    )
    with pytest.raises(DataUnavailable) as exc:
        client.get_company_profile("AAPL")
    assert type(exc.value) is DataUnavailable
    assert f"{read} down" in str(exc.value)


def test_get_company_profile_nan_employees_nulled_not_fatal() -> None:
    info = {**FULL_INFO, "fullTimeEmployees": float("nan")}
    client = make_client(factory=fake_ticker_factory(info=info))
    p = client.get_company_profile("AAPL")
    assert p.employees is None  # junk field nulls; profile still returned
    assert p.name == "Apple Inc."


def test_get_company_profile_float_employees_coerced_to_int() -> None:
    info = {**FULL_INFO, "fullTimeEmployees": 166000.0}
    client = make_client(factory=fake_ticker_factory(info=info))
    assert client.get_company_profile("AAPL").employees == 166000


def test_get_company_profile_caches_within_ttl() -> None:
    factory, calls = counting(fake_ticker_factory(info=FULL_INFO))
    clock = FakeClock()
    client = make_client(factory, clock=clock, fundamentals_ttl=3600.0)
    client.get_company_profile("AAPL")
    client.get_company_profile("AAPL")
    assert len(calls) == 1
    clock.advance(3601.0)
    client.get_company_profile("AAPL")
    assert len(calls) == 2


def test_get_company_profile_dividends_below_cap_returns_all() -> None:
    div = make_series(["2023-02-01", "2023-05-01", "2023-08-01"], [0.23, 0.24, 0.25])
    client = make_client(factory=fake_ticker_factory(info=FULL_INFO, dividends=div))
    p = client.get_company_profile("AAPL")
    assert [d.date for d in p.recent_dividends] == ["2023-02-01", "2023-05-01", "2023-08-01"]
