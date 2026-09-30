"""YFinanceClient.get_key_metrics."""

import pytest
from yfinance.exceptions import (
    YFException,
)

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    KeyMetrics,
)
from finance_mcp.data.yfinance_client import (
    YFinanceClient,
)
from tests.fakes import (
    SAP_INFO,
    FakeClock,
    fake_ticker_factory,
    make_client,
)

METRICS_INFO = {
    "longName": "Apple Inc.",
    "trailingPE": 37.73,
    "forwardPE": 32.48,
    "priceToBook": 42.98,
    "priceToSalesTrailing12Months": 10.15,
    "pegRatio": 2.72,
    "enterpriseValue": 4599540350976,
    "enterpriseToEbitda": 28.75,
    "enterpriseToRevenue": 10.19,
    "returnOnEquity": 1.41,
    "returnOnAssets": 0.26,
    "grossMargins": 0.478,
    "operatingMargins": 0.322,
    "profitMargins": 0.271,
    "ebitdaMargins": 0.354,
    "debtToEquity": 79.55,
    "currentRatio": 1.07,
    "quickRatio": 0.906,
    "totalDebt": 84710998016,
    "totalCash": 68507000832,
    "freeCashflow": 101090746368,
    "ebitda": 159975997440,
    "trailingEps": 8.27,
    "forwardEps": 9.61,
    "revenuePerShare": 30.53,
    "bookValue": 7.26,
}


def test_get_key_metrics_maps_fields() -> None:
    m = make_client(factory=fake_ticker_factory(info=METRICS_INFO)).get_key_metrics("AAPL")
    assert isinstance(m, KeyMetrics)
    assert m.symbol == "AAPL"
    assert m.trailing_pe == 37.73 and m.forward_pe == 32.48 and m.price_to_book == 42.98
    assert m.price_to_sales == 10.15 and m.peg_ratio == 2.72
    assert m.enterprise_value == 4599540350976 and m.ev_to_ebitda == 28.75
    assert m.ev_to_revenue == 10.19
    assert m.return_on_equity == 1.41 and m.return_on_assets == 0.26
    assert m.gross_margins == 0.478 and m.operating_margins == 0.322
    assert m.profit_margins == 0.271 and m.ebitda_margins == 0.354
    assert m.debt_to_equity == 79.55 and m.current_ratio == 1.07 and m.quick_ratio == 0.906
    assert m.total_debt == 84710998016 and m.total_cash == 68507000832
    assert m.free_cashflow == 101090746368 and m.ebitda == 159975997440
    assert m.trailing_eps == 8.27 and m.forward_eps == 9.61
    assert m.revenue_per_share == 30.53 and m.book_value == 7.26


def test_get_key_metrics_missing_and_nan_are_none() -> None:
    info = {"longName": "X Corp", "trailingPE": float("nan")}
    m = make_client(factory=fake_ticker_factory(info=info)).get_key_metrics("X")
    assert m.symbol == "X" and m.trailing_pe is None
    assert m.ebitda is None and m.profit_margins is None


def test_get_key_metrics_no_name_raises_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info={"trailingPegRatio": None}))
    with pytest.raises(SymbolNotFound):
        client.get_key_metrics("BAD")


def test_get_key_metrics_typed_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=YFException("rate limited")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_key_metrics("AAPL")
    assert "rate limited" in str(exc.value)


def test_get_key_metrics_raw_error_is_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(info_error=KeyError("boom")))
    with pytest.raises(SymbolNotFound):
        client.get_key_metrics("AAPL")


def test_get_key_metrics_mapping_failure_is_data_unavailable() -> None:
    # A value that survives the name check but fails float() coercion in mapping.
    info = {"longName": "Apple Inc.", "trailingPE": object()}
    client = make_client(factory=fake_ticker_factory(info=info))
    with pytest.raises(DataUnavailable) as exc:
        client.get_key_metrics("AAPL")
    assert "Failed to parse metrics for 'AAPL'" in str(exc.value)


def test_get_key_metrics_caches_within_ttl() -> None:
    calls = {"n": 0}

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(info=METRICS_INFO)(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_key_metrics("AAPL")
    client.get_key_metrics("AAPL")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.get_key_metrics("AAPL")
    assert calls["n"] == 2


def test_profile_and_metrics_caches_do_not_collide() -> None:
    calls = {"n": 0}

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(info=METRICS_INFO)(symbol)

    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    prof = client.get_company_profile("AAPL")
    metr = client.get_key_metrics("AAPL")
    assert calls["n"] == 2  # distinct cache keys -> two fetches
    assert prof.symbol == "AAPL" and metr.symbol == "AAPL"


def test_key_metrics_carry_both_quote_and_financial_currency() -> None:
    client = make_client(factory=fake_ticker_factory(info=SAP_INFO))
    metrics = client.get_key_metrics("SAP")
    # EBITDA/debt/cash/FCF come from Yahoo's financialData (EUR); EV is derived from
    # market cap and is in the quote currency (USD). One currency field would misreport half.
    assert metrics.currency == "USD"
    assert metrics.financial_currency == "EUR"
    assert metrics.ebitda == 1.18e10
    assert metrics.enterprise_value == 3.42e12


def test_key_metrics_currencies_are_none_when_absent() -> None:
    client = make_client(factory=fake_ticker_factory(info={"longName": "X"}))
    metrics = client.get_key_metrics("X")
    assert metrics.currency is None and metrics.financial_currency is None
