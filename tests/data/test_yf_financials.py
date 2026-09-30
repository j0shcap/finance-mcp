"""YFinanceClient.get_financials: statements, line-item filtering and reporting currency."""

from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
import pytest
from yfinance.exceptions import (
    YFRateLimitError,
)

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.yfinance_client import (
    YFinanceClient,
)
from tests.fakes import (
    INCOME,
    SAP_INFO,
    FakeClock,
    fake_ticker_factory,
    make_client,
    make_financials_df,
)


def test_get_financials_parses_periods_and_line_items() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials("AAPL", "income", "annual")
    assert fs.symbol == "AAPL" and fs.statement == "income" and fs.period == "annual"
    assert fs.period_ends == ["2024-09-30", "2023-09-30"]
    assert fs.line_items["Total Revenue"] == [400.0, 380.0]
    assert fs.line_items["Net Income"] == [100.0, None]  # NaN -> None


def test_get_financials_line_items_filter() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials("AAPL", "income", "annual", line_items=["Total Revenue", "Nope"])
    assert list(fs.line_items.keys()) == ["Total Revenue"]  # only matching labels, "Nope" dropped


def test_get_financials_quarterly_attr() -> None:
    df = make_financials_df({"Total Revenue": [100.0]}, ["2025-03-31"])
    client = make_client(factory=fake_ticker_factory(financials={"quarterly_balance_sheet": df}))
    fs = client.get_financials("AAPL", "balance", "quarterly")
    assert fs.period_ends == ["2025-03-31"]


def test_get_financials_empty_raises_symbol_not_found() -> None:
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": pd.DataFrame()}))
    with pytest.raises(SymbolNotFound):
        client.get_financials("BAD", "income", "annual")


def test_get_financials_fetch_error_is_data_unavailable() -> None:
    client = make_client(factory=fake_ticker_factory(financials_error=RuntimeError("yahoo down")))
    with pytest.raises(DataUnavailable) as exc:
        client.get_financials("AAPL", "income", "annual")
    assert "yahoo down" in str(exc.value)


def test_get_financials_parse_error_is_data_unavailable() -> None:
    # Non-datetime columns make `col.date()` raise inside the parse block.
    df = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["Total Revenue"])
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    with pytest.raises(DataUnavailable) as exc:
        client.get_financials("AAPL", "income", "annual")
    assert "AAPL" in str(exc.value)


def test_get_financials_cached_within_ttl() -> None:
    calls = {"n": 0}
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])

    def counting(symbol: str) -> Any:
        calls["n"] += 1
        return fake_ticker_factory(financials={"income_stmt": df})(symbol)

    clock = FakeClock()
    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=clock,
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    client.get_financials("AAPL", "income", "annual")
    client.get_financials("AAPL", "income", "annual")
    assert calls["n"] == 1
    clock.advance(3601.0)
    client.get_financials("AAPL", "income", "annual")
    assert calls["n"] == 2


def test_get_financials_filter_reuses_cached_fetch() -> None:
    calls = {"n": 0}
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])

    def counting(symbol: str) -> object:
        calls["n"] += 1
        return fake_ticker_factory(financials={"income_stmt": df})(symbol)

    client = YFinanceClient(
        ticker_factory=counting,
        time_fn=FakeClock(),
        quote_ttl=30.0,
        history_ttl=300.0,
        fundamentals_ttl=3600.0,
    )
    full = client.get_financials("AAPL", "income", "annual")
    f1 = client.get_financials("AAPL", "income", "annual", line_items=["Total Revenue"])
    f2 = client.get_financials("AAPL", "income", "annual", line_items=["Net Income"])
    assert calls["n"] == 1  # one fetch; both filters reuse the cached statement
    assert list(f1.line_items) == ["Total Revenue"]
    assert list(f2.line_items) == ["Net Income"]
    assert set(full.line_items) == {"Total Revenue", "Net Income"}  # cached object un-mutated


@pytest.mark.parametrize(
    ("statement", "period", "attr"),
    [
        ("income", "annual", "income_stmt"),
        ("income", "quarterly", "quarterly_income_stmt"),
        ("balance", "annual", "balance_sheet"),
        ("balance", "quarterly", "quarterly_balance_sheet"),
        ("cashflow", "annual", "cashflow"),
        ("cashflow", "quarterly", "quarterly_cashflow"),
    ],
)
def test_get_financials_all_statement_period_combos(
    statement: Literal["income", "balance", "cashflow"],
    period: Literal["annual", "quarterly"],
    attr: str,
) -> None:
    df = make_financials_df({"X": [1.0]}, ["2024-12-31"])
    client = make_client(factory=fake_ticker_factory(financials={attr: df}))
    fs = client.get_financials("AAPL", statement, period)
    assert fs.statement == statement and fs.period == period
    assert fs.period_ends == ["2024-12-31"] and fs.line_items["X"] == [1.0]


def test_get_financials_line_items_preserve_order() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials(
        "AAPL", "income", "annual", line_items=["Net Income", "Total Revenue"]
    )
    assert list(fs.line_items) == ["Net Income", "Total Revenue"]


def test_get_financials_line_items_all_miss_empty() -> None:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))
    fs = client.get_financials("AAPL", "income", "annual", line_items=["Nonexistent"])
    assert fs.line_items == {}


def test_financial_statement_is_labelled_with_the_reporting_currency() -> None:
    df = make_financials_df(INCOME, ["2024-12-31", "2023-12-31"])
    client = make_client(factory=fake_ticker_factory(financials={"income_stmt": df}, info=SAP_INFO))
    assert client.get_financials("SAP", "income", "annual").currency == "EUR"


def test_financial_statement_currency_falls_back_to_quote_currency() -> None:
    df = make_financials_df(INCOME, ["2024-12-31", "2023-12-31"])
    client = make_client(
        factory=fake_ticker_factory(
            financials={"income_stmt": df}, info={"longName": "Apple Inc.", "currency": "USD"}
        )
    )
    assert client.get_financials("AAPL", "income", "annual").currency == "USD"


def test_financial_statement_currency_is_none_when_info_is_unusable() -> None:
    df = make_financials_df(INCOME, ["2024-12-31", "2023-12-31"])
    for factory in (
        fake_ticker_factory(financials={"income_stmt": df}, info={}),
        fake_ticker_factory(financials={"income_stmt": df}, info_error=OSError("no network")),
    ):
        client = make_client(factory=factory)
        # An unlabelled statement beats a failed one: the values are still correct.
        fs = client.get_financials("AAPL", "income", "annual")
        assert fs.currency is None
        assert fs.line_items["Total Revenue"] == [400.0, 380.0]


# --- unknown line-item labels are surfaced, not silently dropped (item 4) ---


def _income_client() -> YFinanceClient:
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    return make_client(factory=fake_ticker_factory(financials={"income_stmt": df}))


def test_unknown_line_items_are_reported_with_suggestions() -> None:
    fs = _income_client().get_financials(
        "AAPL", "income", "annual", line_items=["Total Revenue", "Revenue"]
    )
    assert list(fs.line_items) == ["Total Revenue"]
    assert fs.missing_line_items == ["Revenue"]
    assert fs.available_line_items == ["Total Revenue", "Net Income"]
    assert fs.line_item_suggestions["Revenue"] == ["Total Revenue"]


def test_matching_line_items_report_no_misses() -> None:
    fs = _income_client().get_financials("AAPL", "income", "annual", line_items=["Net Income"])
    assert list(fs.line_items) == ["Net Income"]
    assert fs.missing_line_items == []
    # Redundant with line_items when nothing is missing, so it stays empty.
    assert fs.available_line_items == []
    assert fs.line_item_suggestions == {}


def test_line_items_match_case_and_whitespace_insensitively() -> None:
    fs = _income_client().get_financials(
        "AAPL", "income", "annual", line_items=["total revenue", "  Net   Income  "]
    )
    # Keyed by the statement's canonical label, whatever spelling was requested.
    assert list(fs.line_items) == ["Total Revenue", "Net Income"]
    assert fs.missing_line_items == []


def test_repeated_line_items_collapse() -> None:
    fs = _income_client().get_financials(
        "AAPL", "income", "annual", line_items=["Total Revenue", "TOTAL REVENUE", "Nope", "Nope"]
    )
    assert list(fs.line_items) == ["Total Revenue"]
    assert fs.missing_line_items == ["Nope"]


def test_line_item_with_no_close_match_gets_no_suggestion() -> None:
    fs = _income_client().get_financials("AAPL", "income", "annual", line_items=["zzzzzzzz"])
    assert fs.line_items == {}
    assert fs.missing_line_items == ["zzzzzzzz"]
    assert fs.available_line_items == ["Total Revenue", "Net Income"]
    assert "zzzzzzzz" not in fs.line_item_suggestions


def test_unfiltered_statement_reports_no_misses() -> None:
    fs = _income_client().get_financials("AAPL", "income", "annual")
    assert fs.missing_line_items == [] and fs.available_line_items == []


# --- the statement currency is read once per symbol, and failures are not cached ---


def _statement_currency_factory(
    info_reads: list[str], info: dict[str, Any], fail_first: int = 0
) -> Callable[[str], Any]:
    """A ticker factory that records every ``.info`` read and can fail the first N of them."""
    df = make_financials_df(INCOME, ["2024-09-30", "2023-09-30"])
    state = {"failures_left": fail_first}

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self._symbol = symbol

        @property
        def info(self) -> Any:
            info_reads.append(self._symbol)
            if state["failures_left"] > 0:
                state["failures_left"] -= 1
                raise YFRateLimitError()
            return info

        def __getattr__(self, name: str) -> Any:
            return df

    return _Ticker


def test_statement_currency_is_fetched_once_per_symbol() -> None:
    info_reads: list[str] = []
    client = make_client(factory=_statement_currency_factory(info_reads, SAP_INFO))
    for statement in ("income", "balance", "cashflow"):
        for period in ("annual", "quarterly"):
            fs = client.get_financials("SAP", statement, period)
            assert fs.currency == "EUR"
    # All six statement/period combinations share one currency, so one .info request.
    assert info_reads == ["SAP"]


def test_failed_statement_currency_read_is_not_cached() -> None:
    info_reads: list[str] = []
    client = make_client(factory=_statement_currency_factory(info_reads, SAP_INFO, fail_first=1))
    first = client.get_financials("SAP", "income", "annual")
    assert first.currency is None  # unlabelled beats failing the statement
    # A rate limit says nothing about the reporting currency, so the next call retries
    # instead of asserting "Yahoo does not report it" for the whole fundamentals TTL.
    second = client.get_financials("SAP", "balance", "annual")
    assert second.currency == "EUR"
    assert info_reads == ["SAP", "SAP"]


def test_absent_statement_currency_is_cached() -> None:
    info_reads: list[str] = []
    client = make_client(factory=_statement_currency_factory(info_reads, {}))
    assert client.get_financials("X", "income", "annual").currency is None
    assert client.get_financials("X", "balance", "annual").currency is None
    # A genuine "Yahoo reports no currency" IS cacheable - no repeat request.
    assert info_reads == ["X"]


def test_empty_line_items_filter_returns_the_whole_statement() -> None:
    # An empty filter cannot mean "return nothing useful": that was the one silent-drop
    # case left. Library callers get the full statement; the tool rejects [] outright.
    fs = _income_client().get_financials("AAPL", "income", "annual", line_items=[])
    assert list(fs.line_items) == ["Total Revenue", "Net Income"]
    assert fs.missing_line_items == []
