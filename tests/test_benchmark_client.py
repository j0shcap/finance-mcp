"""Mocked-client tests for the benchmark comparison and the multi-ticker table.

Kept out of test_yfinance_client.py, which is already past 2 000 lines.
"""

import math
import threading
from typing import Any

import pytest

from finance_mcp.data import analytics
from finance_mcp.data.errors import DataUnavailable, InvalidInput
from finance_mcp.data.models import BenchmarkComparison
from tests.fakes import fake_multi_ticker_factory, make_client, make_history_df

# 200 consecutive calendar days from 2024-01-01, so both legs clear the 90-day gate.
DAYS = 200


def _walk(seed: float, step: float, wobble: float, n: int = DAYS) -> list[float]:
    """A deterministic, non-monotonic series: a drift plus a repeating wobble."""
    return [seed + step * i + wobble * math.sin(i) for i in range(n)]


def _compare(
    asset_closes: list[float],
    bench_closes: list[float],
    *,
    asset_freq: str = "D",
    bench_freq: str = "D",
    risk_free_rate: float = 0.0,
) -> BenchmarkComparison:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(asset_closes, freq=asset_freq)},
            "SPY": {"history_df": make_history_df(bench_closes, freq=bench_freq)},
        }
    )
    return make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y", risk_free_rate)


def test_compare_to_benchmark_reports_the_relative_statistics() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0)
    result = _compare(asset, bench)
    assert result.symbol == "AAPL" and result.benchmark == "SPY" and result.period == "1y"
    assert result.overlapping_observations == DAYS
    assert result.start_date == "2024-01-01"
    assert result.periods_per_year is not None
    assert result.beta == pytest.approx(analytics.beta(asset, bench))
    assert result.correlation == pytest.approx(analytics.correlation(asset, bench))
    assert result.tracking_error_percent == pytest.approx(
        analytics.tracking_error(asset, bench, result.periods_per_year)
    )
    assert result.information_ratio == pytest.approx(
        analytics.information_ratio(asset, bench, result.periods_per_year)
    )


def test_compare_to_benchmark_reports_excess_return_as_a_difference() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0)
    result = _compare(asset, bench)
    assert result.excess_return_percent == pytest.approx(
        result.total_return_percent - result.benchmark_total_return_percent
    )


def test_compare_to_benchmark_alpha_is_computed_from_the_aligned_cagrs() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0)
    result = _compare(asset, bench, risk_free_rate=0.04)
    assert result.risk_free_rate == 0.04
    assert result.beta is not None
    assert result.annualized_return_percent is not None
    assert result.benchmark_annualized_return_percent is not None
    assert result.alpha_percent == pytest.approx(
        analytics.jensen_alpha(
            result.annualized_return_percent,
            result.benchmark_annualized_return_percent,
            result.beta,
            0.04,
        )
    )


def test_compare_to_benchmark_defaults_the_risk_free_rate_to_zero() -> None:
    result = _compare(_walk(100.0, 0.30, 2.0), _walk(400.0, 0.80, 4.0))
    assert result.risk_free_rate == 0.0


def test_compare_to_benchmark_aligns_a_seven_day_asset_to_a_weekday_benchmark() -> None:
    """BTC-USD vs SPY: the overlap is the benchmark's weekdays, so periods_per_year is read
    off a 5-day calendar -- not the 365 the crypto leg alone implies.

    The expected figure is ~261, not the ~252 real market data gives: pandas' "B" frequency
    is every Mon-Fri, and it is the ~9 annual market holidays that take a real exchange from
    261 down to 252. The point of the assertion is the calendar the factor came from.
    """
    asset = _walk(100.0, 0.30, 2.0)  # freq="D": every calendar day
    bench = _walk(400.0, 0.80, 4.0)  # freq="B": weekdays only
    result = _compare(asset, bench, bench_freq="B")
    assert result.overlapping_observations < DAYS
    assert result.periods_per_year is not None
    # 365.25 / 7 * 5 = 260.9 weekdays a year, nowhere near the crypto leg's 365.
    assert result.periods_per_year == pytest.approx(261.0, rel=0.02)
    # The reported dates are the OVERLAP, not either input window.
    assert result.end_date <= "2024-07-18"


def test_compare_to_benchmark_needs_two_overlapping_dates() -> None:
    """A one-day overlap has no return to compare, so it must report the overlap count
    rather than indexing into an empty return list."""
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df([100.0, 101.0, 102.0], start="2024-01-01")},
            "SPY": {"history_df": make_history_df([50.0, 51.0, 52.0], start="2024-01-03")},
        }
    )
    with pytest.raises(DataUnavailable, match="overlapping"):
        make_client(factory).compare_to_benchmark("AAPL", "SPY", "5d", 0.0)


def test_compare_to_benchmark_with_no_shared_dates_names_both_symbols() -> None:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df([100.0, 101.0], start="2024-01-01")},
            "SPY": {"history_df": make_history_df([50.0, 51.0], start="2025-06-01")},
        }
    )
    with pytest.raises(DataUnavailable) as excinfo:
        make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y", 0.0)
    assert "AAPL" in str(excinfo.value) and "SPY" in str(excinfo.value)


def test_compare_to_benchmark_nulls_annualized_figures_on_a_short_overlap() -> None:
    asset, bench = _walk(100.0, 0.30, 2.0, n=30), _walk(400.0, 0.80, 4.0, n=30)
    result = _compare(asset, bench)  # 30 calendar days: under the 90-day gate
    assert result.periods_per_year is None
    assert result.annualized_return_percent is None
    assert result.benchmark_annualized_return_percent is None
    assert result.alpha_percent is None
    assert result.tracking_error_percent is None
    assert result.information_ratio is None
    # Beta and correlation need no annualization, so the gate must not blank them.
    assert result.beta is not None
    assert result.correlation is not None
    # Nor do the total returns.
    assert result.excess_return_percent is not None


def test_compare_to_benchmark_flat_benchmark_gives_no_beta_or_alpha() -> None:
    """A benchmark that never moves has zero variance."""
    result = _compare(_walk(100.0, 0.30, 2.0), [400.0] * DAYS)
    assert result.beta is None
    assert result.correlation is None
    assert result.alpha_percent is None  # alpha is undefined without a beta
    assert result.tracking_error_percent is not None  # active return still varies


def test_compare_to_benchmark_rejects_comparing_a_symbol_to_itself() -> None:
    factory = fake_multi_ticker_factory(
        {"SPY": {"history_df": make_history_df(_walk(400.0, 0.8, 4.0))}}
    )
    with pytest.raises(InvalidInput, match="two different"):
        make_client(factory).compare_to_benchmark("SPY", "spy", "1y", 0.0)


def test_compare_to_benchmark_propagates_an_unknown_benchmark() -> None:
    factory = fake_multi_ticker_factory(
        {"AAPL": {"history_df": make_history_df(_walk(100.0, 0.3, 2.0))}}
    )
    with pytest.raises(DataUnavailable):
        make_client(factory).compare_to_benchmark("AAPL", "NOPE", "1y", 0.0)


def test_compare_to_benchmark_normalizes_both_symbols() -> None:
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(_walk(100.0, 0.30, 2.0))},
            "SPY": {"history_df": make_history_df(_walk(400.0, 0.80, 4.0))},
        }
    )
    result = make_client(factory).compare_to_benchmark(" aapl ", "spy", "1y", 0.0)
    assert (result.symbol, result.benchmark) == ("AAPL", "SPY")


def test_compare_to_benchmark_fetches_both_legs_concurrently() -> None:
    # Both fetches wait on a 2-party barrier, so a sequential implementation deadlocks
    # until the timeout rather than returning.
    gate = threading.Barrier(2, timeout=10)
    factory = fake_multi_ticker_factory(
        {
            "AAPL": {"history_df": make_history_df(_walk(100.0, 0.30, 2.0))},
            "SPY": {"history_df": make_history_df(_walk(400.0, 0.80, 4.0))},
        },
        gate=gate,
    )
    result = make_client(factory).compare_to_benchmark("AAPL", "SPY", "1y", 0.0)
    assert result.overlapping_observations == DAYS


METRICS_INFO = {
    "longName": "Apple Inc.",
    "currency": "USD",
    "financialCurrency": "USD",
    "trailingPE": 37.7,
    "forwardPE": 30.1,
    "priceToBook": 51.2,
    "profitMargins": 0.271,
    "returnOnEquity": 1.41,
    "debtToEquity": 79.55,
}


def _rows_factory(**overrides: dict[str, Any]) -> Any:
    """Two healthy comparable tickers, with per-symbol overrides merged in."""
    base: dict[str, dict[str, Any]] = {
        "AAPL": {"history_df": make_history_df(_walk(100.0, 0.30, 2.0)), "info": METRICS_INFO},
        "MSFT": {"history_df": make_history_df(_walk(200.0, 0.50, 3.0)), "info": METRICS_INFO},
    }
    base.update(overrides)
    return fake_multi_ticker_factory(base)


def test_compare_tickers_returns_one_row_per_ticker_in_request_order() -> None:
    table = make_client(_rows_factory()).compare_tickers(["msft", " aapl "], "1y")
    assert [row.symbol for row in table.rows] == ["MSFT", "AAPL"]
    assert table.errors == []
    assert table.period == "1y"
    assert table.risk_free_rate == 0.0


def test_compare_tickers_rows_carry_performance_and_valuation() -> None:
    table = make_client(_rows_factory()).compare_tickers(["AAPL", "MSFT"], "1y")
    row = table.rows[0]
    assert row.total_return_percent is not None
    assert row.annualized_volatility_percent is not None
    assert row.sharpe_ratio is not None
    assert row.max_drawdown_percent < 0
    assert row.trailing_pe == 37.7
    assert row.forward_pe == 30.1
    assert row.profit_margins == 0.271
    assert row.metrics_error is None


def test_compare_tickers_rows_report_the_calendar_each_was_annualized_on() -> None:
    """A 24/7 row and a weekday row in one table are scaled differently; the row says which.

    Without periods_per_year on the row there is nothing in the table to warn that its
    volatility and sharpe_ratio columns are not on a common footing.
    """
    factory = _rows_factory(
        MSFT={
            "history_df": make_history_df(_walk(200.0, 0.50, 3.0), freq="B"),
            "info": METRICS_INFO,
        }
    )
    table = make_client(factory).compare_tickers(["AAPL", "MSFT"], "1y")
    crypto_like, equity_like = table.rows
    assert crypto_like.periods_per_year == pytest.approx(365.25, rel=0.02)
    assert equity_like.periods_per_year == pytest.approx(261.0, rel=0.02)


def test_compare_tickers_short_window_row_has_no_periods_per_year() -> None:
    """Under the annualization floor the row reports no calendar, matching its null ratios."""
    factory = _rows_factory(AAPL={"history_df": make_history_df(_walk(100.0, 0.3, 2.0, n=30))})
    table = make_client(factory).compare_tickers(["AAPL"], "1mo")
    row = table.rows[0]
    assert row.periods_per_year is None
    assert row.sharpe_ratio is None


def test_compare_tickers_applies_the_risk_free_rate_to_every_row() -> None:
    raw = make_client(_rows_factory()).compare_tickers(["AAPL", "MSFT"], "1y")
    excess = make_client(_rows_factory()).compare_tickers(
        ["AAPL", "MSFT"], "1y", risk_free_rate=0.05
    )
    assert excess.risk_free_rate == 0.05
    assert raw.rows[0].sharpe_ratio is not None and excess.rows[0].sharpe_ratio is not None
    assert excess.rows[0].sharpe_ratio < raw.rows[0].sharpe_ratio


def test_compare_tickers_deduplicates_equivalent_spellings() -> None:
    table = make_client(_rows_factory()).compare_tickers(["AAPL", "aapl"], "1y")
    assert [row.symbol for row in table.rows] == ["AAPL"]


def test_compare_tickers_reports_a_failed_history_as_an_error_not_a_row() -> None:
    """The row's backbone is gone, so there is no row."""
    factory = _rows_factory(NOPE={"history_error": KeyError("exchangeTimezoneName")})
    table = make_client(factory).compare_tickers(["AAPL", "NOPE"], "1y")
    assert [row.symbol for row in table.rows] == ["AAPL"]
    assert [err.symbol for err in table.errors] == ["NOPE"]
    assert "NOPE" in table.errors[0].error


def test_compare_tickers_keeps_the_row_when_only_the_valuation_metrics_fail() -> None:
    """Performance still stands, so the row survives with null valuation fields and a
    per-row reason - not silently blank, and not a dropped row."""
    factory = _rows_factory(
        BADINFO={
            "history_df": make_history_df(_walk(300.0, 0.20, 1.0)),
            "info_error": RuntimeError("Yahoo 503"),
        }
    )
    table = make_client(factory).compare_tickers(["AAPL", "BADINFO"], "1y")
    row = next(r for r in table.rows if r.symbol == "BADINFO")
    assert table.errors == []  # not double-reported
    assert row.total_return_percent is not None
    assert row.trailing_pe is None
    assert row.metrics_error is not None and "BADINFO" in row.metrics_error


def test_compare_tickers_reports_an_unnormalizable_symbol_without_fetching() -> None:
    table = make_client(_rows_factory()).compare_tickers(["AAPL", "  "], "1y")
    assert [row.symbol for row in table.rows] == ["AAPL"]
    assert table.errors[0].symbol == "  "
    assert "Empty ticker symbol" in table.errors[0].error


def test_compare_tickers_empty_list_is_an_empty_table() -> None:
    table = make_client(_rows_factory()).compare_tickers([], "1y")
    assert table.rows == [] and table.errors == []
    assert table.base_currency is None
    assert table.mixed_currencies is False


def test_compare_tickers_flags_a_currency_difference_per_row() -> None:
    foreign = dict(METRICS_INFO, currency="EUR", financialCurrency="EUR")
    factory = _rows_factory(
        SAP={"history_df": make_history_df(_walk(150.0, 0.25, 2.0)), "info": foreign}
    )
    table = make_client(factory).compare_tickers(["AAPL", "SAP"], "1y")
    by_symbol = {row.symbol: row for row in table.rows}
    assert table.base_currency == "USD"  # the first row that reported one
    assert table.mixed_currencies is True
    assert by_symbol["AAPL"].currency == "USD" and by_symbol["AAPL"].currency_differs is False
    assert by_symbol["SAP"].currency == "EUR" and by_symbol["SAP"].currency_differs is True


def test_compare_tickers_single_currency_batch_is_not_flagged() -> None:
    table = make_client(_rows_factory()).compare_tickers(["AAPL", "MSFT"], "1y")
    assert table.mixed_currencies is False
    assert all(row.currency_differs is False for row in table.rows)


def test_compare_tickers_unknown_currency_is_not_flagged_as_a_difference() -> None:
    # Yahoo sometimes omits the currency; a null is "unlabelled", not "different".
    no_currency = {k: v for k, v in METRICS_INFO.items() if k != "currency"}
    factory = _rows_factory(
        MYST={"history_df": make_history_df(_walk(10.0, 0.05, 0.4)), "info": no_currency}
    )
    table = make_client(factory).compare_tickers(["AAPL", "MYST"], "1y")
    by_symbol = {row.symbol: row for row in table.rows}
    assert by_symbol["MYST"].currency is None
    assert by_symbol["MYST"].currency_differs is False
    assert table.mixed_currencies is False


def test_compare_tickers_base_currency_comes_from_the_first_row_that_has_one() -> None:
    no_currency = {k: v for k, v in METRICS_INFO.items() if k != "currency"}
    factory = _rows_factory(
        MYST={"history_df": make_history_df(_walk(10.0, 0.05, 0.4)), "info": no_currency}
    )
    table = make_client(factory).compare_tickers(["MYST", "AAPL"], "1y")
    assert table.base_currency == "USD"


def test_compare_tickers_fetches_rows_concurrently() -> None:
    # Each row constructs a ticker twice (history, then info), so the barrier cycles; it
    # only ever clears if the rows are in flight together. A sequential implementation
    # blocks on the first wait until the timeout.
    symbols = ["AAPL", "MSFT", "GOOG"]
    gate = threading.Barrier(len(symbols), timeout=10)
    factory = fake_multi_ticker_factory(
        {
            s: {
                "history_df": make_history_df(_walk(100.0 + i * 10, 0.30, 2.0)),
                "info": METRICS_INFO,
            }
            for i, s in enumerate(symbols)
        },
        gate=gate,
    )
    table = make_client(factory).compare_tickers(symbols, "1y")
    assert [row.symbol for row in table.rows] == symbols
