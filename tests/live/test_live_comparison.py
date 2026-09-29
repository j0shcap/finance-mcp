"""Live contract: compare_to_benchmark, compare_tickers.

Both are ours, computed from live bars, so these assertions check our statistics against
real market data rather than a recorded payload. Two things can only be checked here: that
the inner join really does reconcile a 24/7 calendar with a weekday one, and that a batch
survives a symbol Yahoo rejects.
"""

import datetime

import pytest

from tests.live.conftest import AAPL, BTC, SAP, SPY, UNKNOWN, Layer, require_present

BENCHMARK_ANNUALIZED_FIELDS = (
    "alpha_percent",
    "tracking_error_percent",
    "information_ratio",
    "periods_per_year",
)


async def test_compare_to_benchmark_equity_shape_and_units(layer: Layer) -> None:
    result = await layer.call("compare_to_benchmark", ticker=AAPL, benchmark=SPY, period="1y")

    assert result.symbol == AAPL
    assert result.benchmark == SPY
    assert result.period == "1y"
    assert result.risk_free_rate == 0.0, "the default rate must be echoed, not omitted"
    datetime.date.fromisoformat(result.start_date)
    datetime.date.fromisoformat(result.end_date)

    require_present(result, ("beta", "correlation", "excess_return_percent"))
    require_present(result, BENCHMARK_ANNUALIZED_FIELDS)

    assert result.overlapping_observations > 200, (
        f"two US equities share a full year of sessions, so the inner join should keep ~252 "
        f"dates, got {result.overlapping_observations}"
    )

    assert result.periods_per_year is not None
    assert 220 < result.periods_per_year < 275, (
        f"periods_per_year {result.periods_per_year} is not a weekday trading calendar"
    )

    # A large-cap US equity against its own index: sensitive to it, and explained by it.
    assert result.beta is not None
    assert 0 < result.beta < 4, f"AAPL's beta to SPY should be order-1, got {result.beta}"
    assert result.correlation is not None
    assert 0.3 < result.correlation <= 1, (
        f"AAPL and SPY must be positively correlated, got {result.correlation}"
    )

    assert result.tracking_error_percent is not None
    assert 0 < result.tracking_error_percent < 100, (
        f"tracking error {result.tracking_error_percent} is not a plausible percent"
    )


async def test_compare_to_benchmark_inner_joins_a_247_calendar_onto_a_weekday_one(
    layer: Layer,
) -> None:
    """BTC-USD against SPY: the shared dates are SPY's weekdays, so ~252 a year, not ~365.

    This is the calendar reconciliation the tool description and the glossary promise, and
    only a live run has two genuinely different calendars to join. Inferring ~365 here would
    mean the annualization read BTC's own span instead of the aligned series, inflating
    every annualized figure by sqrt(365/252).
    """
    result = await layer.call("compare_to_benchmark", ticker=BTC, benchmark=SPY, period="1y")

    require_present(result, ("beta", "correlation", "periods_per_year"))
    assert result.periods_per_year is not None
    assert 220 < result.periods_per_year < 275, (
        f"the aligned series is weekday-only, so periods_per_year should be ~252, got "
        f"{result.periods_per_year} - a ~365 here means the join was not used"
    )
    assert 200 < result.overlapping_observations < 275, (
        f"the overlap is the benchmark's sessions, not the union, got "
        f"{result.overlapping_observations}"
    )


async def test_compare_tickers_returns_a_row_per_ticker_with_performance_and_valuation(
    layer: Layer,
) -> None:
    result = await layer.call("compare_tickers", tickers=[AAPL, SPY], period="1y")

    assert result.period == "1y"
    assert result.risk_free_rate == 0.0
    assert result.errors == [], f"both symbols are real, got errors {result.errors}"
    assert [row.symbol for row in result.rows] == [AAPL, SPY], (
        "rows must come back in request order, since the model reads them positionally"
    )

    apple = result.rows[0]
    require_present(
        apple,
        (
            "annualized_return_percent",
            "annualized_volatility_percent",
            "periods_per_year",
            "sharpe_ratio",
            "trailing_pe",
            "currency",
        ),
    )
    assert apple.periods_per_year is not None
    assert 220 < apple.periods_per_year < 275, (
        f"a weekday-traded equity row should annualize on ~252, got {apple.periods_per_year}"
    )
    assert apple.max_drawdown_percent <= 0
    assert apple.trailing_pe is not None and 0 < apple.trailing_pe < 500
    assert apple.metrics_error is None

    # Both rows quote in USD, so nothing is flagged and there is a single base currency.
    assert result.base_currency == "USD"
    assert result.mixed_currencies is False
    assert all(row.currency_differs is False for row in result.rows)


async def test_compare_tickers_keeps_the_rows_that_worked_when_one_symbol_fails(
    layer: Layer,
) -> None:
    """The partial-failure contract get_quote set: a bad symbol is an error, not a wipeout."""
    result = await layer.call("compare_tickers", tickers=[AAPL, UNKNOWN], period="1y")

    assert [row.symbol for row in result.rows] == [AAPL], (
        "a symbol Yahoo has no history for must not take the working row with it"
    )
    assert [e.symbol for e in result.errors] == [UNKNOWN]
    assert result.errors[0].error, "an error entry must say why the symbol produced no row"


async def test_compare_tickers_flags_a_cross_listing_reporting_in_another_currency(
    layer: Layer,
) -> None:
    """SAP quotes in USD but reports in EUR, so its row's absolute amounts are mixed.

    financial_currency is the flag the prompt has the model read before comparing a
    cross-listing's valuation figures against a domestic peer's.
    """
    result = await layer.call("compare_tickers", tickers=[AAPL, SAP], period="1y")

    by_symbol = {row.symbol: row for row in result.rows}
    assert SAP in by_symbol, f"SAP produced no row; errors: {result.errors}"
    sap = by_symbol[SAP]
    require_present(sap, ("currency", "financial_currency"))
    assert sap.currency == "USD", f"the US listing quotes in USD, got {sap.currency}"
    assert sap.financial_currency == "EUR", (
        f"SAP reports in EUR; got {sap.financial_currency}. A match with `currency` here "
        f"means the cross-listing signal has been lost."
    )


async def test_compare_tickers_applies_the_risk_free_rate_to_every_row(layer: Layer) -> None:
    """A non-zero rate lowers every Sharpe, and the rate used is echoed on the table."""
    raw = await layer.call("compare_tickers", tickers=[AAPL, SPY], period="1y")
    excess = await layer.call(
        "compare_tickers", tickers=[AAPL, SPY], period="1y", risk_free_rate=0.05
    )

    assert excess.risk_free_rate == pytest.approx(0.05)
    for before, after in zip(raw.rows, excess.rows, strict=True):
        assert before.sharpe_ratio is not None and after.sharpe_ratio is not None
        assert after.sharpe_ratio < before.sharpe_ratio, (
            f"{before.symbol}: charging 5% for cash must lower the Sharpe, "
            f"got {before.sharpe_ratio} -> {after.sharpe_ratio}"
        )
