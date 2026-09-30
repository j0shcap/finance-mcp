"""The data-layer error and result models."""

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    CompanyProfile,
    DividendEvent,
    FinancialStatement,
    PriceBar,
    PriceHistory,
    PriceSummary,
    Quote,
    SplitEvent,
)


def test_models_and_errors_exist() -> None:
    q = Quote(
        symbol="AAPL",
        currency="USD",
        price=190.0,
        previous_close=188.0,
        change=2.0,
        change_percent=1.06,
        day_high=191.0,
        day_low=187.0,
        year_high=200.0,
        year_low=150.0,
        market_cap=3.0e12,
        volume=50_000_000,
    )
    assert q.symbol == "AAPL"
    bar = PriceBar(date="2024-01-02", open=1.0, high=2.0, low=0.5, close=1.5, volume=100)
    summary = PriceSummary(
        start_date="2024-01-02",
        end_date="2024-01-03",
        start_close=1.5,
        end_close=1.6,
        total_return_percent=6.67,
        period_high=2.0,
        period_low=0.5,
        bars=2,
    )
    hist = PriceHistory(
        symbol="AAPL", period="1mo", interval="1d", bars=[bar], summary=summary, truncated=False
    )
    assert hist.summary.bars == 2
    assert issubclass(SymbolNotFound, DataUnavailable)
    assert str(DataUnavailable("boom")) == "boom"


def test_fundamentals_and_profile_models() -> None:
    stmt = FinancialStatement(
        symbol="AAPL",
        statement="income",
        period="annual",
        period_ends=["2024-09-30", "2023-09-30"],
        line_items={"Total Revenue": [391_035.0, None]},
    )
    assert stmt.statement == "income"
    assert stmt.period == "annual"
    assert stmt.period_ends[0] == "2024-09-30"
    assert stmt.line_items["Total Revenue"] == [391_035.0, None]

    profile = CompanyProfile(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        market_cap=3.0e12,
        recent_dividends=[DividendEvent(date="2024-08-12", amount=0.25)],
        splits=[SplitEvent(date="2020-08-31", ratio=4.0)],
    )
    assert profile.symbol == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.recent_dividends[0].amount == 0.25
    assert profile.splits[0].ratio == 4.0

    empty = CompanyProfile(symbol="MSFT")
    assert empty.recent_dividends == []
    assert empty.splits == []
    assert empty.name is None
