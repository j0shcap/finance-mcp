"""DataService runs on any provider that implements the ports - not only on Yahoo.

The rest of the suite drives DataService through YahooProvider over yfinance-shaped fakes.
This provider is plain Python, so these flows prove the logic layer needs nothing from
yfinance: quotes, news relevance (two ports combined), and a default risk-free rate.
"""

from datetime import date, timedelta

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import (
    AnalystData,
    CompanyProfile,
    FinancialStatement,
    Identity,
    KeyMetrics,
    NewsArticle,
    NewsSource,
    PriceBar,
    Quote,
    Statement,
    StatementPeriod,
    SymbolSearchResult,
)
from finance_mcp.data.service import DataService


def _daily(closes: list[float]) -> list[PriceBar]:
    start = date(2025, 1, 6)
    return [
        PriceBar(
            date=(start + timedelta(days=i)).isoformat(),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
        )
        for i, c in enumerate(closes)
    ]


class InMemoryProvider:
    """Serves fixed data through every port, in the ports' own vocabulary."""

    def quote(self, symbol: str) -> Quote:
        if symbol != "ACME":
            raise SymbolNotFound(f"No quote for '{symbol}'.")
        return Quote(symbol=symbol, currency="USD", price=10.0)

    def bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]:
        return _daily([100.0 * 1.001**i for i in range(300)])

    def treasury_bill_yields(self, period: str) -> list[PriceBar]:
        return _daily([4.0] * 300)

    def financial_statement(
        self, symbol: str, statement: Statement, period: StatementPeriod
    ) -> FinancialStatement:
        raise DataUnavailable("not served by this provider")

    def statement_currency(self, symbol: str) -> str | None:
        return None

    def profile(self, symbol: str) -> CompanyProfile:
        raise DataUnavailable("not served by this provider")

    def key_metrics(self, symbol: str) -> KeyMetrics:
        raise DataUnavailable("not served by this provider")

    def analyst_data(self, symbol: str) -> AnalystData:
        raise DataUnavailable("not served by this provider")

    def identity(self, symbol: str) -> Identity:
        return Identity(is_company=True, long_name="Acme Corporation", short_name="Acme")

    def search(self, query: str, max_results: int) -> SymbolSearchResult:
        raise DataUnavailable("not served by this provider")

    def news(self, symbol: str, count: int) -> tuple[list[NewsArticle], NewsSource]:
        return [
            NewsArticle(title="Acme beats on revenue"),
            NewsArticle(title="Stocks slip as yields rise"),
        ], "ticker"


def _service() -> DataService:
    provider = InMemoryProvider()
    return DataService(market=provider, news=provider)


def test_quotes_come_from_the_market_port() -> None:
    result = _service().get_quote(["acme", "NOPE"])
    assert [q.symbol for q in result.quotes] == ["ACME"]
    assert [e.symbol for e in result.errors] == ["NOPE"]


def test_news_relevance_combines_the_news_and_market_ports() -> None:
    result = _service().get_news("ACME")
    assert result.relevance_check == "applied"
    assert [a.mentions_company for a in result.articles] == [True, False]


def test_the_default_risk_free_rate_comes_from_the_treasury_bill_port() -> None:
    stats = _service().analyze_performance("ACME", "1y")
    assert stats.risk_free_rate_source == "treasury_bill"
    assert stats.risk_free_rate is not None and 0.03 < stats.risk_free_rate < 0.05
