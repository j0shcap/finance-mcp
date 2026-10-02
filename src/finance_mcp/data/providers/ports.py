"""The interfaces a data provider implements: all the logic layer knows about providers.

DataService (data/service.py) orchestrates and caches over these; an adapter such as
providers/yahoo.py turns one provider's API into them. There is one port per source that
could be replaced independently.

Every method returns this package's provider-neutral models and raises only
SymbolNotFound (the provider has no such instrument) or DataUnavailable (anything else).
A provider's own exceptions, types and field names never cross a port. Symbols arrive
normalized (upper-case, stripped); nothing here caches.

The period, interval and statement vocabularies in models.py, and NewsSource's values, are
the neutral vocabulary: an adapter maps them to its provider's terms.
"""

from typing import Protocol

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


class MarketDataProvider(Protocol):
    """Prices, fundamentals and symbol lookup."""

    def quote(self, symbol: str) -> Quote: ...

    def bars(self, symbol: str, period: str, interval: str) -> list[PriceBar]: ...

    def treasury_bill_yields(self, period: str) -> list[PriceBar]:
        """Daily 13-week US Treasury bill yields over ``period``, for the default risk-free
        rate: each close is the bank-discount yield in percent (4.03 = 4.03%)."""
        ...

    def financial_statement(
        self, symbol: str, statement: Statement, period: StatementPeriod
    ) -> FinancialStatement: ...

    def statement_currency(self, symbol: str) -> str | None:
        """The currency ``symbol`` reports its statements in, if known.

        Separate from financial_statement only because Yahoo serves it from another
        endpoint; a provider that returns it with the statement can answer it from there.
        """
        ...

    def profile(self, symbol: str) -> CompanyProfile: ...

    def key_metrics(self, symbol: str) -> KeyMetrics: ...

    def analyst_data(self, symbol: str) -> AnalystData: ...

    def identity(self, symbol: str) -> Identity:
        """The names and kind of instrument ``symbol`` is, for news relevance."""
        ...

    def search(self, query: str, max_results: int) -> SymbolSearchResult: ...


class NewsProvider(Protocol):
    """Recent news about a symbol."""

    def news(self, symbol: str, count: int) -> tuple[list[NewsArticle], NewsSource]:
        """Up to ``count`` articles, newest first, and which kind of feed served them."""
        ...
