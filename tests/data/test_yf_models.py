"""The data-layer error and result models."""

from finance_mcp.data.errors import DataUnavailable, SymbolNotFound
from finance_mcp.data.models import CompanyProfile


def test_symbol_not_found_is_a_data_unavailable() -> None:
    # The tools translate DataUnavailable, so a missing symbol reaches the model the same way.
    assert issubclass(SymbolNotFound, DataUnavailable)
    assert str(SymbolNotFound("boom")) == "boom"


def test_company_profile_defaults_to_no_events() -> None:
    empty = CompanyProfile(symbol="MSFT")
    assert empty.recent_dividends == []
    assert empty.splits == []
    assert empty.name is None
