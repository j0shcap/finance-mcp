"""Tests for analysis prompts (in-memory Client; no network)."""

from fastmcp import Client

from finance_mcp.conventions import CONVENTIONS_URI, UNITS_GLOSSARY
from finance_mcp.server import create_server


async def test_analyze_stock_is_registered_with_expected_arguments() -> None:
    async with Client(create_server()) as client:
        prompts = await client.list_prompts()
        by_name = {p.name: p for p in prompts}
        assert "analyze_stock" in by_name
        args = {a.name: a for a in (by_name["analyze_stock"].arguments or [])}
        assert set(args) == {"ticker", "horizon"}
        assert args["ticker"].required is True
        assert args["horizon"].required is False


async def test_analyze_stock_interpolates_ticker_and_default_horizon() -> None:
    async with Client(create_server()) as client:
        result = await client.get_prompt("analyze_stock", {"ticker": "AAPL"})
        assert len(result.messages) == 1
        text = result.messages[0].content.text
        assert "AAPL" in text
        assert "12mo" in text  # default horizon
        assert "{" not in text  # no unrendered format placeholders remain


async def test_analyze_stock_interpolates_custom_horizon() -> None:
    async with Client(create_server()) as client:
        result = await client.get_prompt("analyze_stock", {"ticker": "MSFT", "horizon": "3y"})
        text = result.messages[0].content.text
        assert "MSFT" in text
        assert "3y" in text


async def test_analyze_stock_references_the_tools_it_orchestrates() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        for tool in (
            "get_company_profile",
            "get_financials",
            "get_key_metrics",
            "analyze_performance",
            "get_analyst_data",
            "get_news",
            "get_quote",
            "compare_tickers",
            "compare_to_benchmark",
        ):
            assert tool in text


async def test_analyze_stock_embeds_unit_guardrails() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "FRACTIONS" in text  # margins/ROE are fractions
        assert "ALREADY A PERCENT" in text  # debt_to_equity / dividend_yield
        assert "INVERTED" in text  # recommendation_mean scale
        assert "auto-adjusted" in text  # no dividend double-count


async def test_analyze_stock_warns_that_annualized_figures_can_be_null() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "periods_per_year" in text
        assert "null" in text  # short windows do not report annualized figures


async def test_analyze_stock_ends_with_disclaimer() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert text.rstrip().endswith("not investment advice. Always do your own due diligence.")


async def test_analyze_stock_embeds_the_shared_units_glossary() -> None:
    """The glossary has one definition (conventions.UNITS_GLOSSARY); the prompt renders it."""
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert UNITS_GLOSSARY in text
        assert CONVENTIONS_URI in text


async def test_analyze_stock_states_the_forward_pe_check_correctly() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "forward_eps above trailing_eps" in text
        assert "expected earnings growth" in text
        assert "forward P/E below the trailing P/E" in text
        assert "forward P/E / forward_eps" not in text  # the garbled original


async def test_analyze_stock_qualifies_the_forward_pe_shortcut_for_negative_eps() -> None:
    """A forward P/E below trailing P/E only implies growth while trailing_eps > 0; with
    negative trailing EPS the trailing P/E is meaningless and the shortcut inverts."""
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "when trailing_eps is zero or negative" in text
        assert "compare the EPS figures directly" in text


async def test_analyze_stock_computes_implied_return_arithmetically() -> None:
    """A 12-month implied return is target/price - 1; routing it through the TVM
    calculator (nper=1) computes the same thing with more ways to get the signs wrong."""
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "(mean target / current price) - 1" in text
        assert "time_value_of_money" not in text


async def test_analyze_stock_no_longer_claims_sharpe_is_unavailable() -> None:
    """The caveat this change exists to delete: the server now computes a Sharpe ratio."""
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "no Sharpe" not in text


async def test_analyze_stock_uses_the_comparison_tools() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "compare_tickers" in text
        assert "compare_to_benchmark" in text


async def test_analyze_stock_reads_risk_posture_from_sharpe_and_beta() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "sharpe_ratio" in text
        assert "beta" in text
        # beta must come from the benchmark comparison, not only the profile's ~5y figure.
        assert "compare_to_benchmark's beta" in text


async def test_analyze_stock_warns_that_a_thin_overlap_makes_beta_noisy() -> None:
    async with Client(create_server()) as client:
        text = (
            (await client.get_prompt("analyze_stock", {"ticker": "AAPL"})).messages[0].content.text
        )
        assert "overlapping_observations" in text
