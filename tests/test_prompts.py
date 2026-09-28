"""Tests for analysis prompts (in-memory Client; no network)."""

from fastmcp import Client

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
            "time_value_of_money",
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
