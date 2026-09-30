"""Tests for analysis prompts (in-memory Client; no network)."""

import re

import pytest
from fastmcp import Client
from mcp.shared.exceptions import McpError
from mcp.types import TextContent

from finance_mcp.conventions import CALCULATOR_CONVENTIONS, CONVENTIONS_URI, UNITS_GLOSSARY
from finance_mcp.prompts._render import DISCLAIMER
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


# --- shared contract for the prompts that point at the conventions resource ---------

#: Sample arguments for each prompt that references finance://conventions instead of
#: embedding the glossary (analyze_stock embeds it, by design, and is tested above).
REFERENCING_PROMPTS: dict[str, dict[str, str]] = {
    "investment_cashflows": {"cashflows": "-1000 now, then 300 a year for 5 years"},
    "bond_analysis": {"bond": "UST 4% due 2036-01-15, settles 2026-03-15, clean price 92.30"},
    "loan_planner": {"principal": "400000", "annual_rate": "6.5%", "term_months": "360"},
    "compare_stocks": {"tickers": "aapl, msft  googl"},
}


async def _render(name: str, args: dict[str, str]) -> str:
    async with Client(create_server()) as client:
        result = await client.get_prompt(name, args)
        assert len(result.messages) == 1
        content = result.messages[0].content
        assert isinstance(content, TextContent)
        return content.text


@pytest.mark.parametrize("name", sorted(REFERENCING_PROMPTS))
async def test_prompt_references_conventions_instead_of_duplicating_them(name: str) -> None:
    text = await _render(name, REFERENCING_PROMPTS[name])
    assert CONVENTIONS_URI in text
    assert UNITS_GLOSSARY not in text
    assert CALCULATOR_CONVENTIONS not in text


@pytest.mark.parametrize("name", sorted(REFERENCING_PROMPTS))
async def test_prompt_renders_completely_and_ends_with_the_disclaimer(name: str) -> None:
    text = await _render(name, REFERENCING_PROMPTS[name])
    assert not re.search(r"\{[a-z_]+\}", text)  # no unrendered placeholder
    assert text.rstrip().endswith(DISCLAIMER)


@pytest.mark.parametrize("name", sorted(REFERENCING_PROMPTS))
async def test_prompt_insists_on_tool_computation_not_mental_arithmetic(name: str) -> None:
    text = await _render(name, REFERENCING_PROMPTS[name])
    assert "never do that arithmetic yourself" in text


# --- investment_cashflows ------------------------------------------------------------


async def _prompt_arguments(name: str) -> dict[str, bool]:
    async with Client(create_server()) as client:
        by_name = {p.name: p for p in await client.list_prompts()}
        return {a.name: bool(a.required) for a in (by_name[name].arguments or [])}


async def test_investment_cashflows_arguments() -> None:
    assert await _prompt_arguments("investment_cashflows") == {
        "cashflows": True,
        "discount_rate": False,
        "reinvest_rate": False,
    }


async def test_investment_cashflows_echoes_inputs_and_marks_missing_rates() -> None:
    text = await _render("investment_cashflows", {"cashflows": "-500, 200, 200, 200"})
    assert "-500, 200, 200, 200" in text
    assert "Discount (hurdle) rate: not given" in text
    assert "Reinvestment rate for MIRR: not given" in text
    text = await _render(
        "investment_cashflows",
        {"cashflows": "-500, 200", "discount_rate": "8%", "reinvest_rate": "5%"},
    )
    assert "Discount (hurdle) rate: 8%" in text
    assert "Reinvestment rate for MIRR: 5%" in text


async def test_investment_cashflows_pins_the_timing_and_period_conventions() -> None:
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "npv treats cashflows[0] as today" in text
    assert "Excel's NPV() discounts its first" in text  # the classic off-by-one-period
    assert '"effective_to_nominal"' in text  # per-period rate from an annual one
    assert "xnpv / xirr" in text  # irregular dates


async def test_investment_cashflows_makes_npv_the_decision_rule() -> None:
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "NPV is the decision rule" in text
    assert "NPV wins" in text
    assert "NPV profile" in text


async def test_investment_cashflows_explains_multiple_and_borrowing_type_irrs() -> None:
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "Descartes" in text  # sign changes bound the number of IRRs
    assert "is_unique" in text
    assert "all_irrs" in text
    assert "there is NO single IRR" in text
    # irr returns the same root for a loan as for its mirror-image investment.
    assert "Borrowing-type" in text
    assert "higher IRR is WORSE" in text


async def test_investment_cashflows_says_when_to_prefer_mirr_and_its_limits() -> None:
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "Prefer MIRR when" in text
    assert "finance_rate" in text
    assert "reinvest_rate" in text
    assert "MIRR exceeds the hurdle exactly when NPV is positive" in text
    assert "There is no dated MIRR tool" in text
    assert "crossover rate" in text  # ranking mutually exclusive projects


async def test_investment_cashflows_verifies_the_irr_by_repricing() -> None:
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "npv (or xnpv) at the IRR must be ~0" in text


async def test_investment_cashflows_annualizes_through_convert_rate() -> None:
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "convert_rate(rate=r x k, periods_per_year=k" in text


async def test_investment_cashflows_states_the_hurdle_cushion_per_flow_type() -> None:
    """Borrowing-type NPV rises with the rate, so its cushion runs the other way."""
    text = await _render("investment_cashflows", REFERENCING_PROMPTS["investment_cashflows"])
    assert "investment-type NPV turns negative as the hurdle RISES to the IRR" in text
    assert "borrowing-type NPV as the hurdle FALLS to it" in text


# --- bond_analysis -------------------------------------------------------------------


async def test_bond_analysis_arguments() -> None:
    assert await _prompt_arguments("bond_analysis") == {"bond": True, "shock_bp": False}


async def test_bond_analysis_echoes_the_bond_and_defaults_to_a_100bp_shock() -> None:
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert REFERENCING_PROMPTS["bond_analysis"]["bond"] in text
    assert "+/-100bp" in text
    text = await _render("bond_analysis", {"bond": "a 5y 3% corporate at 97", "shock_bp": "50"})
    assert "+/-50bp" in text
    assert "+/-100bp" not in text


async def test_bond_analysis_normalizes_the_shock() -> None:
    text = await _render("bond_analysis", {"bond": "a 5y 3% corporate at 97", "shock_bp": " 25BP "})
    assert "+/-25bp" in text


@pytest.mark.parametrize("shock_bp", ["1%", "0.01", "-50", "abc", "inf", ""])
async def test_bond_analysis_rejects_a_shock_that_is_not_basis_points(shock_bp: str) -> None:
    """The error reaches the client despite mask_error_details, naming the problem."""
    with pytest.raises(McpError, match="shock_bp must be a number of basis points"):
        await _render("bond_analysis", {"bond": "a 5y 3% corporate at 97", "shock_bp": shock_bp})


async def test_bond_analysis_picks_the_dated_tools_and_conventions() -> None:
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert "bond_price_dated" in text
    assert "bond_ytm_dated" in text
    assert '"30/360"' in text  # US corporates/munis, not the actual/actual default
    assert "market quotes are CLEAN" in text
    assert "subtract accrued interest" in text  # dirty -> clean before solving
    assert "read accrued_interest from bond_price_dated at any ytm" in text


async def test_bond_analysis_knows_ytm_dated_returns_no_risk_figures() -> None:
    """bond_ytm_dated returns prices and a yield only; duration needs a repricing call."""
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert "bond_ytm_dated returns the yield and the prices but no duration" in text


async def test_bond_analysis_sanity_checks_premium_and_discount() -> None:
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert "a coupon_rate above the yield means a premium" in text
    # A dated bond with coupon == yield prices a few thousandths below 100 clean.
    assert "Near par that is only approximate" in text


async def test_bond_analysis_measures_dv01_and_convexity_on_the_dirty_price() -> None:
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert "DV01 = modified_duration x dirty_price x 0.0001" in text
    assert "measured on the DIRTY price" in text
    assert "do not rescale it" in text  # convexity is already in years^2


async def test_bond_analysis_cross_checks_the_shock_estimate_by_repricing() -> None:
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert "-modified_duration x dy + 0.5 x convexity x dy^2" in text
    assert "Reprice exactly" in text
    assert "The exact repricing is authoritative" in text
    assert "same currency amount" in text  # clean and dirty move by the same amount


async def test_bond_analysis_names_where_option_free_math_breaks() -> None:
    text = await _render("bond_analysis", REFERENCING_PROMPTS["bond_analysis"])
    assert "callable" in text
    assert "negative convexity" in text
    assert "not an expected return" in text  # YTM is a promised yield
    assert '"nominal_to_effective"' in text  # compare yields on one compounding basis


# --- loan_planner --------------------------------------------------------------------


async def test_loan_planner_arguments() -> None:
    assert await _prompt_arguments("loan_planner") == {
        "principal": True,
        "annual_rate": True,
        "term_months": True,
        "extra_payment": False,
    }


async def test_loan_planner_echoes_inputs_and_marks_a_missing_extra_payment() -> None:
    text = await _render("loan_planner", REFERENCING_PROMPTS["loan_planner"])
    assert "Principal: 400000" in text
    assert "Annual rate: 6.5%" in text
    assert "Term: 360 months" in text
    assert "Extra monthly principal payment: not given" in text
    text = await _render(
        "loan_planner", {**REFERENCING_PROMPTS["loan_planner"], "extra_payment": "250"}
    )
    assert "Extra monthly principal payment: 250" in text


async def test_loan_planner_separates_the_note_rate_from_a_disclosed_apr() -> None:
    text = await _render("loan_planner", REFERENCING_PROMPTS["loan_planner"])
    assert "NOT the rate to amortize at" in text  # a TILA APR folds in fees
    assert '"nominal_to_effective"' in text  # EAR of the note rate


async def test_loan_planner_converts_non_monthly_compounding() -> None:
    text = await _render("loan_planner", REFERENCING_PROMPTS["loan_planner"])
    assert "Canadian fixed-rate mortgages compound semiannually" in text
    assert '"effective_to_nominal"' in text
    assert "adjustable-rate" in text  # the tools assume one fixed rate


async def test_loan_planner_verifies_the_payment_and_reads_balances_from_tvm() -> None:
    text = await _render("loan_planner", REFERENCING_PROMPTS["loan_planner"])
    assert 'time_value_of_money(solve_for="pmt"' in text
    assert 'time_value_of_money(solve_for="fv"' in text
    assert "NEGATIVE of the fv" in text  # the balance comes back as a negative fv


async def test_loan_planner_frames_extra_payments_honestly() -> None:
    text = await _render("loan_planner", REFERENCING_PROMPTS["loan_planner"])
    assert "undiscounted sum" in text
    assert "exactly the loan's note rate" in text
    assert "only a recast" in text
    assert "prepayment penalties" in text


async def test_loan_planner_rejects_the_naive_refinance_break_even() -> None:
    text = await _render("loan_planner", REFERENCING_PROMPTS["loan_planner"])
    assert "naive break-even" in text
    assert "wrong when the term resets" in text
    assert "same remaining term" in text  # isolates the rate effect
    assert "rolled into the loan are already in its balance" in text  # no double count


# --- compare_stocks ------------------------------------------------------------------


async def test_compare_stocks_arguments() -> None:
    assert await _prompt_arguments("compare_stocks") == {"tickers": True, "horizon": False}


async def test_compare_stocks_renders_the_parsed_ticker_list_into_the_tool_calls() -> None:
    text = await _render("compare_stocks", REFERENCING_PROMPTS["compare_stocks"])
    assert 'compare_tickers(tickers=["AAPL", "MSFT", "GOOGL"], period="1y"' in text
    assert 'get_quote(tickers=["AAPL", "MSFT", "GOOGL"])' in text
    assert "AAPL, MSFT, GOOGL" in text  # the readable list in the framing
    assert "12mo" in text  # default horizon


async def test_compare_stocks_uses_the_custom_horizon() -> None:
    text = await _render("compare_stocks", {"tickers": "KO PEP", "horizon": "5y"})
    assert "for a 5y investment horizon" in text


@pytest.mark.parametrize(
    ("tickers", "message"),
    [("AAPL", "at least 2"), (",".join(f"T{i}" for i in range(11)), "at most 10")],
)
async def test_compare_stocks_rejects_a_ticker_count_compare_tickers_cannot_take(
    tickers: str, message: str
) -> None:
    """The error reaches the client despite mask_error_details, naming the problem."""
    with pytest.raises(McpError, match=message):
        await _render("compare_stocks", {"tickers": tickers})


async def test_compare_stocks_checks_rank_stability_across_two_windows() -> None:
    text = await _render("compare_stocks", REFERENCING_PROMPTS["compare_stocks"])
    assert 'period="1y"' in text
    assert 'period="5y"' in text
    assert "If the ranking flips between windows" in text


async def test_compare_stocks_screens_comparability_and_fixes_the_rubric_first() -> None:
    text = await _render("compare_stocks", REFERENCING_PROMPTS["compare_stocks"])
    assert "Comparability screen (before any ranking)" in text
    assert "split the list into comparable groups" in text
    assert "Declare the rubric BEFORE reading the results" in text


async def test_compare_stocks_derives_growth_adjustment_instead_of_trusting_peg() -> None:
    text = await _render("compare_stocks", REFERENCING_PROMPTS["compare_stocks"])
    assert "peg_ratio from Yahoo is often null or stale" in text
    assert "trailing_pe / forward_pe - 1" in text
    assert "years is the number of annual periods returned minus 1" in text
    assert "undefined when either endpoint is zero or negative" in text
    assert "ONLY when both are" in text
    assert "Value-trap check" in text


async def test_compare_stocks_reads_partial_results_calendars_and_currency() -> None:
    text = await _render("compare_stocks", REFERENCING_PROMPTS["compare_stocks"])
    for field in ("metrics_error", "periods_per_year", "currency_differs", "mixed_currencies"):
        assert field in text


async def test_compare_stocks_ranks_honestly() -> None:
    text = await _render("compare_stocks", REFERENCING_PROMPTS["compare_stocks"])
    assert '"insufficient data" bucket - not last place' in text
    assert "call it a tie" in text
    assert "what would change its rank" in text
