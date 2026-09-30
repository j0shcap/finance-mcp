"""In-memory MCP protocol tests for the calculator tools."""

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from fastmcp.exceptions import ToolError

from finance_mcp.data.errors import InvalidInput
from finance_mcp.tools._dispatch import run_calc


async def test_tools_are_registered(client: Client[FastMCPTransport]) -> None:
    names = {tool.name for tool in await client.list_tools()}
    assert {"time_value_of_money", "loan_schedule"} <= names


async def test_time_value_of_money_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "time_value_of_money",
        {"solve_for": "fv", "pv": -1000.0, "pmt": 0.0, "rate": 0.05, "nper": 10.0},
    )
    assert result.data.solved_for == "fv"
    assert result.data.solved_value == pytest.approx(1628.894627, rel=1e-6)


async def test_loan_schedule_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "loan_schedule",
        {"principal": 200000.0, "annual_rate": 0.06, "term_months": 360},
    )
    assert result.data.monthly_payment == pytest.approx(1199.101, rel=1e-5)
    assert result.data.n_payments == 360


async def test_invalid_input_surfaces_as_tool_error(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(
            "time_value_of_money",
            {"solve_for": "fv", "pv": -1000.0, "pmt": 0.0, "nper": 10.0},  # missing rate
        )


async def test_loan_schedule_invalid_input_surfaces_as_tool_error(
    client: Client[FastMCPTransport],
) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(
            "loan_schedule",
            {"principal": 1000.0, "annual_rate": 0.05, "term_months": 0},
        )


async def test_new_tools_registered(client: Client[FastMCPTransport]) -> None:
    names = {t.name for t in await client.list_tools()}
    assert {"npv", "irr", "xnpv", "xirr", "convert_rate"} <= names


async def test_npv_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "npv", {"rate": 0.10, "cashflows": [-1000.0, 500.0, 500.0, 500.0]}
    )
    assert result.data.npv == pytest.approx(243.426, rel=1e-4)


async def test_irr_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool("irr", {"cashflows": [-100.0, 110.0]})
    assert result.data.irr == pytest.approx(0.10, rel=1e-9)


async def test_xirr_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "xirr",
        {
            "cashflows": [
                {"date": "2021-01-01", "amount": -1000.0},
                {"date": "2022-01-01", "amount": 1100.0},
            ]
        },
    )
    assert result.data.irr == pytest.approx(0.10, rel=1e-6)


async def test_convert_rate_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "convert_rate",
        {"rate": 0.12, "periods_per_year": 12, "direction": "nominal_to_effective"},
    )
    assert result.data.converted_rate == pytest.approx(0.12682503, rel=1e-7)


async def test_loan_schedule_tool_summary_default(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "loan_schedule", {"principal": 200000.0, "annual_rate": 0.06, "term_months": 360}
    )
    assert result.data.schedule == []
    assert result.data.n_payments == 360


async def test_loan_schedule_tool_with_rows(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "loan_schedule",
        {
            "principal": 200000.0,
            "annual_rate": 0.06,
            "term_months": 360,
            "include_schedule": True,
        },
    )
    assert len(result.data.schedule) == 360


async def test_irr_tool_no_sign_change_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool("irr", {"cashflows": [100.0, 200.0]})


async def test_npv_tool_empty_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool("npv", {"rate": 0.1, "cashflows": []})


async def test_xnpv_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "xnpv",
        {
            "rate": 0.10,
            "cashflows": [
                {"date": "2020-01-01", "amount": -1000.0},
                {"date": "2021-01-01", "amount": 1100.0},
            ],
        },
    )
    # 366 days apart (2020 is a leap year): -1000 + 1100 / 1.1 ** (366 / 365)
    assert result.data.npv == pytest.approx(-0.2609, abs=1e-3)


async def test_xnpv_tool_empty_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool("xnpv", {"rate": 0.1, "cashflows": []})


async def test_xirr_tool_no_sign_change_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(
            "xirr",
            {
                "cashflows": [
                    {"date": "2021-01-01", "amount": 100.0},
                    {"date": "2022-01-01", "amount": 200.0},
                ]
            },
        )


async def test_convert_rate_tool_invalid_nominal_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(
            "convert_rate",
            {"rate": -20.0, "periods_per_year": 12, "direction": "nominal_to_effective"},
        )


async def test_bond_tools_registered(client: Client[FastMCPTransport]) -> None:
    names = {t.name for t in await client.list_tools()}
    assert {"bond_price", "bond_ytm"} <= names


async def test_bond_price_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "bond_price",
        {
            "face": 1000.0,
            "coupon_rate": 0.06,
            "years_to_maturity": 10.0,
            "ytm": 0.06,
            "frequency": 2,
        },
    )
    assert result.data.price == pytest.approx(1000.0, rel=1e-6)


async def test_bond_ytm_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "bond_ytm",
        {
            "face": 1000.0,
            "coupon_rate": 0.06,
            "years_to_maturity": 10.0,
            "price": 1000.0,
            "frequency": 2,
        },
    )
    assert result.data.yield_to_maturity == pytest.approx(0.06, rel=1e-6)


async def test_tvm_when_begin_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "time_value_of_money",
        {"solve_for": "fv", "pv": 0.0, "pmt": -100.0, "rate": 0.05, "nper": 10.0, "when": "begin"},
    )
    assert result.data.solved_value == pytest.approx(1320.679, rel=1e-5)


async def test_bond_price_tool_invalid_errors(client: Client[FastMCPTransport]) -> None:
    # ytm = -frequency makes 1 + ytm/frequency zero. Not schema-blocked, so it reaches
    # the data layer and surfaces as ToolError.
    with pytest.raises(ToolError):
        await client.call_tool(
            "bond_price",
            {
                "face": 1000.0,
                "coupon_rate": 0.05,
                "years_to_maturity": 10.0,
                "ytm": -2.0,
                "frequency": 2,
            },
        )


async def test_convert_rate_tool_continuous(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "convert_rate",
        {
            "rate": 0.12,
            "periods_per_year": 1,
            "direction": "nominal_to_effective",
            "compounding": "continuous",
        },
    )
    assert result.data.converted_rate == pytest.approx(0.12749685, rel=1e-7)
    assert result.data.compounding == "continuous"


async def test_bond_ytm_tool_non_integer_periods_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(
            "bond_ytm",
            {
                "face": 1000.0,
                "coupon_rate": 0.05,
                "years_to_maturity": 2.5,
                "price": 950.0,
                "frequency": 1,
            },
        )


async def test_mirr_tool_registered(client: Client[FastMCPTransport]) -> None:
    names = {t.name for t in await client.list_tools()}
    assert "mirr" in names


async def test_mirr_tool(client: Client[FastMCPTransport]) -> None:
    result = await client.call_tool(
        "mirr",
        {
            "cashflows": [-1000.0, 500.0, 400.0, 300.0, 100.0],
            "finance_rate": 0.10,
            "reinvest_rate": 0.12,
        },
    )
    assert result.data.mirr == pytest.approx(0.13168560, rel=1e-6)


async def test_mirr_tool_invalid_errors(client: Client[FastMCPTransport]) -> None:
    with pytest.raises(ToolError):
        await client.call_tool(
            "mirr",
            {"cashflows": [-100.0, -50.0], "finance_rate": 0.1, "reinvest_rate": 0.1},
        )


def test_run_calc_translates_zero_division() -> None:
    def boom() -> float:
        return 1.0 / 0.0

    with pytest.raises(ToolError, match="out of range"):
        run_calc(boom)


def test_run_calc_translates_overflow() -> None:
    def boom() -> float:
        raise OverflowError("(34, 'Result too large')")

    with pytest.raises(ToolError, match="out of range"):
        run_calc(boom)


def test_run_calc_translates_value_error() -> None:
    # Covers pydantic ValidationError too, which subclasses ValueError.
    def boom() -> float:
        raise ValueError("math domain error")

    with pytest.raises(ToolError, match="out of range"):
        run_calc(boom)


def test_run_calc_passes_through_invalid_input_message() -> None:
    def boom() -> float:
        raise InvalidInput("principal must be positive.")

    with pytest.raises(ToolError, match=r"principal must be positive\."):
        run_calc(boom)


async def test_loan_schedule_rate_overflow_surfaces_as_tool_error(
    client: Client[FastMCPTransport],
) -> None:
    # annual_rate=1e4 satisfies the Field(ge=0) bound, so it reaches the data layer.
    # The actionable, model-facing message must survive the run_calc -> ToolError hop.
    with pytest.raises(ToolError, match="annual_rate is too large for this term"):
        await client.call_tool(
            "loan_schedule",
            {"principal": 1000.0, "annual_rate": 1e4, "term_months": 360},
        )


async def test_time_value_of_money_overflow_surfaces_as_tool_error(
    client: Client[FastMCPTransport],
) -> None:
    # No Field bound can screen this: (1 + 1e5)**1e5 overflows inside the calculator.
    # Unhandled, the client saw only the bare "(34, 'Result too large')"; the
    # defence-in-depth clause in run_calc must frame it as an input problem.
    with pytest.raises(ToolError, match="out of range for this calculation"):
        await client.call_tool(
            "time_value_of_money",
            {"solve_for": "fv", "pv": -1000.0, "pmt": 0.0, "rate": 1e5, "nper": 1e5},
        )


async def test_dated_bond_tools_registered(client: Client[FastMCPTransport]) -> None:
    names = {t.name for t in await client.list_tools()}
    assert {"bond_price_dated", "bond_ytm_dated"} <= names


async def test_bond_price_dated_tool(client: Client[FastMCPTransport]) -> None:
    """The Excel PRICE documentation example, over the protocol. Dates cross the wire as
    ISO 8601 strings in both directions."""
    result = await client.call_tool(
        "bond_price_dated",
        {
            "settlement": "2008-02-15",
            "maturity": "2017-11-15",
            "coupon_rate": 0.0575,
            "ytm": 0.065,
            "face": 100.0,
            "frequency": 2,
            "day_count": "30/360",
        },
    )
    assert result.data.clean_price == pytest.approx(94.634362, abs=1e-6)
    assert result.data.accrued_interest == pytest.approx(1.4375, abs=1e-9)
    assert result.data.dirty_price == pytest.approx(96.071862, abs=1e-6)
    assert result.data.previous_coupon_date == "2007-11-15"
    assert result.data.next_coupon_date == "2008-05-15"
    assert result.data.periods_remaining == 20


async def test_bond_price_dated_tool_defaults_to_actual_actual(
    client: Client[FastMCPTransport],
) -> None:
    result = await client.call_tool(
        "bond_price_dated",
        {
            "settlement": "2008-02-15",
            "maturity": "2017-11-15",
            "coupon_rate": 0.0575,
            "ytm": 0.065,
        },
    )
    assert result.data.day_count == "actual/actual"
    assert result.data.period_days == 182.0


async def test_bond_ytm_dated_tool(client: Client[FastMCPTransport]) -> None:
    """The Excel YIELD documentation example, over the protocol."""
    result = await client.call_tool(
        "bond_ytm_dated",
        {
            "settlement": "2008-02-15",
            "maturity": "2016-11-15",
            "coupon_rate": 0.0575,
            "clean_price": 95.04287,
            "face": 100.0,
            "frequency": 2,
            "day_count": "30/360",
        },
    )
    assert result.data.yield_to_maturity == pytest.approx(0.065, abs=1e-6)
    assert result.data.dirty_price == pytest.approx(96.48037, abs=1e-5)


async def test_bond_price_dated_tool_settlement_after_maturity_errors(
    client: Client[FastMCPTransport],
) -> None:
    # A two-argument relationship no Field bound can express, so it reaches the data
    # layer and surfaces as a ToolError carrying the calculator's message.
    with pytest.raises(ToolError, match="strictly before maturity"):
        await client.call_tool(
            "bond_price_dated",
            {
                "settlement": "2018-02-15",
                "maturity": "2017-11-15",
                "coupon_rate": 0.0575,
                "ytm": 0.065,
            },
        )


async def test_bond_ytm_dated_tool_odd_frequency_errors(
    client: Client[FastMCPTransport],
) -> None:
    with pytest.raises(ToolError, match="divide 12 evenly"):
        await client.call_tool(
            "bond_ytm_dated",
            {
                "settlement": "2008-02-15",
                "maturity": "2017-11-15",
                "coupon_rate": 0.0575,
                "clean_price": 95.0,
                "frequency": 5,
            },
        )
