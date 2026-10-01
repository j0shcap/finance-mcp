"""Every calculator, called over stdio on the installed server, plus the review regressions.

Offline: calculators are pure, so this half of the e2e suite needs no Yahoo and runs in CI
on every PR. The regression block re-proves each calculator finding from the 2026-09-27
review against the shipped artifact rather than the source tree: bad inputs must come back
as a clear tool error, never a ZeroDivisionError, a complex number, an OverflowError, or
the masked "internal error" an unexpected exception turns into.
"""

import re
from typing import Any

import pytest

from finance_mcp.conventions import CALCULATOR_TOOLS
from tests.e2e.conftest import Server
from tests.e2e.golden import GOLDEN

#: Tight enough to catch a wrong convention (those move results by percent), loose enough
#: for the root-finders' convergence and the reference examples' published rounding.
REL_TOL = 1e-6

#: Text that would mean the error escaped as an unhandled exception, not a domain error.
#: Exception class names rather than words: a clean message may well say "overflowed".
_UNHANDLED = (
    "ZeroDivisionError",
    "division by zero",
    "OverflowError",
    "math range error",
    "complex",
    "internal error",
    "Traceback",
)


def test_golden_table_covers_every_calculator() -> None:
    """A calculator added without a golden call would ship untested over the wire."""
    assert set(GOLDEN) == set(CALCULATOR_TOOLS)


async def _structured(server: Server, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await server.client.call_tool(tool, args)
    assert result.structured_content is not None, f"{tool} returned no structured content"
    return result.structured_content


@pytest.mark.parametrize("tool", sorted(GOLDEN))
async def test_golden_call(default_server: Server, tool: str) -> None:
    args, expected = GOLDEN[tool]
    result = await _structured(default_server, tool, args)

    for field, value in expected.items():
        want = pytest.approx(value, rel=REL_TOL) if isinstance(value, float) else value
        assert result[field] == want, f"{tool}.{field}: got {result[field]}, want {value}"


async def _tool_error(server: Server, tool: str, args: dict[str, Any]) -> str:
    """The error text a bad call reaches the model with, asserting it is a clean one."""
    result = await server.client.call_tool(tool, args, raise_on_error=False)
    assert result.is_error, f"{tool}({args}) should have failed, got {result.structured_content}"
    text = " ".join(getattr(block, "text", "") for block in result.content)
    for marker in _UNHANDLED:
        assert marker.lower() not in text.lower(), f"{tool}({args}) leaked {marker!r}: {text}"
    return text


TVM = "time_value_of_money"


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        pytest.param(
            TVM,
            {"solve_for": "pmt", "pv": -100, "fv": 0, "rate": 0.05, "nper": 0},
            "zero periods",
            id="tvm-pmt-nper-0",
        ),
        pytest.param(
            TVM,
            {"solve_for": "rate", "pv": -100, "fv": 110, "pmt": 0, "nper": 0},
            "zero periods|nper",
            id="tvm-rate-nper-0",
        ),
        pytest.param(
            TVM,
            {"solve_for": "fv", "pv": -100, "pmt": 0, "rate": -1, "nper": 5},
            "greater than -1",
            id="tvm-rate-minus-1",
        ),
        pytest.param(
            TVM,
            {"solve_for": "fv", "pv": -100, "pmt": 0, "rate": -1.5, "nper": 2.5},
            "greater than -1",
            id="tvm-rate-below-minus-1-fractional-nper",
        ),
        pytest.param(
            TVM,
            {"solve_for": "nper", "pv": 0, "pmt": 0, "fv": 100, "rate": 0.05},
            "pv and pmt are both zero",
            id="tvm-nper-pv-pmt-zero",
        ),
        pytest.param(
            "convert_rate",
            {
                "rate": 1000,
                "periods_per_year": 1,
                "direction": "nominal_to_effective",
                "compounding": "continuous",
            },
            "too large",
            id="convert-rate-overflow",
        ),
        pytest.param(
            "loan_schedule",
            {"principal": 1e6, "annual_rate": 1e6, "term_months": 600},
            "too large",
            id="loan-schedule-overflow",
        ),
    ],
)
async def test_bad_input_is_a_clear_error(
    default_server: Server, tool: str, args: dict[str, Any], message: str
) -> None:
    text = await _tool_error(default_server, tool, args)
    assert re.search(message, text), f"{tool}({args}): expected /{message}/, got: {text}"


@pytest.mark.parametrize(
    ("cashflows", "expected"),
    [
        # A double root at 0: the scan must find the tangency, not report "no IRR".
        pytest.param([-1, 2, -1], 0.0, id="double-root-at-zero"),
        # NPV is -(1 - 1.15/(1+r))**2: a double root at 15%.
        pytest.param([-1, 2.3, -1.3225], 0.15, id="double-root-at-15pc"),
        # Past the linear root grid's upper end (10.0), into its log-spaced tail.
        pytest.param([-1, 12], 11.0, id="irr-above-1000pc"),
    ],
)
async def test_irr_review_cases(
    default_server: Server, cashflows: list[float], expected: float
) -> None:
    result = await _structured(default_server, "irr", {"cashflows": cashflows})
    # abs, not rel: one expected root is exactly 0, and the tangent scan lands within ~1e-8.
    assert result["irr"] == pytest.approx(expected, abs=1e-6)


async def test_bond_price_accepts_a_negative_yield_above_the_per_period_floor(
    default_server: Server,
) -> None:
    """ytm=-1.5 at frequency 2 is -75% per period: extreme but defined, so it must price."""
    args = {"face": 100, "coupon_rate": 0.05, "years_to_maturity": 10, "ytm": -1.5}
    result = await _structured(default_server, "bond_price", {**args, "frequency": 2})
    assert result["price"] > 0
