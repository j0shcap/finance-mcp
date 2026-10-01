"""Yahoo-sourced results are rounded to their real precision on output, and only on output."""

import inspect
import json
import math

import pytest
from fastmcp import Client
from pydantic import BaseModel

from finance_mcp.data import models
from finance_mcp.data.models import (
    FinancialStatement,
    MarketData,
    PriceBar,
    round_significant,
)
from finance_mcp.server import create_server
from tests.fakes import QUOTE_FI, fake_ticker_factory, make_client, make_history_df

# Yahoo's prices are float32 values widened to float64, so the digits past the 7th are
# conversion noise: float32(316.98) is 316.9800109863281.
NOISY_PRICE = 316.9800109863281

#: Results of the pure calculators. They keep full precision: callers check them
#: against Excel and textbook answers to many places.
CALCULATOR_MODELS = {
    "AmortizationRow",
    "BondAnalytics",
    "BondDatedAnalytics",
    "BondDatedYTM",
    "BondYTM",
    "DatedCashflow",
    "IRRResult",
    "LoanSchedule",
    "MIRRResult",
    "NPVResult",
    "RateConversionResult",
    "TVMResult",
}


def _all_models() -> dict[str, type[BaseModel]]:
    return {
        name: cls
        for name, cls in inspect.getmembers(models, inspect.isclass)
        if issubclass(cls, BaseModel) and cls.__module__ == models.__name__
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (NOISY_PRICE, 316.98),
        (7651.5400390625, 7651.54),
        (1.133529782295227, 1.13353),  # FX rates keep their pips
        (0.09812236577272415, 0.09812237),  # sub-1 values keep 7 significant digits
        (-0.8758833482631699, -0.8758833),
        (1.2e-9, 1.2e-9),
        (49875295.0, 49875295.0),  # integer digits are never rounded away
        (1065966441922.2999, 1065966441922.0),
        (0.0, 0.0),
    ],
)
def test_round_significant(value: float, expected: float) -> None:
    assert round_significant(value) == expected


@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_round_significant_passes_infinities_through(value: float) -> None:
    assert round_significant(value) == value


def test_round_significant_passes_nan_through() -> None:
    assert math.isnan(round_significant(math.nan))


def test_serialized_output_is_rounded_but_attributes_are_not() -> None:
    bar = PriceBar(
        date="2026-09-01",
        open=NOISY_PRICE,
        high=NOISY_PRICE,
        low=NOISY_PRICE,
        close=NOISY_PRICE,
        volume=53167400.0,
    )
    assert json.loads(bar.model_dump_json())["close"] == 316.98
    assert bar.model_dump()["close"] == 316.98
    # Analytics read attributes, so they compute on the unrounded value.
    assert bar.close == NOISY_PRICE


def test_rounding_reaches_floats_nested_in_dicts_and_lists() -> None:
    statement = FinancialStatement(
        symbol="AAPL",
        statement="income",
        period="annual",
        period_ends=["2025-09-30", "2024-09-30"],
        line_items={"Diluted EPS": [7.4600000381469727, None]},
    )
    assert statement.model_dump()["line_items"] == {"Diluted EPS": [7.46, None]}


def test_every_market_data_model_rounds_and_no_calculator_model_does() -> None:
    all_models = _all_models()
    assert all_models.keys() >= CALCULATOR_MODELS
    for name, cls in all_models.items():
        if name == "MarketData":
            continue
        assert issubclass(cls, MarketData) is (name not in CALCULATOR_MODELS), name


def test_rounding_leaves_every_published_schema_unchanged() -> None:
    # MCP clients read a tool's output schema from the serialization schema; the
    # rounding serializer must not loosen any field's declared type.
    for name, cls in _all_models().items():
        serialization = cls.model_json_schema(mode="serialization")
        assert serialization == cls.model_json_schema(mode="validation"), name


async def test_tool_results_are_rounded_over_the_protocol() -> None:
    noisy_quote = {**QUOTE_FI, "last_price": NOISY_PRICE, "previous_close": 314.1499938964844}
    factory = fake_ticker_factory(
        fast_info=noisy_quote, history_df=make_history_df([NOISY_PRICE, 330.1401062011719])
    )
    async with Client(create_server(yf_client=make_client(factory=factory))) as client:
        quote = await client.call_tool("get_quote", {"tickers": ["AAPL"]})
        history = await client.call_tool("get_price_history", {"ticker": "AAPL"})

    quote_text = json.loads(quote.content[0].text)
    assert quote_text["quotes"][0]["price"] == 316.98
    assert quote_text["quotes"][0]["change"] == 2.830017  # 2.8300170898437...
    assert quote.structured_content is not None
    assert quote.structured_content["quotes"][0]["price"] == 316.98

    bars = json.loads(history.content[0].text)["bars"]
    assert [bar["close"] for bar in bars] == [316.98, 330.1401]


async def test_calculator_results_keep_full_precision() -> None:
    async with Client(create_server()) as client:
        result = await client.call_tool(
            "bond_price", {"face": 1000, "coupon_rate": 0.05, "years_to_maturity": 10, "ytm": 0.06}
        )
    price = json.loads(result.content[0].text)["price"]
    assert price == pytest.approx(925.6126256977221, rel=1e-15)
    assert price != round_significant(price)
