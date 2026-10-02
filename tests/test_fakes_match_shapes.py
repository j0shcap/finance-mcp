"""The test fakes must look like what yfinance really returns (tests/shapes/yahoo.json).

The unit suite parses data from tests/fakes.py. Holding those builders to recorded shapes is
what makes that suite a test of the parsers against real payloads rather than imagined ones.
When Yahoo, yfinance or pandas changes a shape, the nightly job's `record_shapes --check`
fails; re-record with `make record-shapes` and these tests say which fakes to update.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.yahoo_shapes import (
    exception_shape,
    frame_shape,
    keys_read,
    mapping_shape,
    series_shape,
    value_type,
)

from tests.fakes import (
    QUOTE_FI,
    FakeHTTPError,
    make_earnings_summary,
    make_financials_df,
    make_history_df,
    make_intraday_df,
    make_news_item,
    make_recommendations_df,
    make_search_news_item,
    make_series,
)

RECORDED: dict[str, Any] = json.loads(
    (Path(__file__).parent / "shapes" / "yahoo.json").read_text(encoding="utf-8")
)


def _within(fake: dict[str, list[str]], recorded: dict[str, list[str]]) -> list[str]:
    """Keys whose fake value types were never seen from Yahoo."""
    return [
        f"{key}: fake {types}, Yahoo {recorded.get(key)}"
        for key, types in fake.items()
        if not set(types) <= set(recorded.get(key, []))
    ]


def test_daily_history_fake_matches() -> None:
    assert frame_shape(make_history_df([100.0, 101.0])) == RECORDED["history_daily"]


def test_intraday_history_fake_matches() -> None:
    assert frame_shape(make_intraday_df([100.0, 101.0])) == RECORDED["history_intraday"]


@pytest.mark.parametrize("recorded", ["statement_annual", "statement_quarterly"])
def test_statement_fake_matches(recorded: str) -> None:
    fake = make_financials_df({"Total Revenue": [1.0, 2.0]}, ["2024-09-30", "2023-09-30"])
    assert frame_shape(fake, named_columns=False) == RECORDED[recorded]


@pytest.mark.parametrize("recorded", ["dividends", "splits"])
def test_corporate_action_fake_matches(recorded: str) -> None:
    fake = series_shape(make_series(["2024-02-09"], [0.24]))
    real = RECORDED[recorded]
    assert {k: v for k, v in fake.items() if k != "times_of_day"} == {
        k: v for k, v in real.items() if k != "times_of_day"
    }
    assert set(fake["times_of_day"]) <= set(real.get("times_of_day", []))


def test_recommendations_fake_matches() -> None:
    fake = make_recommendations_df([("0m", 10, 20, 5, 1, 0)])
    assert frame_shape(fake) == RECORDED["recommendations"]


def test_quote_fake_has_yahoo_types() -> None:
    fake = {name: [value_type(value)] for name, value in QUOTE_FI.items()}
    assert _within(fake, RECORDED["fast_info"]) == []


def test_news_fakes_have_yahoo_types() -> None:
    item = make_news_item("Title", "Reuters", "https://x/a", "2026-09-30T00:00:00Z", "Summary.")
    content = item["content"]
    nested = {
        "provider.displayName": [value_type(content["provider"]["displayName"])],
        "canonicalUrl.url": [value_type(content["canonicalUrl"]["url"])],
    }
    fake_content = mapping_shape([content], [k for k in content if k in RECORDED["news_content"]])
    assert _within({**fake_content, **nested}, RECORDED["news_content"]) == []
    search_item = make_search_news_item("Title", "Reuters", "https://x/a", 1790647283)
    keys = [k for k in search_item if k in RECORDED["search_news_item"]]
    assert _within(mapping_shape([search_item], keys), RECORDED["search_news_item"]) == []


@pytest.mark.parametrize("call", ["fast_info.last_price", "history"])
def test_unknown_symbol_fake_raises_what_yahoo_raises(call: str) -> None:
    real = RECORDED["unknown_symbol"][call]
    fake = exception_shape(FakeHTTPError(404))
    assert fake["status_code"] == real["status_code"]
    assert real["class"].endswith("HTTPError")


def test_earnings_fake_has_yahoo_types() -> None:
    keys = keys_read()
    summary = make_earnings_summary()
    trend = summary["earningsTrend"]["trend"]
    eps = [item["earningsEstimate"] for item in trend]
    revenue = [item["revenueEstimate"] for item in trend]
    parts = {
        "earnings_modules": ([summary], "modules"),
        "earnings_quote_type": ([summary["quoteType"]], "quote_type"),
        "earnings_calendar": ([summary["calendarEvents"]], "calendar"),
        "earnings_next": ([summary["calendarEvents"]["earnings"]], "calendar_earnings"),
        "earnings_trend_item": (trend, "trend_item"),
        "earnings_estimate": (eps + revenue, "estimate"),
        "earnings_eps_estimate": (eps, "eps_estimate"),
        "earnings_revenue_estimate": (revenue, "revenue_estimate"),
        "earnings_quarter": (summary["earningsHistory"]["history"], "earnings_quarter"),
    }
    mismatches = {
        recorded: _within(mapping_shape(items, keys[receiver]), RECORDED[recorded])
        for recorded, (items, receiver) in parts.items()
    }
    assert {name: found for name, found in mismatches.items() if found} == {}
