"""scripts/yahoo_shapes.py: describing payload shapes, merging recordings, spotting drift."""

import math
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.record_shapes import is_transient
from scripts.yahoo_shapes import (
    attribute_shape,
    drift,
    exception_shape,
    frame_shape,
    keys_read,
    mapping_shape,
    merge,
    series_shape,
    value_type,
)
from yfinance.exceptions import YFRateLimitError


def test_keys_read_follows_get_calls_and_getattr_by_receiver() -> None:
    source = (
        "_FAST_INFO_FIELDS = ('last_price',)\n"
        "def f(info, content):\n"
        "    a = info.get('trailingPE')\n"
        "    b = info.get('currency')\n"
        "    d = (content.get('provider') or {}).get('displayName')\n"
        "    e = info.get(dynamic_key)\n"
    )
    assert keys_read(source) == {
        "content": ["provider"],
        "fi": ["last_price"],
        "info": ["currency", "trailingPE"],
    }


def test_keys_read_on_the_real_parser_covers_what_it_reads() -> None:
    keys = keys_read()
    assert {"trailingPE", "financialCurrency", "longName"} <= set(keys["info"])
    assert {"last_price", "previous_close"} <= set(keys["fi"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "null"),
        (True, "bool"),
        (3, "number"),
        (2.5, "number"),
        (math.nan, "nan"),
        ("x", "str"),
        ({}, "object"),
        ([], "list"),
    ],
)
def test_value_type(value: object, expected: str) -> None:
    assert value_type(value) == expected


def test_mapping_shape_unions_types_across_items() -> None:
    items: list[dict[str, object]] = [{"sector": "Tech", "score": 1}, {"score": 2.5}]
    assert mapping_shape(items, ["score", "sector"]) == {
        "score": ["number"],
        "sector": ["missing", "str"],
    }


def test_attribute_shape_records_a_raising_attribute() -> None:
    class Lazy:
        price = 1.0

        @property
        def broken(self) -> float:
            raise KeyError("x")

    assert attribute_shape(Lazy(), ["price", "broken"]) == {
        "price": ["number"],
        "broken": ["raises KeyError"],
    }


def test_frame_shape_hides_the_timezone_but_keeps_awareness_and_unit() -> None:
    index = pd.date_range("2024-01-02", periods=2, freq="D", tz="America/New_York").as_unit("s")
    frame = pd.DataFrame({"Close": [1.0, 2.0], "Volume": [1, 2]}, index=index)
    assert frame_shape(frame) == {
        "index": {
            "class": "DatetimeIndex",
            "dtype": "datetime64[s, <tz>]",
            "tz_aware": True,
            "unit": "s",
        },
        "columns": {"class": "Index", "dtype": frame.columns.dtype.name},
        "dtypes": {"Close": "float64", "Volume": "int64"},
    }


def test_series_shape_records_times_of_day() -> None:
    index = pd.DatetimeIndex(["2024-02-09 09:30", "2024-05-10 09:30"], tz="America/New_York")
    assert series_shape(pd.Series([0.24, 0.25], index=index))["times_of_day"] == ["09:30"]


def test_exception_shape_names_the_class_and_status() -> None:
    exc = RuntimeError("HTTP Error 404")
    exc.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
    assert exception_shape(exc) == {"class": "builtins.RuntimeError", "status_code": 404}


def test_merge_unions_observations_and_replaces_the_rest() -> None:
    old = {"news": {"summary": ["str"]}, "index": {"unit": "ns"}}
    new = {"news": {"summary": ["null"]}, "index": {"unit": "s"}}
    assert merge(old, new) == {"news": {"summary": ["null", "str"]}, "index": {"unit": "s"}}


def test_drift_ignores_fewer_observations_but_reports_new_ones() -> None:
    committed = {"news": {"summary": ["null", "str"]}}
    assert drift(committed, {"news": {"summary": ["str"]}}) == []
    assert drift(committed, {"news": {"summary": ["number"]}}) == [
        "news.summary: newly seen ['number']"
    ]


def test_drift_reports_structural_changes_exactly() -> None:
    committed = {"history": {"unit": "s", "tz_aware": True}, "gone": {"x": ["str"]}}
    fresh = {"history": {"unit": "ns", "tz_aware": True}, "added": {"y": ["str"]}}
    assert drift(committed, fresh) == [
        "added: new",
        "gone: no longer reported",
        "history.unit: 's' -> 'ns'",
    ]


@pytest.mark.parametrize(
    "exc",
    [
        YFRateLimitError(),
        type("Timeout", (Exception,), {})(),
        type("ConnectionError", (Exception,), {})(),
    ],
    ids=["rate-limit", "timeout", "connection"],
)
def test_throttles_timeouts_and_dropped_connections_are_transient(exc: Exception) -> None:
    assert is_transient(exc)


@pytest.mark.parametrize(("status", "transient"), [(503, True), (429, True), (404, False)])
def test_an_http_status_is_transient_only_for_throttling_and_server_errors(
    status: int, transient: bool
) -> None:
    exc = RuntimeError(f"HTTP Error {status}")
    exc.response = SimpleNamespace(status_code=status)  # type: ignore[attr-defined]
    assert is_transient(exc) is transient


def test_a_no_data_error_is_not_transient() -> None:
    assert not is_transient(KeyError("exchangeTimezoneName"))
