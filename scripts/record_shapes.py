"""Record the shapes of real yfinance payloads into tests/shapes/yahoo.json (no values).

    uv run python -m scripts.record_shapes          # rewrite tests/shapes/yahoo.json
    uv run python -m scripts.record_shapes --check  # exit 1, with a diff, if Yahoo drifted

About fifteen live calls cover every kind of payload src/finance_mcp/data/providers/yahoo.py
parses. The nightly live job runs --check, so a change in what Yahoo, yfinance or pandas
return shows up as a readable diff, and the test fakes - held to this file by
tests/test_fakes_match_shapes.py - follow it once it is re-recorded.
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yfinance as yf

from finance_mcp.data.providers import yahoo
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

SHAPES = Path(__file__).resolve().parent.parent / "tests" / "shapes" / "yahoo.json"
UNKNOWN = "NOTATICKER.XX"
#: Nested reads written as ``(content.get("provider") or {}).get("displayName")``, which the
#: key scan cannot attribute to a receiver.
NESTED_NEWS_KEYS = {"provider": "displayName", "canonicalUrl": "url", "clickThroughUrl": "url"}


def record() -> dict[str, Any]:
    keys = keys_read()
    yf.config.debug.hide_exceptions = False  # as YahooProvider sets it
    apple = yf.Ticker("AAPL")
    news = _fetch(lambda: apple.get_news(count=5, tab="news"))
    search = _fetch(lambda: yf.Search("apple", max_results=3, news_count=3, lists_count=0))
    recommendations = _fetch(lambda: apple.recommendations)
    dividends, splits = _fetch(lambda: yahoo.corporate_actions(apple))
    shapes: dict[str, Any] = {
        "fast_info": attribute_shape(
            apple.fast_info, keys["fi"], lambda obj, name: _fetch(lambda: getattr(obj, name))
        ),
        "history_daily": frame_shape(
            _fetch(lambda: apple.history(period="1mo", interval="1d", auto_adjust=True))
        ),
        "history_intraday": frame_shape(
            _fetch(
                lambda: yf.Ticker("BTC-USD").history(period="5d", interval="1h", auto_adjust=True)
            )
        ),
        "history_index": frame_shape(
            _fetch(lambda: yf.Ticker("^IRX").history(period="1mo", interval="1d", auto_adjust=True))
        ),
        "statement_annual": frame_shape(_fetch(lambda: apple.income_stmt), named_columns=False),
        "statement_quarterly": frame_shape(
            _fetch(lambda: apple.quarterly_balance_sheet), named_columns=False
        ),
        "dividends": series_shape(dividends),
        "splits": series_shape(splits),
        "recommendations": frame_shape(recommendations),
        "recommendation_row": mapping_shape(recommendations.to_dict("records"), keys["row"]),
        "info_equity": mapping_shape([_fetch(lambda: apple.info)], keys["info"]),
        "info_cross_listing": mapping_shape([_fetch(lambda: yf.Ticker("SAP").info)], keys["info"]),
        "info_etf": mapping_shape([_fetch(lambda: yf.Ticker("SPY").info)], keys["info"]),
        "news_item": mapping_shape(news, keys["item"]),
        "news_content": _news_content_shape([item["content"] for item in news], keys["content"]),
        "search_quote": mapping_shape(search.quotes, keys["q"]),
        "search_news_item": mapping_shape(search.news, keys["item"]),
        "unknown_symbol": {
            "fast_info.last_price": _raised(lambda: yf.Ticker(UNKNOWN).fast_info.last_price),
            "history": _raised(lambda: yf.Ticker(UNKNOWN).history(period="1mo")),
        },
    }
    shapes["_recorded_with"] = {pkg: version(pkg) for pkg in ("yfinance", "pandas")}
    return shapes


def _news_content_shape(contents: list[dict[str, Any]], keys: list[str]) -> dict[str, list[str]]:
    shape = mapping_shape(contents, keys)
    for outer, inner in NESTED_NEWS_KEYS.items():
        shape[f"{outer}.{inner}"] = sorted(
            {
                value_type(content[outer].get(inner))
                if isinstance(content.get(outer), dict)
                else "missing"
                for content in contents
            }
        )
    return shape


#: Backoff between attempts when Yahoo is slow, unreachable or throttling.
RETRY_DELAYS = (5.0, 15.0, 45.0)


class Unreachable(Exception):
    """Yahoo kept failing transiently, so this run says nothing about payload shapes."""


def _fetch[T](call: Callable[[], T]) -> T:
    """``call()``, retried on transient failures: those say nothing about payload shapes."""
    for delay in (*RETRY_DELAYS, None):
        try:
            return call()
        except Exception as exc:
            if not is_transient(exc):
                raise
            if delay is None:
                raise Unreachable(f"{type(exc).__name__}: {exc}") from exc
            print(f"transient {type(exc).__name__}; retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


def is_transient(exc: BaseException) -> bool:
    """The server's definition, plus slow read timeouts: a script has no client to keep waiting."""
    return yahoo.is_transient(exc) or type(exc).__name__ in {"Timeout", "ReadTimeout"}


def _raised(call: Callable[[], Any]) -> dict[str, Any]:
    try:
        result = _fetch(call)
    except Unreachable:
        raise
    except Exception as exc:
        return exception_shape(exc)
    return {"no exception": value_type(result) if not hasattr(result, "empty") else "frame"}


def dumps(shapes: dict[str, Any]) -> str:
    return json.dumps(shapes, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare instead of writing")
    args = parser.parse_args()
    try:
        fresh = record()
    except Unreachable as exc:
        # Like the live suite's skips: throttling is not drift, so --check does not fail on it.
        print(f"Yahoo kept failing transiently ({exc}); shapes not checked.", file=sys.stderr)
        return 0 if args.check else 1
    committed = json.loads(SHAPES.read_text(encoding="utf-8")) if SHAPES.exists() else {}
    if not args.check:
        SHAPES.parent.mkdir(parents=True, exist_ok=True)
        SHAPES.write_text(dumps(merge(committed, fresh)), encoding="utf-8")
        print(f"recorded {SHAPES.name}: merged with the committed shapes")
        return 0
    committed.pop("_recorded_with", None)
    fresh.pop("_recorded_with", None)
    problems = drift(committed, fresh)
    if not problems:
        print("Yahoo payload shapes match tests/shapes/yahoo.json.")
        return 0
    for problem in problems:
        print(f"drift: {problem}")
    print(
        "\nYahoo payload shapes drifted. Run `make record-shapes`, fix the fakes and parsers "
        "it affects, and commit both.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
