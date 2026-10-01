"""Describe the SHAPE of what yfinance returns - never its values.

A shape is the part of a payload that parsing depends on: index kinds, timezones and units,
column dtypes, which keys are present and the type of their values, which exception an
unknown symbol raises. tests/shapes/yahoo.json records the real shapes
(scripts/record_shapes.py); tests/test_fakes_match_shapes.py holds the test fakes to them, and
the nightly job compares fresh shapes with the committed ones.

The keys worth describing are read off src/finance_mcp/data/yahoo.py itself: every
``<receiver>.get("key")`` and ``getattr(fi, "attr")`` there, so the recording follows the parser.
"""

import ast
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

YAHOO_PY = Path(__file__).resolve().parent.parent / "src" / "finance_mcp" / "data" / "yahoo.py"


def keys_read(source: str | None = None) -> dict[str, list[str]]:
    """Receiver name -> the literal keys yahoo.py reads from it, sorted.

    ``info.get("trailingPE")`` lands under "info", ``content.get("title")`` under "content",
    and so on. The fast_info attributes a quote reads are listed in ``_FAST_INFO_FIELDS``
    and land under "fi".
    """
    tree = ast.parse(source if source is not None else YAHOO_PY.read_text(encoding="utf-8"))
    found: defaultdict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        if _is_fast_info_fields(node):
            assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
            found["fi"].update(
                elt.value
                for elt in node.value.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            )
            continue
        if not isinstance(node, ast.Call) or not node.args:
            continue
        key = node.args[1] if _is_getattr(node) else node.args[0]
        receiver = node.args[0] if _is_getattr(node) else _get_receiver(node)
        if (
            isinstance(receiver, ast.Name)
            and isinstance(key, ast.Constant)
            and isinstance(key.value, str)
        ):
            found[receiver.id].add(key.value)
    return {receiver: sorted(keys) for receiver, keys in sorted(found.items())}


def _is_fast_info_fields(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_FAST_INFO_FIELDS" for t in node.targets)
        and isinstance(node.value, ast.Tuple)
    )


def _is_getattr(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2


def _get_receiver(node: ast.Call) -> ast.expr | None:
    if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        return node.func.value
    return None


def value_type(value: Any) -> str:
    """A JSON-stable type name. int and float are one "number": yahoo.py reads both as float."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float):
        return "nan" if isinstance(value, float) and math.isnan(value) else "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def mapping_shape(items: list[dict[str, Any]], keys: list[str]) -> dict[str, list[str]]:
    """Each key's value types across ``items`` ("missing" where an item lacks the key).

    Across every item, not the first: payloads mix instruments and optional fields, and a
    shape taken from one item would change with whichever item came first.
    """
    return {
        key: sorted({value_type(item[key]) if key in item else "missing" for item in items})
        for key in keys
    }


def attribute_shape(
    obj: Any, names: list[str], read: Callable[[Any, str], Any] = getattr
) -> dict[str, list[str]]:
    """Each attribute's value type; the exception class name if reading it raises."""
    shape = {}
    for name in names:
        try:
            shape[name] = [value_type(read(obj, name))]
        except Exception as exc:  # a lazy attribute that fails is itself part of the shape
            shape[name] = [f"raises {type(exc).__name__}"]
    return shape


def merge(old: Any, new: Any) -> Any:
    """Combine two recordings: lists (sets of observations) are unioned, the rest replaced."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {key: merge(old[key], new[key]) if key in old else new[key] for key in new}
    if isinstance(old, list) and isinstance(new, list):
        return sorted(set(old) | set(new))
    return new


def drift(committed: Any, fresh: Any, path: str = "") -> list[str]:
    """What ``fresh`` shows that ``committed`` never recorded.

    A list is a set of observations, so seeing fewer of them is not drift - today's sample
    may simply lack an optional field - but a type never seen before is. Everything else
    (dtypes, index kinds, exception classes) must match exactly.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        problems = [f"{path}{key}: no longer reported" for key in committed.keys() - fresh]
        problems += [f"{path}{key}: new" for key in fresh.keys() - committed]
        for key in committed.keys() & fresh:
            problems += drift(committed[key], fresh[key], f"{path}{key}.")
        return sorted(problems)
    if isinstance(committed, list) and isinstance(fresh, list):
        unseen = sorted(set(fresh) - set(committed))
        return [f"{path.rstrip('.')}: newly seen {unseen}"] if unseen else []
    if committed != fresh:
        return [f"{path.rstrip('.')}: {committed!r} -> {fresh!r}"]
    return []


def index_shape(index: pd.Index) -> dict[str, Any]:
    shape: dict[str, Any] = {"class": type(index).__name__, "dtype": _dtype_name(index.dtype)}
    if isinstance(index, pd.DatetimeIndex):
        shape["tz_aware"] = index.tz is not None
        shape["unit"] = index.unit
    return shape


def frame_shape(frame: pd.DataFrame, *, named_columns: bool = True) -> dict[str, Any]:
    """Index and column structure plus dtypes; per-column dtypes only for fixed column names."""
    shape: dict[str, Any] = {
        "index": index_shape(frame.index),
        "columns": index_shape(frame.columns),
    }
    if named_columns:
        shape["dtypes"] = {str(col): _dtype_name(dtype) for col, dtype in frame.dtypes.items()}
    else:
        shape["value_dtypes"] = sorted({_dtype_name(dtype) for dtype in frame.dtypes})
    return shape


def series_shape(series: pd.Series) -> dict[str, Any]:
    shape: dict[str, Any] = {"index": index_shape(series.index), "dtype": str(series.dtype)}
    if isinstance(series.index, pd.DatetimeIndex) and len(series):
        shape["times_of_day"] = sorted({ts.strftime("%H:%M") for ts in series.index})
    return shape


def exception_shape(exc: BaseException) -> dict[str, Any]:
    response = getattr(exc, "response", None)
    return {
        "class": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "status_code": getattr(response, "status_code", None),
    }


def _dtype_name(dtype: Any) -> str:
    """The dtype, with any timezone replaced by "<tz>": the zone is the exchange's, not ours."""
    tz = getattr(dtype, "tz", None)
    return str(dtype).replace(str(tz), "<tz>") if tz is not None else str(dtype)
