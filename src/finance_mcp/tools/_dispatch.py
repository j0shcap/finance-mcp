"""Helpers that run a data/calculator call and translate domain errors to ToolError.

These take a thunk rather than wrapping the tool function, so each ``@mcp.tool`` keeps
the annotated signature FastMCP builds its schema from.
"""

import asyncio
from collections.abc import Callable

from fastmcp.exceptions import ToolError

from finance_mcp.data.errors import DataUnavailable, InvalidInput


async def run_data[T](call: Callable[[], T]) -> T:
    """Run a blocking data-layer ``call`` off the event loop.

    Translates DataUnavailable (and its SymbolNotFound subclass) and InvalidInput into a
    ToolError whose message is surfaced to the model. InvalidInput comes from data-layer
    checks that span several arguments, which no Field bound can express.
    """
    try:
        return await asyncio.to_thread(call)
    except (DataUnavailable, InvalidInput) as exc:
        raise ToolError(str(exc)) from exc


def run_calc[T](call: Callable[[], T]) -> T:
    """Run a pure calculator ``call``, translating input errors into a ToolError.

    InvalidInput carries a message written for the model, so it is forwarded verbatim.
    Arithmetic failures on extreme inputs that no Field bound screens out (overflow, a
    factor underflowing to zero, a result model rejecting the value -- pydantic's
    ValidationError is a ValueError) are reported as out of range rather than as a bare
    "(34, 'Result too large')".
    """
    try:
        return call()
    except InvalidInput as exc:
        raise ToolError(str(exc)) from exc
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        raise ToolError(f"The inputs are out of range for this calculation: {exc}") from exc
