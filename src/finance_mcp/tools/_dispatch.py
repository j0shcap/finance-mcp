"""Helpers that run a data/calculator call and translate domain errors to ToolError.

These take a thunk rather than wrapping the tool function, so each ``@mcp.tool`` keeps
its full annotated signature and FastMCP's schema introspection is unaffected (a
signature-erasing decorator would break it). One home for the data-layer -> ToolError
translation that otherwise repeats in every tool.
"""

import asyncio
from collections.abc import Callable

from fastmcp.exceptions import ToolError

from finance_mcp.data.errors import DataUnavailable, InvalidInput


async def run_data[T](call: Callable[[], T]) -> T:
    """Run a blocking yfinance-backed ``call`` off the event loop.

    Translates DataUnavailable (and its SymbolNotFound subclass) into a ToolError whose
    message is surfaced to the model.
    """
    try:
        return await asyncio.to_thread(call)
    except DataUnavailable as exc:
        raise ToolError(str(exc)) from exc


def run_calc[T](call: Callable[[], T]) -> T:
    """Run a pure calculator ``call``, translating input errors into a ToolError.

    InvalidInput carries a message written for the model, so it is forwarded verbatim.
    The numeric clause is defence in depth: the calculators validate their preconditions
    explicitly, but an extreme argument that no Field bound can screen (a power that
    overflows, a factor that underflows to zero, a result the model rejects) must still
    reach the client as a clear ToolError rather than a bare "(34, 'Result too large')".
    pydantic's ValidationError subclasses ValueError, so result-model failures are
    covered too.
    """
    try:
        return call()
    except InvalidInput as exc:
        raise ToolError(str(exc)) from exc
    except (ZeroDivisionError, OverflowError, ValueError) as exc:
        raise ToolError(f"The inputs are out of range for this calculation: {exc}") from exc
