"""Turn a tool call's argument-validation failure into one message the model can act on.

FastMCP validates arguments with pydantic and forwards pydantic's own text: a header naming
an internal validator (``call[get_quote]``), ``[type=..., input_value=..., input_type=...]``
diagnostics and a documentation URL per error. This middleware replaces that with
``Invalid arguments for get_quote: tickers: List should have at least 1 item ...``.

Only argument errors are rewritten. FastMCP 3.0-3.3 raise pydantic's ValidationError
itself; 3.4 wraps it in ``fastmcp.exceptions.ValidationError`` with pydantic's error as
the cause. Either way the pydantic error's title is ``call[<tool name>]``, the title of the
validator FastMCP builds around the tool function. A pydantic error raised inside a tool
body carries its model's title instead: that is a server bug, so it is left for FastMCP's
error handling rather than reported as the caller's mistake.
"""

from typing import Any

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from pydantic import ValidationError
from pydantic_core import ErrorDetails

#: Longest offending value quoted back in a pattern error; a longer one is cut short.
MAX_QUOTED_VALUE = 20


class ArgumentErrorMiddleware(Middleware):
    """Rewrite argument-validation failures as a ToolError with a clean message."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        try:
            return await call_next(context)
        except Exception as exc:
            name = context.message.name
            invalid = _argument_error(exc, name)
            if invalid is None:
                raise
            raise ToolError(describe_argument_error(name, invalid)) from exc


def _argument_error(exc: BaseException, tool_name: str) -> ValidationError | None:
    """The pydantic error behind ``exc`` if it rejected ``tool_name``'s arguments."""
    for candidate in (exc, exc.__cause__):
        if isinstance(candidate, ValidationError) and candidate.title == f"call[{tool_name}]":
            return candidate
    return None


def describe_argument_error(tool_name: str, error: ValidationError) -> str:
    """One line per failed argument, joined: ``Invalid arguments for <tool>: <loc>: <msg>``."""
    parts = [_describe(detail) for detail in error.errors(include_url=False)]
    return f"Invalid arguments for {tool_name}: " + "; ".join(parts) + "."


def _describe(detail: ErrorDetails) -> str:
    location = _location(detail["loc"])
    if detail["type"] == "string_pattern_mismatch":
        # pydantic's message is the raw regex; the parameter's description says the format
        # in words, so name the value and point there instead.
        message = (
            f"{_quote(detail['input'])} is not in the expected format; "
            "see the parameter's description"
        )
    else:
        message = detail["msg"].rstrip(".")
    return f"{location}: {message}" if location else message


def _location(loc: tuple[int | str, ...]) -> str:
    """("cashflows", 0, "date") -> "cashflows[0].date"."""
    text = ""
    for part in loc:
        if isinstance(part, int):
            text += f"[{part}]"
        else:
            text += f".{part}" if text else part
    return text


def _quote(value: Any) -> str:
    text = str(value)
    if len(text) > MAX_QUOTED_VALUE:
        text = text[:MAX_QUOTED_VALUE] + "..."
    return repr(text)
