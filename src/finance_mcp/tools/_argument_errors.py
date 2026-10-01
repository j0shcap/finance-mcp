"""Turn a tool call's argument-validation failure into one message the model can act on.

FastMCP validates arguments with pydantic and forwards pydantic's own text: a header naming
an internal validator (``call[get_quote]``), ``[type=..., input_value=..., input_type=...]``
diagnostics and a documentation URL per error. This middleware replaces that with
``Invalid arguments for get_quote: tickers: List should have at least 1 item ...``.

Only argument errors are rewritten. FastMCP wraps them in
``fastmcp.exceptions.ValidationError`` with pydantic's error as the cause, and that error's
title is ``call[<tool name>]``, the title of the validator FastMCP builds around the tool
function. A pydantic error raised inside a tool body carries its model's title instead: that
is a server bug, not the caller's mistake. FastMCP 4 re-raises it as it is, and the MCP SDK
then answers with a JSON-RPC "Invalid request parameters" error, past error masking. So it
is re-raised here as the tool error masking gives any other unexpected failure.
"""

from typing import Any

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from fastmcp.utilities.logging import get_logger
from pydantic import ValidationError
from pydantic_core import ErrorDetails

logger = get_logger(__name__)

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
            if invalid is not None:
                raise ToolError(describe_argument_error(name, invalid)) from exc
            if isinstance(exc, ValidationError):
                # FastMCP's own masked wording and log line. Its masking runs inside
                # call_next, before this point, so the error has to leave here already a
                # tool error; and it logs this one only as an argument warning, so the
                # traceback naming the rejecting model is logged here.
                logger.exception(f"Error calling tool {name!r}")
                raise ToolError(f"Error calling tool {name!r}") from exc
            raise


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
