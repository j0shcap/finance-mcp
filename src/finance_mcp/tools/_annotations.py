"""ToolAnnotations builders for the two tool families this server exposes.

Every tool here is read-only, so the families differ only in the two hints a client
actually acts on: whether repeating a call is guaranteed to give the same answer, and
whether the call reaches outside the process.
"""

from mcp.types import ToolAnnotations


def calculator(title: str) -> ToolAnnotations:
    """Annotations for a pure calculator: same inputs, same answer, no network."""
    return ToolAnnotations(
        title=title,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )


def market_data(title: str) -> ToolAnnotations:
    """Annotations for a Yahoo-backed lookup: read-only, but live and external.

    Deliberately not idempotent: a second call can return a different price, a new
    headline, or a newly filed statement.
    """
    return ToolAnnotations(
        title=title,
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    )
