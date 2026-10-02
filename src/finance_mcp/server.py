"""Server assembly: build the FastMCP instance and register every tool/prompt module.

Also the composition root: the one place that chooses which provider serves each port.
"""

from fastmcp import FastMCP

from finance_mcp import __version__, conventions
from finance_mcp.data.providers.yahoo import YahooProvider
from finance_mcp.data.service import DataService
from finance_mcp.prompts import analysis, calculations
from finance_mcp.settings import get_settings
from finance_mcp.tools import analytics, calculators, equities
from finance_mcp.tools._argument_errors import ArgumentErrorMiddleware


def build_data_service() -> DataService:
    """The DataService over Yahoo, configured from settings/env."""
    s = get_settings()
    # One instance serves every port, so its request gate bounds all Yahoo calls together.
    yahoo = YahooProvider(
        max_concurrent_requests=s.max_concurrent_requests,
        request_retries=s.request_retries,
    )
    return DataService(
        market=yahoo,
        news=yahoo,
        earnings=yahoo,
        quote_ttl=float(s.quote_cache_ttl_seconds),
        history_ttl=float(s.history_cache_ttl_seconds),
        fundamentals_ttl=float(s.fundamentals_cache_ttl_seconds),
        max_bars=s.max_history_bars,
    )


def create_server(service: DataService | None = None) -> FastMCP:
    """Create and configure the finance-mcp FastMCP server."""
    mcp: FastMCP = FastMCP(
        "finance-mcp",
        instructions=conventions.SERVER_INSTRUCTIONS,
        version=__version__,
        # Domain errors already reach the model as ToolError (see tools/_dispatch.py),
        # which masking preserves. Masking covers everything else: an unexpected
        # provider/transport failure must not put a traceback's internals, URLs, or
        # credentials in front of the model.
        mask_error_details=True,
    )
    mcp.add_middleware(ArgumentErrorMiddleware())
    if service is None:
        service = build_data_service()
    conventions.register(mcp)
    calculators.register(mcp)
    equities.register(mcp, service)
    analytics.register(mcp, service)
    analysis.register(mcp)
    calculations.register(mcp)
    return mcp


def main() -> None:
    """Console entry point: run the server over stdio.

    Without fastmcp's banner, which would print on every launch and check PyPI for a
    fastmcp release this package may not support.
    """
    create_server().run(show_banner=False)


if __name__ == "__main__":
    main()
