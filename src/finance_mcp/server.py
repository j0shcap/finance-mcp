"""Server assembly: build the FastMCP instance and register every tool/prompt module."""

from fastmcp import FastMCP

from finance_mcp import __version__, conventions
from finance_mcp.data.yfinance_client import YFinanceClient
from finance_mcp.prompts import analysis
from finance_mcp.settings import get_settings
from finance_mcp.tools import analytics, calculators, equities


def build_default_client() -> YFinanceClient:
    """Build a YFinanceClient with cache TTLs sourced from settings/env."""
    s = get_settings()
    return YFinanceClient(
        quote_ttl=float(s.quote_cache_ttl_seconds),
        history_ttl=float(s.history_cache_ttl_seconds),
        fundamentals_ttl=float(s.fundamentals_cache_ttl_seconds),
        max_bars=s.max_history_bars,
    )


def create_server(yf_client: YFinanceClient | None = None) -> FastMCP:
    """Create and configure the finance-mcp FastMCP server."""
    mcp: FastMCP = FastMCP(
        "finance-mcp",
        instructions=conventions.SERVER_INSTRUCTIONS,
        version=__version__,
        # Domain errors already reach the model as ToolError (see tools/_dispatch.py),
        # which masking preserves. Masking covers everything else: an unexpected
        # yfinance/transport failure must not put a traceback's internals, URLs, or
        # credentials in front of the model.
        mask_error_details=True,
    )
    client = yf_client if yf_client is not None else build_default_client()
    conventions.register(mcp)
    calculators.register(mcp)
    equities.register(mcp, client)
    analytics.register(mcp, client)
    analysis.register(mcp)
    return mcp


def main() -> None:
    """Console entry point: run the server over stdio."""
    create_server().run()


if __name__ == "__main__":
    main()
