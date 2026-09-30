"""Shared pytest fixtures. Fakes and data builders live in tests/fakes.py."""

from collections.abc import AsyncIterator

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from finance_mcp.server import create_server


@pytest.fixture
async def client() -> AsyncIterator[Client[FastMCPTransport]]:
    """An in-memory MCP client connected to a fresh finance-mcp server."""
    async with Client(create_server()) as connected:
        yield connected
