"""Shared pytest fixtures. Fakes and data builders live in tests/fakes.py."""

import os
from collections.abc import AsyncIterator

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from hypothesis import settings

from finance_mcp.server import create_server

# Derandomized so `make check` explores the same examples on every run and machine: a
# property failure is then a reproducible regression, never a flake that goes away on retry.
# No deadline, because the root-finders' run time varies with the input by design.
# HYPOTHESIS_PROFILE=explore searches afresh, and harder, when hunting for new failures.
settings.register_profile("default", derandomize=True, max_examples=200, deadline=None)
settings.register_profile("explore", max_examples=5_000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture
async def client() -> AsyncIterator[Client[FastMCPTransport]]:
    """An in-memory MCP client connected to a fresh finance-mcp server."""
    async with Client(create_server()) as connected:
        yield connected
