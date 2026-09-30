"""Pins the configuration and completeness of the opt-in live and e2e suites.

Three things keep `tests/live/` (and, the same way, `tests/e2e/`) out of `make check`: the
marker is registered, `addopts` deselects it, and the marker is applied only to items under
that directory. Each is one edit away from being lost - the first two would make the offline gate
network-dependent, the third would empty it while leaving the config looking correct. So the
first two are checked against the loaded config and the third against a real collection pass.

Also pins that every market-data tool has a live test, since one added without it is a tool
whose schema drift is invisible again.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from finance_mcp.conventions import MARKET_DATA_TOOLS

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = REPO_ROOT / "tests" / "live"


def _collect(*args: str) -> str:
    """Node ids a real collection pass selects. --collect-only never runs a test."""
    result = subprocess.run(  # fixed argv, no shell, no user input
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def test_default_run_keeps_the_offline_suite_and_drops_the_live_one() -> None:
    """The marker's whole point, asserted against an actual selection.

    The config checks below cannot catch what this one does: the marking hook is handed
    every item in the session, so without a path filter it marks the offline suite live and
    `-m 'not live'` deselects all of it - a gate that passes having run nothing.
    """
    collected = _collect()

    assert "tests/test_server.py::" in collected, (
        "the offline suite vanished from the default run - most likely the live marker is "
        "being applied to items outside tests/live/"
    )
    assert "tests/live/" not in collected, "live tests must not be selected by default"
    assert "tests/e2e/" not in collected, "e2e tests must not be selected by default"


def test_live_run_selects_only_the_live_suite() -> None:
    """`make test-live` reaches the live tests and nothing else."""
    collected = _collect("-m", "live")

    assert "tests/live/" in collected, "`-m live` must select the live suite"
    assert "tests/test_server.py::" not in collected, (
        "`-m live` must not drag the offline suite along"
    )


def test_e2e_run_selects_only_the_offline_e2e_suite() -> None:
    """`make e2e` reaches the stdio suite, minus its Yahoo-backed half, and nothing else."""
    collected = _collect("-m", "e2e and not live")

    assert "tests/e2e/test_stdio_protocol.py::" in collected, (
        "`-m 'e2e and not live'` must select the offline e2e suite"
    )
    assert "tests/e2e/test_stdio_market_data.py::" not in collected, (
        "the market-data e2e tests hit Yahoo, so they belong to `make test-live`"
    )
    assert "tests/test_server.py::" not in collected
    assert "tests/live/" not in collected


def test_live_run_includes_the_market_data_e2e_tests() -> None:
    """The Yahoo-backed stdio tests are live tests too, so the nightly run covers them."""
    assert "tests/e2e/test_stdio_market_data.py::" in _collect("-m", "live")


@pytest.mark.parametrize("marker", ["live", "e2e"])
def test_marker_is_registered(pytestconfig: pytest.Config, marker: str) -> None:
    """A declared marker, not an ad-hoc one `--strict-markers` would reject."""
    markers = pytestconfig.getini("markers")
    assert any(m.startswith(f"{marker}:") for m in markers), (
        f"the {marker!r} marker must be registered in [tool.pytest.ini_options] markers, "
        f"got {markers}"
    )


@pytest.mark.parametrize("marker", ["live", "e2e"])
def test_marker_is_deselected_by_default(pytestconfig: pytest.Config, marker: str) -> None:
    """The default run excludes both opt-in suites, so `make check` stays fast and offline."""
    addopts = pytestconfig.getini("addopts")
    assert "-m" in addopts, f"addopts must pass a -m expression, got {addopts}"
    assert f"not {marker}" in addopts[addopts.index("-m") + 1], (
        f"addopts must deselect the {marker!r} marker by default, got {addopts}"
    )


def test_strict_markers_is_enabled(pytestconfig: pytest.Config) -> None:
    """With --strict-markers, a misspelled marker errors instead of quietly running."""
    assert "--strict-markers" in pytestconfig.getini("addopts")


def _tools_called_by_the_live_suite() -> set[str]:
    """Tool names passed to `layer.call(...)`, read from the AST rather than grepped.

    A substring search over the sources would also match a name in a comment or a message,
    which is how a "covered" tool ends up with no call behind it.
    """
    called: set[str] = set()
    for path in sorted(LIVE_DIR.glob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "call"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                called.add(node.args[0].value)
    return called


def test_every_market_data_tool_has_live_coverage() -> None:
    """A tool with no live test is a tool whose schema drift is invisible.

    Checked against the same MARKET_DATA_TOOLS the server instructions render from, so
    adding one without live coverage fails the offline gate rather than going unnoticed
    until the nightly run.
    """
    uncovered = set(MARKET_DATA_TOOLS) - _tools_called_by_the_live_suite()

    assert not uncovered, (
        f"market-data tools never called by tests/live/: {sorted(uncovered)}. Add a contract "
        f"test so Yahoo changing their payload shape is still caught."
    )
