"""Pins the configuration and completeness of the opt-in live suite.

`tests/live/` hits real Yahoo endpoints. What keeps it out of `make check` (and therefore
out of CI, every contributor's machine, and the coverage gate) is:

1. the `live` marker is registered, so `--strict-markers` cannot silently swallow a typo;
2. `addopts` carries `-m "not live"`, so the marker is deselected unless asked for;
3. the marker is applied only to items under `tests/live/`.

Each is one edit away from being lost. Losing (1) or (2) turns the offline quality gate
into a network-dependent one that fails on a plane or behind a firewall; losing (3) empties
it entirely while leaving the config looking correct. So the first two are asserted against
the loaded config and the third against a real collection pass.

This module also pins that the live suite covers every market-data tool, since a tool
added without one is a tool whose schema drift is invisible again.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from finance_mcp.conventions import MARKET_DATA_TOOLS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _collect(*args: str) -> str:
    """Node ids a real collection pass selects, so the assertions below are not theoretical.

    --collect-only, so this never executes a test and never reaches the network.
    """
    result = subprocess.run(  # fixed argv, no shell, no user input
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def test_default_run_keeps_the_offline_suite_and_drops_the_live_one() -> None:
    """The whole point of the marker, asserted against an actual selection.

    The config checks below cannot catch the failure this one does: a
    pytest_collection_modifyitems hook in tests/live/conftest.py is handed EVERY collected
    item in the session, so marking without a path filter marks the entire offline suite
    live, and `-m 'not live'` then deselects all of it. That leaves `make check` reporting
    success having run nothing at all, with the config still looking perfect.
    """
    collected = _collect()

    assert "tests/test_server.py::" in collected, (
        "the offline suite vanished from the default run - most likely the live marker is "
        "being applied to items outside tests/live/"
    )
    assert "tests/live/" not in collected, "live tests must not be selected by default"


def test_live_run_selects_only_the_live_suite() -> None:
    """`make test-live` must reach the live tests and nothing else."""
    collected = _collect("-m", "live")

    assert "tests/live/" in collected, "`-m live` must select the live suite"
    assert "tests/test_server.py::" not in collected, (
        "`-m live` must not drag the offline suite along"
    )


def test_live_marker_is_registered(pytestconfig: pytest.Config) -> None:
    """`live` is a declared marker, not an ad-hoc one `--strict-markers` would reject."""
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("live:") for m in markers), (
        f"the 'live' marker must be registered in [tool.pytest.ini_options] markers, got {markers}"
    )


def test_live_tests_are_deselected_by_default(pytestconfig: pytest.Config) -> None:
    """The default run excludes the live suite, so `make check` never touches the network."""
    addopts = pytestconfig.getini("addopts")
    assert "-m" in addopts, f"addopts must pass a -m expression, got {addopts}"
    assert "not live" in addopts[addopts.index("-m") + 1], (
        f"addopts must deselect the live marker by default, got {addopts}"
    )


def test_strict_markers_is_enabled(pytestconfig: pytest.Config) -> None:
    """With --strict-markers, a misspelled marker errors instead of quietly running."""
    assert "--strict-markers" in pytestconfig.getini("addopts")


def test_every_market_data_tool_has_live_coverage() -> None:
    """Each market-data tool is exercised by the live suite.

    The live suite is the only thing watching Yahoo's payload shape, so a tool added
    without a live test is a tool whose schema drift is invisible again. Checked against
    the same MARKET_DATA_TOOLS tuple the server instructions render from, so adding a tool
    there without covering it here fails in the offline gate rather than going unnoticed
    until the nightly run.
    """
    sources = "\n".join(
        path.read_text() for path in sorted((REPO_ROOT / "tests" / "live").glob("test_*.py"))
    )

    uncovered = [tool for tool in MARKET_DATA_TOOLS if f'"{tool}"' not in sources]
    assert not uncovered, (
        f"market-data tools with no live contract test: {uncovered}. Add one to tests/live/ "
        f"so Yahoo changing their payload shape is still caught."
    )
