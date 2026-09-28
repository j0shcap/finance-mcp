"""Pins the configuration that keeps the live suite out of the default test run.

`tests/live/` hits real Yahoo endpoints. Two config facts keep it from running in
`make check` (and therefore in CI, on every contributor's machine, and inside the
coverage gate):

1. the `live` marker is registered, so `--strict-markers` cannot silently swallow a typo;
2. `addopts` carries `-m "not live"`, so the marker is deselected unless asked for.

Both are one edit away from being lost, and losing either turns the offline quality gate
into a network-dependent one that fails on a plane or behind a firewall. So they are
asserted here rather than left to a code review to notice.
"""

import pytest


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
