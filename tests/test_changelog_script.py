"""scripts/changelog.py: release notes come from, and releases are cut in, CHANGELOG.md."""

import re
from pathlib import Path

import pytest
from scripts.changelog import (
    ChangelogError,
    cut_release,
    main,
    release_notes,
    released_versions,
)

BASE = "https://github.com/j0shcap/finance-mcp"

CHANGELOG = f"""# Changelog

Intro prose.

## [Unreleased]

### Changed
- Requires fastmcp 4.x.

## [0.4.2] - 2026-10-01

0.4.1 was never published; its changes ship here.

### Breaking changes
- The default risk-free rate is the T-bill.

### Fixed
- A bug.

## [0.4.0] - 2026-10-01

### Added
- A tool.

[Unreleased]: {BASE}/compare/v0.4.2...HEAD
[0.4.2]: {BASE}/compare/v0.4.0...v0.4.2
[0.4.0]: {BASE}/compare/v0.3.0...v0.4.0
"""


def test_notes_are_the_section_body_verbatim() -> None:
    assert release_notes(CHANGELOG, "0.4.2") == (
        "0.4.1 was never published; its changes ship here.\n"
        "\n"
        "### Breaking changes\n"
        "- The default risk-free rate is the T-bill.\n"
        "\n"
        "### Fixed\n"
        "- A bug.\n"
    )


def test_notes_for_the_last_section_stop_before_the_link_references() -> None:
    assert release_notes(CHANGELOG, "0.4.0") == "### Added\n- A tool.\n"


@pytest.mark.parametrize("version", ["9.9.9", "Unreleased"])
def test_notes_for_a_version_without_a_released_section_fail(version: str) -> None:
    with pytest.raises(ChangelogError, match=re.escape(version)):
        release_notes(CHANGELOG, version)


def test_notes_for_an_empty_section_fail() -> None:
    empty = CHANGELOG.replace("### Added\n- A tool.\n", "")
    with pytest.raises(ChangelogError, match="empty"):
        release_notes(empty, "0.4.0")


def test_cutting_a_release_moves_unreleased_under_the_version() -> None:
    cut = cut_release(CHANGELOG, "0.5.0", "2026-10-02")
    assert "## [Unreleased]\n\n## [0.5.0] - 2026-10-02\n\n### Changed\n" in cut
    assert release_notes(cut, "0.5.0") == "### Changed\n- Requires fastmcp 4.x.\n"
    assert release_notes(cut, "0.4.2") == release_notes(CHANGELOG, "0.4.2")  # untouched


def test_cutting_a_release_updates_the_compare_links() -> None:
    cut = cut_release(CHANGELOG, "0.5.0", "2026-10-02")
    assert cut.endswith(
        f"[Unreleased]: {BASE}/compare/v0.5.0...HEAD\n"
        f"[0.5.0]: {BASE}/compare/v0.4.2...v0.5.0\n"
        f"[0.4.2]: {BASE}/compare/v0.4.0...v0.4.2\n"
        f"[0.4.0]: {BASE}/compare/v0.3.0...v0.4.0\n"
    )


def test_cutting_a_release_leaves_everything_else_unchanged() -> None:
    cut = cut_release(CHANGELOG, "0.5.0", "2026-10-02")
    expected = CHANGELOG.replace(
        "## [Unreleased]\n\n### Changed",
        "## [Unreleased]\n\n## [0.5.0] - 2026-10-02\n\n### Changed",
    ).replace(
        "compare/v0.4.2...HEAD\n",
        f"compare/v0.5.0...HEAD\n[0.5.0]: {BASE}/compare/v0.4.2...v0.5.0\n",
    )
    assert cut == expected


def test_an_empty_unreleased_section_cannot_be_released() -> None:
    once = cut_release(CHANGELOG, "0.5.0", "2026-10-02")
    with pytest.raises(ChangelogError, match="Unreleased"):
        cut_release(once, "0.5.1", "2026-10-03")


def test_an_existing_version_cannot_be_released_again() -> None:
    with pytest.raises(ChangelogError, match=re.escape("0.4.2")):
        cut_release(CHANGELOG, "0.4.2", "2026-10-02")


@pytest.mark.parametrize("version", ["v0.5.0", "0.5", "0.5.0rc1", "0.5.0.dev1"])
def test_only_plain_semantic_versions_are_accepted(version: str) -> None:
    with pytest.raises(ChangelogError, match=re.escape("X.Y.Z")):
        cut_release(CHANGELOG, version, "2026-10-02")


def test_every_released_section_of_the_real_changelog_has_notes() -> None:
    real = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = released_versions(real)
    assert "0.4.2" in versions
    for version in versions:
        assert release_notes(real, version).strip()


def test_the_cli_prints_notes_and_cuts_a_release(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    assert main(["notes", "0.4.0", "--file", str(path)]) == 0
    assert capsys.readouterr().out == "### Added\n- A tool.\n"
    assert main(["release", "0.5.0", "--date", "2026-10-02", "--file", str(path)]) == 0
    assert "## [0.5.0] - 2026-10-02" in path.read_text(encoding="utf-8")


def test_the_cli_reports_an_error_and_exits_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    assert main(["notes", "9.9.9", "--file", str(path)]) == 1
    assert "no released section for 9.9.9" in capsys.readouterr().err
