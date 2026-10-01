"""Release notes from, and release cuts in, CHANGELOG.md (Keep a Changelog format).

    python scripts/changelog.py notes 0.5.0     # print the body of ## [0.5.0]
    python scripts/changelog.py release 0.5.0   # move [Unreleased] under ## [0.5.0] - today

The changelog is curated prose, so this only moves and slices it: a section's body is
returned verbatim, and cutting a release moves the [Unreleased] entries under a dated
heading and updates the compare links at the bottom of the file.
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)
_LINK = re.compile(r"^\[(?P<version>[^\]]+)\]: (?P<url>\S+)$", re.MULTILINE)


class ChangelogError(Exception):
    """The changelog cannot provide what was asked for."""


def released_versions(text: str) -> list[str]:
    """Every released version with a section, newest first."""
    return [m["version"] for m in _HEADING.finditer(text) if m["version"] != "Unreleased"]


def release_notes(text: str, version: str) -> str:
    """The body of ``## [version]``, verbatim, without surrounding blank lines."""
    if version == "Unreleased" or version not in released_versions(text):
        raise ChangelogError(f"CHANGELOG.md has no released section for {version}")
    body = _section_body(text, version).strip("\n")
    if not body.strip():
        raise ChangelogError(f"the CHANGELOG.md section for {version} is empty")
    return body + "\n"


def cut_release(text: str, version: str, date: str) -> str:
    """Move the [Unreleased] entries under ``## [version] - date`` and update the links."""
    if not _VERSION.match(version):
        raise ChangelogError(f"{version!r} is not a release version: expected X.Y.Z")
    if version in released_versions(text):
        raise ChangelogError(f"CHANGELOG.md already has a section for {version}")
    unreleased = _section_body(text, "Unreleased")
    if not unreleased.strip():
        raise ChangelogError("the [Unreleased] section is empty: nothing to release")
    previous = released_versions(text)[0]
    text = text.replace(
        f"## [Unreleased]{unreleased}",
        f"## [Unreleased]\n\n## [{version}] - {date}\n\n{unreleased.lstrip(chr(10))}",
        1,
    )
    links = {m["version"]: m for m in _LINK.finditer(text)}
    if "Unreleased" not in links:
        raise ChangelogError("CHANGELOG.md has no [Unreleased] compare link")
    old_link = links["Unreleased"].group(0)
    base = links["Unreleased"]["url"].split("/compare/")[0]
    return text.replace(
        old_link,
        f"[Unreleased]: {base}/compare/v{version}...HEAD\n"
        f"[{version}]: {base}/compare/v{previous}...v{version}",
        1,
    )


def _section_body(text: str, version: str) -> str:
    """Everything after ``## [version]...`` up to the next section or the link block."""
    headings = list(_HEADING.finditer(text))
    for index, heading in enumerate(headings):
        if heading["version"] != version:
            continue
        start = text.index("\n", heading.end())
        if index + 1 < len(headings):
            end = headings[index + 1].start()
        else:
            first_link = _LINK.search(text, start)
            end = first_link.start() if first_link else len(text)
        return text[start:end]
    raise ChangelogError(f"CHANGELOG.md has no section for {version}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    notes = commands.add_parser("notes", help="print a released section's body")
    notes.add_argument("version")
    release = commands.add_parser("release", help="move [Unreleased] under a new version")
    release.add_argument("version")
    release.add_argument("--date", default=datetime.date.today().isoformat())
    for command in (notes, release):
        command.add_argument("--file", type=Path, default=CHANGELOG, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path: Path = args.file
    text = path.read_text(encoding="utf-8")
    try:
        if args.command == "notes":
            sys.stdout.write(release_notes(text, args.version))
        else:
            path.write_text(cut_release(text, args.version, args.date), encoding="utf-8")
            print(f"{path.name}: [Unreleased] moved under [{args.version}] - {args.date}")
    except ChangelogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
