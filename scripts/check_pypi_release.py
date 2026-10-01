"""Check that PyPI serves exactly the files the release workflow uploaded.

    python scripts/check_pypi_release.py release.json dist/

``release.json`` is PyPI's JSON for the release (``/pypi/mcp-finance/X.Y.Z/json``);
``dist/`` holds the wheel and sdist that were verified and uploaded. Exits non-zero, naming
the file, if PyPI is missing one, has an extra one, or serves different bytes.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def mismatches(release: dict[str, Any], dist: Path) -> list[str]:
    """Every difference between PyPI's files for the release and the local ``dist``."""
    served = {entry["filename"]: entry["digests"]["sha256"] for entry in release["urls"]}
    built = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in dist.iterdir()}
    problems = [f"{name}: built but not on PyPI" for name in sorted(built.keys() - served)]
    problems += [f"{name}: on PyPI but not built here" for name in sorted(served.keys() - built)]
    problems += [
        f"{name}: PyPI serves sha256 {served[name]}, the upload was {built[name]}"
        for name in sorted(built.keys() & served.keys())
        if built[name] != served[name]
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    release_json, dist = argv if argv is not None else sys.argv[1:]
    problems = mismatches(json.loads(Path(release_json).read_text()), Path(dist))
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print("PyPI serves exactly the uploaded files.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
