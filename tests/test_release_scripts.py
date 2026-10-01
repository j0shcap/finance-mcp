"""scripts/check_pypi_release.py: PyPI must serve exactly the files the release uploaded."""

import hashlib
import json
from pathlib import Path

import pytest
from scripts.check_pypi_release import main, mismatches


def _dist(tmp_path: Path, files: dict[str, bytes]) -> Path:
    dist = tmp_path / "dist"
    dist.mkdir()
    for name, data in files.items():
        (dist / name).write_bytes(data)
    return dist


def _release(files: dict[str, bytes]) -> dict[str, object]:
    return {
        "urls": [
            {"filename": name, "digests": {"sha256": hashlib.sha256(data).hexdigest()}}
            for name, data in files.items()
        ]
    }


FILES = {"mcp_finance-0.5.0-py3-none-any.whl": b"wheel", "mcp_finance-0.5.0.tar.gz": b"sdist"}


def test_identical_files_match(tmp_path: Path) -> None:
    assert mismatches(_release(FILES), _dist(tmp_path, FILES)) == []


def test_different_bytes_name_the_file(tmp_path: Path) -> None:
    served = {**FILES, "mcp_finance-0.5.0.tar.gz": b"other"}
    [problem] = mismatches(_release(served), _dist(tmp_path, FILES))
    assert problem.startswith("mcp_finance-0.5.0.tar.gz: PyPI serves sha256")


def test_missing_and_extra_files_are_reported(tmp_path: Path) -> None:
    served = {"mcp_finance-0.5.0-py3-none-any.whl": b"wheel", "stray.whl": b"x"}
    problems = mismatches(_release(served), _dist(tmp_path, FILES))
    assert "mcp_finance-0.5.0.tar.gz: built but not on PyPI" in problems
    assert "stray.whl: on PyPI but not built here" in problems


def test_the_cli_exits_non_zero_on_a_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    release = tmp_path / "release.json"
    release.write_text(json.dumps(_release({**FILES, "mcp_finance-0.5.0.tar.gz": b"other"})))
    dist = _dist(tmp_path, FILES)
    assert main([str(release), str(dist)]) == 1
    assert "PyPI serves sha256" in capsys.readouterr().err
    release.write_text(json.dumps(_release(FILES)))
    assert main([str(release), str(dist)]) == 0
