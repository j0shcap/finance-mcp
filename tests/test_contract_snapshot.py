"""The model-facing contract must match the committed snapshot in tests/snapshots/contract/.

What a client receives - instructions, tool and prompt descriptions, schemas, rendered prompt
text and the conventions resource - is what steers the model, so any change to it has to be a
reviewed diff. On a mismatch: run `make snapshot` and review the diff (`git diff --word-diff`
reads best). A fastmcp, mcp or pydantic bump can change it too; that is intended.
"""

import json
import re
from pathlib import Path

import pytest
from fastmcp import Client

from finance_mcp import __version__
from finance_mcp.server import create_server
from tests.contract_snapshot import (
    MAX_DIFF_LINES,
    SNAPSHOT_ROOT,
    collect_contract,
    compare,
    update,
)
from tests.fakes import make_client


async def _contract() -> dict[str, str]:
    async with Client(create_server(service=make_client())) as client:
        return await collect_contract(client)


async def test_contract_matches_the_snapshot(request: pytest.FixtureRequest) -> None:
    expected = await _contract()
    if request.config.getoption("--update-snapshots"):
        written, deleted = update(SNAPSHOT_ROOT, expected)
        pytest.skip(f"snapshot updated: {written} file(s) written, {deleted} stale removed")
    problems = compare(SNAPSHOT_ROOT, expected)
    assert not problems, (
        "The model-facing contract changed. If that is intended (including a fastmcp, mcp "
        "or pydantic bump), run `make snapshot` and review the diff.\n\n" + "\n\n".join(problems)
    )


async def test_contract_does_not_depend_on_the_release() -> None:
    # The server version reaches clients in serverInfo and, in newer protocol eras, in every
    # get_prompt result's _meta. Neither is steering text, and pinning it would fail the
    # snapshot on every release.
    files = await _contract()
    version = re.compile(rf"(?<![\d.]){re.escape(__version__)}(?![\d.])")
    assert not [path for path, text in files.items() if version.search(text)]


async def test_contract_files_are_exactly_what_the_index_lists() -> None:
    files = await _contract()
    index = json.loads(files["index.json"])
    expected = {"server.md", "server.json", "index.json", "resource_templates.json"}
    for kind, folder in (("tools", "tools"), ("prompts", "prompts"), ("resources", "resources")):
        expected |= {
            f"{folder}/{name}{suffix}" for name in index[kind] for suffix in (".json", ".md")
        }
    assert set(files) == expected


# --- the guard's own behaviour ---------------------------------------------------------


async def test_a_prompt_without_sample_arguments_says_what_to_add() -> None:
    server = create_server(service=make_client())

    @server.prompt
    def unsampled() -> str:
        return "text"

    async with Client(server) as client:
        with pytest.raises(AssertionError, match=r"'unsampled'.*tests/prompt_samples.py"):
            await collect_contract(client)


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")


def test_compare_passes_on_an_identical_snapshot(tmp_path: Path) -> None:
    files = {"tools/a.json": '{"x": 1}\n', "server.md": "hi\n"}
    _write(tmp_path, files)
    assert compare(tmp_path, files) == []


def test_compare_names_a_changed_file_with_a_diff(tmp_path: Path) -> None:
    _write(tmp_path, {"tools/a.json": '{"description": "old"}\n'})
    [problem] = compare(tmp_path, {"tools/a.json": '{"description": "new"}\n'})
    assert "tools/a.json" in problem
    assert '-{"description": "old"}' in problem and '+{"description": "new"}' in problem


def test_compare_reports_a_missing_and_a_stale_file(tmp_path: Path) -> None:
    _write(tmp_path, {"tools/gone.json": "{}\n"})
    problems = compare(tmp_path, {"tools/new.json": "{}\n"})
    assert any("tools/new.json" in p and "missing" in p for p in problems)
    assert any("tools/gone.json" in p and "stale" in p for p in problems)


def test_compare_notices_a_trailing_newline_change(tmp_path: Path) -> None:
    _write(tmp_path, {"server.md": "text\n"})
    assert compare(tmp_path, {"server.md": "text"})


def test_compare_ignores_files_that_are_not_snapshots(tmp_path: Path) -> None:
    _write(tmp_path, {"server.md": "hi\n", ".DS_Store": "junk", "tools/notes.txt": "x"})
    assert compare(tmp_path, {"server.md": "hi\n"}) == []


def test_update_writes_overwrites_and_removes_stale_files(tmp_path: Path) -> None:
    _write(tmp_path, {"tools/old.json": "{}\n", "server.md": "before\n", ".DS_Store": "junk"})
    expected = {"server.md": "after\n", "tools/new.json": '{"y": 2}\n'}
    assert update(tmp_path, expected) == (2, 1)
    assert compare(tmp_path, expected) == []
    assert (tmp_path / ".DS_Store").exists()  # not ours to delete
    assert update(tmp_path, expected) == (0, 0)  # idempotent


def test_snapshot_text_round_trips_byte_for_byte(tmp_path: Path) -> None:
    files = {"prompts/p.md": "line one\r\nline two\n€ and ü\n"}
    update(tmp_path, files)
    assert compare(tmp_path, files) == []


def test_compare_cuts_a_long_diff_short(tmp_path: Path) -> None:
    old = "".join(f"line {i}\n" for i in range(200))
    _write(tmp_path, {"prompts/p.md": old})
    [problem] = compare(tmp_path, {"prompts/p.md": old.replace("line", "row")})
    assert "more diff lines" in problem
    assert problem.count("\n") < MAX_DIFF_LINES + 10


def test_update_removes_directories_left_empty(tmp_path: Path) -> None:
    _write(tmp_path, {"resources/gone.json": "{}\n", "server.md": "x\n"})
    update(tmp_path, {"server.md": "x\n"})
    assert not (tmp_path / "resources").exists()


async def test_a_duplicate_resource_name_is_rejected() -> None:
    server = create_server(service=make_client())

    @server.resource("finance://other", name="finance_conventions")
    def other() -> str:
        return "text"

    async with Client(server) as client:
        with pytest.raises(AssertionError, match="duplicate resource names"):
            await collect_contract(client)


async def test_a_binary_resource_is_rejected_rather_than_snapshotted_empty() -> None:
    server = create_server(service=make_client())

    @server.resource("finance://blob", name="blob", mime_type="application/octet-stream")
    def blob() -> bytes:
        return b"\x00\x01"

    async with Client(server) as client:
        with pytest.raises(AssertionError, match="only text resources"):
            await collect_contract(client)
