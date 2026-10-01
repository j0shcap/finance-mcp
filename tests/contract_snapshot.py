"""Collect the model-facing contract over the protocol and compare it with a committed snapshot.

Everything is read through an MCP client and dumped the way the server puts it on the wire
(``model_dump(by_alias=True, mode="json", exclude_none=True)``, as ``mcp.server.runner``
does), whole: a field the protocol or fastmcp adds later shows up in the diff instead of
slipping past an allow-list. Only the keys named in this module are dropped.
"""

import difflib
import json
import re
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport
from mcp.types import TextContent, TextResourceContents
from pydantic import BaseModel

from tests.prompt_samples import SAMPLE_ARGS

SNAPSHOT_ROOT = Path(__file__).parent / "snapshots" / "contract"
#: Only these files belong to the snapshot; anything else in the directory is left alone.
SNAPSHOT_SUFFIXES = (".json", ".md")
#: Longest diff shown per changed file before it is cut short.
MAX_DIFF_LINES = 80
#: Names become file names, so they must be unique and stay inside the snapshot root.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


async def collect_contract(client: Client[FastMCPTransport]) -> dict[str, str]:
    """Every snapshot file, keyed by its path under the snapshot root, with its exact text."""
    server_info = client.server_info
    assert server_info is not None
    # The version is left out: it changes every release and steers nothing.
    files = {
        "server.md": f"# {server_info.name}\n\n{client.instructions or ''}",
        "server.json": _dumps(
            {key: value for key, value in _wire(server_info).items() if key != "version"}
        ),
    }

    tools = await client.list_tools()
    prompts = await client.list_prompts()
    resources = await client.list_resources()
    for kind, names in (
        ("tool", [tool.name for tool in tools]),
        ("prompt", [prompt.name for prompt in prompts]),
        ("resource", [resource.name for resource in resources]),
    ):
        _check_names(kind, names)

    for tool in tools:
        files[f"tools/{tool.name}.json"] = _dumps(_wire(tool))
        # The JSON is the faithful record; a verbatim copy of the description diffs readably.
        files[f"tools/{tool.name}.md"] = tool.description or ""

    for prompt in prompts:
        if prompt.name not in SAMPLE_ARGS:
            raise AssertionError(
                f"Prompt {prompt.name!r} has no sample arguments: add them to "
                "SAMPLE_ARGS in tests/prompt_samples.py, then run `make snapshot`."
            )
        arguments = SAMPLE_ARGS[prompt.name]
        rendered = await client.get_prompt(prompt.name, arguments)
        texts = []
        for message in rendered.messages:
            assert isinstance(message.content, TextContent), "only text prompts are snapshotted"
            texts.append(message.content.text)
        # From the rendered result only the description and messages: its _meta carries the
        # server version in newer protocol eras.
        files[f"prompts/{prompt.name}.json"] = _dumps(
            {
                **_wire(prompt),
                "sample_arguments": arguments,
                "rendered_description": rendered.description,
                "rendered_roles": [message.role for message in rendered.messages],
            }
        )
        files[f"prompts/{prompt.name}.md"] = "\n\n---\n\n".join(texts)

    for resource in resources:
        files[f"resources/{resource.name}.json"] = _dumps(_wire(resource))
        contents = await client.read_resource(resource.uri)
        texts = []
        for content in contents:
            assert isinstance(content, TextResourceContents), "only text resources are snapshotted"
            texts.append(content.text)
        files[f"resources/{resource.name}.md"] = "".join(texts)
    templates = await client.list_resource_templates()
    files["resource_templates.json"] = _dumps([_wire(template) for template in templates])

    files["index.json"] = _dumps(
        {
            "tools": [tool.name for tool in tools],
            "prompts": [prompt.name for prompt in prompts],
            "resources": [resource.name for resource in resources],
        }
    )
    return files


def compare(root: Path, expected: dict[str, str]) -> list[str]:
    """One message per missing, changed or stale snapshot file; empty when they all match."""
    problems = []
    for relative, text in sorted(expected.items()):
        path = root / relative
        if not path.exists():
            problems.append(f"{relative}: missing from the snapshot")
            continue
        actual = path.read_text(encoding="utf-8", newline="")
        if actual != text:
            problems.append(f"{relative}: changed\n{_diff(relative, actual, text)}")
    problems.extend(
        f"{relative}: stale (nothing produces it any more)"
        for relative in sorted(_snapshot_files(root) - expected.keys())
    )
    return problems


def update(root: Path, expected: dict[str, str]) -> tuple[int, int]:
    """Write the expected files and delete stale ones; returns (written, deleted).

    Every ``.json``/``.md`` file under ``root`` belongs to the snapshot, so one that no longer
    corresponds to the contract is deleted, along with directories left empty.
    """
    written = 0
    for relative, text in expected.items():
        path = root / relative
        if path.exists() and path.read_text(encoding="utf-8", newline="") == text:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        written += 1
    stale = _snapshot_files(root) - expected.keys()
    for relative in stale:
        (root / relative).unlink()
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return written, len(stale)


def _check_names(kind: str, names: list[str]) -> None:
    unsafe = [name for name in names if not _SAFE_NAME.match(name)]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not unsafe, f"{kind} names unusable as snapshot file names: {unsafe}"
    assert not duplicates, f"duplicate {kind} names would overwrite each other: {duplicates}"


def _wire(model: BaseModel) -> dict[str, Any]:
    """The object as the server sends it, minus fastmcp's own bookkeeping."""
    data: dict[str, Any] = model.model_dump(by_alias=True, mode="json", exclude_none=True)
    meta = data.pop("_meta", None)
    if isinstance(meta, dict):
        # _meta.fastmcp holds fastmcp's internal tags; any other _meta key is kept.
        meta = {key: value for key, value in meta.items() if key != "fastmcp"}
        if meta:
            data["_meta"] = meta
    return data


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _snapshot_files(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SNAPSHOT_SUFFIXES
    }


def _diff(relative: str, actual: str, expected: str) -> str:
    lines = list(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"snapshot/{relative}",
            tofile=f"server/{relative}",
        )
    )
    if len(lines) > MAX_DIFF_LINES:
        lines = [*lines[:MAX_DIFF_LINES], f"... {len(lines) - MAX_DIFF_LINES} more diff lines\n"]
    return "".join(lines)
