"""Harness for the end-to-end suite: the built wheel, driven over stdio as a client runs it.

Everything else in tests/ talks to `create_server()` in-process, which cannot see a broken
entry point, a module missing from the wheel, a dependency pyproject forgot, or a stray print
corrupting the JSON-RPC stream on stdout. So this suite builds the wheel, installs it into a
fresh venv, launches the server the ways users do and speaks MCP to it over a real pipe.

Nothing here imports finance_mcp for the code under test: the server only ever runs from
the installed wheel. The in-repo package is imported solely for the tool-name registry the
assertions are checked against, which is data, not behaviour.

The wheel and venv are built once per session; every async test and fixture here shares
the session event loop, so one server subprocess serves a whole module rather than paying
the import cost of pandas and yfinance per test.
"""

import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from dataclasses import dataclass
from email.parser import HeaderParser
from pathlib import Path

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.messages import MessageHandler
from fastmcp.client.transports import StdioTransport

_E2E_DIR = Path(__file__).parent
REPO_ROOT = _E2E_DIR.parent.parent

#: Point this at a prebuilt wheel to test that artifact instead of building one.
WHEEL_ENV_VAR = "FINANCE_MCP_E2E_WHEEL"
#: The exact version the wheel must carry. The release workflow sets it to the tag; unset,
#: the wheel's own metadata is the expectation (the version comes from git).
EXPECT_VERSION_ENV_VAR = "FINANCE_MCP_E2E_EXPECT_VERSION"

#: Generous: the first launch of a fresh venv compiles bytecode for pandas and yfinance.
INIT_TIMEOUT_SECONDS = 120.0
REQUEST_TIMEOUT_SECONDS = 60.0


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the tests in this directory `e2e` and run them on the session event loop.

    Path-filtered for the reason tests/live/conftest.py spells out: pytest hands this hook
    every item in the session, and marking all of them would deselect the offline suite.
    """
    session_loop = pytest.mark.asyncio(loop_scope="session")
    for item in items:
        if item.path.is_relative_to(_E2E_DIR):
            item.add_marker(pytest.mark.e2e)
            if pytest_asyncio.is_async_test(item):
                item.add_marker(session_loop, append=False)


def _require(tool: str) -> str:
    """The absolute path of a CLI tool. A missing one fails the run: skipping would let the
    suite pass having launched nothing."""
    path = shutil.which(tool)
    if path is None:
        pytest.fail(f"`{tool}` must be on PATH to run the e2e suite (install uv)")
    return path


def _run(*argv: str, cwd: Path = REPO_ROOT) -> None:
    """Run a build/install step, surfacing its output only when it fails."""
    result = subprocess.run(  # fixed argv, no shell, no user input
        argv, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        pytest.fail(f"{' '.join(argv)} failed ({result.returncode}):\n{result.stderr}")


@pytest.fixture(scope="session")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The wheel under test: freshly built, never whatever is lying around in dist/.

    A stale dist/*.whl from an earlier version would otherwise pass for this one; the
    version assertion in the protocol tests ties the server to the artifact's metadata.
    """
    override = os.environ.get(WHEEL_ENV_VAR)
    if override:
        return Path(override).resolve()
    out = tmp_path_factory.mktemp("dist")
    _run(_require("uv"), "build", "--wheel", "--out-dir", str(out))
    (built,) = out.glob("mcp_finance-*.whl")
    return built


@pytest.fixture(scope="session")
def project_version(wheel: Path) -> str:
    """The version the wheel under test carries, which the server must report."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        version = HeaderParser().parsestr(archive.read(metadata).decode())["Version"]
    expected = os.environ.get(EXPECT_VERSION_ENV_VAR)
    if expected and version != expected:
        pytest.fail(f"the wheel is version {version}, but {expected} was expected")
    return str(version)


@pytest.fixture(scope="session")
def venv(wheel: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fresh venv holding only the wheel and what its metadata pulls in.

    Built on the interpreter running the tests, so the CI matrix exercises each version.
    """
    path = tmp_path_factory.mktemp("venv")
    uv = _require("uv")
    _run(uv, "venv", "--python", sys.executable, str(path))
    _run(uv, "pip", "install", "--python", str(path / "bin" / "python"), str(wheel))
    return path


@dataclass(frozen=True)
class Launcher:
    """One way a user starts the server: an MCP client config's command + args."""

    name: str
    command: str
    args: tuple[str, ...]


LAUNCHERS = ("console-script", "python-m", "uvx")


def make_launcher(name: str, wheel: Path, venv: Path) -> Launcher:
    """The argv for each supported way of starting the server."""
    if name == "console-script":
        return Launcher(name, str(venv / "bin" / "mcp-finance"), ())
    if name == "python-m":
        return Launcher(name, str(venv / "bin" / "python"), ("-m", "finance_mcp"))
    if name == "uvx":
        # --python pins the interpreter so uvx cannot pick a different one from a
        # .python-version it finds on the way up from its working directory.
        args = ("--python", sys.executable, "--from", str(wheel), "mcp-finance")
        return Launcher(name, _require("uvx"), args)
    raise ValueError(f"unknown launcher {name!r}")


class _ProtocolErrors(MessageHandler):
    """Records what the client could not parse off the server's stdout.

    The SDK's stdio reader does not fail on a line that is not JSON-RPC - it hands the
    parse error to the message handler and reads on - so a stray print() in the server
    would otherwise pass unnoticed here and then break a stricter client.
    """

    def __init__(self) -> None:
        super().__init__()
        self.errors: list[Exception] = []

    async def on_exception(self, message: Exception) -> None:
        self.errors.append(message)


@dataclass
class Server:
    """A connected client plus what the clean-start assertions need to see."""

    client: Client[StdioTransport]
    launcher: Launcher
    stderr_log: Path
    protocol: _ProtocolErrors

    def stderr(self) -> str:
        return self.stderr_log.read_text()


class Launch:
    """Starts, and caches for the session, one server subprocess per launcher."""

    def __init__(self, wheel: Path, venv: Path, tmp: pytest.TempPathFactory) -> None:
        self._wheel, self._venv, self._tmp = wheel, venv, tmp
        self._servers: dict[str, Server] = {}
        self.stack = AsyncExitStack()

    async def __call__(self, name: str, extra_env: dict[str, str] | None = None) -> Server:
        extra = extra_env or {}
        key = f"{name}{sorted(extra.items())}"
        if key not in self._servers:
            launcher = make_launcher(name, self._wheel, self._venv)
            workdir = self._tmp.mktemp(f"cwd-{name}")
            stderr_log = workdir / "stderr.log"
            transport = StdioTransport(
                command=launcher.command,
                args=list(launcher.args),
                # Only uv's own settings (cache dir, index) on top of the minimal environment
                # the MCP SDK passes a server; nothing else from the test process leaks in.
                env={**{k: v for k, v in os.environ.items() if k.startswith("UV_")}, **extra},
                # A neutral working directory, as a client launching from anywhere would use:
                # nothing may depend on being started inside the repo.
                cwd=str(workdir),
                log_file=stderr_log,
            )
            protocol = _ProtocolErrors()
            client = Client(
                transport,
                timeout=REQUEST_TIMEOUT_SECONDS,
                init_timeout=INIT_TIMEOUT_SECONDS,
                message_handler=protocol,
            )
            await self.stack.enter_async_context(client)
            self._servers[key] = Server(client, launcher, stderr_log, protocol)
        return self._servers[key]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def launch(
    wheel: Path, venv: Path, tmp_path_factory: pytest.TempPathFactory
) -> AsyncIterator[Launch]:
    """The session's launcher cache; closing it shuts every server down."""
    launch = Launch(wheel, venv, tmp_path_factory)
    async with launch.stack:
        yield launch


@pytest_asyncio.fixture(loop_scope="session", params=LAUNCHERS)
async def server(request: pytest.FixtureRequest, launch: Launch) -> Server:
    """The server started by each launcher in turn."""
    return await launch(request.param)


@pytest_asyncio.fixture(loop_scope="session")
async def default_server(launch: Launch) -> Server:
    """The server as most clients start it: the installed console script.

    Behaviour past the handshake does not depend on the launcher, so the per-call tests run
    once here instead of three times.
    """
    return await launch("console-script")


#: A proxy on the discard port, which nothing listens on: every request fails to connect
#: at once, so a network outage can be reproduced offline and deterministically.
UNREACHABLE_PROXY = "http://127.0.0.1:9"


@pytest_asyncio.fixture(loop_scope="session")
async def offline_server(launch: Launch) -> Server:
    """The installed server with its network cut: every Yahoo request fails to connect."""
    proxy = {var: UNREACHABLE_PROXY for var in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")}
    return await launch("console-script", {**proxy, **{k.lower(): v for k, v in proxy.items()}})
