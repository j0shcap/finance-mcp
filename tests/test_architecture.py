"""Provider specifics stay in their adapter: the logic layer and tools import none of them.

data/service.py and everything above it know providers only through providers/ports.py.
yfinance, its transport and pandas belong to providers/yahoo.py; server.py, the composition
root, is the one other module allowed to name the Yahoo adapter, and tools and prompts reach
data only through DataService. Relative imports are banned (ruff TID252), so every import
this scans is absolute.
"""

import ast
from pathlib import Path

import pytest

import finance_mcp

PACKAGE = Path(finance_mcp.__file__).parent
PROVIDER_ONLY = ("yfinance", "curl_cffi", "pandas", "finance_mcp.data.providers.yahoo")
ALLOWED = {
    "data/providers/yahoo.py": PROVIDER_ONLY,
    "server.py": ("finance_mcp.data.providers.yahoo",),
}
#: Layers above DataService, which must not reach for a provider or its ports directly.
ABOVE_THE_SERVICE = ("tools/", "prompts/")


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            # `from finance_mcp.data.providers import yahoo` names the module as an alias.
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _is_within(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: p.relative_to(PACKAGE).as_posix()
)
def test_only_the_adapter_imports_provider_specifics(path: Path) -> None:
    relative = path.relative_to(PACKAGE).as_posix()
    forbidden = [package for package in PROVIDER_ONLY if package not in ALLOWED.get(relative, ())]
    if relative.startswith(ABOVE_THE_SERVICE):
        forbidden.append("finance_mcp.data.providers")
    leaks = sorted(
        module for module in _imports(path) for package in forbidden if _is_within(module, package)
    )
    assert leaks == [], f"{relative} imports provider specifics: {leaks}"


def test_the_scan_sees_an_aliased_module_import(tmp_path: Path) -> None:
    source = tmp_path / "leak.py"
    source.write_text("from finance_mcp.data.providers import yahoo\n", encoding="utf-8")
    assert "finance_mcp.data.providers.yahoo" in _imports(source)
