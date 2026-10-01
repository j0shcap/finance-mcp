"""Install a published mcp-finance by name, the way users run it, and use it once.

    uv run --no-project --with mcp scripts/smoke_release.py 0.5.0

Launches ``uvx --refresh --from mcp-finance==X.Y.Z mcp-finance`` over stdio, checks the
server reports that version and lists its tools, and calls one calculator (no network
needed beyond the install). ``--from-spec`` launches something else instead, e.g. a local
wheel, to rehearse the script.

Exit status: 0 when it all checks out, 1 when the installed release is wrong, and 75
(EX_TEMPFAIL) when it could not be installed or launched at all, which is worth retrying
while the package index catches up.
"""

import argparse
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def smoke(version: str, from_spec: str) -> list[str]:
    server = StdioServerParameters(
        command="uvx", args=["--refresh", "--from", from_spec, "mcp-finance"]
    )
    problems: list[str] = []
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        if init.server_info.version != version:
            problems.append(f"the server reports {init.server_info.version}, not {version}")
        tools = {tool.name for tool in (await session.list_tools()).tools}
        if "loan_schedule" not in tools:
            problems.append(f"loan_schedule is not among the {len(tools)} tools listed")
        result = await session.call_tool(
            "loan_schedule", {"principal": 400000, "annual_rate": 0.065, "term_months": 360}
        )
        payment = json.loads("".join(getattr(c, "text", "") for c in result.content))
        if result.is_error or payment.get("monthly_payment") != 2528.27:
            problems.append(f"loan_schedule returned {payment}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("version")
    parser.add_argument("--from-spec", help="what uvx installs (default: mcp-finance==VERSION)")
    args = parser.parse_args()
    try:
        problems = asyncio.run(
            smoke(args.version, args.from_spec or f"mcp-finance=={args.version}")
        )
    except Exception as exc:  # the install or launch failed; nothing was checked
        print(f"could not install or launch mcp-finance {args.version}: {exc!r}", file=sys.stderr)
        return 75
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if not problems:
        print(f"mcp-finance {args.version}: installs, reports its version, and answers a call.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
