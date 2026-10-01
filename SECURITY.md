# Security policy

## Supported versions

Security fixes go into the latest release only. Upgrade to the newest `mcp-finance` from PyPI
before reporting.

| Version | Supported |
| --- | --- |
| 0.4.x (latest) | yes |
| older | no |

## Reporting a vulnerability

Please don't open a public issue. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/j0shcap/finance-mcp/security/advisories/new),
with the version, how you run the server, and steps to reproduce. The advisory stays private
until a fixed release is out.

## Scope

finance-mcp is a local stdio server. Every tool is read-only; it stores no credentials and needs
none, and its only network calls go to Yahoo Finance through `yfinance`. Unexpected exceptions
are masked before they reach the client, so tracebacks, URLs and internal details aren't passed
to the model.

In scope: anything that lets tool input reach beyond that, such as code execution, reading or
writing local files, requests to hosts other than Yahoo, or leaking environment or process
details through results or errors. Out of scope: the accuracy of Yahoo's data (see the
disclaimer in the [README](README.md#data-source--disclaimer)), and vulnerabilities in
dependencies that this server doesn't expose, which belong upstream.
