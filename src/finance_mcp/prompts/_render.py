"""Helpers shared by the prompt modules: placeholder filling, the closing disclaimer, and
parsing a free-text ticker list into the exact list a tool call needs."""

import re

from fastmcp.exceptions import PromptError

from finance_mcp.conventions import CONVENTIONS_URI
from finance_mcp.tools._inputs import MAX_COMPARE_TICKERS, TICKER_PATTERN

DISCLAIMER = (
    "Disclaimer: This is quantitative analysis for research purposes, not investment advice. "
    "Always do your own due diligence."
)

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")
_TICKER_SEPARATORS = re.compile(r"[,;\s]+")


def fill(template: str, **values: str) -> str:
    """Replace each ``{name}`` placeholder in ``template`` with ``values[name]``.

    Not ``str.format``: methodology prose may contain literal braces, and only
    ``{lowercase_identifier}`` tokens are placeholders. The substitution is a single pass,
    so a user-supplied value that happens to contain ``{name}`` is inserted verbatim and
    never expanded again. The placeholder set must match ``values`` exactly - a typo on
    either side would otherwise drop an argument without a trace.
    """
    names = set(_PLACEHOLDER.findall(template))
    if missing := names - values.keys():
        raise ValueError(f"template placeholders missing a value: {sorted(missing)}")
    if unused := values.keys() - names:
        raise ValueError(f"values with no placeholder in the template (unused): {sorted(unused)}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def render(template: str, **values: str) -> str:
    """``fill`` with the two values every prompt shares: the conventions resource URI and
    the closing disclaimer. Since ``fill`` rejects unused values, a template that omits
    ``{conventions_uri}`` or ``{disclaimer}`` fails to render, so every prompt points at the
    conventions resource and ends with the disclaimer by construction.
    """
    return fill(template, conventions_uri=CONVENTIONS_URI, disclaimer=DISCLAIMER, **values)


def parse_tickers(raw: str) -> list[str]:
    """Split a free-text ticker list ("aapl, msft googl") into distinct upper-case symbols.

    Raises PromptError (which reaches the client despite error masking) when a symbol is
    malformed or the count is outside what compare_tickers accepts, so the user fixes the
    argument instead of the model discovering the problem one tool call later.
    """
    tickers: list[str] = []
    for token in _TICKER_SEPARATORS.split(raw.strip()):
        if not token:
            continue
        if not re.fullmatch(TICKER_PATTERN, token):
            raise PromptError(f"{token!r} is not a valid ticker symbol.")
        symbol = token.upper()
        if symbol not in tickers:
            tickers.append(symbol)
    if len(tickers) < 2:
        raise PromptError(f"compare_stocks needs at least 2 distinct tickers; got {tickers}.")
    if len(tickers) > MAX_COMPARE_TICKERS:
        raise PromptError(
            f"compare_stocks takes at most {MAX_COMPARE_TICKERS} tickers; got {len(tickers)}."
        )
    return tickers
