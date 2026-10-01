"""Unit tests for the shared prompt-rendering helpers (no server needed)."""

import pytest
from fastmcp.exceptions import PromptError

from finance_mcp.conventions import CONVENTIONS_URI
from finance_mcp.prompts._render import DISCLAIMER, fill, parse_tickers, render
from finance_mcp.tools._inputs import MAX_COMPARE_TICKERS


def test_fill_substitutes_every_placeholder() -> None:
    assert fill("{a} and {b}", a="x", b="y") == "x and y"


def test_fill_leaves_literal_braces_that_are_not_placeholders_alone() -> None:
    # Methodology prose may contain braces (a set, a JSON example); only {identifier}
    # tokens are placeholders.
    assert fill("{a} {1, 2} {}", a="x") == "x {1, 2} {}"


def test_fill_does_not_re_expand_placeholders_inside_substituted_values() -> None:
    """User text is data: a value containing '{b}' must not be substituted again."""
    assert fill("{a} {b}", a="{b}", b="y") == "{b} y"


def test_fill_rejects_a_placeholder_with_no_value() -> None:
    with pytest.raises(ValueError, match="missing"):
        fill("{a} {b}", a="x")


def test_fill_rejects_a_value_with_no_placeholder() -> None:
    """A typo in a placeholder name would otherwise drop the argument silently."""
    with pytest.raises(ValueError, match="unused"):
        fill("{a}", a="x", b="y")


def test_render_supplies_the_conventions_uri_and_disclaimer() -> None:
    text = render("see {conventions_uri}; {x}. End with exactly: {disclaimer}", x="1")
    assert text == f"see {CONVENTIONS_URI}; 1. End with exactly: {DISCLAIMER}"


def test_render_requires_every_prompt_to_point_at_conventions_and_end_with_the_disclaimer() -> None:
    """Omitting either is a hard error rather than a review comment."""
    with pytest.raises(ValueError, match="unused"):
        render("{x} with no conventions pointer. {disclaimer}", x="1")
    with pytest.raises(ValueError, match="unused"):
        render("{x} {conventions_uri} with no disclaimer", x="1")


def test_disclaimer_wording() -> None:
    assert DISCLAIMER == (
        "Disclaimer: This is quantitative analysis for research purposes, not investment "
        "advice. Always do your own due diligence."
    )


def test_parse_tickers_splits_on_commas_and_whitespace_and_upper_cases() -> None:
    assert parse_tickers("aapl, msft  googl") == ["AAPL", "MSFT", "GOOGL"]
    assert parse_tickers(", aapl,,msft, ") == ["AAPL", "MSFT"]  # stray separators


def test_parse_tickers_accepts_yahoo_prefixes_and_suffixes() -> None:
    assert parse_tickers("BRK-B;RY.TO,^GSPC") == ["BRK-B", "RY.TO", "^GSPC"]


def test_parse_tickers_drops_duplicates_keeping_first_order() -> None:
    assert parse_tickers("MSFT, aapl, msft, AAPL") == ["MSFT", "AAPL"]


def test_parse_tickers_needs_at_least_two_distinct_symbols() -> None:
    with pytest.raises(PromptError, match="at least 2"):
        parse_tickers("AAPL, aapl")


def test_parse_tickers_caps_the_count_at_the_compare_tickers_bound() -> None:
    too_many = ",".join(f"T{i}" for i in range(MAX_COMPARE_TICKERS + 1))
    with pytest.raises(PromptError, match=f"at most {MAX_COMPARE_TICKERS}"):
        parse_tickers(too_many)


def test_parse_tickers_rejects_a_malformed_symbol_by_name() -> None:
    with pytest.raises(PromptError, match="AAPL!"):
        parse_tickers("MSFT, AAPL!")
