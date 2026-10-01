"""Whether a headline names a company: the text match behind NewsArticle.mentions_company."""

import pytest

from finance_mcp.data.relevance import company_aliases, mentions_company, symbol_aliases


@pytest.mark.parametrize(
    ("long_name", "short_name", "expected"),
    [
        ("Tesla, Inc.", "Tesla, Inc.", ("Tesla",)),
        ("NVIDIA Corporation", "NVIDIA Corporation", ("NVIDIA",)),
        ("The Coca-Cola Company", "Coca-Cola Company (The)", ("Coca-Cola",)),
        ("Apple Inc.", None, ("Apple",)),
        ("JPMorgan Chase & Co.", None, ("JPMorgan Chase", "JPMorgan")),
        (
            "Berkshire Hathaway Inc.",
            "Berkshire Hathaway Inc. New",
            ("Berkshire Hathaway", "Berkshire"),
        ),
        ("Alphabet Inc.", "Alphabet Inc. Class A", ("Alphabet",)),
        ("Amazon.com, Inc.", None, ("Amazon.com", "Amazon")),
        ("Royal Bank of Canada", None, ("Royal Bank of Canada",)),  # generic first word
        ("General Motors Company", None, ("General Motors",)),
        ("3M Company", None, ("3M",)),
        ("AT&T Inc.", None, ("AT&T",)),
        ("Sony Group Corporation", None, ("Sony",)),
        (None, None, ()),
        ("Inc.", None, ()),  # nothing left once the legal form is stripped
    ],
)
def test_company_aliases(
    long_name: str | None, short_name: str | None, expected: tuple[str, ...]
) -> None:
    assert company_aliases(long_name, short_name) == expected


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("TSLA", ("TSLA",)),
        ("BRK-B", ("BRK-B", "BRK.B", "BRK")),
        ("RY.TO", ("RY.TO", "RY")),
        ("F", ("F",)),
    ],
)
def test_symbol_aliases(symbol: str, expected: tuple[str, ...]) -> None:
    assert symbol_aliases(symbol) == expected


TESLA = {"names": ("Tesla",), "symbols": ("TSLA",)}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tesla Postpones Roadster Reveal Event", True),
        ("Wall Street expects a drop in tesla deliveries", True),  # names are case-insensitive
        ("Why $TSLA keeps rising", True),
        ("Shares of TSLA fell 2%", True),
        ("NASDAQ:TSLA hits a new high", True),
        ("S&P 500 dips, Nasdaq higher after inflation data", False),
        ("Teslas everywhere", False),  # whole words only
        ("tsla is up", False),  # tickers are case-sensitive
        ("", False),
    ],
)
def test_mentions_company(text: str, expected: bool) -> None:
    assert mentions_company(text, **TESLA) is expected


def test_a_one_letter_ticker_needs_a_cashtag_or_parentheses() -> None:
    ford = {"names": ("Ford Motor",), "symbols": ("F",)}
    assert mentions_company("Shares of Ford Motor (F) rose", **ford)
    assert mentions_company("$F jumps after earnings", **ford)
    assert not mentions_company("F-150 recall widens", **ford)
    assert not mentions_company("Grade F for the economy", **ford)


def test_a_short_name_is_matched_case_sensitively() -> None:
    three_m = {"names": ("3M",), "symbols": ("MMM",)}
    assert mentions_company("3M settles earplug suits", **three_m)
    assert not mentions_company("Posted 3m ago", **three_m)


def test_names_with_punctuation_match_as_whole_words() -> None:
    assert mentions_company("AT&T raises its dividend", names=("AT&T",), symbols=("T",))
    assert mentions_company("Coca-Cola beats estimates", names=("Coca-Cola",), symbols=("KO",))
