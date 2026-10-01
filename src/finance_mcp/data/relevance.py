"""Does a news headline name the company it was filed under? Pure text matching.

Yahoo's per-ticker news stream mixes market-wide stories ("S&P 500 dips...") in with
company news, and its payload carries no related-ticker metadata to tell them apart, so the
only signal is the text. This matches the company's name and ticker as whole words.

It is deliberately a flag, not a filter: brand and executive names ("Google" for
Alphabet, "Musk" for Tesla) are not matched, so a False is a hint to read the title, not a
verdict that the story is unrelated. The rules lean toward False, because the prompts
discount only False articles: a wrong True would pass a market-wide story off as company
news. A company named by a common word can still collide with a Title-Case headline
("Price Target" for Target), which no text rule can rule out.
"""

import re

#: Legal forms, share-class markers and filler stripped from the END of a company name.
_TRAILING_NOISE = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "ltd",
        "limited",
        "plc",
        "llc",
        "lp",
        "l.p",
        "holdings",
        "holding",
        "group",
        "n.v",
        "nv",
        "s.a",
        "sa",
        "ag",
        "se",
        "asa",
        "ab",
        "the",  # Yahoo's "Coca-Cola Company (The)"
        "new",  # Yahoo's "Berkshire Hathaway Inc. New"
        "&",
        "-",
    }
)

#: First words too common to stand for a company on their own: "General Motors" is
#: matched in full, never as "General".
_GENERIC_FIRST_WORDS = frozenset(
    {
        "american",
        "general",
        "united",
        "first",
        "national",
        "international",
        "bank",
        "global",
        "federal",
        "royal",
        "canadian",
        "china",
        "japan",
        "new",
        "southern",
        "northern",
        "western",
        "eastern",
        "pacific",
        "atlantic",
        "public",
        "capital",
        "energy",
        "digital",
        "advanced",
        "applied",
        "texas",
        "union",
        "standard",
        # Also words and surnames that stand for other things: "US home sales" is not Home
        # Depot, "J.P. Morgan" is not Morgan Stanley.
        "home",
        "morgan",
        "philip",
        "johnson",
        "wells",
        "best",
        "dollar",
    }
)

#: A first word shorter than this is too likely to be a common word to match alone.
_MIN_FIRST_WORD = 4

#: Tickers this short are everyday words or acronyms (F, T, AI, ET, PM, US), so they
#: count only as a cashtag or in parentheses: "$ET", "(AI)".
_MAX_SHORT_TICKER = 2

_NOT_ALNUM_BEFORE = r"(?<![A-Za-z0-9])"
_NOT_ALNUM_AFTER = r"(?![A-Za-z0-9])"


def company_aliases(long_name: str | None, short_name: str | None) -> tuple[str, ...]:
    """The forms of a company's name a headline would use, most specific first.

    "Tesla, Inc." -> ("Tesla",); "JPMorgan Chase & Co." -> ("JPMorgan Chase", "JPMorgan");
    "Amazon.com, Inc." -> ("Amazon.com", "Amazon").
    """
    aliases: list[str] = []
    for name in (long_name, short_name):
        core = _core_name(name or "")
        if not core:
            continue
        candidates = [core]
        first = core.split()[0]
        if (
            first != core
            and len(first) >= _MIN_FIRST_WORD
            and first.lower() not in _GENERIC_FIRST_WORDS
        ):
            candidates.append(first)
        if core.lower().endswith(".com"):
            candidates.append(core[: -len(".com")])
        aliases.extend(c for c in candidates if c not in aliases)
    return tuple(aliases)


def _core_name(name: str) -> str:
    """Strip the legal form: "The Coca-Cola Company" -> "Coca-Cola"."""
    tokens = name.split(",")[0].split()
    if tokens and tokens[0].lower() == "the":
        tokens = tokens[1:]
    while tokens:
        last = tokens[-1].lower().strip(".()")
        if last in _TRAILING_NOISE:
            tokens.pop()
        elif len(tokens) >= 2 and tokens[-2].lower() == "class" and len(last) == 1:
            del tokens[-2:]
        else:
            break
    return " ".join(tokens)


def symbol_aliases(symbol: str) -> tuple[str, ...]:
    """The ticker as headlines write it: "BRK-B" -> ("BRK-B", "BRK.B", "BRK")."""
    aliases = [symbol, symbol.replace("-", "."), re.split(r"[.\-]", symbol)[0]]
    return tuple(dict.fromkeys(a for a in aliases if a))


def mentions_company(text: str, names: tuple[str, ...], symbols: tuple[str, ...]) -> bool:
    """True if ``text`` names the company or its ticker as a whole word.

    Names match in their own capitalisation, so "price target" is not Target and "to
    block" is not Block; an all-caps name ("NVIDIA") also matches however a headline
    capitalises it ("Nvidia"). Tickers match case-sensitively, since a lower-case "tsla" is
    not a ticker, and one of one or two letters (F, T, AI, ET) only as a cashtag or in
    parentheses -- "$F", "(AI)" -- because those are everyday words and acronyms.
    """
    for name in names:
        letters = [c for c in name if c.isalpha()]
        all_caps = len(letters) >= 2 and all(c.isupper() for c in letters)
        flags = re.IGNORECASE if all_caps else 0
        if re.search(_NOT_ALNUM_BEFORE + re.escape(name) + _NOT_ALNUM_AFTER, text, flags):
            return True
    for symbol in symbols:
        escaped = re.escape(symbol)
        pattern = (
            rf"\${escaped}{_NOT_ALNUM_AFTER}|\({escaped}\)"
            if len(symbol) <= _MAX_SHORT_TICKER
            else _NOT_ALNUM_BEFORE + escaped + _NOT_ALNUM_AFTER
        )
        if re.search(pattern, text):
            return True
    return False
