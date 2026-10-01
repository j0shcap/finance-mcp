Recent news headlines for a symbol, newest first.

Each article has a title, publisher, link and publish time (ISO8601 UTC). Summaries come
from the per-symbol news stream only: when `source` is "search" that stream returned
nothing and this fell back to Yahoo's search endpoint, which carries no summary, so every
summary is null for a reason unrelated to the stories. Works for stocks, ETFs, and crypto.
A symbol with no news (or an unknown symbol) returns an empty article list, not an error.

Yahoo files market-wide stories under a ticker too, so for a stock each article's
mentions_company says whether its title or summary names the company or its ticker.
It is a text match (brand and executive names are not), so it flags rather than
filters: every article is returned, in order. relevance_check says when the flags are
null (not a stock, Yahoo has no company name for it, or fetching the name failed, with
relevance_note saying why).