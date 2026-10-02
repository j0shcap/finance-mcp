"""Bounded input types for the tool boundary.

Two jobs. First, a malformed argument should be rejected by the schema, before a fetch or
a numeric loop starts: the model then gets a validation error naming the field instead of
a data-layer error, and a typo costs no network call. Second, every bound here caps work
per call - an unbounded cashflow list or bond maturity otherwise turns one tool call into
a CPU-bound loop that blocks the server.

Bounds are deliberately generous: they are there to exclude nonsense, not to second-guess
a legitimate model. The calculators still validate their own preconditions, since several
(``ytm > -frequency``, a required sign change) depend on more than one argument and cannot
be expressed as a static field constraint.
"""

from typing import Annotated

from pydantic import Field

#: Ticker symbols are alphanumeric with '.', '-', '^' and '=' (BRK-B, ^GSPC, RY.TO,
#: BTC-USD, EURUSD=X, 005930.KS). Surrounding whitespace is allowed because the data
#: layer normalizes (strip + upper) before fetching; whitespace *inside* is not, since
#: that means two symbols were passed in one argument.
TICKER_PATTERN = r"^\s*[A-Za-z0-9.\-^=]{1,20}\s*$"

Ticker = Annotated[
    str,
    Field(
        min_length=1,
        max_length=24,
        pattern=TICKER_PATTERN,
        description=(
            "Ticker symbol, e.g. 'AAPL'. Case-insensitive. Exchange suffixes and prefixes are "
            "allowed (BRK-B, ^GSPC, RY.TO, BTC-USD, EURUSD=X); one symbol per value."
        ),
    ),
]

#: Most cashflows one npv/irr/mirr/xnpv/xirr call may take. The IRR solvers scan a fixed
#: grid of ~1400 candidate rates and evaluate the whole series at each, so cost is linear
#: in this bound; 1000 covers a monthly series over 80+ years.
MAX_CASHFLOWS = 1000

#: bond_price walks every coupon period (years_to_maturity * frequency) one at a time,
#: so both factors have to be bounded for the loop to be.
MAX_BOND_YEARS = 100
MAX_COUPON_FREQUENCY = 12

#: loan_schedule walks one row per month.
MAX_LOAN_TERM_MONTHS = 1200

#: Discrete compounding no finer than daily; use compounding='continuous' for the limit.
MAX_PERIODS_PER_YEAR = 365

#: Enough to name every line item on any financial statement, several times over.
MAX_LINE_ITEMS = 100

#: Most tickers one get_quote call may take; they are fetched in parallel.
MAX_QUOTE_TICKERS = 25

#: Most tickers one compare_tickers call may take. Each row costs two data lookups, so the
#: bound is what keeps a single tool call from opening dozens of connections.
MAX_COMPARE_TICKERS = 10

#: An annual risk-free rate for the risk-adjusted statistics, or None for the T-bill default.
#: Bounded well outside any real policy rate but away from -100%, where de-annualizing the
#: rate is undefined.
RiskFreeRate = Annotated[
    float | None,
    Field(
        ge=-0.5,
        le=1.0,
        description=(
            "Annual risk-free rate as a DECIMAL, not a percent: 0.045 means 4.5%. Leave it out "
            "to use the 13-week US T-bill yield averaged over the measured dates, so the "
            "Sharpe, Sortino and alpha are excess return over cash. Pass 0 for RAW return per "
            "unit of risk."
        ),
    ),
]
