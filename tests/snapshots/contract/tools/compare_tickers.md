Side-by-side performance and key valuation metrics for 2-10 tickers.

Each row carries total/annualized return, volatility, max drawdown and the
risk-adjusted ratios over `period` - measured against the caller's risk_free_rate,
or by default the 13-week T-bill yield over that row's own dates (each row echoes
its rate) - plus Yahoo's valuation metrics (P/E, forward P/E, P/B, P/S, PEG,
EV/EBITDA, margins, ROE, debt/equity) in their as-reported units - margins and ROE
are fractions, debt_to_equity is already a percent. Rank peers on PEG or
growth-vs-multiple rather than raw P/E.

Tickers are fetched in parallel and results are partial: a ticker whose price
history could not be fetched is named in `errors` with the reason and has no row,
and a row whose valuation metrics failed is still present with those fields null
and `metrics_error` set. One bad ticker never invalidates the rest.

Rows whose quote currency differs from the table's base_currency are flagged with
currency_differs, and mixed_currencies summarises it: those returns carry an FX
component the other rows do not, so compare such rows on ratios and say so. A row
whose financial_currency differs from its currency is a cross-listing: its P/S, P/B
and EV multiples mix two currencies, so rank it on P/E, PEG and the margins.