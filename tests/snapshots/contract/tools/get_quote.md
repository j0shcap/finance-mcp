Current price snapshots for up to 25 tickers (price, change, ranges, market cap).

Tickers are fetched in parallel and results are partial: successful quotes come back
in `quotes`, and any ticker that could not be fetched is named in `errors` with the
reason (invalid/delisted symbol vs. a source failure). One bad ticker does not
invalidate the rest, so there is no need to retry the whole batch.