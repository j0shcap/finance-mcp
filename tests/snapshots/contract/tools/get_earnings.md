When a company reports next, what analysts expect, and whether it beat lately.

next_report: the next report's date-time in the exchange's timezone, and whether the
company has confirmed it (date_is_estimate false; true or null means it hasn't).
estimates: EPS and revenue consensus (average, low, high, analyst count, year-ago value,
growth) for the quarter the next report covers, the quarter after, and their fiscal
years, each with fiscal_period_end. history: the last four quarters' EPS against the
consensus, oldest first, with surprise_percent. growth_percent and surprise_percent are
PERCENTS; EPS, revenue and history can each be in a different currency, so check
eps_currency, revenue_currency and history_currency. Companies only: ETFs, funds,
indices, currencies and crypto return an error.