Return and risk stats from daily auto-adjusted closes over the window.

Includes total and annualized return, annualized volatility, max drawdown
(negative percent), 50/200-day SMAs (null if insufficient history), and the
risk-adjusted set: Sharpe, Sortino, downside deviation and Calmar (CAGR per unit
of max drawdown).

Annualized figures use the actual calendar span between the first and last bar,
so over a one-year window the annualized return equals the total return for any
instrument. Volatility is scaled by an observations-per-year factor inferred from
the data and reported as periods_per_year (roughly 252 for a weekday-traded
equity, 365 for a 24/7 instrument such as crypto). The annualized figures,
periods_per_year and every risk-adjusted ratio are null when the window spans
under 85 days (just under three months), because annualizing a sub-quarter move
extrapolates noise into a yearly rate.

Left out, risk_free_rate is the 13-week US T-bill yield averaged over the same
dates, so Sharpe, Sortino and downside deviation are excess over cash; pass 0 for
raw figures. The rate and its source are echoed in the result, and if the T-bill
average cannot be formed those three are null with risk_free_rate_note saying why.
For beta, alpha or a comparison against an index, use compare_to_benchmark.