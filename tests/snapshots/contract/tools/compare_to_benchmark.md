Beta, correlation, Jensen's alpha, tracking error, information ratio and excess
return versus a benchmark, over the dates the two instruments share.

The two daily close series are inner-joined on date, so a 24/7 instrument compared
against an equity benchmark contributes only its weekday closes (the weekend move
lands in the Monday return). overlapping_observations reports how many dates were
actually used - a thin overlap makes every figure noisy, so read it first.

Annualized figures (both CAGRs, alpha, tracking error, information ratio) are null
when the overlap spans under 85 days; beta, correlation and excess return are not,
since they need no annualization. risk_free_rate only affects alpha; left out, it is
the 13-week T-bill yield averaged over the overlapping dates (alpha is null if that
cannot be formed). Returns are in each instrument's own quote currency, so a
cross-currency pair folds an FX move into every figure - say so rather than reading
it straight.