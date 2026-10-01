Valuation, profitability, and leverage ratios (as reported by Yahoo).

Note units differ by field: P/E, P/B, P/S, EV/EBITDA, PEG are plain ratios;
margins and ROE/ROA are fractions (0.27 = 27%); debt_to_equity is a percent
(79.5 = 79.5%); EV, total debt/cash, FCF, EBITDA are absolute amounts. Those
amounts are not all in one currency: debt/cash/FCF/EBITDA and the per-share
revenue/book value are in `financial_currency`, the EPS fields in `currency`. They
differ for ADRs and other cross-listings, where Yahoo computes P/S, P/B, EV and the
EV multiples across both currencies: use P/E, PEG and the margins for those.