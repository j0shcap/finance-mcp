# finance-mcp

Finance tools over Yahoo Finance market data plus offline financial calculators.

Two tool families:
- Market data (open world, live, one Yahoo call each): search_symbols, get_quote, get_price_history, get_financials, get_company_profile, get_key_metrics, get_analyst_data, analyze_performance, compare_to_benchmark, compare_tickers, get_news.
  Resolve a name to a ticker with search_symbols first; get_quote prices up to 25 tickers in
  one call. Tickers are Yahoo symbols, case-insensitive, with the usual prefixes and
  suffixes: BRK-B, ^GSPC, RY.TO, BTC-USD, EURUSD=X.
- Calculators (pure, deterministic, no network): time_value_of_money, loan_schedule, npv, irr, mirr, xnpv, xirr, bond_price, bond_ytm, bond_price_dated, bond_ytm_dated, convert_rate.

Conventions that change the answer:
- Signs follow Excel: cash received is positive, cash paid is negative. A deposit or a loan principal you pay out is negative pv; the balance you get back is positive fv. Get this wrong and the sign of the answer flips (or a rate solve has no solution).
- Rates are decimals, never percents: 0.05 means 5%.
- npv, irr, mirr and time_value_of_money take a PER-PERIOD rate matching the cashflow spacing (monthly flows -> monthly rate). xnpv, xirr, bond_price, bond_ytm, bond_price_dated, bond_ytm_dated, loan_schedule and convert_rate take ANNUAL rates; loan_schedule's annual_rate is a nominal APR compounded monthly.
- irr and xirr return a per-period and an annualized rate respectively; both can have several roots for non-conventional flows (see all_irrs/is_unique) - prefer mirr then.
- convert_rate moves between a nominal annual rate (APR) and an effective annual rate (APY/EAR).
- Bond prices come in two flavours and mixing them up misstates the cash by up to a full coupon. The CLEAN price is what the market quotes; the DIRTY (or full/invoice) price is clean + accrued interest, and is what the buyer actually pays. bond_price_dated reports both, per face and per 100 of face; bond_ytm_dated solves from the CLEAN price, so subtract accrued interest first if you were given a dirty one. On a coupon date nothing has accrued and the two coincide, which is why bond_price/bond_ytm report a single price.
- bond_price_dated and bond_ytm_dated default to the Actual/Actual ICMA day count (US Treasuries and most sovereigns), NOT Excel's default of 30/360 - pass day_count='30/360' to match Excel's PRICE/YIELD or to price a US corporate or municipal bond. They also default to the street convention for the part period before the next coupon; pass first_period_discount='simple' to match the US Treasury's own published prices, or to match Excel inside the FINAL coupon period, where Excel too uses simple interest over the stub.

Yahoo's market-data units are inconsistent: margins and ROE/ROA are fractions (0.27 = 27%) but
debt_to_equity and dividend_yield are ALREADY PERCENTS (79.5 = 79.5%, 5.92 = 5.92%), and
recommendation_mean is inverted (1 = strong buy, 5 = strong sell). Absolute amounts are not all
in one currency - check each result's currency/financial_currency. Read the finance://conventions
resource for the full per-field glossary before converting or comparing figures.

Every tool is read-only. Unavailable data is reported as an error or a null field; never fill a
gap with a guess.
