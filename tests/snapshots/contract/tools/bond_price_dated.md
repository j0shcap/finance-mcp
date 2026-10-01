Price a bond for a settlement date that may fall BETWEEN coupon dates.

Use this for a real bond quoted by its maturity date; use bond_price only when
settlement lands exactly on a coupon date. Returns the CLEAN price (quoted, excludes
accrued interest), the DIRTY price (clean + accrued = the cash the buyer pays), the
accrued interest, and duration/convexity computed with the fractional first period
discounted as first_period_discount says. Each price is given per 'face' and per 100
of face.

Day count defaults to Actual/Actual ICMA (US Treasuries and most sovereigns); pass
day_count='30/360' for the US corporate/municipal convention, which reproduces
Excel's PRICE with basis=0 -- except in the final coupon period, where Excel uses
simple interest over the stub: pass first_period_discount='simple' to match it there.

Assumes a regular schedule -- every coupon period a whole 12/frequency months. Bonds
with an odd (long or short) first or last coupon period are not supported.