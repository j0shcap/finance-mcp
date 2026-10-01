Solve the annual yield to maturity from a bond's CLEAN price at a settlement date.

The dated counterpart to bond_ytm, and the inverse of bond_price_dated: use it when
settlement may fall between coupon dates. Also returns the accrued interest and the
dirty price, so a clean-price quote still tells you the cash amount. Day count
defaults to Actual/Actual ICMA; '30/360' reproduces Excel's YIELD with basis=0, except
in the final coupon period -- pass first_period_discount='simple' to match Excel there.