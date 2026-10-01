Compute the monthly payment, total interest, and (optionally) the full schedule.

annual_rate is a nominal APR compounded monthly (periodic rate = annual_rate/12),
with monthly payments. With an extra_payment, interest_saved and payments_saved
compare against the same loan without it. By default returns just the summary; set
include_schedule=True for every row.