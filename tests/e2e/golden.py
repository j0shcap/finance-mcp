"""One golden call per calculator tool: arguments and the result fields they must produce.

The expected figures are published ones (Microsoft's Excel function reference examples, or
closed forms a reader can check by hand), never numbers this server produced. Deriving
them from the implementation, or from another library at test time, would let a shared
mistake pass; a literal from the reference cannot drift with either.
"""

from typing import Any

#: tool name -> (arguments, {result field: expected value}).
GOLDEN: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    # 100 paid now at 5% for 10 periods: 100 * 1.05**10.
    "time_value_of_money": (
        {"solve_for": "fv", "pv": -100, "pmt": 0, "rate": 0.05, "nper": 10},
        {"solved_for": "fv", "solved_value": 162.889462677744},
    ),
    # 30-year 300k mortgage at 6% APR: Excel PMT(0.06/12, 360, 300000) = -1798.65.
    "loan_schedule": (
        {"principal": 300_000, "annual_rate": 0.06, "term_months": 360},
        {"monthly_payment": 1798.65157545827, "n_payments": 360},
    ),
    # -100 + 60/1.1 + 60/1.1**2.
    "npv": (
        {"rate": 0.10, "cashflows": [-100, 60, 60]},
        {"npv": 4.13223140495868},
    ),
    # Root of 60x + 60x**2 = 100 with x = 1/(1+r): r = 2/(sqrt(23/3) - 1) - 1.
    "irr": (
        {"cashflows": [-100, 60, 60]},
        {"irr": 0.130662386291808, "is_unique": True},
    ),
    # Excel MIRR reference example (five years, 10% finance, 12% reinvest): 12.61%.
    "mirr": (
        {
            "cashflows": [-120_000, 39_000, 30_000, 21_000, 37_000, 46_000],
            "finance_rate": 0.10,
            "reinvest_rate": 0.12,
        },
        {"mirr": 0.126094130365},
    ),
    # Excel XNPV reference example: 2086.65.
    "xnpv": (
        {
            "rate": 0.09,
            "cashflows": [
                {"date": "2008-01-01", "amount": -10_000},
                {"date": "2008-03-01", "amount": 2_750},
                {"date": "2008-10-30", "amount": 4_250},
                {"date": "2009-02-15", "amount": 3_250},
                {"date": "2009-04-01", "amount": 2_750},
            ],
        },
        {"npv": 2086.64760203},
    ),
    # Excel XIRR reference example (same flows): 37.34%.
    "xirr": (
        {
            "cashflows": [
                {"date": "2008-01-01", "amount": -10_000},
                {"date": "2008-03-01", "amount": 2_750},
                {"date": "2008-10-30", "amount": 4_250},
                {"date": "2009-02-15", "amount": 3_250},
                {"date": "2009-04-01", "amount": 2_750},
            ]
        },
        {"irr": 0.373362533519},
    ),
    # A bond priced at its own coupon rate trades at par. Macaulay duration of a par bond
    # is (1+y)/y * (1 - (1+y)**-n) periods: y = 0.025, n = 20 -> 15.9789 periods / 2.
    "bond_price": (
        {
            "face": 1000,
            "coupon_rate": 0.05,
            "years_to_maturity": 10,
            "ytm": 0.05,
            "frequency": 2,
        },
        {
            "price": 1000.0,
            "current_yield": 0.05,
            "macaulay_duration": 7.98944567139,
            "modified_duration": 7.79458114282,
        },
    ),
    "bond_ytm": (
        {
            "face": 1000,
            "coupon_rate": 0.05,
            "years_to_maturity": 10,
            "price": 1000,
            "frequency": 2,
        },
        {"yield_to_maturity": 0.05},
    ),
    # Excel PRICE reference example (basis 0 = 30/360 US): 94.63436.
    "bond_price_dated": (
        {
            "settlement": "2008-02-15",
            "maturity": "2017-11-15",
            "coupon_rate": 0.0575,
            "ytm": 0.065,
            "frequency": 2,
            "day_count": "30/360",
        },
        {"clean_price_per_100": 94.6343616213221},
    ),
    # Excel YIELD reference example (basis 0): 6.50%.
    "bond_ytm_dated": (
        {
            "settlement": "2008-02-15",
            "maturity": "2016-11-15",
            "coupon_rate": 0.0575,
            "clean_price": 95.04287,
            "frequency": 2,
            "day_count": "30/360",
        },
        {"yield_to_maturity": 0.065},
    ),
    # 12% APR compounded monthly: 1.01**12 - 1.
    "convert_rate": (
        {"rate": 0.12, "periods_per_year": 12, "direction": "nominal_to_effective"},
        {"converted_rate": 0.126825030131970},
    ),
}
