"""Arguments every prompt is rendered with by the tests that read rendered prompt text.

Shared by the prompt drift guard, the contract snapshot and the e2e protocol test. It must
name every registered prompt: each of those tests fails on a prompt missing from it, so a new
prompt cannot bypass them by never being rendered.
"""

SAMPLE_ARGS: dict[str, dict[str, str]] = {
    "analyze_stock": {"ticker": "AAPL", "horizon": "3y"},
    "compare_stocks": {"tickers": "KO, PEP, MDLZ", "horizon": "3y"},
    "investment_cashflows": {
        "cashflows": "-1000, 300, 300, 400",
        "discount_rate": "8%",
        "reinvest_rate": "6%",
    },
    "bond_analysis": {"bond": "UST 4% due 2036-01-15, clean 92.30", "shock_bp": "50"},
    "loan_planner": {
        "principal": "400000",
        "annual_rate": "6.5%",
        "term_months": "360",
        "extra_payment": "250",
    },
}
