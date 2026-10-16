"""
Step 18-19: Expected Loss (EL) and Stress Testing.

EXPECTED LOSS - THE CORE FORMULA OF CREDIT RISK:

    Expected Loss = PD x LGD x EAD

    PD  = Probability of Default (our model's output)
    LGD = Loss Given Default - the % of the exposure the lender actually
          loses if the borrower defaults (the rest might be recovered via
          collateral, collections, etc). We don't have real recovery data,
          so we use a standard industry assumption for unsecured consumer
          credit: LGD = 45% (this is the Basel II regulatory default
          assumption for unsecured retail exposures when a bank has no
          internal recovery data of its own - a defensible, explainable
          choice, not an arbitrary one).
    EAD = Exposure at Default - how much money is actually at risk. This
          dataset has no actual loan amount column, so for portfolio-level
          dollar figures we ASSUME an average exposure per applicant
          (clearly labeled as an assumption below). For a single applicant
          in the Streamlit app, the user will input their own requested
          loan amount, which becomes the real EAD for that calculation.

STRESS TESTING:
A real risk model must be tested against "what if the economy gets worse."
We simulate this the standard simple way: scale every applicant's PD up by
a stress multiplier (representing higher macro-driven default risk) and see
how portfolio-level Expected Loss changes. This is a simplified version of
what banks call "PD stress testing under adverse macroeconomic scenarios"
(e.g. CCAR/stress-test style exercises use GDP/unemployment shocks mapped to
PD multipliers; we use illustrative multipliers here since we have no macro
model component).
"""

import pandas as pd
import numpy as np

test_raw = pd.read_csv("data/test_raw.csv", index_col=0)
y_test = test_raw["SeriousDlqin2yrs"]
pd_xgb = pd.read_csv("outputs/xgboost_pd_test.csv", index_col=0)["pd_xgboost"]

LGD = 0.45  # Basel II standard assumption, unsecured retail exposure
ASSUMED_AVG_EXPOSURE = 10_000  # illustrative avg loan/credit-line $ per applicant

# --- Baseline Expected Loss ---
el_rate = pd_xgb * LGD  # expected loss as a % of exposure, per applicant
portfolio_el_dollars = (el_rate * ASSUMED_AVG_EXPOSURE).sum()
portfolio_exposure = ASSUMED_AVG_EXPOSURE * len(pd_xgb)

print(f"Assumptions: LGD = {LGD:.0%}, avg exposure/applicant = ${ASSUMED_AVG_EXPOSURE:,}")
print(f"Portfolio size (test set): {len(pd_xgb):,} applicants, ${portfolio_exposure:,.0f} total exposure")
print(f"\nBASELINE Expected Loss: ${portfolio_el_dollars:,.0f} "
      f"({portfolio_el_dollars/portfolio_exposure:.2%} of total exposure)")

# --- Stress testing: scale PD under adverse scenarios ---
scenarios = {
    "Baseline": 1.0,
    "Mild recession (PD x1.5)": 1.5,
    "Moderate recession (PD x2.0)": 2.0,
    "Severe recession (PD x3.0)": 3.0,
}

rows = []
for name, multiplier in scenarios.items():
    stressed_pd = np.clip(pd_xgb * multiplier, 0, 1)
    stressed_el_rate = stressed_pd * LGD
    stressed_el_dollars = (stressed_el_rate * ASSUMED_AVG_EXPOSURE).sum()
    rows.append(
        {
            "scenario": name,
            "avg_PD": stressed_pd.mean(),
            "expected_loss_$": stressed_el_dollars,
            "EL_as_%_of_exposure": stressed_el_dollars / portfolio_exposure,
        }
    )

stress_table = pd.DataFrame(rows)
print("\n--- Stress Test Results ---")
print(stress_table.to_string(index=False))
stress_table.to_csv("outputs/stress_test_results.csv", index=False)

print(
    f"\nUnder a severe recession scenario, portfolio Expected Loss would grow "
    f"from ${portfolio_el_dollars:,.0f} to ${rows[-1]['expected_loss_$']:,.0f} "
    f"-> a {rows[-1]['expected_loss_$']/portfolio_el_dollars - 1:.1%} increase, "
    f"with no change in the actual customers, purely from a worse macro environment."
)
