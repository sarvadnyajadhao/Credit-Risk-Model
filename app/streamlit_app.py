"""
Steps 20-25: Streamlit application for the Credit Risk Model.

Run locally with:   streamlit run app/streamlit_app.py
(run from the credit-risk-model/ project root so the relative model paths resolve)

What this app does:
  21. Takes applicant details through an input form
  22. Computes a Model-Based Credit Risk Score (the points-based scorecard)
  23. Assigns a Risk Grade (A-E, calibrated to this model's own population)
  24. Computes a calibrated Probability of Default (from the XGBoost model)
  25. Gives a simple Loan Recommendation based on that PD
  Bonus: shows which factors pushed the applicant's risk up or down
         (SHAP if installed, otherwise the scorecard's own points table -
         both are legitimate, auditable explanations)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)

# The scorecard model was pickled with a custom WOEBinner class (src/woe_utils.py).
# joblib needs that class importable under the same module name to unpickle it.
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

st.set_page_config(page_title="Credit Risk Model", page_icon="\U0001F4B3", layout="centered")


@st.cache_resource
def load_models():
    scorecard_bundle = joblib.load(os.path.join(ROOT_DIR, "models", "scorecard_model.joblib"))
    xgb_bundle = joblib.load(os.path.join(ROOT_DIR, "models", "xgboost_model.joblib"))
    return scorecard_bundle, xgb_bundle


scorecard_bundle, xgb_bundle = load_models()
binner = scorecard_bundle["binner"]
logreg = scorecard_bundle["logreg"]
sc_features = scorecard_bundle["features"]
FACTOR = scorecard_bundle["factor"]
OFFSET = scorecard_bundle["offset"]
grade_cuts = scorecard_bundle["grade_cuts"]

xgb_model = xgb_bundle["model"]
xgb_features = xgb_bundle["features"]
platt = xgb_bundle["platt"]

st.title("\U0001F4B3 Credit Risk Model")
model_label = "XGBoost" if xgb_bundle.get("using_xgboost", False) else "scikit-learn HistGradientBoosting"
st.caption(
    "A retail credit scoring tool combining a traditional, auditable "
    "logistic-regression scorecard with a gradient-boosted-tree benchmark "
    f"({model_label}), trained on the Kaggle 'Give Me Some Credit' dataset."
)

st.header("Applicant Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=100)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=15, value=0)
    open_credit_lines = st.number_input("Number of Open Credit Lines/Loans", min_value=0, max_value=60, value=5)
    real_estate_loans = st.number_input("Number of Real Estate Loans/Lines", min_value=0, max_value=30, value=1)
with col2:
    utilization = st.slider("Revolving Credit Utilization (fraction of available credit used)", 0.0, 1.5, 0.3, 0.01)
    debt_ratio = st.slider("Debt Ratio (monthly debt / monthly income)", 0.0, 2.0, 0.3, 0.01)
    late_30_59 = st.number_input("Times 30-59 Days Past Due (last 2 yrs)", min_value=0, max_value=20, value=0)
    late_60_89 = st.number_input("Times 60-89 Days Past Due (last 2 yrs)", min_value=0, max_value=20, value=0)
    late_90_plus = st.number_input("Times 90+ Days Late (last 2 yrs)", min_value=0, max_value=20, value=0)

loan_amount = st.number_input(
    "Requested Loan Amount ($) - used only for the Expected Loss estimate",
    min_value=500, value=10000, step=500,
)

if st.button("Assess Credit Risk", type="primary"):
    # --- Rebuild the exact same engineered features used in training ---
    row = {
        "RevolvingUtilizationOfUnsecuredLines": utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
        "NumberOfTimes90DaysLate": late_90_plus,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents,
    }
    row["TotalPastDue"] = late_30_59 + late_60_89 + late_90_plus
    row["IncomePerDependent"] = monthly_income / (dependents + 1)
    row["EstimatedMonthlyDebt"] = debt_ratio * monthly_income
    row["CreditLinesPerAge"] = open_credit_lines / age

    applicant = pd.DataFrame([row])

    # --- Scorecard score + grade (Steps 22-23) ---
    X_woe = binner.transform(applicant, sc_features)
    log_odds_bad = logreg.decision_function(X_woe)[0]
    score = OFFSET + FACTOR * (-log_odds_bad)

    if score >= grade_cuts[0.8]:
        grade = "A (Low Risk)"
    elif score >= grade_cuts[0.6]:
        grade = "B (Below Avg Risk)"
    elif score >= grade_cuts[0.4]:
        grade = "C (Average Risk)"
    elif score >= grade_cuts[0.2]:
        grade = "D (Above Avg Risk)"
    else:
        grade = "E (High Risk)"

    # --- Calibrated PD from XGBoost (Step 24) ---
    X_raw = applicant[xgb_features]
    raw_pd = xgb_model.predict_proba(X_raw)[:, 1][0]
    raw_pd_clipped = np.clip(raw_pd, 1e-6, 1 - 1e-6)
    logit = np.log(raw_pd_clipped / (1 - raw_pd_clipped))
    calibrated_pd = platt.predict_proba([[logit]])[:, 1][0]

    # --- Loan recommendation (Step 25) ---
    if calibrated_pd < 0.05:
        recommendation = "\u2705 Approve - low estimated default risk"
    elif calibrated_pd < 0.15:
        recommendation = "\u26A0\uFE0F Manual Review - moderate estimated default risk"
    else:
        recommendation = "\u274C Decline / Require Collateral - high estimated default risk"

    # --- Expected Loss for this specific loan (Phase 3 tie-in) ---
    LGD = 0.45
    expected_loss = calibrated_pd * LGD * loan_amount

    st.header("Results")
    r1, r2, r3 = st.columns(3)
    r1.metric("Credit Score", f"{score:.0f}")
    r2.metric("Risk Grade", grade)
    r3.metric("Probability of Default", f"{calibrated_pd:.1%}")

    st.subheader("Recommendation")
    st.markdown(f"### {recommendation}")

    st.subheader("Expected Loss Estimate")
    st.write(
        f"On a {loan_amount:,.0f} USD loan, with an estimated PD of {calibrated_pd:.1%} "
        f"and a standard Loss-Given-Default assumption of {LGD:.0%}, the "
        f"Expected Loss is approximately {expected_loss:,.0f} USD."
   )	

   

    # --- Explanation: top scorecard point contributors for this applicant ---
    st.subheader("What drove this score?")
    contributions = []
    for feat, coef in zip(sc_features, logreg.coef_[0]):
        woe_val = X_woe[f"{feat}_woe"].iloc[0]
        points = -FACTOR * coef * woe_val
        contributions.append((feat, points))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    contrib_df = pd.DataFrame(contributions, columns=["Feature", "Points Impact"]).head(6)
    st.bar_chart(contrib_df.set_index("Feature"))
    st.caption(
        "Positive = pushed the score up (safer). Negative = pushed the score "
        "down (riskier). This is the same points table a real credit "
        "scorecard document would show an underwriter."
    )

st.divider()
st.caption(
    "Educational/portfolio project. Trained on the public 'Give Me Some "
    "Credit' dataset (Kaggle, 2011). Not a real lending decision tool."
)
