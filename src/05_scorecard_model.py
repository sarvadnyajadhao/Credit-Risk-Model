"""
Step 9-10: Logistic Regression Scorecard + Probability of Default (PD).

WHY LOGISTIC REGRESSION + WOE (not straight to a fancy model)?
This mirrors exactly what banks actually deploy for regulatory-facing credit
scoring: it's simple, monotonic per feature (thanks to WOE), and every single
point in the final score can be traced back to "this applicant is in the
risky bin for feature X, which costs them Y points." That auditability is
the entire reason this 1960s technique is still industry standard alongside
modern ML.

STEPS:
1. Train/test split (stratified, because the target is imbalanced)
2. Fit WOE bins on the TRAINING set only
3. Transform both sets to WOE features
4. Fit a logistic regression on the WOE features -> this predicts PD directly
5. Convert the logistic regression's log-odds into a points-based score,
   using the same PDO (Points to Double the Odds) method real scorecards use
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from woe_utils import WOEBinner

df = pd.read_csv("data/featured_train.csv", index_col=0)
target = "SeriousDlqin2yrs"

features = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalPastDue",
    "IncomePerDependent",
    "EstimatedMonthlyDebt",
    "CreditLinesPerAge",
]
# NumberRealEstateLoansOrLines had IV < 0.02 (useless) - drop it for the
# scorecard, keeping the model lean and each feature justified
features.remove("NumberRealEstateLoansOrLines")

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# --- Step 8 (applied properly this time: fit on TRAIN only) ---
train_df_for_woe = X_train.copy()
train_df_for_woe[target] = y_train
binner = WOEBinner().fit(train_df_for_woe, features, target)

print("\nFinal IV table (computed on training data only):")
print(binner.iv_table())

X_train_woe = binner.transform(X_train, features)
X_test_woe = binner.transform(X_test, features)

# --- Step 9: Logistic Regression on WOE features ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_woe, y_train)

# --- Step 10: Probability of Default ---
pd_train = logreg.predict_proba(X_train_woe)[:, 1]
pd_test = logreg.predict_proba(X_test_woe)[:, 1]
print(f"\nMean predicted PD - train: {pd_train.mean():.4f}, test: {pd_test.mean():.4f}")
print(f"Actual default rate - train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")

# --- Convert to a points-based score (PDO method, same as real scorecards) ---
BASE_SCORE = 600      # score assigned at the base odds below
BASE_ODDS = 50        # 50 good : 1 bad at the base score
PDO = 20              # points needed to double the good:bad odds

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

# logreg gives log-odds of DEFAULT (bad). Log-odds of GOOD is the negative.
log_odds_bad_train = logreg.decision_function(X_train_woe)
log_odds_good_train = -log_odds_bad_train
score_train = OFFSET + FACTOR * log_odds_good_train

log_odds_bad_test = logreg.decision_function(X_test_woe)
score_test = OFFSET + FACTOR * (-log_odds_bad_test)

print(f"\nScore range (test set): {score_test.min():.0f} - {score_test.max():.0f}")
print(f"Score mean (test set): {score_test.mean():.0f}")


def risk_grade(score, cuts):
    if score >= cuts[0.8]:
        return "A (Low Risk)"
    elif score >= cuts[0.6]:
        return "B (Below Avg Risk)"
    elif score >= cuts[0.4]:
        return "C (Average Risk)"
    elif score >= cuts[0.2]:
        return "D (Above Avg Risk)"
    else:
        return "E (High Risk)"


# NOTE: grade cutoffs are set from this population's OWN score distribution
# (quintiles of the training scores), not copied from generic FICO-style
# numbers. A textbook cutoff like "700 = low risk" means nothing unless it
# actually separates this specific applicant pool - here, training scores
# range ~420-640, so a fixed cutoff of 700 would (and did, on a first pass)
# put nobody in the top grades. Calibrate cutoffs to the data, always.
log_odds_bad_train_for_cuts = logreg.decision_function(X_train_woe)
score_train_for_cuts = OFFSET + FACTOR * (-log_odds_bad_train_for_cuts)
cuts = {
    q: float(np.quantile(score_train_for_cuts, q)) for q in [0.2, 0.4, 0.6, 0.8]
}
print("\nGrade cutoffs (from training score quintiles):", cuts)

grades_test = pd.Series(score_test).apply(lambda s: risk_grade(s, cuts))
print("\nRisk grade distribution (test set):")
print(grades_test.value_counts())

# --- Build the actual points table (bank-style scorecard document) ---
points_rows = []
for feat, coef in zip(features, logreg.coef_[0]):
    for bin_label, woe_val in binner.woe_maps_[feat].items():
        points = -FACTOR * coef * woe_val
        points_rows.append(
            {"feature": feat, "bin": str(bin_label), "woe": woe_val, "points": points}
        )
points_table = pd.DataFrame(points_rows)
points_table.to_csv("outputs/scorecard_points_table.csv", index=False)
print("\nSaved full points table -> outputs/scorecard_points_table.csv")
print("Base points (from intercept):", OFFSET - FACTOR * logreg.intercept_[0])

# --- Save everything needed to reuse this model later (e.g. in Streamlit) ---
scorecard_bundle = {
    "binner": binner,
    "logreg": logreg,
    "features": features,
    "factor": FACTOR,
    "offset": OFFSET,
    "grade_cuts": cuts,
}
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
joblib.dump(scorecard_bundle, "outputs/scorecard_model.joblib")
# Keep the app's deployed model path in sync whenever the pipeline is retrained.
joblib.dump(scorecard_bundle, "models/scorecard_model.joblib")

# also persist the train/test split + WOE features for the next scripts
X_train_woe.assign(**{target: y_train.values}).to_csv("data/train_woe.csv")
X_test_woe.assign(**{target: y_test.values}).to_csv("data/test_woe.csv")
X_train.assign(**{target: y_train.values}).to_csv("data/train_raw.csv")
X_test.assign(**{target: y_test.values}).to_csv("data/test_raw.csv")
pd.Series(pd_test, index=X_test.index, name="pd_scorecard").to_csv(
    "outputs/scorecard_pd_test.csv"
)
pd.Series(score_test, index=X_test.index, name="score").to_csv(
    "outputs/scorecard_score_test.csv"
)
print("\nSaved model -> outputs/scorecard_model.joblib and models/scorecard_model.joblib")
