"""
Step 1-4 of the Credit Risk Model project: Load data & Exploratory Data Analysis (EDA).

Dataset: "Give Me Some Credit" (Kaggle, 2011, Credit Fusion)
Goal: Predict whether a borrower will experience serious delinquency
      (90+ days past due) within the next 2 years.

Target column: SeriousDlqin2yrs (1 = defaulted, 0 = did not)
"""

import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------------------
# Step 3: Load the data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/cs-training.csv", index_col=0)

print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------------------
# Step 4: EDA - understand distributions, target balance, and data quality
# ---------------------------------------------------------------------------

print("\n--- Target balance (SeriousDlqin2yrs) ---")
print(df["SeriousDlqin2yrs"].value_counts(normalize=True))
# ~6.7% defaulted. This is an IMBALANCED classification problem.
# Accuracy alone would be a misleading metric here - a model that always
# predicts "no default" would already score ~93% accuracy while being useless.

print("\n--- Summary statistics ---")
print(df.describe().T)

print("\n--- Missing values ---")
print(df.isna().sum())
# MonthlyIncome: ~20% missing
# NumberOfDependents: ~2.6% missing

print("\n--- Data quality issues found ---")

# 1. age == 0 is impossible for a credit applicant
print("Rows with age == 0:", (df["age"] == 0).sum())

# 2. The three delinquency-count columns contain sentinel/placeholder values
#    98 and 96 appear far too often to be genuine counts of times late.
for col in [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]:
    n_98 = (df[col] == 98).sum()
    n_96 = (df[col] == 96).sum()
    print(f"{col}: count==98 -> {n_98}, count==96 -> {n_96}")

# 3. RevolvingUtilizationOfUnsecuredLines should logically be <= 1 (it's a ratio)
print(
    "RevolvingUtilizationOfUnsecuredLines > 1:",
    (df["RevolvingUtilizationOfUnsecuredLines"] > 1).sum(),
    "| max:",
    df["RevolvingUtilizationOfUnsecuredLines"].max(),
)

# 4. DebtRatio blows up when MonthlyIncome is missing/near-zero
print(
    "DebtRatio > 1:",
    (df["DebtRatio"] > 1).sum(),
    "out of",
    len(df),
    "| max:",
    df["DebtRatio"].max(),
)

if __name__ == "__main__":
    pass
