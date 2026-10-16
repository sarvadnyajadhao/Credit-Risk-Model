"""
Step 5: Clean the raw data.

Issues found during EDA (see outputs/01_eda_overview.png and src/01_eda.py):
  1. age == 0 for one row               -> drop (impossible value)
  2. 98 / 96 sentinel codes in the three
     "days past due" columns            -> these placeholder values also come
                                            with NumberOfTime30-59... == 98, so
                                            we treat all three as "unreliable
                                            history" for that row and cap them
  3. RevolvingUtilizationOfUnsecuredLines
     values > ~2 (should be a 0-1 ratio) -> cap (winsorize) at 99.5th percentile
  4. DebtRatio extreme outliers          -> cap at 99.5th percentile
  5. MonthlyIncome missing (~20%)        -> median imputation BY AGE GROUP,
                                            plus a flag column so the model can
                                            learn "income was unknown" as its
                                            own signal (this matters: people
                                            who don't report income may behave
                                            differently from those who do)
  6. NumberOfDependents missing (~2.6%)  -> median imputation (0, the mode)

We keep a flag column for every imputation so we never hide missingness from
the model - missingness itself can be predictive.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("data/cs-training.csv", index_col=0)
print("Before cleaning:", df.shape)

# 1. Drop the impossible age
df = df[df["age"] > 0].copy()

# 2. Cap the sentinel-coded delinquency columns.
#    98/96 are not real counts; cap all three "past due" columns at a
#    realistic max (we use 17, the highest genuine value seen in the data)
late_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
REALISTIC_MAX = 17
for col in late_cols:
    df[f"{col}_was_capped"] = (df[col] > REALISTIC_MAX).astype(int)
    df[col] = df[col].clip(upper=REALISTIC_MAX)

# 3. Cap RevolvingUtilization at 99.5th percentile (winsorizing, not deleting)
cap_util = df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.995)
df["RevolvingUtilizationOfUnsecuredLines"] = df[
    "RevolvingUtilizationOfUnsecuredLines"
].clip(upper=cap_util)

# 4. Cap DebtRatio at 99.5th percentile
cap_debt = df["DebtRatio"].quantile(0.995)
df["DebtRatio"] = df["DebtRatio"].clip(upper=cap_debt)

# 5. MonthlyIncome: flag + median-by-age-group imputation
df["MonthlyIncome_missing"] = df["MonthlyIncome"].isna().astype(int)
age_bin = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 70, 120])
median_income_by_age = df.groupby(age_bin, observed=True)["MonthlyIncome"].transform(
    "median"
)
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(median_income_by_age)
# any remaining (shouldn't be any) fall back to global median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

# 6. NumberOfDependents: flag + median (mode is 0) imputation
df["NumberOfDependents_missing"] = df["NumberOfDependents"].isna().astype(int)
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
    df["NumberOfDependents"].median()
)

print("After cleaning:", df.shape)
print("\nRemaining missing values:\n", df.isna().sum().sum(), "total nulls")

df.to_csv("data/cleaned_train.csv")
print("Saved -> data/cleaned_train.csv")
