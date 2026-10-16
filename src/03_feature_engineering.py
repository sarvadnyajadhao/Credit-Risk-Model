"""
Step 7: Feature engineering.

We create a small number of features that are known, in real credit risk
practice, to carry strong signal beyond the raw columns:

- TotalPastDue: sum of all three "days past due" counts. A single combined
  delinquency signal is often more stable than three separate sparse columns.
- HasBeenLate: binary flag, "has this person ever been late at all"
- IncomePerDependent: income stretched thinner by more dependents
- DebtToIncomeMonthly: monthly debt payment estimated from DebtRatio*Income
  (sanity feature - correlated with DebtRatio but on an absolute $ scale)
- CreditLinesPerAge: how much credit exposure relative to how long they've
  had time to build a credit history
- UtilizationBucket / AgeBucket: coarse bins, useful later for the WOE step
"""

import pandas as pd

df = pd.read_csv("data/cleaned_train.csv", index_col=0)

df["TotalPastDue"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"]
    + df["NumberOfTime60-89DaysPastDueNotWorse"]
    + df["NumberOfTimes90DaysLate"]
)
df["HasBeenLate"] = (df["TotalPastDue"] > 0).astype(int)

df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

df["EstimatedMonthlyDebt"] = df["DebtRatio"] * df["MonthlyIncome"]

df["CreditLinesPerAge"] = df["NumberOfOpenCreditLinesAndLoans"] / df["age"]

df["AgeBucket"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, 65, 120],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
)

print("New shape:", df.shape)
print("\nNew feature preview:")
print(
    df[
        [
            "TotalPastDue",
            "HasBeenLate",
            "IncomePerDependent",
            "EstimatedMonthlyDebt",
            "CreditLinesPerAge",
            "AgeBucket",
        ]
    ].head()
)

df.to_csv("data/featured_train.csv")
print("Saved -> data/featured_train.csv")
