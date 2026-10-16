"""
Step 8: Weight of Evidence (WOE) encoding + Information Value (IV).

WHAT THIS DOES, IN PLAIN WORDS:
For each feature, we split its values into bins (e.g. age: 18-25, 26-35, ...).
In each bin we compare: "what fraction of ALL non-defaulters fall in this
bin" vs "what fraction of ALL defaulters fall in this bin". If a bin has
relatively MORE defaulters than non-defaulters, that bin is risky -> negative
WOE. If it has relatively FEWER defaulters, it's safe -> positive WOE.

    WOE = ln( %non-defaulters-in-bin / %defaulters-in-bin )

Information Value (IV) sums up how much each feature's bins differ from each
other, across the whole feature. It tells us how PREDICTIVE the whole
feature is, using this standard interpretation:

    IV < 0.02        -> useless, drop it
    0.02 - 0.1        -> weak
    0.1  - 0.3        -> medium, keep
    0.3  - 0.5        -> strong, keep
    > 0.5             -> suspiciously strong, check for a data leak

This is exactly the method real bank scorecards use for feature selection,
because it's simple, monotonic-friendly, and easy for a regulator to audit.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/featured_train.csv", index_col=0)
target = "SeriousDlqin2yrs"
# Keep the descriptive IV report aligned with the model's deterministic split:
# feature-strength estimates must not use labels from the held-out test set.
df, _ = train_test_split(
    df, test_size=0.2, stratify=df[target], random_state=42
)

# Features we'll WOE-encode (numeric + the categorical AgeBucket)
numeric_features = [
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


def bin_feature(series, n_bins=10):
    """
    Bin a numeric feature for WOE.

    IMPORTANT FIX: naive quantile binning (qcut) silently fails on
    zero-inflated count columns (e.g. NumberOfTimes90DaysLate, where ~94%
    of values are 0). When most values are identical, qcut's bin edges
    collapse into a single duplicate-dropped bin -> the feature looks
    artificially "useless" (IV = 0), which is WRONG, not a real finding.

    Fix: if a column has few distinct values (<=15, typical of count
    features), bin on the EXACT integer values instead of quantiles, with
    a "3+" style tail bucket for rare high counts. Otherwise use quantiles
    as normal for genuinely continuous features.
    """
    top_value_share = series.value_counts(normalize=True).iloc[0]
    if series.nunique() <= 20 or top_value_share > 0.4:
        # cap long tail into a single bucket so it has enough volume to
        # produce a stable WOE (avoids single-person bins)
        cap = series[series > 0].quantile(0.99) if (series > 0).any() else 0
        cap = max(int(cap), 3)
        capped = series.clip(upper=cap)
        return capped.astype(str).where(capped < cap, f"{cap}+")
    try:
        return pd.qcut(series, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.qcut(series, q=5, duplicates="drop")


def compute_woe_iv(df, feature, target, is_categorical=False):
    if is_categorical:
        binned = df[feature].astype(str)
    else:
        binned = bin_feature(df[feature])

    grouped = pd.DataFrame({"bin": binned, "target": df[target]})
    total_good = (grouped["target"] == 0).sum()
    total_bad = (grouped["target"] == 1).sum()

    summary = grouped.groupby("bin", observed=True)["target"].agg(
        ["count", "sum"]
    )
    summary.columns = ["total", "bad"]
    summary["good"] = summary["total"] - summary["bad"]

    # Laplace smoothing (+0.5) so bins with zero bads/goods don't break log()
    summary["pct_good"] = (summary["good"] + 0.5) / (total_good + 0.5)
    summary["pct_bad"] = (summary["bad"] + 0.5) / (total_bad + 0.5)
    summary["woe"] = np.log(summary["pct_good"] / summary["pct_bad"])
    summary["iv_contribution"] = (summary["pct_good"] - summary["pct_bad"]) * summary[
        "woe"
    ]

    iv = summary["iv_contribution"].sum()
    return summary, iv


iv_results = {}
woe_maps = {}

for feat in numeric_features:
    summary, iv = compute_woe_iv(df, feat, target)
    iv_results[feat] = iv
    woe_maps[feat] = summary

# AgeBucket categorical version too (for interpretability comparison)
summary_cat, iv_cat = compute_woe_iv(df, "AgeBucket", target, is_categorical=True)
iv_results["AgeBucket"] = iv_cat
woe_maps["AgeBucket"] = summary_cat

iv_table = (
    pd.Series(iv_results, name="IV")
    .sort_values(ascending=False)
    .to_frame()
)


def iv_strength(iv):
    if iv < 0.02:
        return "useless"
    elif iv < 0.1:
        return "weak"
    elif iv < 0.3:
        return "medium"
    elif iv < 0.5:
        return "strong"
    else:
        return "suspicious (check leak)"


iv_table["strength"] = iv_table["IV"].apply(iv_strength)
print(iv_table)
iv_table.to_csv("outputs/iv_table.csv")

# Show the WOE bins for the single most predictive feature, for interpretability
top_feature = iv_table.index[0]
print(f"\n--- WOE bins for top feature: {top_feature} ---")
print(
    woe_maps[top_feature][["total", "bad", "good", "woe"]].round(3)
)
