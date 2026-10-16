"""
Step 13-16: Evaluate both models with the metrics credit risk actually uses.

WHY NOT JUST ACCURACY:
With a 6.7% default rate, a model that predicts "no one defaults" scores
~93% accuracy while catching zero real risk. So we use:

- AUC-ROC: probability that a randomly chosen defaulter gets a HIGHER risk
  score than a randomly chosen non-defaulter. 0.5 = random guessing, 1.0 =
  perfect separation.
- Gini coefficient: a simple rescale of AUC used throughout banking:
  Gini = 2*AUC - 1. Same information, different convention/audience.
- KS (Kolmogorov-Smirnov) statistic: the MAXIMUM gap between the cumulative
  % of defaulters caught and the cumulative % of non-defaulters caught, as
  you move down a list sorted by risk score. It answers "at the single best
  possible cutoff, how well does this model separate good from bad?" Banks
  often care about this more than AUC because it's tied to an actual
  decision threshold.
- Confusion matrix (at a chosen threshold): how many actual defaulters we'd
  catch vs miss, and how many safe applicants we'd wrongly flag.
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

test_raw = pd.read_csv("data/test_raw.csv", index_col=0)
y_test = test_raw["SeriousDlqin2yrs"]

pd_scorecard = pd.read_csv("outputs/scorecard_pd_test.csv", index_col=0)["pd_scorecard"]
pd_xgb = pd.read_csv("outputs/xgboost_pd_test.csv", index_col=0)["pd_xgboost"]


def ks_statistic(y_true, y_score):
    """Max separation between cumulative good% and cumulative bad%."""
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values(
        "score", ascending=False
    )
    df["cum_bad"] = (df["y"] == 1).cumsum() / (df["y"] == 1).sum()
    df["cum_good"] = (df["y"] == 0).cumsum() / (df["y"] == 0).sum()
    ks = (df["cum_bad"] - df["cum_good"]).abs().max()
    return ks


results = {}
for name, scores in [("Logistic Scorecard", pd_scorecard), ("XGBoost (gradient-boosted trees)", pd_xgb)]:
    auc = roc_auc_score(y_test, scores)
    gini = 2 * auc - 1
    ks = ks_statistic(y_test, scores)
    results[name] = {"AUC": auc, "Gini": gini, "KS": ks}

results_df = pd.DataFrame(results).T
print(results_df.round(4))
results_df.to_csv("outputs/model_comparison_metrics.csv")

# --- Confusion matrix at a business-relevant threshold ---
# We don't use 0.5 (that would flag almost nobody, since defaults are rare).
# Instead, flag the riskiest 10% of applicants for manual review/decline -
# a realistic policy choice a bank would actually set based on capacity.
threshold_xgb = pd_xgb.quantile(0.90)
pred_flag = (pd_xgb >= threshold_xgb).astype(int)
cm = confusion_matrix(y_test, pred_flag)
print(f"\nConfusion matrix (XGBoost, flagging top 10% riskiest, threshold={threshold_xgb:.4f}):")
print(pd.DataFrame(cm, index=["Actual: No Default", "Actual: Default"],
                    columns=["Predicted: Safe", "Predicted: Flagged"]))

tn, fp, fn, tp = cm.ravel()
print(f"\nOf {tp+fn} actual defaulters, we caught {tp} ({tp/(tp+fn):.1%}) by reviewing only the top 10% riskiest applicants.")

# --- Plots: ROC curves + KS curve ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, scores in [("Logistic Scorecard", pd_scorecard), ("XGBoost", pd_xgb)]:
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

df_ks = pd.DataFrame({"y": y_test, "score": pd_xgb}).sort_values("score", ascending=False).reset_index(drop=True)
df_ks["cum_bad"] = (df_ks["y"] == 1).cumsum() / (df_ks["y"] == 1).sum()
df_ks["cum_good"] = (df_ks["y"] == 0).cumsum() / (df_ks["y"] == 0).sum()
df_ks["pct_population"] = (df_ks.index + 1) / len(df_ks)
axes[1].plot(df_ks["pct_population"], df_ks["cum_bad"], label="Cumulative % Defaulters Caught")
axes[1].plot(df_ks["pct_population"], df_ks["cum_good"], label="Cumulative % Non-Defaulters")
ks_idx = (df_ks["cum_bad"] - df_ks["cum_good"]).abs().idxmax()
axes[1].axvline(df_ks.loc[ks_idx, "pct_population"], color="gray", linestyle="--",
                label=f"KS = {results['XGBoost (gradient-boosted trees)']['KS']:.3f}")
axes[1].set_xlabel("% of population (sorted riskiest first)")
axes[1].set_ylabel("Cumulative %")
axes[1].set_title("KS Curve (XGBoost)")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/07_evaluation_curves.png", dpi=110)
print("\nSaved -> outputs/07_evaluation_curves.png")
