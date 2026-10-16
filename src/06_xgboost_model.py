"""
Step 11-12: Gradient-boosted tree model (XGBoost) + class imbalance handling.

WHY A SECOND MODEL AT ALL?
The logistic regression scorecard is interpretable and auditable, but it can
only capture additive, linear-in-WOE relationships. A tree-based model can
capture interactions (e.g. "high utilization is only risky when income is
ALSO low") that a scorecard structurally cannot. Comparing the two tells us
how much predictive power we're giving up for interpretability - a real,
quantified trade-off, not a vague claim.

CLASS IMBALANCE (step 12):
Only ~6.7% of the training data are defaults. A tree model trained naively
will happily predict "no default" most of the time and still look accurate.
We fix this with scale_pos_weight = (negatives / positives), which tells the
model to treat each default example as worth that many times more during
training - it's the tree-model equivalent of class_weight='balanced'.

ENVIRONMENT NOTE: this sandbox has no internet access, so the `xgboost`
package can't be installed here. The code below uses real XGBoost when it's
available (which it will be on your machine after `pip install -r
requirements.txt`), and automatically falls back to scikit-learn's
HistGradientBoostingClassifier - a very similar gradient-boosted-tree model -
only so results can be demonstrated in this sandbox. Don't be surprised if
exact numbers shift slightly when you re-run this with real XGBoost locally.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

USING_XGBOOST = True
try:
    import xgboost as xgb
except ImportError:
    USING_XGBOOST = False
    from sklearn.ensemble import HistGradientBoostingClassifier

print("Using XGBoost:" , USING_XGBOOST)

# Reuse the exact same train/test split and raw (non-WOE) features as the
# scorecard model, so the comparison is apples-to-apples. Tree models don't
# need WOE encoding - they can split on raw values directly.
train_raw = pd.read_csv("data/train_raw.csv", index_col=0)
test_raw = pd.read_csv("data/test_raw.csv", index_col=0)
target = "SeriousDlqin2yrs"

feature_cols = [c for c in train_raw.columns if c != target]
X_train, y_train = train_raw[feature_cols], train_raw[target]
X_test, y_test = test_raw[feature_cols], test_raw[target]

# --- Step 12: class imbalance handling ---
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"scale_pos_weight (neg/pos ratio): {scale_pos_weight:.2f}")

if USING_XGBOOST:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
else:
    # HistGradientBoostingClassifier has no scale_pos_weight; we approximate
    # the same "pay more attention to defaults" idea with sample_weight.
    sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )

if USING_XGBOOST:
    model.fit(X_train, y_train)
else:
    model.fit(X_train, y_train, sample_weight=sample_weight)

pd_train_raw = model.predict_proba(X_train)[:, 1]
pd_test_raw = model.predict_proba(X_test)[:, 1]

print(
    f"\n[BEFORE CALIBRATION] Mean predicted PD - train: {pd_train_raw.mean():.4f}, "
    f"test: {pd_test_raw.mean():.4f}"
)
print(f"Actual default rate - train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")

# --- CALIBRATION (important, often-skipped step) ---
# Using scale_pos_weight / sample_weight to fix class imbalance improves the
# model's ability to RANK risky vs safe applicants (what AUC/KS/Gini measure)
# but distorts the actual probability values - they get pulled toward the
# weighted class, as the mismatch above shows (predicted ~32% average default
# vs a true ~6.7%). A probability that's off by 5x is useless as a real PD
# (e.g. for Expected Loss math later), even if the ranking is fine.
#
# Fix: Platt scaling - fit a simple 1-feature logistic regression that maps
# the model's raw score back onto the true probability scale, learned on
# the training set only.
from sklearn.linear_model import LogisticRegression as _PlattLR

logit_train = np.log(np.clip(pd_train_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(pd_train_raw, 1e-6, 1 - 1e-6)))
logit_test = np.log(np.clip(pd_test_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(pd_test_raw, 1e-6, 1 - 1e-6)))

platt = _PlattLR()
platt.fit(logit_train.reshape(-1, 1), y_train)

pd_train = platt.predict_proba(logit_train.reshape(-1, 1))[:, 1]
pd_test = platt.predict_proba(logit_test.reshape(-1, 1))[:, 1]

print(
    f"[AFTER CALIBRATION]  Mean predicted PD - train: {pd_train.mean():.4f}, "
    f"test: {pd_test.mean():.4f}"
)

# Feature importance (works for both XGBoost and HistGradientBoosting via
# a consistent attribute name where possible)
if USING_XGBOOST:
    importances = model.feature_importances_
else:
    # sklearn's HGB doesn't expose feature_importances_ the same way;
    # use permutation importance as a robust substitute
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, scoring="roc_auc"
    )
    importances = r.importances_mean

imp_table = (
    pd.Series(importances, index=feature_cols, name="importance")
    .sort_values(ascending=False)
)
print("\nFeature importance:")
print(imp_table)
imp_table.to_csv("outputs/xgboost_feature_importance.csv")

xgb_bundle = {
    "model": model,
    "platt": platt,
    "features": feature_cols,
    "using_xgboost": USING_XGBOOST,
}
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_bundle, "outputs/xgboost_model.joblib")
# Keep the app's deployed model path in sync whenever the pipeline is retrained.
joblib.dump(xgb_bundle, "models/xgboost_model.joblib")
pd.Series(pd_test, index=X_test.index, name="pd_xgboost").to_csv(
    "outputs/xgboost_pd_test.csv"
)
print("\nSaved model -> outputs/xgboost_model.joblib and models/xgboost_model.joblib")
