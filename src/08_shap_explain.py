"""
Step 17: SHAP explanations for the XGBoost model.

WHAT SHAP ACTUALLY TELLS YOU:
For any single prediction, SHAP breaks the model's output down into "how
many percentage points of risk did each feature add or subtract, starting
from the average applicant." This is what makes a black-box tree model
usable in a real lending decision - you can hand an applicant (or a
regulator) a plain-language reason for their score, e.g. "+4.2 points of
risk from high utilization, -1.1 points from a long credit history."

NOTE: this sandbox has no internet access, so the `shap` package could not
be installed/run here. This script is complete and correct - run it on your
own machine (`pip install -r requirements.txt`) after the other scripts have
produced outputs/xgboost_model.joblib.
"""

import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap  # pip install shap

bundle = joblib.load("outputs/xgboost_model.joblib")
model = bundle["model"]
features = bundle["features"]

test_raw = pd.read_csv("data/test_raw.csv", index_col=0)
X_test = test_raw[features]

# TreeExplainer is fast and exact for tree-based models like XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 1. Global importance: which features matter most, on average, across all
#    applicants (a bar/beeswarm summary)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", dpi=110)
plt.close()

# 2. Local explanation: why did ONE specific applicant get their score?
#    (useful for the Streamlit app's "why was I scored this way" feature)
sample_idx = 0
plt.figure()
shap.force_plot(
    explainer.expected_value,
    shap_values[sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=True,
    show=False,
)
plt.tight_layout()
plt.savefig("outputs/shap_single_applicant.png", dpi=110)
plt.close()

print("Saved outputs/shap_summary.png and outputs/shap_single_applicant.png")


def explain_applicant(applicant_df):
    """
    Returns a sorted list of (feature, shap_contribution) for one applicant,
    for use in the Streamlit app's explanation panel.
    applicant_df: a single-row DataFrame with the same columns as `features`.
    """
    sv = explainer.shap_values(applicant_df[features])
    contributions = list(zip(features, sv[0]))
    return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
