# Credit Risk Model — Review and Validation Notes

## Fixes applied

1. **WOE transform compatibility:** `WOEBinner.transform()` now converts mapped categorical WOE values to numeric before filling unseen bins with `0.0`. This fixes a runtime error with pandas categorical data.
2. **Consistent model paths:** the scorecard and tree-model training scripts now save model bundles to both `outputs/` and `models/`. The Streamlit app loads from `models/`, so rerunning training no longer leaves the app using stale model files.
3. **Accurate model label in app:** the app caption identifies whether the saved tree model is XGBoost or scikit-learn HistGradientBoosting.
4. **Training-only IV report:** the standalone IV report now uses the same deterministic, stratified 80/20 split as the model scripts and computes IV from the training portion only.

## Checks performed

- Python syntax compilation passed for all source scripts and the Streamlit app.
- Pipeline steps 01–07 ran successfully on the included dataset after the WOE fix.
- Scorecard and XGBoost model bundles were regenerated and saved to both expected locations.
- A sample applicant was passed through both saved model bundles successfully.
- On the fixed pipeline run, the held-out test metrics were approximately:
  - Logistic scorecard: AUC 0.8629, Gini 0.7259, KS 0.5756.
  - XGBoost: AUC 0.8709, Gini 0.7417, KS 0.5823.
- The Streamlit UI itself was not launched in this environment because Streamlit is not installed here. The model-loading and inference logic was tested separately.

## Remaining limitations to be aware of

1. **Preprocessing leakage:** cleaning parameters such as percentile caps and imputation medians are currently estimated before the train/test split. These steps do not use the target label, but a more rigorous pipeline would estimate preprocessing parameters on training data only and apply them unchanged to test data.
2. **Probability calibration:** Platt scaling is fitted to predictions from the same data used to train the tree model. For a more robust calibration estimate, use a separate calibration/validation split or out-of-fold predictions.
3. **Missingness flags:** the cleaning script creates missingness indicators, but the current model feature lists do not include them. Either include and validate these indicators or adjust the comments to avoid implying the model uses them.
4. **Decision thresholds:** the app's approve/review/decline rules are illustrative project thresholds, not a validated lending policy. This project is educational and should not be used for real credit decisions.
5. **Dataset scope:** the dataset is historical US consumer-credit data. The results should not be assumed to generalize to current Indian applicants or other populations without additional data and validation.
