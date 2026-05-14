# RQ1 Modeling Summary

## Research Question

RQ1 asks whether wearable-derived sleep, activity, HRV, and SpO2 features can predict student stress level.

## Data

- Feature set: `rq1_all_wearable`
- Number of wearable features: 12
- Train rows: 1097
- Test rows: 287
- Train label counts: {'Low': 273, 'Medium': 406, 'High': 418}
- Test label counts: {'Low': 86, 'Medium': 85, 'High': 116}

## Models

- Majority baseline
- Logistic Regression
- SVM
- Random Forest
- MLP

## Main Results

| model | accuracy | macro_precision | macro_recall | macro_f1 | low_f1 | medium_f1 | high_f1 | low_support | medium_support | high_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.411 | 0.410 | 0.409 | 0.409 | 0.426 | 0.352 | 0.448 | 86 | 85 | 116 |
| svm | 0.366 | 0.364 | 0.360 | 0.361 | 0.368 | 0.275 | 0.440 | 86 | 85 | 116 |
| logistic_regression | 0.362 | 0.353 | 0.352 | 0.352 | 0.337 | 0.282 | 0.437 | 86 | 85 | 116 |
| random_forest | 0.324 | 0.341 | 0.322 | 0.317 | 0.275 | 0.333 | 0.343 | 86 | 85 | 116 |
| majority_baseline | 0.404 | 0.135 | 0.333 | 0.192 | 0.000 | 0.000 | 0.576 | 86 | 85 | 116 |

## Key Findings

- Best model by macro-F1: `mlp`.
- Best macro-F1: 0.409.
- Majority baseline macro-F1: 0.192.
- Majority baseline accuracy: 0.404.
- The hardest class for the best model is `Medium` by per-class F1.
- Accuracy alone is not sufficient here because the majority baseline has competitive accuracy but weak macro-F1.

## Confusion Matrix for Best Model: `mlp`

| actual | predicted_Low | predicted_Medium | predicted_High |
| --- | --- | --- | --- |
| actual_Low | 36 | 25 | 25 |
| actual_Medium | 21 | 32 | 32 |
| actual_High | 26 | 40 | 50 |

## Report Note

The initial RQ1 result suggests that wearable-only features provide limited but measurable predictive signal. The best model improves macro-F1 over the majority baseline, but the overall performance remains modest, so the report should discuss the difficulty of predicting self-reported stress from passive wearable data alone.
