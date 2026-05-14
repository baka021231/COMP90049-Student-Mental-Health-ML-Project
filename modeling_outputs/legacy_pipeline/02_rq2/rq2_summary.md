# RQ2 Feature Group Summary

## Research Question

RQ2 asks which wearable feature groups contribute most to student stress prediction.

## Experimental Design

- The same subject-aware train/test split is used as RQ1.
- Each feature group is evaluated with the full RQ1 model set.
- Feature groups tested: sleep_only, activity_only, hrv_spo2_only, all_wearable.
- Main comparison metric: macro-F1.

## Feature Groups

- `sleep_only`: 2 features
- `activity_only`: 5 features
- `hrv_spo2_only`: 5 features
- `all_wearable`: 12 features

## Best Model by Feature Group

| feature_group | n_features | model | accuracy | macro_f1 | low_f1 | medium_f1 | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_wearable | 12 | mlp | 0.411 | 0.409 | 0.426 | 0.352 | 0.448 |
| hrv_spo2_only | 5 | logistic_regression | 0.401 | 0.401 | 0.439 | 0.344 | 0.421 |
| activity_only | 5 | random_forest | 0.401 | 0.386 | 0.346 | 0.344 | 0.467 |
| sleep_only | 2 | random_forest | 0.359 | 0.354 | 0.287 | 0.404 | 0.370 |

## Full Result Table

| feature_group | n_features | model | accuracy | macro_f1 | low_f1 | medium_f1 | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| activity_only | 5 | random_forest | 0.401 | 0.386 | 0.346 | 0.344 | 0.467 |
| activity_only | 5 | svm | 0.345 | 0.336 | 0.298 | 0.304 | 0.405 |
| activity_only | 5 | mlp | 0.338 | 0.312 | 0.215 | 0.306 | 0.415 |
| activity_only | 5 | logistic_regression | 0.317 | 0.297 | 0.199 | 0.306 | 0.388 |
| activity_only | 5 | majority_baseline | 0.404 | 0.192 | 0.000 | 0.000 | 0.576 |
| all_wearable | 12 | mlp | 0.411 | 0.409 | 0.426 | 0.352 | 0.448 |
| all_wearable | 12 | svm | 0.366 | 0.361 | 0.368 | 0.275 | 0.440 |
| all_wearable | 12 | logistic_regression | 0.362 | 0.352 | 0.337 | 0.282 | 0.437 |
| all_wearable | 12 | random_forest | 0.324 | 0.317 | 0.275 | 0.333 | 0.343 |
| all_wearable | 12 | majority_baseline | 0.404 | 0.192 | 0.000 | 0.000 | 0.576 |
| hrv_spo2_only | 5 | logistic_regression | 0.401 | 0.401 | 0.439 | 0.344 | 0.421 |
| hrv_spo2_only | 5 | svm | 0.401 | 0.388 | 0.468 | 0.298 | 0.398 |
| hrv_spo2_only | 5 | mlp | 0.369 | 0.366 | 0.344 | 0.361 | 0.393 |
| hrv_spo2_only | 5 | random_forest | 0.352 | 0.352 | 0.347 | 0.347 | 0.361 |
| hrv_spo2_only | 5 | majority_baseline | 0.404 | 0.192 | 0.000 | 0.000 | 0.576 |
| sleep_only | 2 | random_forest | 0.359 | 0.354 | 0.287 | 0.404 | 0.370 |
| sleep_only | 2 | svm | 0.373 | 0.342 | 0.268 | 0.291 | 0.466 |
| sleep_only | 2 | logistic_regression | 0.338 | 0.333 | 0.289 | 0.373 | 0.337 |
| sleep_only | 2 | mlp | 0.411 | 0.326 | 0.108 | 0.335 | 0.534 |
| sleep_only | 2 | majority_baseline | 0.404 | 0.192 | 0.000 | 0.000 | 0.576 |

## Key Findings

- Strongest feature group by macro-F1: `all_wearable` using `mlp`.
- Best macro-F1 overall: 0.409.
- Weakest best-performing feature group: `sleep_only`.
- `hrv_spo2_only` is close to `all_wearable`: macro-F1 gap = 0.008.
- This suggests physiological features may carry substantial stress-related signal, while adding all wearable features gives only a small improvement in this run.

## Report Note

RQ2 should be presented as an ablation study. The important point is not only which score is highest, but how much each modality contributes. The current result supports a cautious interpretation: wearable modalities contain some useful signal, but the differences are modest and should be discussed alongside the limited number of students.
