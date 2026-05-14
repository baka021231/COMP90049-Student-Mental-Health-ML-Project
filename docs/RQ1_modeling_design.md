# RQ1 Modeling Design Document

## 1. Purpose

This document defines the modeling workflow for RQ1:

> Can wearable sensor features predict student stress level?

In Chinese:

> 只使用可穿戴设备数据，能否预测学生当天的压力等级？

The goal of RQ1 is not only to obtain a high score, but to build a correct and reproducible machine learning experiment. The experiment should show whether sleep, activity, HRV, and SpO2 features contain useful information for predicting self-reported student stress.

## 2. Research Question

### 2.1 Main Question

RQ1 investigates whether wearable-derived features can classify student stress into three levels:

- Low
- Medium
- High

### 2.2 Machine Learning Task

This is a supervised multi-class classification task.

In simple terms:

- Input: wearable features for one student on one day
- Output: predicted stress label for that student-day

The target column is:

```text
stress_label
```

The label rule currently used in the project is:

```text
Low: stress <= 17
Medium: 18 <= stress <= 38
High: stress >= 39
```

This threshold must be kept consistent between code, results, and report writing.

## 3. Scope

### 3.1 In Scope

RQ1 includes:

- Reading the cleaned modeling table
- Using the fixed train/test split
- Training baseline and machine learning models
- Evaluating models on the test set
- Comparing model performance
- Saving result tables and confusion matrices
- Writing a short summary for the report

### 3.2 Out of Scope

RQ1 does not include:

- Comparing separate feature groups such as sleep-only or activity-only
- Adding temporal feature variants such as lag or rolling features
- Using anxiety as an input feature
- Using Fitbit's own stress score as an input feature
- Creating a new train/test split

Those belong to later research questions or auxiliary analysis.

## 4. Input Data

### 4.1 Required Files

The RQ1 script should read:

```text
modeling_outputs/clean_model_data.csv
modeling_outputs/feature_sets.json
```

### 4.2 Required Columns

The data must contain:

```text
stress_label
split
```

The `split` column must contain:

```text
train
test
```

### 4.3 RQ1 Feature Set

RQ1 should use the feature set named:

```text
rq1_all_wearable
```

This feature set currently contains:

```text
sleep_score
deep_sleep_minutes
total_steps
sedentary_minutes
lightly_active_minutes
moderately_active_minutes
very_active_minutes
avg_rmssd
avg_low_frequency
avg_high_frequency
avg_oxygen
std_oxygen
```

These features represent:

| Group | Features |
| --- | --- |
| Sleep | sleep_score, deep_sleep_minutes |
| Activity | total_steps, sedentary_minutes, lightly_active_minutes, moderately_active_minutes, very_active_minutes |
| HRV | avg_rmssd, avg_low_frequency, avg_high_frequency |
| SpO2 | avg_oxygen, std_oxygen |

## 5. Excluded Columns

The following columns must not be used as model inputs:

```text
student_id
date
stress
stress_label
anxiety
STRESS_SCORE
CALCULATION_FAILED
```

Reasons:

| Column | Reason for Exclusion |
| --- | --- |
| student_id | The model may memorize individual students instead of learning general patterns. |
| date | Raw date strings should not be directly used in RQ1. |
| stress | This is the numeric source of the target label. Using it would be data leakage. |
| stress_label | This is the answer to be predicted. |
| anxiety | It is a same-day questionnaire variable and is highly related to stress. |
| STRESS_SCORE | It is Fitbit's own stress estimate and overlaps conceptually with the target. |
| CALCULATION_FAILED | It is metadata related to Fitbit stress score calculation. |

Data leakage means the model receives information that would not be available in a real prediction scenario. It is similar to giving the answer to the model during the exam.

## 6. Train/Test Split Design

RQ1 must use the existing `split` column from:

```text
modeling_outputs/clean_model_data.csv
```

Do not create a new random split.

Current split summary:

| Split | Students | Rows |
| --- | ---: | ---: |
| Train | 24 | 1097 |
| Test | 8 | 287 |

This split is subject-aware, meaning one student appears only in train or only in test.

This matters because if the same student appears in both train and test, the model may remember that student's pattern. A subject-aware split better tests whether the model can generalize to unseen students.

## 7. Modeling Plan

### 7.1 Baseline Model

The first model should be a majority-class baseline.

This baseline always predicts the most common class in the training set.

Purpose:

- Provides the minimum performance that real models should beat
- Helps judge whether wearable features provide useful predictive signal

### 7.2 Candidate Models

RQ1 should train the following models:

| Model | Purpose |
| --- | --- |
| Majority Baseline | Simple reference point |
| Logistic Regression | Simple and interpretable classifier |
| SVM | Strong model for small or medium-sized tabular data |
| Random Forest | Captures non-linear feature interactions |
| MLP | Small neural network for comparison with more complex learning |

### 7.3 Recommended Pipelines

Some models need feature scaling because they are sensitive to feature magnitude.

| Model | Pipeline |
| --- | --- |
| Logistic Regression | StandardScaler + LogisticRegression |
| SVM | StandardScaler + SVC |
| MLP | StandardScaler + MLPClassifier |
| Random Forest | RandomForestClassifier |
| Majority Baseline | DummyClassifier |

Feature scaling should be fitted only on the training data. The same fitted scaler is then applied to the test data.

## 8. Training Workflow

The RQ1 script should follow this workflow:

1. Load `clean_model_data.csv`.
2. Load `feature_sets.json`.
3. Read the `rq1_all_wearable` feature list.
4. Split rows by the existing `split` column.
5. Build `X_train`, `y_train`, `X_test`, and `y_test`.
6. Train each model on `X_train` and `y_train`.
7. Predict labels for `X_test`.
8. Evaluate predictions against `y_test`.
9. Save metrics, classification reports, and confusion matrices.
10. Write a short Markdown summary.

Pseudo-code:

```text
data = read clean_model_data.csv
features = read rq1_all_wearable from feature_sets.json

train = rows where split == "train"
test = rows where split == "test"

X_train = train[features]
y_train = train["stress_label"]

X_test = test[features]
y_test = test["stress_label"]

for each model:
    train model on X_train, y_train
    predict y_pred on X_test
    evaluate y_test vs y_pred
    save results
```

## 9. Hyperparameter Strategy

### 9.1 Minimum Version

The minimum acceptable version should use reasonable default settings for all models.

This version is enough to answer the basic RQ1 question.

### 9.2 Recommended Version

The recommended version should perform a small grid search inside the training set.

Possible parameter grids:

| Model | Parameters |
| --- | --- |
| Logistic Regression | C = 0.1, 1, 10 |
| SVM | C = 0.1, 1, 10; kernel = linear, rbf |
| Random Forest | n_estimators = 100, 300; max_depth = None, 5, 10 |
| MLP | hidden_layer_sizes = (32,), (64,), (32, 16); alpha = 0.0001, 0.001 |

The test set must not be used for hyperparameter selection.

Recommended validation method:

```text
GridSearchCV on training data only
```

If time is limited, default models are acceptable, but the report should state that only limited tuning was performed.

## 10. Evaluation Metrics

RQ1 should not rely only on accuracy.

The main metric should be:

```text
macro-F1
```

Macro-F1 gives equal importance to Low, Medium, and High classes. This is useful because the class distribution is not perfectly balanced.

Required metrics:

| Metric | Purpose |
| --- | --- |
| Accuracy | Overall percentage of correct predictions |
| Macro Precision | Average precision across classes |
| Macro Recall | Average recall across classes |
| Macro-F1 | Main metric for comparing models |
| Per-class F1 | Shows which stress level is hardest to predict |
| Confusion Matrix | Shows common prediction mistakes |

Expected result table:

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | Low F1 | Medium F1 | High F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority Baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SVM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MLP | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 11. Expected Outputs

The RQ1 modeling script should generate:

```text
modeling_outputs/rq1_results.csv
modeling_outputs/rq1_classification_report.csv
modeling_outputs/rq1_confusion_matrix_majority_baseline.csv
modeling_outputs/rq1_confusion_matrix_logistic_regression.csv
modeling_outputs/rq1_confusion_matrix_svm.csv
modeling_outputs/rq1_confusion_matrix_random_forest.csv
modeling_outputs/rq1_confusion_matrix_mlp.csv
modeling_outputs/rq1_summary.md
```

Recommended script path:

```text
scripts/run_rq1_models.py
```

The script should be runnable from the project root:

```bash
python scripts/run_rq1_models.py
```

## 12. Report Writing Plan

The RQ1 section in the final report should include:

### 12.1 Research Question

State that RQ1 tests whether wearable features can predict student stress level.

### 12.2 Method

Describe:

- Input features: sleep, activity, HRV, and SpO2
- Target: Low, Medium, High stress label
- Train/test split: subject-aware split
- Models: baseline, Logistic Regression, SVM, Random Forest, MLP
- Main metric: macro-F1

### 12.3 Results

Include:

- One result table
- One confusion matrix for the best model or most interpretable model

### 12.4 Discussion

Discuss:

- Whether models beat the majority baseline
- Which stress class is hardest to predict
- Whether wearable features alone seem sufficient
- Why results may be limited by small number of students

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Models perform only slightly better than baseline | Results may look weak | Emphasize that wearable-only prediction is difficult and discuss limitations honestly |
| Medium class is hard to predict | Macro-F1 may be low | Use confusion matrix and per-class F1 to explain the issue |
| MLP overfits | Test performance may be unstable | Keep the MLP small and use regularization |
| Test set is small | Results may vary depending on selected students | Mention subject-level sample size as a limitation |
| Hyperparameter tuning takes too long | Delays later RQs | Start with default models, then add small grid search only if time allows |

## 14. Time Estimate

| Task | Estimated Time |
| --- | ---: |
| Create RQ1 script structure | 30 minutes |
| Load data and feature sets | 30 minutes |
| Implement baseline and models | 1 to 2 hours |
| Add pipelines and metrics | 1 to 2 hours |
| Add optional grid search | 2 to 3 hours |
| Save result files | 1 hour |
| Check outputs and debug | 1 to 2 hours |
| Write report-ready summary | 2 to 4 hours |

Overall estimate:

| Version | Time |
| --- | ---: |
| Minimum version | Half day |
| Solid version | 1 day |
| Tuned and report-ready version | 1 to 1.5 days |

## 15. Recommended Implementation Order

1. Implement baseline and default models first.
2. Save metrics and confusion matrices.
3. Check whether real models beat the baseline.
4. Add small grid search if time allows.
5. Generate `rq1_summary.md`.
6. Transfer the key result table and discussion into the final report.

## 16. Definition of Done

RQ1 is considered complete when:

- `scripts/run_rq1_models.py` can run from the project root.
- It uses `modeling_outputs/clean_model_data.csv`.
- It uses `rq1_all_wearable` from `feature_sets.json`.
- It uses the existing `split` column.
- It trains at least one baseline and three real models.
- It reports accuracy, macro-F1, and per-class F1.
- It saves result files under `modeling_outputs/`.
- The best model and main findings can be clearly explained in the report.
