# Model Data Cleaning Report

## Purpose

This module prepares a reproducible modelling table from `final_student_day_table_v01_processed.csv`.
The processed CSV is useful for modelling because its main wearable features have already been
completed and standardised, but it still needs a final modelling-specific cleaning layer.

## Inputs

- `final_student_day_table_v01_processed.csv`: primary modelling source.
- `final_student_day_table_v01.csv`: raw merged student-day table, used for context and auditing only.

## Cleaning Principles

1. Keep source CSV files unchanged.
2. Resolve duplicate `student_id + date` rows before any split or modelling.
3. Recreate `stress_label` from the numeric `stress` score so the label rule is explicit.
4. Exclude leakage-prone columns from model feature sets.
5. Create all RQ-specific feature sets from the same cleaned table.
6. Use subject-aware train/test splits so the same student cannot appear in both sets.

## Planned Outputs

- `clean_model_data.csv`: cleaned student-day modelling table.
- `feature_sets.json`: feature columns for RQ1, RQ2, and RQ3.
- `split_assignments.csv`: train/test assignment by student.
- `cleaning_audit.json`: row counts, duplicate handling, class counts, and split summary.

## Workflow

The cleaning pipeline is implemented in `scripts/prepare_model_data.py`.

Run it from the project root:

```bash
python3 scripts/prepare_model_data.py
```

The script performs the following steps:

1. Loads `final_student_day_table_v01_processed.csv`.
2. Converts `date` to datetime and `student_id` to string.
3. Resolves duplicate `student_id + date` rows by averaging numeric columns.
4. Rebuilds `stress_label` from `stress` using:
   - Low: `stress <= 17`
   - Medium: `18 <= stress <= 38`
   - High: `stress >= 39`
5. Adds temporal features:
   - `semester_week`
   - `day_of_week`
   - `is_weekend`
   - one-observation lag features for sleep score, total steps, and RMSSD
   - 3-observation and 7-observation rolling means using past observations only
6. Defines feature sets for RQ1, RQ2, and RQ3.
7. Creates a subject-aware train/test split using `GroupShuffleSplit`.
8. Writes all modelling artefacts into `modeling_outputs/`.

## Generated Outputs

- `clean_model_data.csv`: cleaned student-day table with temporal features and a `split` column.
- `feature_sets.json`: standard feature sets for all RQs.
- `split_assignments.csv`: student-level train/test assignments.
- `cleaning_audit.json`: machine-readable cleaning audit.

## Cleaning Results

| Item | Value |
| --- | ---: |
| Source rows | 1,394 |
| Source students | 32 |
| Duplicate `student_id + date` rows | 10 |
| Clean rows | 1,384 |
| Clean students | 32 |
| Train students | 24 |
| Test students | 8 |
| Train rows | 1,097 |
| Test rows | 287 |

Clean label distribution:

| Label | Count |
| --- | ---: |
| Low | 359 |
| Medium | 491 |
| High | 534 |

Split label distribution:

| Split | Low | Medium | High | Rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 273 | 406 | 418 | 1,097 |
| Test | 86 | 85 | 116 | 287 |

## Feature Sets

RQ1 uses all wearable features:

- sleep features
- activity features
- HRV features
- SpO2 features

RQ2 uses feature-group ablation:

- `sleep_only`
- `activity_only`
- `hrv_spo2_only`
- `all_wearable`

RQ3 uses temporal variants:

- `no_temporal`
- `semester_week`
- `calendar`
- `lag_1`
- `rolling_3`
- `rolling_7`

## Leakage Policy

The following columns must not be used as model inputs:

- `student_id`
- `date`
- `stress`
- `stress_label`
- `anxiety`
- `STRESS_SCORE`
- `CALCULATION_FAILED`

`anxiety` is excluded because it is a same-day questionnaire variable closely related to the target.
`STRESS_SCORE` is excluded because it is Fitbit's own stress estimate and may overlap conceptually
with the prediction target.

## Notes for Modelling

- Use the existing `split` column in `clean_model_data.csv`; do not create a new split per model.
- For RQ1 and RQ2, the core wearable features contain no missing values.
- For RQ3, lag and rolling features have 32 missing values, corresponding to the first observation
  for each student. Handle these inside the modelling pipeline, preferably with train-set-only median
  imputation.
- Logistic Regression, SVM, and MLP should use scaling inside a pipeline.
- Random Forest does not require scaling, but using the same train/test split is still mandatory.

## Current Status

Data cleaning module complete. The next step is to build modelling scripts that consume
`clean_model_data.csv`, `feature_sets.json`, and the fixed `split` column.
