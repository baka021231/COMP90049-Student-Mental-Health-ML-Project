# Raw SSAQS Data Audit

## Purpose

This audit records the structure, coverage, and missingness of the original `SSAQS dataset/` directory before any modeling preprocessing.

## Dataset Scale

- Student directories: 35
- Students with all expected per-student files: 32
- Total target days from daily questionnaires: 3118
- Target days with complete wearable file coverage: 1482 (47.53%)

## Expected File Counts

| file | student_file_count |
| --- | --- |
| activity_level.csv | 32 |
| daily_questions.csv | 35 |
| hrv.csv | 32 |
| oxygen.csv | 32 |
| sleep.csv | 32 |
| steps.csv | 32 |
| stress.csv | 32 |

## Missing Expected Files

| file | missing_student_count |
| --- | --- |
| activity_level.csv | 3 |
| hrv.csv | 3 |
| oxygen.csv | 3 |
| sleep.csv | 3 |
| steps.csv | 3 |
| stress.csv | 3 |

## Coverage Against Daily Questionnaire Target Days

| file | total_target_days | covered_target_days | mean_student_coverage_percent | overall_target_coverage_percent |
| --- | --- | --- | --- | --- |
| activity_level.csv | 3118 | 2257 | 69.20 | 72.39 |
| daily_questions.csv | 3118 | 3118 | 100.00 | 100.00 |
| hrv.csv | 3118 | 1637 | 48.99 | 52.50 |
| oxygen.csv | 3118 | 2009 | 61.16 | 64.43 |
| sleep.csv | 3118 | 1501 | 44.61 | 48.14 |
| steps.csv | 3118 | 2232 | 68.49 | 71.58 |
| stress.csv | 3118 | 1466 | 43.68 | 47.02 |

## Students With Lowest Complete Wearable Coverage

| student_id | target_days | complete_wearable_target_days | complete_wearable_target_percent | has_all_expected_files |
| --- | --- | --- | --- | --- |
| 3 | 14 | 0 | 0.00 | 0 |
| 12 | 55 | 0 | 0.00 | 0 |
| 14 | 39 | 0 | 0.00 | 0 |
| 2 | 102 | 9 | 8.82 | 1 |
| 4 | 118 | 13 | 11.02 | 1 |
| 17 | 116 | 13 | 11.21 | 1 |
| 31 | 50 | 8 | 16.00 | 1 |
| 1 | 121 | 30 | 24.79 | 1 |
| 23 | 76 | 22 | 28.95 | 1 |
| 21 | 59 | 18 | 30.51 | 1 |

## Daily Questionnaire Audit

| metric | value |
| --- | --- |
| daily_question_rows | 3133 |
| daily_question_students | 35 |
| daily_question_unique_student_days | 3118 |
| daily_question_duplicate_student_day_pairs | 15 |
| has_stress_label_column | False |
| stress_label_mismatches_under_current_rule | N/A |
| stress_min | 0.0 |
| stress_max | 100.0 |
| stress_missing_rows | 0 |
| anxiety_missing_rows | 0 |

## Implications for Modeling

- The original data contains substantial wearable missingness and should not be globally standardized before splitting.
- A leakage-safe modeling pipeline should split by student first, then fit imputation and scaling only within the training folds.
- Students with missing wearable files or very low coverage should be explicitly discussed as a limitation or handled by a documented filtering rule.
