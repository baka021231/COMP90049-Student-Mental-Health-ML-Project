# SSAQS 原始数据审计

## 目的

本报告记录 `SSAQS dataset/` 原始目录在任何建模预处理之前的数据结构、覆盖率和缺失情况。

## 数据规模

- 学生目录数：35
- 拥有全部预期学生文件的学生数：32
- daily questionnaire 中有目标标签的 student-day 总数：3118
- 同一天同时有完整 wearable 文件覆盖的目标天数：1489 (47.75%)

## 预期文件数量

| file | student_file_count |
| --- | --- |
| activity_level.csv | 32 |
| daily_questions.csv | 35 |
| hrv.csv | 32 |
| oxygen.csv | 32 |
| sleep.csv | 32 |
| steps.csv | 32 |
| stress.csv | 32 |

## 缺失的预期文件

| file | missing_student_count |
| --- | --- |
| activity_level.csv | 3 |
| hrv.csv | 3 |
| oxygen.csv | 3 |
| sleep.csv | 3 |
| steps.csv | 3 |
| stress.csv | 3 |

## 相对于问卷目标天的覆盖率

| file | total_target_days | covered_target_days | mean_student_coverage_percent | overall_target_coverage_percent |
| --- | --- | --- | --- | --- |
| activity_level.csv | 3118 | 2257 | 69.20 | 72.39 |
| daily_questions.csv | 3118 | 3118 | 100.00 | 100.00 |
| hrv.csv | 3118 | 1637 | 48.99 | 52.50 |
| oxygen.csv | 3118 | 2009 | 61.16 | 64.43 |
| sleep.csv | 3118 | 1508 | 44.79 | 48.36 |
| steps.csv | 3118 | 2232 | 68.49 | 71.58 |
| stress.csv | 3118 | 1466 | 43.68 | 47.02 |

## 完整 wearable 覆盖率最低的学生

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

## Daily Questionnaire 审计

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

## 对后续建模的影响

- 原始数据存在明显的 wearable 缺失，不能在 train/test 划分前直接做全局标准化。
- 为避免 preprocessing leakage，后续建模应先按学生划分 train/test，再在训练折内部拟合填补和标准化步骤。
- 对于缺失 wearable 文件或覆盖率很低的学生，需要在报告中作为 limitation 说明，或使用明确且可复现的过滤规则处理。
