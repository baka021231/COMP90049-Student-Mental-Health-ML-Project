# 严格清洗 Student-Day 表报告

## 目的

本步骤从 `raw_student_day_table.csv` 生成严格清洗表。清洗只处理标签和重复 student-day，不做任何全局填补或全局标准化，从而避免 preprocessing leakage。

## 清洗规则

- 根据数值型 `stress` 重新生成 `stress_label`：Low = 0-17，Medium = 18-38，High = 39-100。
- 对同一 `student_id + date` 的多条问卷记录，所有数值列取平均值。
- `CALCULATION_FAILED` 使用 any-true 规则：同一天任一记录为 True，则清洗后为 True。
- wearable 缺失值保持缺失，不在这里填补。
- wearable 特征保持原始单位，不在这里标准化。

## 清洗结果

- 原始行数：3133
- 清洗后行数：3118
- 学生数：35
- 删除的重复行数：15
- 重复 student-day pair 数：15
- stress 数值冲突的重复 pair 数：15
- stress label 冲突的重复 pair 数：7
- wearable 完整行数：1482 (47.53%)
- wearable 缺失单元格比例：38.02%

## 标签分布

| stress_label | count |
| --- | --- |
| Low | 1082 |
| High | 1052 |
| Medium | 984 |

## 缺失值概览

| column | missing_rows | missing_percent |
| --- | --- | --- |
| CALCULATION_FAILED | 1652 | 52.98 |
| STRESS_SCORE | 1652 | 52.98 |
| sleep_score | 1617 | 51.86 |
| deep_sleep_minutes | 1617 | 51.86 |
| avg_low_frequency | 1481 | 47.50 |
| avg_rmssd | 1481 | 47.50 |
| avg_high_frequency | 1481 | 47.50 |
| std_oxygen | 1111 | 35.63 |
| avg_oxygen | 1109 | 35.57 |
| total_steps | 886 | 28.42 |
| moderately_active_minutes | 861 | 27.61 |
| lightly_active_minutes | 861 | 27.61 |
| sedentary_minutes | 861 | 27.61 |
| very_active_minutes | 861 | 27.61 |

## 重复记录预览

| student_id | date | n_rows | stress_values | anxiety_values | label_values | stress_conflict | label_conflict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | 2025-04-29 | 2 | [35, 46] | [36, 49] | ['High', 'Medium'] | True | True |
| 13 | 2025-05-15 | 2 | [42, 67] | [38, 73] | ['High'] | True | False |
| 14 | 2025-03-24 | 2 | [9, 31] | [16, 27] | ['Low', 'Medium'] | True | True |
| 15 | 2025-03-26 | 2 | [18, 22] | [24, 33] | ['Medium'] | True | False |
| 18 | 2025-03-15 | 2 | [46, 57] | [39, 53] | ['High'] | True | False |
| 19 | 2025-05-18 | 2 | [0, 4] | [0, 12] | ['Low'] | True | False |
| 19 | 2025-06-16 | 2 | [0, 18] | [0, 13] | ['Low', 'Medium'] | True | True |
| 27 | 2025-03-15 | 2 | [86, 100] | [82, 83] | ['High'] | True | False |
| 27 | 2025-03-17 | 2 | [23, 51] | [24, 34] | ['High', 'Medium'] | True | True |
| 27 | 2025-05-14 | 2 | [27, 76] | [34, 82] | ['High', 'Medium'] | True | True |
| 34 | 2025-05-17 | 2 | [41, 82] | [26, 66] | ['High'] | True | False |
| 5 | 2025-03-11 | 2 | [13, 21] | [7, 17] | ['Low', 'Medium'] | True | True |
| 5 | 2025-05-04 | 2 | [23, 95] | [15, 84] | ['High', 'Medium'] | True | True |
| 6 | 2025-05-28 | 2 | [3, 8] | [3, 5] | ['Low'] | True | False |
| 7 | 2025-04-20 | 2 | [1, 15] | [2, 7] | ['Low'] | True | False |

## 后续建模注意事项

- 后续应基于这张严格清洗表先做 subject-aware train/test split。
- 缺失值填补和标准化必须放进 scikit-learn pipeline，在训练数据或训练折内部拟合。
- `student_id`、`date`、`stress`、`stress_label`、`anxiety`、`STRESS_SCORE`、`CALCULATION_FAILED` 不应作为主模型输入。
