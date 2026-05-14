# 原始 Student-Day 表合并报告

## 目的

本步骤从 `SSAQS dataset/` 中每个学生的原始 CSV 文件重新合并 student-day 表。该表不做填补、不做标准化，也不删除缺失 wearable 的日期，目的是为后续 leakage-safe preprocessing 提供干净起点。

## 合并规则

- `daily_questions.csv`：使用 `timeStampStart` 转成日期；每条问卷记录保留为一行，因此同一学生同一天多次问卷会暂时保留重复行。
- `stress_label`：根据 `stress` 重新生成，规则为 Low = 0-17，Medium = 18-38，High = 39-100。
- `sleep.csv`：按 student-date 聚合，取 `overall_score` 和 `deep_sleep_in_minutes` 的日均值。
- `steps.csv`：按 student-date 汇总每日总步数。
- `activity_level.csv`：按 student-date 统计各活动等级出现次数，作为对应活动分钟数。
- `hrv.csv`：按 student-date 计算 RMSSD、low frequency、high frequency 的日均值。
- `oxygen.csv`：按 student-date 计算血氧均值和标准差。
- `stress.csv`：按 student-date 合并 Fitbit stress score；该列后续不作为主模型输入，仅保留用于审计。

## 合并结果

- 行数：3133
- 学生数：35
- unique student-day 数：3118
- 日期范围：2025-02-14 到 2025-07-09
- 重复 student-day 行数：15
- 重复 student-day pair 数：15
- wearable 缺失单元格比例：37.98%

## 标签分布

| stress_label | count |
| --- | --- |
| Low | 1086 |
| High | 1057 |
| Medium | 990 |

## 缺失值概览

| column | missing_rows | missing_percent |
| --- | --- | --- |
| CALCULATION_FAILED | 1657 | 52.89 |
| STRESS_SCORE | 1657 | 52.89 |
| sleep_score | 1623 | 51.80 |
| deep_sleep_minutes | 1623 | 51.80 |
| avg_low_frequency | 1487 | 47.46 |
| avg_rmssd | 1487 | 47.46 |
| avg_high_frequency | 1487 | 47.46 |
| std_oxygen | 1115 | 35.59 |
| avg_oxygen | 1113 | 35.53 |
| total_steps | 889 | 28.38 |
| moderately_active_minutes | 864 | 27.58 |
| lightly_active_minutes | 864 | 27.58 |
| sedentary_minutes | 864 | 27.58 |
| very_active_minutes | 864 | 27.58 |

## 与旧版 `final_student_day_table_v01.csv` 对比

| metric | value |
| --- | --- |
| reference_exists | True |
| reference_rows | 3135 |
| reference_unique_student_days | 3118 |
| rebuilt_rows | 3133 |
| rebuilt_unique_student_days | 3118 |
| common_unique_student_days | 3118 |
| reference_only_student_days | 0 |
| rebuilt_only_student_days | 0 |
| common_grouped_student_days | 3118 |
| common_student_days_with_different_row_counts | 2 |
| common_student_days_with_different_stress_values | 2 |
| common_student_days_with_different_anxiety_values | 2 |

## 后续处理建议

- 下一步应先处理同一学生同一天的多次问卷记录，再做 subject-aware train/test split。
- 不应在这张原始合并表上做全局填补或全局标准化。
- 缺失值应放到模型 pipeline 中处理，确保 imputer/scaler 只在训练数据或训练折内部拟合。
