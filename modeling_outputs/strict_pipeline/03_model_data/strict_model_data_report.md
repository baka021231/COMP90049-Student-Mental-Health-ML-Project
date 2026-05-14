# 严格建模数据准备报告

## 目的

本步骤从 `strict_clean_student_day_table.csv` 生成后续建模使用的严格数据表。关键原则是：先按学生划分 train/test，但不在这里做任何缺失值填补或标准化。

## 处理内容

- 保留原始 wearable 特征单位。
- 保留 wearable 缺失值。
- 添加确定性的日历特征：`semester_week`、`day_of_week`、`is_weekend`。
- 使用 subject-aware split，确保同一个学生不会同时出现在 train 和 test。
- 生成严格版 feature sets，供 RQ1/RQ2/RQ3 使用。

## 数据规模

- 行数：3118
- 学生数：35
- unique student-day 数：3118
- 随机种子：49
- test_size：0.25

## Train/Test Split

| split | students | rows | low | medium | high |
| --- | --- | --- | --- | --- | --- |
| test | 9 | 901 | 386 | 267 | 248 |
| train | 26 | 2217 | 696 | 717 | 804 |

## Wearable 缺失值

| feature | missing_rows | missing_percent |
| --- | --- | --- |
| sleep_score | 1617 | 51.86 |
| deep_sleep_minutes | 1617 | 51.86 |
| avg_low_frequency | 1481 | 47.50 |
| avg_rmssd | 1481 | 47.50 |
| avg_high_frequency | 1481 | 47.50 |
| std_oxygen | 1111 | 35.63 |
| avg_oxygen | 1109 | 35.57 |
| total_steps | 886 | 28.42 |
| sedentary_minutes | 861 | 27.61 |
| very_active_minutes | 861 | 27.61 |
| lightly_active_minutes | 861 | 27.61 |
| moderately_active_minutes | 861 | 27.61 |

## Feature Sets

- RQ1：全部 wearable 特征。
- RQ2：sleep_only、activity_only、hrv_spo2_only、all_wearable。
- RQ3：当前先保留 no_temporal 和 calendar 两个版本；lag/rolling 特征可在后续单独加入并处理缺失。

## 泄漏控制

- 本步骤没有做全局 imputation。
- 本步骤没有做全局 standardization。
- 后续模型必须在 pipeline 中使用 `SimpleImputer` / `StandardScaler`，并只在训练数据或交叉验证训练折内拟合。
- `student_id`、`date`、`stress`、`stress_label`、`anxiety`、`STRESS_SCORE`、`CALCULATION_FAILED` 不作为主模型输入。
