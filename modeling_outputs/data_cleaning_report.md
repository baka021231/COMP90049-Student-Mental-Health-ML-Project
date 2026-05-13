# 建模数据清理报告

## 目标

本模块的目标是从 `final_student_day_table_v01_processed.csv` 生成一份可复现、可直接用于后续 RQ 建模的 clean modelling table。

`final_student_day_table_v01_processed.csv` 已经适合做建模基础，因为主要 wearable 特征已经完成填补和标准化；但它仍然需要一层“建模前清理”，包括去重、统一标签、生成时间特征、固定 train/test split，以及明确哪些列不能作为模型输入。

## 输入文件

- `final_student_day_table_v01_processed.csv`：主要建模来源。
- `final_student_day_table_v01.csv`：未 fully processed 的 merged student-day 表，主要用于理解 raw missingness 和数据审计，不作为当前主建模表。

## 清理原则

1. 不直接修改原始 CSV。
2. 在任何 split 或建模之前，先处理重复的 `student_id + date`。
3. 用数值型 `stress` 重新生成 `stress_label`，让标签规则显式可查。
4. 明确排除可能造成 leakage 的列。
5. 所有 RQ 的 feature set 都从同一份 clean table 派生。
6. 使用 subject-aware train/test split，确保同一个学生不会同时出现在 train 和 test 中。

## 工作流程

清理脚本位于 `scripts/prepare_model_data.py`。

在项目根目录运行：

```bash
python3 scripts/prepare_model_data.py
```

脚本执行流程：

1. 读取 `final_student_day_table_v01_processed.csv`。
2. 将 `date` 转换为 datetime，将 `student_id` 转换为 string。
3. 对重复的 `student_id + date` 行进行处理：所有数值列取平均。
4. 根据 `stress` 重新生成 `stress_label`：
   - Low: `stress <= 17`
   - Medium: `18 <= stress <= 38`
   - High: `stress >= 39`
5. 添加时间特征：
   - `semester_week`
   - `day_of_week`
   - `is_weekend`
   - sleep score、total steps、RMSSD 的 one-observation lag 特征
   - sleep score、total steps、RMSSD 的 3-observation / 7-observation rolling mean
6. 定义 RQ1、RQ2、RQ3 所需的 feature sets。
7. 使用 `GroupShuffleSplit` 创建 subject-aware train/test split。
8. 将所有建模产物写入 `modeling_outputs/`。

## 生成文件

- `clean_model_data.csv`：清理后的 student-day 建模表，包含时间特征和 `split` 列。
- `feature_sets.json`：RQ1、RQ2、RQ3 的标准特征列定义。
- `split_assignments.csv`：每个 student 的 train/test 分配。
- `cleaning_audit.json`：机器可读的数据清理审计结果。

## 清理结果

| 项目 | 数值 |
| --- | ---: |
| Source rows | 1,394 |
| Source students | 32 |
| 重复 `student_id + date` 行数 | 10 |
| Clean rows | 1,384 |
| Clean students | 32 |
| Train students | 24 |
| Test students | 8 |
| Train rows | 1,097 |
| Test rows | 287 |

清理后的标签分布：

| Label | Count |
| --- | ---: |
| Low | 359 |
| Medium | 491 |
| High | 534 |

Train/test 中的标签分布：

| Split | Low | Medium | High | Rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 273 | 406 | 418 | 1,097 |
| Test | 86 | 85 | 116 | 287 |

## Feature Sets 设计

RQ1 使用全部 wearable 特征：

- sleep features
- activity features
- HRV features
- SpO2 features

RQ2 做 feature-group ablation：

- `sleep_only`
- `activity_only`
- `hrv_spo2_only`
- `all_wearable`

RQ3 比较时间特征版本：

- `no_temporal`
- `semester_week`
- `calendar`
- `lag_1`
- `rolling_3`
- `rolling_7`

## Leakage Policy

以下列不能作为模型输入：

- `student_id`
- `date`
- `stress`
- `stress_label`
- `anxiety`
- `STRESS_SCORE`
- `CALCULATION_FAILED`

原因：

- `stress` 和 `stress_label` 是预测目标本身。
- `anxiety` 是同一天问卷变量，和 stress 高度相关，主实验中使用它会让任务变得不公平。
- `STRESS_SCORE` 是 Fitbit 自己计算出的压力分数，和目标概念重叠，可能造成 leakage。
- `student_id` 可能让模型记住个体，而不是学习可泛化的 wearable pattern。
- `date` 不能直接作为原始字符串输入；需要用明确的时间特征替代。

## 给后续建模的注意事项

- 后续模型必须使用 `clean_model_data.csv` 里的 `split` 列，不要每个模型重新随机划分。
- RQ1 和 RQ2 的核心 wearable 特征没有缺失值。
- RQ3 的 lag / rolling 特征有 32 个缺失值，对应每个学生的第一条 observation。后续建模时应在 pipeline 内处理，建议使用 train-set-only median imputation。
- Logistic Regression、SVM、MLP 需要在 pipeline 内做 scaling。
- Random Forest 不需要 scaling，但必须使用同一个 train/test split。

## 当前状态

数据清理模块已完成。下一步可以开始写建模脚本，直接读取：

- `modeling_outputs/clean_model_data.csv`
- `modeling_outputs/feature_sets.json`
- `modeling_outputs/split_assignments.csv`

后续 RQ1、RQ2、RQ3 都应该基于这套固定数据和固定 split 进行实验。
