# 严格建模流程验证记录

## 验证命令

```bash
python -m py_compile scripts/audit_raw_ssaqs_data.py scripts/build_raw_student_day_table.py scripts/clean_raw_student_day_table.py scripts/prepare_strict_model_data.py scripts/modeling_utils.py scripts/run_strict_rq1_models.py scripts/run_strict_rq2_feature_groups.py
python scripts/prepare_strict_model_data.py
python scripts/run_strict_rq1_models.py
python scripts/run_strict_rq2_feature_groups.py
```

## 验证结果

- 关键脚本均能通过 Python 编译检查。
- 严格建模数据可以从 `strict_clean_student_day_table.csv` 重新生成。
- 严格版 RQ1 可以完整运行，并重新生成结果、调参记录、confusion matrix 和中文总结。
- 严格版 RQ2 可以完整运行，并重新生成 feature group 结果、调参记录和中文总结。
- 重新运行严格流程后没有产生 git diff，说明当前输出具有可复现性。
- 严格流程输出已整理到 `modeling_outputs/strict_pipeline/` 下的分阶段文件夹。

## 当前严格版核心结果

### RQ1

- 最佳模型：`mlp`
- test macro-F1：0.357
- majority baseline test macro-F1：0.144
- 解释重点：严格版 RQ1 使用原始未标准化数据，并在 pipeline 内部完成填补、标准化和调参，因此比旧版结果更保守。

### RQ2

- 最佳 feature group + 模型：`hrv_spo2_only` + `random_forest`
- test macro-F1：0.393
- `sleep_only` 的最佳 test macro-F1：0.391
- `all_wearable` 的最佳 test macro-F1：0.357
- 解释重点：更多 wearable 特征不一定带来更好表现，可能因为缺失、噪声和学生数量有限导致全特征模型不稳定。

## 对报告的意义

这条严格流程可以直接回应三个主要方法风险：

- 没有真正调参：已加入 GroupKFold + GridSearchCV。
- test set leakage：模型和超参数选择只在训练集内部完成，test set 只用于最终评估。
- preprocessing leakage：缺失值填补和标准化放入 pipeline，不在 train/test split 前全局拟合。
