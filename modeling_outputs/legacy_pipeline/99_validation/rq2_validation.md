# 旧版 RQ2 验证记录

## 运行命令

```bash
python -m py_compile scripts/modeling_utils.py scripts/run_rq2_feature_groups.py
python scripts/run_rq2_feature_groups.py
```

## 验证结果

- 共享建模工具和 RQ2 脚本可以通过 Python 编译检查。
- RQ2 脚本可以从项目根目录运行。
- 脚本读取 `modeling_outputs/legacy_pipeline/00_model_data/clean_model_data.csv`。
- 脚本从 `modeling_outputs/legacy_pipeline/00_model_data/feature_sets.json` 读取 `rq2_feature_groups`。
- 脚本评估 4 个 feature groups。
- 每个 feature group 训练 5 个模型。
- 完整结果表包含 20 行。
- best-by-group 表包含 4 行。
- 重新运行脚本后没有产生 git diff。

## 主要输出

- `modeling_outputs/legacy_pipeline/02_rq2/rq2_feature_group_results.csv`
- `modeling_outputs/legacy_pipeline/02_rq2/rq2_best_by_feature_group.csv`
- `modeling_outputs/legacy_pipeline/02_rq2/rq2_summary.md`

## 主要结果

- 最好的 feature group + 模型：`all_wearable` + `mlp`
- 最好 macro-F1：0.409
- `hrv_spo2_only` 最好 macro-F1：0.401
- `activity_only` 最好 macro-F1：0.386
- `sleep_only` 最好 macro-F1：0.354

## 报告解释

旧版 RQ2 可以支持 ablation 讨论，但这个版本基于旧 processed 表，不作为最终主结果。最终报告应优先使用严格流程中的 RQ2 结果。
