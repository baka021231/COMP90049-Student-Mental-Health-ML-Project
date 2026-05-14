# 旧版 RQ1 验证记录

## 运行命令

```bash
python -m py_compile scripts/run_rq1_models.py
python scripts/run_rq1_models.py
```

## 验证结果

- RQ1 脚本可以通过 Python 编译检查。
- RQ1 脚本可以从项目根目录运行。
- 脚本读取 `modeling_outputs/legacy_pipeline/00_model_data/clean_model_data.csv`。
- 脚本从 `modeling_outputs/legacy_pipeline/00_model_data/feature_sets.json` 读取 `rq1_all_wearable`。
- 脚本使用已有的 `split` 列。
- 脚本训练五个模型：majority baseline、Logistic Regression、SVM、Random Forest 和 MLP。
- 脚本输出到 `modeling_outputs/legacy_pipeline/01_rq1/`。
- 每个模型都会输出一个 confusion matrix CSV。
- 重新运行脚本后没有产生 git diff。

## 主要结果

- macro-F1 最好的模型：`mlp`
- 最好 macro-F1：0.409
- majority baseline macro-F1：0.192
- 最难预测的类别：`Medium`

## 报告解释

旧版 RQ1 说明 wearable-only 特征有一定预测信号，因为最佳模型明显高于 majority baseline。但这个版本基于旧 processed 表，不作为最终主结果，最终报告应优先使用严格流程结果。
