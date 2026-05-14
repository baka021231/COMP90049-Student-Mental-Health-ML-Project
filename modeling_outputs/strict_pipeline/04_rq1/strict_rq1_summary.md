# 严格版 RQ1 建模总结

## 研究问题

RQ1 关注：只使用 wearable 特征，能否预测学生当天的压力等级。

## 严格实验设计

- 输入数据来自 `strict_model_data.csv`，该表没有提前做全局填补或全局标准化。
- 使用 `strict_feature_sets.json` 中的 `rq1_all_wearable`。
- train/test 已按学生划分，同一个学生不会同时出现在 train 和 test。
- 缺失值填补使用 `SimpleImputer`，并放在模型 pipeline 内部。
- Logistic Regression、SVM、kNN、MLP 的标准化也放在 pipeline 内部。
- 除 majority baseline 外，模型超参数通过训练集内部的 GroupKFold 交叉验证选择。
- test set 只用于最终评估，不用于选择模型或调参。

## 数据规模

- 特征数：12
- 训练行数：2217
- 测试行数：901
- 训练标签分布：{'Low': 696, 'Medium': 717, 'High': 804}
- 测试标签分布：{'Low': 386, 'Medium': 267, 'High': 248}

## 结果

| model | accuracy | macro_f1 | low_f1 | medium_f1 | high_f1 | cv_macro_f1 | best_params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mlp | 0.374 | 0.357 | 0.480 | 0.261 | 0.330 | 0.362 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [64]} |
| svm | 0.375 | 0.356 | 0.490 | 0.251 | 0.327 | 0.353 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| gradient_boosting | 0.374 | 0.355 | 0.479 | 0.238 | 0.347 | 0.353 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| random_forest | 0.367 | 0.349 | 0.468 | 0.231 | 0.348 | 0.368 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| logistic_regression | 0.290 | 0.283 | 0.327 | 0.301 | 0.220 | 0.289 | {"classifier__C": 0.1} |
| knn | 0.285 | 0.277 | 0.172 | 0.301 | 0.358 | 0.296 | {"classifier__n_neighbors": 3} |
| majority_baseline | 0.275 | 0.144 | 0.000 | 0.000 | 0.432 | N/A | {} |

## 关键发现

- test macro-F1 最高的模型是 `mlp`。
- 最佳 test macro-F1：0.357。
- majority baseline test macro-F1：0.144。
- 最佳模型最难预测的类别是 `Medium`。
- 因为本实验使用原始未标准化数据，并在 pipeline 内部完成填补和标准化，所以比旧版实验更能避免 preprocessing leakage。

## 最佳模型 `mlp` 的 Confusion Matrix

| actual | predicted_Low | predicted_Medium | predicted_High |
| --- | --- | --- | --- |
| actual_Low | 178 | 85 | 123 |
| actual_Medium | 81 | 61 | 125 |
| actual_High | 96 | 54 | 98 |

## 报告写作建议

报告中应强调：严格版 RQ1 的目标不是追求高 accuracy，而是验证 wearable-only signal 是否在无 leakage 的设置下仍然提供一定预测能力。若分数下降，也应解释为更严格评估带来的合理结果。
