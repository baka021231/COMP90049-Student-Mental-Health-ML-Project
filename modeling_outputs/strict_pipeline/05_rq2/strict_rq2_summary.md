# 严格版 RQ2 Feature Group 总结

## 研究问题

RQ2 关注：不同 wearable feature group 对压力预测的贡献有多大。

## 严格实验设计

- 输入数据来自 `strict_model_data.csv`，没有提前做全局填补或标准化。
- 每个 feature group 都使用同一套模型集合和同一套 train/test split。
- 每个模型都在训练集内部使用 GroupKFold 进行超参数选择。
- test set 只用于最终评估，不用于选择 feature group、模型或超参数。
- 缺失值填补和必要的标准化都放在模型 pipeline 内部。

## Feature Groups

- `sleep_only`：2 个特征
- `activity_only`：5 个特征
- `hrv_spo2_only`：5 个特征
- `all_wearable`：12 个特征

## 每个 Feature Group 的最佳结果

| feature_group | n_features | model | accuracy | macro_f1 | low_f1 | medium_f1 | high_f1 | cv_macro_f1 | best_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hrv_spo2_only | 5 | random_forest | 0.423 | 0.393 | 0.564 | 0.281 | 0.335 | 0.364 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| sleep_only | 2 | random_forest | 0.438 | 0.391 | 0.578 | 0.228 | 0.367 | 0.352 | {"classifier__max_depth": 5, "classifier__n_estimators": 100} |
| all_wearable | 12 | mlp | 0.374 | 0.357 | 0.480 | 0.261 | 0.330 | 0.362 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [64]} |
| activity_only | 5 | gradient_boosting | 0.377 | 0.356 | 0.485 | 0.237 | 0.347 | 0.374 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |

## 完整结果表

| feature_group | n_features | model | accuracy | macro_f1 | cv_macro_f1 | best_params |
| --- | --- | --- | --- | --- | --- | --- |
| activity_only | 5 | gradient_boosting | 0.377 | 0.356 | 0.374 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| activity_only | 5 | random_forest | 0.366 | 0.345 | 0.381 | {"classifier__max_depth": null, "classifier__n_estimators": 100} |
| activity_only | 5 | svm | 0.362 | 0.343 | 0.362 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| activity_only | 5 | mlp | 0.352 | 0.333 | 0.361 | {"classifier__alpha": 0.0001, "classifier__hidden_layer_sizes": [32, 16]} |
| activity_only | 5 | logistic_regression | 0.302 | 0.290 | 0.291 | {"classifier__C": 1.0} |
| activity_only | 5 | knn | 0.270 | 0.263 | 0.286 | {"classifier__n_neighbors": 5} |
| activity_only | 5 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| all_wearable | 12 | mlp | 0.374 | 0.357 | 0.362 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [64]} |
| all_wearable | 12 | svm | 0.375 | 0.356 | 0.353 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| all_wearable | 12 | gradient_boosting | 0.374 | 0.355 | 0.353 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| all_wearable | 12 | random_forest | 0.367 | 0.349 | 0.368 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| all_wearable | 12 | logistic_regression | 0.290 | 0.283 | 0.289 | {"classifier__C": 0.1} |
| all_wearable | 12 | knn | 0.285 | 0.277 | 0.296 | {"classifier__n_neighbors": 3} |
| all_wearable | 12 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| hrv_spo2_only | 5 | random_forest | 0.423 | 0.393 | 0.364 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| hrv_spo2_only | 5 | mlp | 0.426 | 0.389 | 0.377 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [32, 16]} |
| hrv_spo2_only | 5 | gradient_boosting | 0.412 | 0.381 | 0.352 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| hrv_spo2_only | 5 | svm | 0.416 | 0.359 | 0.367 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| hrv_spo2_only | 5 | knn | 0.294 | 0.290 | 0.306 | {"classifier__n_neighbors": 5} |
| hrv_spo2_only | 5 | logistic_regression | 0.266 | 0.258 | 0.246 | {"classifier__C": 10.0} |
| hrv_spo2_only | 5 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| sleep_only | 2 | random_forest | 0.438 | 0.391 | 0.352 | {"classifier__max_depth": 5, "classifier__n_estimators": 100} |
| sleep_only | 2 | svm | 0.428 | 0.370 | 0.349 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| sleep_only | 2 | gradient_boosting | 0.434 | 0.364 | 0.355 | {"classifier__learning_rate": 0.05, "classifier__max_depth": 2, "classifier__n_estimators": 100} |
| sleep_only | 2 | mlp | 0.424 | 0.358 | 0.357 | {"classifier__alpha": 0.0001, "classifier__hidden_layer_sizes": [32, 16]} |
| sleep_only | 2 | knn | 0.295 | 0.267 | 0.326 | {"classifier__n_neighbors": 3} |
| sleep_only | 2 | logistic_regression | 0.270 | 0.200 | 0.265 | {"classifier__C": 0.1} |
| sleep_only | 2 | majority_baseline | 0.275 | 0.144 | N/A | {} |

## 关键发现

- 整体 test macro-F1 最高的是 `hrv_spo2_only` + `random_forest`。
- 最高 test macro-F1：0.393。
- 最弱的 feature group 是 `activity_only`。
- RQ2 应作为 ablation study 来写，重点解释不同模态的相对贡献，而不是只比较绝对分数。

## 报告写作建议

如果严格版分数低于旧版结果，应解释为：旧版 processed 表可能提前做了全局处理，而严格版把填补、标准化和调参都限制在训练流程内部，因此评估更保守、更可信。
