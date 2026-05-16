# 严格版 RQ3 时序特征实验总结

## 研究问题

RQ3 关注：在 wearable-only 压力等级预测任务中，加入学期周次、前一天 wearable 特征、过去 3 天/7 天 wearable 均值后，是否能改善模型表现？

## 严格实验设计

- 随机种子：`49`，与严格版 RQ1/RQ2 保持一致。
- 输入数据来自 `modeling_outputs/strict_pipeline/03_model_data/strict_model_data.csv`。
- 使用严格版 RQ1/RQ2 相同的 `split` 列：同一个学生不会同时出现在 train 和 test。
- train：26 名学生，2217 行；test：9 名学生，901 行。
- 缺失值填补和必要的标准化都放在 scikit-learn pipeline 内部。
- 模型和超参数选择只在训练集内部通过 GroupKFold + GridSearchCV 完成。
- test set 只用于最终评估。
- 不使用当天 `stress`、`anxiety`、`STRESS_SCORE`、`CALCULATION_FAILED`，也不使用 `stress_lag1`，避免把压力自评或设备压力分数带入模型。
- lag 特征使用 `groupby(student_id).shift(1)`，rolling 特征使用 `shift(1).rolling(...).mean()`，避免包含当天信息。

## 时序条件

- `no_temporal`：12 个特征。
- `semester_week_only`：13 个特征。
- `lag1_wearable`：24 个特征。
- `rolling3_wearable`：24 个特征。
- `rolling7_wearable`：24 个特征。

## 每个条件的最佳结果

| temporal_condition | n_features | model | accuracy | macro_f1 | low_f1 | medium_f1 | high_f1 | cv_macro_f1 | best_params | delta_vs_no_temporal_best | delta_vs_strict_rq1_mlp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling7_wearable | 24 | gradient_boosting | 0.398 | 0.392 | 0.462 | 0.316 | 0.399 | 0.353 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 2, "classifier__n_estimators": 200} | 0.035 | 0.035 |
| rolling3_wearable | 24 | gradient_boosting | 0.401 | 0.390 | 0.468 | 0.308 | 0.395 | 0.363 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} | 0.033 | 0.033 |
| lag1_wearable | 24 | svm | 0.400 | 0.389 | 0.464 | 0.323 | 0.381 | 0.373 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} | 0.032 | 0.032 |
| semester_week_only | 13 | mlp | 0.373 | 0.369 | 0.425 | 0.293 | 0.388 | 0.364 | {"classifier__alpha": 0.0001, "classifier__hidden_layer_sizes": [32, 16]} | 0.011 | 0.012 |
| no_temporal | 12 | mlp | 0.374 | 0.357 | 0.480 | 0.261 | 0.330 | 0.362 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [64]} | 0.000 | 0.000 |

## 完整模型结果

| temporal_condition | n_features | model | accuracy | macro_f1 | cv_macro_f1 | best_params |
| --- | --- | --- | --- | --- | --- | --- |
| lag1_wearable | 24 | svm | 0.400 | 0.389 | 0.373 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| lag1_wearable | 24 | gradient_boosting | 0.378 | 0.358 | 0.352 | {"classifier__learning_rate": 0.05, "classifier__max_depth": 2, "classifier__n_estimators": 100} |
| lag1_wearable | 24 | random_forest | 0.371 | 0.354 | 0.342 | {"classifier__max_depth": 10, "classifier__n_estimators": 300} |
| lag1_wearable | 24 | mlp | 0.358 | 0.344 | 0.369 | {"classifier__alpha": 0.0001, "classifier__hidden_layer_sizes": [32]} |
| lag1_wearable | 24 | knn | 0.304 | 0.293 | 0.296 | {"classifier__n_neighbors": 9} |
| lag1_wearable | 24 | logistic_regression | 0.270 | 0.266 | 0.302 | {"classifier__C": 1.0} |
| lag1_wearable | 24 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| no_temporal | 12 | mlp | 0.374 | 0.357 | 0.362 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [64]} |
| no_temporal | 12 | svm | 0.375 | 0.356 | 0.353 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| no_temporal | 12 | gradient_boosting | 0.374 | 0.355 | 0.353 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| no_temporal | 12 | random_forest | 0.367 | 0.349 | 0.368 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| no_temporal | 12 | logistic_regression | 0.290 | 0.283 | 0.289 | {"classifier__C": 0.1} |
| no_temporal | 12 | knn | 0.285 | 0.277 | 0.296 | {"classifier__n_neighbors": 3} |
| no_temporal | 12 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| rolling3_wearable | 24 | gradient_boosting | 0.401 | 0.390 | 0.363 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| rolling3_wearable | 24 | svm | 0.394 | 0.385 | 0.346 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| rolling3_wearable | 24 | random_forest | 0.377 | 0.367 | 0.345 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| rolling3_wearable | 24 | mlp | 0.367 | 0.362 | 0.376 | {"classifier__alpha": 0.0001, "classifier__hidden_layer_sizes": [32, 16]} |
| rolling3_wearable | 24 | knn | 0.292 | 0.287 | 0.316 | {"classifier__n_neighbors": 3} |
| rolling3_wearable | 24 | logistic_regression | 0.289 | 0.286 | 0.301 | {"classifier__C": 1.0} |
| rolling3_wearable | 24 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| rolling7_wearable | 24 | gradient_boosting | 0.398 | 0.392 | 0.353 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 2, "classifier__n_estimators": 200} |
| rolling7_wearable | 24 | svm | 0.393 | 0.384 | 0.349 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| rolling7_wearable | 24 | random_forest | 0.387 | 0.379 | 0.350 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| rolling7_wearable | 24 | mlp | 0.355 | 0.343 | 0.361 | {"classifier__alpha": 0.001, "classifier__hidden_layer_sizes": [32, 16]} |
| rolling7_wearable | 24 | logistic_regression | 0.316 | 0.317 | 0.298 | {"classifier__C": 0.1} |
| rolling7_wearable | 24 | knn | 0.302 | 0.301 | 0.309 | {"classifier__n_neighbors": 5} |
| rolling7_wearable | 24 | majority_baseline | 0.275 | 0.144 | N/A | {} |
| semester_week_only | 13 | mlp | 0.373 | 0.369 | 0.364 | {"classifier__alpha": 0.0001, "classifier__hidden_layer_sizes": [32, 16]} |
| semester_week_only | 13 | gradient_boosting | 0.375 | 0.367 | 0.356 | {"classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200} |
| semester_week_only | 13 | random_forest | 0.382 | 0.367 | 0.359 | {"classifier__max_depth": 10, "classifier__n_estimators": 100} |
| semester_week_only | 13 | logistic_regression | 0.363 | 0.350 | 0.308 | {"classifier__C": 0.1} |
| semester_week_only | 13 | knn | 0.347 | 0.342 | 0.338 | {"classifier__n_neighbors": 9} |
| semester_week_only | 13 | svm | 0.347 | 0.342 | 0.358 | {"classifier__C": 10.0, "classifier__kernel": "rbf"} |
| semester_week_only | 13 | majority_baseline | 0.275 | 0.144 | N/A | {} |

## 学期周压力趋势

- 平均压力最低：第 21 周，M=13.62，SD=12.92。
- 平均压力最高：第 6 周，M=40.59，SD=27.21。
- 趋势图：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_weekly_stress_trend.svg`。

## 关键结论

- 严格版 RQ3 的最佳条件是 `rolling7_wearable` + `gradient_boosting`。
- 最佳 test macro-F1 为 0.392。
- 相比本 RQ3 的 no-temporal 最佳模型，变化为 +0.035。
- 相比严格版 RQ1 MLP baseline macro-F1=0.357，变化为 +0.035。
- 如果时序条件没有明显提升，应在报告中如实写成 negative or weak result：严格防泄漏设置下，时序 wearable 特征未必稳定改善 unseen-student 压力预测。

## 输出文件

- 结果表：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_temporal_feature_results.csv`
- 每个条件最佳结果：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_best_by_temporal_condition.csv`
- 调参记录：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_tuning_results.csv`
- 最佳模型 confusion matrix：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_confusion_matrix_best_overall.csv`
- 周压力趋势图：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_weekly_stress_trend.svg`
- 周压力统计表：`modeling_outputs\strict_pipeline\06_rq3\strict_rq3_weekly_stress_summary.csv`
