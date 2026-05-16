# RQ3 素材：Temporal context 与 lag/rolling features

## 研究问题

RQ3：在 wearable-only stress classification 中，加入 semester progression、previous-day wearable features 和 rolling wearable summaries 是否能改善预测？

## 实验设计

| 项 | 内容 |
|---|---|
| 数据 | strict model data |
| Split | 与 RQ1/RQ2 相同的 subject-aware train/test split |
| Train/test | 26 train students / 9 test students |
| 基线条件 | no temporal：12 个 wearable features |
| Temporal features | semester week、lag1 wearable、rolling3 mean、rolling7 mean |
| 模型集合 | 与 RQ1/RQ2 相同 |
| 调参 | GroupKFold + GridSearchCV |
| 主指标 | test macro-F1 |

## Temporal feature 构造

Temporal conditions:

| Condition | N features | 含义 |
|---|---:|---|
| no_temporal | 12 | 原始 wearable features |
| semester_week_only | 13 | wearable features + semester week |
| lag1_wearable | 24 | wearable features + 前一天同一学生 wearable features |
| rolling3_wearable | 24 | wearable features + 过去 3 天 wearable rolling mean |
| rolling7_wearable | 24 | wearable features + 过去 7 天 wearable rolling mean |

构造原则：

- Lag features 使用同一学生上一天信息：`groupby(student_id).shift(1)`。
- Rolling features 使用过去信息：`shift(1).rolling(window=N).mean()`。
- Rolling features 不包含当天 wearable value。
- 不使用当天或历史 self-reported `stress`。
- 不使用 `anxiety` 或 Fitbit `STRESS_SCORE`。

这个设计使 RQ3 真正测试 wearable temporal context，而不是利用问卷历史或目标泄漏。

## 每个 temporal condition 的最佳结果

| Temporal condition | N features | Best model | Accuracy | Macro-F1 | Low F1 | Medium F1 | High F1 | CV Macro-F1 | Best params |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| rolling7_wearable | 24 | Gradient Boosting | 0.398 | 0.392 | 0.462 | 0.316 | 0.399 | 0.353 | `learning_rate=0.1`, `max_depth=2`, `n_estimators=200` |
| rolling3_wearable | 24 | Gradient Boosting | 0.401 | 0.390 | 0.468 | 0.308 | 0.395 | 0.363 | `learning_rate=0.1`, `max_depth=3`, `n_estimators=200` |
| lag1_wearable | 24 | SVM | 0.400 | 0.389 | 0.464 | 0.323 | 0.381 | 0.373 | `C=10.0`, `kernel=rbf` |
| semester_week_only | 13 | MLP | 0.373 | 0.369 | 0.425 | 0.293 | 0.388 | 0.364 | `alpha=0.0001`, `hidden_layer_sizes=(32,16)` |
| no_temporal | 12 | MLP | 0.374 | 0.357 | 0.480 | 0.261 | 0.330 | 0.362 | `alpha=0.001`, `hidden_layer_sizes=(64,)` |

图：

![RQ3 temporal condition macro-F1](figures/figure5_rq3_temporal_condition_macro_f1.svg)

## 主要发现

| 项 | 结果 |
|---|---|
| Best temporal condition | rolling7_wearable |
| Best model | Gradient Boosting |
| Best test macro-F1 | 0.392 |
| No-temporal baseline macro-F1 | 0.357 |
| Improvement over no-temporal | +0.035 |
| Best rolling3 macro-F1 | 0.390 |
| Best lag1 macro-F1 | 0.389 |
| Best semester_week_only macro-F1 | 0.369 |

结果解释：

- rolling7、rolling3、lag1 都优于 no-temporal。
- rolling7 最好，但只比 rolling3 高 0.002。
- Temporal features 的提升幅度小，应写成 modest improvement。
- Semester week alone 有一定提升，但不如 wearable lag/rolling features。

## Best temporal model confusion matrix

Best condition: rolling7_wearable + Gradient Boosting.

| Actual | Predicted Low | Predicted Medium | Predicted High |
|---|---:|---:|---:|
| Low (n=386) | 157 (40.7%) | 119 (30.8%) | 110 (28.5%) |
| Medium (n=267) | 68 (25.5%) | 84 (31.5%) | 115 (43.1%) |
| High (n=248) | 69 (27.8%) | 61 (24.6%) | 118 (47.6%) |

Error pattern:

- Medium F1 = 0.316，仍然是三个类别中最低。
- Actual Medium 最常被预测为 High：115 个，占 Medium 行的 43.1%。
- Actual High 中 118 个被正确预测为 High，占 High 行的 47.6%；69 个被误分为 Low，占 27.8%。
- Actual Low 中 157 个被正确预测为 Low，占 Low 行的 40.7%；其余被分散误分为 Medium 和 High。
- High 类 F1 = 0.399，相比 RQ1 best model 的 High F1 = 0.330 有明显改善。
- Low F1 从 RQ1 的 0.480 降到 0.462，说明 temporal features 提升主要来自 Medium/High，而不是 Low。

## Weekly stress trend

Strict EDA 显示压力存在周级变化：

| 指标 | 结果 |
|---|---|
| Lowest weekly mean stress | Week 21, M = 13.62, SD = 12.92 |
| Highest weekly mean stress | Week 6, M = 40.59, SD = 27.21 |

图：

![Strict weekly stress trend](figures/strict_weekly_stress_trend.svg)

如何使用：

- 这张图可以支持 RQ3 的动机：学生压力不是时间上完全静态的。
- 周级趋势说明 semester progression 可能含有信息。
- 但最终模型结果显示，仅加入 semester week 的提升有限，wearable history 更有用。

## 为什么 rolling features 有帮助

可用解释：

- 压力反应可能不是单日即时信号，而是多日睡眠、生理恢复和活动模式的累积结果。
- Rolling mean 能平滑 wearable 日级噪声。
- 7 天窗口可能捕捉一周生活节律，比单日 lag 更稳定。
- Gradient Boosting 适合利用 rolling features 中的非线性分割。

## 为什么提升有限

可用解释：

- temporal features 将特征数从 12 增加到 24，但学生数仍只有 35。
- wearable history 仍然不包含个体主观压力标准，无法完全解决自报标签噪声。
- Subject-aware split 要求模型泛化到 unseen students，历史特征只能提供通用 temporal signal。
- 缺失 wearable 数据会影响 lag/rolling features 的稳定性。

## 可用于 Results 的要点

- rolling7_wearable + Gradient Boosting 取得最高 test macro-F1 = 0.392。
- 与 no_temporal best macro-F1 = 0.357 相比，提升 +0.035。
- rolling3 和 lag1 结果接近，分别为 0.390 和 0.389。
- semester_week_only 达到 0.369，说明 calendar time 有一定信息，但不如 wearable temporal history。

## 可用于 Discussion 的要点

- RQ3 支持 temporal context 对 stress prediction 有小幅帮助。
- rolling features 可能通过平滑噪声和捕捉累积生理状态改善预测。
- 结果不能写成大幅提升，因为提升只有 +0.035 macro-F1。
- Medium 类仍然最难预测，说明 temporal context 没有完全解决类别边界模糊问题。

## 可用于 Conclusion 的要点

RQ3 的回答：加入 wearable temporal context 可以带来小幅改善，尤其是 7-day rolling features，但提升有限，应解释为 weak-to-moderate evidence rather than strong temporal predictability。
