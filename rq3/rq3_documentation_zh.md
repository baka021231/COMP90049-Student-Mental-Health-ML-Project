# RQ3 中文说明文档：时间模式与滞后特征

## 1. 研究问题

RQ3 关注的问题是：**学生压力是否存在明显的学期时间变化模式，以及加入时间相关特征后，压力分类模型的表现是否会提升。**

本部分分析分为两个任务：

1. 画出整个学期中每一周的平均自评压力变化趋势，即 Figure 4。
2. 比较不同时间特征配置下的分类性能，即 Table 5。

分类任务的目标变量是 `stress_label`，包含三个类别：

- `Low`
- `Medium`
- `High`

连续变量 `stress` 用于描述每周压力趋势，也用于构造“前一天压力”等历史特征。但是，当前当天的 `stress` 没有直接放入 baseline 模型，因为 `stress_label` 很可能是由 `stress` 分数划分出来的。如果把当天 `stress` 作为输入，会造成明显的数据泄漏。

## 2. 数据结构

当前数据是 student-day level 的表格数据，也就是说，每一行代表某个学生在某一天的记录。

主要字段包括：

| 字段 | 含义 |
|---|---|
| `student_id` | 学生编号 |
| `date` | 日期 |
| `stress` | 连续型自评压力分数 |
| `stress_label` | 压力分类标签，即预测目标 |
| `anxiety` | 焦虑相关变量 |
| `sleep_score` | 睡眠得分 |
| `deep_sleep_minutes` | 深睡眠时长 |
| `total_steps` | 总步数 |
| `sedentary_minutes` | 久坐时长 |
| `lightly_active_minutes` | 轻度活动时长 |
| `moderately_active_minutes` | 中度活动时长 |
| `very_active_minutes` | 高强度活动时长 |
| `avg_rmssd` | HRV 指标 |
| `avg_low_frequency` | 低频 HRV 指标 |
| `avg_high_frequency` | 高频 HRV 指标 |
| `avg_oxygen` | 平均血氧 |
| `std_oxygen` | 血氧标准差 |
| `STRESS_SCORE` | 设备或系统生成的压力分数 |
| `CALCULATION_FAILED` | 压力分数计算是否失败 |

## 3. 学期周变量的构造

为了分析压力随学期推进的变化，代码根据 `date` 构造了 `semester_week`：

```python
df["semester_week"] = ((df["date"] - df["date"].min()).dt.days // 7) + 1
```

含义是：

- 数据集中最早的一天属于第 1 周。
- 每 7 天作为一个新的 semester week。
- 这样可以把每日数据聚合到“学期周”层面。

## 4. Figure 4：学期压力趋势

Figure 4 展示的是按 `semester_week` 聚合后的平均自评压力水平。

每一周计算以下统计量：

- 该周观测数量 `n`
- 平均压力 `mean_stress`
- 压力标准差 `sd_stress`
- 该周起始日期
- 该周结束日期

图中：

- 折线表示每周平均压力。
- 阴影区域表示正负一个标准差，即 `mean ± SD`。

### 4.1 主要发现

从当前数据结果看，平均自评压力在第 3 周最低：

```text
M = 27.31, SD = 20.30
```

平均自评压力在第 19 周最高：

```text
M = 51.63, SD = 31.27
```

整体趋势可以概括为：

> 学期初压力处于中等水平，随后在早期下降，并在第 3 周达到最低点；之后压力在第 5-8 周出现明显上升，可能对应学期中早期任务或阶段性评估压力；第 9-12 周压力有所回落；进入学期后期后，压力再次上升，并在第 17-19 周达到最高水平。

这个趋势说明，学生压力并不是随机波动，而是和学期进程存在一定关系。尤其是学期末阶段，平均压力明显升高，可能反映了期末考试、课程项目、论文截止日期等集中压力源。

## 5. Table 5：时间特征配置对比

Table 5 比较了五种特征配置下的分类表现。为了保证比较公平，五种配置都使用同一个模型，即 Tabular Transformer。

五种配置如下：

| 配置 | 说明 |
|---|---|
| No temporal features | 非时间 baseline，只使用当天的非泄漏基础特征 |
| Semester week only | 在 baseline 上加入 `semester_week` |
| One-day lag features | 加入前一天的历史特征 |
| Three-day rolling means | 加入过去三天的滚动均值 |
| Seven-day rolling means | 加入过去七天的滚动均值 |

## 6. 滞后特征与滚动均值特征

### 6.1 One-day lag features

One-day lag features 表示同一个学生前一天的变量值。例如，前一天压力通过下面的方式计算：

```python
df[f"{col}_lag1"] = df.groupby("student_id")[col].shift(1)
```

这里使用 `groupby("student_id")` 是为了确保每个学生的历史只来自自己，而不会把其他学生的数据错接进来。

### 6.2 Rolling mean features

Rolling mean features 表示过去几天的平均状态。例如，过去七天均值的计算方式是：

```python
df[f"{col}_roll7_mean"] = df.groupby("student_id")[col].transform(
    lambda s: s.shift(1).rolling(7, min_periods=1).mean()
)
```

这里有一个非常重要的细节：先 `shift(1)`，再 `rolling()`。

这样做的原因是：

- `shift(1)` 确保只使用预测日之前的数据。
- `rolling()` 再计算过去窗口内的均值。
- 这样可以避免使用当天信息预测当天标签，从而防止数据泄漏。

## 7. 模型：Tabular Transformer

当前代码使用的是基于 PyTorch 实现的 Tabular Transformer，而不是传统机器学习模型。

模型类是：

```python
class TabularTransformer(nn.Module):
```

模型逻辑如下：

1. 每个数值特征会被转换成一个 feature token。
2. 类别特征 `CALCULATION_FAILED` 通过 embedding 表示。
3. 模型加入一个可学习的 `[CLS]` token。
4. 所有 token 输入到 `TransformerEncoder`。
5. 最终使用 `[CLS]` token 的输出表示进行三分类预测。

也就是说，这个模型不是把所有特征简单拼接后直接输入全连接层，而是把表格特征看作一组 token，让 Transformer 学习不同特征之间的关系。

## 8. 模型训练与评估

训练设置如下：

| 项目 | 设置 |
|---|---|
| 框架 | PyTorch |
| 模型 | Tabular Transformer |
| 损失函数 | CrossEntropyLoss |
| 优化器 | AdamW |
| 评估方式 | 5-fold Stratified Cross-Validation |
| 指标 | Macro-F1 |
| 类别不平衡处理 | class weights |

使用 Macro-F1 的原因是，压力类别存在一定不平衡。Macro-F1 会分别计算每个类别的 F1，再取平均，因此不会让样本最多的类别主导整体结果。

## 9. 当前 Table 5 结果

当前 Transformer 结果如下：

| Feature configuration | Model | Macro-F1 | SD | Delta vs baseline |
|---|---:|---:|---:|---:|
| No temporal features | Tabular Transformer | 0.6096 | 0.0176 | 0.0000 |
| Semester week only | Tabular Transformer | 0.6330 | 0.0343 | +0.0234 |
| One-day lag features | Tabular Transformer | 0.6451 | 0.0261 | +0.0355 |
| Three-day rolling means | Tabular Transformer | 0.6375 | 0.0295 | +0.0279 |
| Seven-day rolling means | Tabular Transformer | 0.6309 | 0.0268 | +0.0213 |

最佳配置是：

```text
One-day lag features
```

对应的 Macro-F1 为：

```text
0.6451
```

相比非时间 baseline 的提升为：

```text
+0.0355
```

## 10. 结果解释

结果说明，加入时间信息后，模型对压力类别的预测能力确实有所提升。

首先，只加入 `semester_week` 就能把 Macro-F1 从 0.6096 提升到 0.6330。这说明学生压力和学期进程存在关系，模型可以从“当前处于学期第几周”中获得有用信息。

其次，加入 one-day lag features 后取得最佳表现，Macro-F1 达到 0.6451。这表明学生前一天的压力、睡眠、活动和生理状态对当天压力分类具有较强预测价值。

相比之下，三天和七天滚动均值虽然也优于 baseline，但没有超过 one-day lag features。这可能说明，在当前数据中，短期即时变化比更长窗口的平滑平均值更有信息量。换句话说，学生压力状态可能更受最近一天状态影响，而不是简单由过去多天平均水平决定。

## 11. 可直接写入报告的中文表述

RQ3 分析了学生自评压力在学期时间维度上的变化趋势，并进一步检验时间特征是否能够提升压力分类表现。Figure 4 显示，平均自评压力在学期早期下降，并在第 3 周达到最低水平（M = 27.31, SD = 20.30）；随后压力在第 5-8 周上升，第 9-12 周有所回落，并在学期后期再次升高。最高平均压力出现在第 19 周（M = 51.63, SD = 31.27），说明学期末阶段可能是学生压力最集中的时期。Table 5 使用固定的 Tabular Transformer 模型比较了五种时间特征配置。结果显示，加入时间信息整体上提升了分类表现，其中 one-day lag features 取得最高 Macro-F1（0.645），相比非时间 baseline（0.610）提升了 +0.036。这表明学生前一天的压力和行为生理状态对当前压力类别具有重要预测价值。

## 12. 可直接写入报告的英文表述

Figure 4 illustrates the mean self-reported stress level aggregated by semester week across all students. The trend shows an early-semester decline, reaching its lowest point in week 3 (M = 27.31, SD = 20.30), followed by a noticeable increase around weeks 5-8. Stress then drops again during weeks 9-12 before rising toward the end of the semester, with the highest mean stress observed in week 19 (M = 51.63, SD = 31.27). Table 5 compares classification performance across five feature configurations using a fixed Tabular Transformer model. Adding temporal information improved performance over the non-temporal baseline. The one-day lag feature configuration achieved the highest Macro-F1 of 0.645, representing a +0.036 improvement over the baseline Macro-F1 of 0.610.

## 13. 文件位置说明

当前项目结构中，RQ3 相关脚本和文档位于：

```text
rq3/
```

主要文件包括：

| 文件 | 说明 |
|---|---|
| `rq3/rq3_temporal_analysis.py` | RQ3 分析脚本 |
| `rq3/rq3_documentation.md` | 英文说明文档 |
| `rq3/rq3_documentation_zh.md` | 中文说明文档 |

运行脚本后会生成以下输出：

| 输出文件 | 说明 |
|---|---|
| `rq3_outputs/figure4_temporal_stress_trend.png` | Figure 4 折线图 |
| `rq3_outputs/figure4_weekly_stress_summary.csv` | 每周压力统计表 |
| `rq3_outputs/table5_temporal_feature_variants.csv` | Table 5 结果 |
| `rq3_outputs/rq3_summary.txt` | 自动生成的结果摘要 |

