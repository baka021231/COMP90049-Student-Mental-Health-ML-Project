# COMP90049 A2 最终报告素材总览

本文档是写作最终 ACL 报告的素材入口。所有数字、图表和结论要点均来自 strict pipeline，目的是支持完整报告写作：Introduction、Literature Review、Method、Results、Discussion、Conclusion、GenAI Declaration 和 Bibliography。

## 项目基本信息

| 项目项 | 内容 |
|---|---|
| 课程 | COMP90049 Introduction to Machine Learning |
| 任务 | 使用 Fitbit 可穿戴传感器数据预测大学生每日自报压力等级 |
| 数据集 | SSAQS |
| 数据集 URL | `https://doi.org/10.1038/s41597-026-07085-7` |
| 数据采集对象 | 35 名本科生 |
| 数据采集环境 | 两所墨西哥大学，一个完整学期 |
| 设备 | Fitbit Inspire 3 |
| 原始模态 | daily questionnaire、sleep、activity level、steps、HRV、SpO2、Fitbit stress score |
| 最终主任务输入 | wearable-derived features，不使用同日 `stress`、`anxiety`、Fitbit `STRESS_SCORE` |
| 预测目标 | 每个 student-day 的自报压力等级：Low / Medium / High |
| 主评估指标 | macro-F1 |

## 研究问题

| RQ | 问题 | 对应素材 |
|---|---|---|
| RQ1 | 仅使用 wearable 特征能否预测学生当天自报压力等级？ | `rq1/rq1_report.md` |
| RQ2 | 不同 wearable feature group 对压力预测的贡献是否不同？ | `rq2/rq2_report.md` |
| RQ3 | 加入时间上下文、lag 特征和 rolling 特征是否改善预测？ | `rq3/rq3_report.md` |

EDA 和数据处理素材位于 `eda/eda_report.md`。
补充素材位于 `supplementary/supplementary_report.md`，包括超参数汇总表、RQ3 最佳模型混淆矩阵图和 RQ2 feature importance 图。
Sensitivity / robustness 素材位于 `sensitivity/sensitivity_report.md`，包括 student-level bootstrap CI、leave-one-student-out sensitivity 和 within-person deviation feature experiment。

## 核心数据事实

| 项 | 值 |
|---|---:|
| strict cleaned student-day observations | 3118 |
| students | 35 |
| unique `student_id + date` pairs | 3118 |
| duplicate student-days after cleaning | 0 |
| date range | 2025-02-14 to 2025-07-09 |
| train students | 26 |
| test students | 9 |
| train rows | 2217 |
| test rows | 901 |
| random seed | 49 |

## 标签定义

| Label | Stress score range | Count | Percent |
|---|---:|---:|---:|
| Low | 0-17 | 1082 | 34.7% |
| Medium | 18-38 | 984 | 31.6% |
| High | 39-100 | 1052 | 33.7% |

## 主要结果总览

| 实验 | 最佳条件 | 最佳模型 | Test macro-F1 | 关键解释 |
|---|---|---|---:|---|
| RQ1 | all wearable features | MLP | 0.357 | wearable-only 信号优于 majority baseline，但绝对性能 modest |
| RQ2 | HRV + SpO2 only | Random Forest | 0.393 | autonomic / oxygenation 特征比完整 wearable 特征组更有效 |
| RQ3 | rolling7 wearable | Gradient Boosting | 0.392 | rolling temporal context 带来小幅提升 |
| Sensitivity | within-person deviation features | Gradient Boosting | 0.407 | 相对个人历史基线的 wearable deviation 带来额外提升 |

Majority baseline 在 RQ1 的 test macro-F1 为 0.144。

## 推荐报告结构与素材映射

| 报告章节 | 应使用素材 |
|---|---|
| Introduction | 数据集背景、任务目标、三个 RQ、公共数据集 URL |
| Literature Review | SSAQS 原始数据集论文 + wearable stress detection / student stress prediction 文献 |
| Method: Dataset | `eda/eda_report.md` 的数据规模、标签分布、split、missingness |
| Method: Preprocessing | `eda/eda_report.md` 的重复处理、subject-aware split、pipeline imputation/scaling |
| Method: Models | `rq1/rq1_report.md` 的模型集合和 `supplementary/supplementary_report.md` 的正式超参数表 |
| Results: RQ1 | `rq1/rq1_report.md` 的模型结果表和 confusion matrix |
| Results: RQ2 | `rq2/rq2_report.md` 的 feature group 结果和图 |
| Results: RQ3 | `rq3/rq3_report.md` 的 temporal condition 结果和 weekly trend |
| Sensitivity / Robustness | `sensitivity/sensitivity_report.md` 的 bootstrap CI、LOSO sensitivity、within-person deviation |
| Discussion | 各模块末尾的“可用于 Discussion 的要点” |
| Conclusion | 汇总三个 RQ 的回答：wearable 有弱信号，HRV/SpO2 和 sleep 更有效，temporal rolling 有小幅收益 |

## 可直接放入报告的图

| 建议编号 | 图 | 路径 | 用途 |
|---|---|---|---|
| Figure 1 | Label distribution | `eda/figures/strict_label_distribution.svg` | Dataset statistics |
| Figure 2 | RQ1 confusion matrix | `rq1/figures/figure2_rq1_mlp_confusion_matrix.svg` | RQ1 error analysis |
| Figure 3 | RQ2 feature group macro-F1 | `rq2/figures/figure4_rq2_feature_group_macro_f1.svg` | Feature ablation |
| Figure 4 | RQ3 temporal condition macro-F1 | `rq3/figures/figure5_rq3_temporal_condition_macro_f1.svg` | Temporal feature comparison |
| Optional | Weekly stress trend | `rq3/figures/strict_weekly_stress_trend.svg` | Temporal EDA / RQ3 context |
| Optional | Missingness | `eda/figures/strict_feature_missingness.svg` | Preprocessing justification |
| Optional | RQ3 best confusion matrix | `supplementary/figures/rq3_best_confusion_matrix_rolling7_gb.svg` | RQ3 error analysis |
| Optional | RQ2 RF feature importance | `supplementary/figures/rq2_hrv_spo2_rf_feature_importance.svg` | RQ2 feature interpretation |
| Optional | Bootstrap CI | `sensitivity/figures/bootstrap_macro_f1_ci.svg` | Test-set uncertainty |
| Optional | Within-person deviation comparison | `sensitivity/figures/within_person_deviation_macro_f1.svg` | Personalised feature sensitivity |

## 推荐放入报告的表

| 表 | 内容 | 来源 |
|---|---|---|
| Dataset summary | N、students、split、label distribution | `eda/eda_report.md` |
| Hyperparameter summary | model、preprocessing、调参方法、候选范围、RQ1最终值 | `supplementary/supplementary_report.md` |
| RQ1 model comparison | model、accuracy、macro-F1、per-class F1、CV macro-F1 | `rq1/rq1_report.md` |
| RQ2 feature group results | feature group、best model、macro-F1、per-class F1 | `rq2/rq2_report.md` |
| RQ3 temporal feature results | temporal condition、best model、macro-F1、delta | `rq3/rq3_report.md` |
| Bootstrap CI | 三个主结果的 point estimate 和 95% CI | `sensitivity/sensitivity_report.md` |
| Within-person deviation | no-temporal vs deviation features | `sensitivity/sensitivity_report.md` |

## 参考文献素材

可在 Bibliography 中使用的文献条目需最终核对作者和格式：

| 文献 | 报告用途 |
|---|---|
| Garcia-Ceja et al. (2026), SSAQS dataset, Scientific Data, DOI `10.1038/s41597-026-07085-7` | 数据集来源 |
| Saylam and Incel (2024), smartwatch/wristband stress detection review | wearable stress detection 背景；HRV、ensemble、SVM、temporal features |
| Razavi et al. (2023), college student wearable stress prediction | naturalistic student stress prediction benchmark |
| Liu et al. (2026), Fitbit-based student mental health screening | Fitbit stress classification benchmark |
| Bloomfield et al. (2024), sleep/HRV/academic calendar and student stress | 支撑 RQ2/RQ3 对 sleep、HRV、semester week 的解释 |

## 文件夹结构

```text
report_materials/
├── README.md
├── eda/
│   ├── eda_report.md
│   └── figures/
├── rq1/
│   ├── rq1_report.md
│   └── figures/
├── rq2/
│   ├── rq2_report.md
│   └── figures/
└── rq3/
    ├── rq3_report.md
    └── figures/
└── supplementary/
    ├── supplementary_report.md
    ├── hyperparameter_summary_table.csv
    ├── rq2_hrv_spo2_rf_feature_importance.csv
    └── figures/
└── sensitivity/
    ├── sensitivity_report.md
    ├── student_bootstrap_ci.csv
    ├── train_group_cv_best_model_sensitivity.csv
    ├── within_person_deviation_results.csv
    ├── within_person_deviation_tuning.csv
    └── figures/
```
