# modeling_outputs 目录说明

这个目录现在按“旧流程”和“严格防泄漏流程”分开保存，避免把中间数据、模型结果和验证记录混在一起。

## 推荐阅读顺序

优先看 `strict_pipeline/`。这是从 `SSAQS dataset/` 原始文件重新生成的版本，并且把缺失值填补、标准化、模型调参都放进训练流程内部，主要用于最终报告。

`legacy_pipeline/` 是早期基于 `final_student_day_table_v01_processed.csv` 的旧版本结果，保留它是为了追踪项目演进，不建议作为最终主结果。

## strict_pipeline

| 文件夹 | 内容 | 对应脚本 |
| --- | --- | --- |
| `00_raw_audit/` | 检查原始 SSAQS 文件是否齐全、每个学生有多少可用天数 | `scripts/audit_raw_ssaqs_data.py` |
| `01_raw_student_day/` | 从原始文件合并出的 student-day 原始表 | `scripts/build_raw_student_day_table.py` |
| `02_cleaned_student_day/` | 处理重复 student-day、重新生成压力标签后的严格清洗表 | `scripts/clean_raw_student_day_table.py` |
| `03_model_data/` | 加入日历特征、按学生划分 train/test 后的建模数据 | `scripts/prepare_strict_model_data.py` |
| `04_rq1/` | RQ1 模型比较结果、调参结果、混淆矩阵和中文总结 | `scripts/run_strict_rq1_models.py` |
| `05_rq2/` | RQ2 特征组对比结果、调参结果和中文总结 | `scripts/run_strict_rq2_feature_groups.py` |
| `99_validation/` | 严格流程的验证记录 | 手动记录 |

## legacy_pipeline

| 文件夹 | 内容 |
| --- | --- |
| `00_model_data/` | 旧版 processed 数据清理后的建模表和特征配置 |
| `01_rq1/` | 旧版 RQ1 结果 |
| `02_rq2/` | 旧版 RQ2 结果 |
| `99_validation/` | 旧版验证记录 |

## 关于“填补数据”

严格流程中的 `strict_model_data.csv` 仍然保留缺失值，没有在这一步提前填补。模型运行时会使用 scikit-learn pipeline 里的 `SimpleImputer` 填补缺失值：

- 在交叉验证时，每个训练折只用本折训练数据学习填补规则。
- 在最终测试时，只用训练集学习填补规则，再应用到测试集。
- 测试集不会提前参与填补、标准化或模型选择。

所以这不是“凭空造出更多样本”，也不是把测试答案泄漏给模型；它只是让模型在遇到缺失特征时有一个统一处理规则。
