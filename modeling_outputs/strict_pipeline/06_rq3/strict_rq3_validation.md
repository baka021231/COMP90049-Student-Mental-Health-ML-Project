# 严格版 RQ3 验证记录

## 已验证内容

- 脚本可以通过 Python 编译检查。
- 脚本可以从项目根目录运行。
- 输入数据来自严格版建模数据目录。
- 输出文件写入 `modeling_outputs/strict_pipeline/06_rq3/`。
- RQ3 使用与 RQ1/RQ2 相同的 subject-aware train/test split。
- 缺失值填补和标准化在模型 pipeline 内完成。
- lag/rolling 特征只使用同一学生过去日期的数据。
- test set 不参与模型选择或超参数选择。

## 实验规模

- temporal condition 数：5
- 模型结果行数：35
