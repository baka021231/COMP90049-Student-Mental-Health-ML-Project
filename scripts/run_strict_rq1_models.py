from __future__ import annotations

from pathlib import Path

import pandas as pd
from modeling_utils import LABEL_ORDER
from modeling_utils import build_strict_model_specs
from modeling_utils import dataframe_to_markdown
from modeling_utils import evaluate_models
from modeling_utils import load_feature_sets
from modeling_utils import tune_models_with_group_cv


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT_DIR = ROOT / "modeling_outputs" / "strict_pipeline"
DATA_DIR = BASE_OUT_DIR / "03_model_data"
OUT_DIR = BASE_OUT_DIR / "04_rq1"
DATA_PATH = DATA_DIR / "strict_model_data.csv"
FEATURE_SETS_PATH = DATA_DIR / "strict_feature_sets.json"
RESULTS_PATH = OUT_DIR / "strict_rq1_results.csv"
TUNING_PATH = OUT_DIR / "strict_rq1_tuning_results.csv"
SUMMARY_PATH = OUT_DIR / "strict_rq1_summary.md"

RQ1_FEATURE_SET_NAME = "rq1_all_wearable"
TARGET_COLUMN = "stress_label"
SPLIT_COLUMN = "split"
GROUP_COLUMN = "student_id"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    return data


def load_features() -> list[str]:
    feature_sets = load_feature_sets(FEATURE_SETS_PATH)
    if RQ1_FEATURE_SET_NAME not in feature_sets:
        raise KeyError(f"Missing feature set: {RQ1_FEATURE_SET_NAME}")
    return feature_sets[RQ1_FEATURE_SET_NAME]


def validate_feature_columns(data: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing strict RQ1 feature columns: {missing}")


def make_train_test(
    data: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    test = data[data[SPLIT_COLUMN] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Both train and test splits must contain rows.")

    X_train = train[features]
    y_train = train[TARGET_COLUMN]
    groups_train = train[GROUP_COLUMN]
    X_test = test[features]
    y_test = test[TARGET_COLUMN]
    return X_train, y_train, groups_train, X_test, y_test


def confusion_matrix_path(model_name: str) -> Path:
    return OUT_DIR / f"strict_rq1_confusion_matrix_{model_name}.csv"


def save_outputs(
    results: pd.DataFrame,
    tuning_results: pd.DataFrame,
    confusion_matrices: dict[str, pd.DataFrame],
) -> None:
    merged = results.merge(tuning_results, on="model", how="left")
    merged.to_csv(RESULTS_PATH, index=False)
    tuning_results.to_csv(TUNING_PATH, index=False)
    for model_name, matrix in confusion_matrices.items():
        matrix.to_csv(confusion_matrix_path(model_name))


def write_summary(
    results: pd.DataFrame,
    tuning_results: pd.DataFrame,
    features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    merged = results.merge(tuning_results, on="model", how="left")
    best = merged.sort_values("macro_f1", ascending=False).iloc[0]
    baseline = merged[merged["model"] == "majority_baseline"].iloc[0]
    best_model = str(best["model"])
    best_matrix = pd.read_csv(confusion_matrix_path(best_model), index_col=0)
    hardest_class = (
        best[["low_f1", "medium_f1", "high_f1"]]
        .astype(float)
        .rename(index={"low_f1": "Low", "medium_f1": "Medium", "high_f1": "High"})
        .idxmin()
    )

    compact_results = merged[
        [
            "model",
            "accuracy",
            "macro_f1",
            "low_f1",
            "medium_f1",
            "high_f1",
            "cv_macro_f1",
            "best_params",
        ]
    ].sort_values("macro_f1", ascending=False)
    numeric_columns = [
        "accuracy",
        "macro_f1",
        "low_f1",
        "medium_f1",
        "high_f1",
        "cv_macro_f1",
    ]
    for column in numeric_columns:
        compact_results[column] = pd.to_numeric(compact_results[column], errors="coerce")

    lines = [
        "# 严格版 RQ1 建模总结",
        "",
        "## 研究问题",
        "",
        "RQ1 关注：只使用 wearable 特征，能否预测学生当天的压力等级。",
        "",
        "## 严格实验设计",
        "",
        "- 输入数据来自 `strict_model_data.csv`，该表没有提前做全局填补或全局标准化。",
        "- 使用 `strict_feature_sets.json` 中的 `rq1_all_wearable`。",
        "- train/test 已按学生划分，同一个学生不会同时出现在 train 和 test。",
        "- 缺失值填补使用 `SimpleImputer`，并放在模型 pipeline 内部。",
        "- Logistic Regression、SVM、kNN、MLP 的标准化也放在 pipeline 内部。",
        "- 除 majority baseline 外，模型超参数通过训练集内部的 GroupKFold 交叉验证选择。",
        "- test set 只用于最终评估，不用于选择模型或调参。",
        "",
        "## 数据规模",
        "",
        f"- 特征数：{len(features)}",
        f"- 训练行数：{len(X_train)}",
        f"- 测试行数：{len(X_test)}",
        f"- 训练标签分布：{y_train.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}",
        f"- 测试标签分布：{y_test.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}",
        "",
        "## 结果",
        "",
        dataframe_to_markdown(compact_results),
        "",
        "## 关键发现",
        "",
        f"- test macro-F1 最高的模型是 `{best_model}`。",
        f"- 最佳 test macro-F1：{float(best['macro_f1']):.3f}。",
        f"- majority baseline test macro-F1：{float(baseline['macro_f1']):.3f}。",
        f"- 最佳模型最难预测的类别是 `{hardest_class}`。",
        "- 因为本实验使用原始未标准化数据，并在 pipeline 内部完成填补和标准化，所以比旧版实验更能避免 preprocessing leakage。",
        "",
        f"## 最佳模型 `{best_model}` 的 Confusion Matrix",
        "",
        dataframe_to_markdown(best_matrix.reset_index().rename(columns={"index": "actual"})),
        "",
        "## 报告写作建议",
        "",
        "报告中应强调：严格版 RQ1 的目标不是追求高 accuracy，而是验证 wearable-only signal 是否在无 leakage 的设置下仍然提供一定预测能力。若分数下降，也应解释为更严格评估带来的合理结果。",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    features = load_features()
    validate_feature_columns(data, features)
    X_train, y_train, groups_train, X_test, y_test = make_train_test(data, features)

    fitted_models, tuning_results = tune_models_with_group_cv(
        build_strict_model_specs(),
        X_train,
        y_train,
        groups_train,
    )
    results, confusion_matrices = evaluate_models(fitted_models, X_test, y_test)
    save_outputs(results, tuning_results, confusion_matrices)
    write_summary(results, tuning_results, features, X_train, X_test, y_train, y_test)

    print("Strict RQ1 modeling complete.")
    print(f"Features: {len(features)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {TUNING_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
