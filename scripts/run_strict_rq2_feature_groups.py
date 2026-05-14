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
OUT_DIR = ROOT / "modeling_outputs"
DATA_PATH = OUT_DIR / "strict_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "strict_feature_sets.json"
RESULTS_PATH = OUT_DIR / "strict_rq2_feature_group_results.csv"
BEST_RESULTS_PATH = OUT_DIR / "strict_rq2_best_by_feature_group.csv"
TUNING_PATH = OUT_DIR / "strict_rq2_tuning_results.csv"
SUMMARY_PATH = OUT_DIR / "strict_rq2_summary.md"

RQ2_FEATURE_GROUPS_NAME = "rq2_feature_groups"
TARGET_COLUMN = "stress_label"
SPLIT_COLUMN = "split"
GROUP_COLUMN = "student_id"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    return data


def load_feature_groups() -> dict[str, list[str]]:
    feature_sets = load_feature_sets(FEATURE_SETS_PATH)
    if RQ2_FEATURE_GROUPS_NAME not in feature_sets:
        raise KeyError(f"Missing feature group set: {RQ2_FEATURE_GROUPS_NAME}")
    return feature_sets[RQ2_FEATURE_GROUPS_NAME]


def validate_feature_columns(data: pd.DataFrame, features: list[str], group_name: str) -> None:
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing strict RQ2 feature columns for {group_name}: {missing}")


def run_feature_group(
    data: pd.DataFrame, feature_group: str, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_feature_columns(data, features, feature_group)
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    test = data[data[SPLIT_COLUMN] == "test"].copy()
    X_train = train[features]
    y_train = train[TARGET_COLUMN]
    groups_train = train[GROUP_COLUMN]
    X_test = test[features]
    y_test = test[TARGET_COLUMN]

    fitted_models, tuning_results = tune_models_with_group_cv(
        build_strict_model_specs(),
        X_train,
        y_train,
        groups_train,
    )
    results, _ = evaluate_models(fitted_models, X_test, y_test)

    results.insert(0, "feature_group", feature_group)
    results.insert(1, "n_features", len(features))
    tuning_results.insert(0, "feature_group", feature_group)
    tuning_results.insert(1, "n_features", len(features))
    return results, tuning_results


def run_all_feature_groups(
    data: pd.DataFrame, feature_groups: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames = []
    tuning_frames = []
    for feature_group, features in feature_groups.items():
        results, tuning = run_feature_group(data, feature_group, features)
        result_frames.append(results)
        tuning_frames.append(tuning)
    return (
        pd.concat(result_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
    )


def best_results_by_feature_group(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["feature_group", "macro_f1"], ascending=[True, False])
        .groupby("feature_group", as_index=False, observed=True)
        .head(1)
        .reset_index(drop=True)
    )


def save_outputs(
    results: pd.DataFrame,
    tuning_results: pd.DataFrame,
    best_by_group: pd.DataFrame,
) -> pd.DataFrame:
    merged = results.merge(
        tuning_results,
        on=["feature_group", "n_features", "model"],
        how="left",
    )
    merged.to_csv(RESULTS_PATH, index=False)
    tuning_results.to_csv(TUNING_PATH, index=False)
    best_by_group.merge(
        tuning_results,
        on=["feature_group", "n_features", "model"],
        how="left",
    ).to_csv(BEST_RESULTS_PATH, index=False)
    return merged


def write_summary(
    merged_results: pd.DataFrame,
    best_by_group: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> None:
    best_table = best_by_group.merge(
        merged_results[["feature_group", "n_features", "model", "cv_macro_f1", "best_params"]],
        on=["feature_group", "n_features", "model"],
        how="left",
    )
    overall_best = merged_results.sort_values("macro_f1", ascending=False).iloc[0]
    weakest_group = str(best_by_group.sort_values("macro_f1").iloc[0]["feature_group"])

    compact_best = best_table[
        [
            "feature_group",
            "n_features",
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
    compact_results = merged_results[
        [
            "feature_group",
            "n_features",
            "model",
            "accuracy",
            "macro_f1",
            "cv_macro_f1",
            "best_params",
        ]
    ].sort_values(["feature_group", "macro_f1"], ascending=[True, False])
    for frame in [compact_best, compact_results]:
        for column in ["accuracy", "macro_f1", "low_f1", "medium_f1", "high_f1", "cv_macro_f1"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    lines = [
        "# 严格版 RQ2 Feature Group 总结",
        "",
        "## 研究问题",
        "",
        "RQ2 关注：不同 wearable feature group 对压力预测的贡献有多大。",
        "",
        "## 严格实验设计",
        "",
        "- 输入数据来自 `strict_model_data.csv`，没有提前做全局填补或标准化。",
        "- 每个 feature group 都使用同一套模型集合和同一套 train/test split。",
        "- 每个模型都在训练集内部使用 GroupKFold 进行超参数选择。",
        "- test set 只用于最终评估，不用于选择 feature group、模型或超参数。",
        "- 缺失值填补和必要的标准化都放在模型 pipeline 内部。",
        "",
        "## Feature Groups",
        "",
    ]
    for group_name, features in feature_groups.items():
        lines.append(f"- `{group_name}`：{len(features)} 个特征")

    lines.extend(
        [
            "",
            "## 每个 Feature Group 的最佳结果",
            "",
            dataframe_to_markdown(compact_best),
            "",
            "## 完整结果表",
            "",
            dataframe_to_markdown(compact_results),
            "",
            "## 关键发现",
            "",
            f"- 整体 test macro-F1 最高的是 `{overall_best['feature_group']}` + `{overall_best['model']}`。",
            f"- 最高 test macro-F1：{float(overall_best['macro_f1']):.3f}。",
            f"- 最弱的 feature group 是 `{weakest_group}`。",
            "- RQ2 应作为 ablation study 来写，重点解释不同模态的相对贡献，而不是只比较绝对分数。",
            "",
            "## 报告写作建议",
            "",
            "如果严格版分数低于旧版结果，应解释为：旧版 processed 表可能提前做了全局处理，而严格版把填补、标准化和调参都限制在训练流程内部，因此评估更保守、更可信。",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_data()
    feature_groups = load_feature_groups()
    results, tuning_results = run_all_feature_groups(data, feature_groups)
    best_by_group = best_results_by_feature_group(results)
    merged = save_outputs(results, tuning_results, best_by_group)
    write_summary(merged, best_by_group, feature_groups)

    print("Strict RQ2 feature group modeling complete.")
    print(f"Feature groups: {', '.join(feature_groups)}")
    print(f"Experiment rows: {len(merged)}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {BEST_RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {TUNING_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
