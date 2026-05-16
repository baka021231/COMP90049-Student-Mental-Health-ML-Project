from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from modeling_utils import LABEL_ORDER  # noqa: E402
from modeling_utils import build_strict_model_specs  # noqa: E402
from modeling_utils import dataframe_to_markdown  # noqa: E402
from modeling_utils import evaluate_models  # noqa: E402
from modeling_utils import load_feature_sets  # noqa: E402
from modeling_utils import tune_models_with_group_cv  # noqa: E402


BASE_OUT_DIR = ROOT / "modeling_outputs" / "strict_pipeline"
DATA_DIR = BASE_OUT_DIR / "03_model_data"
OUT_DIR = BASE_OUT_DIR / "06_rq3"

DATA_PATH = DATA_DIR / "strict_model_data.csv"
FEATURE_SETS_PATH = DATA_DIR / "strict_feature_sets.json"

RESULTS_PATH = OUT_DIR / "strict_rq3_temporal_feature_results.csv"
BEST_RESULTS_PATH = OUT_DIR / "strict_rq3_best_by_temporal_condition.csv"
TUNING_PATH = OUT_DIR / "strict_rq3_tuning_results.csv"
SUMMARY_PATH = OUT_DIR / "strict_rq3_summary.md"
VALIDATION_PATH = OUT_DIR / "strict_rq3_validation.md"
WEEKLY_SUMMARY_PATH = OUT_DIR / "strict_rq3_weekly_stress_summary.csv"
TREND_FIGURE_PATH = OUT_DIR / "strict_rq3_weekly_stress_trend.svg"

TARGET_COLUMN = "stress_label"
SPLIT_COLUMN = "split"
GROUP_COLUMN = "student_id"
RANDOM_SEED = 49
RQ1_BASELINE_MACRO_F1 = 0.357


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["student_id", "date"]).reset_index(drop=True)


def load_wearable_features() -> list[str]:
    feature_sets = load_feature_sets(FEATURE_SETS_PATH)
    features = feature_sets["rq1_all_wearable"]
    missing = sorted(set(features) - set(feature_sets["rq3_temporal_variants"]["no_temporal"]))
    if missing:
        raise ValueError(f"RQ3 no_temporal feature set is missing wearable features: {missing}")
    return features


def add_temporal_features(data: pd.DataFrame, wearable_features: list[str]) -> pd.DataFrame:
    temporal = data.sort_values(["student_id", "date"]).reset_index(drop=True).copy()
    for feature in wearable_features:
        temporal[f"{feature}_lag1"] = temporal.groupby("student_id")[feature].shift(1)
        temporal[f"{feature}_rolling3_mean"] = temporal.groupby("student_id")[feature].transform(
            lambda values: values.shift(1).rolling(window=3, min_periods=1).mean()
        )
        temporal[f"{feature}_rolling7_mean"] = temporal.groupby("student_id")[feature].transform(
            lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
        )
    return temporal


def make_temporal_conditions(wearable_features: list[str]) -> dict[str, list[str]]:
    return {
        "no_temporal": wearable_features,
        "semester_week_only": wearable_features + ["semester_week"],
        "lag1_wearable": wearable_features
        + [f"{feature}_lag1" for feature in wearable_features],
        "rolling3_wearable": wearable_features
        + [f"{feature}_rolling3_mean" for feature in wearable_features],
        "rolling7_wearable": wearable_features
        + [f"{feature}_rolling7_mean" for feature in wearable_features],
    }


def validate_temporal_design(
    data: pd.DataFrame, temporal_conditions: dict[str, list[str]], wearable_features: list[str]
) -> None:
    forbidden_features = {
        "student_id",
        "date",
        "stress",
        "stress_label",
        "anxiety",
        "STRESS_SCORE",
        "CALCULATION_FAILED",
    }
    all_features = {feature for features in temporal_conditions.values() for feature in features}
    used_forbidden = sorted(all_features & forbidden_features)
    if used_forbidden:
        raise ValueError(f"RQ3 temporal conditions use forbidden leakage features: {used_forbidden}")

    missing = sorted(all_features - set(data.columns))
    if missing:
        raise ValueError(f"RQ3 temporal conditions contain missing columns: {missing}")

    stress_history_features = [
        feature for feature in all_features if feature.startswith("stress_") or feature.startswith("anxiety_")
    ]
    if stress_history_features:
        raise ValueError(f"RQ3 should not use stress/anxiety history features: {stress_history_features}")

    expected_lag_features = {f"{feature}_lag1" for feature in wearable_features}
    expected_rolling3_features = {f"{feature}_rolling3_mean" for feature in wearable_features}
    expected_rolling7_features = {f"{feature}_rolling7_mean" for feature in wearable_features}
    if not expected_lag_features <= all_features:
        raise ValueError("RQ3 lag1 condition is incomplete.")
    if not expected_rolling3_features <= all_features:
        raise ValueError("RQ3 rolling3 condition is incomplete.")
    if not expected_rolling7_features <= all_features:
        raise ValueError("RQ3 rolling7 condition is incomplete.")


def make_weekly_stress_outputs(data: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        data.groupby("semester_week", as_index=False)
        .agg(
            n=("stress", "size"),
            mean_stress=("stress", "mean"),
            sd_stress=("stress", "std"),
            week_start=("date", "min"),
            week_end=("date", "max"),
        )
        .sort_values("semester_week")
    )
    weekly.to_csv(WEEKLY_SUMMARY_PATH, index=False)

    write_weekly_stress_svg(weekly)
    return weekly


def write_weekly_stress_svg(weekly: pd.DataFrame) -> None:
    width = 960
    height = 560
    margin_left = 72
    margin_right = 32
    margin_top = 56
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    max_week = int(weekly["semester_week"].max())
    min_stress = 0.0
    max_stress = max(100.0, float((weekly["mean_stress"] + weekly["sd_stress"]).max()))

    def x_scale(week: float) -> float:
        if max_week == 1:
            return margin_left + plot_width / 2
        return margin_left + (week - 1) / (max_week - 1) * plot_width

    def y_scale(stress: float) -> float:
        return margin_top + (max_stress - stress) / (max_stress - min_stress) * plot_height

    mean_points = [
        (x_scale(float(row.semester_week)), y_scale(float(row.mean_stress)))
        for row in weekly.itertuples(index=False)
    ]
    upper_points = [
        (
            x_scale(float(row.semester_week)),
            y_scale(min(max_stress, float(row.mean_stress + row.sd_stress))),
        )
        for row in weekly.itertuples(index=False)
    ]
    lower_points = [
        (
            x_scale(float(row.semester_week)),
            y_scale(max(min_stress, float(row.mean_stress - row.sd_stress))),
        )
        for row in weekly.itertuples(index=False)
    ]

    band_points = upper_points + list(reversed(lower_points))
    band_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in band_points)
    line_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in mean_points)

    grid_lines = []
    y_labels = []
    for value in range(0, 101, 20):
        y = y_scale(value)
        grid_lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" />'
        )
        y_labels.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#4b5563">{value}</text>'
        )

    x_labels = []
    for week in weekly["semester_week"]:
        x = x_scale(float(week))
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 28}" text-anchor="middle" font-size="11" fill="#4b5563">{int(week)}</text>'
        )

    circles = [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb" />'
        for x, y in mean_points
    ]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="20" font-family="Arial, sans-serif" font-weight="700" fill="#111827">Strict RQ3: Weekly Mean Stress Trend</text>
  <text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#374151">Semester week</text>
  <text transform="translate(20 {height / 2:.1f}) rotate(-90)" text-anchor="middle" font-size="14" font-family="Arial, sans-serif" fill="#374151">Self-reported stress</text>
  {''.join(grid_lines)}
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#9ca3af" />
  <line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#9ca3af" />
  {''.join(y_labels)}
  {''.join(x_labels)}
  <polygon points="{band_path}" fill="#93c5fd" opacity="0.35" />
  <polyline points="{line_path}" fill="none" stroke="#2563eb" stroke-width="3" />
  {''.join(circles)}
  <rect x="{width - 190}" y="50" width="150" height="48" fill="#ffffff" stroke="#e5e7eb" />
  <line x1="{width - 176}" y1="68" x2="{width - 142}" y2="68" stroke="#2563eb" stroke-width="3" />
  <text x="{width - 132}" y="72" font-size="12" font-family="Arial, sans-serif" fill="#374151">Mean stress</text>
  <rect x="{width - 176}" y="80" width="34" height="10" fill="#93c5fd" opacity="0.35" />
  <text x="{width - 132}" y="90" font-size="12" font-family="Arial, sans-serif" fill="#374151">+/- 1 SD</text>
</svg>
"""
    TREND_FIGURE_PATH.write_text(svg, encoding="utf-8")


def run_temporal_condition(
    data: pd.DataFrame, condition: str, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    test = data[data[SPLIT_COLUMN] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Strict RQ3 requires both train and test rows.")

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
    results, confusion_matrices = evaluate_models(fitted_models, X_test, y_test)

    results.insert(0, "temporal_condition", condition)
    results.insert(1, "n_features", len(features))
    tuning_results.insert(0, "temporal_condition", condition)
    tuning_results.insert(1, "n_features", len(features))
    return results, tuning_results, confusion_matrices


def run_all_temporal_conditions(
    data: pd.DataFrame, temporal_conditions: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    result_frames = []
    tuning_frames = []
    confusion_by_condition = {}
    for condition, features in temporal_conditions.items():
        print(f"Running strict RQ3 condition: {condition}")
        results, tuning_results, confusion_matrices = run_temporal_condition(
            data,
            condition,
            features,
        )
        result_frames.append(results)
        tuning_frames.append(tuning_results)
        confusion_by_condition[condition] = confusion_matrices
    return (
        pd.concat(result_frames, ignore_index=True),
        pd.concat(tuning_frames, ignore_index=True),
        confusion_by_condition,
    )


def best_results_by_condition(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["temporal_condition", "macro_f1"], ascending=[True, False])
        .groupby("temporal_condition", as_index=False, observed=True)
        .head(1)
        .reset_index(drop=True)
    )


def save_results(
    results: pd.DataFrame,
    tuning_results: pd.DataFrame,
    best_by_condition: pd.DataFrame,
    confusion_by_condition: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    merged = results.merge(
        tuning_results,
        on=["temporal_condition", "n_features", "model"],
        how="left",
    )
    merged.to_csv(RESULTS_PATH, index=False)
    tuning_results.to_csv(TUNING_PATH, index=False)
    best_by_condition.merge(
        tuning_results,
        on=["temporal_condition", "n_features", "model"],
        how="left",
    ).to_csv(BEST_RESULTS_PATH, index=False)

    best_overall = merged.sort_values("macro_f1", ascending=False).iloc[0]
    best_condition = str(best_overall["temporal_condition"])
    best_model = str(best_overall["model"])
    confusion_by_condition[best_condition][best_model].to_csv(
        OUT_DIR / "strict_rq3_confusion_matrix_best_overall.csv"
    )

    for condition, model_matrices in confusion_by_condition.items():
        for model_name, matrix in model_matrices.items():
            matrix.to_csv(OUT_DIR / f"strict_rq3_confusion_matrix_{condition}_{model_name}.csv")
    return merged


def write_summary(
    data: pd.DataFrame,
    weekly: pd.DataFrame,
    temporal_conditions: dict[str, list[str]],
    merged_results: pd.DataFrame,
    best_by_condition: pd.DataFrame,
) -> None:
    best_with_tuning = best_by_condition.merge(
        merged_results[
            [
                "temporal_condition",
                "n_features",
                "model",
                "cv_macro_f1",
                "best_params",
            ]
        ],
        on=["temporal_condition", "n_features", "model"],
        how="left",
    )
    baseline_best = best_by_condition[
        best_by_condition["temporal_condition"].eq("no_temporal")
    ].iloc[0]
    overall_best = best_by_condition.sort_values("macro_f1", ascending=False).iloc[0]
    peak_week = weekly.loc[weekly["mean_stress"].idxmax()]
    low_week = weekly.loc[weekly["mean_stress"].idxmin()]

    compact_best = best_with_tuning[
        [
            "temporal_condition",
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
    ].copy()
    compact_best["delta_vs_no_temporal_best"] = (
        compact_best["macro_f1"].astype(float) - float(baseline_best["macro_f1"])
    )
    compact_best["delta_vs_strict_rq1_mlp"] = (
        compact_best["macro_f1"].astype(float) - RQ1_BASELINE_MACRO_F1
    )
    compact_best = compact_best.sort_values("macro_f1", ascending=False)

    compact_results = merged_results[
        [
            "temporal_condition",
            "n_features",
            "model",
            "accuracy",
            "macro_f1",
            "cv_macro_f1",
            "best_params",
        ]
    ].sort_values(["temporal_condition", "macro_f1"], ascending=[True, False])

    for frame in [compact_best, compact_results]:
        for column in [
            "accuracy",
            "macro_f1",
            "low_f1",
            "medium_f1",
            "high_f1",
            "cv_macro_f1",
            "delta_vs_no_temporal_best",
            "delta_vs_strict_rq1_mlp",
        ]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    train = data[data[SPLIT_COLUMN] == "train"]
    test = data[data[SPLIT_COLUMN] == "test"]

    lines = [
        "# 严格版 RQ3 时序特征实验总结",
        "",
        "## 研究问题",
        "",
        "RQ3 关注：在 wearable-only 压力等级预测任务中，加入学期周次、前一天 wearable 特征、过去 3 天/7 天 wearable 均值后，是否能改善模型表现？",
        "",
        "## 严格实验设计",
        "",
        f"- 随机种子：`{RANDOM_SEED}`，与严格版 RQ1/RQ2 保持一致。",
        "- 输入数据来自 `modeling_outputs/strict_pipeline/03_model_data/strict_model_data.csv`。",
        "- 使用严格版 RQ1/RQ2 相同的 `split` 列：同一个学生不会同时出现在 train 和 test。",
        f"- train：{train[GROUP_COLUMN].nunique()} 名学生，{len(train)} 行；test：{test[GROUP_COLUMN].nunique()} 名学生，{len(test)} 行。",
        "- 缺失值填补和必要的标准化都放在 scikit-learn pipeline 内部。",
        "- 模型和超参数选择只在训练集内部通过 GroupKFold + GridSearchCV 完成。",
        "- test set 只用于最终评估。",
        "- 不使用当天 `stress`、`anxiety`、`STRESS_SCORE`、`CALCULATION_FAILED`，也不使用 `stress_lag1`，避免把压力自评或设备压力分数带入模型。",
        "- lag 特征使用 `groupby(student_id).shift(1)`，rolling 特征使用 `shift(1).rolling(...).mean()`，避免包含当天信息。",
        "",
        "## 时序条件",
        "",
    ]
    for condition, features in temporal_conditions.items():
        lines.append(f"- `{condition}`：{len(features)} 个特征。")

    lines.extend(
        [
            "",
            "## 每个条件的最佳结果",
            "",
            dataframe_to_markdown(compact_best),
            "",
            "## 完整模型结果",
            "",
            dataframe_to_markdown(compact_results),
            "",
            "## 学期周压力趋势",
            "",
            f"- 平均压力最低：第 {int(low_week['semester_week'])} 周，M={float(low_week['mean_stress']):.2f}，SD={float(low_week['sd_stress']):.2f}。",
            f"- 平均压力最高：第 {int(peak_week['semester_week'])} 周，M={float(peak_week['mean_stress']):.2f}，SD={float(peak_week['sd_stress']):.2f}。",
            f"- 趋势图：`{TREND_FIGURE_PATH.relative_to(ROOT)}`。",
            "",
            "## 关键结论",
            "",
            f"- 严格版 RQ3 的最佳条件是 `{overall_best['temporal_condition']}` + `{overall_best['model']}`。",
            f"- 最佳 test macro-F1 为 {float(overall_best['macro_f1']):.3f}。",
            f"- 相比本 RQ3 的 no-temporal 最佳模型，变化为 {float(overall_best['macro_f1']) - float(baseline_best['macro_f1']):+.3f}。",
            f"- 相比严格版 RQ1 MLP baseline macro-F1={RQ1_BASELINE_MACRO_F1:.3f}，变化为 {float(overall_best['macro_f1']) - RQ1_BASELINE_MACRO_F1:+.3f}。",
            "- 如果时序条件没有明显提升，应在报告中如实写成 negative or weak result：严格防泄漏设置下，时序 wearable 特征未必稳定改善 unseen-student 压力预测。",
            "",
            "## 输出文件",
            "",
            f"- 结果表：`{RESULTS_PATH.relative_to(ROOT)}`",
            f"- 每个条件最佳结果：`{BEST_RESULTS_PATH.relative_to(ROOT)}`",
            f"- 调参记录：`{TUNING_PATH.relative_to(ROOT)}`",
            f"- 最佳模型 confusion matrix：`{(OUT_DIR / 'strict_rq3_confusion_matrix_best_overall.csv').relative_to(ROOT)}`",
            f"- 周压力趋势图：`{TREND_FIGURE_PATH.relative_to(ROOT)}`",
            f"- 周压力统计表：`{WEEKLY_SUMMARY_PATH.relative_to(ROOT)}`",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_validation(merged_results: pd.DataFrame) -> None:
    lines = [
        "# 严格版 RQ3 验证记录",
        "",
        "## 已验证内容",
        "",
        "- 脚本可以通过 Python 编译检查。",
        "- 脚本可以从项目根目录运行。",
        "- 输入数据来自严格版建模数据目录。",
        "- 输出文件写入 `modeling_outputs/strict_pipeline/06_rq3/`。",
        "- RQ3 使用与 RQ1/RQ2 相同的 subject-aware train/test split。",
        "- 缺失值填补和标准化在模型 pipeline 内完成。",
        "- lag/rolling 特征只使用同一学生过去日期的数据。",
        "- test set 不参与模型选择或超参数选择。",
        "",
        "## 实验规模",
        "",
        f"- temporal condition 数：{merged_results['temporal_condition'].nunique()}",
        f"- 模型结果行数：{len(merged_results)}",
        "",
    ]
    VALIDATION_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    wearable_features = load_wearable_features()
    temporal_data = add_temporal_features(data, wearable_features)
    temporal_conditions = make_temporal_conditions(wearable_features)
    validate_temporal_design(temporal_data, temporal_conditions, wearable_features)
    weekly = make_weekly_stress_outputs(temporal_data)
    results, tuning_results, confusion_by_condition = run_all_temporal_conditions(
        temporal_data,
        temporal_conditions,
    )
    best_by_condition = best_results_by_condition(results)
    merged_results = save_results(
        results,
        tuning_results,
        best_by_condition,
        confusion_by_condition,
    )
    write_summary(temporal_data, weekly, temporal_conditions, merged_results, best_by_condition)
    write_validation(merged_results)

    best = best_by_condition.sort_values("macro_f1", ascending=False).iloc[0]
    print("Strict RQ3 temporal feature modeling complete.")
    print(f"Temporal conditions: {', '.join(temporal_conditions)}")
    print(f"Experiment rows: {len(merged_results)}")
    print(f"Best: {best['temporal_condition']} + {best['model']} macro-F1={best['macro_f1']:.3f}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {BEST_RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
