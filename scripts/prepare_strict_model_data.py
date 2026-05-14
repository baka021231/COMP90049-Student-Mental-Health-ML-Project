from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modeling_outputs"
SOURCE_PATH = OUT_DIR / "strict_clean_student_day_table.csv"
STRICT_MODEL_DATA_PATH = OUT_DIR / "strict_model_data.csv"
STRICT_FEATURE_SETS_PATH = OUT_DIR / "strict_feature_sets.json"
STRICT_SPLIT_PATH = OUT_DIR / "strict_split_assignments.csv"
STRICT_PREP_AUDIT_PATH = OUT_DIR / "strict_model_data_audit.json"
STRICT_PREP_REPORT_PATH = OUT_DIR / "strict_model_data_report.md"

RANDOM_SEED = 49
TEST_SIZE = 0.25

SLEEP_FEATURES = ["sleep_score", "deep_sleep_minutes"]
ACTIVITY_FEATURES = [
    "total_steps",
    "sedentary_minutes",
    "lightly_active_minutes",
    "moderately_active_minutes",
    "very_active_minutes",
]
HRV_SPO2_FEATURES = [
    "avg_rmssd",
    "avg_low_frequency",
    "avg_high_frequency",
    "avg_oxygen",
    "std_oxygen",
]
WEARABLE_FEATURES = SLEEP_FEATURES + ACTIVITY_FEATURES + HRV_SPO2_FEATURES
CALENDAR_FEATURES = ["semester_week", "day_of_week", "is_weekend"]
LEAKAGE_COLUMNS = [
    "student_id",
    "date",
    "stress",
    "stress_label",
    "anxiety",
    "STRESS_SCORE",
    "CALCULATION_FAILED",
]


def read_source() -> pd.DataFrame:
    data = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    data["CALCULATION_FAILED"] = data["CALCULATION_FAILED"].astype("boolean")
    return data


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["student_id", "date"]).copy()
    first_date = out["date"].min()
    out["semester_week"] = ((out["date"] - first_date).dt.days // 7 + 1).astype(int)
    out["day_of_week"] = out["date"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def make_subject_split(data: pd.DataFrame) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    row_numbers = pd.Series(range(len(data)), index=data.index)
    train_idx, test_idx = next(
        splitter.split(data, data["stress_label"], groups=data["student_id"])
    )
    train_rows = set(row_numbers.iloc[train_idx])
    test_rows = set(row_numbers.iloc[test_idx])

    rows = []
    for student_id, group in data.groupby("student_id", observed=True):
        group_rows = set(row_numbers.loc[group.index])
        split = "train" if group_rows <= train_rows else "test" if group_rows <= test_rows else "mixed"
        rows.append(
            {
                "student_id": student_id,
                "split": split,
                "n_rows": len(group),
                "low_count": int((group["stress_label"] == "Low").sum()),
                "medium_count": int((group["stress_label"] == "Medium").sum()),
                "high_count": int((group["stress_label"] == "High").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "student_id"]).reset_index(drop=True)


def make_feature_sets() -> dict[str, Any]:
    return {
        "leakage_columns_excluded_from_model_inputs": LEAKAGE_COLUMNS,
        "rq1_all_wearable": WEARABLE_FEATURES,
        "rq2_feature_groups": {
            "sleep_only": SLEEP_FEATURES,
            "activity_only": ACTIVITY_FEATURES,
            "hrv_spo2_only": HRV_SPO2_FEATURES,
            "all_wearable": WEARABLE_FEATURES,
        },
        "rq3_temporal_variants": {
            "no_temporal": WEARABLE_FEATURES,
            "calendar": WEARABLE_FEATURES + CALENDAR_FEATURES,
        },
    }


def build_audit(data: pd.DataFrame, split: pd.DataFrame) -> dict[str, Any]:
    split_summary = (
        split.groupby("split", observed=True)
        .agg(
            students=("student_id", "count"),
            rows=("n_rows", "sum"),
            low=("low_count", "sum"),
            medium=("medium_count", "sum"),
            high=("high_count", "sum"),
        )
        .to_dict(orient="index")
    )
    missing_by_feature = data[WEARABLE_FEATURES].isna().sum().to_dict()
    return {
        "source_path": SOURCE_PATH.name,
        "strict_model_data_path": STRICT_MODEL_DATA_PATH.name,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "rows": int(len(data)),
        "students": int(data["student_id"].nunique()),
        "unique_student_days": int(data[["student_id", "date"]].drop_duplicates().shape[0]),
        "label_counts": {key: int(value) for key, value in data["stress_label"].value_counts().to_dict().items()},
        "split_summary": split_summary,
        "wearable_missing_by_feature": {key: int(value) for key, value in missing_by_feature.items()},
        "calendar_features_added": CALENDAR_FEATURES,
        "preprocessing_policy": "不在此步骤做填补或标准化；缺失值填补和 scaling 必须放入后续模型 pipeline。",
    }


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    display = data.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.2f}")
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        clean_row = []
        for item in row:
            text = str(item)
            clean_row.append("N/A" if text.lower() in {"nan", "none", "<na>"} else text)
        lines.append("| " + " | ".join(clean_row) + " |")
    return "\n".join(lines)


def write_report(data: pd.DataFrame, split: pd.DataFrame, audit: dict[str, Any]) -> None:
    split_table = (
        split.groupby("split", observed=True)
        .agg(
            students=("student_id", "count"),
            rows=("n_rows", "sum"),
            low=("low_count", "sum"),
            medium=("medium_count", "sum"),
            high=("high_count", "sum"),
        )
        .reset_index()
    )
    missing_table = pd.DataFrame(
        [
            {
                "feature": key,
                "missing_rows": value,
                "missing_percent": value / len(data) * 100,
            }
            for key, value in audit["wearable_missing_by_feature"].items()
        ]
    ).sort_values("missing_percent", ascending=False)

    lines = [
        "# 严格建模数据准备报告",
        "",
        "## 目的",
        "",
        "本步骤从 `strict_clean_student_day_table.csv` 生成后续建模使用的严格数据表。关键原则是：先按学生划分 train/test，但不在这里做任何缺失值填补或标准化。",
        "",
        "## 处理内容",
        "",
        "- 保留原始 wearable 特征单位。",
        "- 保留 wearable 缺失值。",
        "- 添加确定性的日历特征：`semester_week`、`day_of_week`、`is_weekend`。",
        "- 使用 subject-aware split，确保同一个学生不会同时出现在 train 和 test。",
        "- 生成严格版 feature sets，供 RQ1/RQ2/RQ3 使用。",
        "",
        "## 数据规模",
        "",
        f"- 行数：{audit['rows']}",
        f"- 学生数：{audit['students']}",
        f"- unique student-day 数：{audit['unique_student_days']}",
        f"- 随机种子：{audit['random_seed']}",
        f"- test_size：{audit['test_size']}",
        "",
        "## Train/Test Split",
        "",
        dataframe_to_markdown(split_table),
        "",
        "## Wearable 缺失值",
        "",
        dataframe_to_markdown(missing_table),
        "",
        "## Feature Sets",
        "",
        "- RQ1：全部 wearable 特征。",
        "- RQ2：sleep_only、activity_only、hrv_spo2_only、all_wearable。",
        "- RQ3：当前先保留 no_temporal 和 calendar 两个版本；lag/rolling 特征可在后续单独加入并处理缺失。",
        "",
        "## 泄漏控制",
        "",
        "- 本步骤没有做全局 imputation。",
        "- 本步骤没有做全局 standardization。",
        "- 后续模型必须在 pipeline 中使用 `SimpleImputer` / `StandardScaler`，并只在训练数据或交叉验证训练折内拟合。",
        "- `student_id`、`date`、`stress`、`stress_label`、`anxiety`、`STRESS_SCORE`、`CALCULATION_FAILED` 不作为主模型输入。",
        "",
    ]
    STRICT_PREP_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = read_source()
    data = add_calendar_features(data)
    split = make_subject_split(data)
    data["split"] = data["student_id"].map(split.set_index("student_id")["split"])
    feature_sets = make_feature_sets()
    audit = build_audit(data, split)

    data.to_csv(STRICT_MODEL_DATA_PATH, index=False)
    split.to_csv(STRICT_SPLIT_PATH, index=False)
    STRICT_FEATURE_SETS_PATH.write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    STRICT_PREP_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(data, split, audit)

    print(f"Wrote {STRICT_MODEL_DATA_PATH.relative_to(ROOT)}")
    print(f"Wrote {STRICT_SPLIT_PATH.relative_to(ROOT)}")
    print(f"Wrote {STRICT_FEATURE_SETS_PATH.relative_to(ROOT)}")
    print(f"Wrote {STRICT_PREP_AUDIT_PATH.relative_to(ROOT)}")
    print(f"Wrote {STRICT_PREP_REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
