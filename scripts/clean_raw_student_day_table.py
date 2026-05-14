from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT_DIR = ROOT / "modeling_outputs" / "strict_pipeline"
RAW_STUDENT_DAY_PATH = BASE_OUT_DIR / "01_raw_student_day" / "raw_student_day_table.csv"
OUT_DIR = BASE_OUT_DIR / "02_cleaned_student_day"
STRICT_CLEAN_PATH = OUT_DIR / "strict_clean_student_day_table.csv"
CLEANING_AUDIT_PATH = OUT_DIR / "strict_cleaning_audit.json"
CLEANING_REPORT_PATH = OUT_DIR / "strict_cleaning_report.md"

WEARABLE_COLUMNS = [
    "sleep_score",
    "deep_sleep_minutes",
    "total_steps",
    "sedentary_minutes",
    "lightly_active_minutes",
    "moderately_active_minutes",
    "very_active_minutes",
    "avg_rmssd",
    "avg_low_frequency",
    "avg_high_frequency",
    "avg_oxygen",
    "std_oxygen",
]

LEAKAGE_COLUMNS = [
    "student_id",
    "date",
    "stress",
    "stress_label",
    "anxiety",
    "STRESS_SCORE",
    "CALCULATION_FAILED",
]


def stress_to_label(stress: float) -> str:
    if pd.isna(stress):
        return ""
    if stress <= 17:
        return "Low"
    if stress <= 38:
        return "Medium"
    return "High"


def read_raw_student_day_table() -> pd.DataFrame:
    data = pd.read_csv(RAW_STUDENT_DAY_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    data["CALCULATION_FAILED"] = data["CALCULATION_FAILED"].astype("boolean")
    data["stress_label"] = data["stress"].map(stress_to_label)
    return data


def boolean_any(values: pd.Series) -> bool | pd.NA:
    non_missing = values.dropna()
    if non_missing.empty:
        return pd.NA
    return bool(non_missing.astype(bool).max())


def duplicate_group_summary(data: pd.DataFrame) -> pd.DataFrame:
    duplicated = data[data.duplicated(["student_id", "date"], keep=False)].copy()
    if duplicated.empty:
        return pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "n_rows",
                "stress_values",
                "anxiety_values",
                "label_values",
                "stress_conflict",
                "label_conflict",
            ]
        )

    grouped = (
        duplicated.groupby(["student_id", "date"], observed=True)
        .agg(
            n_rows=("stress", "size"),
            stress_values=("stress", lambda values: sorted(values.tolist())),
            anxiety_values=("anxiety", lambda values: sorted(values.tolist())),
            label_values=("stress_label", lambda values: sorted(set(values.astype(str)))),
        )
        .reset_index()
    )
    grouped["stress_conflict"] = grouped["stress_values"].map(lambda values: len(set(values)) > 1)
    grouped["label_conflict"] = grouped["label_values"].map(lambda values: len(values) > 1)
    return grouped


def resolve_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in data.select_dtypes(include="number").columns
        if column != "CALCULATION_FAILED"
    ]
    resolved = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)[numeric_columns]
        .mean()
        .sort_values(["student_id", "date"])
        .reset_index(drop=True)
    )

    calc_failed = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)
        .agg(CALCULATION_FAILED=("CALCULATION_FAILED", boolean_any))
        .reset_index(drop=True)
    )
    resolved = resolved.merge(calc_failed, on=["student_id", "date"], how="left")
    resolved["stress_label"] = resolved["stress"].map(stress_to_label)

    ordered_columns = [
        "student_id",
        "date",
        "stress",
        "stress_label",
        "anxiety",
        "sleep_score",
        "deep_sleep_minutes",
        "total_steps",
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
        "avg_rmssd",
        "avg_low_frequency",
        "avg_high_frequency",
        "avg_oxygen",
        "std_oxygen",
        "STRESS_SCORE",
        "CALCULATION_FAILED",
    ]
    return resolved[ordered_columns]


def build_audit(raw: pd.DataFrame, clean: pd.DataFrame, duplicates: pd.DataFrame) -> dict[str, Any]:
    wearable_missing_cells = int(clean[WEARABLE_COLUMNS].isna().sum().sum())
    wearable_total_cells = int(len(clean) * len(WEARABLE_COLUMNS))
    complete_wearable_rows = int(clean[WEARABLE_COLUMNS].notna().all(axis=1).sum())

    return {
        "source_path": RAW_STUDENT_DAY_PATH.name,
        "clean_path": STRICT_CLEAN_PATH.name,
        "source_rows": int(len(raw)),
        "source_students": int(raw["student_id"].nunique()),
        "source_unique_student_days": int(raw[["student_id", "date"]].drop_duplicates().shape[0]),
        "duplicate_student_day_pairs": int(len(duplicates)),
        "duplicate_pairs_with_stress_conflict": int(duplicates["stress_conflict"].sum())
        if not duplicates.empty
        else 0,
        "duplicate_pairs_with_label_conflict": int(duplicates["label_conflict"].sum())
        if not duplicates.empty
        else 0,
        "clean_rows": int(len(clean)),
        "clean_students": int(clean["student_id"].nunique()),
        "clean_unique_student_days": int(clean[["student_id", "date"]].drop_duplicates().shape[0]),
        "removed_duplicate_rows": int(len(raw) - len(clean)),
        "label_counts": {key: int(value) for key, value in clean["stress_label"].value_counts().to_dict().items()},
        "complete_wearable_rows": complete_wearable_rows,
        "complete_wearable_percent": round(complete_wearable_rows / len(clean) * 100, 2),
        "wearable_missing_cells": wearable_missing_cells,
        "wearable_total_cells": wearable_total_cells,
        "wearable_missing_percent": round(wearable_missing_cells / wearable_total_cells * 100, 2),
        "leakage_columns_excluded_from_model_inputs": LEAKAGE_COLUMNS,
        "missing_by_column": {
            column: int(clean[column].isna().sum())
            for column in clean.columns
            if int(clean[column].isna().sum()) > 0
        },
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


def write_report(clean: pd.DataFrame, duplicates: pd.DataFrame, audit: dict[str, Any]) -> None:
    label_counts = pd.DataFrame(
        [{"stress_label": key, "count": value} for key, value in audit["label_counts"].items()]
    )
    missing_summary = (
        pd.DataFrame(
            [
                {
                    "column": column,
                    "missing_rows": missing,
                    "missing_percent": missing / len(clean) * 100,
                }
                for column, missing in audit["missing_by_column"].items()
            ]
        )
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )
    duplicate_preview = duplicates.head(20).copy()

    lines = [
        "# 严格清洗 Student-Day 表报告",
        "",
        "## 目的",
        "",
        "本步骤从 `raw_student_day_table.csv` 生成严格清洗表。清洗只处理标签和重复 student-day，不做任何全局填补或全局标准化，从而避免 preprocessing leakage。",
        "",
        "## 清洗规则",
        "",
        "- 根据数值型 `stress` 重新生成 `stress_label`：Low = 0-17，Medium = 18-38，High = 39-100。",
        "- 对同一 `student_id + date` 的多条问卷记录，所有数值列取平均值。",
        "- `CALCULATION_FAILED` 使用 any-true 规则：同一天任一记录为 True，则清洗后为 True。",
        "- wearable 缺失值保持缺失，不在这里填补。",
        "- wearable 特征保持原始单位，不在这里标准化。",
        "",
        "## 清洗结果",
        "",
        f"- 原始行数：{audit['source_rows']}",
        f"- 清洗后行数：{audit['clean_rows']}",
        f"- 学生数：{audit['clean_students']}",
        f"- 删除的重复行数：{audit['removed_duplicate_rows']}",
        f"- 重复 student-day pair 数：{audit['duplicate_student_day_pairs']}",
        f"- stress 数值冲突的重复 pair 数：{audit['duplicate_pairs_with_stress_conflict']}",
        f"- stress label 冲突的重复 pair 数：{audit['duplicate_pairs_with_label_conflict']}",
        f"- wearable 完整行数：{audit['complete_wearable_rows']} ({audit['complete_wearable_percent']:.2f}%)",
        f"- wearable 缺失单元格比例：{audit['wearable_missing_percent']:.2f}%",
        "",
        "## 标签分布",
        "",
        dataframe_to_markdown(label_counts),
        "",
        "## 缺失值概览",
        "",
        dataframe_to_markdown(missing_summary),
        "",
        "## 重复记录预览",
        "",
        dataframe_to_markdown(duplicate_preview)
        if not duplicate_preview.empty
        else "没有重复 student-day 记录。",
        "",
        "## 后续建模注意事项",
        "",
        "- 后续应基于这张严格清洗表先做 subject-aware train/test split。",
        "- 缺失值填补和标准化必须放进 scikit-learn pipeline，在训练数据或训练折内部拟合。",
        "- `student_id`、`date`、`stress`、`stress_label`、`anxiety`、`STRESS_SCORE`、`CALCULATION_FAILED` 不应作为主模型输入。",
        "",
    ]
    CLEANING_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_raw_student_day_table()
    duplicates = duplicate_group_summary(raw)
    clean = resolve_duplicates(raw)
    audit = build_audit(raw, clean, duplicates)

    clean.to_csv(STRICT_CLEAN_PATH, index=False)
    CLEANING_AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(clean, duplicates, audit)

    print(f"Wrote {STRICT_CLEAN_PATH.relative_to(ROOT)}")
    print(f"Wrote {CLEANING_AUDIT_PATH.relative_to(ROOT)}")
    print(f"Wrote {CLEANING_REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
