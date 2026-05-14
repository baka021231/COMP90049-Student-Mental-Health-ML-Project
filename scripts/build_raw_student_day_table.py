from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "SSAQS dataset"
OUT_DIR = ROOT / "modeling_outputs"

RAW_STUDENT_DAY_PATH = OUT_DIR / "raw_student_day_table.csv"
MERGE_AUDIT_JSON_PATH = OUT_DIR / "raw_student_day_merge_audit.json"
MERGE_REPORT_PATH = OUT_DIR / "raw_student_day_merge_report.md"
REFERENCE_TABLE_PATH = ROOT / "final_student_day_table_v01.csv"

OUTPUT_COLUMNS = [
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


def student_dirs() -> list[Path]:
    return sorted(
        [path for path in RAW_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_dates(series: pd.Series, unix_seconds: bool = False) -> pd.Series:
    if unix_seconds:
        return pd.to_datetime(series, unit="s", errors="coerce", utc=True).dt.date
    return pd.to_datetime(series, errors="coerce", utc=True).dt.date


def stress_to_label(stress: float) -> str:
    if pd.isna(stress):
        return ""
    if stress <= 17:
        return "Low"
    if stress <= 38:
        return "Medium"
    return "High"


def aggregate_daily_questions(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "daily_questions.csv")
    if data is None or data.empty:
        return pd.DataFrame(columns=["student_id", "date", "stress", "stress_label", "anxiety"])

    out = data[["timeStampStart", "stress", "anxiety"]].copy()
    out["student_id"] = student_dir.name
    out["date"] = parse_dates(out["timeStampStart"], unix_seconds=True)
    out = out[out["date"].notna()].copy()
    out["stress_label"] = out["stress"].map(stress_to_label)
    return out[["student_id", "date", "stress", "stress_label", "anxiety"]]


def aggregate_sleep(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "sleep.csv")
    if data is None or data.empty:
        return pd.DataFrame(columns=["student_id", "date", "sleep_score", "deep_sleep_minutes"])

    data = data.copy()
    data["student_id"] = student_dir.name
    data["date"] = parse_dates(data["timestamp"])
    grouped = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)
        .agg(
            sleep_score=("overall_score", "mean"),
            deep_sleep_minutes=("deep_sleep_in_minutes", "mean"),
        )
        .reset_index(drop=True)
    )
    return grouped


def aggregate_steps(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "steps.csv")
    if data is None or data.empty:
        return pd.DataFrame(columns=["student_id", "date", "total_steps"])

    data = data.copy()
    data["student_id"] = student_dir.name
    data["date"] = parse_dates(data["timestamp"])
    grouped = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)
        .agg(total_steps=("steps", "sum"))
        .reset_index(drop=True)
    )
    return grouped


def aggregate_activity(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "activity_level.csv")
    columns = [
        "student_id",
        "date",
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
    ]
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)

    data = data.copy()
    data["student_id"] = student_dir.name
    data["date"] = parse_dates(data["timestamp"])
    counts = (
        data.groupby(["student_id", "date", "level"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    rename_map = {
        "SEDENTARY": "sedentary_minutes",
        "LIGHTLY_ACTIVE": "lightly_active_minutes",
        "MODERATELY_ACTIVE": "moderately_active_minutes",
        "VERY_ACTIVE": "very_active_minutes",
    }
    counts = counts.rename(columns=rename_map)
    for column in columns:
        if column not in counts.columns:
            counts[column] = 0
    return counts[columns]


def aggregate_hrv(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "hrv.csv")
    if data is None or data.empty:
        return pd.DataFrame(
            columns=["student_id", "date", "avg_rmssd", "avg_low_frequency", "avg_high_frequency"]
        )

    data = data.copy()
    data["student_id"] = student_dir.name
    data["date"] = parse_dates(data["timestamp"])
    grouped = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)
        .agg(
            avg_rmssd=("rmssd", "mean"),
            avg_low_frequency=("low_frequency", "mean"),
            avg_high_frequency=("high_frequency", "mean"),
        )
        .reset_index(drop=True)
    )
    return grouped


def aggregate_oxygen(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "oxygen.csv")
    if data is None or data.empty:
        return pd.DataFrame(columns=["student_id", "date", "avg_oxygen", "std_oxygen"])

    data = data.copy()
    data["student_id"] = student_dir.name
    data["date"] = parse_dates(data["timestamp"])
    grouped = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)
        .agg(avg_oxygen=("value", "mean"), std_oxygen=("value", "std"))
        .reset_index(drop=True)
    )
    return grouped


def aggregate_fitbit_stress(student_dir: Path) -> pd.DataFrame:
    data = read_csv_if_exists(student_dir / "stress.csv")
    if data is None or data.empty:
        return pd.DataFrame(columns=["student_id", "date", "STRESS_SCORE", "CALCULATION_FAILED"])

    data = data.copy()
    data["student_id"] = student_dir.name
    data["date"] = parse_dates(data["DATE"])
    data["CALCULATION_FAILED"] = data["CALCULATION_FAILED"].astype("boolean")
    grouped = (
        data.groupby(["student_id", "date"], as_index=False, observed=True)
        .agg(
            STRESS_SCORE=("STRESS_SCORE", "mean"),
            CALCULATION_FAILED=("CALCULATION_FAILED", "max"),
        )
        .reset_index(drop=True)
    )
    return grouped


def merge_student(student_dir: Path) -> pd.DataFrame:
    table = aggregate_daily_questions(student_dir)
    for feature_table in [
        aggregate_sleep(student_dir),
        aggregate_steps(student_dir),
        aggregate_activity(student_dir),
        aggregate_hrv(student_dir),
        aggregate_oxygen(student_dir),
        aggregate_fitbit_stress(student_dir),
    ]:
        table = table.merge(feature_table, on=["student_id", "date"], how="left")
    return table


def build_raw_student_day_table() -> pd.DataFrame:
    frames = [merge_student(student_dir) for student_dir in student_dirs()]
    table = pd.concat(frames, ignore_index=True)
    table["date"] = pd.to_datetime(table["date"])
    table = table.sort_values(["student_id", "date"]).reset_index(drop=True)
    return table[OUTPUT_COLUMNS]


def compare_with_reference(table: pd.DataFrame) -> dict[str, Any]:
    if not REFERENCE_TABLE_PATH.exists():
        return {"reference_exists": False}

    reference = pd.read_csv(REFERENCE_TABLE_PATH, encoding="utf-8-sig")
    reference["student_id"] = reference["student_id"].astype(str)
    reference["date"] = pd.to_datetime(reference["date"])

    table_pairs = set(zip(table["student_id"], table["date"]))
    reference_pairs = set(zip(reference["student_id"], reference["date"]))

    reference_grouped = (
        reference.groupby(["student_id", "date"], observed=True)
        .agg(
            reference_row_count=("stress", "size"),
            reference_stress_values=("stress", lambda values: sorted(values.tolist())),
            reference_anxiety_values=("anxiety", lambda values: sorted(values.tolist())),
        )
        .reset_index()
    )
    table_grouped = (
        table.groupby(["student_id", "date"], observed=True)
        .agg(
            rebuilt_row_count=("stress", "size"),
            rebuilt_stress_values=("stress", lambda values: sorted(values.tolist())),
            rebuilt_anxiety_values=("anxiety", lambda values: sorted(values.tolist())),
        )
        .reset_index()
    )
    common = reference_grouped.merge(table_grouped, on=["student_id", "date"], how="inner")
    row_count_different = int(
        (common["reference_row_count"] != common["rebuilt_row_count"]).sum()
    )
    stress_values_different = int(
        (
            common["reference_stress_values"].astype(str)
            != common["rebuilt_stress_values"].astype(str)
        ).sum()
    )
    anxiety_values_different = int(
        (
            common["reference_anxiety_values"].astype(str)
            != common["rebuilt_anxiety_values"].astype(str)
        ).sum()
    )

    return {
        "reference_exists": True,
        "reference_rows": int(len(reference)),
        "reference_unique_student_days": int(reference[["student_id", "date"]].drop_duplicates().shape[0]),
        "rebuilt_rows": int(len(table)),
        "rebuilt_unique_student_days": int(table[["student_id", "date"]].drop_duplicates().shape[0]),
        "common_unique_student_days": int(len(reference_pairs & table_pairs)),
        "reference_only_student_days": int(len(reference_pairs - table_pairs)),
        "rebuilt_only_student_days": int(len(table_pairs - reference_pairs)),
        "common_grouped_student_days": int(len(common)),
        "common_student_days_with_different_row_counts": row_count_different,
        "common_student_days_with_different_stress_values": stress_values_different,
        "common_student_days_with_different_anxiety_values": anxiety_values_different,
    }


def build_audit(table: pd.DataFrame) -> dict[str, Any]:
    duplicate_rows = int(table.duplicated(["student_id", "date"]).sum())
    duplicate_pairs = int(
        table[table.duplicated(["student_id", "date"], keep=False)][["student_id", "date"]]
        .drop_duplicates()
        .shape[0]
    )
    missing_by_column = {
        column: int(table[column].isna().sum())
        for column in OUTPUT_COLUMNS
        if int(table[column].isna().sum()) > 0
    }
    wearable_missing_cells = int(table[WEARABLE_COLUMNS].isna().sum().sum())
    wearable_total_cells = int(len(table) * len(WEARABLE_COLUMNS))

    return {
        "source_directory": RAW_DIR.name,
        "rows": int(len(table)),
        "students": int(table["student_id"].nunique()),
        "unique_student_days": int(table[["student_id", "date"]].drop_duplicates().shape[0]),
        "date_min": str(table["date"].min().date()),
        "date_max": str(table["date"].max().date()),
        "duplicate_student_day_rows": duplicate_rows,
        "duplicate_student_day_pairs": duplicate_pairs,
        "label_counts": {key: int(value) for key, value in table["stress_label"].value_counts().to_dict().items()},
        "wearable_missing_cells": wearable_missing_cells,
        "wearable_total_cells": wearable_total_cells,
        "wearable_missing_percent": round(wearable_missing_cells / wearable_total_cells * 100, 2),
        "missing_by_column": missing_by_column,
        "comparison_with_reference_final_student_day_table_v01": compare_with_reference(table),
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
            clean_row.append("N/A" if text.lower() in {"nan", "none"} else text)
        lines.append("| " + " | ".join(clean_row) + " |")
    return "\n".join(lines)


def write_report(table: pd.DataFrame, audit: dict[str, Any]) -> None:
    missing_summary = (
        pd.DataFrame(
            [
                {
                    "column": column,
                    "missing_rows": missing,
                    "missing_percent": missing / len(table) * 100,
                }
                for column, missing in audit["missing_by_column"].items()
            ]
        )
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )
    reference_comparison = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in audit["comparison_with_reference_final_student_day_table_v01"].items()]
    )

    lines = [
        "# 原始 Student-Day 表合并报告",
        "",
        "## 目的",
        "",
        "本步骤从 `SSAQS dataset/` 中每个学生的原始 CSV 文件重新合并 student-day 表。该表不做填补、不做标准化，也不删除缺失 wearable 的日期，目的是为后续 leakage-safe preprocessing 提供干净起点。",
        "",
        "## 合并规则",
        "",
        "- `daily_questions.csv`：使用 `timeStampStart` 转成日期；每条问卷记录保留为一行，因此同一学生同一天多次问卷会暂时保留重复行。",
        "- `stress_label`：根据 `stress` 重新生成，规则为 Low = 0-17，Medium = 18-38，High = 39-100。",
        "- `sleep.csv`：按 student-date 聚合，取 `overall_score` 和 `deep_sleep_in_minutes` 的日均值。",
        "- `steps.csv`：按 student-date 汇总每日总步数。",
        "- `activity_level.csv`：按 student-date 统计各活动等级出现次数，作为对应活动分钟数。",
        "- `hrv.csv`：按 student-date 计算 RMSSD、low frequency、high frequency 的日均值。",
        "- `oxygen.csv`：按 student-date 计算血氧均值和标准差。",
        "- `stress.csv`：按 student-date 合并 Fitbit stress score；该列后续不作为主模型输入，仅保留用于审计。",
        "",
        "## 合并结果",
        "",
        f"- 行数：{audit['rows']}",
        f"- 学生数：{audit['students']}",
        f"- unique student-day 数：{audit['unique_student_days']}",
        f"- 日期范围：{audit['date_min']} 到 {audit['date_max']}",
        f"- 重复 student-day 行数：{audit['duplicate_student_day_rows']}",
        f"- 重复 student-day pair 数：{audit['duplicate_student_day_pairs']}",
        f"- wearable 缺失单元格比例：{audit['wearable_missing_percent']:.2f}%",
        "",
        "## 标签分布",
        "",
        dataframe_to_markdown(
            pd.DataFrame(
                [{"stress_label": key, "count": value} for key, value in audit["label_counts"].items()]
            )
        ),
        "",
        "## 缺失值概览",
        "",
        dataframe_to_markdown(missing_summary),
        "",
        "## 与旧版 `final_student_day_table_v01.csv` 对比",
        "",
        dataframe_to_markdown(reference_comparison),
        "",
        "## 后续处理建议",
        "",
        "- 下一步应先处理同一学生同一天的多次问卷记录，再做 subject-aware train/test split。",
        "- 不应在这张原始合并表上做全局填补或全局标准化。",
        "- 缺失值应放到模型 pipeline 中处理，确保 imputer/scaler 只在训练数据或训练折内部拟合。",
        "",
    ]
    MERGE_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw data directory: {RAW_DIR}")

    OUT_DIR.mkdir(exist_ok=True)
    table = build_raw_student_day_table()
    audit = build_audit(table)

    table.to_csv(RAW_STUDENT_DAY_PATH, index=False)
    MERGE_AUDIT_JSON_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(table, audit)

    print(f"Wrote {RAW_STUDENT_DAY_PATH.relative_to(ROOT)}")
    print(f"Wrote {MERGE_AUDIT_JSON_PATH.relative_to(ROOT)}")
    print(f"Wrote {MERGE_REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
