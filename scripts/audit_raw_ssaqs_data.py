from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "SSAQS dataset"
OUT_DIR = ROOT / "modeling_outputs"

FILE_AUDIT_PATH = OUT_DIR / "raw_file_audit_by_student.csv"
TARGET_COVERAGE_PATH = OUT_DIR / "raw_target_coverage_by_student_file.csv"
STUDENT_AVAILABILITY_PATH = OUT_DIR / "raw_student_day_availability.csv"
SUMMARY_JSON_PATH = OUT_DIR / "raw_data_audit.json"
SUMMARY_MD_PATH = OUT_DIR / "raw_data_audit.md"

EXPECTED_STUDENT_FILES = [
    "daily_questions.csv",
    "sleep.csv",
    "activity_level.csv",
    "steps.csv",
    "hrv.csv",
    "oxygen.csv",
    "stress.csv",
]
WEARABLE_FILES = [
    "sleep.csv",
    "activity_level.csv",
    "steps.csv",
    "hrv.csv",
    "oxygen.csv",
]


def student_dirs() -> list[Path]:
    return sorted(
        [path for path in RAW_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def parse_date_series(series: pd.Series, unix_seconds: bool = False) -> pd.Series:
    if unix_seconds:
        return pd.to_datetime(series, unit="s", errors="coerce", utc=True).dt.date
    return pd.to_datetime(series, errors="coerce", utc=True).dt.date


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def dates_for_file(path: Path, file_name: str) -> set:
    data = read_csv_if_exists(path)
    if data is None or data.empty:
        return set()

    if file_name == "daily_questions.csv":
        if "stress" in data.columns:
            data = data[data["stress"].notna()]
        dates = parse_date_series(data["timeStampStart"], unix_seconds=True)
    elif file_name == "stress.csv":
        dates = parse_date_series(data["DATE"])
    else:
        dates = parse_date_series(data["timestamp"])
    return set(dates.dropna())


def value_columns(data: pd.DataFrame, file_name: str) -> list[str]:
    if file_name == "daily_questions.csv":
        return [column for column in ["stress", "anxiety"] if column in data.columns]
    if file_name == "stress.csv":
        return [
            column
            for column in ["STRESS_SCORE", "CALCULATION_FAILED"]
            if column in data.columns
        ]
    return [column for column in data.columns if column not in {"timestamp", "DATE", "UPDATED_AT"}]


def date_column(data: pd.DataFrame, file_name: str) -> pd.Series:
    if file_name == "daily_questions.csv":
        return parse_date_series(data["timeStampStart"], unix_seconds=True)
    if file_name == "stress.csv":
        return parse_date_series(data["DATE"])
    return parse_date_series(data["timestamp"])


def audit_file(student_dir: Path, file_name: str) -> dict[str, Any]:
    path = student_dir / file_name
    data = read_csv_if_exists(path)
    base_row: dict[str, Any] = {
        "student_id": student_dir.name,
        "file": file_name,
        "exists": int(data is not None),
        "rows": 0,
        "columns": "",
        "unique_dates": 0,
        "missing_value_cells": 0,
        "total_value_cells": 0,
        "missing_value_percent": 0.0,
        "min_date": "",
        "max_date": "",
    }
    if data is None:
        return base_row

    value_cols = value_columns(data, file_name)
    dates = date_column(data, file_name) if not data.empty else pd.Series([], dtype=object)
    valid_dates = dates.dropna()
    missing_cells = int(data[value_cols].isna().sum().sum()) if value_cols else 0
    total_cells = int(len(data) * len(value_cols)) if value_cols else 0

    base_row.update(
        {
            "rows": int(len(data)),
            "columns": ", ".join(data.columns),
            "unique_dates": int(valid_dates.nunique()),
            "missing_value_cells": missing_cells,
            "total_value_cells": total_cells,
            "missing_value_percent": round(
                missing_cells / total_cells * 100, 2
            )
            if total_cells
            else 0.0,
            "min_date": str(min(valid_dates)) if len(valid_dates) else "",
            "max_date": str(max(valid_dates)) if len(valid_dates) else "",
        }
    )
    return base_row


def build_audit_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    file_rows = []
    coverage_rows = []
    availability_rows = []

    for student_dir in student_dirs():
        target_dates = dates_for_file(student_dir / "daily_questions.csv", "daily_questions.csv")
        file_date_sets = {
            file_name: dates_for_file(student_dir / file_name, file_name)
            for file_name in EXPECTED_STUDENT_FILES
        }
        complete_wearable_dates = set(target_dates)
        for file_name in WEARABLE_FILES:
            complete_wearable_dates &= file_date_sets[file_name]

        availability_rows.append(
            {
                "student_id": student_dir.name,
                "target_days": len(target_dates),
                "complete_wearable_target_days": len(complete_wearable_dates),
                "complete_wearable_target_percent": round(
                    len(complete_wearable_dates) / len(target_dates) * 100, 2
                )
                if target_dates
                else 0.0,
                "has_all_expected_files": int(
                    all((student_dir / file_name).exists() for file_name in EXPECTED_STUDENT_FILES)
                ),
            }
        )

        for file_name in EXPECTED_STUDENT_FILES:
            file_rows.append(audit_file(student_dir, file_name))
            file_dates = file_date_sets[file_name]
            coverage_rows.append(
                {
                    "student_id": student_dir.name,
                    "file": file_name,
                    "target_days": len(target_dates),
                    "covered_target_days": len(target_dates & file_dates),
                    "target_coverage_percent": round(
                        len(target_dates & file_dates) / len(target_dates) * 100, 2
                    )
                    if target_dates
                    else 0.0,
                }
            )

    return (
        pd.DataFrame(file_rows),
        pd.DataFrame(coverage_rows),
        pd.DataFrame(availability_rows),
    )


def label_from_stress(stress: float) -> str:
    if pd.isna(stress):
        return ""
    if stress <= 17:
        return "Low"
    if stress <= 38:
        return "Medium"
    return "High"


def audit_daily_questions() -> dict[str, Any]:
    frames = []
    for student_dir in student_dirs():
        data = read_csv_if_exists(student_dir / "daily_questions.csv")
        if data is None:
            continue
        data = data.copy()
        data["student_id"] = student_dir.name
        data["date"] = parse_date_series(data["timeStampStart"], unix_seconds=True)
        frames.append(data)

    if not frames:
        return {}

    daily = pd.concat(frames, ignore_index=True)
    duplicate_pairs = (
        daily[daily.duplicated(["student_id", "date"], keep=False)][["student_id", "date"]]
        .drop_duplicates()
        .shape[0]
    )
    rebuilt_label = daily["stress"].map(label_from_stress)
    label_mismatches = None
    if "stress_label" in daily.columns:
        label_mismatches = int((rebuilt_label != daily["stress_label"]).sum())

    return {
        "daily_question_rows": int(len(daily)),
        "daily_question_students": int(daily["student_id"].nunique()),
        "daily_question_unique_student_days": int(
            daily[["student_id", "date"]].drop_duplicates().shape[0]
        ),
        "daily_question_duplicate_student_day_pairs": int(duplicate_pairs),
        "has_stress_label_column": "stress_label" in daily.columns,
        "stress_label_mismatches_under_current_rule": label_mismatches,
        "stress_min": float(daily["stress"].min()),
        "stress_max": float(daily["stress"].max()),
        "stress_missing_rows": int(daily["stress"].isna().sum()),
        "anxiety_missing_rows": int(daily["anxiety"].isna().sum()),
    }


def build_summary(
    file_audit: pd.DataFrame,
    coverage: pd.DataFrame,
    availability: pd.DataFrame,
) -> dict[str, Any]:
    file_counts = (
        file_audit[file_audit["exists"] == 1]
        .groupby("file", observed=True)
        .size()
        .to_dict()
    )
    coverage_summary = (
        coverage.groupby("file", observed=True)
        .agg(
            total_target_days=("target_days", "sum"),
            covered_target_days=("covered_target_days", "sum"),
            mean_student_coverage_percent=("target_coverage_percent", "mean"),
        )
        .reset_index()
    )
    coverage_summary["overall_target_coverage_percent"] = (
        coverage_summary["covered_target_days"]
        / coverage_summary["total_target_days"]
        * 100
    ).round(2)

    return {
        "raw_dir": RAW_DIR.name,
        "student_dirs": len(student_dirs()),
        "expected_student_files": EXPECTED_STUDENT_FILES,
        "file_counts": {key: int(value) for key, value in file_counts.items()},
        "students_with_all_expected_files": int(
            availability["has_all_expected_files"].sum()
        ),
        "total_target_days": int(availability["target_days"].sum()),
        "complete_wearable_target_days": int(
            availability["complete_wearable_target_days"].sum()
        ),
        "complete_wearable_target_percent": round(
            availability["complete_wearable_target_days"].sum()
            / availability["target_days"].sum()
            * 100,
            2,
        ),
        "daily_questions": audit_daily_questions(),
        "coverage_by_file": coverage_summary.to_dict(orient="records"),
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


def write_markdown_summary(
    summary: dict[str, Any],
    file_audit: pd.DataFrame,
    coverage: pd.DataFrame,
    availability: pd.DataFrame,
) -> None:
    coverage_summary = pd.DataFrame(summary["coverage_by_file"])
    missing_file_summary = (
        file_audit[file_audit["exists"] == 0]
        .groupby("file", observed=True)
        .size()
        .rename("missing_student_count")
        .reset_index()
    )
    low_availability = availability.sort_values("complete_wearable_target_percent").head(10)

    lines = [
        "# Raw SSAQS Data Audit",
        "",
        "## Purpose",
        "",
        "This audit records the structure, coverage, and missingness of the original `SSAQS dataset/` directory before any modeling preprocessing.",
        "",
        "## Dataset Scale",
        "",
        f"- Student directories: {summary['student_dirs']}",
        f"- Students with all expected per-student files: {summary['students_with_all_expected_files']}",
        f"- Total target days from daily questionnaires: {summary['total_target_days']}",
        f"- Target days with complete wearable file coverage: {summary['complete_wearable_target_days']} ({summary['complete_wearable_target_percent']:.2f}%)",
        "",
        "## Expected File Counts",
        "",
        dataframe_to_markdown(
            pd.DataFrame(
                [{"file": key, "student_file_count": value} for key, value in summary["file_counts"].items()]
            ).sort_values("file")
        ),
        "",
        "## Missing Expected Files",
        "",
        dataframe_to_markdown(missing_file_summary)
        if not missing_file_summary.empty
        else "No expected per-student files are missing.",
        "",
        "## Coverage Against Daily Questionnaire Target Days",
        "",
        dataframe_to_markdown(coverage_summary.sort_values("file")),
        "",
        "## Students With Lowest Complete Wearable Coverage",
        "",
        dataframe_to_markdown(low_availability),
        "",
        "## Daily Questionnaire Audit",
        "",
        dataframe_to_markdown(
            pd.DataFrame([summary["daily_questions"]]).T.reset_index().rename(
                columns={"index": "metric", 0: "value"}
            )
        ),
        "",
        "## Implications for Modeling",
        "",
        "- The original data contains substantial wearable missingness and should not be globally standardized before splitting.",
        "- A leakage-safe modeling pipeline should split by student first, then fit imputation and scaling only within the training folds.",
        "- Students with missing wearable files or very low coverage should be explicitly discussed as a limitation or handled by a documented filtering rule.",
        "",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Missing raw data directory: {RAW_DIR}")

    OUT_DIR.mkdir(exist_ok=True)
    file_audit, coverage, availability = build_audit_tables()
    summary = build_summary(file_audit, coverage, availability)

    file_audit.to_csv(FILE_AUDIT_PATH, index=False)
    coverage.to_csv(TARGET_COVERAGE_PATH, index=False)
    availability.to_csv(STUDENT_AVAILABILITY_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, file_audit, coverage, availability)

    print(f"Wrote {FILE_AUDIT_PATH.relative_to(ROOT)}")
    print(f"Wrote {TARGET_COVERAGE_PATH.relative_to(ROOT)}")
    print(f"Wrote {STUDENT_AVAILABILITY_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_JSON_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_MD_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
