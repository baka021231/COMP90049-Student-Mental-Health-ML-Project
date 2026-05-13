from __future__ import annotations

import os
from pathlib import Path
from html import escape

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "final_student_day_table_v01_processed.csv"
OUT_DIR = ROOT / "eda_outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
RAW_DIR = ROOT / "SSAQS dataset"


FEATURE_GROUPS = {
    "Sleep": ["sleep_score", "deep_sleep_minutes"],
    "Activity": [
        "total_steps",
        "sedentary_minutes",
        "lightly_active_minutes",
        "moderately_active_minutes",
        "very_active_minutes",
    ],
    "HRV": ["avg_rmssd", "avg_low_frequency", "avg_high_frequency"],
    "SpO2": ["avg_oxygen", "std_oxygen"],
    "Fitbit stress score": ["STRESS_SCORE", "CALCULATION_FAILED"],
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / ".matplotlib-cache").mkdir(parents=True, exist_ok=True)


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["student_id"] = df["student_id"].astype(str)
    df["stress_label"] = pd.Categorical(
        df["stress_label"], categories=["Low", "Medium", "High"], ordered=True
    )
    return df


def student_dirs() -> list[Path]:
    return sorted(
        [path for path in RAW_DIR.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def parse_date_series(series: pd.Series, unix_seconds: bool = False) -> pd.Series:
    if unix_seconds:
        return pd.to_datetime(series, unit="s", errors="coerce", utc=True).dt.date
    return pd.to_datetime(series, errors="coerce", utc=True).dt.date


def raw_file_date_info(student_dir: Path, file_name: str) -> tuple[int, int, int, str, str]:
    path = student_dir / file_name
    if not path.exists():
        return 0, 0, 0, "", ""

    data = pd.read_csv(path, encoding="utf-8-sig")
    if data.empty:
        return 0, 0, 0, "", ""

    if file_name == "daily_questions.csv":
        dates = parse_date_series(data["timeStampStart"], unix_seconds=True)
        value_cols = ["stress", "anxiety"]
    elif file_name == "stress.csv":
        dates = parse_date_series(data["DATE"])
        value_cols = ["STRESS_SCORE", "CALCULATION_FAILED"]
    else:
        dates = parse_date_series(data["timestamp"])
        value_cols = [col for col in data.columns if col != "timestamp"]

    missing_cells = int(data[value_cols].isna().sum().sum())
    unique_dates = dates.dropna().nunique()
    valid_dates = dates.dropna()
    min_date = str(min(valid_dates)) if len(valid_dates) else ""
    max_date = str(max(valid_dates)) if len(valid_dates) else ""
    return len(data), int(unique_dates), missing_cells, min_date, max_date


def raw_dates_for_file(student_dir: Path, file_name: str) -> set:
    path = student_dir / file_name
    if not path.exists():
        return set()
    data = pd.read_csv(path, encoding="utf-8-sig")
    if data.empty:
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


def save_raw_data_audit(processed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    file_names = [
        "daily_questions.csv",
        "sleep.csv",
        "activity_level.csv",
        "steps.csv",
        "hrv.csv",
        "oxygen.csv",
        "stress.csv",
    ]
    rows = []
    target_coverage_rows = []
    raw_target_rows = []

    processed_pairs = set(zip(processed["student_id"], processed["date"].dt.date))

    for student_dir in student_dirs():
        student = student_dir.name
        target_dates = raw_dates_for_file(student_dir, "daily_questions.csv")
        processed_dates = {date for sid, date in processed_pairs if sid == student}
        raw_target_rows.append(
            {
                "student_id": student,
                "raw_target_days": len(target_dates),
                "processed_days": len(processed_dates),
                "target_days_kept_in_processed": len(target_dates & processed_dates),
                "target_days_not_in_processed": len(target_dates - processed_dates),
                "processed_days_not_in_raw_target": len(processed_dates - target_dates),
            }
        )
        for file_name in file_names:
            row_count, unique_dates, missing_cells, min_date, max_date = raw_file_date_info(
                student_dir, file_name
            )
            rows.append(
                {
                    "student_id": student,
                    "file": file_name,
                    "exists": int((student_dir / file_name).exists()),
                    "raw_rows": row_count,
                    "unique_dates": unique_dates,
                    "missing_value_cells": missing_cells,
                    "min_date": min_date,
                    "max_date": max_date,
                }
            )
            if target_dates:
                file_dates = raw_dates_for_file(student_dir, file_name)
                target_coverage_rows.append(
                    {
                        "student_id": student,
                        "file": file_name,
                        "target_days": len(target_dates),
                        "covered_target_days": len(target_dates & file_dates),
                        "coverage_percent": round(
                            len(target_dates & file_dates) / len(target_dates) * 100, 2
                        ),
                    }
                )

    raw_file_audit = pd.DataFrame(rows)
    raw_file_audit.to_csv(TABLE_DIR / "raw_file_audit_by_student.csv", index=False)

    raw_target_audit = pd.DataFrame(raw_target_rows)
    raw_target_audit.to_csv(TABLE_DIR / "raw_target_to_processed_audit.csv", index=False)

    target_coverage = pd.DataFrame(target_coverage_rows)
    target_coverage.to_csv(TABLE_DIR / "raw_modality_coverage_against_target_days.csv", index=False)

    modality_summary = (
        target_coverage.groupby("file", observed=True)
        .agg(
            students=("student_id", "nunique"),
            total_target_days=("target_days", "sum"),
            covered_target_days=("covered_target_days", "sum"),
            mean_student_coverage_percent=("coverage_percent", "mean"),
        )
        .reset_index()
    )
    modality_summary["overall_target_day_coverage_percent"] = (
        modality_summary["covered_target_days"]
        / modality_summary["total_target_days"]
        * 100
    ).round(2)
    modality_summary["mean_student_coverage_percent"] = modality_summary[
        "mean_student_coverage_percent"
    ].round(2)
    modality_summary.to_csv(TABLE_DIR / "raw_modality_coverage_summary.csv", index=False)

    return raw_file_audit, raw_target_audit, modality_summary


def save_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    per_student = df.groupby("student_id", observed=True).size()
    label_counts = df["stress_label"].value_counts(sort=False)
    label_pct = label_counts / len(df) * 100

    summary = pd.DataFrame(
        {
            "Metric": [
                "Students",
                "Student-day observations",
                "Date range",
                "Mean days per student",
                "Median days per student",
                "Min days per student",
                "Max days per student",
                "Low stress count",
                "Medium stress count",
                "High stress count",
                "Low stress percent",
                "Medium stress percent",
                "High stress percent",
            ],
            "Value": [
                df["student_id"].nunique(),
                len(df),
                f"{df['date'].min().date()} to {df['date'].max().date()}",
                round(per_student.mean(), 2),
                round(per_student.median(), 2),
                int(per_student.min()),
                int(per_student.max()),
                int(label_counts["Low"]),
                int(label_counts["Medium"]),
                int(label_counts["High"]),
                round(label_pct["Low"], 2),
                round(label_pct["Medium"], 2),
                round(label_pct["High"], 2),
            ],
        }
    )
    summary.to_csv(TABLE_DIR / "dataset_summary.csv", index=False)
    per_student.rename("student_day_count").to_csv(
        TABLE_DIR / "student_day_counts.csv", header=True
    )
    return summary


def save_duplicate_check(df: pd.DataFrame) -> pd.DataFrame:
    duplicated = df[df.duplicated(["student_id", "date"], keep=False)].sort_values(
        ["student_id", "date"]
    )
    if duplicated.empty:
        duplicate_groups = pd.DataFrame(
            columns=[
                "student_id",
                "date",
                "n_rows",
                "stress_values",
                "stress_labels",
                "same_stress_label",
                "same_stress_value",
            ]
        )
    else:
        duplicate_groups = (
            duplicated.groupby(["student_id", "date"], observed=True)
            .agg(
                n_rows=("stress", "size"),
                stress_values=("stress", lambda x: "; ".join(map(str, x.tolist()))),
                stress_labels=(
                    "stress_label",
                    lambda x: "; ".join(map(str, x.astype(str).tolist())),
                ),
                same_stress_label=("stress_label", lambda x: x.nunique() == 1),
                same_stress_value=("stress", lambda x: x.nunique() == 1),
            )
            .reset_index()
        )
    duplicate_summary = pd.DataFrame(
        {
            "metric": [
                "duplicate_student_day_rows",
                "duplicate_student_day_pairs",
                "duplicate_pairs_with_conflicting_stress_value",
                "duplicate_pairs_with_conflicting_stress_label",
            ],
            "value": [
                len(duplicated),
                duplicated[["student_id", "date"]].drop_duplicates().shape[0],
                int((~duplicate_groups["same_stress_value"]).sum()),
                int((~duplicate_groups["same_stress_label"]).sum()),
            ],
        }
    )
    duplicate_summary.to_csv(TABLE_DIR / "processed_duplicate_check.csv", index=False)
    duplicated.to_csv(TABLE_DIR / "processed_duplicate_rows.csv", index=False)
    duplicate_groups.to_csv(TABLE_DIR / "processed_duplicate_groups.csv", index=False)
    return duplicate_summary


def save_label_threshold_check(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = (
        df.groupby("stress_label", observed=True)["stress"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .round(3)
    )
    thresholds.to_csv(TABLE_DIR / "stress_label_threshold_check.csv")
    return thresholds


def save_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, cols in FEATURE_GROUPS.items():
        existing = [col for col in cols if col in df.columns]
        if not existing:
            continue
        total_cells = len(df) * len(existing)
        missing_cells = int(df[existing].isna().sum().sum())
        rows.append(
            {
                "feature_group": group,
                "features": ", ".join(existing),
                "missing_cells": missing_cells,
                "total_cells": total_cells,
                "missing_percent": round(missing_cells / total_cells * 100, 2),
            }
        )

    group_missing = pd.DataFrame(rows)
    group_missing.to_csv(TABLE_DIR / "missingness_by_feature_group.csv", index=False)

    col_missing = (
        df.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percent=lambda x: (x["missing_count"] / len(df) * 100).round(2))
    )
    col_missing.to_csv(TABLE_DIR / "missingness_by_column.csv")
    return group_missing


def save_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col
        for cols in FEATURE_GROUPS.values()
        for col in cols
        if col in df.columns and col != "CALCULATION_FAILED"
    ]
    summary = df[feature_cols].describe().T.round(3)
    summary.to_csv(TABLE_DIR / "feature_descriptive_statistics.csv")
    return summary


def save_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        col
        for cols in FEATURE_GROUPS.values()
        for col in cols
        if col in df.columns and col not in {"CALCULATION_FAILED"}
    ]
    rows = []
    for col in feature_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "feature": col,
                "q1": round(q1, 3),
                "q3": round(q3, 3),
                "iqr": round(iqr, 3),
                "lower_fence": round(lower, 3),
                "upper_fence": round(upper, 3),
                "outlier_count": outlier_count,
                "outlier_percent": round(outlier_count / len(series) * 100, 2),
            }
        )
    outliers = pd.DataFrame(rows).sort_values("outlier_percent", ascending=False)
    outliers.to_csv(TABLE_DIR / "outlier_summary_iqr.csv", index=False)
    return outliers


def save_feature_by_label(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
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
    available = [col for col in feature_cols if col in df.columns]
    grouped = df.groupby("stress_label", observed=True)[available].agg(["mean", "median"])
    grouped.columns = [f"{feature}_{stat}" for feature, stat in grouped.columns]
    grouped = grouped.round(3)
    grouped.to_csv(TABLE_DIR / "feature_summary_by_stress_label.csv")
    return grouped


def save_correlations(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number").drop(
        columns=["student_id"], errors="ignore"
    )
    corr = numeric.corr(method="spearman")["stress"].sort_values(ascending=False)
    corr.to_csv(TABLE_DIR / "spearman_correlation_with_stress.csv", header=["rho"])
    return corr


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#222">{escape(str(text))}</text>'
    )


def to_markdown_table(obj: pd.DataFrame | pd.Series) -> str:
    if isinstance(obj, pd.Series):
        frame = obj.to_frame()
    else:
        frame = obj.copy()
    frame = frame.reset_index()
    headers = [str(col) for col in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join(parts), encoding="utf-8")


def plot_stress_histogram(df: pd.DataFrame) -> None:
    bins = list(range(0, 101, 5))
    counts = pd.cut(df["stress"], bins=bins, right=False, include_lowest=True).value_counts(sort=False)
    width, height = 880, 470
    left, right, top, bottom = 70, 35, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = counts.max()
    bar_w = plot_w / len(counts)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Continuous Stress Score Distribution", 20),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        svg_text(width / 2, height - 25, "Stress score", 13),
        svg_text(24, top + plot_h / 2, "Count", 13),
    ]
    for i, count in enumerate(counts.values):
        x = left + i * bar_w
        bar_h = count / max_count * plot_h if max_count else 0
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{bar_h:.1f}" fill="#4C78A8"/>'
        )
    for tick in range(0, 101, 10):
        x = left + tick / 100 * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#333"/>')
        parts.append(svg_text(x, top + plot_h + 22, str(tick), 10))
    parts.append("</svg>")
    write_svg(FIG_DIR / "stress_continuous_distribution.svg", parts)


def plot_student_day_counts(df: pd.DataFrame) -> None:
    counts = df.groupby("student_id", observed=True).size().sort_values()
    width, height = 960, 560
    left, right, top, bottom = 65, 30, 55, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = counts.max()
    bar_w = plot_w / len(counts)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Student-Day Observations per Student", 20),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        svg_text(22, top + plot_h / 2, "Days", 13),
        svg_text(width / 2, height - 22, "Student ID", 13),
    ]
    for i, (student, count) in enumerate(counts.items()):
        x = left + i * bar_w
        bar_h = count / max_count * plot_h
        y = top + plot_h - bar_h
        color = "#D9534F" if count < 14 else "#4C78A8"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" height="{bar_h:.1f}" fill="{color}"/>'
        )
        if i % 2 == 0:
            parts.append(svg_text(x + bar_w / 2, top + plot_h + 18, student, 9))
    parts.append(svg_text(width - 145, 75, "Red: fewer than 14 days", 12, "start"))
    parts.append("</svg>")
    write_svg(FIG_DIR / "student_day_counts.svg", parts)


def plot_feature_histograms(df: pd.DataFrame) -> None:
    selected = [
        "sleep_score",
        "deep_sleep_minutes",
        "total_steps",
        "sedentary_minutes",
        "avg_rmssd",
        "avg_low_frequency",
        "avg_oxygen",
        "std_oxygen",
    ]
    available = [col for col in selected if col in df.columns]
    width, height = 1000, 720
    cols = 2
    rows = 4
    margin_x, margin_y = 70, 55
    cell_w = (width - margin_x * 2) / cols
    cell_h = (height - margin_y * 2) / rows
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Feature Distributions", 20),
    ]
    for idx, feature in enumerate(available):
        r, c = divmod(idx, cols)
        x0 = margin_x + c * cell_w
        y0 = margin_y + r * cell_h
        inner_w = cell_w - 55
        inner_h = cell_h - 50
        series = df[feature].dropna()
        bins = pd.cut(series, bins=12)
        counts = bins.value_counts(sort=False)
        max_count = counts.max()
        parts.append(svg_text(x0 + inner_w / 2, y0 + 5, feature, 13))
        parts.append(f'<line x1="{x0}" y1="{y0 + inner_h}" x2="{x0 + inner_w}" y2="{y0 + inner_h}" stroke="#333"/>')
        parts.append(f'<line x1="{x0}" y1="{y0 + 20}" x2="{x0}" y2="{y0 + inner_h}" stroke="#333"/>')
        bw = inner_w / len(counts)
        for i, count in enumerate(counts.values):
            bar_h = count / max_count * (inner_h - 25) if max_count else 0
            x = x0 + i * bw
            y = y0 + inner_h - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bw - 1, 1):.1f}" height="{bar_h:.1f}" fill="#72B7B2"/>'
            )
    parts.append("</svg>")
    write_svg(FIG_DIR / "feature_distributions.svg", parts)


def plot_raw_modality_coverage(modality_summary: pd.DataFrame) -> None:
    data = modality_summary.sort_values("overall_target_day_coverage_percent")
    width, height = 860, 420
    left, right, top, bottom = 190, 40, 55, 55
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(data)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Raw Modality Coverage Against Target Days", 20),
    ]
    for i, row in enumerate(data.itertuples()):
        y = top + i * row_h + 8
        pct = float(row.overall_target_day_coverage_percent)
        bar_w = pct / 100 * plot_w
        parts.append(svg_text(left - 10, y + 14, row.file, 12, "end"))
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="18" fill="#eee"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="18" fill="#4C78A8"/>')
        parts.append(svg_text(left + bar_w + 8, y + 14, f"{pct:.1f}%", 11, "start"))
    parts.append("</svg>")
    write_svg(FIG_DIR / "raw_modality_coverage.svg", parts)


def plot_weekday_weekend(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["day_type"] = tmp["date"].dt.dayofweek.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    summary = tmp.groupby("day_type", observed=True)["stress"].agg(["count", "mean", "median", "std"]).round(3)
    summary.to_csv(TABLE_DIR / "weekday_weekend_stress_summary.csv")
    width, height = 560, 380
    left, right, top, bottom = 80, 40, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_mean = summary["mean"].max()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Mean Stress: Weekday vs Weekend", 20),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    labels = ["Weekday", "Weekend"]
    bar_w = 105
    gap = (plot_w - bar_w * 2) / 3
    for i, label in enumerate(labels):
        if label not in summary.index:
            continue
        mean = summary.loc[label, "mean"]
        x = left + gap + i * (bar_w + gap)
        bar_h = mean / max_mean * plot_h
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="#F58518"/>')
        parts.append(svg_text(x + bar_w / 2, y - 8, f"{mean:.2f}", 12))
        parts.append(svg_text(x + bar_w / 2, top + plot_h + 25, label, 12))
    parts.append("</svg>")
    write_svg(FIG_DIR / "weekday_weekend_stress.svg", parts)
    return summary


def plot_label_distribution(df: pd.DataFrame) -> None:
    counts = df["stress_label"].value_counts(sort=False)
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    width, height = 760, 460
    left, right, top, bottom = 90, 40, 55, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = counts.max()
    bar_w = 100
    gap = (plot_w - bar_w * len(counts)) / (len(counts) + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Stress Label Distribution", 20),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        svg_text(24, top + plot_h / 2, "Count", 13, "middle"),
        svg_text(width / 2, height - 25, "Stress label", 13),
    ]
    for i, (label, count) in enumerate(counts.items()):
        x = left + gap + i * (bar_w + gap)
        bar_h = count / max_count * plot_h
        y = top + plot_h - bar_h
        pct = count / len(df) * 100
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{colors[i]}"/>',
                svg_text(x + bar_w / 2, y - 10, f"{int(count)} ({pct:.1f}%)", 12),
                svg_text(x + bar_w / 2, top + plot_h + 28, str(label), 13),
            ]
        )
    parts.append("</svg>")
    (FIG_DIR / "figure_1_stress_label_distribution.svg").write_text(
        "\n".join(parts), encoding="utf-8"
    )


def plot_temporal_trend(df: pd.DataFrame) -> pd.DataFrame:
    start = df["date"].min()
    tmp = df.copy()
    tmp["semester_week"] = ((tmp["date"] - start).dt.days // 7) + 1
    weekly = (
        tmp.groupby("semester_week", observed=True)["stress"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    weekly.to_csv(TABLE_DIR / "weekly_stress_trend.csv", index=False)

    width, height = 860, 500
    left, right, top, bottom = 80, 35, 55, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min, x_max = weekly["semester_week"].min(), weekly["semester_week"].max()
    y_min = max(0, weekly["mean"].min() - 3)
    y_max = weekly["mean"].max() + 3

    def x_scale(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def y_scale(v: float) -> float:
        return top + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    points = " ".join(
        f"{x_scale(row.semester_week):.1f},{y_scale(row.mean):.1f}"
        for row in weekly.itertuples()
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Mean Self-Reported Stress by Semester Week", 20),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
        svg_text(28, top + plot_h / 2, "Mean stress", 13),
        svg_text(width / 2, height - 25, "Semester week", 13),
        f'<polyline points="{points}" fill="none" stroke="#4C78A8" stroke-width="3"/>',
    ]
    for row in weekly.itertuples():
        x, y = x_scale(row.semester_week), y_scale(row.mean)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#4C78A8"/>')
        if int(row.semester_week) % 2 == 1:
            parts.append(svg_text(x, top + plot_h + 22, str(int(row.semester_week)), 11))
    for tick in range(int(y_min // 5 * 5), int(y_max) + 1, 5):
        y = y_scale(tick)
        parts.extend(
            [
                f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#333"/>',
                svg_text(left - 10, y + 4, str(tick), 11, "end"),
                f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#ddd"/>',
            ]
        )
    parts.append("</svg>")
    (FIG_DIR / "figure_4_temporal_stress_trend.svg").write_text(
        "\n".join(parts), encoding="utf-8"
    )
    return weekly


def plot_feature_means_by_label(df: pd.DataFrame) -> None:
    selected = [
        "sleep_score",
        "deep_sleep_minutes",
        "total_steps",
        "sedentary_minutes",
        "avg_rmssd",
        "avg_oxygen",
    ]
    available = [col for col in selected if col in df.columns]
    means = df.groupby("stress_label", observed=True)[available].mean()
    z_means = (means - df[available].mean()) / df[available].std(ddof=0)
    z_means.to_csv(TABLE_DIR / "standardised_feature_means_by_stress_label.csv")

    width, height = 920, 520
    left, right, top, bottom = 220, 170, 55, 45
    row_h = (height - top - bottom) / len(available)
    x0 = left + (width - left - right) / 2
    scale = 125
    colors = {"Low": "#4C78A8", "Medium": "#F58518", "High": "#54A24B"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 30, "Standardised Feature Means by Stress Label", 20),
        f'<line x1="{x0}" y1="{top - 10}" x2="{x0}" y2="{height - bottom + 10}" stroke="#999"/>',
    ]
    for i, feature in enumerate(available):
        y = top + i * row_h + row_h / 2
        parts.append(svg_text(left - 12, y + 4, feature, 12, "end"))
        for label in ["Low", "Medium", "High"]:
            value = float(z_means.loc[label, feature])
            x = x0 + value * scale
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[label]}"/>'
            )
    legend_x = width - right + 30
    for j, label in enumerate(["Low", "Medium", "High"]):
        y = top + 20 + j * 24
        parts.extend(
            [
                f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{colors[label]}"/>',
                svg_text(legend_x + 14, y + 4, label, 12, "start"),
            ]
        )
    parts.append(svg_text(x0 - scale, height - 12, "-1 SD", 11))
    parts.append(svg_text(x0, height - 12, "Mean", 11))
    parts.append(svg_text(x0 + scale, height - 12, "+1 SD", 11))
    parts.append("</svg>")
    (FIG_DIR / "feature_means_by_stress_label.svg").write_text(
        "\n".join(parts), encoding="utf-8"
    )


def write_report(
    df: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    missingness: pd.DataFrame,
    correlations: pd.Series,
    weekly: pd.DataFrame,
) -> None:
    label_counts = df["stress_label"].value_counts(sort=False)
    label_pct = label_counts / len(df) * 100
    top_corr = correlations.drop(labels=["stress"], errors="ignore").head(8)
    bottom_corr = correlations.tail(8)
    peak = weekly.loc[weekly["mean"].idxmax()]
    low = weekly.loc[weekly["mean"].idxmin()]

    lines = [
        "# EDA Summary",
        "",
        "## Dataset Scale",
        f"- Students: {df['student_id'].nunique()}",
        f"- Student-day observations: {len(df)}",
        f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}",
        f"- Mean days per student: {df.groupby('student_id', observed=True).size().mean():.2f}",
        "",
        "## Stress Label Distribution",
        f"- Low: {label_counts['Low']} ({label_pct['Low']:.1f}%)",
        f"- Medium: {label_counts['Medium']} ({label_pct['Medium']:.1f}%)",
        f"- High: {label_counts['High']} ({label_pct['High']:.1f}%)",
        "",
        "## Label Threshold Check",
        to_markdown_table(thresholds),
        "",
        "The processed table uses thresholds equivalent to Low = 0-17, "
        "Medium = 18-38, and High = 39-100. This differs from the draft "
        "report text, which currently describes 0-33 / 34-66 / 67-100 bins.",
        "",
        "## Missingness",
        to_markdown_table(missingness),
        "",
        "All modelling feature columns in the processed table have already been "
        "imputed or otherwise completed. The only remaining missing fields are "
        "Fitbit's own STRESS_SCORE and CALCULATION_FAILED, each missing in 153 rows "
        "(10.98%).",
        "",
        "## Strongest Spearman Correlations With Stress",
        to_markdown_table(top_corr.round(3)),
        "",
        "## Most Negative Spearman Correlations With Stress",
        to_markdown_table(bottom_corr.round(3)),
        "",
        "## Temporal Trend",
        f"- Peak mean stress: week {int(peak['semester_week'])}, mean = {peak['mean']:.2f}, n = {int(peak['count'])}",
        f"- Lowest mean stress: week {int(low['semester_week'])}, mean = {low['mean']:.2f}, n = {int(low['count'])}",
        "",
        "Generated figures:",
        "- figures/figure_1_stress_label_distribution.svg",
        "- figures/stress_continuous_distribution.svg",
        "- figures/figure_4_temporal_stress_trend.svg",
        "- figures/feature_means_by_stress_label.svg",
        "- figures/student_day_counts.svg",
        "- figures/feature_distributions.svg",
        "- figures/raw_modality_coverage.svg",
        "- figures/weekday_weekend_stress.svg",
    ]

    (OUT_DIR / "EDA_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_detailed_chinese_report(
    df: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    missingness: pd.DataFrame,
    raw_target_audit: pd.DataFrame,
    modality_summary: pd.DataFrame,
    outliers: pd.DataFrame,
    correlations: pd.Series,
    weekly: pd.DataFrame,
    weekday_weekend: pd.DataFrame,
) -> None:
    label_counts = df["stress_label"].value_counts(sort=False)
    label_pct = label_counts / len(df) * 100
    per_student = df.groupby("student_id", observed=True).size().sort_values()
    low_coverage_students = per_student[per_student < 14]
    duplicate_rows = int(
        duplicate_summary.loc[
            duplicate_summary["metric"] == "duplicate_student_day_rows", "value"
        ].iloc[0]
    )
    duplicate_pairs = int(
        duplicate_summary.loc[
            duplicate_summary["metric"] == "duplicate_student_day_pairs", "value"
        ].iloc[0]
    )
    conflicting_values = int(
        duplicate_summary.loc[
            duplicate_summary["metric"] == "duplicate_pairs_with_conflicting_stress_value",
            "value",
        ].iloc[0]
    )
    conflicting_labels = int(
        duplicate_summary.loc[
            duplicate_summary["metric"] == "duplicate_pairs_with_conflicting_stress_label",
            "value",
        ].iloc[0]
    )
    raw_totals = raw_target_audit[
        [
            "raw_target_days",
            "processed_days",
            "target_days_kept_in_processed",
            "target_days_not_in_processed",
        ]
    ].sum()
    peak = weekly.loc[weekly["mean"].idxmax()]
    trough = weekly.loc[weekly["mean"].idxmin()]
    top_outliers = outliers.head(8)
    top_corr = correlations.drop(labels=["stress"], errors="ignore").head(8).round(3)
    bottom_corr = correlations.tail(8).round(3)

    lines = [
        "# EDA Detailed Report",
        "",
        "## 1. Processed Table 审计",
        "",
        f"- 当前 processed table 为 `final_student_day_table_v01_processed.csv`。",
        f"- 样本规模：{df['student_id'].nunique()} 名学生，{len(df)} 条 student-day observations。",
        f"- 日期范围：{df['date'].min().date()} 至 {df['date'].max().date()}。",
        f"- 重复 `student_id + date` 行数：{duplicate_rows}，涉及 {duplicate_pairs} 个重复 student-day。",
        f"- 其中 {conflicting_values} 个重复 student-day 的 stress 数值不同，{conflicting_labels} 个重复 student-day 的 stress label 不同。",
        f"- 每名学生贡献天数：均值 {per_student.mean():.2f}，中位数 {per_student.median():.1f}，最少 {per_student.min()}，最多 {per_student.max()}。",
        "",
        "结论：processed table 目前存在少量重复 student-day。多数重复行的 wearable 特征相同，但问卷 stress/anxiety 不同，说明同一学生同一天可能提交了多次问卷。建模前不应保留这些重复行；应选择一个明确规则，例如按同日问卷平均 stress/anxiety 后重新分箱，或保留当天最后一次问卷记录。学生贡献天数差异也较大，少数学生样本很少，建模时需要使用 subject-aware split，并在 limitation 中说明测试集估计可能不稳定。",
        "",
        "数据特别少的学生（少于 14 天）：",
        to_markdown_table(low_coverage_students.rename("processed_days")),
        "",
        "相关图表：`figures/student_day_counts.svg`。",
        "",
        "相关重复检查表格：",
        "- `tables/processed_duplicate_check.csv`",
        "- `tables/processed_duplicate_groups.csv`",
        "- `tables/processed_duplicate_rows.csv`",
        "",
        "## 2. Raw Data Coverage / Missingness 审计",
        "",
        "这里的 raw audit 针对 `SSAQS dataset/` 下每个学生原始 CSV，而不是 processed table。统计口径是：以每个学生有 self-reported stress 的 raw target days 为基准，检查各模态在这些 target days 上是否有同日数据。",
        "",
        to_markdown_table(modality_summary),
        "",
        f"Raw target days 总数为 {int(raw_totals['raw_target_days'])}，processed table 中保留了 {int(raw_totals['processed_days'])} 个 student-days。raw target days 中有 {int(raw_totals['target_days_kept_in_processed'])} 天出现在 processed table，另有 {int(raw_totals['target_days_not_in_processed'])} 天没有进入 processed table。",
        "",
        "结论：不能只用 processed table 的 0% 缺失来描述原始数据缺失情况。processed table 已经经过筛选、聚合或填补；报告中应区分 raw coverage 和 processed missingness。",
        "",
        "相关表格：",
        "- `tables/raw_file_audit_by_student.csv`",
        "- `tables/raw_modality_coverage_against_target_days.csv`",
        "- `tables/raw_modality_coverage_summary.csv`",
        "- `tables/raw_target_to_processed_audit.csv`",
        "",
        "相关图表：`figures/raw_modality_coverage.svg`。",
        "",
        "## 3. Target EDA",
        "",
        f"- Low: {label_counts['Low']} ({label_pct['Low']:.1f}%)",
        f"- Medium: {label_counts['Medium']} ({label_pct['Medium']:.1f}%)",
        f"- High: {label_counts['High']} ({label_pct['High']:.1f}%)",
        "",
        "当前 processed table 的标签阈值为：",
        to_markdown_table(thresholds),
        "",
        "结论：类别存在轻度不平衡，但不是极端不平衡。模型评估应以 macro-F1 为主要指标。注意：当前实际分箱是 Low = 0-17、Medium = 18-38、High = 39-100，与草稿报告中 0-33 / 34-66 / 67-100 的写法不一致，必须修改。",
        "",
        "相关图表：",
        "- `figures/figure_1_stress_label_distribution.svg`",
        "- `figures/stress_continuous_distribution.svg`",
        "",
        "## 4. Processed Missingness EDA",
        "",
        to_markdown_table(missingness),
        "",
        "结论：processed table 中主要建模特征已经没有缺失值。剩余缺失集中在 Fitbit 自带的 `STRESS_SCORE` 和 `CALCULATION_FAILED`，各缺失 153 行。`STRESS_SCORE` 不建议作为主模型输入，因为它本身是 Fitbit 的压力估计，和目标变量有概念重叠，存在 leakage 风险。",
        "",
        "## 5. Feature Distribution / Outlier EDA",
        "",
        "特征描述统计显示，多数建模特征已经标准化为均值约 0、标准差约 1。IQR 异常值检查结果如下：",
        "",
        to_markdown_table(top_outliers),
        "",
        "结论：activity minutes、HRV frequency 等变量存在一定极端值，但这类纵向 wearable 数据本身容易出现长尾分布。建议暂不直接删除异常值，而是在建模中使用对异常值相对稳健的模型或标准化后的特征；若后续模型表现异常，再考虑 winsorization 作为敏感性分析。",
        "",
        "相关图表：",
        "- `figures/feature_distributions.svg`",
        "- `figures/feature_means_by_stress_label.svg`",
        "",
        "## 6. Feature 与 Stress 的关系",
        "",
        "Spearman 相关性最高的变量：",
        to_markdown_table(top_corr),
        "",
        "Spearman 相关性最低的变量：",
        to_markdown_table(bottom_corr),
        "",
        "结论：`anxiety` 与 stress 的相关性较高，rho = 0.64，但它是同日问卷变量，不属于 wearable passive sensing 特征。若研究目标是用 Fitbit/行为数据预测 stress，主实验中应谨慎排除 `anxiety`，可作为辅助分析或 upper-bound comparison。其他 wearable 单变量相关性普遍较弱，说明需要多特征模型，而不是依赖单个强预测变量。",
        "",
        "## 7. RQ3 Temporal EDA",
        "",
        f"- 平均压力最低：semester week {int(trough['semester_week'])}，mean = {trough['mean']:.2f}，n = {int(trough['count'])}。",
        f"- 平均压力最高：semester week {int(peak['semester_week'])}，mean = {peak['mean']:.2f}，n = {int(peak['count'])}。",
        "",
        "Weekday / weekend 对比：",
        to_markdown_table(weekday_weekend),
        "",
        "结论：总体压力在学期后期明显升高，支持 RQ3 中加入 semester week、lag features 和 rolling means。weekday/weekend 差异可以作为辅助观察，但更核心的时间信号来自 semester progression。",
        "",
        "相关图表：",
        "- `figures/figure_4_temporal_stress_trend.svg`",
        "- `figures/weekday_weekend_stress.svg`",
        "",
        "## 8. EDA-driven Preprocessing Decisions",
        "",
        "| 决策点 | 建议 | 理由 |",
        "| --- | --- | --- |",
        "| target 缺失 | 删除该 student-day | 无标签样本不能用于监督学习训练/评估 |",
        "| feature 缺失 | 不建议简单整行删除；优先 per-student median imputation | 样本量小，整行删除可能引入偏差并破坏时间连续性 |",
        f"| 重复 student-day | 需要处理；建议按同日问卷平均 stress/anxiety 后重新分箱，或保留当天最后一次问卷 | 当前存在 {duplicate_pairs} 个重复 student-day，其中 {conflicting_labels} 个 stress label 冲突 |",
        "| stress label 分箱 | 报告改为当前实际阈值，或重新确认是否要改回等宽分箱 | 当前数据实际是 0-17 / 18-38 / 39-100 |",
        "| 数据很少的学生 | 暂不自动删除，建模时使用 subject-aware split，并做 limitation 说明 | 删除学生会进一步缩小样本；但要承认估计不稳定 |",
        "| `STRESS_SCORE` | 主实验排除 | Fitbit 自带压力估计可能造成 leakage |",
        "| `anxiety` | 主实验建议排除或单独作为辅助实验 | 同日问卷变量与 stress 高相关，不符合 pure wearable prediction 设定 |",
        "| 时间特征 | 建议加入 semester week、day-of-week、lag/rolling features | EDA 显示学期后期压力明显升高 |",
        "",
        "## 9. 可直接写进报告的简短总结",
        "",
        "The EDA shows that the processed dataset contains 1,394 rows from 32 students, corresponding to 1,384 unique student-day pairs because 10 student-days appear twice. These duplicates mostly reflect multiple questionnaire responses on the same day and should be resolved before modelling. The stress-label distribution is moderately imbalanced, while the main wearable-derived modelling features contain no missing values in the processed table. Raw modality coverage, however, varies substantially before preprocessing. Individual wearable features show weak monotonic correlations with self-reported stress, suggesting that prediction should rely on multivariate models rather than single-feature rules. Temporal analysis reveals a clear increase in mean stress toward the end of semester, motivating the inclusion of semester-week and lagged temporal features.",
        "",
    ]

    (OUT_DIR / "EDA_detailed_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = read_data()
    raw_file_audit, raw_target_audit, modality_summary = save_raw_data_audit(df)
    dataset_summary = save_dataset_summary(df)
    duplicate_summary = save_duplicate_check(df)
    thresholds = save_label_threshold_check(df)
    missingness = save_missingness(df)
    save_feature_summary(df)
    outliers = save_outlier_summary(df)
    save_feature_by_label(df)
    correlations = save_correlations(df)
    plot_stress_histogram(df)
    plot_label_distribution(df)
    weekly = plot_temporal_trend(df)
    plot_feature_means_by_label(df)
    plot_student_day_counts(df)
    plot_feature_histograms(df)
    plot_raw_modality_coverage(modality_summary)
    weekday_weekend = plot_weekday_weekend(df)
    write_report(df, dataset_summary, thresholds, missingness, correlations, weekly)
    write_detailed_chinese_report(
        df,
        duplicate_summary,
        thresholds,
        missingness,
        raw_target_audit,
        modality_summary,
        outliers,
        correlations,
        weekly,
        weekday_weekend,
    )


if __name__ == "__main__":
    main()
