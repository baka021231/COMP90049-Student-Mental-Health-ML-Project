from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE_OUT_DIR = ROOT / "modeling_outputs" / "strict_pipeline"
DATA_PATH = BASE_OUT_DIR / "03_model_data" / "strict_model_data.csv"
RAW_COVERAGE_PATH = (
    BASE_OUT_DIR / "00_raw_audit" / "raw_target_coverage_by_student_file.csv"
)
OUT_DIR = BASE_OUT_DIR / "eda"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

LABEL_ORDER = ["Low", "Medium", "High"]
WEARABLE_FEATURES = [
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
}


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    data["stress_label"] = pd.Categorical(
        data["stress_label"], categories=LABEL_ORDER, ordered=True
    )
    return data


def svg_text(
    x: float,
    y: float,
    text: object,
    size: int = 12,
    anchor: str = "middle",
    weight: str = "400",
    fill: str = "#111827",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape(str(text))}</text>'
    )


def write_svg(path: Path, parts: list[str]) -> None:
    path.write_text("\n".join(parts), encoding="utf-8")


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    display = data.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    headers = [str(column) for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in display.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_bar(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    ylabel: str,
    color: str,
    ymax: float | None = None,
    value_format: str = "{:.0f}",
) -> None:
    width, height = 900, 520
    left, right, top, bottom = 88, 36, 58, 104
    plot_w = width - left - right
    plot_h = height - top - bottom
    ymax = ymax or max(values) * 1.18
    gap = 22
    bar_w = (plot_w - gap * (len(values) - 1)) / len(values)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 32, title, 20, weight="700"),
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>',
        svg_text(24, top + plot_h / 2, ylabel, 13),
    ]
    for tick in range(6):
        value = ymax * tick / 5
        y = top + plot_h - value / ymax * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(svg_text(left - 12, y + 4, f"{value:.2f}", 11, "end", fill="#4b5563"))

    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * (bar_w + gap)
        bar_h = value / ymax * plot_h if ymax else 0
        y = top + plot_h - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>'
        )
        label_lines = str(label).replace("_", " ").split(" ")
        if len(label_lines) > 2:
            label_lines = [" ".join(label_lines[:2]), " ".join(label_lines[2:])]
        for line_index, line in enumerate(label_lines[:2]):
            parts.append(
                svg_text(
                    x + bar_w / 2,
                    height - bottom + 24 + line_index * 15,
                    line,
                    11,
                    fill="#374151",
                )
            )
        parts.append(
            svg_text(
                x + bar_w / 2,
                y - 8,
                value_format.format(value),
                12,
                weight="700",
            )
        )
    parts.append("</svg>")
    write_svg(path, parts)


def plot_student_days(data: pd.DataFrame) -> None:
    counts = data.groupby("student_id", observed=True).size().sort_values()
    width, height = 980, 560
    left, right, top, bottom = 70, 35, 58, 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_count = int(counts.max())
    bar_w = plot_w / len(counts)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 32, "Strict Dataset Student-Day Counts", 20, weight="700"),
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>',
        svg_text(24, top + plot_h / 2, "Student-days", 13),
        svg_text(width / 2, height - 24, "Student ID", 13),
    ]
    for tick in range(0, max_count + 1, 20):
        y = top + plot_h - tick / max_count * plot_h
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(svg_text(left - 12, y + 4, tick, 11, "end", fill="#4b5563"))
    for index, (student, value) in enumerate(counts.items()):
        x = left + index * bar_w
        bar_h = value / max_count * plot_h
        y = top + plot_h - bar_h
        color = "#2563eb" if value >= 30 else "#d97706"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" height="{bar_h:.1f}" fill="{color}"/>'
        )
        if index % 2 == 0:
            parts.append(svg_text(x + bar_w / 2, height - bottom + 18, student, 9))
    parts.append(svg_text(width - 175, 72, "Orange: fewer than 30 days", 12, "start"))
    parts.append("</svg>")
    write_svg(FIG_DIR / "strict_student_day_counts.svg", parts)


def plot_weekly_stress(data: pd.DataFrame, weekly: pd.DataFrame) -> None:
    width, height = 940, 540
    left, right, top, bottom = 78, 34, 58, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = int(weekly["semester_week"].min())
    x_max = int(weekly["semester_week"].max())
    y_min = 0.0
    y_max = max(100.0, float((weekly["mean_stress"] + weekly["sd_stress"]).max()))

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def y_scale(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    mean_points = [
        (x_scale(row.semester_week), y_scale(row.mean_stress))
        for row in weekly.itertuples(index=False)
    ]
    upper_points = [
        (
            x_scale(row.semester_week),
            y_scale(min(y_max, row.mean_stress + row.sd_stress)),
        )
        for row in weekly.itertuples(index=False)
    ]
    lower_points = [
        (
            x_scale(row.semester_week),
            y_scale(max(y_min, row.mean_stress - row.sd_stress)),
        )
        for row in weekly.itertuples(index=False)
    ]
    band = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper_points + list(reversed(lower_points)))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in mean_points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 32, "Strict Dataset Weekly Mean Stress Trend", 20, weight="700"),
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9ca3af"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9ca3af"/>',
        svg_text(width / 2, height - 24, "Semester week", 13),
        svg_text(24, top + plot_h / 2, "Stress score", 13),
    ]
    for value in range(0, 101, 20):
        y = y_scale(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        parts.append(svg_text(left - 12, y + 4, value, 11, "end", fill="#4b5563"))
    for week in weekly["semester_week"]:
        if int(week) % 2 == 1:
            x = x_scale(float(week))
            parts.append(svg_text(x, height - bottom + 24, int(week), 10, fill="#4b5563"))
    parts.extend(
        [
            f'<polygon points="{band}" fill="#93c5fd" opacity="0.35"/>',
            f'<polyline points="{line}" fill="none" stroke="#2563eb" stroke-width="3"/>',
        ]
    )
    for x, y in mean_points:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#2563eb"/>')
    parts.append("</svg>")
    write_svg(FIG_DIR / "strict_weekly_stress_trend.svg", parts)


def plot_feature_means_by_label(data: pd.DataFrame) -> None:
    selected = [
        "sleep_score",
        "deep_sleep_minutes",
        "total_steps",
        "sedentary_minutes",
        "avg_rmssd",
        "avg_oxygen",
        "std_oxygen",
    ]
    means = data.groupby("stress_label", observed=True)[selected].mean()
    z_means = (means - data[selected].mean()) / data[selected].std(ddof=0)
    z_means.to_csv(TABLE_DIR / "strict_standardised_feature_means_by_label.csv")

    width, height = 980, 560
    left, right, top, bottom = 245, 170, 58, 46
    row_h = (height - top - bottom) / len(selected)
    x0 = left + (width - left - right) / 2
    scale = 130
    colors = {"Low": "#2563eb", "Medium": "#d97706", "High": "#059669"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 32, "Strict Dataset Standardised Feature Means by Stress Label", 19, weight="700"),
        f'<line x1="{x0}" y1="{top - 10}" x2="{x0}" y2="{height - bottom + 10}" stroke="#9ca3af"/>',
    ]
    for index, feature in enumerate(selected):
        y = top + index * row_h + row_h / 2
        parts.append(svg_text(left - 12, y + 4, feature, 12, "end"))
        for label in LABEL_ORDER:
            value = float(z_means.loc[label, feature])
            x = x0 + value * scale
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{colors[label]}"/>')
    legend_x = width - right + 35
    for index, label in enumerate(LABEL_ORDER):
        y = top + 24 + index * 24
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{colors[label]}"/>')
        parts.append(svg_text(legend_x + 14, y + 4, label, 12, "start"))
    parts.append("</svg>")
    write_svg(FIG_DIR / "strict_feature_means_by_stress_label.svg", parts)


def build_tables(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    label_counts = data["stress_label"].value_counts(sort=False).reindex(LABEL_ORDER)
    label_distribution = pd.DataFrame(
        {
            "stress_label": label_counts.index,
            "count": label_counts.values,
            "percent": (label_counts.values / len(data) * 100).round(2),
        }
    )
    label_distribution.to_csv(TABLE_DIR / "strict_label_distribution.csv", index=False)

    per_student = (
        data.groupby("student_id", observed=True)
        .agg(
            student_days=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            low=("stress_label", lambda values: int((values == "Low").sum())),
            medium=("stress_label", lambda values: int((values == "Medium").sum())),
            high=("stress_label", lambda values: int((values == "High").sum())),
        )
        .reset_index()
        .sort_values("student_days")
    )
    per_student.to_csv(TABLE_DIR / "strict_student_day_counts.csv", index=False)

    split_summary = (
        data.groupby("split", observed=True)
        .agg(
            students=("student_id", "nunique"),
            rows=("student_id", "size"),
            low=("stress_label", lambda values: int((values == "Low").sum())),
            medium=("stress_label", lambda values: int((values == "Medium").sum())),
            high=("stress_label", lambda values: int((values == "High").sum())),
        )
        .reset_index()
    )
    split_summary.to_csv(TABLE_DIR / "strict_split_summary.csv", index=False)

    missing_rows = []
    for feature in WEARABLE_FEATURES:
        missing = int(data[feature].isna().sum())
        missing_rows.append(
            {
                "feature": feature,
                "missing_rows": missing,
                "missing_percent": round(missing / len(data) * 100, 2),
            }
        )
    missing_by_feature = pd.DataFrame(missing_rows).sort_values(
        "missing_percent", ascending=False
    )
    missing_by_feature.to_csv(TABLE_DIR / "strict_missingness_by_feature.csv", index=False)

    missing_group_rows = []
    for group, features in FEATURE_GROUPS.items():
        missing_cells = int(data[features].isna().sum().sum())
        total_cells = len(data) * len(features)
        complete_rows = int(data[features].notna().all(axis=1).sum())
        missing_group_rows.append(
            {
                "feature_group": group,
                "n_features": len(features),
                "missing_cells": missing_cells,
                "total_cells": total_cells,
                "missing_percent": round(missing_cells / total_cells * 100, 2),
                "complete_rows": complete_rows,
                "complete_percent": round(complete_rows / len(data) * 100, 2),
            }
        )
    missing_by_group = pd.DataFrame(missing_group_rows)
    missing_by_group.to_csv(TABLE_DIR / "strict_missingness_by_feature_group.csv", index=False)

    threshold_check = (
        data.groupby("stress_label", observed=True)["stress"]
        .agg(["count", "min", "max", "mean", "median", "std"])
        .reset_index()
        .round(3)
    )
    threshold_check.to_csv(TABLE_DIR / "strict_stress_label_threshold_check.csv", index=False)

    feature_summary = data[WEARABLE_FEATURES].describe().T.round(3)
    feature_summary.to_csv(TABLE_DIR / "strict_feature_descriptive_statistics.csv")

    feature_by_label = (
        data.groupby("stress_label", observed=True)[WEARABLE_FEATURES]
        .agg(["mean", "median", "std"])
        .round(3)
    )
    feature_by_label.columns = [f"{feature}_{stat}" for feature, stat in feature_by_label.columns]
    feature_by_label.reset_index().to_csv(
        TABLE_DIR / "strict_feature_summary_by_stress_label.csv", index=False
    )

    numeric_for_corr = data[["stress"] + WEARABLE_FEATURES].copy()
    correlations = (
        numeric_for_corr.corr(method="spearman", min_periods=20)["stress"]
        .drop(labels=["stress"], errors="ignore")
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", "stress": "spearman_rho_with_stress"})
    )
    correlations.to_csv(TABLE_DIR / "strict_spearman_correlation_with_stress.csv", index=False)

    weekly = (
        data.groupby("semester_week", as_index=False)
        .agg(
            n=("stress", "size"),
            mean_stress=("stress", "mean"),
            sd_stress=("stress", "std"),
            week_start=("date", "min"),
            week_end=("date", "max"),
        )
        .round({"mean_stress": 3, "sd_stress": 3})
    )
    weekly.to_csv(TABLE_DIR / "strict_weekly_stress_summary.csv", index=False)

    weekday_weekend = (
        data.assign(day_type=lambda frame: frame["is_weekend"].map({0: "weekday", 1: "weekend"}))
        .groupby("day_type", as_index=False)
        .agg(n=("stress", "size"), mean_stress=("stress", "mean"), sd_stress=("stress", "std"))
        .round({"mean_stress": 3, "sd_stress": 3})
    )
    weekday_weekend.to_csv(TABLE_DIR / "strict_weekday_weekend_stress.csv", index=False)

    raw_coverage_summary = pd.DataFrame()
    if RAW_COVERAGE_PATH.exists():
        coverage = pd.read_csv(RAW_COVERAGE_PATH)
        coverage_percent_column = (
            "coverage_percent"
            if "coverage_percent" in coverage.columns
            else "target_coverage_percent"
        )
        raw_coverage_summary = (
            coverage.groupby("file", as_index=False)
            .agg(
                students=("student_id", "nunique"),
                total_target_days=("target_days", "sum"),
                covered_target_days=("covered_target_days", "sum"),
                mean_student_coverage_percent=(coverage_percent_column, "mean"),
            )
            .round({"mean_student_coverage_percent": 2})
        )
        raw_coverage_summary["overall_target_day_coverage_percent"] = (
            raw_coverage_summary["covered_target_days"]
            / raw_coverage_summary["total_target_days"]
            * 100
        ).round(2)
        raw_coverage_summary.to_csv(
            TABLE_DIR / "strict_raw_modality_coverage_summary.csv", index=False
        )

    return {
        "label_distribution": label_distribution,
        "per_student": per_student,
        "split_summary": split_summary,
        "missing_by_feature": missing_by_feature,
        "missing_by_group": missing_by_group,
        "threshold_check": threshold_check,
        "correlations": correlations,
        "weekly": weekly,
        "weekday_weekend": weekday_weekend,
        "raw_coverage_summary": raw_coverage_summary,
    }


def build_figures(data: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    labels = tables["label_distribution"]["stress_label"].astype(str).tolist()
    counts = tables["label_distribution"]["count"].astype(float).tolist()
    plot_bar(
        FIG_DIR / "strict_label_distribution.svg",
        "Strict Dataset Stress Label Distribution",
        labels,
        counts,
        "Student-day count",
        "#2563eb",
        value_format="{:.0f}",
    )

    missing = tables["missing_by_feature"].head(12)
    plot_bar(
        FIG_DIR / "strict_feature_missingness.svg",
        "Strict Dataset Wearable Missingness by Feature",
        missing["feature"].tolist(),
        missing["missing_percent"].astype(float).tolist(),
        "Missing percent",
        "#d97706",
        ymax=60,
        value_format="{:.1f}%",
    )

    correlations = tables["correlations"].copy()
    correlations["abs_rho"] = correlations["spearman_rho_with_stress"].abs()
    top_corr = correlations.sort_values("abs_rho", ascending=False).head(8)
    plot_bar(
        FIG_DIR / "strict_top_spearman_correlations.svg",
        "Top Wearable Spearman Correlations With Stress",
        top_corr["feature"].tolist(),
        top_corr["spearman_rho_with_stress"].abs().astype(float).tolist(),
        "Absolute Spearman rho",
        "#059669",
        ymax=0.18,
        value_format="{:.3f}",
    )

    plot_student_days(data)
    plot_weekly_stress(data, tables["weekly"])
    plot_feature_means_by_label(data)


def write_summary(data: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    per_student = tables["per_student"]
    weekly = tables["weekly"]
    peak_week = weekly.loc[weekly["mean_stress"].idxmax()]
    low_week = weekly.loc[weekly["mean_stress"].idxmin()]
    top_corr = tables["correlations"].copy()
    top_corr["abs_rho"] = top_corr["spearman_rho_with_stress"].abs()
    top_corr = top_corr.sort_values("abs_rho", ascending=False).head(5)[
        ["feature", "spearman_rho_with_stress"]
    ]

    lines = [
        "# Strict Pipeline EDA Summary",
        "",
        "This EDA is generated only from `modeling_outputs/strict_pipeline/03_model_data/strict_model_data.csv` and is aligned with the final strict modelling pipeline.",
        "",
        "## Dataset Scale",
        "",
        f"- Student-day observations: {len(data)}",
        f"- Students: {data['student_id'].nunique()}",
        f"- Unique student-day pairs: {data[['student_id', 'date']].drop_duplicates().shape[0]}",
        f"- Date range: {data['date'].min().date()} to {data['date'].max().date()}",
        f"- Mean student-days per student: {per_student['student_days'].mean():.2f}",
        f"- Median student-days per student: {per_student['student_days'].median():.1f}",
        f"- Min/max student-days per student: {int(per_student['student_days'].min())} / {int(per_student['student_days'].max())}",
        "",
        "## Label Distribution",
        "",
        dataframe_to_markdown(tables["label_distribution"]),
        "",
        "The strict target bins are Low = 0-17, Medium = 18-38, and High = 39-100.",
        "",
        "## Train/Test Split",
        "",
        dataframe_to_markdown(tables["split_summary"]),
        "",
        "## Missingness",
        "",
        dataframe_to_markdown(tables["missing_by_group"]),
        "",
        "Feature-level missingness:",
        "",
        dataframe_to_markdown(tables["missing_by_feature"]),
        "",
        "## Stress Label Threshold Check",
        "",
        dataframe_to_markdown(tables["threshold_check"]),
        "",
        "## Strongest Wearable Correlations With Stress",
        "",
        dataframe_to_markdown(top_corr),
        "",
        "These correlations are weak in absolute size. This supports using multivariate models and reporting modest performance expectations.",
        "",
        "## Weekly Stress Trend",
        "",
        f"- Lowest mean stress: week {int(low_week['semester_week'])}, M = {float(low_week['mean_stress']):.2f}, SD = {float(low_week['sd_stress']):.2f}.",
        f"- Highest mean stress: week {int(peak_week['semester_week'])}, M = {float(peak_week['mean_stress']):.2f}, SD = {float(peak_week['sd_stress']):.2f}.",
        "",
        "## Generated Tables",
        "",
    ]
    for path in sorted(TABLE_DIR.glob("*.csv")):
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(["", "## Generated Figures", ""])
    for path in sorted(FIG_DIR.glob("*.svg")):
        lines.append(f"- `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Figure Previews",
            "",
            "![Strict label distribution](figures/strict_label_distribution.svg)",
            "",
            "![Strict feature missingness](figures/strict_feature_missingness.svg)",
            "",
            "![Strict student-day counts](figures/strict_student_day_counts.svg)",
            "",
            "![Strict feature means by stress label](figures/strict_feature_means_by_stress_label.svg)",
            "",
            "![Strict weekly stress trend](figures/strict_weekly_stress_trend.svg)",
            "",
            "![Strict top Spearman correlations](figures/strict_top_spearman_correlations.svg)",
            "",
        ]
    )
    (OUT_DIR / "strict_eda_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    data = load_data()
    tables = build_tables(data)
    build_figures(data, tables)
    write_summary(data, tables)
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
