from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "final_student_day_table_v01_processed.csv"
OUT_DIR = ROOT / "modeling_outputs"
CLEAN_PATH = OUT_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "feature_sets.json"
SPLIT_PATH = OUT_DIR / "split_assignments.csv"
AUDIT_PATH = OUT_DIR / "cleaning_audit.json"

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
TEMPORAL_FEATURES = [
    "semester_week",
    "day_of_week",
    "is_weekend",
    "lag_1_sleep_score",
    "lag_1_total_steps",
    "lag_1_avg_rmssd",
    "rolling_3_sleep_score",
    "rolling_3_total_steps",
    "rolling_3_avg_rmssd",
    "rolling_7_sleep_score",
    "rolling_7_total_steps",
    "rolling_7_avg_rmssd",
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
    if stress <= 17:
        return "Low"
    if stress <= 38:
        return "Medium"
    return "High"


def read_source() -> pd.DataFrame:
    df = pd.read_csv(SOURCE_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df["student_id"] = df["student_id"].astype(str)
    return df


def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    resolved = (
        df.groupby(["student_id", "date"], as_index=False, observed=True)[numeric_cols]
        .mean()
        .sort_values(["student_id", "date"])
        .reset_index(drop=True)
    )
    resolved["stress_label"] = resolved["stress"].map(stress_to_label)
    return resolved


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["student_id", "date"]).copy()
    first_date = out["date"].min()
    out["semester_week"] = ((out["date"] - first_date).dt.days // 7 + 1).astype(int)
    out["day_of_week"] = out["date"].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    lag_sources = ["sleep_score", "total_steps", "avg_rmssd"]
    grouped = out.groupby("student_id", observed=True)
    for feature in lag_sources:
        out[f"lag_1_{feature}"] = grouped[feature].shift(1)
        out[f"rolling_3_{feature}"] = grouped[feature].transform(
            lambda values: values.shift(1).rolling(3, min_periods=1).mean()
        )
        out[f"rolling_7_{feature}"] = grouped[feature].transform(
            lambda values: values.shift(1).rolling(7, min_periods=1).mean()
        )

    return out


def make_feature_sets() -> dict[str, list[str] | dict[str, list[str]]]:
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
            "semester_week": WEARABLE_FEATURES + ["semester_week"],
            "calendar": WEARABLE_FEATURES + ["semester_week", "day_of_week", "is_weekend"],
            "lag_1": WEARABLE_FEATURES
            + ["semester_week", "day_of_week", "is_weekend"]
            + ["lag_1_sleep_score", "lag_1_total_steps", "lag_1_avg_rmssd"],
            "rolling_3": WEARABLE_FEATURES
            + ["semester_week", "day_of_week", "is_weekend"]
            + [
                "lag_1_sleep_score",
                "lag_1_total_steps",
                "lag_1_avg_rmssd",
                "rolling_3_sleep_score",
                "rolling_3_total_steps",
                "rolling_3_avg_rmssd",
            ],
            "rolling_7": WEARABLE_FEATURES + TEMPORAL_FEATURES,
        },
    }


def make_subject_split(df: pd.DataFrame) -> pd.DataFrame:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    row_numbers = pd.Series(range(len(df)), index=df.index)
    train_idx, test_idx = next(splitter.split(df, df["stress_label"], groups=df["student_id"]))
    train_rows = set(row_numbers.iloc[train_idx])
    test_rows = set(row_numbers.iloc[test_idx])

    assignments = []
    for student_id, group in df.groupby("student_id", observed=True):
        group_rows = set(row_numbers.loc[group.index])
        split = "train" if group_rows <= train_rows else "test" if group_rows <= test_rows else "mixed"
        assignments.append(
            {
                "student_id": student_id,
                "split": split,
                "n_rows": len(group),
                "low_count": int((group["stress_label"] == "Low").sum()),
                "medium_count": int((group["stress_label"] == "Medium").sum()),
                "high_count": int((group["stress_label"] == "High").sum()),
            }
        )
    return pd.DataFrame(assignments).sort_values(["split", "student_id"])


def build_audit(source: pd.DataFrame, clean: pd.DataFrame, split: pd.DataFrame) -> dict:
    duplicate_rows = int(source.duplicated(["student_id", "date"]).sum())
    label_changes = int(
        (
            source.assign(rebuilt_label=source["stress"].map(stress_to_label))["stress_label"]
            != source.assign(rebuilt_label=source["stress"].map(stress_to_label))["rebuilt_label"]
        ).sum()
    )
    clean_counts = clean["stress_label"].value_counts().to_dict()
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
    temporal_missing = clean[TEMPORAL_FEATURES].isna().sum().to_dict()

    return {
        "source_path": SOURCE_PATH.name,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "source_rows": int(len(source)),
        "source_students": int(source["student_id"].nunique()),
        "source_duplicate_student_day_rows": duplicate_rows,
        "source_label_values_changed_by_rebuild": label_changes,
        "clean_rows": int(len(clean)),
        "clean_students": int(clean["student_id"].nunique()),
        "clean_label_counts": {key: int(value) for key, value in clean_counts.items()},
        "split_summary": split_summary,
        "temporal_missing_counts": {key: int(value) for key, value in temporal_missing.items()},
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    source = read_source()
    clean = resolve_duplicates(source)
    clean = add_temporal_features(clean)
    split = make_subject_split(clean)
    clean["split"] = clean["student_id"].map(split.set_index("student_id")["split"])
    feature_sets = make_feature_sets()
    audit = build_audit(source, clean, split)

    clean.to_csv(CLEAN_PATH, index=False)
    split.to_csv(SPLIT_PATH, index=False)
    FEATURE_SETS_PATH.write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Wrote {CLEAN_PATH.relative_to(ROOT)}")
    print(f"Wrote {SPLIT_PATH.relative_to(ROOT)}")
    print(f"Wrote {FEATURE_SETS_PATH.relative_to(ROOT)}")
    print(f"Wrote {AUDIT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
