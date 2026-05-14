from __future__ import annotations

from pathlib import Path

import pandas as pd
from modeling_utils import LABEL_ORDER
from modeling_utils import build_models
from modeling_utils import dataframe_to_markdown
from modeling_utils import evaluate_models
from modeling_utils import fit_models
from modeling_utils import load_feature_sets
from modeling_utils import load_modeling_data
from modeling_utils import make_train_test_data
from modeling_utils import validate_features


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modeling_outputs"
DATA_PATH = OUT_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "feature_sets.json"
RESULTS_PATH = OUT_DIR / "rq2_feature_group_results.csv"
BEST_RESULTS_PATH = OUT_DIR / "rq2_best_by_feature_group.csv"
SUMMARY_PATH = OUT_DIR / "rq2_summary.md"

RQ2_FEATURE_GROUPS_NAME = "rq2_feature_groups"


def load_feature_groups() -> dict[str, list[str]]:
    feature_sets = load_feature_sets(FEATURE_SETS_PATH)
    if RQ2_FEATURE_GROUPS_NAME not in feature_sets:
        raise KeyError(f"Missing feature group set: {RQ2_FEATURE_GROUPS_NAME}")
    return feature_sets[RQ2_FEATURE_GROUPS_NAME]


def run_feature_group_experiments(
    data: pd.DataFrame, feature_groups: dict[str, list[str]]
) -> pd.DataFrame:
    result_frames = []
    for feature_group, features in feature_groups.items():
        validate_features(data, features, f"RQ2 {feature_group}")
        X_train, y_train, X_test, y_test = make_train_test_data(data, features)
        fitted_models = fit_models(build_models(), X_train, y_train)
        results, _ = evaluate_models(fitted_models, X_test, y_test)
        results.insert(0, "feature_group", feature_group)
        results.insert(1, "n_features", len(features))
        result_frames.append(results)

    return pd.concat(result_frames, ignore_index=True)


def best_results_by_feature_group(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.sort_values(["feature_group", "macro_f1"], ascending=[True, False])
        .groupby("feature_group", as_index=False, observed=True)
        .head(1)
        .reset_index(drop=True)
    )


def save_results(results: pd.DataFrame, best_by_group: pd.DataFrame) -> None:
    results.to_csv(RESULTS_PATH, index=False)
    best_by_group.to_csv(BEST_RESULTS_PATH, index=False)


def write_summary(
    results: pd.DataFrame,
    best_by_group: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> None:
    overall_best = results.sort_values("macro_f1", ascending=False).iloc[0]
    strongest_group = str(overall_best["feature_group"])
    weakest_group = str(best_by_group.sort_values("macro_f1").iloc[0]["feature_group"])
    all_wearable = best_by_group[best_by_group["feature_group"] == "all_wearable"].iloc[0]
    hrv_spo2 = best_by_group[best_by_group["feature_group"] == "hrv_spo2_only"].iloc[0]
    macro_f1_gap = float(all_wearable["macro_f1"]) - float(hrv_spo2["macro_f1"])

    compact_best = best_by_group[
        ["feature_group", "n_features", "model", "accuracy", "macro_f1", "low_f1", "medium_f1", "high_f1"]
    ].sort_values("macro_f1", ascending=False)
    compact_results = results[
        ["feature_group", "n_features", "model", "accuracy", "macro_f1", "low_f1", "medium_f1", "high_f1"]
    ].sort_values(["feature_group", "macro_f1"], ascending=[True, False])

    lines = [
        "# RQ2 Feature Group Summary",
        "",
        "## Research Question",
        "",
        "RQ2 asks which wearable feature groups contribute most to student stress prediction.",
        "",
        "## Experimental Design",
        "",
        "- The same subject-aware train/test split is used as RQ1.",
        "- Each feature group is evaluated with the full RQ1 model set.",
        f"- Feature groups tested: {', '.join(feature_groups)}.",
        "- Main comparison metric: macro-F1.",
        "",
        "## Feature Groups",
        "",
    ]

    for group_name, features in feature_groups.items():
        lines.append(f"- `{group_name}`: {len(features)} features")

    lines.extend(
        [
            "",
            "## Best Model by Feature Group",
            "",
            dataframe_to_markdown(compact_best),
            "",
            "## Full Result Table",
            "",
            dataframe_to_markdown(compact_results),
            "",
            "## Key Findings",
            "",
            f"- Strongest feature group by macro-F1: `{strongest_group}` using `{overall_best['model']}`.",
            f"- Best macro-F1 overall: {float(overall_best['macro_f1']):.3f}.",
            f"- Weakest best-performing feature group: `{weakest_group}`.",
            f"- `hrv_spo2_only` is close to `all_wearable`: macro-F1 gap = {macro_f1_gap:.3f}.",
            "- This suggests physiological features may carry substantial stress-related signal, while adding all wearable features gives only a small improvement in this run.",
            "",
            "## Report Note",
            "",
            "RQ2 should be presented as an ablation study. The important point is not only which score is highest, but how much each modality contributes. The current result supports a cautious interpretation: wearable modalities contain some useful signal, but the differences are modest and should be discussed alongside the limited number of students.",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_modeling_data(DATA_PATH)
    feature_groups = load_feature_groups()
    results = run_feature_group_experiments(data, feature_groups)
    best_by_group = best_results_by_feature_group(results)
    save_results(results, best_by_group)
    write_summary(results, best_by_group, feature_groups)

    print("Loaded RQ2 feature group data.")
    print(f"Input data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Feature sets: {FEATURE_SETS_PATH.relative_to(ROOT)}")
    print(f"Feature group key: {RQ2_FEATURE_GROUPS_NAME}")
    print(f"Feature groups: {', '.join(feature_groups)}")
    print(f"Models per group: {len(build_models())}")
    print(f"Experiment rows: {len(results)}")
    print(f"Train/test labels use order: {LABEL_ORDER}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {BEST_RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    print("Best model by feature group:")
    for _, row in best_by_group.iterrows():
        print(
            f"- {row['feature_group']}: {row['model']} "
            f"(macro-F1={row['macro_f1']:.3f}, accuracy={row['accuracy']:.3f})"
        )


if __name__ == "__main__":
    main()
