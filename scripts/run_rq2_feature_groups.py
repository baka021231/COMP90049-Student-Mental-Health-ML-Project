from __future__ import annotations

from pathlib import Path

import pandas as pd
from modeling_utils import LABEL_ORDER
from modeling_utils import build_models
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


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_modeling_data(DATA_PATH)
    feature_groups = load_feature_groups()
    results = run_feature_group_experiments(data, feature_groups)
    best_by_group = best_results_by_feature_group(results)
    save_results(results, best_by_group)

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
    print("Best model by feature group:")
    for _, row in best_by_group.iterrows():
        print(
            f"- {row['feature_group']}: {row['model']} "
            f"(macro-F1={row['macro_f1']:.3f}, accuracy={row['accuracy']:.3f})"
        )


if __name__ == "__main__":
    main()
