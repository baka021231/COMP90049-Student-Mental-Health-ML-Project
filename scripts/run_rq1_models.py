from __future__ import annotations

from pathlib import Path

import pandas as pd
from modeling_utils import LABEL_ORDER
from modeling_utils import build_models
from modeling_utils import dataframe_to_markdown
from modeling_utils import evaluate_models
from modeling_utils import fit_models
from modeling_utils import format_metric
from modeling_utils import load_feature_sets
from modeling_utils import load_modeling_data
from modeling_utils import make_train_test_data
from modeling_utils import validate_features


ROOT = Path(__file__).resolve().parents[1]
BASE_OUT_DIR = ROOT / "modeling_outputs" / "legacy_pipeline"
DATA_DIR = BASE_OUT_DIR / "00_model_data"
OUT_DIR = BASE_OUT_DIR / "01_rq1"
DATA_PATH = DATA_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = DATA_DIR / "feature_sets.json"
RESULTS_PATH = OUT_DIR / "rq1_results.csv"
SUMMARY_PATH = OUT_DIR / "rq1_summary.md"

RQ1_FEATURE_SET_NAME = "rq1_all_wearable"


def load_feature_names() -> list[str]:
    feature_sets = load_feature_sets(FEATURE_SETS_PATH)
    if RQ1_FEATURE_SET_NAME not in feature_sets:
        raise KeyError(f"Missing feature set: {RQ1_FEATURE_SET_NAME}")
    return feature_sets[RQ1_FEATURE_SET_NAME]


def confusion_matrix_path(model_name: str) -> Path:
    return OUT_DIR / f"rq1_confusion_matrix_{model_name}.csv"


def save_results(results: pd.DataFrame, confusion_matrices: dict[str, pd.DataFrame]) -> None:
    results.to_csv(RESULTS_PATH, index=False)
    for model_name, matrix in confusion_matrices.items():
        matrix.to_csv(confusion_matrix_path(model_name))


def write_summary(
    results: pd.DataFrame,
    features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    best = results.iloc[0]
    baseline = results[results["model"] == "majority_baseline"].iloc[0]
    best_model = str(best["model"])
    best_matrix = pd.read_csv(confusion_matrix_path(best_model), index_col=0)
    hardest_class = (
        best[["low_f1", "medium_f1", "high_f1"]]
        .astype(float)
        .rename(index={"low_f1": "Low", "medium_f1": "Medium", "high_f1": "High"})
        .idxmin()
    )

    lines = [
        "# RQ1 Modeling Summary",
        "",
        "## Research Question",
        "",
        "RQ1 asks whether wearable-derived sleep, activity, HRV, and SpO2 features can predict student stress level.",
        "",
        "## Data",
        "",
        f"- Feature set: `{RQ1_FEATURE_SET_NAME}`",
        f"- Number of wearable features: {len(features)}",
        f"- Train rows: {len(X_train)}",
        f"- Test rows: {len(X_test)}",
        f"- Train label counts: {y_train.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}",
        f"- Test label counts: {y_test.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}",
        "",
        "## Models",
        "",
        "- Majority baseline",
        "- Logistic Regression",
        "- SVM",
        "- Random Forest",
        "- MLP",
        "",
        "## Main Results",
        "",
        dataframe_to_markdown(results),
        "",
        "## Key Findings",
        "",
        f"- Best model by macro-F1: `{best_model}`.",
        f"- Best macro-F1: {format_metric(float(best['macro_f1']))}.",
        f"- Majority baseline macro-F1: {format_metric(float(baseline['macro_f1']))}.",
        f"- Majority baseline accuracy: {format_metric(float(baseline['accuracy']))}.",
        f"- The hardest class for the best model is `{hardest_class}` by per-class F1.",
        "- Accuracy alone is not sufficient here because the majority baseline has competitive accuracy but weak macro-F1.",
        "",
        f"## Confusion Matrix for Best Model: `{best_model}`",
        "",
        dataframe_to_markdown(best_matrix.reset_index().rename(columns={"index": "actual"})),
        "",
        "## Report Note",
        "",
        "The initial RQ1 result suggests that wearable-only features provide limited but measurable predictive signal. The best model improves macro-F1 over the majority baseline, but the overall performance remains modest, so the report should discuss the difficulty of predicting self-reported stress from passive wearable data alone.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_modeling_data(DATA_PATH)
    features = load_feature_names()
    validate_features(data, features, "RQ1")
    X_train, y_train, X_test, y_test = make_train_test_data(data, features)
    fitted_models = fit_models(build_models(), X_train, y_train)
    results, confusion_matrices = evaluate_models(fitted_models, X_test, y_test)
    save_results(results, confusion_matrices)
    write_summary(results, features, X_train, X_test, y_train, y_test)

    print("Loaded RQ1 modeling data.")
    print(f"Input data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Feature set: {RQ1_FEATURE_SET_NAME}")
    print(f"Features: {len(features)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Train labels: {y_train.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}")
    print(f"Test labels: {y_test.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}")
    print(f"Fitted models: {', '.join(fitted_models)}")
    print(f"Wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
