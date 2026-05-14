from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modeling_outputs"
DATA_PATH = OUT_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "feature_sets.json"
RESULTS_PATH = OUT_DIR / "rq1_results.csv"
SUMMARY_PATH = OUT_DIR / "rq1_summary.md"

TARGET_COLUMN = "stress_label"
SPLIT_COLUMN = "split"
RQ1_FEATURE_SET_NAME = "rq1_all_wearable"
LABEL_ORDER = ["Low", "Medium", "High"]
RANDOM_SEED = 49


def load_feature_names() -> list[str]:
    feature_sets = json.loads(FEATURE_SETS_PATH.read_text(encoding="utf-8"))
    if RQ1_FEATURE_SET_NAME not in feature_sets:
        raise KeyError(f"Missing feature set: {RQ1_FEATURE_SET_NAME}")
    return feature_sets[RQ1_FEATURE_SET_NAME]


def load_modeling_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required_columns = {TARGET_COLUMN, SPLIT_COLUMN}
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in modeling data: {missing}")
    return data


def validate_features(data: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing RQ1 feature columns: {missing}")

    feature_missing_counts = data[features].isna().sum()
    missing_features = feature_missing_counts[feature_missing_counts > 0]
    if not missing_features.empty:
        raise ValueError(
            "RQ1 wearable features contain missing values:\n"
            + missing_features.to_string()
        )


def make_train_test_data(
    data: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    test = data[data[SPLIT_COLUMN] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("Both train and test splits must contain rows.")

    X_train = train[features]
    y_train = train[TARGET_COLUMN]
    X_test = test[features]
    y_test = test[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def build_models() -> dict[str, object]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "svm": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "mlp": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64,),
                        alpha=0.001,
                        max_iter=3000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def fit_models(
    models: dict[str, object], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, object]:
    fitted_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
    return fitted_models


def confusion_matrix_path(model_name: str) -> Path:
    return OUT_DIR / f"rq1_confusion_matrix_{model_name}.csv"


def evaluate_models(
    fitted_models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    confusion_matrices = {}
    for model_name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        _, _, per_class_f1, per_class_support = precision_recall_fscore_support(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
        )

        rows.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "low_f1": per_class_f1[0],
                "medium_f1": per_class_f1[1],
                "high_f1": per_class_f1[2],
                "low_support": int(per_class_support[0]),
                "medium_support": int(per_class_support[1]),
                "high_support": int(per_class_support[2]),
            }
        )

        matrix = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
        confusion_matrices[model_name] = pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in LABEL_ORDER],
            columns=[f"predicted_{label}" for label in LABEL_ORDER],
        )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return results, confusion_matrices


def save_results(results: pd.DataFrame, confusion_matrices: dict[str, pd.DataFrame]) -> None:
    results.to_csv(RESULTS_PATH, index=False)
    for model_name, matrix in confusion_matrices.items():
        matrix.to_csv(confusion_matrix_path(model_name))


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    display = data.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(format_metric)
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


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
    OUT_DIR.mkdir(exist_ok=True)
    data = load_modeling_data()
    features = load_feature_names()
    validate_features(data, features)
    X_train, y_train, X_test, y_test = make_train_test_data(data, features)
    models = build_models()
    fitted_models = fit_models(models, X_train, y_train)
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
