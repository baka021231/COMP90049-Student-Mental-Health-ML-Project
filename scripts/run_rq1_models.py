from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "modeling_outputs"
DATA_PATH = OUT_DIR / "clean_model_data.csv"
FEATURE_SETS_PATH = OUT_DIR / "feature_sets.json"
RESULTS_PATH = OUT_DIR / "rq1_results.csv"

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
                        max_iter=1000,
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


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    data = load_modeling_data()
    features = load_feature_names()
    validate_features(data, features)
    X_train, y_train, X_test, y_test = make_train_test_data(data, features)
    models = build_models()
    fitted_models = fit_models(models, X_train, y_train)

    print("Loaded RQ1 modeling data.")
    print(f"Input data: {DATA_PATH.relative_to(ROOT)}")
    print(f"Feature set: {RQ1_FEATURE_SET_NAME}")
    print(f"Features: {len(features)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Train labels: {y_train.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}")
    print(f"Test labels: {y_test.value_counts().reindex(LABEL_ORDER, fill_value=0).to_dict()}")
    print(f"Fitted models: {', '.join(fitted_models)}")


if __name__ == "__main__":
    main()
