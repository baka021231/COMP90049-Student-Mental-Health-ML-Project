from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TARGET_COLUMN = "stress_label"
SPLIT_COLUMN = "split"
LABEL_ORDER = ["Low", "Medium", "High"]
RANDOM_SEED = 49


def load_feature_sets(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_modeling_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {TARGET_COLUMN, SPLIT_COLUMN}
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in modeling data: {missing}")
    return data


def validate_features(data: pd.DataFrame, features: list[str], context: str) -> None:
    missing = sorted(set(features) - set(data.columns))
    if missing:
        raise ValueError(f"Missing {context} feature columns: {missing}")

    feature_missing_counts = data[features].isna().sum()
    missing_features = feature_missing_counts[feature_missing_counts > 0]
    if not missing_features.empty:
        raise ValueError(
            f"{context} features contain missing values:\n"
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


def build_strict_model_specs() -> dict[str, tuple[object, dict[str, list]]]:
    return {
        "majority_baseline": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", DummyClassifier(strategy="most_frequent")),
                ]
            ),
            {},
        ),
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "svm": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        SVC(
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"],
            },
        ),
        "knn": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier()),
                ]
            ),
            {"classifier__n_neighbors": [3, 5, 9]},
        ),
        "random_forest": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "classifier__n_estimators": [100, 300],
                "classifier__max_depth": [5, 10, None],
            },
        ),
        "gradient_boosting": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("classifier", GradientBoostingClassifier(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
        "mlp": (
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        MLPClassifier(
                            max_iter=3000,
                            random_state=RANDOM_SEED,
                        ),
                    ),
                ]
            ),
            {
                "classifier__hidden_layer_sizes": [(32,), (64,), (32, 16)],
                "classifier__alpha": [0.0001, 0.001],
            },
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


def tune_models_with_group_cv(
    model_specs: dict[str, tuple[object, dict[str, list]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_splits: int = 5,
) -> tuple[dict[str, object], pd.DataFrame]:
    unique_groups = groups_train.nunique()
    if unique_groups < 2:
        raise ValueError("Group-aware tuning requires at least two training students.")

    cv = GroupKFold(n_splits=min(n_splits, unique_groups))
    fitted_models = {}
    tuning_rows = []
    for model_name, (estimator, param_grid) in model_specs.items():
        if param_grid:
            search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train, groups=groups_train)
            fitted_models[model_name] = search.best_estimator_
            tuning_rows.append(
                {
                    "model": model_name,
                    "cv_macro_f1": search.best_score_,
                    "best_params": json.dumps(search.best_params_, sort_keys=True),
                    "n_candidates": len(search.cv_results_["params"]),
                }
            )
        else:
            fitted = clone(estimator)
            fitted.fit(X_train, y_train)
            fitted_models[model_name] = fitted
            tuning_rows.append(
                {
                    "model": model_name,
                    "cv_macro_f1": pd.NA,
                    "best_params": "{}",
                    "n_candidates": 1,
                }
            )

    tuning_results = pd.DataFrame(tuning_rows)
    return fitted_models, tuning_results


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
