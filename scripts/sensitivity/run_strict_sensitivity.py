from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from modeling_utils import LABEL_ORDER  # noqa: E402
from modeling_utils import build_strict_model_specs  # noqa: E402
from modeling_utils import dataframe_to_markdown  # noqa: E402
from modeling_utils import evaluate_models  # noqa: E402
from modeling_utils import load_feature_sets  # noqa: E402
from modeling_utils import tune_models_with_group_cv  # noqa: E402


BASE_OUT_DIR = ROOT / "modeling_outputs" / "strict_pipeline"
DATA_DIR = BASE_OUT_DIR / "03_model_data"
OUT_DIR = BASE_OUT_DIR / "07_sensitivity"

DATA_PATH = DATA_DIR / "strict_model_data.csv"
FEATURE_SETS_PATH = DATA_DIR / "strict_feature_sets.json"

BOOTSTRAP_PATH = OUT_DIR / "student_bootstrap_ci.csv"
BOOTSTRAP_DRAWS_PATH = OUT_DIR / "student_bootstrap_draws.csv"
GROUP_CV_PATH = OUT_DIR / "train_group_cv_best_model_sensitivity.csv"
DEVIATION_RESULTS_PATH = OUT_DIR / "within_person_deviation_results.csv"
DEVIATION_TUNING_PATH = OUT_DIR / "within_person_deviation_tuning.csv"
SUMMARY_PATH = OUT_DIR / "strict_sensitivity_summary.md"

RANDOM_SEED = 49
N_BOOTSTRAP = 2000
TARGET_COLUMN = "stress_label"
GROUP_COLUMN = "student_id"
SPLIT_COLUMN = "split"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    data["student_id"] = data["student_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["student_id", "date"]).reset_index(drop=True)


def make_fixed_estimators() -> dict[str, Pipeline]:
    return {
        "rq1_all_wearable_mlp": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
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
        "rq2_hrv_spo2_random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "rq3_rolling7_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=2,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def add_temporal_features(data: pd.DataFrame, wearable_features: list[str]) -> pd.DataFrame:
    out = data.sort_values(["student_id", "date"]).reset_index(drop=True).copy()
    for feature in wearable_features:
        out[f"{feature}_rolling7_mean"] = out.groupby("student_id")[feature].transform(
            lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
        )
    return out


def add_within_person_deviation_features(
    data: pd.DataFrame, wearable_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    out = data.sort_values(["student_id", "date"]).reset_index(drop=True).copy()
    deviation_features = []
    for feature in wearable_features:
        expanding_mean = out.groupby("student_id")[feature].transform(
            lambda values: values.shift(1).expanding(min_periods=1).mean()
        )
        deviation_feature = f"{feature}_deviation_from_past_mean"
        out[deviation_feature] = out[feature] - expanding_mean
        deviation_features.append(deviation_feature)
    return out, deviation_features


def fit_final_models(
    data: pd.DataFrame,
    feature_sets: dict,
) -> dict[str, dict[str, object]]:
    wearable_features = feature_sets["rq1_all_wearable"]
    temporal = add_temporal_features(data, wearable_features)
    estimators = make_fixed_estimators()

    specs = {
        "rq1_all_wearable_mlp": {
            "estimator": estimators["rq1_all_wearable_mlp"],
            "features": wearable_features,
            "data": data,
            "description": "RQ1 best: all wearable + MLP",
        },
        "rq2_hrv_spo2_random_forest": {
            "estimator": estimators["rq2_hrv_spo2_random_forest"],
            "features": feature_sets["rq2_feature_groups"]["hrv_spo2_only"],
            "data": data,
            "description": "RQ2 best: HRV + SpO2 + Random Forest",
        },
        "rq3_rolling7_gradient_boosting": {
            "estimator": estimators["rq3_rolling7_gradient_boosting"],
            "features": wearable_features
            + [f"{feature}_rolling7_mean" for feature in wearable_features],
            "data": temporal,
            "description": "RQ3 best: rolling7 wearable + Gradient Boosting",
        },
    }

    fitted = {}
    for name, spec in specs.items():
        spec_data = spec["data"]
        train = spec_data[spec_data[SPLIT_COLUMN] == "train"].copy()
        test = spec_data[spec_data[SPLIT_COLUMN] == "test"].copy()
        X_train = train[spec["features"]]
        y_train = train[TARGET_COLUMN]
        X_test = test[spec["features"]]
        y_test = test[TARGET_COLUMN]
        estimator = clone(spec["estimator"])
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        fitted[name] = {
            **spec,
            "estimator": estimator,
            "test": test,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": pd.Series(y_pred, index=test.index, name="prediction"),
            "point_macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        }
    return fitted


def student_bootstrap_ci(fitted_specs: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = pd.Series(range(N_BOOTSTRAP)).sample(frac=1, random_state=RANDOM_SEED).to_numpy()
    summary_rows = []
    draw_rows = []
    for spec_name, spec in fitted_specs.items():
        test = spec["test"].copy()
        test["prediction"] = spec["y_pred"]
        test_students = sorted(test[GROUP_COLUMN].unique())
        for draw_idx, seed_value in enumerate(rng):
            sampled_students = pd.Series(test_students).sample(
                n=len(test_students),
                replace=True,
                random_state=int(seed_value) + RANDOM_SEED,
            )
            sampled_frames = []
            for replicate_id, student_id in enumerate(sampled_students):
                student_rows = test[test[GROUP_COLUMN] == student_id].copy()
                student_rows["bootstrap_replicate"] = replicate_id
                sampled_frames.append(student_rows)
            sampled = pd.concat(sampled_frames, ignore_index=True)
            macro_f1 = f1_score(
                sampled[TARGET_COLUMN],
                sampled["prediction"],
                average="macro",
                zero_division=0,
            )
            draw_rows.append(
                {
                    "analysis": spec_name,
                    "draw": draw_idx,
                    "macro_f1": macro_f1,
                }
            )

        draws = pd.DataFrame([row for row in draw_rows if row["analysis"] == spec_name])
        summary_rows.append(
            {
                "analysis": spec_name,
                "description": spec["description"],
                "point_macro_f1": spec["point_macro_f1"],
                "bootstrap_mean_macro_f1": draws["macro_f1"].mean(),
                "ci_lower_2_5": draws["macro_f1"].quantile(0.025),
                "ci_upper_97_5": draws["macro_f1"].quantile(0.975),
                "n_test_students": len(test_students),
                "n_bootstrap_draws": N_BOOTSTRAP,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(draw_rows)


def fixed_group_cv_sensitivity(data: pd.DataFrame, feature_sets: dict) -> pd.DataFrame:
    train = data[data[SPLIT_COLUMN] == "train"].copy()
    wearable_features = feature_sets["rq1_all_wearable"]
    configs = {
        "rq1_all_wearable_mlp": {
            "estimator": make_fixed_estimators()["rq1_all_wearable_mlp"],
            "features": wearable_features,
            "data": train,
            "description": "RQ1 best fixed estimator on train GroupKFold",
        },
        "rq2_hrv_spo2_random_forest": {
            "estimator": make_fixed_estimators()["rq2_hrv_spo2_random_forest"],
            "features": feature_sets["rq2_feature_groups"]["hrv_spo2_only"],
            "data": train,
            "description": "RQ2 best fixed estimator on train GroupKFold",
        },
    }

    temporal = add_temporal_features(data, wearable_features)
    configs["rq3_rolling7_gradient_boosting"] = {
        "estimator": make_fixed_estimators()["rq3_rolling7_gradient_boosting"],
        "features": wearable_features + [f"{feature}_rolling7_mean" for feature in wearable_features],
        "data": temporal[temporal[SPLIT_COLUMN] == "train"].copy(),
        "description": "RQ3 best fixed estimator on train GroupKFold",
    }

    rows = []
    logo = LeaveOneGroupOut()
    for config_name, config in configs.items():
        config_data = config["data"]
        X = config_data[config["features"]]
        y = config_data[TARGET_COLUMN]
        groups = config_data[GROUP_COLUMN]
        fold_scores = []
        fold_rows = []
        for fold, (train_idx, val_idx) in enumerate(logo.split(X, y, groups=groups), start=1):
            estimator = clone(config["estimator"])
            estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = estimator.predict(X.iloc[val_idx])
            macro_f1 = f1_score(y.iloc[val_idx], y_pred, average="macro", zero_division=0)
            fold_scores.append(macro_f1)
            fold_rows.append(
                {
                    "analysis": config_name,
                    "fold": fold,
                    "held_out_student": groups.iloc[val_idx].iloc[0],
                    "n_validation_rows": len(val_idx),
                    "macro_f1": macro_f1,
                }
            )
        fold_frame = pd.DataFrame(fold_rows)
        fold_frame.to_csv(OUT_DIR / f"{config_name}_leave_one_train_student_out_folds.csv", index=False)
        rows.append(
            {
                "analysis": config_name,
                "description": config["description"],
                "n_train_students": groups.nunique(),
                "mean_macro_f1": pd.Series(fold_scores).mean(),
                "sd_macro_f1": pd.Series(fold_scores).std(ddof=1),
                "min_macro_f1": pd.Series(fold_scores).min(),
                "max_macro_f1": pd.Series(fold_scores).max(),
            }
        )
    return pd.DataFrame(rows)


def within_person_deviation_experiment(data: pd.DataFrame, feature_sets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    wearable_features = feature_sets["rq1_all_wearable"]
    deviation_data, deviation_features = add_within_person_deviation_features(data, wearable_features)
    conditions = {
        "no_temporal": wearable_features,
        "deviation_only": wearable_features + deviation_features,
    }
    result_frames = []
    tuning_frames = []
    for condition, features in conditions.items():
        train = deviation_data[deviation_data[SPLIT_COLUMN] == "train"].copy()
        test = deviation_data[deviation_data[SPLIT_COLUMN] == "test"].copy()
        X_train = train[features]
        y_train = train[TARGET_COLUMN]
        groups_train = train[GROUP_COLUMN]
        X_test = test[features]
        y_test = test[TARGET_COLUMN]
        fitted_models, tuning = tune_models_with_group_cv(
            build_strict_model_specs(),
            X_train,
            y_train,
            groups_train,
        )
        results, _ = evaluate_models(fitted_models, X_test, y_test)
        results.insert(0, "condition", condition)
        results.insert(1, "n_features", len(features))
        tuning.insert(0, "condition", condition)
        tuning.insert(1, "n_features", len(features))
        result_frames.append(results)
        tuning_frames.append(tuning)
    return pd.concat(result_frames, ignore_index=True), pd.concat(tuning_frames, ignore_index=True)


def write_summary(
    bootstrap: pd.DataFrame,
    group_cv: pd.DataFrame,
    deviation_results: pd.DataFrame,
    deviation_tuning: pd.DataFrame,
) -> None:
    deviation_best = (
        deviation_results.sort_values(["condition", "macro_f1"], ascending=[True, False])
        .groupby("condition", as_index=False, observed=True)
        .head(1)
        .reset_index(drop=True)
    )
    lines = [
        "# Strict Sensitivity Analysis Summary",
        "",
        "This folder contains supplementary robustness analyses for the strict modelling pipeline.",
        "",
        "## 1. Student-level bootstrap confidence intervals",
        "",
        dataframe_to_markdown(bootstrap),
        "",
        "Interpretation: the confidence intervals are based on resampling the 9 held-out test students with replacement. They quantify how sensitive the reported test macro-F1 is to the small test-student set.",
        "",
        "## 2. Leave-one-train-student-out sensitivity",
        "",
        dataframe_to_markdown(group_cv),
        "",
        "Interpretation: these values evaluate fixed best-model configurations inside the training students by leaving one training student out at a time. They are not a replacement for the final held-out test set; they show variation across students.",
        "",
        "## 3. Leakage-safe within-person deviation feature experiment",
        "",
        "Deviation features are constructed as current wearable value minus the same student's expanding past mean. The expanding mean is shifted by one day, so it uses only prior observations.",
        "",
        dataframe_to_markdown(
            deviation_best[
                [
                    "condition",
                    "n_features",
                    "model",
                    "accuracy",
                    "macro_f1",
                    "low_f1",
                    "medium_f1",
                    "high_f1",
                ]
            ]
        ),
        "",
        "Interpretation: this experiment tests whether a simple personalised-baseline feature representation helps unseen-student prediction under the same strict split.",
        "",
        "## Output files",
        "",
        f"- `{BOOTSTRAP_PATH.relative_to(ROOT)}`",
        f"- `{BOOTSTRAP_DRAWS_PATH.relative_to(ROOT)}`",
        f"- `{GROUP_CV_PATH.relative_to(ROOT)}`",
        f"- `{DEVIATION_RESULTS_PATH.relative_to(ROOT)}`",
        f"- `{DEVIATION_TUNING_PATH.relative_to(ROOT)}`",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    feature_sets = load_feature_sets(FEATURE_SETS_PATH)
    fitted_specs = fit_final_models(data, feature_sets)
    bootstrap, bootstrap_draws = student_bootstrap_ci(fitted_specs)
    group_cv = fixed_group_cv_sensitivity(data, feature_sets)
    deviation_results, deviation_tuning = within_person_deviation_experiment(data, feature_sets)

    bootstrap.to_csv(BOOTSTRAP_PATH, index=False)
    bootstrap_draws.to_csv(BOOTSTRAP_DRAWS_PATH, index=False)
    group_cv.to_csv(GROUP_CV_PATH, index=False)
    deviation_results.to_csv(DEVIATION_RESULTS_PATH, index=False)
    deviation_tuning.to_csv(DEVIATION_TUNING_PATH, index=False)
    write_summary(bootstrap, group_cv, deviation_results, deviation_tuning)

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
