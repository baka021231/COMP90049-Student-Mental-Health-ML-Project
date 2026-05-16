# Strict Sensitivity Analysis Summary

This folder contains supplementary robustness analyses for the strict modelling pipeline.

## 1. Student-level bootstrap confidence intervals

| analysis | description | point_macro_f1 | bootstrap_mean_macro_f1 | ci_lower_2_5 | ci_upper_97_5 | n_test_students | n_bootstrap_draws |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rq1_all_wearable_mlp | RQ1 best: all wearable + MLP | 0.357 | 0.345 | 0.274 | 0.415 | 9 | 2000 |
| rq2_hrv_spo2_random_forest | RQ2 best: HRV + SpO2 + Random Forest | 0.393 | 0.381 | 0.301 | 0.454 | 9 | 2000 |
| rq3_rolling7_gradient_boosting | RQ3 best: rolling7 wearable + Gradient Boosting | 0.392 | 0.378 | 0.289 | 0.469 | 9 | 2000 |

Interpretation: the confidence intervals are based on resampling the 9 held-out test students with replacement. They quantify how sensitive the reported test macro-F1 is to the small test-student set.

## 2. Leave-one-train-student-out sensitivity

| analysis | description | n_train_students | mean_macro_f1 | sd_macro_f1 | min_macro_f1 | max_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| rq1_all_wearable_mlp | RQ1 best fixed estimator on train GroupKFold | 26 | 0.282 | 0.104 | 0.000 | 0.443 |
| rq2_hrv_spo2_random_forest | RQ2 best fixed estimator on train GroupKFold | 26 | 0.275 | 0.105 | 0.000 | 0.423 |
| rq3_rolling7_gradient_boosting | RQ3 best fixed estimator on train GroupKFold | 26 | 0.261 | 0.111 | 0.000 | 0.417 |

Interpretation: these values evaluate fixed best-model configurations inside the training students by leaving one training student out at a time. They are not a replacement for the final held-out test set; they show variation across students.

## 3. Leakage-safe within-person deviation feature experiment

Deviation features are constructed as current wearable value minus the same student's expanding past mean. The expanding mean is shifted by one day, so it uses only prior observations.

| condition | n_features | model | accuracy | macro_f1 | low_f1 | medium_f1 | high_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deviation_only | 24 | gradient_boosting | 0.416 | 0.407 | 0.488 | 0.344 | 0.390 |
| no_temporal | 12 | mlp | 0.374 | 0.357 | 0.480 | 0.261 | 0.330 |

Interpretation: this experiment tests whether a simple personalised-baseline feature representation helps unseen-student prediction under the same strict split.

## Output files

- `modeling_outputs\strict_pipeline\07_sensitivity\student_bootstrap_ci.csv`
- `modeling_outputs\strict_pipeline\07_sensitivity\student_bootstrap_draws.csv`
- `modeling_outputs\strict_pipeline\07_sensitivity\train_group_cv_best_model_sensitivity.csv`
- `modeling_outputs\strict_pipeline\07_sensitivity\within_person_deviation_results.csv`
- `modeling_outputs\strict_pipeline\07_sensitivity\within_person_deviation_tuning.csv`
