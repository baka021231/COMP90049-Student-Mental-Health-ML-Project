# RQ2 Validation Notes

## Commands Run

```bash
python -m py_compile scripts/modeling_utils.py scripts/run_rq2_feature_groups.py
python scripts/run_rq2_feature_groups.py
```

## Validation Results

- The shared modeling utilities compile successfully.
- The RQ2 script compiles successfully.
- The RQ2 script runs successfully from the project root.
- The script reads `modeling_outputs/clean_model_data.csv`.
- The script reads `rq2_feature_groups` from `modeling_outputs/feature_sets.json`.
- The script evaluates 4 feature groups.
- The script trains 5 models per feature group.
- The full result table contains 20 rows.
- The best-by-group table contains 4 rows.
- Re-running the script after committing generated outputs produced no Git diff.

## Main Checked Outputs

- `modeling_outputs/rq2_feature_group_results.csv`
- `modeling_outputs/rq2_best_by_feature_group.csv`
- `modeling_outputs/rq2_summary.md`

## Key Result Check

- Best overall feature group/model: `all_wearable` with `mlp`
- Best overall macro-F1: 0.409
- Best `hrv_spo2_only` macro-F1: 0.401
- Best `activity_only` macro-F1: 0.386
- Best `sleep_only` macro-F1: 0.354

## Interpretation for Report

RQ2 supports an ablation-style discussion. `all_wearable` gives the best result, but `hrv_spo2_only` is very close, suggesting that physiological signals may carry a large share of the available wearable-based stress signal. The differences are modest, so this should be framed cautiously rather than as a strong claim.
