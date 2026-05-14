# RQ1 Validation Notes

## Commands Run

```bash
python -m py_compile scripts/run_rq1_models.py
python scripts/run_rq1_models.py
```

## Validation Results

- The RQ1 script compiles successfully.
- The RQ1 script runs successfully from the project root.
- The script reads `modeling_outputs/clean_model_data.csv`.
- The script reads `rq1_all_wearable` from `modeling_outputs/feature_sets.json`.
- The script uses the existing `split` column.
- The script trains five models: majority baseline, Logistic Regression, SVM, Random Forest, and MLP.
- The script writes `modeling_outputs/rq1_results.csv`.
- The script writes one confusion matrix CSV per model.
- The script writes `modeling_outputs/rq1_summary.md`.
- Re-running the script after committing generated outputs produced no Git diff.

## Main Checked Outputs

- Best model by macro-F1: `mlp`
- Best macro-F1: 0.409
- Majority baseline macro-F1: 0.192
- Hardest class for the best model: `Medium`

## Interpretation for Report

The RQ1 results suggest that wearable-only features contain some predictive signal because the best real model improves macro-F1 over the majority baseline. However, overall performance is modest, so the report should avoid claiming strong stress prediction performance from wearable data alone.
