# Model Data Cleaning Report

## Purpose

This module prepares a reproducible modelling table from `final_student_day_table_v01_processed.csv`.
The processed CSV is useful for modelling because its main wearable features have already been
completed and standardised, but it still needs a final modelling-specific cleaning layer.

## Inputs

- `final_student_day_table_v01_processed.csv`: primary modelling source.
- `final_student_day_table_v01.csv`: raw merged student-day table, used for context and auditing only.

## Cleaning Principles

1. Keep source CSV files unchanged.
2. Resolve duplicate `student_id + date` rows before any split or modelling.
3. Recreate `stress_label` from the numeric `stress` score so the label rule is explicit.
4. Exclude leakage-prone columns from model feature sets.
5. Create all RQ-specific feature sets from the same cleaned table.
6. Use subject-aware train/test splits so the same student cannot appear in both sets.

## Planned Outputs

- `clean_model_data.csv`: cleaned student-day modelling table.
- `feature_sets.json`: feature columns for RQ1, RQ2, and RQ3.
- `split_assignments.csv`: train/test assignment by student.
- `cleaning_audit.json`: row counts, duplicate handling, class counts, and split summary.

## Current Status

Scaffold created. The next step is to implement the cleaning script and generate the planned outputs.
