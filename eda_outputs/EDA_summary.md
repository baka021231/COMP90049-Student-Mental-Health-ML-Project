# EDA Summary

## Dataset Scale
- Students: 32
- Student-day observations: 1394
- Date range: 2025-02-19 to 2025-06-30
- Mean days per student: 43.56

## Stress Label Distribution
- Low: 361 (25.9%)
- Medium: 496 (35.6%)
- High: 537 (38.5%)

## Label Threshold Check
| stress_label | count | min | max | mean | median | std |
| --- | --- | --- | --- | --- | --- | --- |
| Low | 361 | 0 | 17 | 9.873 | 11.0 | 5.246 |
| Medium | 496 | 18 | 38 | 27.349 | 28.0 | 5.752 |
| High | 537 | 39 | 100 | 62.482 | 57.0 | 18.258 |

The processed table uses thresholds equivalent to Low = 0-17, Medium = 18-38, and High = 39-100. This differs from the draft report text, which currently describes 0-33 / 34-66 / 67-100 bins.

## Missingness
| index | feature_group | features | missing_cells | total_cells | missing_percent |
| --- | --- | --- | --- | --- | --- |
| 0 | Sleep | sleep_score, deep_sleep_minutes | 0 | 2788 | 0.0 |
| 1 | Activity | total_steps, sedentary_minutes, lightly_active_minutes, moderately_active_minutes, very_active_minutes | 0 | 6970 | 0.0 |
| 2 | HRV | avg_rmssd, avg_low_frequency, avg_high_frequency | 0 | 4182 | 0.0 |
| 3 | SpO2 | avg_oxygen, std_oxygen | 0 | 2788 | 0.0 |
| 4 | Fitbit stress score | STRESS_SCORE, CALCULATION_FAILED | 306 | 2788 | 10.98 |

All modelling feature columns in the processed table have already been imputed or otherwise completed. The only remaining missing fields are Fitbit's own STRESS_SCORE and CALCULATION_FAILED, each missing in 153 rows (10.98%).

## Strongest Spearman Correlations With Stress
| index | stress |
| --- | --- |
| anxiety | 0.64 |
| STRESS_SCORE | 0.079 |
| std_oxygen | 0.069 |
| avg_oxygen | 0.061 |
| moderately_active_minutes | 0.058 |
| sedentary_minutes | 0.05 |
| total_steps | 0.047 |
| avg_high_frequency | 0.044 |

## Most Negative Spearman Correlations With Stress
| index | stress |
| --- | --- |
| total_steps | 0.047 |
| avg_high_frequency | 0.044 |
| lightly_active_minutes | 0.033 |
| avg_rmssd | 0.03 |
| very_active_minutes | 0.006 |
| sleep_score | -0.007 |
| deep_sleep_minutes | -0.017 |
| avg_low_frequency | -0.171 |

## Temporal Trend
- Peak mean stress: week 19, mean = 51.63, n = 19
- Lowest mean stress: week 3, mean = 27.31, n = 59

Generated figures:
- Stress label distribution

  ![Stress label distribution](figures/figure_1_stress_label_distribution.svg)

- Continuous stress distribution

  ![Continuous stress distribution](figures/stress_continuous_distribution.svg)

- Temporal stress trend

  ![Temporal stress trend](figures/figure_4_temporal_stress_trend.svg)

- Feature means by stress label

  ![Feature means by stress label](figures/feature_means_by_stress_label.svg)

- Student-day counts

  ![Student-day counts](figures/student_day_counts.svg)

- Feature distributions

  ![Feature distributions](figures/feature_distributions.svg)

- Raw modality coverage

  ![Raw modality coverage](figures/raw_modality_coverage.svg)

- Weekday/weekend stress

  ![Weekday/weekend stress](figures/weekday_weekend_stress.svg)
