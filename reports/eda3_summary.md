# EDA Phase 3 Summary — Menstrual Physiology and Hormonal Signals

**Project:** CycleInsight — Menstrual Intelligence System  
**Dataset:** `datasets/cleaned_dataset.csv`  
**Analysis grain:** day-level records; repeated records from a participant are not independent.  
**Scope:** descriptive EDA only. No machine learning, feature engineering, predictive modeling, causal claims, or clinical diagnosis.

## Dataset Overview
### Overview

| metric | value |
| --- | --- |
| Rows (day-level records) | 2,937 |
| Columns | 72 |
| Participants (`id`) | 42 |
| Required columns present | 28 / 28 |
| Phase / phase.1 non-null disagreements | 0 |
| heavy_bleeding_flag_proxy values > 1 | 5 |

## 1. Data Validation
### Requested Columns, Dtypes, and Missingness

| index | present | dtype | missing | missing_pct | unique_non_null |
| --- | --- | --- | --- | --- | --- |
| phase | True | object | 0 | 0.0 | 4 |
| phase.1 | True | object | 1 | 0.034048348655090224 | 4 |
| flow intensity | True | object | 0 | 0.0 | 8 |
| flow_color | True | object | 0 | 0.0 | 9 |
| Cervical mucus code | True | object | 1 | 0.034048348655090224 | 5 |
| bleeding_days | True | int64 | 0 | 0.0 | 10 |
| inter_cycle_length | True | int64 | 0 | 0.0 | 8 |
| cycle_variability | True | float64 | 0 | 0.0 | 2937 |
| day_from_last_menses | True | int64 | 0 | 0.0 | 4 |
| pad_change_rate | True | float64 | 0 | 0.0 | 91 |
| heavy_bleeding_flag_proxy | True | float64 | 0 | 0.0 | 6 |
| anovulation_proxy | True | int64 | 0 | 0.0 | 2 |
| HMB_label | True | float64 | 0 | 0.0 | 91 |
| LH_result | True | object | 0 | 0.0 | 2 |
| lh | True | float64 | 0 | 0.0 | 236 |
| estrogen | True | float64 | 0 | 0.0 | 1726 |
| BBT | True | float64 | 0 | 0.0 | 2391 |
| lag1_BBT | True | float64 | 0 | 0.0 | 2370 |
| lag2_BBT | True | float64 | 0 | 0.0 | 2369 |
| BBT_mean_7d | True | float64 | 0 | 0.0 | 2895 |
| BBT_std_7d | True | float64 | 0 | 0.0 | 2805 |
| nightly_temperature | True | float64 | 0 | 0.0 | 2723 |
| temperature_samples | True | int64 | 0 | 0.0 | 525 |
| baseline_relative_sample_sum | True | float64 | 0 | 0.0 | 2610 |
| baseline_relative_sample_sum_of_squares | True | float64 | 0 | 0.0 | 2679 |
| baseline_relative_nightly_standard_deviation | True | float64 | 0 | 0.0 | 2679 |
| baseline_relative_sample_standard_deviation | True | float64 | 0 | 0.0 | 2679 |
| type | True | object | 0 | 0.0 | 1 |

### Observed Categorical Levels

| column | category | count | pct |
| --- | --- | --- | --- |
| phase | Luteal | 970 | 33.02689819543752 |
| phase | Follicular | 745 | 25.36601974804222 |
| phase | Fertility | 655 | 22.301668369084098 |
| phase | Menstrual | 567 | 19.30541368743616 |
| phase.1 | Luteal | 970 | 33.02689819543752 |
| phase.1 | Follicular | 745 | 25.36601974804222 |
| phase.1 | Fertility | 654 | 22.26762002042901 |
| phase.1 | Menstrual | 567 | 19.30541368743616 |
| phase.1 | <missing> | 1 | 0.03404834865509023 |
| flow intensity | Not at all | 2277 | 77.52808988764045 |
| flow intensity | Spotting / Very Light | 184 | 6.264896152536602 |
| flow intensity | Moderate | 131 | 4.4603336738168196 |
| flow intensity | Light | 122 | 4.153898535921008 |
| flow intensity | Somewhat Light | 84 | 2.860061287027579 |
| flow intensity | Somewhat Heavy | 60 | 2.0429009193054135 |
| flow intensity | Heavy | 55 | 1.8726591760299625 |
| flow intensity | Very Heavy | 24 | 0.8171603677221655 |
| flow_color | Not at all | 2193 | 74.66802860061287 |
| flow_color | Dark Brown / Dark Red | 289 | 9.839972761321077 |
| flow_color | Bright Red | 207 | 7.048008171603677 |
| flow_color | Yellow | 127 | 4.324140279196459 |
| flow_color | Other | 73 | 2.4855294518215865 |
| flow_color | Pink | 29 | 0.9874021109976167 |
| flow_color | Black | 7 | 0.23833844058563158 |
| flow_color | Orange | 7 | 0.23833844058563158 |
| flow_color | Grey | 5 | 0.17024174327545114 |
| Cervical mucus code | Dry | 970 | 33.02689819543752 |
| Cervical mucus code | Creamy | 744 | 25.33197139938713 |
| Cervical mucus code | Clear | 654 | 22.26762002042901 |
| Cervical mucus code | Bloody | 567 | 19.30541368743616 |
| Cervical mucus code |  Creamy | 1 | 0.03404834865509023 |
| Cervical mucus code | <missing> | 1 | 0.03404834865509023 |
| LH_result | Negative | 2283 | 77.732379979571 |
| LH_result | Positive | 654 | 22.26762002042901 |
| type | SKIN | 2937 | 100.0 |

### Configured Suspicious-Value Checks

| column | rule | count | examples |
| --- | --- | --- | --- |
| bleeding_days | expected exploratory range 0–7 days | 52 | 8, 8, 8, 8, 8, 8, 8, 8 |
| heavy_bleeding_flag_proxy | proxy outside binary 0–1 range | 5 | 2.16, 2.01, 2.58, 1.36, 2.67 |
| Cervical mucus code | unexpected or whitespace-padded category | 1 | ' Creamy' |

### Findings

The dataset contains 2,937 day-level rows across 42 participants. 28 of 28 requested columns are present. 3 configured value/category checks identified at least one issue; inspect the table above. These broad screening rules are data-quality prompts, not diagnostic criteria. These checks are exploratory and do not define clinical normality.

## 2. Menstrual Feature Analysis
### Categorical Frequencies and Percentages

| column | category | count | pct |
| --- | --- | --- | --- |
| phase | Luteal | 970 | 33.02689819543752 |
| phase | Follicular | 745 | 25.36601974804222 |
| phase | Fertility | 655 | 22.301668369084098 |
| phase | Menstrual | 567 | 19.30541368743616 |
| phase.1 | Luteal | 970 | 33.02689819543752 |
| phase.1 | Follicular | 745 | 25.36601974804222 |
| phase.1 | Fertility | 654 | 22.26762002042901 |
| phase.1 | Menstrual | 567 | 19.30541368743616 |
| phase.1 | <missing> | 1 | 0.03404834865509023 |
| flow intensity | Not at all | 2277 | 77.52808988764045 |
| flow intensity | Spotting / Very Light | 184 | 6.264896152536602 |
| flow intensity | Moderate | 131 | 4.4603336738168196 |
| flow intensity | Light | 122 | 4.153898535921008 |
| flow intensity | Somewhat Light | 84 | 2.860061287027579 |
| flow intensity | Somewhat Heavy | 60 | 2.0429009193054135 |
| flow intensity | Heavy | 55 | 1.8726591760299625 |
| flow intensity | Very Heavy | 24 | 0.8171603677221655 |
| flow_color | Not at all | 2193 | 74.66802860061287 |
| flow_color | Dark Brown / Dark Red | 289 | 9.839972761321077 |
| flow_color | Bright Red | 207 | 7.048008171603677 |
| flow_color | Yellow | 127 | 4.324140279196459 |
| flow_color | Other | 73 | 2.4855294518215865 |
| flow_color | Pink | 29 | 0.9874021109976167 |
| flow_color | Black | 7 | 0.23833844058563158 |
| flow_color | Orange | 7 | 0.23833844058563158 |
| flow_color | Grey | 5 | 0.17024174327545114 |
| Cervical mucus code | Dry | 970 | 33.02689819543752 |
| Cervical mucus code | Creamy | 744 | 25.33197139938713 |
| Cervical mucus code | Clear | 654 | 22.26762002042901 |
| Cervical mucus code | Bloody | 567 | 19.30541368743616 |
| Cervical mucus code |  Creamy | 1 | 0.03404834865509023 |
| Cervical mucus code | <missing> | 1 | 0.03404834865509023 |
| bleeding_days | 0 | 2369 | 80.66053796390875 |
| bleeding_days | 6 | 151 | 5.141300646918625 |
| bleeding_days | 5 | 133 | 4.528430371127 |
| bleeding_days | 4 | 86 | 2.9281579843377594 |
| bleeding_days | 7 | 78 | 2.6557711950970377 |
| bleeding_days | 8 | 40 | 1.361933946203609 |
| bleeding_days | 3 | 36 | 1.2257405515832482 |
| bleeding_days | 2 | 23 | 0.7831120190670753 |
| bleeding_days | 12 | 12 | 0.40858018386108275 |
| bleeding_days | 1 | 9 | 0.30643513789581206 |

### Numeric Summary Statistics

| column | count | missing | missing_pct | mean | median | std | min | q1 | q3 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bleeding_days | 2937 | 0 | 0.0 | 1.051 | 0.0 | 2.292 | 0.0 | 0.0 | 0.0 | 12.0 |
| inter_cycle_length | 2937 | 0 | 0.0 | 24.859 | 25.0 | 2.551 | 21.0 | 22.0 | 27.0 | 28.0 |
| cycle_variability | 2937 | 0 | 0.0 | 3.024 | 3.028 | 1.153 | 1.0 | 2.028 | 4.028 | 4.999 |
| day_from_last_menses | 2937 | 0 | 0.0 | 12.702 | 13.0 | 5.838 | 5.0 | 5.0 | 20.0 | 20.0 |
| pad_change_rate | 2937 | 0 | 0.0 | 0.555 | 0.56 | 0.261 | 0.1 | 0.33 | 0.78 | 1.0 |

### Numeric IQR Outlier Counts

| column | lower_fence | upper_fence | outlier_count | outlier_pct |
| --- | --- | --- | --- | --- |
| bleeding_days | 0.0 | 0.0 | 568 | 19.339 |
| inter_cycle_length | 14.5 | 34.5 | 0 | 0.0 |
| cycle_variability | -0.973 | 7.029 | 0 | 0.0 |
| day_from_last_menses | -17.5 | 42.5 | 0 | 0.0 |
| pad_change_rate | -0.345 | 1.455 | 0 | 0.0 |

### Findings

The most frequent phase label is Luteal (970 records; 33.0%). Among the numeric menstrual measures, bleeding_days has the most IQR-flagged records (568); review its distribution and the fences before interpreting extremes.

## 3. Flag Analysis
### Observed Flag and Score Encodings

| column | encoding_note | count | missing | unique_values | nonzero_count | nonzero_pct | min | median | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heavy_bleeding_flag_proxy | continuous/non-binary; no cutoff applied | 2937 | 0 | 6 | 5 | 0.17 | 0.0 | 0.0 | 2.67 |
| anovulation_proxy | binary | 2937 | 0 | 2 | 2283 | 77.732 | 0.0 | 1.0 | 1.0 |
| HMB_label | continuous/non-binary; no cutoff applied | 2937 | 0 | 91 | 2937 | 100.0 | 0.1 | 0.55 | 1.0 |

### Findings

An exploratory 70% majority-class screen flags anovulation_proxy (77.7% majority). Binary-field positive-class shares: anovulation_proxy=77.7%. Exact count/percentage frequencies are shown for all stored values. HMB_label is treated as a continuous score; no clinical threshold or prevalence class is inferred.

## 4. Hormonal and Primary Temperature Analysis
### Hormone Summary Statistics

| column | count | missing | missing_pct | mean | median | std | min | q1 | q3 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lh | 2937 | 0 | 0.0 | 5.2322 | 3.6 | 6.7459 | 0.0 | 2.4 | 5.4 | 97.0 |
| estrogen | 2937 | 0 | 0.0 | 137.1768 | 106.4 | 111.944 | 0.0 | 66.4 | 172.2 | 640.0 |
| BBT | 2937 | 0 | 0.0 | 36.538 | 36.5292 | 0.2285 | 36.0111 | 36.4113 | 36.7663 | 36.9895 |
| nightly_temperature | 2937 | 0 | 0.0 | 33.7628 | 33.878 | 0.9352 | 25.6993 | 33.2387 | 34.4027 | 36.2004 |

### Hormone IQR Outlier Counts

| column | lower_fence | upper_fence | outlier_count | outlier_pct |
| --- | --- | --- | --- | --- |
| lh | -2.1 | 9.9 | 248 | 8.444 |
| estrogen | -92.3 | 330.9 | 175 | 5.9585 |
| BBT | 35.8788 | 37.2987 | 0 | 0.0 |
| nightly_temperature | 31.4928 | 36.1486 | 34 | 1.1576 |

### Findings

Across the four primary signals, lh has the largest IQR-flagged count (248 records). The KDEs are smoothed views of the observed distributions; they do not establish clinical reference ranges or biological effects. Inspect the summary table for scale, spread, and missingness.

## 5. Temperature Feature Analysis
### Temperature Feature Summary Statistics

| column | count | missing | missing_pct | mean | median | std | min | q1 | q3 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag1_BBT | 2937 | 0 | 0.0 | 36.5735 | 36.5398 | 0.1912 | 36.1222 | 36.4418 | 36.7699 | 36.9895 |
| lag2_BBT | 2937 | 0 | 0.0 | 36.577 | 36.5442 | 0.1882 | 36.1222 | 36.4469 | 36.7706 | 36.9895 |
| BBT_mean_7d | 2937 | 0 | 0.0 | 36.6206 | 36.6053 | 0.1399 | 36.3669 | 36.4882 | 36.7711 | 36.9028 |
| BBT_std_7d | 2937 | 0 | 0.0 | 0.082 | 0.0665 | 0.0498 | 0.0 | 0.0462 | 0.111 | 0.3327 |
| temperature_samples | 2937 | 0 | 0.0 | 459.4086 | 465.0 | 112.7662 | 7.0 | 385.0 | 530.0 | 1097.0 |
| baseline_relative_sample_sum | 2937 | 0 | 0.0 | -24.6988 | -4.9497 | 320.9568 | -4436.3138 | -149.2376 | 131.4984 | 1677.2467 |
| baseline_relative_sample_sum_of_squares | 2937 | 0 | 0.0 | 860.4059 | 616.1638 | 1442.5782 | 10.2244 | 358.6054 | 938.2421 | 34309.648 |
| baseline_relative_nightly_standard_deviation | 2937 | 0 | 0.0 | 0.6144 | 0.5744 | 0.2818 | 0.1338 | 0.4628 | 0.6755 | 2.4063 |
| baseline_relative_sample_standard_deviation | 2937 | 0 | 0.0 | 1.2769 | 1.2554 | 0.3296 | 0.4421 | 1.0637 | 1.4361 | 3.0941 |

### Temperature Feature IQR Outlier Counts

| column | lower_fence | upper_fence | outlier_count | outlier_pct |
| --- | --- | --- | --- | --- |
| lag1_BBT | 35.9497 | 37.262 | 0 | 0.0 |
| lag2_BBT | 35.9614 | 37.2561 | 0 | 0.0 |
| BBT_mean_7d | 36.0638 | 37.1954 | 0 | 0.0 |
| BBT_std_7d | -0.051 | 0.2082 | 67 | 2.2812 |
| temperature_samples | 167.5 | 747.5 | 35 | 1.1917 |
| baseline_relative_sample_sum | -570.3416 | 552.6024 | 153 | 5.2094 |
| baseline_relative_sample_sum_of_squares | -510.8496 | 1807.6971 | 188 | 6.4011 |
| baseline_relative_nightly_standard_deviation | 0.1437 | 0.9946 | 196 | 6.6735 |
| baseline_relative_sample_standard_deviation | 0.5052 | 1.9947 | 69 | 2.3493 |

### Findings

`temperature_samples` ranges from 7 to 1097 (median 465); 1 records have 10 or fewer samples. `BBT_std_7d` is at or below 0.001 in 6 records. `nightly_temperature` ranges from 25.70 to 36.20; the dataset's `type` field is constant (SKIN). The scale/units and acquisition meaning should be confirmed before interpreting temperature values physiologically.

## 6. Biological Consistency Checks
### Mean LH, Estrogen, and BBT by Phase

| phase | lh | estrogen | BBT |
| --- | --- | --- | --- |
| Menstrual | 3.8252 | 96.6765 | 36.1973 |
| Follicular | 4.6227 | 106.6465 | 36.4484 |
| Fertility | 9.5502 | 183.9533 | 36.5494 |
| Luteal | 3.607 | 152.713 | 36.7983 |

### Phase-Stratified Count, Mean, Median, and Standard Deviation

| phase | lh_count | lh_mean | lh_median | lh_std | estrogen_count | estrogen_mean | estrogen_median | estrogen_std | BBT_count | BBT_mean | BBT_median | BBT_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Menstrual | 567 | 3.8252 | 3.3 | 2.2928 | 567 | 96.6765 | 79.0 | 70.9192 | 567 | 36.1973 | 36.1777 | 0.129 |
| Follicular | 745 | 4.6227 | 3.9 | 3.6234 | 745 | 106.6465 | 85.3 | 79.9835 | 745 | 36.4484 | 36.4504 | 0.0539 |
| Fertility | 655 | 9.5502 | 5.1 | 11.8463 | 655 | 183.9533 | 138.4 | 143.6471 | 655 | 36.5494 | 36.5499 | 0.0494 |
| Luteal | 970 | 3.607 | 3.0 | 3.6504 | 970 | 152.713 | 124.55 | 113.1195 | 970 | 36.7983 | 36.7971 | 0.0509 |

### Findings

lh phase means span 3.607–9.550 (highest in Fertility, lowest in Luteal; descriptive difference 5.943). estrogen phase means span 96.677–183.953 (highest in Fertility, lowest in Menstrual; descriptive difference 87.277). BBT phase means span 36.197–36.798 (highest in Luteal, lowest in Menstrual; descriptive difference 0.601). These are unadjusted record-level means; differences are descriptive, not causal or inferential evidence.

## 7. Visualization Export
### Exported Figures

| plot file | report link |
| --- | --- |
| eda3_menstrual_category_phase.png | [PNG](figures/eda3_menstrual_category_phase.png) |
| eda3_menstrual_category_phase_1.png | [PNG](figures/eda3_menstrual_category_phase_1.png) |
| eda3_menstrual_category_flow_intensity.png | [PNG](figures/eda3_menstrual_category_flow_intensity.png) |
| eda3_menstrual_category_flow_color.png | [PNG](figures/eda3_menstrual_category_flow_color.png) |
| eda3_menstrual_category_cervical_mucus_code.png | [PNG](figures/eda3_menstrual_category_cervical_mucus_code.png) |
| eda3_menstrual_category_bleeding_days.png | [PNG](figures/eda3_menstrual_category_bleeding_days.png) |
| eda3_menstrual_numeric_bleeding_days.png | [PNG](figures/eda3_menstrual_numeric_bleeding_days.png) |
| eda3_menstrual_numeric_inter_cycle_length.png | [PNG](figures/eda3_menstrual_numeric_inter_cycle_length.png) |
| eda3_menstrual_numeric_cycle_variability.png | [PNG](figures/eda3_menstrual_numeric_cycle_variability.png) |
| eda3_menstrual_numeric_day_from_last_menses.png | [PNG](figures/eda3_menstrual_numeric_day_from_last_menses.png) |
| eda3_menstrual_numeric_pad_change_rate.png | [PNG](figures/eda3_menstrual_numeric_pad_change_rate.png) |
| eda3_flag_values_heavy_bleeding_flag_proxy.png | [PNG](figures/eda3_flag_values_heavy_bleeding_flag_proxy.png) |
| eda3_flag_values_anovulation_proxy.png | [PNG](figures/eda3_flag_values_anovulation_proxy.png) |
| eda3_flag_values_HMB_label.png | [PNG](figures/eda3_flag_values_HMB_label.png) |
| eda3_hormone_lh.png | [PNG](figures/eda3_hormone_lh.png) |
| eda3_hormone_estrogen.png | [PNG](figures/eda3_hormone_estrogen.png) |
| eda3_hormone_bbt.png | [PNG](figures/eda3_hormone_bbt.png) |
| eda3_hormone_nightly_temperature.png | [PNG](figures/eda3_hormone_nightly_temperature.png) |
| eda3_temperature_feature_distributions.png | [PNG](figures/eda3_temperature_feature_distributions.png) |
| eda3_phase_lh_box_violin.png | [PNG](figures/eda3_phase_lh_box_violin.png) |
| eda3_phase_estrogen_box_violin.png | [PNG](figures/eda3_phase_estrogen_box_violin.png) |
| eda3_phase_BBT_box_violin.png | [PNG](figures/eda3_phase_BBT_box_violin.png) |

### Findings

Exported 22 descriptive plots to reports/figures; filenames identify each measure or comparison.

## Data Quality Observations
### Quality Notes

| area | observation |
| --- | --- |
| Validation | The dataset contains 2,937 day-level rows across 42 participants. 28 of 28 requested columns are present. 3 configured value/category checks identified at least one issue; inspect the table above. These broad screening rules are data-quality prompts, not diagnostic criteria. |
| Proxy encodings | An exploratory 70% majority-class screen flags anovulation_proxy (77.7% majority). Binary-field positive-class shares: anovulation_proxy=77.7%. Exact count/percentage frequencies are shown for all stored values. HMB_label is treated as a continuous score; no clinical threshold or prevalence class is inferred. |
| Temperature metadata / ranges | `temperature_samples` ranges from 7 to 1097 (median 465); 1 records have 10 or fewer samples. `BBT_std_7d` is at or below 0.001 in 6 records. `nightly_temperature` ranges from 25.70 to 36.20; the dataset's `type` field is constant (SKIN). The scale/units and acquisition meaning should be confirmed before interpreting temperature values physiologically. |
| Phase-label agreement | 0 records have nonmissing phase and phase.1 labels that disagree. |

## Recommendations for the Next EDA Phase
### Recommended Follow-up

| recommended next EDA step |
| --- |
| Confirm temperature units, sensor semantics, and the meaning of `type` before interpreting temperature values physiologically. |
| Document how `phase` and `phase.1` were generated; resolve missing or discrepant labels before phase-stratified comparisons. |
| Trace the construction and intended interpretation of `heavy_bleeding_flag_proxy` and `HMB_label`; do not apply cutoffs until specified. |
| Review records with sparse temperature sampling and IQR-flagged extremes against source-level acquisition context. |
| Describe within-participant and per-cycle data coverage and variability, accounting for repeated day-level records; keep the next phase descriptive. |

## Findings
The analyses describe observed distributions and phase-group differences only. They do not establish causation, diagnostic thresholds, or independent participant-level effects. Confirm field definitions and temperature units before physiological interpretation.

*Generated by `notebooks/eda_03_menstrual_hormonal.ipynb`.*
