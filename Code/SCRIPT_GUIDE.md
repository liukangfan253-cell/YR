# Yellow River dissolved oxygen workflow

This guide covers the 11 November 2026 analysis batch and the workflow checked on 14 November 2026. See the [submission README](../README.md) for the Python environment and [figure scripts](#figure-scripts) for the current plotting entry points.

Run scripts from the submission package root in dependency order. An arrow indicates which step runs next. A plus sign means that both inputs are required at that point. Independent branches can run in parallel. Figure numbers follow the current English manuscript.

If the archived Earth Engine extracts are available, start at step 02 with the corrected study-period data and the station files in `1991-2020/YR`. To regenerate a figure, use its plotting entry point with upstream results from the same analysis batch.

## Script inventory

`Code` contains 96 Python scripts and one Node.js script. This guide covers the main workflow, Figs. 1–6 and S1–S22, and the shared modules. The three analyses below were run on the 11 November 2026 data.

| Analysis | Script, inputs and outputs |
| --- | --- |
| Table S2(a–c), effect of treating blank suspended sediment concentration (Ss) as zero | [audit_ss_zero_encoding.py](06_Validation_and_sensitivity/06_Sediment_provenance/audit_ss_zero_encoding.py) reads the original hydrology workbooks, quality-controlled daily hydrology, the station mapping, BiLSTM inputs and SHAP event samples. Gauge-day provenance and three summaries are written to `Result/Sediment_provenance/20261111`. |
| Table S4(e–f), sensitivity to unknown discharge exposure | [unknown_q_sensitivity.py](06_Validation_and_sensitivity/07_Q_exposure_support/unknown_q_sensitivity.py) uses Table S4(a–d) daily discharge provenance, completed daily data, reference-period air temperature, Data S8 daily DO and baseline event statistics. It writes the event screen, reference days, sample summary, six contrasts and fit diagnostics to `Result/Q_exposure_support/20261111`. After updating the discharge inputs, rerun the Table S4(a–d) export first. |
| Tables S28–S30, original DO coverage and observation-restricted samples | [observation_coverage_sensitivity.py](06_Validation_and_sensitivity/11_Figure_source_analyses/observation_coverage_sensitivity.py) reads M07 and the post-event eligibility results. It writes the coverage thresholds, sample groups, 18 contrasts and fit diagnostics to `Result/Observation_coverage/20261111/S6` and `S7` and the station-block intervals for overall reference-direction disagreement to a CSV in the parent directory. |

All three scripts accept `--output-dir`. For Table S4(e–f), use `--q-dir` to point to the directory containing the daily discharge CSV exported for Table S4(a–d).

If you download reference-period temperatures again, the two Earth Engine scripts write to `1991-2020/M` and `1991-2020/B`. Check and combine their station files in `1991-2020/YR` before running event detection. The workflow for this batch starts from the archived `YR` files. The supplementary data inventory is maintained with the submission package.

## Workflow order

```text
01a Main-stem weather extraction + 01b tributary weather extraction
    Reuse M_V1 and B_V1 if already available.
  ↓
02 assemble_meteo.py
  ├── normalize_precipitation.py → Data/Meteorology_data_P → 05 merge
  ↓
03 1.WQ_Data_P.py + 04 2.Hy_Data_P.py (04 may run earlier)
  ├── after 03 → audit_qc_order_impact.py --stage qc
  ↓
05 4.WQ_Hy_M_EVI_Land_Data.py
  ├── V_Studyarea.py + V_P.py
  ↓
06 ED_N.py (uses the YR reference-period air temperatures)
  ├── V_event1.py
  ├── analyze_H_shap.py + analyze_F_shap.py + analyze_DC_shap.py
  │   + analyze_H5F_shap.py + analyze_F5H_shap.py (can run in parallel)
  │     → shap_group_validation.py + audit_shap_group_contributions.py
  ↓
07 Lstm_Y.py
  ├── audit_qc_order_impact.py --stage full --events-file Data/Event_N/All_Events_Missing_Status.csv
  ↓
08 LSTM_V1.py
  ├── V_Loss.py + V_NSE.py + V_space1.py
  ├── Monthly summaries before and after reconstruction → V_DO_time1.py
  ↓
09 ED_F.py
  ├── V_do_reconstruction_validation.py
  ├── validate_continuous_do_gaps.py → V_Continuous_gap_validation.py
  ↓
10a decrease.py + 10b recovery.py
  ├── 10a → Shock maps and distributions
  ├── 10a → M04_paired_baselines.py + M06_spatial_independence.py → M07_absolute_DO.py
  ├── 10a → M05_sensitivity.py → V_definition_counts.py + V_definition_responses.py --input-dir <results-directory>
  ├── 10b → Early post-event slope maps and distributions
  ↓
11 compare_event_do_responses.py → Core_Contrasts.csv → V2 Figs. 3d and 6d

Additional branches
  11 + completed DO and quality-controlled daily data → observation_only_references.py (Data S8)
  M04 + M05 + 11 + Data S8 inputs → antecedent reference sensitivity → Fig. S14
  M05 + 11 + Data S8 inputs → post-event eligibility → restricted slope contrasts → Figs. S22 and 6
  M07 + daily provenance and weather → Jiagaiqiao source audit → Fig. S18
  M07 + observed and completed daily series + event registry → real DO gaps and observation provenance → Table S11(d)
  05 + hydrology before and after interpolation → Table S4(a–d) CSV export
  11 + station mapping and completed daily data → shared-gauge clustering → Tables S22b and S23c
```

## 01–05 Data preparation and integration

### 01a Main-stem Earth Engine extracts

[3.M_EVI_Land_Data_V1.py](01_Data_integration/01_Meteorology_ecology/3.M_EVI_Land_Data_V1.py) uses `Config/Stations_M.csv` and the Earth Engine ERA5-Land, MODIS EVI and Dynamic World products. It aggregates instantaneous variables and hourly increments separately by Beijing-time calendar day, extracts EVI and land cover, and can recover station coordinates from the retained water-quality files for the same group.

The script writes station CSVs, including `SSR` and `TP_M`, to `Data/Meteorology_data/2021-2025/M_V1`. Table S1 describes the products and units. Run this step when the main-stem extracts need refreshing. Step 02 can validate the archived files directly.

### 01b Tributary Earth Engine extracts

[3.B_EVI_Land_Data_V1.py](01_Data_integration/01_Meteorology_ecology/3.B_EVI_Land_Data_V1.py) applies the same daily aggregation and extraction to the stations in `Config/Stations_B.csv` and writes the station CSVs to `Data/Meteorology_data/2021-2025/B_V1`. Step 02 selects the 44 tributary and 22 main-stem stations in the study mapping. The same products and units apply to Table S1.

### 02 Assemble study-period weather files

[assemble_meteo.py](01_Data_integration/01_Meteorology_ecology/assemble_meteo.py) validates the 66 stations across `M_V1` and `B_V1` and collects the daily files in `Data/Meteorology_data/2021-2025/YR_V1`.

### 02b Normalize precipitation

[normalize_precipitation.py](01_Data_integration/01_Meteorology_ecology/normalize_precipitation.py) sets negative `TP` or `TP_M` values to zero in the 66 daily files from `Data/Meteorology_data/2021-2025/YR_V1` and writes to `Data/Meteorology_data_P`. The merge in step 05 reads that directory. After merging, `ED_N.py` prepares the SHAP event samples and `Lstm_Y.py` builds the DO reconstruction inputs, including three-day cumulative precipitation.

### 03 Screen and aggregate water-quality observations

[1.WQ_Data_P.py](01_Data_integration/02_Water_quality_QC/1.WQ_Data_P.py) reads timestamp-level measurements from `Data/WaterQuality_data/WQ_data_YR` and matches daily surface pressure (`SP`) from `Data/Meteorology_data/2021-2025/YR_V1`. It screens DO using absolute and saturation limits, applies static bounds to other variables, aggregates to calendar days and screens stations for inclusion. The output in `Data/WaterQuality_data_P/WQ_data_YR` contains `Retained` and `Excluded` branches with `Raw` and `Cleaned` daily data. Missingness summaries are written to `Result/WQ_data_missing`. Tables S1 and S2 describe the variables and QC rules. The screened DO feeds the later analyses.

### Audit the water-quality screening order after step 03

[audit_qc_order_impact.py](06_Validation_and_sensitivity/01_QC_order_audit/audit_qc_order_impact.py) compares timestamp-level retention, daily means, original-observation flags and downstream model and event inputs under the two QC orders. It uses the original DO and water temperature, the same current pressure data and the fixed station list and `Cleaned` daily files. The full audit also reads the current LSTM features and event data.

Run `--stage qc` after generating the QC daily tables. Run `--stage full` after step 07 and before model training. Both update `qc_order_impact.csv` and `changed_record_log.csv` in `Result/M2b_QC_Order`. The `Stage` field identifies the stage. Use the `SUMMARY` rows for Table S2 when the `RUN` record reports `COMPLETED`. See the [QC order audit guide](06_Validation_and_sensitivity/01_QC_order_audit/README_M2b_QC_Order_Audit.md) for the commands and output fields.

### 04 Screen and interpolate hydrology

[2.Hy_Data_P.py](01_Data_integration/03_Hydrological_QC/2.Hy_Data_P.py) reads the 42 original workbooks in `Data/Hydrology_data/Hy_data_YR`, standardizes gauge names, rivers and dates, builds the daily calendar, applies the initial missing-Ss rule and interpolates water level (`WL`), discharge (`Q`) and Ss linearly. It writes uninterpolated and interpolated files to `Data/Hydrology_data_P/Hy_data_YR/Original` and `Interpolated` and missingness summaries to `Result/Hy_data_missing`.

These daily files support Tables S1–S4. Step 04 can run alongside the weather and water-quality steps. Separate scripts generate the Table S4 threshold and episode catalogue and audit the sediment provenance.

### 05 Match stations and merge daily data

[4.WQ_Hy_M_EVI_Land_Data.py](01_Data_integration/04_Spatial_matching_and_merge/4.WQ_Hy_M_EVI_Land_Data.py) matches each screened water-quality station to its nearest gauge on the same river, using river names and coordinates. It merges the screened water-quality data, completed hydrology and daily weather from `Data/Meteorology_data_P`, writing meteorological `TP` as `TP_M`. It checks for a same-river candidate before writing results.

The output in `Data/WQ_Hy_M_DYN_data/YR` includes station-level, reach-level and basin-wide daily tables, plus `Match_Summary_Retained.csv` in the existing column order. Compare that summary with `Config/Study_Station_Mapping.csv`, the fixed mapping used for discharge provenance, formal event registration and clustered analysis. The station and river information supports Table S3. In this batch, Yingugongluqiao is matched to Qingtongxia at a distance of 63.40 km.

### Discharge provenance for Table S4

[export_table_s4_q_support.py](06_Validation_and_sensitivity/07_Q_exposure_support/export_table_s4_q_support.py) uses the current mapping and daily hydrology before and after interpolation. It checks the gauge and discharge in the merged daily tables and calculates the S0 P95 threshold and consecutive high-flow episodes. It writes three CSVs to `Supplementary_Data/TableS4_Q_Support` by default. Use `--output-dir` to write elsewhere.

### Plotting after step 05

| Figure | Script | Inputs |
| --- | --- | --- |
| Study area and station locations | [V_Studyarea.py](Figure_V1/Figure1/a/Scripts/V_Studyarea.py) | Station matching summary from step 05 and the included basemap in `Data/GIS/1.gdb`. |
| Correlation matrix of environmental variables | [V_P.py](Figure_V1/SFigure/SFigure09_Correlation/Scripts/V_P.py) | Merged daily station files. |

## 06 Observed event detection and SHAP

Use the archived daily reference temperatures in `Data/Meteorology_data/1991-2020/YR`. To refresh them, run the two download scripts in parallel, validate the station files and collect the study stations in `YR` before running `ED_N.py`. The precipitation and SSR corrections do not require a new reference-temperature download.

| Optional script | Station list | Output |
| --- | --- | --- |
| [3.M_1991-2020T.py](01_Data_integration/05_Reference_temperature_download/3.M_1991-2020T.py) | `Config/Stations_M.csv` | `Data/Meteorology_data/1991-2020/M` |
| [3.B_1991-2020T.py](01_Data_integration/05_Reference_temperature_download/3.B_1991-2020T.py) | `Config/Stations_B.csv` | `Data/Meteorology_data/1991-2020/B` |

Both scripts download the daily 1991–2020 temperature series through Earth Engine. The data support the thresholds in Tables S1, S5 and S6. The B script uses the tributary configuration and output directory for this batch.

### 06 Detect and label the observed events

[ED_N.py](02_Event_identification/01_Event_registry/ED_N.py) uses the YR reference temperatures and daily records in `Data/WQ_Hy_M_DYN_data/YR/by_station_all` to identify five event types. It labels individual days, summarizes original observation coverage, and separates complete events from those with missing data.

Its outputs under `Data/Event_N` include `By_Site_Labeled`, `Perfect_Events_Data`, `Missing_Events_Data` and the event counts. These support Table S5 and Figs. S1–S2 and provide inputs for reconstruction and the observation-based SHAP analysis. [V_event1.py](Figure_V1/SFigure/SFigure02_Event_frequency/Scripts/V_event1.py) plots event counts by station.

### Tree-model and SHAP analysis

The five event-type scripts can run in parallel after step 06. Each fits random forest, XGBoost and Extra Trees to DO and 16 predictors, uses Bayesian search to select the model for that event type, and exports feature- and group-level SHAP contributions. All inputs are under `Data/Event_N/Perfect_Events_Data/Overall`. Use the corrected `TP_M` and `SSR` in the current batch.

| Event type | Script | Input CSV | Output prefix under `Result/shap` |
| --- | --- | --- | --- |
| Heatwave only | [analyze_H_shap.py](07_SHAP_analysis/03_Tree_model_analysis/analyze_H_shap.py) | `Heatwave_Only_Perfect_Overall.csv` | `Heatwave_ThreeModel_SHAP_Analysis` |
| High flow only | [analyze_F_shap.py](07_SHAP_analysis/03_Tree_model_analysis/analyze_F_shap.py) | `Flood_Only_Perfect_Overall.csv` | `Flood_ThreeModel_SHAP_Analysis` |
| Direct overlap | [analyze_DC_shap.py](07_SHAP_analysis/03_Tree_model_analysis/analyze_DC_shap.py) | `Direct_Compound_Perfect_Overall.csv` | `DC_ThreeModel_SHAP_Analysis` |
| Heatwave followed by high flow | [analyze_H5F_shap.py](07_SHAP_analysis/03_Tree_model_analysis/analyze_H5F_shap.py) | `H5F_Compound_Perfect_Overall.csv` | `H5F_ThreeModel_SHAP_Analysis` |
| High flow followed by heatwave | [analyze_F5H_shap.py](07_SHAP_analysis/03_Tree_model_analysis/analyze_F5H_shap.py) | `F5H_Compound_Perfect_Overall.csv` | `F5H_ThreeModel_SHAP_Analysis` |

The scripts also save individual predictions and SHAP values for Figs. S19 and S20. The search space is documented in Table S32, model performance and parameters in Table S33, and contributions in Table S35. SHAP contributions describe attribution in the fitted concentration prediction models.

### Grouped prediction validation

[shap_group_validation.py](07_SHAP_analysis/02_Grouped_prediction_validation/shap_group_validation.py) uses the five observed-event datasets and the saved hyperparameters and model comparisons in `Result/shap`. It refits the three tree models for each event type with fixed hyperparameters and holds out complete site–event groups. It writes evaluation metrics, individual predictions, split membership and the environment record to `Result/SHAP_Grouped_Validation`. These are the inputs for Table S34. Figs. S19 and S20 use the saved plotting data from the tree-model scripts.

### Audit the saved SHAP contributions

[audit_shap_group_contributions.py](07_SHAP_analysis/01_SHAP_contribution_audit/audit_shap_group_contributions.py) checks the five model configurations against their observed samples and saved `Feature_Importance`, `Group_Contribution`, `Log` and `Model_Comparison` outputs. It verifies sums, mean contribution per feature within each group, and event and site coverage.

Use `Result/Statistical_comparisons/SHAP_group_descriptive_audit/SHAP_Group_Descriptive_Audit.csv` for the descriptive summaries in Table S35(b–d). The script reports audit status and provenance in the terminal and uses the saved SHAP values. It performs no model fitting or inference.

## 07–08 BiLSTM inputs, training and reconstruction

### 07 Prepare model features

[Lstm_Y.py](03_DO_reconstruction/01_Input_features/Lstm_Y.py) reads `Data/Event_N/By_Site_Labeled` and writes `Data/LSTM_Y`. It keeps the DO target and adds calendar and reach codes and rolling features, including `TP_M_sum3` and `Q_mean7`. Regenerate these features after changing the precipitation inputs. Table S9 describes the encodings, and Tables S10–S11 use the resulting workflow.

### Run the full screening-order audit after step 07

Run [audit_qc_order_impact.py](06_Validation_and_sensitivity/01_QC_order_audit/audit_qc_order_impact.py) with the observed-event file from step 06. This audit can run before model training.

```powershell
python "Code/06_Validation_and_sensitivity/01_QC_order_audit/audit_qc_order_impact.py" --stage full --events-file "Data/Event_N/All_Events_Missing_Status.csv"
```

### 08 Train the model and reconstruct daily DO

[LSTM_V1.py](03_DO_reconstruction/02_Model_training/LSTM_V1.py) takes 18 features and the original DO target from `Data/LSTM_Y`. It trains and applies the BiLSTM-Attention model and records its final weights, fitted scalers, feature order, settings and input hashes.

The output in `Result/BiLstm_Att_V1` includes `Imputed_Data`, `Reports/Loss_History`, `Reports/By_Site_Scatter` and `Saved_Models_V1`. The script validates the 66 reconstructed station tables and collects them in `Imputed_Data/By_Site_Imputed/YR` for `ED_F.py`. Table S9 draws on the model configuration, and Table S10 uses the pointwise evaluation. Generate the daily summaries separately for Tables S7 and S8. These files also support Fig. 2 and Figs. S5–S7.

### Summaries and plotting after step 08

[data_N.py](03_DO_reconstruction/03_Descriptive_summaries/data_N.py) summarizes the merged station files before reconstruction in `Result/Mean_N_data`. [data_F.py](03_DO_reconstruction/03_Descriptive_summaries/data_F.py) summarizes the completed station files in `Result/Mean_F_data`. These summaries support Tables S7–S8 and Fig. S10. Regenerate both summaries when switching analysis batches.

| Figure | Script | Inputs |
| --- | --- | --- |
| Training loss, performance maps and station time series | [V_Loss.py](Figure_V1/SFigure/SFigure05_Training_loss/Scripts/V_Loss.py), [V_NSE.py](Figure_V1/SFigure/SFigure06_Performance_maps/Scripts/V_NSE.py), [V_space1.py](Figure_V1/SFigure/SFigure07_Full_station_series/Scripts/V_space1.py) | Training, evaluation and reconstruction outputs from `LSTM_V1.py`. The maps also use the basemap. |
| Monthly DO before and after reconstruction | [V_DO_time1.py](Figure_V1/SFigure/SFigure10_Monthly_DO/Scripts/V_DO_time1.py) | Monthly station-level and basin-wide summaries under `Result/Mean_N_data/DO` and `Result/Mean_F_data/DO`. |

## 09 Event registry and DO validation

### 09 Register events on the completed daily series

[ED_F.py](04_DO_response/01_Completed_event_registry/ED_F.py) uses the YR reference temperatures and `Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR` to identify the five event types. It writes the event registry, antecedent references, event-period records, post-event windows and non-event days to `Data/Event_F/01_Registry_Data` through `05_Non_Event_Data`. It also writes `Data/Event_F/Event_Registry_For_Validation.csv`. The registry and windows support Table S5 and the response calculations for Table S12.

Both validation branches below can run alongside step 10 after `ED_F.py`. The two diagnostic plotting scripts export 600 DPI TIFF files.

### Pointwise reconstruction diagnostics

[V_do_reconstruction_validation.py](Figure_V1/SFigure/SFigure08_Additional_validation/Scripts/V_do_reconstruction_validation.py) reads saved test predictions in `Reports/By_Site_Scatter`, observed daily DO in `Data/LSTM_Y` and `Event_Registry_For_Validation.csv`. It summarizes pointwise errors, missed low-DO detections and observed gap lengths. It writes to `V/Figure_V1/SFigure/SFigure08_Additional_validation/Images` by default.

Change the 5 mg/L low-DO threshold with `--threshold`. Pass `--error-summary-csv`, `--false-negatives-csv` or `--gap-distribution-csv` to use summaries from the same batch.

### Continuous-gap validation

[validate_continuous_do_gaps.py](06_Validation_and_sensitivity/02_Continuous_gap_validation/validate_continuous_do_gaps.py) reads `DO_Model_Manifest_V1.json`, the matching weights, fitted scalers, LSTM inputs, saved test predictions and `Data/Event_F/Event_Registry_For_Validation.csv`. It masks 3-, 7-, 14- and 30-day windows with original DO observations in 2025 and compares masked predictions, unmasked predictions and the last observation carried forward. DO is hidden before constructing lagged features. Inference uses the saved model directly and does not feed predictions back into subsequent windows.

Each run under `Result/Continuous_gap_validation` writes 11 CSVs covering candidate windows, pointwise predictions, errors and low-DO classification. The terminal reports `COMPLETED` on success and the run writes `error_traceback.txt` on failure. Plotting scripts check for a complete set of result tables. The outputs support Tables S10–S11 and Figs. 2b and S8.

### Plot continuous-gap validation

[V_Continuous_gap_validation.py](Figure_V1/SFigure/SFigure08_Additional_validation/Scripts/V_Continuous_gap_validation.py) selects the latest completed run by default. Use `--validation-dir` to select a run and `--metrics-csv`, `--intervals-csv`, `--paired-csv` or `--low-do-csv` to select individual tables. Gap lengths and low-DO thresholds come from the result tables.

Pointwise errors come from the test predictions by default. Use `--pointwise-csv` to supply a saved summary. Both validation plotting scripts accept `--scatter-dir`, `--daily-dir`, `--registry` and `--test-year` for another analysis batch. For the station example, use `--site` or `--site-scatter-csv`. The default destination is the validation-run subdirectory under `V/Figure_V1/SFigure/SFigure08_Additional_validation/Images`.

## 10–11 Event responses, statistics and plots

### 10a Calculate responses relative to antecedent and seasonal references

[decrease.py](04_DO_response/02_Departure_metrics/decrease.py) reads the event registry, antecedent windows, event-period records and non-event days under `Data/Event_F`. It calculates event DO, seasonal and antecedent references, and the two response ratios `Shock-L` and `Shock-S`. It writes event and station summaries under `Result/Decrease/By_Site` and `Result/Decrease/Overall`, including `FINAL_Shock_All_Events_Combined.csv`.

Use these outputs for Table S12, Figs. 3–5 and the maps and distributions in Figs. S12–S13. The MixedLM and analyses M04, M05, M06 and M07 also use the response outputs from this batch.

### 10b Calculate early post-event slopes

[recovery.py](04_DO_response/03_Early_post_event_slopes/recovery.py) reads the event registry and antecedent reference and post-event windows from `Data/Event_F`. It fits the early linear slope using DO on days 1, 3, 5, 7 and 9 after the event and records eligibility and the diagnostic time to first reach the antecedent reference. It writes event and station summaries under `Result/Recovery`, including `FINAL_Recovery_All_Events_Combined.csv`.

The slope results support Table S12 and Fig. 6. Table S21 uses the separate post-event eligibility audit listed below. A positive slope indicates rising DO during the window. The script reports return-to-reference time separately.

### Plot the response metrics

| Figure | Scripts | Inputs |
| --- | --- | --- |
| Shock-L and Shock-S maps and distributions | [V_de_space.py](Figure_V1/SFigure/SFigure12_Seasonal_departure/Scripts/V_de_space.py), [V_de_space1.py](Figure_V1/SFigure/SFigure13_Antecedent_departure/Scripts/V_de_space1.py), [V_de_value.py](Figure_V1/SFigure/SFigure12_Seasonal_departure/Scripts/V_de_value.py), [V_de_value1.py](Figure_V1/SFigure/SFigure13_Antecedent_departure/Scripts/V_de_value1.py) | Station and event outputs from `decrease.py`. The maps also use the basemap. |
| Early post-event slope maps and distributions | [V_re_space.py](Figure_V1/SFigure/SFigure17_Early_slope_maps/Scripts/V_re_space.py), [V_re_value.py](Figure_V1/SFigure/SFigure17_Early_slope_maps/Scripts/V_re_value.py) | Station and event outputs from `recovery.py`. The map also uses the basemap. |

### Supplementary analyses after step 10a

The analyses below use results from `decrease.py`. M07 imports table-reading and export functions from M06. Run the M05 figure scripts after its analysis outputs are available.

#### M04 Common events and agreement between references

[M04_paired_baselines.py](06_Validation_and_sensitivity/03_Paired_references/M04_paired_baselines.py) joins the `Data/Event_F` registry with the two Shock ratios in `Result/Decrease/Overall`. It checks common events with valid ratios under both references, classifies direction and summarizes agreement by stratum with station-block bootstrap intervals. The dated directory under `Result/Decrease/Paired_baselines` contains the common sample, classifications, differences and intervals in CSV and Excel. These support Tables S13–S15 and Fig. S11. Plot labels use event and station counts from the current results.

#### M06 Spatial independence diagnostics

[M06_spatial_independence.py](06_Validation_and_sensitivity/10_Spatial_independence/M06_spatial_independence.py) checks the existing station mapping, shared discharge inputs and discharge coverage. It summarizes event and site counts and calculates descriptive downstream leave-one-site-out (LOSO) sensitivity for Shock-L. It uses the mapping and reach tables in `Data/WQ_Hy_M_DYN_data/YR`, the `Event_F` registry, `Result/Decrease` and original and completed discharge.

The dated directory under `Result/Data_preprocessing/Hydrological_matching` contains `Hydro_Matching`, `Event_Site_Counts`, `Downstream_LOSO` and `Type_LOSO` in CSV or XLSX. These support the mapping in Table S3, the descriptive results in Table S24 and Fig. S16. The analysis documents station matching and sample coverage. Hydrological connectivity is outside its scope.

#### M07 Absolute DO and observation provenance

[M07_absolute_DO.py](06_Validation_and_sensitivity/04_Absolute_DO_and_provenance/M07_absolute_DO.py) uses the `Event_F` registry, completed V1 DO, `Cleaned` observed DO and the Shock ratios for this batch. It computes event means and minimum daily DO, event and day proportions below 5, 3 and 2 mg/L, and summaries by event type and reach. It tracks the observation or reconstruction source and whether most values were reconstructed.

The dated folder under `Result/Decrease/DO_observation_sensitivity` contains event-level absolute DO, summaries by type and reach, `Observed_Imputed`, `Shock_DO_Relationship` and `Below2_Day_Trace` tables. These support Tables S17–S20, the coverage work for Table S28 and Figs. S15 and S21. Use the dedicated site audit and coverage model scripts for Tables S27–S30.

#### M05 Event-definition sensitivity

[M05_sensitivity.py](06_Validation_and_sensitivity/05_Definition_sensitivity/M05_sensitivity.py) reuses `ED_F.py` and `decrease.py` with the reference temperatures, completed V1 DO, current `Event_F` registry and Shock results. It verifies the current S0 baseline before assessing alternative exposure definitions, thresholds and linking rules. It writes S0–S5 event lists and thresholds, Shock comparisons, summaries and notes to a dated folder under `Result/Event/Definition_sensitivity`.

These results support Table S6 and Figs. S3–S4. Run [V_definition_counts.py](Figure_V1/SFigure/SFigure03_Definition_counts/Scripts/V_definition_counts.py) and [V_definition_responses.py](Figure_V1/SFigure/SFigure04_Definition_responses/Scripts/V_definition_responses.py) after the analysis, using `--input-dir` for the M05 results directory.

### 11 Fit the main statistical models

[compare_event_do_responses.py](05_Statistical_analysis/01_Mixed_effects/compare_event_do_responses.py) combines the `Event_F` registry with ratios from `Result/Decrease` and slopes from `Result/Recovery`. For each response it fits a site-random-intercept mixed-effects model to the eligible events, adjusts for reach and season, and returns prespecified contrasts, confidence intervals and Holm-adjusted p values.

Outputs in `Result/Statistical_comparisons` include `Event_Level_Statistical_Data.csv`, `Core_Contrasts.csv`, `Model_Diagnostics.csv` and `Statistical_Analysis_Results.xlsx`. These support Tables S22–S23. The Shock-L and Shock-S contrasts are used in V2 Fig. 3d, and the `Recovery_Slope` contrasts for all eligible events are used in Fig. 6d. Use the observation-coverage branch for the restricted-sample models in Table S30.

## Inputs for dedicated analyses and figures

Use results from the same batch. Several scripts have fixed input and output paths for 11 November 2026. Check these before running a different batch, and work from a copy if you need to preserve existing outputs.

| Analysis and manuscript location | Script | Results and inputs |
| --- | --- | --- |
| Observation-only dual references and daily DO provenance, Table S31 and Data S8 | [observation_only_references.py](06_Validation_and_sensitivity/11_Figure_source_analyses/observation_only_references.py) | `Result/Observation_only_references/20261111`. Uses baseline event statistics, completed DO and original and quality-controlled daily means. |
| Antecedent windows and minimum observation requirements, Table S16, Data S3 and Fig. S14 | [antecedent_reference_sensitivity.py](06_Validation_and_sensitivity/11_Figure_source_analyses/antecedent_reference_sensitivity.py) | `Result/Antecedent_reference_sensitivity/20261111_Analysis`. Uses M04, M05, the main statistics and observation-only results. Use `--output-dir` for the analysis and the same directory as `--run-dir` in the plot. |
| Slope eligibility and restricted slope models, Tables S21 and S30, Data S4, Figs. S22 and 6 | [post_event_eligibility.py](06_Validation_and_sensitivity/11_Figure_source_analyses/post_event_eligibility.py), then [restricted_slope_contrasts.py](06_Validation_and_sensitivity/11_Figure_source_analyses/restricted_slope_contrasts.py) | `Result/Post_event_eligibility/20261111_Analysis`. Uses main statistics, M05 thresholds, the event registry and daily DO provenance. Run the eligibility check with `--output-dir` set to a new directory. The restricted model script reads the archived path by default. Match the paths before using a different batch. |
| Diagnostic site exclusion and Jiagaiqiao provenance, Table S25, Audit Note S26, Table S27, Data S5 and Fig. S18 | [jiagaiqiao_source_audit.py](06_Validation_and_sensitivity/11_Figure_source_analyses/jiagaiqiao_source_audit.py) | `Result/Jiagaiqiao_source_audit/20261111_Audit`. Uses original timestamps, original and screened daily means, completed DO, environmental data and event records. Use `--output-dir` for a new analysis. Site-exclusion summaries come from `existing_site_exclusion_sensitivity.csv`. |
| Real DO gaps and low-DO provenance, Table S11(d), Data S2 and Fig. S8 | [real_do_gap_support.py](06_Validation_and_sensitivity/02_Continuous_gap_validation/real_do_gap_support.py) | `Result/Continuous_gap_validation/Real_DO_gap_support`. Uses M07, the baseline registry and screened and reconstructed daily tables. It also writes `source_gap_distribution.csv` for Fig. S8. Use `--source-gap-csv` to specify another destination for that CSV. |

## Baseline source and shared modules

### LSTM baseline

[LSTM.py](03_DO_reconstruction/02_Model_training/LSTM.py) records the baseline BiLSTM-Attention workflow behind the configuration in Table S9. It trains on 2021–2023, validates in 2024, tests in 2025 and refits on 2021–2024 using the selected number of epochs before filling missing DO. It reads `Data/LSTM_Y` and writes to `Result/BiLstm_Att_V1` when executed. The V1 model manifest records this baseline source. Run `LSTM_V1.py` as the reconstruction entry point for this batch.

### English table output

[english_table_output.py](05_Statistical_analysis/00_Shared_utilities/english_table_output.py) standardizes English headings, categories and place names and serializes publication CSVs. It flags unmapped Chinese text and can escape source paths reversibly. The caller passes a DataFrame or CSV bytes and saves the returned table. M01 and other export scripts use this helper after calculating the statistics.

### Reconstruction diagnostic inputs

[reconstruction_diagnostic_inputs.py](06_Validation_and_sensitivity/02_Continuous_gap_validation/reconstruction_diagnostic_inputs.py) supports both DO diagnostic figures. It aligns V1 station-level test predictions with observed dates, uses the event registry to classify test dates and summarizes real gap lengths.

The SHAP workflow uses [analysis_config.py](07_SHAP_analysis/analysis_config.py) for shared settings and [shap_plot_data.py](07_SHAP_analysis/03_Tree_model_analysis/shap_plot_data.py) to assemble and save predictions and SHAP plotting data.

### M4 Pairing priority sensitivity

[M04_pairing_priority.py](06_Validation_and_sensitivity/08_Pairing_priority/M04_pairing_priority.py) compares the baseline order DC → F5H → H5F with DC → H5F → F5H. It reuses the `ED_F.py` classifier and compares events using stable sets of constituent episode IDs. It holds thresholds, heatwave merging, direct-overlap components and tie-breaking fixed.

Inputs are completed DO with the `Original_DO` flag, the S0 registry, historical `Tmax`, baseline event metrics and the statistical script. The companion [ED_F_pre_M4.py](06_Validation_and_sensitivity/08_Pairing_priority/ED_F_pre_M4.py) verifies the earlier classification. Use `--audit-only` for the classification check. The full run recalculates metrics when events change, including references, non-event dates, next-event cutoffs and early slopes. It writes `pairing_priority_comparison.csv`, `priority_core_results.csv` and model diagnostics under `Result/Event/Pairing_priority/20261111`. It keeps the two configurations separate and uses the existing completed DO and main results as inputs. See Table S6(d–f) and Supplementary Methods S3–S4.

### M5 Sensitivity to clustering by shared gauge

[M5_hydro_cluster_sensitivity.py](06_Validation_and_sensitivity/09_Hydro_cluster_sensitivity/M5_hydro_cluster_sensitivity.py) fits two OLS specifications for each of three outcomes, one with five event types and one with a compound-event indicator. Both include water-quality-station fixed effects and onset season. CR1 covariance is clustered by matched discharge gauge.

The analysis uses the baseline event-level statistics, six main MixedLM contrasts and fixed mapping. It checks `Matched_HS_Name` against the completed series. Finite positive Shock-L and Shock-S values are logged, retaining negative and zero log values. Slopes remain on the original scale. The analysis fits six models with the same events and DO values and uses `t(G−1)` as the reference approximation. Holm adjustment covers the two tests within each response, and 95% confidence intervals are unadjusted.

Results under `Result/Statistical_comparisons/Hydro_cluster/20261111` include `hydro_cluster_contrasts.csv`, `model_support_counts.csv`, event IDs, mapping checks, diagnostics, the software version and covariance options. They support Tables S22(b) and S23(c) and Supplementary Method S13.

## Figure scripts

Generate the analysis results before running the standalone scripts in `Code/Figure_V1`. The figure numbers below follow manuscript V2. Fig. 3 uses scripts under `Figure4`, and Fig. 4 uses scripts under `Figure3`. The script and default output paths follow the original directory numbering.

| V2 figure | Content | Script |
| --- | --- | --- |
| Fig. 1a | Study area and monitoring stations | [V_Studyarea.py](Figure_V1/Figure1/a/Scripts/V_Studyarea.py) |
| Fig. 1b | Update the workflow counts from the event table and export PNG and TIFF | [refresh_workflow_counts.py](Figure_V1/Figure1/b/Scripts/refresh_workflow_counts.py), [V_export_Figure1b.mjs](Figure_V1/Figure1/b/Scripts/V_export_Figure1b.mjs) |
| Fig. 1c | DO observation coverage during events | [V_event_DO_support.py](Figure_V1/Figure1/c/Scripts/V_event_DO_support.py) |
| Fig. 2a–c | Pointwise validation, continuous-gap validation and low-DO recall | [V_pointwise_validation.py](Figure_V1/Figure2/a/Scripts/V_pointwise_validation.py), [V_continuous_gap_validation.py](Figure_V1/Figure2/b/Scripts/V_continuous_gap_validation.py), [V_low_DO_recall.py](Figure_V1/Figure2/c/Scripts/V_low_DO_recall.py) |
| Fig. 3a–d | Minimum DO, proportion of low-DO days, observation provenance and adjusted contrasts | [V_minimum_daily_DO.py](Figure_V1/Figure4/a/Scripts/V_minimum_daily_DO.py), [V_low_day_fraction.py](Figure_V1/Figure4/b/Scripts/V_low_day_fraction.py), [V_low_DO_sources.py](Figure_V1/Figure4/c/Scripts/V_low_DO_sources.py), [V_adjusted_contrasts.py](Figure_V1/Figure4/d/Scripts/V_adjusted_contrasts.py) |
| Fig. 4a–b | Paired reference ratios and classification sensitivity | [V_paired_reference_ratios.py](Figure_V1/Figure3/a/Scripts/V_paired_reference_ratios.py), [V_classification_sensitivity.py](Figure_V1/Figure3/b/Scripts/V_classification_sensitivity.py) |
| Fig. 5a–c | Samples under each definition, common events and daily DO at Jiagaiqiao | [V_eligible_populations.py](Figure_V1/Figure5/a/Scripts/V_eligible_populations.py), [V_common_events.py](Figure_V1/Figure5/b/Scripts/V_common_events.py), [V_Jiagaiqiao_daily_DO.py](Figure_V1/Figure5/c/Scripts/V_Jiagaiqiao_daily_DO.py) |
| Fig. 6a–d | Early slopes, window eligibility, observation coverage and adjusted contrasts | [V_early_slope_distribution.py](Figure_V1/Figure6/a/Scripts/V_early_slope_distribution.py), [V_window_eligibility.py](Figure_V1/Figure6/b/Scripts/V_window_eligibility.py), [V_post_event_source_support.py](Figure_V1/Figure6/c/Scripts/V_post_event_source_support.py), [V_adjusted_slope_contrasts.py](Figure_V1/Figure6/d/Scripts/V_adjusted_slope_contrasts.py) |

For Fig. 1b, the template is `Code/Figure_V1/Figure1/b/Assets/Figure1b_template.pptx`. After updating the counts, the Node.js export script reads `V/Figure_V1/Figure1/b/Images/Figure1b_Workflow.pptx`. It uses `@oai/artifact-tool` and `sharp`. Pass `--node-modules` if those packages are installed outside the default directory.

| Supplementary figure | Script |
| --- | --- |
| SFigure01_Event_composition | [V_event_composition.py](Figure_V1/SFigure/SFigure01_Event_composition/Scripts/V_event_composition.py) |
| SFigure02_Event_frequency | [V_event1.py](Figure_V1/SFigure/SFigure02_Event_frequency/Scripts/V_event1.py) |
| SFigure03_Definition_counts | [V_definition_counts.py](Figure_V1/SFigure/SFigure03_Definition_counts/Scripts/V_definition_counts.py) |
| SFigure04_Definition_responses | [V_definition_responses.py](Figure_V1/SFigure/SFigure04_Definition_responses/Scripts/V_definition_responses.py) |
| SFigure05_Training_loss | [V_Loss.py](Figure_V1/SFigure/SFigure05_Training_loss/Scripts/V_Loss.py) |
| SFigure06_Performance_maps | [V_NSE.py](Figure_V1/SFigure/SFigure06_Performance_maps/Scripts/V_NSE.py) |
| SFigure07_Full_station_series | [V_space1.py](Figure_V1/SFigure/SFigure07_Full_station_series/Scripts/V_space1.py) |
| SFigure08_Additional_validation | [V_same_date_baseline_comparison.py](Figure_V1/SFigure/SFigure08_Additional_validation/Scripts/V_same_date_baseline_comparison.py), [V_Continuous_gap_validation.py](Figure_V1/SFigure/SFigure08_Additional_validation/Scripts/V_Continuous_gap_validation.py), [V_do_reconstruction_validation.py](Figure_V1/SFigure/SFigure08_Additional_validation/Scripts/V_do_reconstruction_validation.py) |
| SFigure09_Correlation | [V_P.py](Figure_V1/SFigure/SFigure09_Correlation/Scripts/V_P.py) |
| SFigure10_Monthly_DO | [V_DO_time1.py](Figure_V1/SFigure/SFigure10_Monthly_DO/Scripts/V_DO_time1.py), [V_station_mean_DO_map.py](Figure_V1/SFigure/SFigure10_Monthly_DO/Scripts/V_station_mean_DO_map.py) |
| SFigure11_Direction_by_type | [V_direction_by_type.py](Figure_V1/SFigure/SFigure11_Direction_by_type/Scripts/V_direction_by_type.py) |
| SFigure12_Seasonal_departure | [V_de_space.py](Figure_V1/SFigure/SFigure12_Seasonal_departure/Scripts/V_de_space.py), [V_de_value.py](Figure_V1/SFigure/SFigure12_Seasonal_departure/Scripts/V_de_value.py) |
| SFigure13_Antecedent_departure | [V_de_space1.py](Figure_V1/SFigure/SFigure13_Antecedent_departure/Scripts/V_de_space1.py), [V_de_value1.py](Figure_V1/SFigure/SFigure13_Antecedent_departure/Scripts/V_de_value1.py) |
| SFigure14_Antecedent_sensitivity | [V_antecedent_reference_sensitivity.py](Figure_V1/SFigure/SFigure14_Antecedent_sensitivity/Scripts/V_antecedent_reference_sensitivity.py) |
| SFigure15_Absolute_DO_relationship | [V_shock_minimum_DO.py](Figure_V1/SFigure/SFigure15_Absolute_DO_relationship/Scripts/V_shock_minimum_DO.py) |
| SFigure16_Spatial_independence | [V_downstream_LOSO.py](Figure_V1/SFigure/SFigure16_Spatial_independence/Scripts/V_downstream_LOSO.py) |
| SFigure17_Early_slope_maps | [V_re_space.py](Figure_V1/SFigure/SFigure17_Early_slope_maps/Scripts/V_re_space.py), [V_re_value.py](Figure_V1/SFigure/SFigure17_Early_slope_maps/Scripts/V_re_value.py) |
| SFigure18_Source_audit | [V_jiagaiqiao_source_audit.py](Figure_V1/SFigure/SFigure18_Source_audit/Scripts/V_jiagaiqiao_source_audit.py) |
| SFigure19_Selected_model_fit | [V_selected_model_fit.py](Figure_V1/SFigure/SFigure19_Selected_model_fit/Scripts/V_selected_model_fit.py) |
| SFigure20_SHAP_exploratory | [V_shap_explanations.py](Figure_V1/SFigure/SFigure20_SHAP_exploratory/Scripts/V_shap_explanations.py) |
| SFigure21_Low_DO_event_occurrence | [V_low_DO_event_occurrence.py](Figure_V1/SFigure/SFigure21_Low_DO_event_occurrence/Scripts/V_low_DO_event_occurrence.py) |
| SFigure22_Post_event_eligibility | [V_post_event_eligibility.py](Figure_V1/SFigure/SFigure22_Post_event_eligibility/Scripts/V_post_event_eligibility.py) |

For Figs. S11, S15 and S16, the scripts use the most recent available results by default. Use `--input-dir` to select the M04, M07 or M06 results. For Fig. S18, use `--run-dir` to select the provenance audit and `--output-dir` to save the figure in a new directory.

The scripts for Figs. S8, S15, S18 and S22 check panel alignment. The Fig. S22 script also exports a panel alignment diagnostic SVG.

Figs. S19 and S20 read `*_Plot_Data.npz` files under `Result/shap`. The five `analyze_*_shap.py` scripts save these files alongside their CSV outputs. For an existing analysis batch, use `--prepare-saved-figures` with the corresponding script to refit the selected model at the saved settings and check the saved prediction metrics and SHAP percentages. Use `--event` to select the event type in the plotting scripts.

Station time series are saved in the `PNG` and `TIFF` subdirectories of `V/Figure_V1/SFigure/SFigure07_Full_station_series/Images/YR`. The combined gap-validation figure and its source tables are saved in `V/Figure_V1/SFigure/SFigure08_Additional_validation/Images` and are used by the Fig. 2 scripts.
