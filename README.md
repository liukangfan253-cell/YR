# Dissolved oxygen responses to single and compound heatwave and high-flow events in the Yellow River Basin: Event sequence and assessment robustness

This repository provides the analysis code, figure-generation scripts and selected supporting inputs for the study. It documents Sections 2.1–2.5 and 3.1–3.7, Supplementary Method S19, main Figures 1–6 and Supplementary Figures S1–S26. The manuscript and supplementary information are submitted separately to *Journal of Hydrology* and are not required to be stored in this repository.

Use [Manuscript and directory structure](#manuscript-and-directory-structure) to find the analyses associated with each section, [Workflow order](#workflow-order) to follow the computational sequence and [Figure scripts](#figure-scripts) to locate the plotting entry points. The full-rerun requirements and data-access limitations are documented below.

## Package contents and reproducibility

The workflow uses the Conda environment `my_pytorch` with Python 3.9.21. See the [environment version record](Code/environments/README.md). Run the commands below from the directory containing `Code/`, `Data/`, `Config/` and `Supplementary_Data/`.

The repository uses the following project-relative paths for inputs, code and supporting results:

| Project-relative path | Description |
| --- | --- |
| `Code/` | Data preparation, event detection, DO reconstruction, statistical analysis and plotting scripts |
| `Data/Meteorology_data/` | Meteorological inputs for 1991–2020 and 2021–2025 |
| `Data/Meteorology_data_P/` | Prepared daily meteorological inputs |
| `Data/WaterQuality_data/` | Timestamped water-quality inputs for the 66 study sites |
| `Data/WaterQuality_data_P/` | Quality-controlled water-quality observations and prepared daily records |
| `Config/` | Station configurations and water-quality-site/hydrological-gauge matching information |
| `Supplementary_Data/` | Contains datasets S1–S10, with analysis outputs and supporting CSV tables corresponding to the separately submitted supplementary information |

**Data-access and reproducibility limits.** Original and completed daily hydrological observations—including discharge (`Q`), water level (`WL`) and suspended sediment concentration (`Ss`)—are not included in the public code package. Scripts that require such inputs remain documented, but a full end-to-end rerun additionally requires access to the appropriate hydrological data from the original provider. Certain steps also require event registries, GIS layers, saved model weights and scalers, or upstream analysis outputs. Some results and figures can be regenerated from the supplied non-restricted summary tables without rerunning earlier restricted-data steps. Availability of code does not grant permission to redistribute independently obtained third-party data.

**Publication files.** The manuscript and supplementary information, including descriptions of Supplementary Data S1–S10, are submitted separately through the journal submission system. They are not required to run the documented code.

## Reproducing supplementary Figures S23–S26

The supplementary water-quality background plots can be regenerated from five supplied S10 summary tables, using the figure script without original hydrological observations. From the repository root, first validate the available inputs:

```bash
python Code/Figure_V1/SFigure/SFigure23_26_WQ_Background/V_wq_background.py --check-inputs
```

Omit `--check-inputs` to redraw Figures S23–S26. Figure S23 uses the M1 summaries. The script writes PDF, SVG, TIFF and PNG files to `Result/Distributed_S10_Figures` by default; use `--output-dir` to select another output directory. Publication figures are included in the supplementary information submitted separately.

## Running the workflow

The core workflow uses the 11 November 2026 analysis batch. Use the `my_pytorch` environment described above and the [figure scripts](#figure-scripts) for plotting entry points.

Run scripts from the submission package root in dependency order. An arrow indicates which step runs next. A plus sign means that both inputs are required at that point. Independent branches can run in parallel. Figure numbers follow the current English manuscript.

If the archived Earth Engine extracts are available, start at step 02 with the corrected study-period data and the station files in `1991-2020/YR`. To regenerate a figure, use its plotting entry point with upstream results from the same analysis batch.

## Script inventory

The catalogue covers the main workflow, Figs. 1–6 and S1–S26, and the shared modules. The three analyses below use the 11 November 2026 batch.

| Analysis | Script, inputs and outputs |
| --- | --- |
| Table S2(a–c), effect of treating blank suspended sediment concentration (Ss) as zero | [audit_ss_zero_encoding.py](Code/01_Study_area_and_data_integration/07_Sediment_provenance/audit_ss_zero_encoding.py) reads the original hydrology workbooks, quality-controlled daily hydrology, the station mapping, BiLSTM inputs and SHAP event samples. Gauge-day provenance and three summaries are written to `Result/Sediment_provenance/20261111`. |
| Table S4(d–e), sensitivity to unknown discharge exposure | [unknown_q_sensitivity.py](Code/05_Event_comparisons_and_robustness/03_Q_exposure_sensitivity/unknown_q_sensitivity.py) uses Table S4(a–c) daily discharge provenance, completed daily data, reference-period air temperature, Data S8 daily DO and baseline event statistics. It writes the event screen, reference days, sample summary, six contrasts and fit diagnostics to `Result/Q_exposure_support/20261111`. After updating the discharge inputs, rerun the Table S4(a–c) export first. |
| Tables S28–S30, original DO coverage and observation-restricted samples | [observation_coverage_sensitivity.py](Code/05_Event_comparisons_and_robustness/06_Observation_coverage/observation_coverage_sensitivity.py) reads M07 and the post-event eligibility results. It writes the coverage thresholds, sample groups, 18 contrasts and fit diagnostics to `Result/Observation_coverage/20261111/S6` and `S7` and the station-block intervals for overall reference-direction disagreement to a CSV in the parent directory. |

All three scripts accept `--output-dir`. For Table S4(d–e), use `--q-dir` to point to the directory containing the daily discharge CSV exported for Table S4(a–c).

If you download reference-period temperatures again, the two Earth Engine scripts write to `1991-2020/M` and `1991-2020/B`. Check and combine their station files in `1991-2020/YR` before running event detection. The workflow for this batch starts from the supplied `YR` files. The separately submitted supplementary information describes the Supplementary Data CSVs.

## Manuscript and directory structure

The five numbered analysis directories follow Sections 2.1–2.5 of the manuscript. Directory 06 follows Supplementary Method S19. Follow the workflow order below when rebuilding the analysis from raw data.

| Methods section | Process directory | Main contents |
| --- | --- | --- |
| 2.1 Study area and multi-source data integration | [01_Study_area_and_data_integration](Code/01_Study_area_and_data_integration/) | Meteorology, water-quality and hydrology preparation, station matching, QC order and sediment provenance |
| 2.2 Extreme-event identification and mutually exclusive classification | [02_Extreme_event_identification](Code/02_Extreme_event_identification/) | Event registries from the observed and completed series |
| 2.3 Reconstruction of daily DO series | [03_Daily_DO_reconstruction](Code/03_Daily_DO_reconstruction/) | Features, BiLSTM training, descriptive summaries and continuous-gap validation |
| 2.4 Event-period DO conditions, relative departures and early post-event changes | [04_DO_conditions_and_post_event_changes](Code/04_DO_conditions_and_post_event_changes/) | Absolute DO and provenance, relative responses, paired references, early slopes and post-event eligibility |
| 2.5 Event-type comparisons and robustness analyses | [05_Event_comparisons_and_robustness](Code/05_Event_comparisons_and_robustness/) | Mixed-effects models, gauge clustering, exposure and spatial sensitivity, observational support, reference and definition sensitivity, restricted slope models and exploratory SHAP |
| Supplementary Method S19 | [06_WQ_Background_Analysis](Code/06_WQ_Background_Analysis/) | Original water-quality summaries and plot functions for Figs. S23–S26 |

| Results section | Analysis entries | Figure scripts |
| --- | --- | --- |
| 3.1 Extreme-event composition and original observation coverage | Event detection in 2.2 and coverage summaries in 2.4–2.5 | [Fig. S1](Code/Figure_V1/SFigure/SFigure01_Event_composition/) and [Fig. S2](Code/Figure_V1/SFigure/SFigure02_Event_frequency/) |
| 3.2 DO reconstruction coverage and event-period validation | Reconstruction and gap validation in 2.3 | [Fig. 2](Code/Figure_V1/Figure2/), [Figs. S6–S7](#figure-scripts) and [Fig. S8](Code/Figure_V1/SFigure/SFigure08_Additional_validation/) |
| 3.3 Absolute DO conditions across event types | Absolute DO and observation provenance in 2.4 | [Fig. 3a–c](Code/Figure_V1/Figure3/) and [Fig. S21](Code/Figure_V1/SFigure/SFigure21_Low_DO_event_occurrence/) |
| 3.4 Relative DO responses across event types and sequences | Relative departures in 2.4 and adjusted comparisons in 2.5 | [Fig. 3d](Code/Figure_V1/Figure3/d/), [Figs. S12–S13](#figure-scripts) |
| 3.5 Early post-event DO changes | Slopes and eligibility in 2.4 and restricted models in 2.5 | [Fig. 4](Code/Figure_V1/Figure4/), [Fig. S17](Code/Figure_V1/SFigure/SFigure17_Early_slope_maps/) and [Fig. S22](Code/Figure_V1/SFigure/SFigure22_Post_event_eligibility/) |
| 3.6 Reach-level DO responses and spatial heterogeneity | Reach summaries in 2.4–2.5 and [water-quality background analysis](Code/06_WQ_Background_Analysis/) for Supplementary Method S19 and Data S10 | [Fig. 5](Code/Figure_V1/Figure5/) and [Figs. S23–S26](Code/Figure_V1/SFigure/SFigure23_26_WQ_Background/V_wq_background.py) |
| 3.7 Sensitivity to reference conditions, observational support and event definitions | Paired references in 2.4 and robustness analyses in 2.5 | [Fig. 6](Code/Figure_V1/Figure6/), [Fig. S14](Code/Figure_V1/SFigure/SFigure14_Antecedent_sensitivity/) and [Fig. S16](Code/Figure_V1/SFigure/SFigure16_Spatial_independence/) |

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
11 compare_event_do_responses.py → Core_Contrasts.csv → Figs. 3d and 4d

Additional branches
  11 + completed DO and quality-controlled daily data → observation_only_references.py (Data S8)
  M04 + M05 + 11 + Data S8 inputs → antecedent reference sensitivity → Fig. S14
  M05 + 11 + Data S8 inputs → post-event eligibility → restricted slope contrasts → Figs. S22 and 4
  M07 + daily provenance and weather → Jiagaiqiao source audit → Fig. S18
  M07 + observed and completed daily series + event registry → real DO gaps and observation provenance → Table S11(d)
  05 + hydrology before and after interpolation → Table S4(a–c) CSV export
  11 + station mapping and completed daily data → shared-gauge clustering → Tables S22c and S23c
```

## 2.1 Study area and multi-source data integration

### 01a Main-stem Earth Engine extracts

[3.M_EVI_Land_Data_V1.py](Code/01_Study_area_and_data_integration/01_Meteorology_ecology/3.M_EVI_Land_Data_V1.py) uses `Config/Stations_M.csv` and the Earth Engine ERA5-Land, MODIS EVI and Dynamic World products. It aggregates instantaneous variables and hourly increments separately by Beijing-time calendar day, extracts EVI and land cover, and can recover station coordinates from the retained water-quality files for the same group.

The script writes station CSVs, including `SSR` and `TP_M`, to `Data/Meteorology_data/2021-2025/M_V1`. Table S1 describes the products and units. Run this step when the main-stem extracts need refreshing. Step 02 can validate the archived files directly.

### 01b Tributary Earth Engine extracts

[3.B_EVI_Land_Data_V1.py](Code/01_Study_area_and_data_integration/01_Meteorology_ecology/3.B_EVI_Land_Data_V1.py) applies the same daily aggregation and extraction to the stations in `Config/Stations_B.csv` and writes the station CSVs to `Data/Meteorology_data/2021-2025/B_V1`. Step 02 selects the 44 tributary and 22 main-stem stations in the study mapping. The same products and units apply to Table S1.

### 02 Assemble study-period weather files

[assemble_meteo.py](Code/01_Study_area_and_data_integration/01_Meteorology_ecology/assemble_meteo.py) validates the 66 stations across `M_V1` and `B_V1` and collects the daily files in `Data/Meteorology_data/2021-2025/YR_V1`.

### 02b Normalize precipitation

[normalize_precipitation.py](Code/01_Study_area_and_data_integration/01_Meteorology_ecology/normalize_precipitation.py) sets negative `TP` or `TP_M` values to zero in the 66 daily files from `Data/Meteorology_data/2021-2025/YR_V1` and writes to `Data/Meteorology_data_P`. The merge in step 05 reads that directory. After merging, `ED_N.py` prepares the SHAP event samples and `Lstm_Y.py` builds the DO reconstruction inputs, including three-day cumulative precipitation.

### 03 Screen and aggregate water-quality observations

[1.WQ_Data_P.py](Code/01_Study_area_and_data_integration/02_Water_quality_QC/1.WQ_Data_P.py) reads timestamp-level measurements from `Data/WaterQuality_data/WQ_data_YR` and matches daily surface pressure (`SP`) from `Data/Meteorology_data/2021-2025/YR_V1`. It screens DO using absolute and saturation limits, applies static bounds to other variables, aggregates to calendar days and screens stations for inclusion. The output in `Data/WaterQuality_data_P/WQ_data_YR` contains `Retained` and `Excluded` branches with `Raw` and `Cleaned` daily data. Missingness summaries are written to `Result/WQ_data_missing`. Tables S1 and S2 describe the variables and QC rules. The screened DO feeds the later analyses.

### Audit the water-quality screening order after step 03

[audit_qc_order_impact.py](Code/01_Study_area_and_data_integration/06_QC_order_audit/audit_qc_order_impact.py) compares timestamp-level retention, daily means, original-observation flags and downstream model and event inputs under the two QC orders. It uses the original DO and water temperature, the same current pressure data and the fixed station list and `Cleaned` daily files. The full audit also reads the current LSTM features and event data.

Run `--stage qc` after generating the QC daily tables. Run `--stage full` after step 07 and before model training. Both update `qc_order_impact.csv` and `changed_record_log.csv` in `Result/M2b_QC_Order`. The `Stage` field identifies the stage. Use the `SUMMARY` rows for Table S2 when the `RUN` record reports `COMPLETED`. The script records the completed run status and stage in its output files.

### 04 Screen and interpolate hydrology

[2.Hy_Data_P.py](Code/01_Study_area_and_data_integration/03_Hydrological_QC/2.Hy_Data_P.py) reads the 42 original workbooks in `Data/Hydrology_data/Hy_data_YR`, standardizes gauge names, rivers and dates, builds the daily calendar, applies the initial missing-Ss rule and interpolates water level (`WL`), discharge (`Q`) and Ss linearly. It writes uninterpolated and interpolated files to `Data/Hydrology_data_P/Hy_data_YR/Original` and `Interpolated` and missingness summaries to `Result/Hy_data_missing`.

These daily files support Tables S1–S4. Step 04 can run alongside the weather and water-quality steps. Separate scripts generate the Table S4 threshold and episode catalogue and audit the sediment provenance.

### 05 Match stations and merge daily data

[4.WQ_Hy_M_EVI_Land_Data.py](Code/01_Study_area_and_data_integration/04_Spatial_matching_and_merge/4.WQ_Hy_M_EVI_Land_Data.py) matches each screened water-quality station to its nearest gauge on the same river, using river names and coordinates. It merges the screened water-quality data, completed hydrology and daily weather from `Data/Meteorology_data_P`, writing meteorological `TP` as `TP_M`. It checks for a same-river candidate before writing results.

The output in `Data/WQ_Hy_M_DYN_data/YR` includes station-level, reach-level and basin-wide daily tables, plus `Match_Summary_Retained.csv` in the existing column order. Compare that summary with `Config/Study_Station_Mapping.csv`, the fixed mapping used for discharge provenance, formal event registration and clustered analysis. The station and river information supports Table S3. In this batch, Yingugongluqiao is matched to Qingtongxia at a distance of 63.40 km.

### Discharge provenance for Table S4

[export_table_s4_q_support.py](Code/05_Event_comparisons_and_robustness/03_Q_exposure_sensitivity/export_table_s4_q_support.py) uses the current mapping and daily hydrology before and after interpolation. It checks the gauge and discharge in the merged daily tables and calculates the S0 P95 threshold and consecutive high-flow episodes. It writes three CSVs to `Result/Q_exposure_support/Discharge_Provenance` by default. Use `--output-dir` to write elsewhere.

### Plotting after step 05

| Figure | Script | Inputs |
| --- | --- | --- |
| Study area and station locations | [V_Studyarea.py](Code/Figure_V1/Figure1/a/V_Studyarea.py) | Station matching summary from step 05 and a separately obtained basemap at `Data/GIS/1.gdb`. |
| Correlation matrix of environmental variables | [V_P.py](Code/Figure_V1/SFigure/SFigure09_Correlation/V_P.py) | Merged daily station files. |

## 2.2 Extreme-event identification and mutually exclusive classification

Use the archived daily reference temperatures in `Data/Meteorology_data/1991-2020/YR`. To refresh them, run the two download scripts in parallel, validate the station files and collect the study stations in `YR` before running `ED_N.py`. The precipitation and SSR corrections do not require a new reference-temperature download.

| Optional script | Station list | Output |
| --- | --- | --- |
| [3.M_1991-2020T.py](Code/01_Study_area_and_data_integration/05_Reference_temperature_download/3.M_1991-2020T.py) | `Config/Stations_M.csv` | `Data/Meteorology_data/1991-2020/M` |
| [3.B_1991-2020T.py](Code/01_Study_area_and_data_integration/05_Reference_temperature_download/3.B_1991-2020T.py) | `Config/Stations_B.csv` | `Data/Meteorology_data/1991-2020/B` |

Both scripts download the daily 1991–2020 temperature series through Earth Engine. The data support the thresholds in Tables S1, S5 and S6. The B script uses the tributary configuration and output directory for this batch.

### 06 Detect and label the observed events

[ED_N.py](Code/02_Extreme_event_identification/01_Event_registry/ED_N.py) uses the YR reference temperatures and daily records in `Data/WQ_Hy_M_DYN_data/YR/by_station_all` to identify five event types. It labels individual days, summarizes original observation coverage, and separates complete events from those with missing data.

Its outputs under `Data/Event_N` include `By_Site_Labeled`, `Perfect_Events_Data`, `Missing_Events_Data` and the event counts. These support Table S5 and Figs. S1–S2 and provide inputs for reconstruction and the observation-based SHAP analysis. [V_event1.py](Code/Figure_V1/SFigure/SFigure02_Event_frequency/V_event1.py) plots event counts by station.

### 09 Register events on the completed daily series

[ED_F.py](Code/02_Extreme_event_identification/02_Completed_event_registry/ED_F.py) uses the YR reference temperatures and `Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR` to identify the five event types. It writes the event registry, antecedent references, event-period records, post-event windows and non-event days to `Data/Event_F/01_Registry_Data` through `05_Non_Event_Data`. It also writes `Data/Event_F/Event_Registry_For_Validation.csv`. The registry and windows support Table S5 and the response calculations for Table S12.

After step 09, the validation branches in Section 2.3 can run alongside step 10. Their two diagnostic plotting scripts export 600 DPI TIFF files.

## 2.3 Reconstruction of daily DO series

### 07 Prepare model features

[Lstm_Y.py](Code/03_Daily_DO_reconstruction/01_Input_features/Lstm_Y.py) reads `Data/Event_N/By_Site_Labeled` and writes `Data/LSTM_Y`. It keeps the DO target and adds calendar and reach codes and rolling features, including `TP_M_sum3` and `Q_mean7`. Regenerate these features after changing the precipitation inputs. Table S9 describes the encodings, and Tables S10–S11 use the resulting workflow.

### Run the full screening-order audit after step 07

Run [audit_qc_order_impact.py](Code/01_Study_area_and_data_integration/06_QC_order_audit/audit_qc_order_impact.py) with the observed-event file from step 06. This audit can run before model training.

```powershell
python "Code/01_Study_area_and_data_integration/06_QC_order_audit/audit_qc_order_impact.py" --stage full --events-file "Data/Event_N/All_Events_Missing_Status.csv"
```

### 08 Train the model and reconstruct daily DO

[LSTM_V1.py](Code/03_Daily_DO_reconstruction/02_Model_training/LSTM_V1.py) takes 18 features and the original DO target from `Data/LSTM_Y`. It trains and applies the BiLSTM-Attention model and records its final weights, fitted scalers, feature order, settings and input hashes.

The output in `Result/BiLstm_Att_V1` includes `Imputed_Data`, `Reports/Loss_History`, `Reports/By_Site_Scatter` and `Saved_Models_V1`. The script validates the 66 reconstructed station tables and collects them in `Imputed_Data/By_Site_Imputed/YR` for `ED_F.py`. Table S9 draws on the model configuration, and Table S10 uses the pointwise evaluation. Generate the daily summaries separately for Tables S7 and S8. These files also support Fig. 2 and Figs. S5–S7.

### Summaries and plotting after step 08

[data_N.py](Code/03_Daily_DO_reconstruction/03_Descriptive_summaries/data_N.py) summarizes the merged station files before reconstruction in `Result/Mean_N_data`. [data_F.py](Code/03_Daily_DO_reconstruction/03_Descriptive_summaries/data_F.py) summarizes the completed station files in `Result/Mean_F_data`. These summaries support Tables S7–S8 and Fig. S10. Regenerate both summaries when switching analysis batches.

| Figure | Script | Inputs |
| --- | --- | --- |
| Training loss, performance maps and station time series | [V_Loss.py](Code/Figure_V1/SFigure/SFigure05_Training_loss/V_Loss.py), [V_NSE.py](Code/Figure_V1/SFigure/SFigure06_Performance_maps/V_NSE.py), [V_space1.py](Code/Figure_V1/SFigure/SFigure07_Full_station_series/V_space1.py) | Training, evaluation and reconstruction outputs from `LSTM_V1.py`. The maps also use the basemap. |
| Monthly DO before and after reconstruction | [V_DO_time1.py](Code/Figure_V1/SFigure/SFigure10_Monthly_DO/V_DO_time1.py) | Monthly station-level and basin-wide summaries under `Result/Mean_N_data/DO` and `Result/Mean_F_data/DO`. |

### Pointwise reconstruction diagnostics

[V_do_reconstruction_validation.py](Code/Figure_V1/SFigure/SFigure08_Additional_validation/V_do_reconstruction_validation.py) reads saved test predictions in `Reports/By_Site_Scatter`, observed daily DO in `Data/LSTM_Y` and `Event_Registry_For_Validation.csv`. It summarizes pointwise errors, missed low-DO detections and observed gap lengths. It writes to `V/Figure_V1/SFigure/SFigure08_Additional_validation/Images` by default.

Change the 5 mg/L low-DO threshold with `--threshold`. Pass `--error-summary-csv`, `--false-negatives-csv` or `--gap-distribution-csv` to use summaries from the same batch.

### Continuous-gap validation

[validate_continuous_do_gaps.py](Code/03_Daily_DO_reconstruction/04_Continuous_gap_validation/validate_continuous_do_gaps.py) reads `DO_Model_Manifest_V1.json`, the matching weights, fitted scalers, LSTM inputs, saved test predictions and `Data/Event_F/Event_Registry_For_Validation.csv`. It masks 3-, 7-, 14- and 30-day windows with original DO observations in 2025 and compares masked predictions, unmasked predictions and the last observation carried forward. DO is hidden before constructing lagged features. Inference uses the saved model directly and does not feed predictions back into subsequent windows.

Each run under `Result/Continuous_gap_validation` writes 11 CSVs covering candidate windows, pointwise predictions, errors and low-DO classification. The terminal reports `COMPLETED` on success and the run writes `error_traceback.txt` on failure. Plotting scripts check for a complete set of result tables. The outputs support Tables S10–S11 and Figs. 2b and S8.

### Plot continuous-gap validation

[V_Continuous_gap_validation.py](Code/Figure_V1/SFigure/SFigure08_Additional_validation/V_Continuous_gap_validation.py) selects the latest completed run by default. Use `--validation-dir` to select a run and `--metrics-csv`, `--intervals-csv`, `--paired-csv` or `--low-do-csv` to select individual tables. Gap lengths and low-DO thresholds come from the result tables.

Pointwise errors come from the test predictions by default. Use `--pointwise-csv` to supply a saved summary. Both validation plotting scripts accept `--scatter-dir`, `--daily-dir`, `--registry` and `--test-year` for another analysis batch. For the station example, use `--site` or `--site-scatter-csv`. The default destination is the validation-run subdirectory under `V/Figure_V1/SFigure/SFigure08_Additional_validation/Images`.

## 2.4 Event-period DO conditions, relative departures and early post-event changes

### Absolute DO and observation provenance

[M07_absolute_DO.py](Code/04_DO_conditions_and_post_event_changes/04_Absolute_DO_and_provenance/M07_absolute_DO.py) uses the `Event_F` registry, completed V1 DO, `Cleaned` observed DO and the Shock ratios for this batch. It computes event means and minimum daily DO, event and day proportions below 5, 3 and 2 mg/L, and summaries by event type and reach. It tracks the observation or reconstruction source and whether most values were reconstructed.

The dated folder under `Result/Decrease/DO_observation_sensitivity` contains event-level absolute DO, summaries by type and reach, `Observed_Imputed`, `Shock_DO_Relationship` and `Below2_Day_Trace` tables. These support Tables S17–S20, the coverage work for Table S28 and Figs. S15 and S21. Use the dedicated site audit and coverage model scripts for Tables S27–S30.

### 10a Calculate responses relative to antecedent and seasonal references

[decrease.py](Code/04_DO_conditions_and_post_event_changes/02_Departure_metrics/decrease.py) reads the event registry, antecedent windows, event-period records and non-event days under `Data/Event_F`. It calculates event DO, seasonal and antecedent references, and the two response ratios `Shock-L` and `Shock-S`. It writes event and station summaries under `Result/Decrease/By_Site` and `Result/Decrease/Overall`, including `FINAL_Shock_All_Events_Combined.csv`.

Use these outputs for Table S12, Figs. 3, 5 and 6 and the maps and distributions in Figs. S12–S13. The MixedLM and analyses M04, M05, M06 and M07 also use the response outputs from this batch.

### 10b Calculate early post-event slopes

[recovery.py](Code/04_DO_conditions_and_post_event_changes/03_Early_post_event_slopes/recovery.py) reads the event registry and antecedent reference and post-event windows from `Data/Event_F`. It fits the early linear slope using DO on days 1, 3, 5, 7 and 9 after the event and records eligibility and the diagnostic time to first reach the antecedent reference. It writes event and station summaries under `Result/Recovery`, including `FINAL_Recovery_All_Events_Combined.csv`.

The slope results support Table S12 and Fig. 4. Table S21 uses the separate post-event eligibility audit listed below. A positive slope indicates rising DO during the window. The script reports return-to-reference time separately.

### Plot the response metrics

| Figure | Scripts | Inputs |
| --- | --- | --- |
| Shock-L and Shock-S maps and distributions | [V_de_space.py](Code/Figure_V1/SFigure/SFigure12_Seasonal_departure/V_de_space.py), [V_de_space1.py](Code/Figure_V1/SFigure/SFigure13_Antecedent_departure/V_de_space1.py), [V_de_value.py](Code/Figure_V1/SFigure/SFigure12_Seasonal_departure/V_de_value.py), [V_de_value1.py](Code/Figure_V1/SFigure/SFigure13_Antecedent_departure/V_de_value1.py) | Station and event outputs from `decrease.py`. The maps also use the basemap. |
| Early post-event slope maps and distributions | [V_re_space.py](Code/Figure_V1/SFigure/SFigure17_Early_slope_maps/V_re_space.py), [V_re_value.py](Code/Figure_V1/SFigure/SFigure17_Early_slope_maps/V_re_value.py) | Station and event outputs from `recovery.py`. The map also uses the basemap. |

### Paired references

[M04_paired_baselines.py](Code/04_DO_conditions_and_post_event_changes/06_Paired_references/M04_paired_baselines.py) joins the `Data/Event_F` registry with the two Shock ratios in `Result/Decrease/Overall`. It checks common events with valid ratios under both references, classifies direction and summarizes agreement by stratum with station-block bootstrap intervals. The dated directory under `Result/Decrease/Paired_baselines` contains the common sample, classifications, differences and intervals in CSV and Excel. These support Tables S13–S15 and Fig. S11. Plot labels use event and station counts from the current results.

## 2.5 Event-type comparisons and robustness analyses

### Main adjusted comparisons

[compare_event_do_responses.py](Code/05_Event_comparisons_and_robustness/01_Mixed_effects/compare_event_do_responses.py) combines the `Event_F` registry with ratios from `Result/Decrease` and slopes from `Result/Recovery`. For each response it fits a site-random-intercept mixed-effects model to the eligible events, adjusts for reach and season, and returns prespecified contrasts, confidence intervals and Holm-adjusted p values.

Outputs in `Result/Statistical_comparisons` include `Event_Level_Statistical_Data.csv`, `Core_Contrasts.csv`, `Model_Diagnostics.csv` and `Statistical_Analysis_Results.xlsx`. These support Tables S22–S23. The Shock-L and Shock-S contrasts are used in V3 Fig. 3d, and the `Recovery_Slope` contrasts for all eligible events are used in Fig. 4d. Use the observation-coverage branch for the restricted-sample models in Table S30.

### Shared-gauge clustering

[M5_hydro_cluster_sensitivity.py](Code/05_Event_comparisons_and_robustness/02_Hydro_cluster_sensitivity/M5_hydro_cluster_sensitivity.py) fits two OLS specifications for each of three outcomes, one with five event types and one with a compound-event indicator. Both include water-quality-station fixed effects and onset season. CR1 covariance is clustered by matched discharge gauge.

The analysis uses the baseline event-level statistics, six main MixedLM contrasts and fixed mapping. It checks `Matched_HS_Name` against the completed series. Finite positive Shock-L and Shock-S values are logged, retaining negative and zero log values. Slopes remain on the original scale. The analysis fits six models with the same events and DO values and uses `t(G−1)` as the reference approximation. Holm adjustment covers the two tests within each response, and 95% confidence intervals are unadjusted.

Results under `Result/Statistical_comparisons/Hydro_cluster/20261111` include `hydro_cluster_contrasts.csv`, `model_support_counts.csv`, event IDs, mapping checks, diagnostics, the software version and covariance options. They support Tables S22(c) and S23(c) and Supplementary Method S13.

### Spatial independence diagnostics

[M06_spatial_independence.py](Code/05_Event_comparisons_and_robustness/04_Spatial_independence/M06_spatial_independence.py) checks the existing station mapping, shared discharge inputs and discharge coverage. It summarizes event and site counts and calculates descriptive downstream leave-one-site-out (LOSO) sensitivity for Shock-L. It uses the mapping and reach tables in `Data/WQ_Hy_M_DYN_data/YR`, the `Event_F` registry, `Result/Decrease` and original and completed discharge.

The dated directory under `Result/Data_preprocessing/Hydrological_matching` contains `Hydro_Matching`, `Event_Site_Counts`, `Downstream_LOSO` and `Type_LOSO` in CSV or XLSX. These support the mapping in Table S3, the descriptive results in Table S24 and Fig. S16. The analysis documents station matching and sample coverage. Hydrological connectivity is outside its scope.

### Event-definition sensitivity

[M05_sensitivity.py](Code/05_Event_comparisons_and_robustness/08_Definition_sensitivity/M05_sensitivity.py) reuses `ED_F.py` and `decrease.py` with the reference temperatures, completed V1 DO, current `Event_F` registry and Shock results. It verifies the current S0 baseline before assessing alternative exposure definitions, thresholds and linking rules. It writes S0–S5 event lists and thresholds, Shock comparisons, summaries and notes to a dated folder under `Result/Event/Definition_sensitivity`.

These results support Table S6 and Figs. S3–S4. Run [V_definition_counts.py](Code/Figure_V1/SFigure/SFigure03_Definition_counts/V_definition_counts.py) and [V_definition_responses.py](Code/Figure_V1/SFigure/SFigure04_Definition_responses/V_definition_responses.py) after the analysis, using `--input-dir` for the M05 results directory.

### Pairing priority sensitivity

[M04_pairing_priority.py](Code/05_Event_comparisons_and_robustness/09_Pairing_priority/M04_pairing_priority.py) compares the baseline order DC → F5H → H5F with DC → H5F → F5H. It reuses the `ED_F.py` classifier and compares events using stable sets of constituent episode IDs. It holds thresholds, heatwave merging, direct-overlap components and tie-breaking fixed.

Inputs are completed DO with the `Original_DO` flag, the S0 registry, historical `Tmax`, baseline event metrics and the statistical script. The companion [ED_F_pre_M4.py](Code/05_Event_comparisons_and_robustness/09_Pairing_priority/ED_F_pre_M4.py) verifies the earlier classification. Use `--audit-only` for the classification check. The full run recalculates metrics when events change, including references, non-event dates, next-event cutoffs and early slopes. It writes `pairing_priority_comparison.csv`, `priority_core_results.csv` and model diagnostics under `Result/Event/Pairing_priority/20261111`. It keeps the two configurations separate and uses the existing completed DO and main results as inputs. See Table S6(d–f) and Supplementary Methods S3–S4.

### Tree-model and SHAP analysis

The five event-type scripts can run in parallel after step 06. Each fits random forest, XGBoost and Extra Trees to DO and 16 predictors, uses Bayesian search to select the model for that event type, and exports feature- and group-level SHAP contributions. All inputs are under `Data/Event_N/Perfect_Events_Data/Overall`. Use the corrected `TP_M` and `SSR` in the current batch.

| Event type | Script | Input CSV | Output prefix under `Result/shap` |
| --- | --- | --- | --- |
| Heatwave only | [analyze_H_shap.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_H_shap.py) | `Heatwave_Only_Perfect_Overall.csv` | `Heatwave_ThreeModel_SHAP_Analysis` |
| High flow only | [analyze_F_shap.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_F_shap.py) | `Flood_Only_Perfect_Overall.csv` | `Flood_ThreeModel_SHAP_Analysis` |
| Direct overlap | [analyze_DC_shap.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_DC_shap.py) | `Direct_Compound_Perfect_Overall.csv` | `DC_ThreeModel_SHAP_Analysis` |
| Heatwave followed by high flow | [analyze_H5F_shap.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_H5F_shap.py) | `H5F_Compound_Perfect_Overall.csv` | `H5F_ThreeModel_SHAP_Analysis` |
| High flow followed by heatwave | [analyze_F5H_shap.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_F5H_shap.py) | `F5H_Compound_Perfect_Overall.csv` | `F5H_ThreeModel_SHAP_Analysis` |

The scripts also save individual predictions and SHAP values for Figs. S19 and S20. The search space is documented in Table S32, model performance and parameters in Table S33, and contributions in Table S35. SHAP contributions describe attribution in the fitted concentration prediction models.

### Grouped prediction validation

[shap_group_validation.py](Code/05_Event_comparisons_and_robustness/12_Grouped_prediction_validation/shap_group_validation.py) uses the five observed-event datasets and the saved hyperparameters and model comparisons in `Result/shap`. It refits the three tree models for each event type with fixed hyperparameters and holds out complete site–event groups. It writes evaluation metrics, individual predictions, split membership and the environment record to `Result/SHAP_Grouped_Validation`. These are the inputs for Table S34. Figs. S19 and S20 use the saved plotting data from the tree-model scripts.

### Audit the saved SHAP contributions

[audit_shap_group_contributions.py](Code/05_Event_comparisons_and_robustness/13_SHAP_contribution_audit/audit_shap_group_contributions.py) checks the five model configurations against their observed samples and saved `Feature_Importance`, `Group_Contribution`, `Log` and `Model_Comparison` outputs. It verifies sums, mean contribution per feature within each group, and event and site coverage.

Use `Result/Statistical_comparisons/SHAP_group_descriptive_audit/SHAP_Group_Descriptive_Audit.csv` for the descriptive summaries in Table S35(b–d). The script reports audit status and provenance in the terminal and uses the saved SHAP values. It performs no model fitting or inference.

## Inputs for dedicated analyses and figures

Use results from the same batch. Several scripts have fixed input and output paths for 11 November 2026. Check these before running a different batch, and work from a copy if you need to preserve existing outputs.

| Analysis and manuscript location | Script | Results and inputs |
| --- | --- | --- |
| Observation-only dual references and daily DO provenance, Table S31 and Data S8 | [observation_only_references.py](Code/05_Event_comparisons_and_robustness/07_Reference_conditions/observation_only_references.py) | `Result/Observation_only_references/20261111`. Uses baseline event statistics, completed DO and original and quality-controlled daily means. |
| Antecedent windows and minimum observation requirements, Table S16, Data S3 and Fig. S14 | [antecedent_reference_sensitivity.py](Code/05_Event_comparisons_and_robustness/07_Reference_conditions/antecedent_reference_sensitivity.py) | `Result/Antecedent_reference_sensitivity/20261111_Analysis`. Uses M04, M05, the main statistics and observation-only results. Use `--output-dir` for the analysis and the same directory as `--run-dir` in the plot. |
| Slope eligibility and restricted slope models, Tables S21 and S30, Data S4, Figs. S22 and 4 | [post_event_eligibility.py](Code/04_DO_conditions_and_post_event_changes/05_Post_event_eligibility/post_event_eligibility.py), then [restricted_slope_contrasts.py](Code/05_Event_comparisons_and_robustness/10_Restricted_slope_models/restricted_slope_contrasts.py) | `Result/Post_event_eligibility/20261111_Analysis`. Uses main statistics, M05 thresholds, the event registry and daily DO provenance. Run the eligibility check with `--output-dir` set to a new directory. The restricted model script reads the archived path by default. Match the paths before using a different batch. |
| Diagnostic site exclusion and Jiagaiqiao provenance, Table S25, Audit Note S26, Table S27, Data S5 and Fig. S18 | [jiagaiqiao_source_audit.py](Code/05_Event_comparisons_and_robustness/05_Site_source_audit/jiagaiqiao_source_audit.py) | `Result/Jiagaiqiao_source_audit/20261111_Audit`. Uses original timestamps, original and screened daily means, completed DO, environmental data and event records. Use `--output-dir` for a new analysis. Site-exclusion summaries come from `existing_site_exclusion_sensitivity.csv`. |
| Real DO gaps and low-DO provenance, Table S11(d), Data S2 and Fig. S8 | [real_do_gap_support.py](Code/03_Daily_DO_reconstruction/04_Continuous_gap_validation/real_do_gap_support.py) | `Result/Continuous_gap_validation/Real_DO_gap_support`. Uses M07, the baseline registry and screened and reconstructed daily tables. It also writes `source_gap_distribution.csv` for Fig. S8. Use `--source-gap-csv` to specify another destination for that CSV. |

## Baseline source and shared modules

### LSTM baseline

[LSTM.py](Code/03_Daily_DO_reconstruction/02_Model_training/LSTM.py) records the baseline BiLSTM-Attention workflow behind the configuration in Table S9. It trains on 2021–2023, validates in 2024, tests in 2025 and refits on 2021–2024 using the selected number of epochs before filling missing DO. It reads `Data/LSTM_Y` and writes to `Result/BiLstm_Att_V1` when executed. The V1 model manifest records this baseline source. Run `LSTM_V1.py` as the reconstruction entry point for this batch.

### English table output

[english_table_output.py](Code/05_Event_comparisons_and_robustness/00_Shared_utilities/english_table_output.py) standardizes English headings, categories and place names and serializes publication CSVs. It flags unmapped Chinese text and can escape source paths reversibly. The caller passes a DataFrame or CSV bytes and saves the returned table. M01 and other export scripts use this helper after calculating the statistics.

### Reconstruction diagnostic inputs

[reconstruction_diagnostic_inputs.py](Code/03_Daily_DO_reconstruction/04_Continuous_gap_validation/reconstruction_diagnostic_inputs.py) supports both DO diagnostic figures. It aligns V1 station-level test predictions with observed dates, uses the event registry to classify test dates and summarizes real gap lengths.

The SHAP workflow uses [analysis_config.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analysis_config.py) for shared settings and [shap_plot_data.py](Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/shap_plot_data.py) to assemble and save predictions and SHAP plotting data.

## Figure scripts

Figure numbers follow the manuscript and supplementary information. The Fig. 1–6 entry points export individual panels. Use the panel order in the figure captions when assembling the manuscript composites.

| Main figure | Content | Script |
| --- | --- | --- |
| Fig. 1a | Study area, 66 water-quality stations and 42 matched gauges | [V_Studyarea.py](Code/Figure_V1/Figure1/a/V_Studyarea.py) |
| Fig. 1b | Update the workflow counts from the event table and export PNG and TIFF | [refresh_workflow_counts.py](Code/Figure_V1/Figure1/b/refresh_workflow_counts.py), [V_export_Figure1b.mjs](Code/Figure_V1/Figure1/b/V_export_Figure1b.mjs) |
| Fig. 2a–c | Pointwise validation, continuous-gap validation and low-DO recall | [V_pointwise_validation.py](Code/Figure_V1/Figure2/a/V_pointwise_validation.py), [V_continuous_gap_validation.py](Code/Figure_V1/Figure2/b/V_continuous_gap_validation.py), [V_low_DO_recall.py](Code/Figure_V1/Figure2/c/V_low_DO_recall.py) |
| Fig. 3a–d | Event-mean DO, proportion of low-DO days, observation provenance and adjusted contrasts | [V_event_mean_DO.py](Code/Figure_V1/Figure3/a/V_event_mean_DO.py), [V_low_day_fraction.py](Code/Figure_V1/Figure3/b/V_low_day_fraction.py), [V_low_DO_sources.py](Code/Figure_V1/Figure3/c/V_low_DO_sources.py), [V_adjusted_contrasts.py](Code/Figure_V1/Figure3/d/V_adjusted_contrasts.py) |
| Fig. 4a–d | Early slopes, window eligibility, observation coverage and adjusted contrasts | [V_early_slope_distribution.py](Code/Figure_V1/Figure4/a/V_early_slope_distribution.py), [V_window_eligibility.py](Code/Figure_V1/Figure4/b/V_window_eligibility.py), [V_post_event_source_support.py](Code/Figure_V1/Figure4/c/V_post_event_source_support.py), [V_adjusted_slope_contrasts.py](Code/Figure_V1/Figure4/d/V_adjusted_slope_contrasts.py) |
| Fig. 5a–c | Samples under each definition, common events and daily DO at Jiagaiqiao | [V_eligible_populations.py](Code/Figure_V1/Figure5/a/V_eligible_populations.py), [V_common_events.py](Code/Figure_V1/Figure5/b/V_common_events.py), [V_Jiagaiqiao_daily_DO.py](Code/Figure_V1/Figure5/c/V_Jiagaiqiao_daily_DO.py) |
| Fig. 6a–b | Paired reference ratios and classification sensitivity | [V_paired_reference_ratios.py](Code/Figure_V1/Figure6/a/V_paired_reference_ratios.py), [V_classification_sensitivity.py](Code/Figure_V1/Figure6/b/V_classification_sensitivity.py) |

For Fig. 1b, run `refresh_workflow_counts.py` before exporting the diagram. It fills the [PowerPoint template](Code/Figure_V1/Figure1/b/Assets/Figure1b_template.pptx) using the supplied S8 event table or a saved analysis table. To choose an input or output explicitly, use `--events-csv` and `--output-pptx`. The Python step uses `defusedxml`. Run `V_export_Figure1b.mjs` on the updated PPTX with `--input-pptx`. The Node.js step uses `@oai/artifact-tool` and `sharp`. Use `--node-modules` to point to an installation of these packages.

| Supplementary figure | Output or scope | Script |
| --- | --- | --- |
| Fig. S1 | Site–event composition and original water-quality observation completeness. | [V_event_composition.py](Code/Figure_V1/SFigure/SFigure01_Event_composition/V_event_composition.py) |
| Fig. S2 | Event frequency by site. | [V_event1.py](Code/Figure_V1/SFigure/SFigure02_Event_frequency/V_event1.py) |
| Fig. S3 | Exposure definitions and event counts. | [V_definition_counts.py](Code/Figure_V1/SFigure/SFigure03_Definition_counts/V_definition_counts.py) |
| Fig. S4 | Two parts showing compound–single and F5H–H5F Shock-L differences. | [V_definition_responses.py](Code/Figure_V1/SFigure/SFigure04_Definition_responses/V_definition_responses.py) |
| Fig. S5 | BiLSTM-Attention development loss. | [V_Loss.py](Code/Figure_V1/SFigure/SFigure05_Training_loss/V_Loss.py) |
| Fig. S6 | Two maps showing site-level MAE and NSE. | [V_NSE.py](Code/Figure_V1/SFigure/SFigure06_Performance_maps/V_NSE.py) |
| Fig. S7 | Five groups of station time series, exported as montages. | [V_space1.py](Code/Figure_V1/SFigure/SFigure07_Full_station_series/V_space1.py) |
| Fig. S8, part 1 | Additional reconstruction diagnostics. | [V_do_reconstruction_validation.py](Code/Figure_V1/SFigure/SFigure08_Additional_validation/V_do_reconstruction_validation.py) |
| Fig. S8, part 2 | Same-date paired baseline comparisons. | [V_same_date_baseline_comparison.py](Code/Figure_V1/SFigure/SFigure08_Additional_validation/V_same_date_baseline_comparison.py) |
| Fig. S9 | Pearson correlations among variables. | [V_P.py](Code/Figure_V1/SFigure/SFigure09_Correlation/V_P.py) |
| Fig. S10 | Part 1 shows the long-term DO map. Part 2 shows monthly comparisons. | [V_station_mean_DO_map.py](Code/Figure_V1/SFigure/SFigure10_Monthly_DO/V_station_mean_DO_map.py), [V_DO_time1.py](Code/Figure_V1/SFigure/SFigure10_Monthly_DO/V_DO_time1.py) |
| Fig. S11 | Paired-reference discordance by event type. | [V_direction_by_type.py](Code/Figure_V1/SFigure/SFigure11_Direction_by_type/V_direction_by_type.py) |
| Fig. S12 | Parts 1–5 show seasonal-reference maps. Part 6 shows the event-level distribution. | [V_de_space.py](Code/Figure_V1/SFigure/SFigure12_Seasonal_departure/V_de_space.py), [V_de_value.py](Code/Figure_V1/SFigure/SFigure12_Seasonal_departure/V_de_value.py) |
| Fig. S13 | Parts 1–5 show antecedent-reference maps. Part 6 shows the event-level distribution. | [V_de_space1.py](Code/Figure_V1/SFigure/SFigure13_Antecedent_departure/V_de_space1.py), [V_de_value1.py](Code/Figure_V1/SFigure/SFigure13_Antecedent_departure/V_de_value1.py) |
| Fig. S14 | Antecedent-reference sensitivity across observation requirements and windows. | [V_antecedent_reference_sensitivity.py](Code/Figure_V1/SFigure/SFigure14_Antecedent_sensitivity/V_antecedent_reference_sensitivity.py) |
| Fig. S15 | Shock-L and minimum daily mean DO. | [V_shock_minimum_DO.py](Code/Figure_V1/SFigure/SFigure15_Absolute_DO_relationship/V_shock_minimum_DO.py) |
| Fig. S16 | Downstream leave-one-site-out sensitivity. | [V_downstream_LOSO.py](Code/Figure_V1/SFigure/SFigure16_Spatial_independence/V_downstream_LOSO.py) |
| Fig. S17 | Five maps of early post-event DO slopes. | [V_re_space.py](Code/Figure_V1/SFigure/SFigure17_Early_slope_maps/V_re_space.py) |
| Fig. S18 | Part 1 shows the provenance of event-day DO records below 2 mg L⁻¹ across the basin. Part 2 shows descriptive associations between original DO and same-day water-quality observations at Jiagaiqiao. | [V_jiagaiqiao_source_audit.py](Code/Figure_V1/SFigure/SFigure18_Source_audit/V_jiagaiqiao_source_audit.py) |
| Fig. S19 | Five event-type model performance plots. | [V_selected_model_fit.py](Code/Figure_V1/SFigure/SFigure19_Selected_model_fit/V_selected_model_fit.py) |
| Fig. S20 | Five event-type SHAP attribution plots. | [V_shap_explanations.py](Code/Figure_V1/SFigure/SFigure20_SHAP_exploratory/V_shap_explanations.py) |
| Fig. S21 | Events with at least one day below 5 mg L⁻¹ by type. | [V_low_DO_event_occurrence.py](Code/Figure_V1/SFigure/SFigure21_Low_DO_event_occurrence/V_low_DO_event_occurrence.py) |
| Fig. S22 | Slope eligibility by event type and three within-type comparisons. | [V_post_event_eligibility.py](Code/Figure_V1/SFigure/SFigure22_Post_event_eligibility/V_post_event_eligibility.py) |
| Figs. S23–S26 | Long-term M1 water-quality background (S23), matched event–non-event changes (S24), concurrent water quality by DO group across reaches (S25) and downstream (S26). | [V_wq_background.py](Code/Figure_V1/SFigure/SFigure23_26_WQ_Background/V_wq_background.py), using [02_WQ_Background_Plot.py](Code/06_WQ_Background_Analysis/02_WQ_Background_Plot.py) |

The [continuous-gap plotting utility](Code/Figure_V1/SFigure/SFigure08_Additional_validation/V_Continuous_gap_validation.py) also exports the Fig. 2 source tables and a combined validation overview. The [slope distribution utility](Code/Figure_V1/SFigure/SFigure17_Early_slope_maps/V_re_value.py) provides an additional distribution plot. Fig. 4a has its own panel script.

For Figs. S11, S15 and S16, use `--input-dir` to select the M04, M07 or M06 results. Without that option, the scripts look for saved results. For Figs. S14 and S22, `--run-dir` selects a completed analysis directory with its run metadata. The maps in Figs. 1a, S6, S10 (part 1), S12, S13 and S17 require the separate GIS basemap.

The scripts for multi-panel figures include layout checks where implemented. The Fig. S22 script also exports a panel alignment diagnostic SVG.

Figs. S19 and S20 read `*_Plot_Data.npz` files under `Result/shap`. The five `analyze_*_shap.py` scripts save these files alongside their CSV outputs. For an existing analysis batch, use `--prepare-saved-figures` with the corresponding script to refit the selected model at the saved settings and check the saved prediction metrics and SHAP percentages. Use `--event` to select the event type in the plotting scripts.

Station time series and montages for Fig. S7 are exported by `V_space1.py`. The Fig. 2 plotting scripts read prepared source tables. `V_Continuous_gap_validation.py` writes these tables under `V/Figure_V1/SFigure/SFigure08_Additional_validation/Images`.


### Plotting from the supplied tables

| Figure | Input arguments |
| --- | --- |
| Fig. 2a | Use `--source-dir Supplementary_Data/S2`. |
| Figs. 5a–c | Set `--events-csv` and `--summary-csv` to `event_observation_verification.csv` and `observation_only_dual_reference_sensitivity.csv` in `Supplementary_Data/S8/`. Set `--daily-csv` to `Supplementary_Data/S5/daily_site_audit.csv`. |
| Figs. 6a–b | Use the same `--events-csv` and `--summary-csv` arguments as for Fig. 5. |
| Fig. S8, part 2 | Use `--input-csv Supplementary_Data/S2/paired_baseline_comparison.csv`. |
| Fig. S18 | Use `--run-dir Supplementary_Data/S5`. The script reads `daily_site_audit.csv`, `timestamp_source_audit.csv`, `basin_low_DO_source_counts.csv` and `review_blocks.csv`. Use `Comment4_low_DO_sources` and `Comment4_observed_chemistry` for parts 1 and 2, respectively. The same run also exports the audit time series and review blocks. |
| Figs. S23–S26 | Follow [the S10 plotting instructions](#reproducing-supplementary-figures-s23s26). |

Use `--output-dir` to select a separate destination for these figure exports.

## Output naming conventions

Gap-validation tables use `Masked_BiLSTM_Attention` and `Unmasked_BiLSTM_Attention`. The `validation_labels.py` utility normalizes legacy output labels when loading earlier results; the meaning of the modeled gap intervals is unchanged.

## Supplementary data organization and access

Supplementary Data S10 contains 26 statistical CSVs directly under its directory, without subfolders. The figure-recreation procedure is described in [Reproducing supplementary Figures S23–S26](#reproducing-supplementary-figures-s23s26). Selected Table S4 summary and sensitivity CSVs are located in `Supplementary_Data/S4`. Gauge-cluster CSVs for Tables S22(c) and S23(c) are located in `Supplementary_Data/S7`. The full Table S4 exporter and unknown-Q reader default to the internal Result/Q_exposure_support/Discharge_Provenance directory, which includes restricted daily Q. Full daily-discharge provenance exports are restricted and are not distributed as Supplementary Data S4. Reproduction of these outputs requires authorized access to the original data.
