# Supplementary Data

Dissolved oxygen responses to heatwave–high-flow extremes in the Yellow River Basin: Roles of reference conditions and observational support

The numbered folders follow the Supplementary Data section of the Supplementary Information. The table-specific folders contain CSVs supporting Tables S4 and S22b.
The [file index](file_index.csv) lists the topic, filename, description and row count for each CSV.

| Folder | Contents | Files |
| --- | --- | ---: |
| S1 | Site-specific reconstruction performance | 1 |
| S2 | Training history and continuous-gap validation | 20 |
| S3 | Antecedent-reference sensitivity | 14 |
| S4 | Post-event slope eligibility comparison | 12 |
| S5 | Jiagaiqiao source and environmental audit | 17 |
| S6 | Observational-coverage threshold sensitivity | 2 |
| S7 | Nested and mutually exclusive observation-coverage populations | 4 |
| S8 | Observation-only dual-reference sensitivity and provenance | 4 |
| S9 | Fixed-hyperparameter site-event grouped prediction validation | 3 |
| TableS4_Q_Support | Gauge summaries, daily discharge provenance, basic high-flow episodes and unknown-discharge sensitivity results for Table S4(a–f) | 5 |
| TableS22b_Hydro_Cluster | Shared-gauge cluster contrasts and sample support for Table S22b | 2 |

In S1, filter the Dataset column to Test(25) to obtain the 2025 holdout metrics reported in the Supplementary Information.
In S6 and S7, the pair of Site_ID and Event_ID identifies a site–event record. Coverage is based on original quality-controlled DO on the event dates.
Shock-L and Shock-S are dimensionless. DO and concentration errors are in mg L⁻¹, early slopes in mg L⁻¹ d⁻¹, and the training-history MSE uses standardised DO.

The S6 and S7 coverage tables use the current event-level analysis outputs. The S7 model files combine the reported full-sample and restricted-slope results with the restricted reference-ratio fits using the original mixed-effects model.
In the S9 metrics file, Input_Path is expressed relative to the analysis project.
In TableS4_Q_Support, Hydro_Site_ID and Date identify daily records. Basic_Episode_ID links high-flow days to Episode_ID in the episode catalogue. Q and P95 are in m³ s⁻¹. A blank original Q denotes a missing daily value. Q_Unknown_Sample_Summary.csv and Q_Unknown_Core_Contrasts.csv contain the results reported in Table S4(e) and Table S4(f), respectively.
