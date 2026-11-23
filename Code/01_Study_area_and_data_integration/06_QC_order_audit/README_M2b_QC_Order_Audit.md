# M2b audit of the water quality screening order

[audit_qc_order_impact.py](audit_qc_order_impact.py) compares two orders for applying water-temperature and surface-pressure checks in DO quality control. Both branches use the same current surface-pressure series (`SP`), using the path configured in `1.WQ_Data_P.py` by default. The absolute DO limits, saturation formula, water-temperature bounds and rounding rules are shared between the branches. The source water-quality class and remarks are retained for traceability. The audit leaves the official QC workflow, input data, model assets and manuscript files intact.

## Run the audit

Run these commands from the submission package root in the scientific Python environment described in the [code README](../../README.md).

```powershell
# Run after generating the quality-controlled daily water-quality tables.
python Code/01_Study_area_and_data_integration/06_QC_order_audit/audit_qc_order_impact.py --stage qc

# Run after ED_N.py and Lstm_Y.py and before model training.
python Code/01_Study_area_and_data_integration/06_QC_order_audit/audit_qc_order_impact.py --stage full --events-file Data/Event_N/All_Events_Missing_Status.csv
```

The `qc` stage marks downstream model and event checks as `NOT_ASSESSED` and leaves their numeric fields empty. The `full` stage compares feature preparation, temporary scaling, model input windows and original event DO inputs. Model training, Shock ratios, slope estimates and significance tests belong to separate analyses.

## Output files

Each run updates two CSVs under `Result/M2b_QC_Order/`. The `Stage` field identifies which stage produced the outputs.

1. `qc_order_impact.csv` contains the supplementary summary, station counts, model-group impacts, run status, rules and input paths and hashes. `record_type` identifies the row type. The metric is in `metric`, the ordinary value is in `value`, and supplementary summary rows use `compared_n` and `changed_n`. Select rows with `record_type=SUMMARY` for the supplementary table.
2. `changed_record_log.csv` contains timestamps whose DO retention changed because of the order of the checks. It records original DO, WT and SP, both sets of rule codes, changes to daily means and model inputs, and event IDs. The file contains only the header if retention never changes.

Detailed observations and errors are recorded in these two files. If the audit fails, the `RUN` row records `FAILED` and the cause. Use the summary only when the `RUN` status is `COMPLETED`.

| `record_type` | Contents |
| --- | --- |
| `SUMMARY` | Supplementary metrics, denominators, changed counts and audit status |
| `STATION` | Station-level effects and correspondence with the official daily tables |
| `MODEL` | Model groups, targets, samples and windows affected in the full audit |
| `RUN` and `INPUT_FILE` | Run status, rules, input files and hashes. The `RUN` row also records errors. |
| `DAILY`, `SCALER` and `EVENT` | Daily averages, scaler diagnostics or event inputs where changes occur |
| `GUARD_ERROR` | Failed checks of the original surface-pressure match rate or valid range |

Use `--details changes`, the default, to include detailed diagnostic rows only where there are changes. Use `--details none` to suppress detail or `--details all` to include all diagnostics, including `BASELINE` and `DEPENDENCY` rows. All three options write the same two CSV files.

If changed inputs require rerunning a downstream analysis, produce `before_after_key_results.csv` from the results of the two completed runs. This audit checks inputs and does not produce that results comparison.

## Interpretation and dependencies

The official daily means are rounded to three decimal places. The audit also compares unrounded means. The numerical tolerance is `1e-12`, and the reproduction tolerance for the official daily DO tables is `2e-6 mg/L`. Record counts and check results come from the supplied inputs. Development and final training groups overlap, so their counts cannot be added as independent samples. Event checks cover original DO inputs and candidate reference days.

This version was tested on example data. Run the audit on the intended batch to obtain its formal results. The precipitation and SSR corrections require their own analysis run.

The audit uses Python 3.9 or later, pandas, NumPy and pypinyin. The `full` stage also uses scikit-learn.
