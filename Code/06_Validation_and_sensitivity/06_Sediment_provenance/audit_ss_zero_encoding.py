"""Trace source-blank Ss zero coding to the inputs in Supplementary Table S2(a–c)."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HY = ROOT / 'Data/Hydrology_data_P/Hy_data_YR'
RAW = ROOT / 'Data/Hydrology_data/Hy_data_YR'
DATES = pd.date_range('2021-01-01', '2025-12-31', name='Date')
sys.path.insert(0, str(ROOT / 'Code/07_SHAP_analysis'))
from analysis_config import read_analysis_config


def source_data(gauges):
    script = ROOT / 'Code/01_Data_integration/03_Hydrological_QC/2.Hy_Data_P.py'
    spec = importlib.util.spec_from_file_location('ss_hydrological_qc', script)
    qc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qc)
    parts = []
    for path in sorted(RAW.iterdir()):
        if path.name.startswith('~$') or path.suffix.lower() not in {'.xlsx', '.xls', '.csv'}:
            continue
        print(f'Reading {path.name}', flush=True)
        raw = qc.read_file(str(path))
        if raw is None:
            raise ValueError(f'Cannot read {path}')
        time_column = next(c for c in ('时间', '监测时间') if c in raw)
        names = {v: qc.translate_hys_name(v) for v in raw['水文站站名'].dropna().unique()}
        date = pd.to_datetime(raw[time_column], errors='coerce').dt.normalize()
        name = raw['水文站站名'].map(names)
        keep = name.isin(gauges) & date.between(DATES[0], DATES[-1])
        ss = raw.loc[keep, '含沙量']
        blank = ss.isna() | ss.astype(str).str.strip().eq('')
        numeric_ss = pd.to_numeric(ss, errors='coerce')
        parts.append(pd.DataFrame({
            'Hydro_Site_ID': name[keep], 'Date': date[keep],
            'WL': pd.to_numeric(raw.loc[keep, '水位'], errors='coerce'),
            'Q': pd.to_numeric(raw.loc[keep, '流量'], errors='coerce'),
            'Ss': numeric_ss, 'Source_Ss_Blanks': blank.astype(int),
            'Source_Ss_Numeric_Zeros': numeric_ss.eq(0).astype(int),
            'Source_Ss_Invalid_Nonblank': (numeric_ss.isna() & ~blank).astype(int),
        }))
    daily = pd.concat(parts, ignore_index=True).groupby(['Hydro_Site_ID', 'Date']).agg(
        {'WL': 'mean', 'Q': 'mean', 'Ss': 'mean', 'Source_Ss_Blanks': 'sum',
         'Source_Ss_Numeric_Zeros': 'sum', 'Source_Ss_Invalid_Nonblank': 'sum'})
    if set(daily.index.get_level_values(0)) != set(gauges):
        raise ValueError('Raw workbooks do not cover all matched gauges')
    daily = daily.reindex(pd.MultiIndex.from_product(
        [gauges, DATES], names=['Hydro_Site_ID', 'Date']))
    source_counts = ['Source_Ss_Blanks', 'Source_Ss_Numeric_Zeros', 'Source_Ss_Invalid_Nonblank']
    daily[source_counts] = daily[source_counts].fillna(0).astype(int)
    daily['Ss_Original_Missing'] = daily.Ss.isna()
    daily['Ss_Zero_Coded'] = daily.Ss.isna() & daily.WL.notna() & daily.Q.notna()
    for gauge in gauges:
        source = daily.loc[gauge]
        official = pd.read_csv(HY / f'Original/Retained/{gauge}.csv', parse_dates=['Date']).set_index('Date')
        official = official.reindex(DATES)
        expected = source[['WL', 'Q', 'Ss']].copy()
        expected.loc[source.Ss_Zero_Coded, 'Ss'] = 0
        # The official export rounds daily means to three decimal places.
        if not np.allclose(expected, official[expected.columns], rtol=0, atol=.000500001, equal_nan=True):
            raise ValueError(f'{gauge}: raw aggregation does not match official pre-interpolation hydrology')
    return daily.reset_index()


def hydrology_summary(daily, mapped):
    rows = []
    for period in [*sorted(daily.Date.dt.year.unique()), 'Total']:
        g = daily if period == 'Total' else daily[daily.Date.dt.year.eq(period)]
        w = mapped if period == 'Total' else mapped[mapped.Date.dt.year.eq(period)]
        rows.append(dict(Period=period, Blank_Ss_Source_Records=int(g.Source_Ss_Blanks.sum()),
                         Numerical_Zero_Ss_Source_Records=int(g.Source_Ss_Numeric_Zeros.sum()),
                         Invalid_Nonblank_Ss_Source_Records=int(g.Source_Ss_Invalid_Nonblank.sum()),
                         Gauge_Days_Missing_Original_Ss=int(g.Ss_Original_Missing.sum()),
                         Gauge_Days_Zero_Coded=int(g.Ss_Zero_Coded.sum()),
                         Gauges_Affected=g.loc[g.Ss_Zero_Coded, 'Hydro_Site_ID'].nunique(),
                         Mapped_Water_Quality_Site_Days_Zero_Coded=int(w.Ss_Zero_Coded.sum())))
    return pd.DataFrame(rows)


def bilstm_summary(mapped):
    manifest_path = ROOT / 'Result/BiLstm_Att_V1/Saved_Models_V1/DO_Model_Manifest_V1.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    history = manifest['configuration']['sequence_length']
    test = manifest['configuration']['test_year']
    validation = max(manifest['training_years'])
    development = set(manifest['training_years']) - {validation}
    groups = ['Development_training', 'Validation', 'Independent_test', 'Final_training', 'Missing_DO_prediction']
    counts = {name: dict(total=0, affected=0, sites=0, dates=0) for name in groups}
    for site, provenance in mapped.groupby('Site_ID'):
        model_input = pd.read_csv(ROOT / f'Data/LSTM_Y/{site}_Labeled.csv',
                                  usecols=['Date', 'DO', 'Ss', 'Matched_HS_Name'], parse_dates=['Date'])
        d = model_input.merge(provenance, on='Date', how='outer', validate='one_to_one',
                              indicator=True).sort_values('Date')
        if (not d._merge.eq('both').all() or not d.Matched_HS_Name.eq(d.Hydro_Site_ID).all()
                or not d.loc[d.Ss_Zero_Coded, 'Ss'].eq(0).all()):
            raise ValueError(f'{site}: mapped Ss provenance disagrees with model inputs')
        flagged = d.Ss_Zero_Coded.to_numpy()
        year = d.Date.dt.year
        observed = d.DO.notna().to_numpy() & (np.arange(len(d)) >= history)
        masks = [observed & year.isin(development), observed & year.eq(validation),
                 observed & year.eq(test), observed & year.isin(manifest['training_years']), d.DO.isna()]
        for name, mask in zip(groups, masks):
            positions = np.flatnonzero(mask)
            affected, used = 0, np.zeros(len(d), dtype=bool)
            for t in positions:
                start, stop = max(0, t - history), max(1, t)
                if flagged[start:stop].any():
                    affected += 1
                    used[start:stop] = True
            counts[name]['total'] += len(positions)
            counts[name]['affected'] += affected
            counts[name]['sites'] += int(affected > 0)
            counts[name]['dates'] += int((used & flagged).sum())
    return pd.DataFrame([
        dict(Model_Use=name, Total_Samples_Or_Prediction_Days=c['total'], Windows_With_Zero_Coded_Inputs=c['affected'],
             Affected_Pct=100 * c['affected'] / c['total'], Water_Quality_Sites_Affected=c['sites'],
             Zero_Coded_Site_Days_Used_As_Inputs=c['dates'])
        for name, c in counts.items()
    ])


def shap_summary(mapped):
    outputs = []
    event_types = {'H': 'Heatwave_Only', 'F': 'Flood_Only', 'DC': 'Direct_Compound',
                   'H5F': 'H5F_Compound', 'F5H': 'F5H_Compound'}
    for kind, event_type in event_types.items():
        script = ROOT / f'Code/07_SHAP_analysis/03_Tree_model_analysis/analyze_{kind}_shap.py'
        config = read_analysis_config(script, ROOT)
        if not config['SHAP_USE_ALL_DATA']:
            raise ValueError(f'{script.name}: explanation sample selection must be traced explicitly')
        data = pd.read_csv(config['DATA_FILE'], parse_dates=['Date']).rename(columns={'WQS_name': 'Site_ID'})
        features = [f for f in config['FEATURE_COLS'] if f in data]
        retained = data.dropna(subset=[config['TARGET_COL'], *features])
        sample = retained.merge(mapped, on=['Site_ID', 'Date'], how='left',
                                validate='many_to_one', indicator=True)
        if (not sample._merge.eq('both').all() or not sample.Matched_HS_Name.eq(sample.Hydro_Site_ID).all()
                or not sample.loc[sample.Ss_Zero_Coded, 'Ss'].eq(0).all()):
            raise ValueError(f'{kind}: SHAP inputs do not align with mapped Ss provenance')
        affected = sample[sample.Ss_Zero_Coded]
        outputs.append(dict(Event_Type=event_type, Days=len(sample),
                            Affected_Days=len(affected), Affected_Day_Pct=100 * len(affected) / len(sample),
                            Events=len(sample[['Site_ID', 'Event_ID']].drop_duplicates()),
                            Affected_Events=len(affected[['Site_ID', 'Event_ID']].drop_duplicates()),
                            Sites=sample.Site_ID.nunique(), Affected_Sites=affected.Site_ID.nunique()))
    return pd.DataFrame(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'Result/Sediment_provenance/20261111')
    args = parser.parse_args()
    mapping = pd.read_csv(ROOT / 'Config/Study_Station_Mapping.csv').query("Status == 'Retained'")
    mapping = mapping[['WQS_name', 'Matched_HS_Name']].rename(
        columns={'WQS_name': 'Site_ID', 'Matched_HS_Name': 'Hydro_Site_ID'})
    if mapping.Site_ID.duplicated().any() or mapping.isna().any().any():
        raise ValueError('Expected one gauge assignment per retained site')
    daily = source_data(sorted(mapping.Hydro_Site_ID.unique()))
    mapped = mapping.merge(daily[['Hydro_Site_ID', 'Date', 'Ss_Zero_Coded']], on='Hydro_Site_ID')
    tables = {
        'S2_Ss_Zero_Coding_By_Year.csv': hydrology_summary(daily, mapped),
        'S2_Ss_Zero_Coding_BiLSTM_Windows.csv': bilstm_summary(mapped),
        'S2_Ss_Zero_Coding_SHAP_Samples.csv': shap_summary(mapped),
        'S2_Ss_Gauge_Day_Provenance.csv': daily,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False, date_format='%Y-%m-%d')
        print(f'{name}: {len(table):,} rows', flush=True)


if __name__ == '__main__':
    main()
