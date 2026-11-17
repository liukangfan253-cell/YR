"""Describe actual post-QC DO gaps; no model fitting or A/B/C recalculation.

python Code/06_Validation_and_sensitivity/02_Continuous_gap_validation/real_do_gap_support.py
Boundary lengths are observed lower bounds, not inferred complete gap lengths.
Event counts are unique within a bin; one event may overlap several bins.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BINS = ['1', '2-3', '4-7', '8-14', '15-30', '>30']
REQUIRED_BINS = ['1-3', '4-7', '8-14', '15-30', '>30']


def gap_records(daily, site, registry):
    """daily has a complete Date calendar and original post-QC DO, not final DO."""
    daily = daily.sort_values('Date').reset_index(drop=True)
    assert not daily.Date.duplicated().any()
    assert daily.Date.diff().dropna().eq(pd.Timedelta(days=1)).all()
    missing = np.flatnonzero(daily.DO.isna().to_numpy())
    runs = [] if not len(missing) else np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    events = registry.loc[registry.Site_ID.eq(site)]
    records = []
    for number, run in enumerate(runs, 1):
        start, end = daily.Date.iloc[run[0]], daily.Date.iloc[run[-1]]
        ids = sorted(events.loc[events.Start_Date.le(end) & events.End_Date.ge(start), 'Event_ID'].unique())
        left, right = bool(run[0] == 0), bool(run[-1] == len(daily) - 1)
        records.append(dict(Site_ID=site, Gap_ID=f'{site}_G{number:04d}', Start_Date=start,
                            End_Date=end, Length_Days=len(run), Left_Censored=left,
                            Right_Censored=right, Boundary_Censored=left or right,
                            Overlapping_Event_IDs=json.dumps(ids), Overlapping_Events=len(ids)))
    return records


def summarize_gaps(records, combined_short=False):
    edges, labels = ([0, 3, 7, 14, 30, np.inf], REQUIRED_BINS) if combined_short else ([0, 1, 3, 7, 14, 30, np.inf], BINS)
    frame = records.copy()
    if frame.empty:
        frame = pd.DataFrame(columns=['Length_Days', 'Overlapping_Event_IDs', 'Boundary_Censored'])
    frame['bin'] = pd.cut(frame.Length_Days, edges, labels=labels)
    rows = []
    for label in labels:
        group = frame.loc[frame['bin'].eq(label)]
        event_ids = {event for ids in group.Overlapping_Event_IDs for event in json.loads(ids)}
        rows.append(dict(Gap_Days=label, Segments=len(group), Missing_Days=int(group.Length_Days.sum()),
                         Boundary_Censored_Gaps=int(group.Boundary_Censored.sum()),
                         Overlapping_Events=len(event_ids)))
    result = pd.DataFrame(rows)
    total = result.Missing_Days.sum()
    result['Missing_Day_Pct'] = result.Missing_Days / total * 100 if total else 0.0
    # Preserve the original plotted CSV column order and append the two requested columns.
    return result[['Gap_Days', 'Segments', 'Missing_Days', 'Missing_Day_Pct',
                   'Boundary_Censored_Gaps', 'Overlapping_Events']]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'Result/Continuous_gap_validation/Real_DO_gap_support')
    ap.add_argument('--source-gap-csv', type=Path, default=ROOT / 'V/Figure_V1/SFigure/SFigure08_Additional_validation/Images/source_gap_distribution.csv')
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def read(path, **kwargs):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path, **kwargs)

    registry = read(ROOT / 'Data/Event_F/Event_Registry_For_Validation.csv', parse_dates=['Start_Date', 'End_Date'])
    assert not registry.Event_ID.duplicated().any()
    calendar = pd.date_range('2021-01-01', '2025-12-31')
    records, daily_rows, low_rows = [], [], []
    for site in sorted(registry.Site_ID.unique()):
        qc = read(ROOT / f'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned/{site}.csv', parse_dates=['Date'])
        labeled = read(ROOT / f'Data/LSTM_Y/{site}_Labeled.csv', parse_dates=['Date'])
        final = read(ROOT / f'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR/{site}_Imputed.csv', parse_dates=['Date'])
        for frame in [qc, labeled, final]:
            assert not frame.Date.duplicated().any(), site
        original = qc.set_index('Date').DO.reindex(calendar)
        labeled_do = labeled.set_index('Date').DO.reindex(calendar)
        original_export = final.set_index('Date').Original_DO.reindex(calendar)
        completed = final.set_index('Date').DO.reindex(calendar)
        assert np.allclose(original, labeled_do, equal_nan=True, rtol=0, atol=2e-6), site
        assert np.allclose(original, original_export, equal_nan=True, rtol=0, atol=2e-6), site
        assert completed.notna().all(), site
        assert np.allclose(original.dropna(), completed.loc[original.notna()], rtol=0, atol=2e-6), site
        daily = pd.DataFrame({'Date': calendar, 'DO': original.to_numpy()})
        site_records = gap_records(daily, site, registry)
        records.extend(site_records)
        gap_days = pd.Series(0, index=calendar, dtype=int)
        censored = pd.Series(False, index=calendar)
        for gap in site_records:
            gap_days.loc[gap['Start_Date']:gap['End_Date']] = gap['Length_Days']
            censored.loc[gap['Start_Date']:gap['End_Date']] = gap['Boundary_Censored']
        events = registry.loc[registry.Site_ID.eq(site)]
        event_id = pd.Series('', index=calendar)
        for e in events.itertuples():
            assert event_id.loc[e.Start_Date:e.End_Date].eq('').all()
            event_id.loc[e.Start_Date:e.End_Date] = e.Event_ID
        for threshold in [5, 3, 2]:
            use = event_id.ne('') & completed.lt(threshold)
            for date in calendar[use]:
                low_rows.append(dict(Site_ID=site, Event_ID=event_id.loc[date], Date=date,
                    Threshold=threshold, DO=completed.loc[date], Is_QC_Observed=bool(original.notna().loc[date]),
                    Gap_Length_Days=int(gap_days.loc[date]), Boundary_Censored=bool(censored.loc[date])))
        daily_rows.append(dict(Site_ID=site, Calendar_Days=len(calendar), Observed_Days=int(original.notna().sum()),
                               Imputed_Days=int(original.isna().sum()), Gap_N=len(site_records)))
    details = pd.DataFrame(records)
    summary = summarize_gaps(details)
    old = pd.read_csv(args.source_gap_csv)
    pd.testing.assert_frame_equal(old[['Gap_Days', 'Segments', 'Missing_Days']].reset_index(drop=True),
                                  summary[['Gap_Days', 'Segments', 'Missing_Days']], check_dtype=False)
    details.to_csv(out / 'real_DO_gap_segments.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(args.source_gap_csv, index=False, encoding='utf-8-sig')
    canonical = summarize_gaps(details, combined_short=True).rename(columns={
        'Gap_Days':'gap_length_bin', 'Segments':'gap_n', 'Missing_Days':'missing_days',
        'Boundary_Censored_Gaps':'boundary_censored_gap_n', 'Overlapping_Events':'overlapping_events',
        'Missing_Day_Pct':'missing_day_pct'})
    canonical['gap_pct'] = canonical.gap_n / canonical.gap_n.sum() * 100
    canonical.to_csv(out / 'real_DO_gap_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(daily_rows).to_csv(out / 'daily_observation_source_summary.csv', index=False, encoding='utf-8-sig')
    low = pd.DataFrame(low_rows)
    low.to_csv(out / 'low_DO_gap_source_audit.csv', index=False, encoding='utf-8-sig')
    values = []
    for threshold, group in low.groupby('Threshold'):
        imputed = group.loc[~group.Is_QC_Observed]
        values.append(dict(Threshold=int(threshold), Event_Low_Days=len(group),
                           Observed_Days=int(group.Is_QC_Observed.sum()), Imputed_Days=len(imputed),
                           Imputed_Days_In_Over30_Gaps=int(imputed.Gap_Length_Days.gt(30).sum())))
    pd.DataFrame(values).to_csv(out / 'low_DO_gap_source_summary.csv', index=False, encoding='utf-8-sig')
    # Reuse existing event-level absolute DO outputs; retain their source categories.
    event_metrics = read(ROOT / 'Result/Decrease/DO_observation_sensitivity/20261111/20261111_M07事件绝对DO指标.csv')
    assert set(event_metrics.Event_ID) == set(registry.Event_ID)
    event_metrics['Observation_Support'] = np.select(
        [event_metrics.Observed_DO_Days.eq(event_metrics.Event_Duration_Days), event_metrics.Observed_DO_Days.eq(0)],
        ['All_event_days_observed', 'Fully_reconstructed'], default='Partly_observed')
    event_sources = []
    for source in ['All_event_days_observed', 'Partly_observed', 'Fully_reconstructed']:
        group = event_metrics.loc[event_metrics.Observation_Support.eq(source)]
        event_sources.append(dict(Observation_Support=source, Event_N=len(group),
            Event_Mean_Below5_N=int(group.Event_DO_Mean.lt(5).sum()),
            Event_Mean_Below3_N=int(group.Event_DO_Mean.lt(3).sum()),
            Event_Mean_Below2_N=int(group.Event_DO_Mean.lt(2).sum()),
            Any_Event_Day_Below2_N=int(group.Any_DO_Below2.sum())))
    pd.DataFrame(event_sources).to_csv(out / 'low_DO_event_source_summary.csv', index=False, encoding='utf-8-sig')
    assert summary.Missing_Days.sum() == sum(r['Imputed_Days'] for r in daily_rows)
    assert sum(r['Observed_Days'] + r['Imputed_Days'] for r in daily_rows) == sum(r['Calendar_Days'] for r in daily_rows)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    report = dict(sources=hashes, sites=len(daily_rows), registry_events=len(registry), gap_n=len(details),
        boundary_censored_gap_n=int(details.Boundary_Censored.sum()),
        left_censored_gap_n=int(details.Left_Censored.sum()), right_censored_gap_n=int(details.Right_Censored.sum()),
        overlapping_events=len({e for ids in details.Overlapping_Event_IDs for e in json.loads(ids)}),
        missing_days=int(summary.Missing_Days.sum()),
        event_count_rule='Unique Event_ID within each bin; counts across bins are not additive',
        censoring_rule='Touches station study-calendar start/end; length is an observed lower bound',
        original_gap_distribution_counts_unchanged=True, model_fitting=False, ABC_recalculation=False)
    (out / 'run_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(canonical.to_string(index=False))
    print(pd.DataFrame(values).to_string(index=False))
    print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
