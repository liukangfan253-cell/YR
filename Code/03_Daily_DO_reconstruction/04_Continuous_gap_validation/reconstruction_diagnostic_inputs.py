"""Read saved V1 predictions and align their observed targets to the V5 calendar."""

from pathlib import Path

import numpy as np
import pandas as pd

from validate_continuous_do_gaps import REGISTRY, ROOT, contiguous_runs, event_labels, load_data, metrics, require, sha
from real_do_gap_support import gap_records, summarize_gaps


SCATTER_DIR = ROOT / 'Result/BiLstm_Att_V1/Reports/By_Site_Scatter'
DAILY_DIR = ROOT / 'Data/LSTM_Y'
TEST_COLUMNS = ['Observed DO', 'Simulated DO']
POINTWISE_GROUPS = ['All', 'Event', 'NonEvent']
GAP_BINS = ['1', '2-3', '4-7', '8-14', '15-30', '>30']


def add_pointwise_arguments(parser):
    parser.add_argument('--scatter-dir', type=Path, default=SCATTER_DIR, help='Directory of per-site saved scatter CSVs')
    parser.add_argument('--daily-dir', type=Path, default=DAILY_DIR, help='Directory of per-site *_Labeled.csv observed daily series')
    parser.add_argument('--registry', type=Path, default=REGISTRY, help='Event registry used to label test dates')
    parser.add_argument('--test-year', type=int, default=2025, help='Calendar year represented by the saved test predictions')
    parser.add_argument('--history-days', type=int, default=30, help='Number of initial daily rows required as model history')


def test_predictions(path):
    data = pd.read_csv(path)
    tests = data.loc[data.Dataset.astype(str).str.startswith('Test'), TEST_COLUMNS].reset_index(drop=True)
    require(len(tests) > 0 and np.isfinite(tests.to_numpy(dtype=float)).all(), f'No valid saved test predictions: {path}')
    return tests


def read_pointwise(args, include_gaps=False):
    require(args.history_days >= 0, 'History length must be nonnegative')
    scatters = sorted(args.scatter_dir.rglob('*_Labeled_Scatter.csv'))
    require(bool(scatters), f'No *_Labeled_Scatter.csv files in {args.scatter_dir}')
    sources = {str(args.registry.resolve()): sha(args.registry)}
    registry = pd.read_csv(args.registry, parse_dates=['Start_Date', 'End_Date'])
    require(registry.End_Date.ge(registry.Start_Date).all(), 'Invalid event dates in registry')
    records, gap_details, seen = [], [], set()
    for scatter in scatters:
        site = scatter.name[:-len('_Labeled_Scatter.csv')]
        require(site not in seen, f'Duplicate scatter file for {site}')
        seen.add(site)
        daily_path = args.daily_dir / f'{site}_Labeled.csv'
        sources[str(scatter.resolve())] = sha(scatter)
        sources[str(daily_path.resolve())] = sha(daily_path)
        daily = load_data(daily_path)
        require(daily.WQS_name.dropna().eq(site).all(), f'Station identifier differs: {daily_path}')
        observed = daily.iloc[args.history_days:]
        observed = observed.loc[observed.Date.dt.year.eq(args.test_year) & observed.DO.notna()]
        tests = test_predictions(scatter)
        require(len(tests) == len(observed) and np.allclose(tests['Observed DO'], observed.DO.round(3), atol=1e-7, rtol=0),
                f'Saved test targets differ from daily observations for {site}. Check the input directories, test year and history length.')
        event_ids, _ = event_labels(observed, site, registry)
        records.append(tests.assign(Site_ID=site, Date=observed.Date.to_numpy(), Is_Event=event_ids != ''))
        if include_gaps:
            gap_details.extend(gap_records(daily, site, registry))
    require(all(sha(path) == digest for path, digest in sources.items()), 'Pointwise input changed during read')
    return pd.concat(records, ignore_index=True), pd.DataFrame(gap_details), sources


def summarize_test_points(points):
    groups = [points, points.loc[points.Is_Event], points.loc[~points.Is_Event]]
    return pd.DataFrame([dict(Group=name, Subset='All_points', **metrics(frame['Observed DO'], frame['Simulated DO']))
                         for name, frame in zip(POINTWISE_GROUPS, groups)])


def summarize_real_gaps(details):
    return summarize_gaps(details)
