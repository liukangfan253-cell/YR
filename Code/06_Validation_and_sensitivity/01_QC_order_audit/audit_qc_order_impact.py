"""M2b: paired WT/SP QC-order audit, with all CSVs under Result.

Run after the official daily water-quality QC:
    python audit_qc_order_impact.py --stage qc
Run again after Lstm_Y and the current event catalogue, before model training:
    python audit_qc_order_impact.py --stage full

Both branches use exactly the same raw DO/WT and current meteorological SP.
The official QC, input tables, model assets and paper are never overwritten.
Full mode also compares the original standardization preprocessing, without
training a neural network, loading weights, or calculating event-response tests.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import ast
import hashlib
import sys
import traceback
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
DATES = pd.date_range('2021-01-01', '2025-12-31')
LOG_FIELDS = ['wq_site_id', 'timestamp', 'DO_raw', 'WT_raw', 'SP_raw',
    'old_reason_code', 'new_reason_code', 'DO_keep_changed', 'daily_DO_changed',
    'model_input_or_target_changed', 'event_ids', 'source_file', 'source_record_index',
    'old_sat_cap', 'new_sat_cap', 'old_DO_kept', 'new_DO_kept', 'WT_valid', 'SP_valid',
    'model_check_status', 'event_link_status', 'old_effective_SP',
    'source_quality_class', 'source_station_note']


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


REPORT_FIELDS = ['record_type', 'wq_site_id', 'model_split', 'event_ids', 'Date',
                 'metric', 'value', 'compared_n', 'changed_n', 'status',
                 'denominator_definition', 'source_file']


class AuditReport:
    """Two CSVs only: a typed long-form report and the timestamp change log."""
    def __init__(self, out):
        self.out = out
        self.tables = []
        pd.DataFrame(columns=LOG_FIELDS).to_csv(out/'changed_record_log.csv', index=False, encoding='utf-8-sig')

    def save(self, name, data):
        if name == 'changed_record_log.csv':
            data.to_csv(self.out/name, index=False, encoding='utf-8-sig', float_format='%.17g')
        else:
            self.tables.append((name, data.copy()))

    def write(self, info):
        rows = run_info_rows(info)
        types = {'qc_order_impact.csv': 'STATION', 'si_qc_order_summary.csv': 'SUMMARY',
                 'model_split_impact.csv': 'MODEL', 'standardization_impact.csv': 'SCALER',
                 'event_observed_do_input_impact.csv': 'EVENT', 'qc_order_daily_comparison.csv': 'DAILY',
                 'invalid_dependency_records.csv': 'DEPENDENCY',
                 'baseline_reproduction_checks.csv': 'BASELINE', 'legacy_sp_guard_failure.csv': 'GUARD_ERROR'}
        identities = ['wq_site_id', 'model_split', 'event_ids', 'Date', 'source_file']
        for name, data in self.tables:
            kind = types[name]
            for item in data.to_dict('records'):
                context = {k: item[k] for k in identities if k in item}
                if kind == 'SUMMARY':
                    rows.append(dict(record_type=kind, **context, **{k: v for k, v in item.items() if k not in identities}))
                else:
                    for key, value in item.items():
                        if key not in identities:
                            rows.append(dict(record_type=kind, **context, metric=key, value=value))
        pd.DataFrame(rows, columns=REPORT_FIELDS).to_csv(self.out/'qc_order_impact.csv',
            index=False, encoding='utf-8-sig', float_format='%.17g')


def run_info_rows(info):
    """Status, rules, errors and source hashes are rows of qc_order_impact.csv."""
    rows = []
    for key, value in info.items():
        if key == 'Sources':
            for item in value:
                rows.append(dict(record_type='INPUT_FILE', metric='SHA256',
                    value=item['SHA256'], source_file=item['source_file']))
            continue
        if value is None:
            value = 'NOT_ASSESSED'
        elif isinstance(value, (list, tuple)):
            value = '; '.join(str(x) for x in value)
        rows.append(dict(record_type='RUN', metric=key, value=str(value), source_file=''))
    return rows


def different(a, b, tolerance=1e-12):
    return ~np.isclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       rtol=0, atol=tolerance, equal_nan=True)


def read_csv(path, **kwargs):
    for encoding in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    raise ValueError('Cannot decode ' + str(path))


def constant(tree, name, project_root=None):
    """Read literal or project-relative configuration without importing the script."""
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
    def value(node):
        if isinstance(node, ast.Name):
            if node.id == 'ROOT' and project_root is not None:
                return project_root
            return value(assignments[node.id])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return value(node.left) / value(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'str' and len(node.args) == 1 and not node.keywords):
            return str(value(node.args[0]))
        return ast.literal_eval(node)
    return value(assignments[name])


def extracted_functions(tree, names, scope, filename):
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'Required source function missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), 'exec'), scope)
    return scope


def compare_records(raw, calc, wt_limits):
    """Pure paired comparison; no saved tables or model are changed."""
    d = raw.copy()
    wt, sp, do = d.WT_raw, d.SP_raw, d.DO_raw
    wt_valid = np.isfinite(wt) & wt.between(*wt_limits)
    sp_valid = np.isfinite(sp) & sp.between(500, 1100)
    valid = wt_valid & sp_valid
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        old_sat = pd.Series(calc(wt, sp), index=d.index).round(3)
        new_sat = pd.Series(np.nan, index=d.index, dtype=float)
        new_sat.loc[valid] = pd.Series(calc(wt[valid], sp[valid]), index=d.index[valid]).round(3)
    old_cap, new_cap = (old_sat * 1.5).round(3), (new_sat * 1.5).round(3)
    absolute = do.lt(0) | do.gt(20)
    old_sat_bad, new_sat_bad = do.gt(old_cap), do.gt(new_cap)
    old_keep = do.notna() & ~absolute & ~old_sat_bad
    new_keep = do.notna() & ~absolute & ~new_sat_bad
    require(not (new_keep & absolute).any(), 'Absolute-invalid DO was restored')
    def reason(cap, saturated, keep):
        codes = pd.Series('', index=d.index, dtype=object)
        def add(mask, label):
            codes.loc[mask] = codes.loc[mask].map(lambda x: x + ('|' if x else '') + label)
        add(do.isna(), 'DO_MISSING_OR_NONNUMERIC')
        add(do.lt(0), 'REJECT_DO_BELOW_0')
        add(do.gt(20), 'REJECT_DO_ABOVE_20')
        add(saturated, 'REJECT_SATURATION_CAP')
        add(keep & cap.notna(), 'KEEP_WITH_SATURATION_CAP')
        add(keep & cap.isna(), 'KEEP_ABSOLUTE_ONLY')
        add(~wt_valid, 'WT_UNAVAILABLE_OR_INVALID')
        add(~sp_valid, 'SP_UNAVAILABLE_OR_INVALID')
        return codes
    d['WT_valid'], d['SP_valid'] = wt_valid, sp_valid
    d['old_sat_cap'], d['new_sat_cap'] = old_cap, new_cap
    d['old_DO_kept'], d['new_DO_kept'] = old_keep, new_keep
    d['old_reason_code'] = reason(old_cap, old_sat_bad, old_keep)
    d['new_reason_code'] = reason(new_cap, new_sat_bad, new_keep)
    d['DO_keep_changed'] = old_keep.ne(new_keep)
    d['old_DO'] = do.where(old_keep)
    d['new_DO'] = do.where(new_keep)
    return d


def daily_comparison(records, calendar):
    old = records.old_DO.groupby(records.Date).mean().reindex(calendar)
    new = records.new_DO.groupby(records.Date).mean().reindex(calendar)
    old_n = records.old_DO.groupby(records.Date).count().reindex(calendar, fill_value=0)
    new_n = records.new_DO.groupby(records.Date).count().reindex(calendar, fill_value=0)
    return pd.DataFrame(dict(Date=calendar, old_DO=old.round(3).to_numpy(), new_DO=new.round(3).to_numpy(),
        old_DO_unrounded=old.to_numpy(), new_DO_unrounded=new.to_numpy(),
        old_observed=old.notna().to_numpy(), new_observed=new.notna().to_numpy(),
        old_valid_timestamp_n=old_n.to_numpy(), new_valid_timestamp_n=new_n.to_numpy(),
        daily_DO_changed=different(old.round(3), new.round(3)),
        unrounded_daily_DO_changed=different(old, new),
        observed_flag_changed=old.notna().ne(new.notna()).to_numpy()))


def changed_windows(row_changes, sequence_length):
    changes = np.asarray(row_changes, dtype=bool)
    ends = np.arange(len(changes))
    sums = np.r_[0, np.cumsum(changes)]
    windows = sums[ends] - sums[np.maximum(0, ends - sequence_length)] > 0
    # Missing-DO prediction pads early windows by repeating the first input row.
    padded = windows | ((ends < sequence_length) & changes[0])
    return windows, padded


def load_events(root, override, tracker):
    if override:
        paths = [Path(override).resolve()]
    elif (root / 'Data/Event_F/Event_Registry_For_Validation.csv').is_file():
        paths = [root / 'Data/Event_F/Event_Registry_For_Validation.csv']
    elif (root / 'Data/Event_N/All_Events_Missing_Status.csv').is_file():
        paths = [root / 'Data/Event_N/All_Events_Missing_Status.csv']
    else:
        return None
    d = pd.concat([read_csv(tracker(p)) for p in paths], ignore_index=True)
    d = d.rename(columns={'WQS_name': 'wq_site_id', 'Site_ID': 'wq_site_id'})
    require({'wq_site_id', 'Event_ID', 'Start_Date', 'End_Date'}.issubset(d), 'Invalid event catalogue columns')
    require(not d.duplicated(['wq_site_id', 'Event_ID']).any(), 'Duplicate site-event keys')
    d.Start_Date = pd.to_datetime(d.Start_Date, errors='raise').dt.normalize()
    d.End_Date = pd.to_datetime(d.End_Date, errors='raise').dt.normalize()
    require(d.Start_Date.le(d.End_Date).all(), 'Invalid event dates')
    require(d.Start_Date.ge(DATES[0]).all() and d.End_Date.le(DATES[-1]).all(), 'Events outside study dates')
    return d


def event_links(events, site, calendar):
    links = pd.Series('', index=calendar, dtype=object)
    if events is not None:
        for e in events.loc[events.wq_site_id.eq(site)].itertuples():
            use = (calendar >= e.Start_Date) & (calendar <= e.End_Date)
            require(links.loc[use].eq('').all(), f'Overlapping events at {site}')
            links.loc[use] = str(e.Event_ID)
    return links


def full_inputs(root, sites, daily, model_path, lstm_dir, events, tracker, output):
    from sklearn.preprocessing import StandardScaler
    tree = ast.parse(Path(model_path).read_text(encoding='utf-8-sig'))
    base = constant(tree, 'BASE_FEATURES')
    seq = int(constant(tree, 'SEQ_LEN'))
    train = list(constant(tree, 'TRAIN_YEARS'))
    val, test = int(constant(tree, 'VAL_YEAR')), int(constant(tree, 'TEST_YEAR'))
    final = list(constant(tree, 'FINAL_TRAIN_YEARS'))
    scope = extracted_functions(tree, ['准备特征', '拟合标准化器'],
        {'np': np, 'pd': pd, 'BASE_FEATURES': base, 'StandardScaler': StandardScaler}, model_path)
    prepare, fit = scope['准备特征'], scope['拟合标准化器']
    old_dfs, new_dfs, bindings = [], [], []
    for site in sites:
        frame = read_csv(tracker(lstm_dir / f'{site}_Labeled.csv'))
        frame.Date = pd.to_datetime(frame.Date, errors='raise')
        frame = frame.sort_values('Date').reset_index(drop=True)
        require(frame.Date.tolist() == list(DATES), f'Incomplete/duplicate LSTM calendar: {site}')
        require(set(frame.WQS_name.dropna()) == {site}, f'LSTM station mismatch: {site}')
        d = daily.loc[daily.wq_site_id.eq(site)].sort_values('Date')
        match_old = not different(frame.DO, d.old_DO, 2e-6).any()
        match_new = not different(frame.DO, d.new_DO, 2e-6).any()
        require(match_old or match_new, f'LSTM DO matches neither branch: {site}; use current data')
        a, b = frame.copy(), frame.copy()
        a['DO'], b['DO'] = d.old_DO.to_numpy(), d.new_DO.to_numpy()
        old_dfs.append(a);new_dfs.append(b)
        bindings.append(dict(wq_site_id=site, saved_lstm_DO_matches='BOTH' if match_old and match_new else 'OLD' if match_old else 'NEW'))
    a, features = prepare(old_dfs, 'DO');b, other = prepare(new_dfs, 'DO')
    require(features == other, 'Branch feature-order mismatch')
    fits, scaler_rows = {}, []
    for label, years in [('Development', train), ('Final', final)]:
        ax, ay = fit(a, years, features, 'DO'); bx, by = fit(b, years, features, 'DO')
        require(all(x is not None for x in [ax, ay, bx, by]), f'No targets available to compare {label} preprocessing')
        fits[label] = (ax, ay, bx, by)
        for kind, sa, sb, names in [('Input', ax, bx, features), ('Target', ay, by, ['DO'])]:
            for j, name in enumerate(names):
                scaler_rows.append(dict(fit_stage=label, scaler=kind, feature=name,
                    old_mean=sa.mean_[j], new_mean=sb.mean_[j], old_scale=sa.scale_[j], new_scale=sb.scale_[j],
                    parameters_changed=bool(different(sa.mean_[j], sb.mean_[j]) or different(sa.scale_[j], sb.scale_[j]))))
    model_rows = []
    for site, x, y in zip(sites, a, b):
        pos = np.arange(len(x)); dates = pd.to_datetime(x.Date)
        raw_row = different(x[features], y[features]).any(axis=1)
        raw_target = different(x.DO, y.DO)
        raw_windows, padded_raw = changed_windows(raw_row, seq)
        old_obs, new_obs = x.DO.notna().to_numpy(), y.DO.notna().to_numpy()
        scaled = {}
        for label, (ax, ay, bx, by) in fits.items():
            changed = different(ax.transform(x[features].values), bx.transform(y[features].values)).any(axis=1)
            windows, padded = changed_windows(changed, seq)
            def targets(df, scaler):
                out = np.full(len(df), np.nan);valid = df.DO.notna().to_numpy()
                if valid.any():
                    out[valid] = scaler.transform(df.loc[valid, ['DO']].values).ravel()
                return out
            changed_y = different(targets(x, ay), targets(y, by))
            scaled[label] = (windows, padded, changed_y)
        for name, years, fit_label in [('Development_train', train, 'Development'), ('Validation', [val], 'Development'),
                                     ('Independent_test', [test], 'Final'), ('Final_train', final, 'Final')]:
            period = dates.dt.year.isin(years).to_numpy() & (pos >= seq)
            union = period & (old_obs | new_obs)
            win, _, changed_y = scaled[fit_label]
            model_rows.append(dict(wq_site_id=site, model_split=name, fit_stage=fit_label,
                old_sample_n=int((period & old_obs).sum()), new_sample_n=int((period & new_obs).sum()),
                compared_union_n=int(union.sum()), sample_membership_changed_n=int((period & (old_obs != new_obs)).sum()),
                raw_target_changed_n=int((union & raw_target).sum()), raw_feature_window_changed_n=int((union & raw_windows).sum()),
                standardized_target_changed_n=int((union & changed_y).sum()), standardized_feature_window_changed_n=int((union & win).sum())))
        union = ~old_obs | ~new_obs
        model_rows.append(dict(wq_site_id=site, model_split='Missing_DO_prediction', fit_stage='Final',
            old_sample_n=int((~old_obs).sum()), new_sample_n=int((~new_obs).sum()), compared_union_n=int(union.sum()),
            sample_membership_changed_n=int((old_obs != new_obs).sum()), raw_target_changed_n=pd.NA,
            raw_feature_window_changed_n=int((union & padded_raw).sum()), standardized_target_changed_n=pd.NA,
            standardized_feature_window_changed_n=int((union & scaled['Final'][1]).sum())))
        ix = daily.wq_site_id.eq(site)
        daily.loc[ix, 'model_feature_row_changed'] = raw_row
        daily.loc[ix, 'model_target_changed'] = raw_target
        daily.loc[ix, 'model_input_or_target_changed'] = raw_row | raw_target
    model_table = pd.DataFrame(model_rows).merge(pd.DataFrame(bindings), on='wq_site_id', validate='many_to_one')
    output('model_split_impact.csv', model_table)
    output('standardization_impact.csv', pd.DataFrame(scaler_rows))
    event_rows = []
    for site in sites:
        d = daily.loc[daily.wq_site_id.eq(site)].set_index('Date')
        registered = d.event_ids.ne('')
        for e in events.loc[events.wq_site_id.eq(site)].itertuples():
            days = d.loc[e.Start_Date:e.End_Date]
            pre = d.loc[e.Start_Date-pd.Timedelta(days=9):e.Start_Date-pd.Timedelta(days=4)]
            pre = pre.loc[pre.event_ids.eq('')]
            event_rows.append(dict(wq_site_id=site, event_ids=str(e.Event_ID),
                event_type=getattr(e, 'Event_Type', ''), event_start=e.Start_Date, event_end=e.End_Date,
                event_calendar_days=len(days), old_observed_event_days=int(days.old_observed.sum()),
                new_observed_event_days=int(days.new_observed.sum()), event_DO_input_days_changed=int(days.daily_DO_changed.sum()),
                event_observed_flag_days_changed=int(days.observed_flag_changed.sum()),
                pre_reference_candidate_days_changed=int(pre.daily_DO_changed.sum()),
                site_non_event_candidate_days_changed=int((~registered & d.daily_DO_changed).sum()),
                response_metrics_recomputed=False))
    output('event_observed_do_input_impact.csv', pd.DataFrame(event_rows))
    return model_table, pd.DataFrame(event_rows), pd.DataFrame(scaler_rows), features


def execute(args, out, report):
    sources = {}
    def track(path):
        p = Path(path).resolve();require(p.is_file(), 'Required input not found: ' + str(p))
        require(not p.is_relative_to(out), 'Audit output must not be used as source input')
        sources[str(p)] = sha(p)
        return p
    def save(name, data):
        # Small default set. Detailed diagnostics are opt-in, or changed rows only.
        detailed = {'invalid_dependency_records.csv', 'qc_order_daily_comparison.csv',
                    'baseline_reproduction_checks.csv', 'standardization_impact.csv',
                    'event_observed_do_input_impact.csv'}
        if name in detailed:
            if args.details == 'none':
                return
            if args.details == 'changes':
                if name in {'invalid_dependency_records.csv', 'baseline_reproduction_checks.csv'}:
                    return  # Already represented in the core station/log tables.
                if name == 'qc_order_daily_comparison.csv':
                    mask = data.daily_DO_changed | data.unrounded_daily_DO_changed | data.observed_flag_changed
                    if 'model_input_or_target_changed' in data:
                        mask = mask | data.model_input_or_target_changed.fillna(False).astype(bool)
                elif name == 'standardization_impact.csv':
                    mask = data.parameters_changed
                else:
                    mask = data[['event_DO_input_days_changed', 'event_observed_flag_days_changed',
                                 'pre_reference_candidate_days_changed', 'site_non_event_candidate_days_changed']].gt(0).any(axis=1)
                data = data.loc[mask].copy()
                if data.empty:
                    return
                data['detail_scope'] = 'CHANGED_ROWS_ONLY'
        report.save(name, data)
    root = args.root.resolve()
    qc = track(args.qc_script or root/'Code/01_Data_integration/02_Water_quality_QC/1.WQ_Data_P.py')
    tree = ast.parse(qc.read_text(encoding='utf-8-sig'))
    from pypinyin import lazy_pinyin
    scope = extracted_functions(tree, ['calc_do_sat', 'translate_site_name'], {'pd': pd, 'np': np, 'lazy_pinyin': lazy_pinyin}, qc)
    bounds = constant(tree, 'quality_bounds')['水温']
    match_min = float(constant(tree, 'MIN_SP_MATCH_RATIO'))
    raw_dir = Path(args.raw_dir or constant(tree, 'input_folder', root)).resolve()
    met_dir = Path(args.meteo_dir or constant(tree, 'ERA5_SP_FOLDER', root)).resolve()
    clean_dirs = [Path(constant(tree, name, root)).resolve() for name in ['output_folder_retained_clean', 'output_folder_excluded_clean']]
    mapping = read_csv(track(root/'Config/Study_Station_Mapping.csv'))
    require(mapping.WQS_name.notna().all() and mapping.WQS_name.is_unique, 'Invalid/duplicate station identifiers')
    sites = sorted(mapping.WQS_name.astype(str))
    events = load_events(root, args.events_file, track) if args.stage == 'full' else None
    if args.stage == 'full':
        require(events is not None, 'Full mode requires current event catalogue; run ED_N or specify --events-file')
        require(set(events.wq_site_id).issubset(set(sites)), 'Event catalogue contains sites outside the study mapping')
    print(f'Stage={args.stage}; SP directory from current QC configuration: {met_dir}', flush=True)
    require(raw_dir.is_dir() and met_dir.is_dir(), 'Raw water-quality or current SP directory does not exist')
    index = {}
    for p in sorted(raw_dir.glob('*.csv')):
        d = read_csv(p, usecols=['站点名称', '监测时间'])
        stamp = pd.to_datetime(d['监测时间'], errors='coerce')
        names = d.loc[stamp.dt.year.between(2021, 2025), '站点名称'].dropna().map(scope['translate_site_name']).unique()
        for site in set(names) & set(sites):
            require(len(names) == 1 and site not in index, f'Ambiguous raw station source: {site}')
            index[site] = p
    require(set(index) == set(sites), 'Unmatched raw sources: ' + str(sorted(set(sites)-set(index))))
    summaries, daily_blocks, logs, invalids, baseline = [], [], [], [], []
    for k, site in enumerate(sites, 1):
        p = track(index[site]);raw = read_csv(p)
        raw['source_record_index'] = np.arange(len(raw)) + 1
        raw['timestamp'] = pd.to_datetime(raw['监测时间'], errors='coerce')
        outside = ~raw.timestamp.dt.year.between(2021, 2025)
        raw = raw.loc[~outside].copy().reset_index(drop=True)
        require({'溶解氧', '水温'}.issubset(raw), f'Missing DO/WT source field: {site}')
        raw['Date'] = raw.timestamp.dt.normalize()
        raw['DO_raw'] = pd.to_numeric(raw['溶解氧'], errors='coerce')
        raw['WT_raw'] = pd.to_numeric(raw['水温'], errors='coerce')
        met = read_csv(track(met_dir/f'{site}.csv'))
        require({'Date', 'SP'}.issubset(met), f'Missing Date/SP: {site}')
        met['Date'] = pd.to_datetime(met.Date, errors='coerce').dt.normalize()
        met = met.dropna(subset=['Date'])
        require(met.Date.is_unique, f'Duplicate meteorological dates: {site}')
        if 'WQS_name' in met:
            require(set(met.WQS_name.dropna()) == {site}, f'Meteorological station mismatch: {site}')
        raw['SP_raw'] = raw.Date.map(pd.to_numeric(met.set_index('Date').SP, errors='coerce'))
        invalid_sp = raw.SP_raw.notna() & (~np.isfinite(raw.SP_raw) | ~raw.SP_raw.between(500, 1100))
        guard = raw.SP_raw.notna().mean() >= match_min and not invalid_sp.any()
        if not guard:
            save('legacy_sp_guard_failure.csv', pd.DataFrame([dict(wq_site_id=site,
                timestamp_records=len(raw), SP_missing_n=int(raw.SP_raw.isna().sum()), SP_invalid_n=int(invalid_sp.sum()),
                SP_match_ratio=float(raw.SP_raw.notna().mean()), action='STOP: original QC guard would reject station; no zero-impact conclusion')]))
            raise ValueError(f'Original SP guard fails at {site}; see GUARD_ERROR rows in qc_order_impact.csv')
        compared = compare_records(raw, scope['calc_do_sat'], bounds)
        d = daily_comparison(compared, DATES);d.insert(0, 'wq_site_id', site)
        saved_paths = [folder/f'{site}.csv' for folder in clean_dirs if (folder/f'{site}.csv').is_file()]
        require(len(saved_paths) == 1, f'Need exactly one official retained/excluded Cleaned file for {site}; run water-quality QC first')
        saved = read_csv(track(saved_paths[0]));saved.Date = pd.to_datetime(saved.Date, errors='raise').dt.normalize()
        require(saved.Date.is_unique and set(saved.Date)==set(DATES), f'Invalid saved QC calendar: {site}')
        saved = saved.set_index('Date').reindex(DATES)
        require('WQS_name' not in saved or set(saved.WQS_name.dropna())=={site}, f'QC station mismatch: {site}')
        same_old = not different(d.old_DO, saved.DO, 2e-6).any()
        same_new = not different(d.new_DO, saved.DO, 2e-6).any()
        require(same_old or same_new, f'Saved QC DO matches neither replay at {site}; check source version/SP/rounding')
        clean_wt = raw.WT_raw.mask(raw.WT_raw.lt(bounds[0]) | raw.WT_raw.gt(bounds[1])).groupby(raw.Date).mean().round(3).reindex(DATES)
        require(not different(clean_wt, saved.WT, 2e-6).any(), f'Saved WT does not reproduce: {site}')
        baseline.append(dict(wq_site_id=site, clean_file=str(saved_paths[0]),
            saved_DO_matches='BOTH' if same_old and same_new else 'OLD' if same_old else 'NEW', WT_reproduced=True))
        links = event_links(events, site, DATES)
        d['event_ids'] = links.to_numpy()
        for field in ['model_feature_row_changed', 'model_target_changed', 'model_input_or_target_changed']:
            d[field] = pd.Series(pd.NA, index=d.index, dtype='boolean')
        for field in ['daily_DO_changed', 'model_input_or_target_changed', 'event_ids']:
            compared[field] = compared.Date.map(d.set_index('Date')[field])
        compared['wq_site_id'] = site;compared['source_file'] = str(p)
        compared['old_effective_SP'] = compared.SP_raw.fillna(1013.25)
        compared['source_quality_class'] = compared['水质'] if '水质' in compared else ''
        compared['source_station_note'] = compared['站点'] if '站点' in compared else ''
        compared['model_check_status'] = 'PENDING' if args.stage=='qc' else 'ASSESSED'
        compared['event_link_status'] = 'NOT_ASSESSED' if events is None else 'ASSESSED'
        logs.append(compared.loc[compared.DO_keep_changed, LOG_FIELDS].copy())
        invalids.append(compared.loc[~compared.WT_valid | ~compared.SP_valid, LOG_FIELDS].copy())
        daily_blocks.append(d)
        summaries.append(dict(wq_site_id=site, source_file=str(p), stage=args.stage, timestamp_records=len(raw),
            outside_study_or_invalid_date_records=int(outside.sum()),
            WT_missing_n=int(raw.WT_raw.isna().sum()),
            WT_invalid_nonmissing_n=int((raw.WT_raw.notna() & ~compared.WT_valid).sum()),
            SP_missing_n=int(raw.SP_raw.isna().sum()), SP_invalid_n=int(invalid_sp.sum()),
            invalid_dependencies_with_absolute_valid_DO_n=int(((~compared.WT_valid | ~compared.SP_valid) & raw.DO_raw.between(0, 20)).sum()),
            old_kept_DO_n=int(compared.old_DO_kept.sum()), new_kept_DO_n=int(compared.new_DO_kept.sum()),
            DO_keep_changed_n=int(compared.DO_keep_changed.sum()), recovered_DO_n=int((~compared.old_DO_kept & compared.new_DO_kept).sum()),
            newly_removed_DO_n=int((compared.old_DO_kept & ~compared.new_DO_kept).sum()), daily_calendar_n=len(d),
            daily_DO_changed_n=int(d.daily_DO_changed.sum()), unrounded_daily_DO_changed_n=int(d.unrounded_daily_DO_changed.sum()),
            observed_flag_changed_n=int(d.observed_flag_changed.sum()), legacy_SP_guard_passed=True,
            saved_QC_matches=baseline[-1]['saved_DO_matches'], saved_WT_reproduced=True,
            saved_QC_file=str(saved_paths[0])))
        print(f'{k}/{len(sites)} {site}: mask changed={compared.DO_keep_changed.sum()}, daily changed={d.daily_DO_changed.sum()}', flush=True)
    summary = pd.DataFrame(summaries);daily = pd.concat(daily_blocks, ignore_index=True)
    models = event_impact = scalers = None
    features = None
    if args.stage == 'full':
        model_path = track(args.model_script or root/'Code/03_DO_reconstruction/02_Model_training/LSTM_V1.py')
        model_tree = ast.parse(model_path.read_text(encoding='utf-8-sig'))
        lstm_dir = Path(args.lstm_dir or constant(model_tree, 'INPUT_DIR', root)).resolve()
        models, event_impact, scalers, features = full_inputs(root, sites, daily, model_path, lstm_dir, events, track, save)
    changed = pd.concat(logs, ignore_index=True)
    invalid = pd.concat(invalids, ignore_index=True)
    if args.stage == 'full':
        lookup = daily.set_index(['wq_site_id', 'Date']).model_input_or_target_changed
        for frame in [changed, invalid]:
            frame['model_input_or_target_changed'] = [lookup.loc[(s, pd.Timestamp(t).normalize())] for s,t in zip(frame.wq_site_id, frame.timestamp)]
    save('qc_order_impact.csv', summary)
    save('changed_record_log.csv', changed)
    save('invalid_dependency_records.csv', invalid)
    save('qc_order_daily_comparison.csv', daily)
    save('baseline_reproduction_checks.csv', pd.DataFrame(baseline))
    si = []
    def row(name, denominator, changed_n, status='ASSESSED', basis='All study station records/dates'):
        si.append(dict(metric=name, compared_n=denominator, changed_n=changed_n, status=status, denominator_definition=basis))
    row('Timestamp_DO_keep_mask', int(summary.timestamp_records.sum()), int(summary.DO_keep_changed_n.sum()))
    row('Daily_DO_rounded_3_decimals', len(daily), int(daily.daily_DO_changed.sum()))
    row('Daily_DO_unrounded', len(daily), int(daily.unrounded_daily_DO_changed.sum()))
    row('Daily_original_DO_observation_flag', len(daily), int(daily.observed_flag_changed.sum()))
    if models is None:
        row('Model_inputs_and_targets', pd.NA, pd.NA, 'NOT_ASSESSED', 'Run --stage full after current LSTM inputs exist')
        row('Event_observed_DO_inputs', pd.NA, pd.NA, 'NOT_ASSESSED', 'Run --stage full with current event catalogue')
    else:
        for split, g in models.groupby('model_split', sort=False):
            row(split+'_standardized_input_windows', int(g.compared_union_n.sum()), int(g.standardized_feature_window_changed_n.sum()), basis='Union of old/new eligible samples; final/development cohorts overlap')
            if split != 'Missing_DO_prediction':
                row(split+'_raw_DO_targets', int(g.compared_union_n.sum()), int(g.raw_target_changed_n.sum()), basis='Union of old/new eligible observed targets')
        row('Event_period_observed_DO_days', int(event_impact.event_calendar_days.sum()), int(event_impact.event_DO_input_days_changed.sum()), basis='Event calendar days; observed daily input only')
        row('Events_with_observed_DO_input_change', len(event_impact), int(event_impact.event_DO_input_days_changed.gt(0).sum()), basis='Current registered site-events; not recomputed Shock/slopes')
    save('si_qc_order_summary.csv', pd.DataFrame(si))
    input_rows = [dict(source_file=p, SHA256=h) for p,h in sources.items()]
    require(all(sha(p)==h for p,h in sources.items()), 'A source file changed during the audit')
    qc_zero = not daily.daily_DO_changed.any() and not daily.observed_flag_changed.any()
    downstream_zero = None if models is None else bool(qc_zero and not models.standardized_feature_window_changed_n.any()
        and not models.standardized_target_changed_n.dropna().any() and not models.sample_membership_changed_n.any())
    decision = 'QC_STAGE_COMPLETE_DOWNSTREAM_NOT_ASSESSED' if models is None else 'NO_QC_ORDER_DOWNSTREAM_INPUT_CHANGE' if downstream_zero else 'INPUT_CHANGE_DETECTED_REVIEW_AND_RERUN_FIXED_PIPELINE'
    return dict(Status='COMPLETED', Stage=args.stage, Detail_Output=args.details, Decision=decision, Study_Stations=len(sites),
        Timestamp_Records=int(summary.timestamp_records.sum()), Daily_Station_Dates=len(daily),
        DO_Mask_Changed=int(summary.DO_keep_changed_n.sum()), Daily_DO_Changed=int(daily.daily_DO_changed.sum()),
        Observation_Flag_Changed=int(daily.observed_flag_changed.sum()), Zero_Daily_Input_Impact=bool(qc_zero),
        Zero_Downstream_Input_Impact=downstream_zero, Meteorology_Directory=str(met_dir),
        QC_Script=str(qc), WT_Valid_Range=list(bounds), SP_Valid_Range=[500,1100],
        DO_Absolute_Range=[0,20], Features=features, Sources=input_rows,
        Daily_Rounding_Decimals=3, Difference_Tolerance_Unrounded=1e-12, Saved_QC_Reproduction_Tolerance=2e-6,
        Comparison_Scope='Only QC order differs; both branches hold the same current meteorology and all other inputs fixed.',
        Standardization='Full mode compares original development/final StandardScaler preprocessing; temporary scalers are not saved to model assets.',
        Model_Log_Flag='Same-date raw target or prepared unscaled feature row change; window/scaling propagation is reported separately.',
        Event_Scope='Observed daily event and candidate-reference inputs; reconstructed DO and Shock/slopes are not recomputed.',
        Raw_Labels='Source water-quality class and station notes are provenance only, not new DO deletion rules.',
        Production_QC_Changed=False, Research_Inputs_Changed=False, Model_Trained=False,
        Event_Response_Or_Significance_Test_Recomputed=False,
        Meteorology_Rerun_Note='Zero QC-order impact does not cancel training reruns required by the independent TP_M/SSR correction.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=PROJECT)
    p.add_argument('--stage', choices=['qc', 'full'], default='qc')
    p.add_argument('--details', choices=['changes', 'all', 'none'], default='changes',
                   help='Controls diagnostic ROWS in qc_order_impact.csv; output always remains two CSVs')
    p.add_argument('--qc-script', type=Path)
    p.add_argument('--model-script', type=Path)
    p.add_argument('--raw-dir', type=Path)
    p.add_argument('--meteo-dir', type=Path, help='Default is ERA5_SP_FOLDER read from the current QC source; no fallback to older meteorology')
    p.add_argument('--lstm-dir', type=Path)
    p.add_argument('--events-file', type=Path)
    p.add_argument('--output-root', type=Path, help='Exact output folder within Result; no timestamp subfolder; replaces the two audit CSVs')
    return p.parse_args()


def main():
    args = parse_args()
    result_root = (args.root.resolve()/'Result').resolve()
    output_root = (args.output_root or result_root/'M2b_QC_Order').resolve()
    require(output_root.is_relative_to(result_root), 'All audit CSVs must stay under this project Result directory')
    now = datetime.now(timezone(timedelta(hours=8)))
    out = output_root
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')
    initial = dict(Status='RUNNING', Stage=args.stage, Started_At=now.isoformat(), Output_Directory=str(out),
                   Script_SHA256=sha(__file__))
    report = AuditReport(out)
    report.write(initial)
    try:
        result = execute(args, out, report)
        result.update(Started_At=initial['Started_At'], Output_Directory=str(out), Script_SHA256=initial['Script_SHA256'])
        report.write(result)
        for key in ['Status','Stage','Decision','DO_Mask_Changed','Daily_DO_Changed','Zero_Downstream_Input_Impact','Output_Directory']:
            print(f'{key}: {result[key]}')
    except Exception as exc:
        initial.update(Status='FAILED', Error_Type=type(exc).__name__, Error=str(exc),
                       SI_Summary_Usable=False, Note='No zero-impact conclusion is valid for a failed/incomplete run.')
        initial['Error_Traceback'] = traceback.format_exc()
        report.write(initial)
        raise


if __name__=='__main__':
    main()
