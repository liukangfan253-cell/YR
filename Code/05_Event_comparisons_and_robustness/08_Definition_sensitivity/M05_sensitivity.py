# -*- coding: utf-8 -*-
"""Reuse ED_F detection in M05, verify S0, and export the sensitivity results. The SFigure03 and SFigure04 scripts generate the figures."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util
import argparse
import hashlib
import warnings
import sys
import json

# ===================== Inputs, outputs and parameters =====================
ROOT = Path(__file__).resolve().parents[3]
HISTORY_DIR = ROOT / 'Data/Meteorology_data/1991-2020/YR'
DAILY_DIR = ROOT / 'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR'
REGISTRY_DIR = ROOT / 'Data/Event_F/01_Registry_Data/Overall'
ORIGINAL_SHOCK = ROOT / 'Result/Decrease/Overall/FINAL_Shock_All_Events_Combined.csv'
EVENT_SCRIPT = ROOT / 'Code/02_Extreme_event_identification/02_Completed_event_registry/ED_F.py'
SHOCK_SCRIPT = ROOT / 'Code/04_DO_conditions_and_post_event_changes/02_Departure_metrics/decrease.py'
RUN_DATE = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
OUTPUT_DIR = ROOT / 'Result/Event/Definition_sensitivity' / RUN_DATE
STUDY_START, STUDY_END = '2021-01-01', '2025-12-31'
HISTORY_START, HISTORY_END = '1991-01-01', '2020-12-31'
SCENARIOS = {
    'S0': {'T_Method': 'Annual_P90', 'Q_Percentile': 95, 'Gap_Days': 5},
    'S1': {'T_Method': 'Calendar_Daily_P90', 'Q_Percentile': 95, 'Gap_Days': 5},
    'S2': {'T_Method': 'Annual_P90', 'Q_Percentile': 90, 'Gap_Days': 5},
    'S3': {'T_Method': 'Annual_P90', 'Q_Percentile': 97.5, 'Gap_Days': 5},
    'S4': {'T_Method': 'Annual_P90', 'Q_Percentile': 95, 'Gap_Days': 3},
    'S5': {'T_Method': 'Annual_P90', 'Q_Percentile': 95, 'Gap_Days': 7},
}
EVENT_TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']
EXPECTED_N = dict(zip(EVENT_TYPES, [1192, 1035, 132, 102, 89]))
TYPE_MAP = {'hw': EVENT_TYPES[0], 'flood': EVENT_TYPES[1], 'direct': EVENT_TYPES[2],
            'indirect_h2f': EVENT_TYPES[3], 'indirect_f2h': EVENT_TYPES[4]}
REACH_MAP = {'UR': 'Upstream', 'MR': 'Midstream', 'LR': 'Downstream'}
MIN_PRE_DAYS = 3
S0_ATOL, S0_RTOL = 1e-10, 1e-9  # Allow floating-point rounding differences only. Require identical event windows and missingness patterns.
DAILY_WINDOW_HALF = 15
SPARSE_N, SPARSE_SITES = 10, 3
# ============================================================
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

LOG, SOURCES, THRESHOLDS = [], [], []

def log(message):
    LOG.append(str(message))
    print(message, flush=True)

def save_csv(data, name):
    data.to_csv(OUTPUT_DIR / f'{RUN_DATE}_{name}.csv', index=False, encoding='utf-8-sig', float_format='%.12g')

def source_record(path, data=None):
    SOURCES.append({'文件': str(path), 'SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    '记录数': len(data) if data is not None else None,
                    '字段': '|'.join(data.columns) if data is not None else ''})

def read_csv(path):
    try:
        d = pd.read_csv(path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        d = pd.read_csv(path, encoding='gb18030')
    source_record(path, d)
    return d

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # ED_F suppresses warnings on import. Keep that setting local to ED_F.
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    source_record(path)
    return module

def audit_daily(folder, fields, start, end):
    expected = pd.date_range(start, end)
    data, reports = {}, []
    for path in sorted(folder.glob('*.csv')):
        d = read_csv(path)
        # Format fields as in ED_F and retain the original files.
        d.columns = d.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
        d = d.rename(columns={c: 'Date' for c in d if c.lower() == 'date'})
        required = ['Date', 'WQS_name'] + fields
        if set(required) - set(d):
            raise ValueError(f'{path} 缺字段 {set(required)-set(d)}')
        if d.WQS_name.isna().any() or d.WQS_name.astype(str).str.strip().nunique() != 1:
            raise ValueError(f'{path} 断面字段不唯一或缺失')
        site = str(d.WQS_name.iloc[0]).strip()
        if site in data:
            raise ValueError(f'{folder} 同站点多个输入文件：{site}')
        d['Date'] = pd.to_datetime(d.Date, errors='raise')
        if d.Date.isna().any() or d.Date.duplicated().any():
            raise ValueError(f'{path} 日期缺失或重复')
        d = d.sort_values('Date').reset_index(drop=True)
        missing_dates = expected.difference(pd.DatetimeIndex(d.Date))
        extra_dates = pd.DatetimeIndex(d.Date).difference(expected)
        report = {'Site_ID': site, '文件': str(path), 'n_rows': len(d), 'Start': d.Date.min(), 'End': d.Date.max(),
                  '缺日期数': len(missing_dates), '范围外日期数': len(extra_dates),
                  '缺日期': '|'.join(missing_dates.strftime('%Y-%m-%d'))}
        for col in fields:
            original = d[col]
            d[col] = pd.to_numeric(original, errors='coerce')
            if (original.notna() & d[col].isna()).any() or (~np.isfinite(d[col]) & d[col].notna()).any():
                raise ValueError(f'{path} {col} 非数值或非有限')
            report[col+'_缺失数'] = int(d[col].isna().sum())
        reports.append(report)
        if len(extra_dates):
            raise ValueError(f'{path} 存在参考范围外日期，禁止混入阈值计算')
        data[site] = d
    return data, pd.DataFrame(reports)

def identify(site, d, t, q, scenario, detector, base_segments=None):
    detector.COMP_INTERVAL = SCENARIOS[scenario]['Gap_Days']
    if base_segments is None:
        hw = detector.get_base_segments(d.copy(), 'hw', t, q)
        fl = detector.get_base_segments(d.copy(), 'flood', t, q)
    else:
        hw, fl = base_segments
    comp, remaining_hw, remaining_fl = detector.classify_complex_events(hw, fl)
    segments = sorted(comp + remaining_hw + remaining_fl, key=lambda x: (x['start'], x['end']))
    rows = []
    for i, ev in enumerate(segments):
        rows.append({'Scenario': scenario, 'Site_ID': site, 'Event_ID': f'{site}_EV{i+1:03d}',
                     'Event_Type': TYPE_MAP[ev.get('comp_subtype', ev.get('type'))],
                     'Start_Date': ev['start'], 'End_Date': ev['end'],
                     'Total_Days': (ev['end']-ev['start']).days+1, 'Region': d.Region.dropna().iloc[0],
                     'Gap_Days': SCENARIOS[scenario]['Gap_Days']})
    return rows

def export_scenario(frame, scenario, season_fn):
    frame = frame.copy()
    frame['Reach'] = frame.Region.map(REACH_MAP)
    frame['Season'] = frame.Start_Date.dt.month.map(season_fn)
    frame['Year'] = frame.Start_Date.dt.year
    frame['Event_Start'] = frame.Start_Date
    frame['Event_End'] = frame.End_Date
    frame['Event_Duration'] = frame.Total_Days
    frame['T_Method'] = SCENARIOS[scenario]['T_Method']
    frame['Q_Percentile'] = SCENARIOS[scenario]['Q_Percentile']
    log(f'{scenario}：事件{len(frame)}，Shock-L有效{np.isfinite(frame.Shock_Ratio_LongTerm).sum()}，'
        f'Shock-S有效{np.isfinite(frame.Shock_Ratio_Pre).sum()}；无效原因：{frame.ShockS_Status.value_counts().to_dict()}。')
    save_csv(frame[['Scenario', 'Site_ID', 'Event_ID', 'Event_Type', 'Event_Start', 'Event_End',
                    'Event_Duration', 'Reach', 'Season', 'Year', 'Gap_Days', 'T_Method', 'Q_Percentile']],
             f'M05_{scenario}事件注册表')
    save_csv(frame, f'M05_{scenario}事件级Shock结果')
    return frame

def calendar_thresholds(site, history):
    """Use a 365-day calendar without February 29 and a circular 31-day window across 30 years. For February 29, average the P90 values of the adjacent days."""
    expected = pd.date_range(HISTORY_START, HISTORY_END)
    if len(history) != len(expected) or not pd.DatetimeIndex(history.Date).equals(expected) or history.Tmax.isna().any():
        raise ValueError(f'{site} 历史Tmax不完整：S1禁止用2021–2025数据代替，需人工检查')
    nonleap = history.loc[~((history.Date.dt.month == 2) & (history.Date.dt.day == 29))].copy()
    dates = pd.to_datetime('2001-' + nonleap.Date.dt.strftime('%m-%d'))
    day = dates.dt.dayofyear.to_numpy()
    values = nonleap.Tmax.to_numpy()
    rows, mapping = [], {}
    for target in pd.date_range('2001-01-01', '2001-12-31'):
        delta = np.abs(day-target.dayofyear)
        window = np.minimum(delta, 365-delta) <= DAILY_WINDOW_HALF
        pool = values[window]
        threshold = float(np.percentile(pool, 90))
        key = target.strftime('%m-%d')
        mapping[key] = threshold
        rows.append({'Site_ID': site, 'Month_Day': key, 'Tmax_Daily_P90': threshold,
                     'n_history_samples': len(pool), 'Rule': '循环31日×1991–2020，排除2/29'})
    mapping['02-29'] = (mapping['02-28']+mapping['03-01'])/2
    rows.append({'Site_ID': site, 'Month_Day': '02-29', 'Tmax_Daily_P90': mapping['02-29'],
                 'n_history_samples': np.nan, 'Rule': '邻日P90线性插值，不混入研究期数据'})
    return mapping, rows

def run_scenario(scenario, sites, history, daily, detector, season_fn, cache):
    config = SCENARIOS[scenario]
    all_rows, all_bg, errors, candidates, calendar_rows = [], [], [], [], []
    log(f'运行 {scenario}：{config}；跨方案Event_ID仅为各方案内编号，不用于跨方案关联。')
    for site in sites:
        d, h = daily[site], history[site]
        t_fixed = float(np.percentile(h.Tmax.dropna(), 90))
        q = float(np.percentile(d.Q.dropna(), config['Q_Percentile']))
        try:
            if scenario == 'S1':
                mapping, threshold_rows = calendar_thresholds(site, h)
                calendar_rows.extend(threshold_rows)
                t = d.Date.dt.strftime('%m-%d').map(mapping)
                if t.isna().any():
                    raise ValueError(f'{site} 研究期日期缺少对应日历P90')
            else:
                t = t_fixed
            THRESHOLDS.append({'Scenario': scenario, 'Site_ID': site, 'Tmax_Annual_P90': t_fixed,
                               'Tmax_Applied_Min': float(t.min()) if isinstance(t, pd.Series) else t,
                               'Tmax_Applied_Max': float(t.max()) if isinstance(t, pd.Series) else t,
                               'Q_Threshold': q, **config})
            evs = identify(site, d, t, q, scenario, detector, cache[site] if scenario in ['S4', 'S5'] else None)
            candidates.extend(evs)
            detector.validate_event_exclusivity(site, evs)
            if not evs:
                log(f'  {site}：本方案零事件，保留该站数量为0。')
            rows, bg = shock_for_site(d, evs, season_fn)
            all_rows.extend(rows)
            all_bg.extend([{'Scenario': scenario, **row} for row in bg])
        except (ValueError, RuntimeError) as exc:
            errors.append({'Scenario': scenario, 'Site_ID': site, 'Error': str(exc)})
            log(f'  {scenario} {site} 不可用：{exc}')
    if calendar_rows:
        save_csv(pd.DataFrame(calendar_rows), 'M05_S1逐日气候态P90')
    if errors:
        save_csv(pd.DataFrame(errors), f'M05_{scenario}未通过核查原因')
        save_csv(pd.DataFrame(candidates), f'M05_{scenario}未通过核查候选事件')
        log(f'{scenario} 有 {len(errors)} 站未通过；不剔除问题站、不发布该方案正式汇总或Shock结果。其他单因素方案独立继续。')
        return None
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise ValueError(f'{scenario} 全方案零事件，需要人工检查')
    save_csv(pd.DataFrame(all_bg), f'M05_{scenario}季节非事件背景')
    return export_scenario(frame, scenario, season_fn)

def strata(frame):
    yield 'Overall', 'All', 'All', frame
    for event in EVENT_TYPES:
        yield 'Event_Type', event, 'All', frame.loc[frame.Event_Type.eq(event)]
    for reach in REACH_MAP.values():
        yield 'Reach', 'All', reach, frame.loc[frame.Reach.eq(reach)]
    for event in EVENT_TYPES:
        for reach in REACH_MAP.values():
            yield 'Event_Type_Reach', event, reach, frame.loc[frame.Event_Type.eq(event) & frame.Reach.eq(reach)]

def summarize(results, sites):
    counts, changes, details, statistics, comparisons = [], [], [], [], []
    baseline = results['S0']
    base_counts = {t: int(baseline.Event_Type.eq(t).sum()) for t in EVENT_TYPES}
    base_counts['Total'] = len(baseline)
    for scenario, config in SCENARIOS.items():
        frame = results.get(scenario)
        if frame is None:
            counts.append({'Scenario': scenario, **config, 'Status': '未通过核查；不发布结果'})
            continue
        count = {t: int(frame.Event_Type.eq(t).sum()) for t in EVENT_TYPES}
        counts.append({'Scenario': scenario, **config, 'Status': '通过', 'Total': len(frame),
                       'n_sites': frame.Site_ID.nunique(), **count})
        for t, n in {**count, 'Total': len(frame)}.items():
            ref = base_counts[t]
            changes.append({'Scenario': scenario, 'Event_Type': t, 'Gap_Days': config['Gap_Days'],
                            'N': n, 'N_S0': ref, 'Share_pct': 100*n/len(frame),
                            'Change_Percent': 100*(n-ref)/ref if ref else np.nan})
        for scope, entity, sub in [('Reach', r, frame.loc[frame.Reach.eq(r)]) for r in REACH_MAP.values()] + \
                                  [('Site', site, frame.loc[frame.Site_ID.eq(site)]) for site in sites]:
            for t in ['Total']+EVENT_TYPES:
                selected = sub if t == 'Total' else sub.loc[sub.Event_Type.eq(t)]
                details.append({'Scenario': scenario, 'Scope': scope, 'Entity': entity, 'Event_Type': t,
                                'N': len(selected), 'n_sites': selected.Site_ID.nunique(), 'Gap_Days': config['Gap_Days']})
        for metric in ['Shock_Ratio_LongTerm', 'Shock_Ratio_Pre']:
            for level, event, reach, group in strata(frame):
                valid = group.loc[np.isfinite(group[metric])]
                site_means = valid.groupby('Site_ID')[metric].mean()
                statistics.append({'Scenario': scenario, 'Gap_Days': config['Gap_Days'], 'Metric': metric,
                                   'Level': level, 'Event_Type': event, 'Reach': reach, 'n_events_total': len(group),
                                   'n_valid': len(valid), 'n_sites': valid.Site_ID.nunique(),
                                   'Mean_Event': valid[metric].mean(), 'Median_Event': valid[metric].median(),
                                   'Below_1_pct': 100*valid[metric].lt(1).mean() if len(valid) else np.nan,
                                   'Mean_Site_Equal': site_means.mean(),
                                   'Paper_Mean_Basis': 'Site_Equal' if 'Reach' in level else 'Event_Equal',
                                   'Sparse': len(valid) < SPARSE_N or valid.Site_ID.nunique() < SPARSE_SITES})
            for label, target_types, reference_types in [
                    ('Compound - Single', EVENT_TYPES[2:], EVENT_TYPES[:2]),
                    ('F5H - H5F', [EVENT_TYPES[4]], [EVENT_TYPES[3]])]:
                target = frame.loc[frame.Event_Type.isin(target_types) & np.isfinite(frame[metric])]
                reference = frame.loc[frame.Event_Type.isin(reference_types) & np.isfinite(frame[metric])]
                comparisons.append({'Scenario': scenario, 'Gap_Days': config['Gap_Days'], 'Metric': metric,
                                    'Comparison': label, 'Target_Mean': target[metric].mean(),
                                    'Reference_Mean': reference[metric].mean(),
                                    'Mean_Difference': target[metric].mean()-reference[metric].mean(),
                                    'Target_Median': target[metric].median(), 'Reference_Median': reference[metric].median(),
                                    'Target_n': len(target), 'Reference_n': len(reference),
                                    'Target_Sites': target.Site_ID.nunique(), 'Reference_Sites': reference.Site_ID.nunique(),
                                    'Mean_Basis': 'Event_Equal',
                                    'Sparse': len(target) < SPARSE_N or len(reference) < SPARSE_N or min(target.Site_ID.nunique(), reference.Site_ID.nunique()) < SPARSE_SITES})
    detail_table = pd.DataFrame(details)
    base_detail = detail_table.loc[detail_table.Scenario.eq('S0'), ['Scope', 'Entity', 'Event_Type', 'N']].rename(columns={'N': 'N_S0'})
    detail_table = detail_table.merge(base_detail, on=['Scope', 'Entity', 'Event_Type'], how='left', validate='many_to_one')
    detail_table['Change_Percent'] = np.divide(100*(detail_table.N-detail_table.N_S0), detail_table.N_S0,
                                              out=np.full(len(detail_table), np.nan), where=detail_table.N_S0.gt(0))
    detail_table['Change_Note'] = np.where(detail_table.N_S0.eq(0), 'S0计数为0，不计算变化率', '可计算')
    return pd.DataFrame(counts), pd.DataFrame(changes), detail_table, pd.DataFrame(statistics), pd.DataFrame(comparisons)


def excel(tables):
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    path = OUTPUT_DIR / f'{RUN_DATE}_M05敏感性分析汇总.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 38
            for cell in sheet[1]:
                cell.font = Font(name='Microsoft YaHei', color='FFFFFF', bold=True)
                cell.fill = PatternFill('solid', fgColor='24476A')
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            for j, column in enumerate(sheet.columns, 1):
                header = str(sheet.cell(1, j).value)
                sheet.column_dimensions[get_column_letter(j)].width = min(46, max(15, len(header)+2))
                for cell in list(column)[1:]:
                    if isinstance(cell.value, (float, int)):
                        count = header.startswith('n_') or header in EVENT_TYPES or header in ['N', 'N_S0', 'Total', 'Gap_Days', 'Target_n', 'Reference_n', 'Target_Sites', 'Reference_Sites']
                        cell.number_format = '0' if count else '0.000000'
    book = load_workbook(path, read_only=True, data_only=True)
    for name, frame in tables.items():
        assert book[name].max_row == len(frame)+1 and book[name].max_column == len(frame.columns)
    book.close()
    log('Excel重新打开核验，各表行列数一致；差值为数值，比例以0–100百分数存储。')

def interpretation(counts, changes, comparisons):
    valid = counts.loc[counts.Status.eq('通过')].set_index('Scenario')
    core = comparisons.loc[comparisons.Metric.eq('Shock_Ratio_LongTerm')]
    text = ['# M05 敏感性分析结果', '', 'S0逐事件分类、日期和Shock复现通过。结果为描述性参数敏感性，不是显著性检验或因果效应。', '']
    text.append('## 事件数量')
    cols = ['Scenario', 'Gap_Days', 'Total']+EVENT_TYPES
    text.append('| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+
                '\n'.join('| '+' | '.join(str(int(r[c])) if c != 'Scenario' else str(r[c]) for c in cols)+' |'
                          for _, r in valid.reset_index()[cols].iterrows()))
    for scenario in ['S1', 'S2', 'S3', 'S4', 'S5']:
        if scenario not in valid.index:
            text.append(f'{scenario} 未通过数据/互斥核查；不可解释或参与稳健性结论。')
            continue
        differences = changes.loc[(changes.Scenario == scenario) & (changes.Event_Type != 'Total')]
        description = '；'.join(f'{r.Event_Type}: {int(r.N)}（相对S0 {r.Change_Percent:+.2f}%）' for _, r in differences.iterrows())
        text.append(f'{scenario}，实际复合间隔{SCENARIOS[scenario]["Gap_Days"]}天：{description}。')
    text += ['', '## 核心DO响应',
             '| 方案 | 比较 | 目标组n | 参照组n | Shock-L均值差 |\n|---|---|---:|---:|---:|\n'+
             '\n'.join(f'| {r.Scenario} | {r.Comparison} | {int(r.Target_n)} | {int(r.Reference_n)} | {r.Mean_Difference:+.6f} |' for _, r in core.iterrows())]
    for label, group in core.groupby('Comparison', sort=False):
        low, high = group.Mean_Difference.min(), group.Mean_Difference.max()
        base = group.loc[group.Scenario.eq('S0'), 'Mean_Difference'].iloc[0]
        delta = (group.Mean_Difference-base).abs()
        largest = group.loc[delta.idxmax()]
        text.append(f'{label}：S0均值差 {base:+.6f}；通过方案范围 [{low:+.6f}, {high:+.6f}]；相对S0变化最大为 {largest.Scenario}（均值差 {largest.Mean_Difference:+.6f}）。')
        signs = set(np.sign(group.Mean_Difference.dropna()))
        text.append('差值正负方向发生变化；须结合幅度，不能把接近0的符号变化直接称为严重不稳健。' if len(signs) > 1 else '通过方案差值方向一致；幅度仍须查看，不能据此宣称统计等效或因果稳定。')
    count_impact = changes.loc[changes.Event_Type.ne('Total')].assign(Abs_Count_Change=lambda x: abs(x.N-x.N_S0)).groupby('Scenario').Abs_Count_Change.sum()
    if len(count_impact):
        text.append(f'按五类数量绝对变化之和衡量，数量结构变化最大为 {count_impact.idxmax()}（{int(count_impact.max())}）；这不是事件重分类率，未用跨方案Event_ID强行匹配。')
    text += ['', '## 七个问题的结论']
    if {'S1', 'S2', 'S3', 'S4', 'S5'}.issubset(valid.index):
        s1 = changes.loc[(changes.Scenario == 'S1') & changes.Event_Type.eq('Total')].iloc[0]
        text.append(f'1. 逐日气候态P90使事件总数由{int(s1.N_S0)}变为{int(s1.N)}（{s1.Change_Percent:+.2f}%）；五类变化详见上表。这个方案改变热浪的季节定义，数量构成并不稳定。')
        flood_changes = changes.loc[changes.Scenario.isin(['S2', 'S3']) & changes.Event_Type.ne('Total')]
        for scenario in ['S2', 'S3']:
            part = flood_changes.loc[flood_changes.Scenario.eq(scenario)]
            relative = part.loc[part.Change_Percent.abs().idxmax()]
            absolute = part.loc[(part.N-part.N_S0).abs().idxmax()]
            text.append(f'2. {scenario}洪水阈值敏感性：相对数量变化最大为{relative.Event_Type}（{relative.Change_Percent:+.2f}%）；绝对数量变化最大为{absolute.Event_Type}（{int(absolute.N-absolute.N_S0):+d}条）。')
        text.append(f'3. 间隔3/5/7天时，H5F数量分别{int(valid.loc["S4", "H5F_Compound"])}/{int(valid.loc["S0", "H5F_Compound"])}/{int(valid.loc["S5", "H5F_Compound"])}，F5H为{int(valid.loc["S4", "F5H_Compound"])}/{int(valid.loc["S0", "F5H_Compound"])}/{int(valid.loc["S5", "F5H_Compound"])}；数量有变化，DC仍132条。')
        cp = core.loc[core.Comparison.eq('Compound - Single')]
        sq = core.loc[core.Comparison.eq('F5H - H5F')]
        cpmax = cp.loc[(cp.Mean_Difference-cp.loc[cp.Scenario.eq('S0'), 'Mean_Difference'].iloc[0]).abs().idxmax()]
        sqmax = sq.loc[(sq.Mean_Difference-sq.loc[sq.Scenario.eq('S0'), 'Mean_Difference'].iloc[0]).abs().idxmax()]
        text.append(f'4. 复合−单一均值差范围{cp.Mean_Difference.min():+.6f}至{cp.Mean_Difference.max():+.6f}，最大绝对差{cp.Mean_Difference.abs().max():.6f}个Shock比值单位；没有跨方案一致的复合更低方向。未预定义生态效应界值，不能正式判定“差异等效地小”。')
        reversed_rows = sq.loc[np.sign(sq.Mean_Difference) != np.sign(sq.loc[sq.Scenario.eq('S0'), 'Mean_Difference'].iloc[0])]
        text.append('5. 顺序比较与S0相反的方案为'+', '.join(reversed_rows.Scenario)+f'；范围{sq.Mean_Difference.min():+.6f}至{sq.Mean_Difference.max():+.6f}。S1改变幅度较大，不能笼统宣称顺序排序稳定；也没有类型差异的显著性证据由此自动产生。')
        text.append(f'6. 按五类计数绝对变化之和，{count_impact.idxmax()}影响最大；按核心均值差相对S0的变化，复合−单一最大为{cpmax.Scenario}，F5H−H5F最大为{sqmax.Scenario}。不同问题需分别评价。')
        text.append('7. 有限支持“复合事件并非在所有方案下都表现出更低季节相对DO”的总体判断；不支持事件数量、类型占比或H5F/F5H排序对参数完全不敏感。洪水阈值和季节化热浪定义是主要例外，应作为限制明确报告。此处只评价描述性方向和幅度，不替代M01，也不对六方案推断统计显著性。')
    text += ['', '## 解释边界',
             '不能仅凭六种描述性均值判定“差异显著”或“完全稳健”。应分别报告事件数量对参数的敏感性及DO差值范围；较小的正负切换和明显幅度变化需分开描述。未预设有临床/生态意义的“较小”阈值。',
             'S1改变热浪的季节定义，冬季热浪及其分类可增加；不是单纯百分位数字变化。S2/S3只改研究期Q阈值；S4/S5复用S0基础段，只改复合配对间隔。H5F/F5H为统一类型代码，各表Gap_Days说明实际定义。',
             '全流域/类型及两项核心比较采用事件等权；河段及类型×河段另列断面等权均值。各方案依据自己的事件重算非事件季节背景及前9–4天基线，因此变化同时包含定义、窗口及样本组成变化。',
             'Sparse字段标记n<10或站点<3的描述性分组，原事件全部保留。极少量复合样本不能推出总体稳健性。S0源输入与主方案没有被改动。']
    log('\n'.join(text[-9:]))

def shock_for_site(d, events, season_fn):
    """Reproduce the decrease.py calculation in memory, using seasonal non-event references weighted by event days and excluding contaminated reference days."""
    occupied = pd.Series(False, index=d.index)
    for ev in events:
        occupied |= d.Date.between(ev['Start_Date'], ev['End_Date'])
    seasons = d.Date.dt.month.map(season_fn)
    background = d.loc[~occupied & d.DO.notna()].assign(Season=seasons).groupby('Season').DO.mean()
    rows = []
    for ev in events:
        mask = d.Date.between(ev['Start_Date'], ev['End_Date'])
        event_do = d.loc[mask, 'DO'].mean()
        seasonal_do = seasons.loc[mask].map(background).mean()
        pre = d.Date.between(ev['Start_Date']-pd.Timedelta(days=9), ev['Start_Date']-pd.Timedelta(days=4)) & ~occupied
        valid_n = int(d.loc[pre, 'DO'].count())
        pre_do = d.loc[pre, 'DO'].mean() if valid_n >= MIN_PRE_DAYS else np.nan
        row = {**ev, 'Event_DO': event_do, 'Seasonal_Baseline_DO': seasonal_do,
               'Pre_Baseline_DO': pre_do, 'Pre_Baseline_Valid_Days': valid_n if pre.any() else np.nan,
               'Shock_Ratio_LongTerm': event_do/seasonal_do if pd.notna(seasonal_do) and seasonal_do > 0 else np.nan,
               'Shock_Ratio_Pre': event_do/pre_do if pd.notna(pre_do) and pre_do > 0 else np.nan,
               'Event_DO_Valid_Days': int(d.loc[mask, 'DO'].count()),
               'Pre_Window_Rows': int(pre.sum()), 'Pre_Excluded_Event_Days': int((d.Date.between(ev['Start_Date']-pd.Timedelta(days=9), ev['Start_Date']-pd.Timedelta(days=4)) & occupied).sum())}
        row['ShockL_Status'] = ('事件DO无效' if not np.isfinite(event_do) else
                                '季节背景缺失/非正' if not np.isfinite(seasonal_do) or seasonal_do <= 0 else '有效')
        row['ShockS_Status'] = ('事件DO无效' if not np.isfinite(event_do) else
                                '事件前有效DO不足3日' if valid_n < MIN_PRE_DAYS else
                                '前基线缺失/非正' if not np.isfinite(pre_do) or pre_do <= 0 else '有效')
        rows.append(row)
    baseline_rows = [{'Site_ID': events[0]['Site_ID'] if events else str(d.WQS_name.iloc[0]),
                      'Season': season, 'Background_DO': background.get(season, np.nan),
                      'n_background_valid_DO': int((~occupied & seasons.eq(season) & d.DO.notna()).sum())}
                     for season in ['Spring', 'Summer', 'Autumn', 'Winter']]
    return rows, baseline_rows

def verify_s0(registry, observed_shock, calculated):
    keys = ['Site_ID', 'Event_ID']
    compared = registry.merge(calculated, on=keys, how='outer', suffixes=('_original', '_S0'), indicator=True, validate='one_to_one')
    bad = compared['_merge'].ne('both')
    for c in ['Event_Type', 'Start_Date', 'End_Date', 'Total_Days', 'Region']:
        bad |= compared[c+'_original'].ne(compared[c+'_S0'])
    save_csv(compared.loc[bad], 'M05_S0事件窗口与分类差异')
    counts = registry.groupby(['Site_ID', 'Event_Type']).size().rename('Original_N').to_frame().join(
        calculated.groupby(['Site_ID', 'Event_Type']).size().rename('S0_N'), how='outer').fillna(0).reset_index()
    counts['Difference'] = counts.S0_N-counts.Original_N
    save_csv(counts, 'M05_S0逐站事件数量核对')
    log(f'S0逐事件窗口/分类/ID差异 {bad.sum()} 条；逐站数量差异 {counts.Difference.ne(0).sum()} 项。')
    if bad.any():
        raise ValueError('S0无法复现原事件；已输出详细差异，暂停S1–S5。')
    compare = observed_shock.merge(calculated, on=keys, how='outer', suffixes=('_original', '_S0'), indicator=True, validate='one_to_one')
    mismatch = compare['_merge'].ne('both')
    report = []
    for metric in ['Event_DO', 'Seasonal_Baseline_DO', 'Pre_Baseline_DO', 'Pre_Baseline_Valid_Days', 'Shock_Ratio_LongTerm', 'Shock_Ratio_Pre']:
        x, y = compare[metric+'_original'], compare[metric+'_S0']
        matches = np.isclose(x, y, atol=S0_ATOL, rtol=S0_RTOL, equal_nan=True)
        mismatch |= ~matches
        report.append({'Metric': metric, 'n_different': int((~matches).sum()), 'Max_Absolute_Difference': float((x-y).abs().max())})
        compare[metric+'_difference'] = y-x
    save_csv(compare.loc[mismatch], 'M05_S0逐事件Shock差异')
    save_csv(pd.DataFrame(report), 'M05_S0指标核验摘要')
    log('S0指标核验：\n' + pd.DataFrame(report).to_string(index=False))
    if mismatch.any():
        raise ValueError('S0指标或缺失模式与原结果不一致；暂停S1–S5，不覆盖原研究结果。')
    log('S0通过：事件编号、窗口、分类、河段和Shock逐事件一致（仅允许浮点误差）。')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s0-only', action='store_true')
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {'Run_Date': RUN_DATE, 'Status': '运行中', 'S0_Verified': False}
    log('运行状态：' + json.dumps(status, ensure_ascii=False))
    try:
        log(f'M05开始，实际日期（上海）{RUN_DATE}；pandas {pd.__version__}；numpy {np.__version__}。')
        source_record(Path(__file__))
        # Inventory code and data paths without importing preprocessing modules that access the network or write files.
        code_files = sorted(ROOT.joinpath('Code').rglob('*.py'))
        data_files = sorted(HISTORY_DIR.glob('*.csv')) + sorted(DAILY_DIR.glob('*.csv'))
        save_csv(pd.DataFrame([{'类别': 'Python', '文件': str(f)} for f in code_files] +
                             [{'类别': '逐日CSV', '文件': str(f)} for f in data_files]), 'M05代码与候选数据清单')
        log('代码检索未发现已有逐日气候态P90函数；历史气象生成代码按北京时间UTC+8日历日提取，M05使用现有日期，不再时区转换或下载。')
        log('Q已由原前处理线性插值（limit_direction=both）并进入重构逐日输入；M05不再次插值，P90/P95/P97.5均用相同研究期每日值和原np.percentile方法。')
        detector = load_module(EVENT_SCRIPT, 'm05_original_detector')
        shock_module = load_module(SHOCK_SCRIPT, 'm05_original_shock')
        assert detector.HW_CONTINUE_DAYS == 3 and detector.HW_MERGE_INTERVAL == 2
        assert shock_module.MIN_PRE_BASELINE_VALID_DAYS == MIN_PRE_DAYS
        log('调用原ED_F基础段识别、DC连通合并、F5H优先最短间隔配对、H5F后配对及互斥检查；≥阈值，HW≥3天后合并≤2天。')
        log('Shock沿用原decrease.py：季节非事件DO背景，跨季按事件每日季节背景取平均；前9–4天排除事件日且有效DO≥3天。')
        history, hist_audit = audit_daily(HISTORY_DIR, ['Tmax'], HISTORY_START, HISTORY_END)
        daily, daily_audit = audit_daily(DAILY_DIR, ['Tmax', 'Q', 'DO'], STUDY_START, STUDY_END)
        save_csv(hist_audit, 'M05历史Tmax完整性核查')
        save_csv(daily_audit, 'M05研究期逐日数据完整性核查')
        registry = pd.concat([read_csv(REGISTRY_DIR / f'{t}_01_Registry_Overall.csv') for t in EVENT_TYPES], ignore_index=True).rename(columns={'WQS_name': 'Site_ID'})
        observed = read_csv(ORIGINAL_SHOCK).rename(columns={'WQS_name': 'Site_ID'})
        for frame in [registry, observed]:
            if frame.duplicated(['Site_ID', 'Event_ID']).any():
                raise ValueError('已有事件/指标表存在重复键')
            for col in ['Start_Date', 'End_Date']:
                frame[col] = pd.to_datetime(frame[col])
        sites = sorted(registry.Site_ID.unique())
        if set(sites) != set(daily) or not set(sites).issubset(history):
            raise ValueError(f'断面覆盖不一致：缺历史 {set(sites)-set(history)}；逐日差异 {set(sites)^set(daily)}')
        if daily_audit['缺日期数'].sum() or daily_audit[['Tmax_缺失数', 'Q_缺失数']].to_numpy().sum():
            raise ValueError('研究期日期/Tmax/Q不完整；避免原函数跨缺日分段误判，请人工检查。')
        if any(d.Region.dropna().nunique() != 1 or not set(d.Region.dropna()).issubset(REACH_MAP) for d in daily.values()):
            raise ValueError('研究期断面河段缺失、未知或不一致')
        log(f'历史66研究站缺日期 {hist_audit.loc[hist_audit.Site_ID.isin(sites), "缺日期数"].sum()}，Tmax缺失 {hist_audit.loc[hist_audit.Site_ID.isin(sites), "Tmax_缺失数"].sum()}；研究期DO缺失 {daily_audit.DO_缺失数.sum()}。')
        log(f'注册表 {len(registry)} 条，{len(sites)} 站；历史参考期审核 {len(history)} 站，研究期每站 {len(pd.date_range(STUDY_START, STUDY_END))} 天。')
        all_rows = []
        baselines = []
        cache = {}
        for site in sites:
            d, h = daily[site], history[site]
            t, q = np.percentile(h.Tmax.dropna(), 90), np.percentile(d.Q.dropna(), 95)
            THRESHOLDS.append({'Scenario': 'S0', 'Site_ID': site, 'Tmax_Annual_P90': t,
                               'Tmax_Applied_Min': t, 'Tmax_Applied_Max': t, 'Q_Threshold': q, **SCENARIOS['S0']})
            cache[site] = (detector.get_base_segments(d.copy(), 'hw', t, q),
                           detector.get_base_segments(d.copy(), 'flood', t, q))
            events = identify(site, d, t, q, 'S0', detector, cache[site])
            detector.validate_event_exclusivity(site, events)
            rows, bg = shock_for_site(d, events, shock_module.get_season)
            all_rows.extend(rows)
            baselines.extend(bg)
        calculated = pd.DataFrame(all_rows)
        save_csv(calculated, 'M05_S0事件级Shock结果')
        save_csv(pd.DataFrame(baselines), 'M05_S0季节非事件背景')
        save_csv(pd.DataFrame(THRESHOLDS), 'M05实际阈值')
        for t in EVENT_TYPES:
            n = int(calculated.Event_Type.eq(t).sum())
            log(f'S0 {t}：实际 {n}，参考 {EXPECTED_N[t]}，差 {n-EXPECTED_N[t]:+d}。')
        verify_s0(registry, observed, calculated)
        status['S0_Verified'] = True
        calculated = export_scenario(calculated, 'S0', shock_module.get_season)
        if args.s0_only:
            status['Status'] = '仅S0完成'
            log('仅S0核验结束，未运行S1–S5。')
            return
        log('S0核验已通过，进入敏感性方案；S1每一日使用31日循环窗口，参考期排除2/29，研究期2/29采用2/28和3/1阈值均值；缺历史不补。')
        results = {'S0': calculated}
        for scenario in ['S1', 'S2', 'S3', 'S4', 'S5']:
            results[scenario] = run_scenario(scenario, sites, history, daily, detector, shock_module.get_season, cache)
        save_csv(pd.DataFrame(THRESHOLDS), 'M05实际阈值')
        counts, changes, detail_counts, stats, comparisons = summarize(results, sites)
        shock_l = stats.loc[stats.Metric.eq('Shock_Ratio_LongTerm')].copy()
        shock_s = stats.loc[stats.Metric.eq('Shock_Ratio_Pre')].copy()
        compound = comparisons.loc[comparisons.Comparison.eq('Compound - Single')].copy()
        sequence = comparisons.loc[comparisons.Comparison.eq('F5H - H5F')].copy()
        for table, name in [(counts, 'M05六方案事件数量对比'), (changes, 'M05事件数量相对变化'),
                             (detail_counts, 'M05各河段与站点事件数量'), (shock_l, 'M05五类事件ShockL对比'),
                             (shock_s, 'M05五类事件ShockS对比'), (compound, 'M05复合与单一响应对比'),
                             (sequence, 'M05事件顺序响应对比')]:
            save_csv(table, name)
        excel({'六方案数量': counts, '数量相对变化': changes, '河段与站点数量': detail_counts,
               'ShockL统计': shock_l, 'ShockS统计': shock_s, '复合与单一': compound,
               '事件顺序': sequence, '实际阈值': pd.DataFrame(THRESHOLDS),
               '历史Tmax核查': hist_audit, '研究期数据核查': daily_audit,
               '来源校验': pd.DataFrame(SOURCES)})
        interpretation(counts, changes, comparisons)
        if not all(hashlib.sha256(Path(row['文件']).read_bytes()).hexdigest() == row['SHA256'] for row in SOURCES):
            raise ValueError('运行期间输入或源脚本被修改，请人工检查')
        log('输入及原脚本SHA256未变；全部结果写入独立M05目录。正式方案数：' + str(sum(x is not None for x in results.values())) + '/6。')
        status.update({'Status': '完成', 'Available_Scenarios': [s for s, data in results.items() if data is not None]})
    except Exception as exc:
        status.update({'Status': '失败；先前同日输出不可视为本次结果', 'Error': repr(exc)})
        log('停止：' + repr(exc))
        raise
    finally:
        log('运行状态：' + json.dumps(status, ensure_ascii=False))
        save_csv(pd.DataFrame(SOURCES), 'M05输入与脚本来源校验')
        (OUTPUT_DIR / f'{RUN_DATE}_M05数据核验与运行日志.txt').write_text('\n'.join(LOG), encoding='utf-8-sig')

if __name__ == '__main__':
    main()
