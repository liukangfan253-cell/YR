# -*- coding: utf-8 -*-
"""Compare the two references on common events in M04 using the saved Shock values and retaining the source files.

Run python Code/06_Validation_and_sensitivity/03_Paired_references/M04_paired_baselines.py.
Use --prepare-only to check and prepare the inputs. The project CSV header scan is enabled by default.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import json
import os
import sys

# ======================= Paths and parameters =======================
ROOT = Path(__file__).resolve().parents[3]
REGISTRY_DIR = ROOT / 'Data/Event_F/01_Registry_Data/Overall'
SHOCK_L_FILE = ROOT / 'Result/Decrease/Overall/FINAL_Shock_All_Events_Combined.csv'
SHOCK_S_FILE = SHOCK_L_FILE  # Support separate event-level files after checking their keys and attributes.
RUN_DATE = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
OUTPUT_DIR = ROOT / 'Result/Decrease/Paired_baselines' / RUN_DATE
EXPECTED_L, EXPECTED_COMMON = 2550, 1898  # Use only for verification.
BOOT_REPS, RANDOM_SEED = 1000, 20261012
MIN_EVENTS, MIN_SITES = 10, 3  # Flag sparsely supported groups while retaining their events.
MIN_BOOT_SITES = 10  # Avoid reporting an interval as reliable below this station count.
EVENT_TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']
EVENT_MAP = {**{t: t for t in EVENT_TYPES}, 'Heatwave': EVENT_TYPES[0],
             'Flood': EVENT_TYPES[1], 'DC': EVENT_TYPES[2], 'H5F': EVENT_TYPES[3], 'F5H': EVENT_TYPES[4]}
REACH_MAP = {'UR': 'Upstream', 'MR': 'Midstream', 'LR': 'Downstream',
             'Upstream': 'Upstream', 'Midstream': 'Midstream', 'Downstream': 'Downstream'}
REACHES = ['Upstream', 'Midstream', 'Downstream']
ALIASES = {'Site_ID': ['Site_ID', 'WQS_name'], 'Event_ID': ['Event_ID'],
           'Event_Type': ['Event_Type'], 'Reach': ['Reach', 'Region'],
           'Start_Date': ['Start_Date'], 'End_Date': ['End_Date'],
           'Shock_L': ['Shock_L', 'Shock-L', 'Shock_Ratio_LongTerm'],
           'Shock_S': ['Shock_S', 'Shock-S', 'Shock_Ratio_Pre']}
COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
LABELS = ['Heatwave only', 'Flood only', 'DC', 'H5F', 'F5H']
# ============================================================== 
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

LOG, SOURCES = [], []
KEY = ['Site_ID', 'Event_ID']
ATTRS = ['Event_Type', 'Reach', 'Start_Date', 'End_Date']

def log(message):
    LOG.append(str(message))
    print(message, flush=True)

def save_csv(frame, name):
    frame.to_csv(OUTPUT_DIR / f'{RUN_DATE}_{name}.csv', index=False,
                 encoding='utf-8-sig', float_format='%.12g')

def read_csv(path):
    for encoding in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=encoding, dtype='string')
        except UnicodeDecodeError:
            continue
    raise ValueError(f'文件无法解码：{path}')

def scan_headers():
    """Scan CSV headers. Verify record and station counts directly for the analysis inputs."""
    rows = []
    for path in sorted(ROOT.rglob('*.csv')):
        if any(x.startswith('Statistics_M') or x.startswith('.venv') or x in {'Statistical_comparisons', 'Paired_baselines', 'Definition_sensitivity', 'Hydrological_matching', 'DO_observation_sensitivity'} for x in path.parts):
            continue
        for encoding in ['utf-8-sig', 'gb18030']:
            try:
                with path.open(encoding=encoding, newline='') as f:
                    fields = next(csv.reader(f), [])
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f'CSV表头解码失败：{path}')
        has_l = bool(set(fields) & set(ALIASES['Shock_L']))
        has_s = bool(set(fields) & set(ALIASES['Shock_S']))
        event = 'Event_ID' in fields and 'Event_Type' in fields
        rows.append({'文件': str(path.relative_to(ROOT)), '字段': '|'.join(fields),
                     '含Shock_L': has_l, '含Shock_S': has_s, '含事件标识': event,
                     '事件指标候选': event and (has_l or has_s)})
    inventory = pd.DataFrame(rows)
    save_csv(inventory, 'M04项目CSV字段扫描')
    log(f'已扫描 {len(inventory)} 个既有CSV表头；事件指标候选 {inventory["事件指标候选"].sum()} 个。'
        '候选含分类型、分站和合并副本，不重复拼接；配置输入逐一核对。')

def normalize(path, metrics=()):
    raw = read_csv(path)
    mapping = {}
    for canonical in KEY + ATTRS + list(metrics):
        found = [c for c in ALIASES[canonical] if c in raw]
        if len(found) != 1:
            raise ValueError(f'{path}：{canonical} 对应字段不明确 {found}；停止，需人工确认。')
        mapping[found[0]] = canonical
    d = raw.rename(columns=mapping)
    for c in KEY:
        if d[c].isna().any() or d[c].str.strip().eq('').any():
            raise ValueError(f'{path} 存在空 {c}')
    duplicate = d.duplicated(KEY, keep=False)
    if duplicate.any():
        save_csv(d.loc[duplicate], 'M04重复记录待检查')
        raise ValueError(f'{path} 有 {duplicate.sum()} 条重复站点—事件记录；已保存，不随意去重。')
    unknown_type = set(d.Event_Type.dropna()) - set(EVENT_MAP)
    unknown_reach = set(d.Reach.dropna()) - set(REACH_MAP)
    if unknown_type or unknown_reach or d[['Event_Type', 'Reach']].isna().any().any():
        raise ValueError(f'{path} 未知事件类型 {unknown_type} 或河段 {unknown_reach}，禁止猜测分类。')
    d['Event_Type'] = d.Event_Type.map(EVENT_MAP)
    d['Reach'] = d.Reach.map(REACH_MAP)
    for c in ['Start_Date', 'End_Date']:
        d[c] = pd.to_datetime(d[c], errors='raise')
    if d[['Start_Date', 'End_Date']].isna().any().any() or d.End_Date.lt(d.Start_Date).any():
        raise ValueError(f'{path} 事件日期缺失或倒置')
    for metric in metrics:
        x = pd.to_numeric(d[metric], errors='coerce')
        malformed = d[metric].notna() & x.isna()
        if malformed.any():
            save_csv(d.loc[malformed], 'M04非数值记录待检查')
            raise ValueError(f'{path} {metric} 有非数值文本，需人工检查')
        d[metric] = x.to_numpy(dtype=float, na_value=np.nan)
    SOURCES.append({'文件': str(path), '记录数': len(d), '断面数': d.Site_ID.nunique(),
                    '映射': json.dumps(mapping, ensure_ascii=False), '字段': '|'.join(raw.columns),
                    '断面': '|'.join(sorted(d.Site_ID.unique())),
                    'SHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
    log(f'读取 {path.relative_to(ROOT)}：{len(d)} 条，{d.Site_ID.nunique()} 个断面；映射 {mapping}。')
    return d

def verify_source(reg, source, name):
    checked = source.merge(reg[KEY + ATTRS], on=KEY, how='left',
                            suffixes=('_source', '_registry'), validate='one_to_one', indicator=True)
    bad = checked['_merge'].ne('both')
    for c in ATTRS:
        bad |= checked[f'{c}_source'].ne(checked[f'{c}_registry'])
    if bad.any():
        save_csv(checked.loc[bad], 'M04匹配冲突待检查')
        raise ValueError(f'{name} 有 {bad.sum()} 条无法匹配或属性冲突，停止。')
    log(f'{name}：所有Site_ID+Event_ID、事件分类、河段、起止日期与注册表一致；没有行号匹配。')

def prepare():
    registry = pd.concat([normalize(REGISTRY_DIR / f'{t}_01_Registry_Overall.csv') for t in EVENT_TYPES], ignore_index=True)
    if registry.duplicated(KEY).any():
        raise ValueError('五类注册表之间存在重复站点—事件键')
    if registry.groupby('Site_ID').Reach.nunique().gt(1).any():
        raise ValueError('站点所属河段不一致')
    if not registry.Start_Date.dt.year.between(2021, 2025).all():
        raise ValueError('注册表事件起始年份超出2021–2025')
    l = normalize(SHOCK_L_FILE, ['Shock_L', 'Shock_S'] if SHOCK_L_FILE == SHOCK_S_FILE else ['Shock_L'])
    s = l if SHOCK_L_FILE == SHOCK_S_FILE else normalize(SHOCK_S_FILE, ['Shock_S'])
    verify_source(registry, l, 'Shock-L来源')
    verify_source(registry, s, 'Shock-S来源')
    data = registry[KEY + ATTRS].merge(l[KEY + ['Shock_L']], on=KEY, how='left', validate='one_to_one')
    data = data.merge(s[KEY + ['Shock_S']], on=KEY, how='left', validate='one_to_one')
    # Retain other numeric and seasonal covariates. Use the existing values of both Shock measures.
    extra = [c for c in ['Season', 'Year', 'Event_DO', 'Seasonal_Baseline_DO', 'Pre_Baseline_DO',
                         'Pre_Baseline_Valid_Days'] if c in l and c not in data]
    if extra:
        data = data.merge(l[KEY + extra], on=KEY, how='left', validate='one_to_one')
    if 'Year' not in data:
        data['Year'] = data.Start_Date.dt.year
    if 'Season' not in data:
        season = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
                  7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
        data['Season'] = data.Start_Date.dt.month.map(season)
        log('Season原文件缺失，保留开始日期并新增开始日期气象学季节；不改变Shock-L已有背景定义。')
    for metric in ['Shock_L', 'Shock_S']:
        x = data[metric]
        log(f'{metric}：缺失 {x.isna().sum()}，零 {(x == 0).sum()}，负 {(x < 0).sum()}，'
            f'非有限且非缺失 {(~np.isfinite(x) & x.notna()).sum()}；有效正值 {(np.isfinite(x) & (x > 0)).sum()}。')
        data[metric + '_Valid'] = np.isfinite(x) & x.gt(0)
    full_l = data.loc[data.Shock_L_Valid].copy()
    common = data.loc[data.Shock_L_Valid & data.Shock_S_Valid].copy()
    if common.empty:
        raise ValueError('不存在双基线共同有效正值事件')
    common['Delta_Shock'] = common.Shock_S - common.Shock_L
    # The log difference equals ln(S/L) without direct division, which can overflow. Both measures are finite and positive.
    common['Log_Ratio'] = np.log(common.Shock_S) - np.log(common.Shock_L)
    if not np.isfinite(common[['Delta_Shock', 'Log_Ratio']]).all().all():
        raise ValueError('配对差值/对数比溢出，需要人工检查数值')
    x, y = common.Shock_L, common.Shock_S
    common['Direction_Class'] = np.select([(x < 1) & (y < 1), (x > 1) & (y > 1),
                                           (x < 1) & (y > 1), (x > 1) & (y < 1)],
                                          ['A', 'B', 'C', 'D'], default='Neutral')
    common['Direction_Flip'] = common.Direction_Class.isin(['C', 'D'])
    log(f'Shock-L全样本 {len(full_l)} / 参考 {EXPECTED_L}（差 {len(full_l)-EXPECTED_L:+d}）；'
        f'共同样本 {len(common)} / 参考 {EXPECTED_COMMON}（差 {len(common)-EXPECTED_COMMON:+d}）。')
    excluded = data.loc[~(data.Shock_L_Valid & data.Shock_S_Valid)].copy()
    excluded['排除原因'] = np.where(~excluded.Shock_L_Valid & ~excluded.Shock_S_Valid, '两指标无效',
                                  np.where(~excluded.Shock_L_Valid, 'Shock-L无效', 'Shock-S无效'))
    save_csv(excluded, 'M04非共同样本及原因')
    save_csv(common, 'M04共同事件配对数据')
    save_csv(pd.DataFrame(SOURCES), 'M04输入来源与字段核查')
    log(f'未进入共同样本 {len(excluded)} 条，逐条保留无效标识，不人为补值或剪裁。')
    return full_l, common

def groups(data):
    yield 'Overall', 'All', 'All', data
    for event in EVENT_TYPES:
        yield 'Event_Type', event, 'All', data.loc[data.Event_Type.eq(event)]
    for reach in REACHES:
        yield 'Reach', 'All', reach, data.loc[data.Reach.eq(reach)]
    for event in EVENT_TYPES:
        for reach in REACHES:
            yield 'Event_Type_Reach', event, reach, data.loc[data.Event_Type.eq(event) & data.Reach.eq(reach)]

def sparsity(data):
    return '样本稀疏，描述性结果须谨慎' if len(data) < MIN_EVENTS or data.Site_ID.nunique() < MIN_SITES else '描述性样本达到配置门槛'

def summaries(full_l, common):
    rows = []
    for sample, metric, data in [('A_Shock_L_Full', 'Shock_L', full_l),
                                  ('B_Shock_L_Common', 'Shock_L', common),
                                  ('C_Shock_S_Common', 'Shock_S', common)]:
        for level, event, reach, sub in groups(data):
            x = sub[metric]
            site_means = sub.groupby('Site_ID')[metric].mean()
            n = len(sub)
            rows.append({'Sample': sample, 'Metric': metric, 'Level': level, 'Event_Type': event,
                         'Reach': reach, 'n_events': n, 'n_sites': sub.Site_ID.nunique(),
                         'Mean_Event': x.mean(), 'Median_Event': x.median(), 'SD_Event': x.std(ddof=1),
                         'Mean_Site_Equal': site_means.mean(), 'Median_Site_Means': site_means.median(),
                         'SD_Site_Means': site_means.std(ddof=1),
                         'n_below_1': int(x.lt(1).sum()), 'n_above_1': int(x.gt(1).sum()), 'n_equal_1': int(x.eq(1).sum()),
                         'Below_1_pct': 100*x.lt(1).mean() if n else np.nan,
                         'Above_1_pct': 100*x.gt(1).mean() if n else np.nan,
                         'Paper_Mean_Basis': 'Site_Equal' if 'Reach' in level else 'Event_Equal',
                         '比例分母': '有效事件数，非断面均值数', 'Sparse_Note': sparsity(sub)})
    stats = pd.DataFrame(rows)
    paired, flips, effects = [], [], []
    for level, event, reach, sub in groups(common):
        identifiers = {'Level': level, 'Event_Type': event, 'Reach': reach}
        site = sub.groupby('Site_ID')[['Delta_Shock', 'Log_Ratio']].mean()
        paired.append({**identifiers, 'n_events': len(sub), 'n_sites': sub.Site_ID.nunique(),
                       'Delta_Mean_Event': sub.Delta_Shock.mean(), 'Delta_Median_Event': sub.Delta_Shock.median(),
                       'Log_Ratio_Mean_Event': sub.Log_Ratio.mean(),
                       'Delta_Mean_Site_Equal': site.Delta_Shock.mean(),
                       'Log_Ratio_Mean_Site_Equal': site.Log_Ratio.mean(), 'Sparse_Note': sparsity(sub)})
        neutral = sub.Direction_Class.eq('Neutral')
        decidable = sub.loc[~neutral]
        changed = sub.loc[sub.Direction_Flip]
        n, denom = len(sub), len(decidable)
        row = {**identifiers, 'n_common': n, 'n_sites_common': sub.Site_ID.nunique(),
               'n_decidable': denom, 'n_sites_decidable': decidable.Site_ID.nunique(),
               'n_flip': len(changed), 'n_sites_flip': changed.Site_ID.nunique(),
               'Flip_Rate_pct': 100*len(changed)/denom if denom else np.nan,
               'Flip_Complete_pct': 100*len(changed)/n if n else np.nan,
               'Sparse_Note': sparsity(sub)}
        for direction in ['A', 'B', 'C', 'D', 'Neutral']:
            cls = sub.loc[sub.Direction_Class.eq(direction)]
            row['n_'+direction] = len(cls)
            row['n_sites_'+direction] = cls.Site_ID.nunique()
            row[direction+'_Common_pct'] = 100*len(cls)/n if n else np.nan
        flips.append(row)
        # For each summary definition, distinguish sample selection from A to B and the reference change from B to C.
        subset = stats.loc[(stats.Level == level) & (stats.Event_Type == event) & (stats.Reach == reach)].set_index('Sample')
        for basis in ['Mean_Event', 'Mean_Site_Equal']:
            a, b, c = [subset.loc[z, basis] for z in ['A_Shock_L_Full', 'B_Shock_L_Common', 'C_Shock_S_Common']]
            effects.append({**identifiers, 'Mean_Basis': basis, 'A_Full_L': a, 'B_Common_L': b, 'C_Common_S': c,
                            'Selection_B_minus_A': b-a, 'Baseline_C_minus_B': c-b, 'Total_C_minus_A': c-a,
                            'A_n_events': int(subset.loc['A_Shock_L_Full', 'n_events']), 'BC_n_events': n,
                            'A_n_sites': int(subset.loc['A_Shock_L_Full', 'n_sites']), 'BC_n_sites': sub.Site_ID.nunique(),
                            '含义': '描述性均值变化，不作因果归因'})
    return stats, pd.DataFrame(paired), pd.DataFrame(flips), pd.DataFrame(effects)

def bootstrap(common):
    """Resample whole stations with replacement. Summing station counts is equivalent to duplicating all events from each sampled station."""
    blocks = common.assign(Decidable=common.Direction_Class.ne('Neutral')).groupby('Site_ID').agg(
        n_common=('Event_ID', 'size'), n_decidable=('Decidable', 'sum'), n_flip=('Direction_Flip', 'sum'))
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.integers(0, len(blocks), size=(BOOT_REPS, len(blocks)))
    full_n = blocks.n_common.to_numpy()[draws].sum(axis=1)
    denom = blocks.n_decidable.to_numpy()[draws].sum(axis=1)
    numer = blocks.n_flip.to_numpy()[draws].sum(axis=1)
    rates = np.divide(100*numer, denom, out=np.full(BOOT_REPS, np.nan), where=denom > 0)
    valid = np.isfinite(rates)
    stable = int(blocks.n_decidable.gt(0).sum()) >= MIN_BOOT_SITES and valid.sum() == BOOT_REPS
    ci = np.percentile(rates[valid], [2.5, 97.5]) if stable else [np.nan, np.nan]
    summary = pd.DataFrame([{'Group': 'Overall', 'n_common': len(common), 'n_sites': len(blocks),
                             'n_sites_decidable': int(blocks.n_decidable.gt(0).sum()),
                             'n_decidable': int(blocks.n_decidable.sum()), 'n_flip': int(blocks.n_flip.sum()),
                             'Flip_Rate_pct': 100*blocks.n_flip.sum()/blocks.n_decidable.sum() if blocks.n_decidable.sum() else np.nan,
                             'CI95_Lower_pct': ci[0], 'CI95_Upper_pct': ci[1], 'Requested_Reps': BOOT_REPS,
                             'Valid_Reps': int(valid.sum()), 'Seed': RANDOM_SEED,
                             'Status': '可用百分位区间' if stable else '不稳定，CI不发布，需人工检查',
                             'Method': 'Site_ID块有放回抽样；每次抽原始站点数并保留全部共同事件；百分位CI'}])
    replicates = pd.DataFrame({'Replicate': np.arange(1, BOOT_REPS+1), 'n_common_resampled': full_n,
                               'n_decidable_resampled': denom, 'n_flip_resampled': numer, 'Flip_Rate_pct': rates})
    log(f'站点块Bootstrap：{len(blocks)} 站，每次有放回抽 {len(blocks)} 站，重复 {BOOT_REPS}，'
        f'有效 {valid.sum()}；种子 {RANDOM_SEED}；CI={ci}。仅总体估计，未添加分类型显著性检验。')
    return summary, replicates

def excel(tables):
    path = OUTPUT_DIR / f'{RUN_DATE}_M04分析结果汇总.xlsx'
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, data in tables.items():
            data.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 38
            for cell in sheet[1]:
                cell.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF')
                cell.fill = PatternFill('solid', fgColor='24476A')
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            for i, column in enumerate(sheet.columns, 1):
                header = str(sheet.cell(1, i).value)
                sheet.column_dimensions[get_column_letter(i)].width = min(46, max(15, len(header)+2))
                for cell in list(column)[1:]:
                    if isinstance(cell.value, (float, int)):
                        cell.number_format = '0' if header.startswith('n_') or header in ['Seed', 'Requested_Reps', 'Valid_Reps', 'Year'] else '0.000000'
                    elif isinstance(cell.value, datetime):
                        cell.number_format = 'yyyy-mm-dd'
    book = load_workbook(path, read_only=True, data_only=True)
    for name, data in tables.items():
        assert book[name].max_row == len(data)+1 and book[name].max_column == len(data.columns)
    book.close()
    log('Excel已重新打开核对各工作表行列数；百分数按0–100数值保存，未用Excel百分比格式再次乘100。')

def interpretations(stats, flips, changes, boot):
    overall = flips.loc[flips.Level.eq('Overall')].iloc[0]
    type_rows = flips.loc[flips.Level.eq('Event_Type')]
    highest = type_rows.loc[type_rows.Flip_Rate_pct.idxmax()]
    downstream = stats.loc[(stats.Level == 'Event_Type_Reach') & (stats.Event_Type == 'F5H_Compound') & (stats.Reach == 'Downstream')].set_index('Sample')
    df = flips.loc[(flips.Level == 'Event_Type_Reach') & (flips.Event_Type == 'F5H_Compound') & (flips.Reach == 'Downstream')].iloc[0]
    change = changes.loc[(changes.Level == 'Overall') & (changes.Mean_Basis == 'Mean_Event')].iloc[0]
    low, high = downstream.loc['B_Shock_L_Common', 'Mean_Site_Equal'], downstream.loc['C_Shock_S_Common', 'Mean_Site_Equal']
    conclusion = '仍呈现Shock-L<1、Shock-S>1的断面等权平均特征' if low < 1 and high > 1 else '未同时呈现Shock-L<1、Shock-S>1的断面等权平均特征'
    b = boot.iloc[0]
    text = ['# M04 结果解读与 §3.4 补充建议', '',
            f'共同有效事件 {int(overall.n_common)} 条、{int(overall.n_sites_common)} 个断面；方向改变 {int(overall.n_flip)} 条、涉及 {int(overall.n_sites_flip)} 个断面。',
            f'Neutral {int(overall.n_Neutral)} 条。方向改变比例分母为 {int(overall.n_decidable)} 条可判定事件：{overall.Flip_Rate_pct:.3f}%；占完整共同样本 {overall.Flip_Complete_pct:.3f}%。',
            f'总体站点块Bootstrap 95%CI [{b.CI95_Lower_pct:.3f}, {b.CI95_Upper_pct:.3f}]%，有效 {int(b.Valid_Reps)}/{int(b.Requested_Reps)} 次。', '',
            f'五类中方向改变比例最高为 {highest.Event_Type}：{highest.Flip_Rate_pct:.3f}%（{int(highest.n_flip)}/{int(highest.n_decidable)} 条，{int(highest.n_sites_decidable)} 个可判定断面）。这是描述性排序，不证明类型间差异显著。', '',
            f'下游F5H共同样本 n={int(df.n_common)}，断面={int(df.n_sites_common)}；Shock-L断面等权均值 {low:.6f}，Shock-S {high:.6f}；{conclusion}。',
            f'该分组 C 类 {int(df.n_C)} 条、D 类 {int(df.n_D)} 条，方向改变比例 {df.Flip_Rate_pct:.3f}%。断面均值方向不同不表示每条事件方向不同；{df.Sparse_Note}。', '',
            f'全流域事件等权均值：A（全样本L）={change.A_Full_L:.6f}；B（共同样本L）={change.B_Common_L:.6f}；C（共同样本S）={change.C_Common_S:.6f}。',
            f'A→B差 {change.Selection_B_minus_A:+.6f} 是保留共同有效样本后的描述性变化；B→C差 {change.Baseline_C_minus_B:+.6f} 是同一批事件改变参考基线后的差异；C−A={change.Total_C_minus_A:+.6f}。',
            '二者是描述性数值分解，不是因果贡献率。河段和事件×河段继续按断面等权均值报告，事件合并均值作为另列结果；不可跨口径计算A/B/C差异。', '',
            '§3.4 建议在原Shock-S及下游F5H描述之后补充一段共同样本结果，依次报告共同样本规模、方向改变数量/比例与总体CI、下游F5H配对样本均值及稀疏性。原恢复斜率段落保持独立。',
            '方法中补充：两指标同时有限且>0；精确等于1记Neutral；方向改变比例排除Neutral；按站点块有放回抽样1000次、固定随机种子、取百分位CI。',
            '原方法关于“跨指标未进行相同样本配对”的表述需更新为“原全样本汇总为描述性比较，并另开展共同事件配对分析”。原全样本数值仍可保留，但不能与共同样本结果混用。',
            '结果与讨论应强调：两个指标共享事件期DO分子，因此共同事件的差异反映参考水平差异；不是同一事件的DO时间轨迹，也不证明事件因果效应。M04没有新增p值或显著性模型。', '',
            '完整各类、河段、事件×河段、两种均值口径及配对量见CSV和Excel；样本稀疏分组仍保留并明确标记。']
    log('\n'.join(text[2:14]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log(f'M04开始，实际运行日期（上海）{RUN_DATE}；pandas {pd.__version__}、numpy {np.__version__}。')
        log('论文既有口径核对：全流域/事件类型按事件等权；河段及类型×河段先求断面均值再断面等权。分别保留两种口径。')
        scan_headers()
        full_l, common = prepare()
        if args.prepare_only:
            return
        stats, paired, flips, changes = summaries(full_l, common)
        boot, reps = bootstrap(common)
        for frame, name in [(stats, 'M04双基线总体与分组统计'), (paired, 'M04分组配对差值'),
                             (flips, 'M04偏离方向改变比例'), (changes, 'M04样本筛选与基线差异分解'),
                             (boot, 'M04站点块Bootstrap结果'), (reps, 'M04站点块Bootstrap重复分布')]:
            save_csv(frame, name)
        excel({'总体统计': stats.loc[stats.Level.eq('Overall')], '事件类型统计': stats.loc[stats.Level.eq('Event_Type')],
               '河段统计': stats.loc[stats.Level.eq('Reach')], '类型与河段统计': stats.loc[stats.Level.eq('Event_Type_Reach')],
               '方向改变比例': flips, '配对差值': paired, '变化分解': changes, '总体Bootstrap': boot,
               '共同事件数据': common, '来源核查': pd.DataFrame(SOURCES)})
        assert len(common) == len(common.drop_duplicates(KEY))
        assert all(hashlib.sha256(Path(s['文件']).read_bytes()).hexdigest() == s['SHA256'] for s in SOURCES)
        log('源文件SHA256复核一致；配对键唯一；未重算Shock或改原事件分类。')
        interpretations(stats, flips, changes, boot)
        log('M04完成：仅描述性配对量、方向比例及总体站点块Bootstrap区间；未添加显著性检验。')
    except Exception as exc:
        log('停止：' + repr(exc) + '；请人工检查，禁止随意匹配/删样本。')
        raise
    finally:
        (OUTPUT_DIR / f'{RUN_DATE}_M04数据核查与分析日志.txt').write_text('\n'.join(LOG), encoding='utf-8-sig')

if __name__ == '__main__':
    main()
