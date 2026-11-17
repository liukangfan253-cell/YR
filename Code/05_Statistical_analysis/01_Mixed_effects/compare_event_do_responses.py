# -*- coding: utf-8 -*-
"""Read existing M01 metrics and fit REML random-intercept models with two-sided Wald intervals and p values.

Run python Code/05_Statistical_analysis/01_Mixed_effects/compare_event_do_responses.py [--metrics Shock-L].
Use --prepare-only to check and prepare the inputs, or --scan to inventory CSV files.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import json
import warnings
import hashlib
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ================= Input paths, output paths and parameters =================
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Code/05_Statistical_analysis/00_Shared_utilities'))
from english_table_output import english_table, FILE_NAMES, SHEETS
REGISTRY_DIR = ROOT / 'Data/Event_F/01_Registry_Data/Overall'
SHOCK_FILE = ROOT / 'Result/Decrease/Overall/FINAL_Shock_All_Events_Combined.csv'
RECOVERY_FILE = ROOT / 'Result/Recovery/Overall/FINAL_Recovery_All_Events_Combined.csv'
RUN_DATE = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d')
OUTPUT_DIR = ROOT / 'Result/Statistical_comparisons'
EVENT_TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']
EVENT_MAP = {**{x: x for x in EVENT_TYPES}, 'DC': 'Direct_Compound',
             'H5F': 'H5F_Compound', 'F5H': 'F5H_Compound',
             'Heatwave': 'Heatwave_Only', 'Flood': 'Flood_Only'}
FIELD_MAP = {'WQS_name': 'Site_ID', 'Region': 'Reach',
             'Shock_Ratio_LongTerm': 'Shock-L', 'Shock_Ratio_Pre': 'Shock-S'}
REACH_MAP = {'UR': 'Upstream', 'MR': 'Midstream', 'LR': 'Downstream'}
REFERENCE_N = {'Shock-L': 2550, 'Shock-S': 1898, 'Recovery_Slope': 1247}
OPTIMIZERS = ['lbfgs', 'bfgs', 'powell']  # Retry only the numerical optimization, with the same model and sample.
MAXITER = 1000
NEAR_ZERO_ICC = 1e-6
ALPHA = 0.05
KEY = ['Site_ID', 'Event_ID']
# ============================================================== 
LOG = []
DIAGNOSTICS = []
SOURCES = []

def log(message):
    LOG.append(str(message))
    print(message, flush=True)

def save_csv(df, name):
    english_table(df).to_csv(OUTPUT_DIR / f'{FILE_NAMES.get(name, name)}.csv', index=False,
              encoding='utf-8-sig', float_format='%.12g')

def read_csv(path):
    for enc in ['utf-8-sig', 'gb18030']:
        try:
            return pd.read_csv(path, encoding=enc, dtype={'Event_ID': 'string', 'WQS_name': 'string'})
        except UnicodeDecodeError:
            continue
    raise ValueError(f'无法解码：{path}')

def scan_csv():
    """Stream the CSV inventory and keep daily records and station means out of the event-level input."""
    rows = []
    for path in sorted(ROOT.rglob('*.csv')):
        if 'Statistical_comparisons' in path.parts or '.venv_m01' in path.parts:
            continue
        for enc in ['utf-8-sig', 'gb18030']:
            try:
                with path.open(encoding=enc, newline='') as f:
                    reader = csv.reader(f)
                    fields = next(reader, [])
                    candidates = [c for c in ['Site_ID', 'WQS_name', 'WQS_Name', 'Station', 'site'] if c in fields]
                    idx = fields.index(candidates[0]) if candidates else None
                    count, sites = 0, set()
                    for row in reader:
                        if not row:
                            continue
                        count += 1
                        if idx is not None and len(row) > idx and row[idx]:
                            sites.add(row[idx])
                rows.append({'文件': str(path.relative_to(ROOT)), '记录数': count,
                             '字段': '|'.join(fields), '断面数': len(sites) if idx is not None else None,
                             '断面': '|'.join(sorted(sites)),
                             '断面识别': candidates[0] if candidates else '无明确字段；不推断文件名'})
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f'清点解码失败：{path}')
    table = pd.DataFrame(rows)
    save_csv(table, 'CSV结构清单')
    log(f'CSV 全项目清点完成：{len(table)} 个文件；断面无字段的文件已明确标记。')

def normalize(path):
    d = read_csv(path).rename(columns=FIELD_MAP)
    required = KEY + ['Event_Type', 'Reach', 'Start_Date', 'End_Date']
    missing = set(required) - set(d)
    if missing:
        raise ValueError(f'{path} 缺字段 {missing}；需人工明确唯一关联键，停止匹配。')
    for c in KEY:
        if d[c].isna().any() or d[c].str.strip().eq('').any():
            raise ValueError(f'{path} 存在空 {c}')
    if d.duplicated(KEY).any():
        raise ValueError(f'{path} 站点—事件重复：{d.loc[d.duplicated(KEY, keep=False), KEY].head().to_dict("records")}')
    d['Event_Type'] = d['Event_Type'].map(EVENT_MAP)
    d['Reach'] = d['Reach'].map(REACH_MAP)
    if d[['Event_Type', 'Reach']].isna().any().any():
        raise ValueError(f'{path} 有未映射的事件类型或河段，停止。')
    for c in ['Start_Date', 'End_Date']:
        d[c] = pd.to_datetime(d[c], errors='raise')
    if (d.End_Date < d.Start_Date).any():
        raise ValueError(f'{path} 存在结束早于开始的事件')
    SOURCES.append({'文件': str(path), '记录数': len(d), '断面数': d.Site_ID.nunique(),
                    '字段': '|'.join(d.columns), 'SHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
    log(f'读取 {path.relative_to(ROOT)}：{len(d)} 条，{d.Site_ID.nunique()} 个断面；键无重复。')
    return d

def prepare():
    reg = pd.concat([normalize(REGISTRY_DIR / f'{t}_01_Registry_Overall.csv') for t in EVENT_TYPES], ignore_index=True)
    if reg.duplicated(KEY).any():
        raise ValueError('五种注册表之间存在重复站点—事件键')
    if reg.groupby('Site_ID').Reach.nunique().gt(1).any():
        raise ValueError('同一断面的河段不一致')
    if not reg.Start_Date.dt.year.between(2021, 2025).all():
        raise ValueError('注册表存在研究范围之外的开始年份，请人工检查')
    d = reg[KEY + ['Event_Type', 'Reach', 'Start_Date', 'End_Date']].copy()
    for path, metrics in [(SHOCK_FILE, ['Shock-L', 'Shock-S']), (RECOVERY_FILE, ['Recovery_Slope'])]:
        src = normalize(path)
        check = src.merge(reg, on=KEY, how='left', suffixes=('_src', '_reg'), validate='one_to_one', indicator=True)
        if check['_merge'].ne('both').any():
            raise ValueError(f'{path} 有无法匹配注册表的事件')
        for col in ['Event_Type', 'Reach', 'Start_Date', 'End_Date']:
            if check[f'{col}_src'].ne(check[f'{col}_reg']).any():
                raise ValueError(f'{path} 关联后 {col} 不一致')
        for metric in metrics:
            numeric = pd.to_numeric(src[metric], errors='coerce')
            if (src[metric].notna() & numeric.isna()).any():
                raise ValueError(f'{path} {metric} 有非数值记录')
            src[metric] = numeric
        if 'Shock-S' in metrics:
            missing_pre = src['Shock-S'].isna()
            log('Shock-S缺失核对：' + str(int(missing_pre.sum())) + ' 条；其中前基线DO缺失 ' +
                str(int(src.loc[missing_pre, 'Pre_Baseline_DO'].isna().sum())) + ' 条；基线有效天数分布：' +
                src.loc[missing_pre, 'Pre_Baseline_Valid_Days'].value_counts(dropna=False).to_string())
        d = d.merge(src[KEY + metrics], on=KEY, how='left', validate='one_to_one')
        log(f'合并 {path.name}：属性完全一致；注册表中缺该结果的事件 {len(reg)-len(src)} 条，保持缺失。')
    season = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
              6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
    d['Season'] = d.Start_Date.dt.month.map(season)
    cross = d.Season.ne(d.End_Date.dt.month.map(season))
    log(f'Season 缺少原始字段：按开始日期划分气象学四季；跨季事件 {cross.sum()} 条，全部保留。')
    d['Compound_Group'] = np.where(d.Event_Type.isin(EVENT_TYPES[2:]), 'Compound', 'Single')
    log(f'统一事件表：{len(d)} 条，{d.Site_ID.nunique()} 个断面；未重算任何指标。')
    log('字段映射：' + json.dumps(FIELD_MAP, ensure_ascii=False))
    save_csv(d, '事件级统计分析数据')
    save_csv(pd.DataFrame(SOURCES), '输入来源与校验')
    return d

def valid_data(d, metric):
    x = d[metric]
    finite = np.isfinite(x)
    positive = x.gt(0) if metric != 'Recovery_Slope' else pd.Series(True, index=d.index)
    cov = d[['Site_ID', 'Event_Type', 'Reach', 'Season']].notna().all(axis=1)
    mask = finite & positive & cov
    log(f'{metric}：总数 {len(d)}，缺失 {x.isna().sum()}，非有限非缺失 {(x.notna() & ~finite).sum()}，'
        f'非正有限 {(finite & ~positive).sum()}，协变量缺失 {(~cov).sum()}，有效 {mask.sum()}，'
        f'参考 {REFERENCE_N[metric]}，差异 {mask.sum()-REFERENCE_N[metric]:+d}。')
    if metric != 'Recovery_Slope':
        log(f'{metric} 所有非缺失有限值为正：{bool(positive[finite].all())}；不加常数、不填0。')
    v = d.loc[mask].copy()
    v['Response'] = np.log(v[metric]) if metric != 'Recovery_Slope' else v[metric]
    for t in EVENT_TYPES:
        sub = v[v.Event_Type.eq(t)]
        log(f'  {t}：n={len(sub)}，断面={sub.Site_ID.nunique()}；' +
            ('断面稀疏（<5），需谨慎解释。' if sub.Site_ID.nunique() < 5 else ''))
    log('  类型×河段有效数量：\n' + pd.crosstab(v.Event_Type, v.Reach).to_string())
    log('  类型×季节有效数量：\n' + pd.crosstab(v.Event_Type, v.Season).to_string())
    log('  每站有效事件数：' + v.groupby('Site_ID').size().describe().to_string())
    return v

def fit_compare(v, metric, model_name):
    is_a = model_name == 'A'
    factor = 'Event_Type' if is_a else 'Compound_Group'
    ref = 'H5F_Compound' if is_a else 'Single'
    target = 'F5H_Compound' if is_a else 'Compound'
    formula = f'Response ~ C({factor}, Treatment(reference="{ref}")) + C(Reach) + C(Season)'
    row = {'指标': metric, '比较': 'F5H vs H5F' if is_a else 'Compound vs Single',
           '模型': model_name, '模型样本量': len(v), '模型断面数': v.Site_ID.nunique(),
           '效应类型': '斜率差' if metric == 'Recovery_Slope' else '响应比',
           '单位': 'mg L^-1 d^-1' if metric == 'Recovery_Slope' else '无量纲',
           '调整后效应量': np.nan, 'CI下限': np.nan, 'CI上限': np.nan, '原始p值': np.nan,
           'Holm校正p值': np.nan, '相对变化百分比': np.nan,
           '显著性': '无法可靠估计', '状态': '失败'}
    for name, group in [('目标组', target), ('参照组', ref)]:
        sub = v[v[factor].eq(group)]
        row.update({name: group, name+'样本量': len(sub), name+'断面数': sub.Site_ID.nunique(),
                    name+'原始均值': sub[metric].mean(), name+'原始中位数': sub[metric].median()})
    diag = {'指标': metric, '模型': model_name, '公式': formula, 'N': len(v),
            '断面数': v.Site_ID.nunique(), '估计方法': 'REML', 'CI/p方法': '双侧渐近正态Wald；95%CI未作多重校正'}
    if v.Site_ID.nunique() < 2 or not {ref, target}.issubset(set(v[factor])) or (is_a and set(v.Event_Type) != set(EVENT_TYPES)):
        diag['错误'] = '断面不足、比较组缺失或五类模型有事件类型缺失'
        DIAGNOSTICS.append(diag)
        return row
    log(f'拟合 {metric} 模型 {model_name}：{formula} + (1|Site_ID)')
    model = smf.mixedlm(formula, v, groups=v.Site_ID, re_formula='1')
    if np.linalg.matrix_rank(model.exog) < model.exog.shape[1]:
        diag['错误'] = '固定效应设计矩阵秩不足'
        DIAGNOSTICS.append(diag)
        return row
    for method in OPTIMIZERS:
        attempt = {**diag, '优化器': method}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = model.fit(reml=True, method=method, maxiter=MAXITER, disp=False)
                covariance = result.cov_params().loc[result.fe_params.index, result.fe_params.index].to_numpy()
                variance = float(result.cov_re.iloc[0, 0])
                icc = variance / (variance + result.scale)
                warn = [str(w.message) for w in caught]
            attempt.update({'收敛': bool(result.converged), '随机截距方差': variance,
                            '残差方差': float(result.scale), 'ICC': icc, '近零方差': icc < NEAR_ZERO_ICC,
                            '警告': ' | '.join(warn), 'logLik': float(result.llf)})
            invalid_warning = any(('singular' in w.lower() or 'not positive definite' in w.lower()
                                   or 'failed to converge' in w.lower()) for w in warn)
            reliable = (result.converged and np.isfinite(result.fe_params).all() and
                        np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance).min() > 0 and
                        np.isfinite(result.llf) and np.isfinite(variance) and variance > 0 and
                        icc >= NEAR_ZERO_ICC and not invalid_warning)
            # statsmodels issues a boundary warning below an absolute variance of 0.01, which depends on the response units.
            # Retain the warning and check both the full parameter covariance and the fit with a second optimizer.
            if reliable and any('boundary' in w.lower() for w in warn):
                full_cov = result.cov_params().to_numpy()
                with warnings.catch_warnings(record=True) as verify_warnings:
                    warnings.simplefilter('always')
                    verify = model.fit(reml=True, method='powell', maxiter=MAXITER, disp=False)
                discrepancy = float(np.max(np.abs(verify.fe_params - result.fe_params)))
                attempt.update({'完整参数协方差最小特征值': float(np.linalg.eigvalsh(full_cov).min()),
                                'Powell核验收敛': bool(verify.converged),
                                'Powell核验随机方差': float(verify.cov_re.iloc[0, 0]),
                                'Powell核验最大固定系数差': discrepancy,
                                'Powell核验logLik差': float(abs(verify.llf-result.llf)),
                                'Powell核验警告': ' | '.join(str(w.message) for w in verify_warnings)})
                reliable = (np.isfinite(full_cov).all() and np.linalg.eigvalsh(full_cov).min() > 0 and
                            verify.converged and float(verify.cov_re.iloc[0, 0]) > 0 and
                            discrepancy < 1e-4 and abs(verify.llf-result.llf) < 1e-3)
                log(f'  边界警告核验（同样本同模型Powell）：最大固定系数差 {discrepancy:.3g}，可靠={reliable}。')
            attempt['可靠'] = bool(reliable)
            DIAGNOSTICS.append(attempt)
            log(f'  {method}：收敛={result.converged}，随机方差={variance:.6g}，ICC={icc:.6g}，可靠={reliable}；警告={warn}')
            if not reliable:
                continue
            term = f'C({factor}, Treatment(reference="{ref}"))[T.{target}]'
            beta = float(result.fe_params[term])
            se = float(np.sqrt(result.cov_params().loc[term, term]))
            if not np.isfinite(se) or se <= 0:
                raise ValueError('比较参数标准误无法估计')
            lo, hi = beta - norm.ppf(.975)*se, beta + norm.ppf(.975)*se
            p = float(2 * norm.sf(abs(beta / se)))
            effect = beta if metric == 'Recovery_Slope' else float(np.exp(beta))
            row.update({'调整后效应量': effect, 'CI下限': lo if metric == 'Recovery_Slope' else np.exp(lo),
                        'CI上限': hi if metric == 'Recovery_Slope' else np.exp(hi), '原始p值': p,
                        '相对变化百分比': np.nan if metric == 'Recovery_Slope' else 100*(effect-1),
                        '状态': '可靠', '显著性': '等待Holm校正', 'log尺度差值或原始斜率差': beta,
                        '标准误': se, '随机截距方差': variance, 'ICC': icc})
            # Use more precision than the default three-decimal summary so small p values are visible.
            coefficients = pd.DataFrame({'估计': result.fe_params,
                                         '标准误': result.bse_fe,
                                         '双侧p': 2*norm.sf(np.abs(result.fe_params/result.bse_fe))})
            log(coefficients.to_string(float_format=lambda x: f'{x:.8g}'))
            return row
        except Exception as exc:
            attempt['错误'] = repr(exc)
            if attempt not in DIAGNOSTICS:
                DIAGNOSTICS.append(attempt)
            log(f'  {method} 失败：{exc}')
    log(f'{metric} 模型 {model_name} 无可靠拟合；不发布效应量/CI/p，需人工检查诊断。')
    return row

def fit_prespecified_contrasts(data, metrics=('Shock-L', 'Shock-S', 'Recovery_Slope')):
    """Return English result and diagnostic tables for one analysis population."""
    diagnostic_start = len(DIAGNOSTICS)
    rows = []
    for metric in metrics:
        valid = valid_data(data, metric)
        estimates = pd.DataFrame([fit_compare(valid, metric, model) for model in ('A', 'B')])
        if estimates['状态'].eq('可靠').all() and estimates['原始p值'].notna().all():
            estimates['Holm校正p值'] = multipletests(estimates['原始p值'], method='holm')[1]
            estimates['显著性'] = np.where(
                estimates['Holm校正p值'] < ALPHA, '显著', '证据不足以支持差异')
        rows.append(estimates)
    return english_table(pd.concat(rows, ignore_index=True)), english_table(
        pd.DataFrame(DIAGNOSTICS[diagnostic_start:]))


def export_excel(tables):
    path = OUTPUT_DIR / 'Statistical_Analysis_Results.xlsx'
    tables = {SHEETS.get(name, name): english_table(frame) for name, frame in tables.items()}
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book:
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
            sheet.row_dimensions[1].height = 40
            for cell in sheet[1]:
                cell.font = Font(name='Microsoft YaHei', color='FFFFFF', bold=True)
                cell.fill = PatternFill('solid', fgColor='24476A')
                cell.alignment = Alignment(wrap_text=True, vertical='center')
            for j, cells in enumerate(sheet.columns, 1):
                header = str(sheet.cell(1, j).value)
                width = min(48, max(14, len(header)*2+2, max((len(str(c.value)) for c in list(cells)[1:101] if c.value is not None), default=0)+2))
                sheet.column_dimensions[get_column_letter(j)].width = width
                for cell in list(cells)[1:]:
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.000E+00' if header.startswith('P_') else ('0' if header.endswith(('_N', '_Sites')) or header in ['N', 'Records', 'Sites'] else '0.000000')
    wb = load_workbook(path, read_only=True, data_only=True)
    for name, frame in tables.items():
        assert wb[name].max_row == len(frame)+1
        assert wb[name].max_column == len(frame.columns)
    wb.close()
    log('工作簿已重新打开检查：所有表的行列数一致；数值为数值单元格，缺失为空。')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--metrics', nargs='+', choices=list(REFERENCE_N), default=list(REFERENCE_N))
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        log('M01 开始；运行日期（上海）：' + RUN_DATE)
        log(f'版本：pandas {pd.__version__}；numpy {np.__version__}；scipy {scipy.__version__}；statsmodels {statsmodels.__version__}')
        log('REML；双侧渐近正态Wald检验；同一指标2项预设比较Holm校正；95%CI为未校正区间。')
        log('CSV全项目清单如已生成保存在同目录；--scan 可刷新。边界警告不隐藏：绝对方差阈值受单位影响，以ICC、协方差正定及同模型第二优化器一致性核验。')
        if args.scan:
            scan_csv()
        data = prepare()
        if args.prepare_only:
            return
        rows, desc = [], []
        for metric in args.metrics:
            v = valid_data(data, metric)
            for factor in ['Event_Type', 'Compound_Group']:
                for group, sub in v.groupby(factor):
                    desc.append({'指标': metric, '分组方式': factor, '组': group, '有效样本量': len(sub),
                                 '断面数': sub.Site_ID.nunique(), '原始均值': sub[metric].mean(),
                                 '原始中位数': sub[metric].median(), '标准差': sub[metric].std(),
                                 '最小值': sub[metric].min(), '最大值': sub[metric].max()})
            rows.extend([fit_compare(v, metric, 'A'), fit_compare(v, metric, 'B')])
        results = pd.DataFrame(rows)
        for metric in args.metrics:
            idx = results.index[results['指标'].eq(metric)]
            if results.loc[idx, '原始p值'].notna().all():
                results.loc[idx, 'Holm校正p值'] = multipletests(results.loc[idx, '原始p值'], method='holm')[1]
                results.loc[idx, '显著性'] = np.where(results.loc[idx, 'Holm校正p值'] < ALPHA, '显著', '证据不足以支持差异')
            else:
                log(f'{metric} 预设两项比较未全部可靠估计：不缩小校正家族，Holm结果留空。')
        descriptive = pd.DataFrame(desc)
        diagnostics = pd.DataFrame(DIAGNOSTICS)
        save_csv(descriptive, '三指标描述性统计')
        save_csv(results, '两组核心比较统计结果')
        save_csv(diagnostics, '模型诊断明细')
        log('来源筛选说明：恢复计算脚本只输出第1、3、5、7、9天全部DO有效的事件；现有恢复值已保留原文件4位小数，不重新计算。')
        log('Shock-S 缺失需结合原结果 Pre_Baseline_Valid_Days 及既有计算条件理解，不据此补值。')
        log('整体复合比较按现有事件记录组成，非五类等权平均；不推出每种复合均强于每种单一。')
        log('只报告统计关联；不显著不是相同或等效。LMM假定条件残差独立和近似正态；未建模跨站同一气象过程相关或残差时间相关。重构DO不确定性未传播。')
        export_excel({'核心比较': results, '描述性统计': descriptive, '事件级数据': data,
                      '模型诊断': diagnostics, '输入来源': pd.DataFrame(SOURCES)})
        log('核心结果：\n' + results[['指标', '比较', '调整后效应量', 'CI下限', 'CI上限', '原始p值', 'Holm校正p值', '状态']].to_string(index=False))
        interpretations = ['# M01 论文表述建议', '', '调整 Reach、开始日期季节及 Site_ID 随机截距；REML 拟合，双侧渐近 Wald 检验，同指标两项比较 Holm 校正。95% CI 未作多重校正。', '']
        for _, r in results.iterrows():
            if r['状态'] != '可靠' or pd.isna(r['Holm校正p值']):
                interpretations.append(f"- {r['指标']} / {r['比较']}：无法可靠完成推断，须人工检查模型诊断。")
                continue
            effect = f"调整后响应比 {r['调整后效应量']:.4f}（相对变化 {r['相对变化百分比']:+.2f}%）" if r['效应类型'] == '响应比' else f"调整后斜率差 {r['调整后效应量']:.4f} mg L⁻¹ d⁻¹"
            verdict = '有统计证据支持组间差异' if r['Holm校正p值'] < ALPHA else '没有足够统计证据支持组间差异；不代表相同或统计等效'
            interpretations.append(f"- {r['指标']} / {r['比较']}：{effect}，95% CI [{r['CI下限']:.4f}, {r['CI上限']:.4f}]；原始 p={r['原始p值']:.6g}，Holm p={r['Holm校正p值']:.6g}。{verdict}。")
        interpretations += ['', '复合组和单一组的事件类型组成、样本数不均衡；总体比较不代表每一种复合事件都更强。响应比小于1表示相对DO比值较低；斜率差只表示早期DO变化速度差，不直接等同于恢复能力。所有结论是统计关联，不能解释为因果。', '', 'statsmodels 方法来源：https://www.statsmodels.org/stable/generated/statsmodels.regression.mixed_linear_model.MixedLM.fit.html']
    except Exception as exc:
        log('停止：' + repr(exc) + '；未随意匹配或删样本，请人工检查。')
        raise
    finally:
        (OUTPUT_DIR / '模型诊断与数据核对.txt').write_text('\n'.join(LOG), encoding='utf-8-sig')

if __name__ == '__main__':
    main()
