# -*- coding: utf-8 -*-
"""Summarize absolute DO and low-DO status in M07 using the saved S0 events, DO values and provenance flags."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib.util
import sys, os, hashlib, json
ROOT = Path(__file__).resolve().parents[3]
# ========================== Configuration ==========================
REGISTRY_DIR = ROOT/'Data/Event_F/01_Registry_Data/Overall'
DAILY_DIR = ROOT/'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR'
OBSERVED_DIR = ROOT/'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned'
SHOCK_FILE = ROOT/'Result/Decrease/Overall/FINAL_Shock_All_Events_Combined.csv'
OUT_BASE = ROOT/'Result/Decrease/DO_observation_sensitivity'
RUN_DATE = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
OUT = OUT_BASE/RUN_DATE
START, END = '2021-01-01','2025-12-31'
TYPES = ['Heatwave_Only','Flood_Only','Direct_Compound','H5F_Compound','F5H_Compound']
LABELS = ['Heatwave','Flood','DC','H5F','F5H']
EXPECTED = dict(zip(TYPES,[1192,1035,132,102,89]))
THRESHOLDS = [5,3,2]  # Use a strict less-than threshold in mg/L.
MIN_PRE_DAYS = 3
MEAN_ATOL = 1e-8
OBSERVED_ATOL = 2e-6  # Allow for existing float32 precision and require identical missing-value masks.
NEAR_REFERENCE_BAND = .05  # Flag ratios within 0.05 of one for description only. Preserve the original ratio.
STANDARD_URL = 'https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/shjbh/shjzlbz/200206/W020061027509896672057.pdf'
# ============================================================
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

# Import only the M06 reading, key normalization and Excel export utilities.
spec=importlib.util.spec_from_file_location('m06_io',ROOT/'Code/05_Event_comparisons_and_robustness/04_Spatial_independence/M06_spatial_independence.py')
io=importlib.util.module_from_spec(spec); spec.loader.exec_module(io)
io.OUT=OUT; io.RUN_DATE=RUN_DATE
read, normalize, require, log = io.read,io.normalize,io.require,io.log
LOG,SOURCES = io.LOG,io.SOURCES
CALENDAR=pd.date_range(START,END)

def issue(reason,detail=None):
    if detail is not None: detail.to_csv(OUT/f'{RUN_DATE}_M07问题清单.csv',index=False,encoding='utf-8-sig')
    raise ValueError(reason)

def daily_read(path,site,fields):
    d=read(path); require(d,['WQS_name','Date']+fields,str(path))
    if set(d.WQS_name.dropna())!={site}: issue(f'{path}站点不一致')
    d['Date']=pd.to_datetime(d.Date,errors='raise').dt.normalize()
    if d.Date.duplicated().any(): issue(f'{path}日期重复',d[d.Date.duplicated(False)])
    d=d.set_index('Date').sort_index()
    if len(CALENDAR.difference(d.index)): issue(f'{path}缺少研究期日历，不能将缺日当观测')
    d=d.reindex(CALENDAR)
    for c in fields:
        value=pd.to_numeric(d[c],errors='coerce')
        if (d[c].notna() & value.isna()).any() or (value.notna() & ~np.isfinite(value)).any():
            issue(f'{path} {c}有非数值/无穷值')
        d[c]=value
    return d

def load_events():
    events=pd.concat([normalize(read(REGISTRY_DIR/f'{t}_01_Registry_Overall.csv'),t) for t in TYPES],ignore_index=True)
    if events.duplicated(['Site_ID','Event_ID']).any(): issue('注册表跨类别键重复')
    if events.groupby('Site_ID').Reach.nunique().gt(1).any(): issue('同一断面河段不一致')
    shock=normalize(read(SHOCK_FILE),'原始Shock')
    require(shock,['Event_DO','Shock_Ratio_LongTerm','Shock_Ratio_Pre','Pre_Baseline_DO','Pre_Baseline_Valid_Days'],'原始Shock')
    merged=events.merge(shock,on=['Site_ID','Event_ID'],how='outer',validate='one_to_one',suffixes=('', '_shock'),indicator=True)
    bad=merged['_merge'].ne('both')
    for c in ['Reach','Event_Type','Start_Date','End_Date']: bad|=merged[c].ne(merged[c+'_shock'])
    if bad.any(): issue('S0与Shock键/属性/日期不一致',merged[bad])
    for c in ['Event_DO','Shock_Ratio_LongTerm','Shock_Ratio_Pre','Pre_Baseline_DO','Pre_Baseline_Valid_Days']:
        value=pd.to_numeric(merged[c],errors='coerce')
        if (merged[c].notna() & value.isna()).any(): issue(f'{c}含非数值')
        merged[c]=value
    for t,n in EXPECTED.items(): log(f'{t}实际{int(events.Event_Type.eq(t).sum())}，参考{n}，差值{int(events.Event_Type.eq(t).sum())-n}')
    log(f'S0 {len(events)}站点—事件、{events.Site_ID.nunique()}断面；全部可靠关联。')
    return merged[['Site_ID','Event_ID','Event_Type','Reach','Start_Date','End_Date','Event_DO','Shock_Ratio_LongTerm','Shock_Ratio_Pre','Pre_Baseline_DO','Pre_Baseline_Valid_Days']]

def metrics(values,suffix=''):
    v=values.dropna(); n=len(v)
    out={'Event_DO_Min'+suffix:v.min() if n else np.nan}
    if suffix: out['Event_DO_Mean'+suffix]=v.mean() if n else np.nan
    for threshold in THRESHOLDS:
        days=int(v.lt(threshold).sum()) if n else np.nan
        out[f'DO_Below{threshold}_Days'+suffix]=days
        out[f'DO_Below{threshold}_Pct'+suffix]=100*days/n if n else np.nan
        out[f'Any_DO_Below{threshold}'+suffix]=bool(days>0) if n else pd.NA
    return out

def calculate(events):
    rows=[]; provenance=[]; daily_audit=[]
    for site,group in events.groupby('Site_ID',sort=True):
        d=daily_read(DAILY_DIR/f'{site}_Imputed.csv',site,['DO','Original_DO','Imputed_DO'])
        raw=daily_read(OBSERVED_DIR/f'{site}.csv',site,['DO'])
        observed=d.Original_DO.notna(); imputed=~observed & d.DO.notna()
        mismatch=observed.ne(raw.DO.notna()) | ~np.isclose(d.Original_DO,raw.DO,atol=OBSERVED_ATOL,rtol=0,equal_nan=True)
        mismatch|=observed & ~np.isclose(d.DO,d.Original_DO,atol=MEAN_ATOL,rtol=0,equal_nan=True)
        mismatch|=d.Imputed_DO.notna().ne(imputed)
        mismatch|=imputed & ~np.isclose(d.DO,d.Imputed_DO,atol=MEAN_ATOL,rtol=0,equal_nan=True)
        if mismatch.any(): issue(f'{site}观测掩码/填补来源与清洗数据不一致',d.loc[mismatch].reset_index())
        occupied=pd.Series(False,index=CALENDAR)
        for ev in group.itertuples():
            window=occupied.index.to_series().between(ev.Start_Date,ev.End_Date).to_numpy()
            if occupied.loc[window].any(): issue(f'{site}S0事件窗口重叠')
            occupied.loc[window]=True
        daily_audit.append({'Site_ID':site,'Daily_N':len(d),'Observed_Days':int(observed.sum()),
                            'Imputed_Days':int(imputed.sum()),'Original_Zero_Days':int(d.Original_DO.eq(0).sum()),
                            'Completed_Negative_Days':int(d.DO.lt(0).sum()),'Provenance_Check':'通过'})
        for ev in group.itertuples():
            dates=pd.date_range(ev.Start_Date,ev.End_Date)
            period=d.loc[dates]; values=period.DO; obs=period.Original_DO
            n=int(values.count()); no=int(obs.count()); ni=int((obs.isna() & values.notna()).sum())
            calculated=values.mean()
            if not np.isclose(calculated,ev.Event_DO,atol=MEAN_ATOL,rtol=0,equal_nan=True):
                issue(f'{ev.Event_ID}事件DO均值不一致',pd.DataFrame([{'Site_ID':site,'Event_ID':ev.Event_ID,'Existing':ev.Event_DO,'Calculated':calculated}]))
            r={'Site_ID':site,'Event_ID':ev.Event_ID,'Event_Type':ev.Event_Type,'Reach':ev.Reach,
               'Event_Start':ev.Start_Date,'Event_End':ev.End_Date,'Shock_L':ev.Shock_Ratio_LongTerm,'Shock_S':ev.Shock_Ratio_Pre,
               'Event_DO_Mean':ev.Event_DO,'Event_DO_Mean_Check':calculated,'Event_Duration_Days':len(dates),
               'Event_DO_Valid_Days':n,'Event_DO_Coverage_Pct':100*n/len(dates),'Observed_DO_Days':no,'Imputed_DO_Days':ni,
               'Observed_Coverage_Pct':100*no/len(dates),'Imputed_Coverage_Pct':100*ni/len(dates),
               'Imputed_Fraction_Valid_Pct':100*ni/n if n else np.nan,'No_Observed_DO':no==0,
               'Majority_Imputed':ni>n/2 if n else pd.NA,**metrics(values),**metrics(obs,'_Observed')}
            for threshold in THRESHOLDS:
                r[f'DO_Below{threshold}_Days_Imputed']=int((values.lt(threshold) & obs.isna()).sum()) if n else np.nan
            r['Minimum_Has_Observed_Tie']=bool((values.eq(values.min()) & obs.notna()).any()) if n else pd.NA
            r['Minimum_Has_Imputed_Tie']=bool((values.eq(values.min()) & obs.isna()).any()) if n else pd.NA
            # Use the original Shock-S window, days 9–4 before onset, excluding all S0 event days and requiring at least three valid DO values.
            pre_dates=pd.date_range(ev.Start_Date-pd.Timedelta(days=9),ev.Start_Date-pd.Timedelta(days=4)).intersection(CALENDAR)
            pre=d.loc[pre_dates].loc[~occupied.reindex(pre_dates).to_numpy()]
            npd=int(pre.DO.count()); eligible=npd>=MIN_PRE_DAYS
            if np.isfinite(ev.Pre_Baseline_Valid_Days) and npd!=ev.Pre_Baseline_Valid_Days:
                issue(f'{ev.Event_ID}事件前有效日数与原结果不一致')
            pm=pre.DO.mean() if eligible else np.nan
            if not np.isclose(pm,ev.Pre_Baseline_DO,atol=MEAN_ATOL,rtol=0,equal_nan=True): issue(f'{ev.Event_ID}事件前基线不一致')
            r.update({'Pre_DO_Valid_Days':npd,'Pre_Window_Eligible':eligible,
                      'Pre_DO_Below5_Any':bool(pre.DO.lt(5).any()) if eligible else pd.NA,
                      'Pre_DO_Below5_Pct':100*pre.DO.lt(5).sum()/npd if eligible else np.nan})
            if no: r['Minimum_Observed_minus_Full']=r['Event_DO_Min_Observed']-r['Event_DO_Min']
            else: r['Minimum_Observed_minus_Full']=np.nan
            rows.append(r)
            # Save provenance for all days with DO below 2, including screened DO and that day's saturation threshold. This step traces records without screening again.
            for date,p in period.loc[values.lt(2)].iterrows():
                provenance.append({'Site_ID':site,'Event_ID':ev.Event_ID,'Event_Type':ev.Event_Type,'Reach':ev.Reach,
                                   'Date':date,'DO':p.DO,'Original_DO':p.Original_DO,'Imputed_DO':p.Imputed_DO,
                                   'Source':'Observed' if pd.notna(p.Original_DO) else 'Imputed',
                                   'Cleaned_DO':raw.loc[date,'DO'],'DO_1.5_sat':raw.loc[date].get('DO_1.5_sat',np.nan)})
    table=pd.DataFrame(rows)
    for suffix in ['', '_Observed']:
        for threshold in THRESHOLDS:
            c=table[f'DO_Below{threshold}_Pct'+suffix].dropna()
            if not c.between(0,100).all(): issue('低DO比例超出0—100')
        if (table['DO_Below2_Days'+suffix]>table['DO_Below3_Days'+suffix]).any() or (table['DO_Below3_Days'+suffix]>table['DO_Below5_Days'+suffix]).any(): issue('低DO阈值嵌套失败')
        if (table['Event_DO_Min'+suffix]>table['Event_DO_Mean'+suffix]+MEAN_ATOL).any(): issue('最低DO大于均值')
    if not (table.Observed_DO_Days+table.Imputed_DO_Days).eq(table.Event_DO_Valid_Days).all(): issue('观测与填补日重复或遗漏')
    log(f'{len(table)}事件平均DO与原结果一致；观测来源逐日核对{len(daily_audit)}站通过；指标嵌套/分母/极值检查通过。')
    columns=['Site_ID','Event_ID','Event_Type','Reach','Date','DO','Original_DO','Imputed_DO','Source','Cleaned_DO','DO_1.5_sat']
    return table,pd.DataFrame(provenance,columns=columns),pd.DataFrame(daily_audit)

def summary(table,suffix='',subset='Full',spatial=False):
    data=table.copy(); rows=[]
    for reach in (['All','Upstream','Midstream','Downstream'] if spatial else ['All']):
        dr=data if reach=='All' else data[data.Reach.eq(reach)]
        for typ in ['All']+TYPES:
            g=dr if typ=='All' else dr[dr.Event_Type.eq(typ)]
            valid=g['Event_DO_Mean'+suffix].notna(); v=g.loc[valid]
            for weighting in (['Event_Equal','Site_Equal'] if spatial else ['Event_Equal']):
                r={'Subset':subset,'Reach':reach,'Event_Type':typ,'Weighting':weighting,
                   'Registry_Event_N':len(g),'Event_N':len(v),'Site_N':v.Site_ID.nunique()}
                for c in ['Event_DO_Mean','Event_DO_Min','DO_Below5_Days','DO_Below5_Pct']:
                    col=c+suffix; vals=v[col].dropna()
                    used=v[v[col].notna()]
                    samples=used.groupby('Site_ID')[col].mean() if weighting=='Site_Equal' else vals
                    r[c+'_Mean']=samples.mean(); r[c+'_Median']=samples.median()
                    r[c+'_Valid_Event_N']=len(vals); r[c+'_Valid_Site_N']=used.Site_ID.nunique()
                for threshold in THRESHOLDS:
                    col=f'Any_DO_Below{threshold}'+suffix; present=v[col].fillna(False).astype(bool)
                    r[f'Below{threshold}_Event_N']=int(present.sum())
                    r[f'Below{threshold}_Site_N']=v.loc[present,'Site_ID'].nunique()
                    r[f'Below{threshold}_Eligible_Event_N']=len(v)
                    if weighting=='Event_Equal': rate=100*present.mean() if len(v) else np.nan
                    else: rate=100*v.assign(flag=present.astype(float)).groupby('Site_ID').flag.mean().mean()
                    r[f'Below{threshold}_Event_Pct']=rate
                    r[f'Below{threshold}_Days_Total']=v[f'DO_Below{threshold}_Days'+suffix].sum() if len(v) else np.nan
                days='Observed_DO_Days' if suffix else 'Event_DO_Valid_Days'
                r['Valid_DO_Days_Total']=int(v[days].sum())
                r['Below5_Pooled_Day_Pct']=100*r['Below5_Days_Total']/r['Valid_DO_Days_Total'] if r['Valid_DO_Days_Total'] else np.nan
                r['Sparse_Note']='站点覆盖有限' if r['Site_N']<5 else ('下游仅5站，谨慎推广' if reach=='Downstream' else '')
                rows.append(r)
    return pd.DataFrame(rows)

def relationships(table):
    rows=[]
    for metric in ['Shock_L','Shock_S']:
        d=table[np.isfinite(table[metric]) & table.Event_DO_Min.notna()]
        log(f'{metric}绝对DO对照有效{len(d)}事件；参考'+('2550' if metric=='Shock_L' else '1898'))
        for rel in ['<1','>=1']:
            for low in [True,False]:
                mask=(d[metric].lt(1) if rel=='<1' else d[metric].ge(1)) & d.Event_DO_Min.lt(5).eq(low)
                g=d[mask]; r={'Metric':metric,'Ratio_Class':rel,'Minimum_DO_Class':'<5' if low else '>=5',
                             'Event_N':len(g),'Eligible_Event_N':len(d),'Event_Pct':100*len(g)/len(d) if len(d) else np.nan,
                             'Site_N':g.Site_ID.nunique()}
                rows.append(r)
        near=d[(d[metric]-1).abs().le(NEAR_REFERENCE_BAND)]
        rows.append({'Metric':metric,'Ratio_Class':'Within_5pct_of_1','Minimum_DO_Class':'<5',
                     'Event_N':int(near.Event_DO_Min.lt(5).sum()),'Eligible_Event_N':len(near),
                     'Event_Pct':100*near.Event_DO_Min.lt(5).mean() if len(near) else np.nan,
                     'Site_N':near.loc[near.Event_DO_Min.lt(5),'Site_ID'].nunique()})
    return pd.DataFrame(rows)

def observed_comparison(table):
    rows=[]
    for typ in ['All']+TYPES:
        g=table if typ=='All' else table[table.Event_Type.eq(typ)]
        paired=g[g.Observed_DO_Days.gt(0)]
        r={'Event_Type':typ,'Event_N':len(g),'Site_N':g.Site_ID.nunique(),'With_Observed_Event_N':len(paired),
           'With_Observed_Site_N':paired.Site_ID.nunique(),'No_Observed_Event_N':int(g.No_Observed_DO.sum()),
           'Majority_Imputed_Event_N':int(g.Majority_Imputed.fillna(False).sum()),
           'Observed_Days_Total':int(g.Observed_DO_Days.sum()),'Imputed_Days_Total':int(g.Imputed_DO_Days.sum()),
           'Paired_Min_Difference_Mean':paired.Minimum_Observed_minus_Full.mean(),
           'Paired_Min_Difference_Median':paired.Minimum_Observed_minus_Full.median(),
           'Paired_Full_Below5_Event_Pct':100*paired.Any_DO_Below5.astype(bool).mean() if len(paired) else np.nan,
           'Paired_Observed_Below5_Event_Pct':100*paired.Any_DO_Below5_Observed.astype(bool).mean() if len(paired) else np.nan,
           'Paired_Full_DayPct_Mean':paired.DO_Below5_Pct.mean(),'Paired_Observed_DayPct_Mean':paired.DO_Below5_Pct_Observed.mean()}
        for t in THRESHOLDS:
            r[f'Below{t}_Observed_Days']=int(g[f'DO_Below{t}_Days_Observed'].sum())
            r[f'Below{t}_Imputed_Days']=int(g[f'DO_Below{t}_Days_Imputed'].sum())
            r[f'Below{t}_Imputed_Only_Event_N']=int((g[f'Any_DO_Below{t}'].fillna(False).astype(bool) & ~g[f'Any_DO_Below{t}_Observed'].fillna(False).astype(bool)).sum())
            r[f'Below{t}_NoObserved_Event_N']=int((g.No_Observed_DO & g[f'Any_DO_Below{t}'].fillna(False)).sum())
        rows.append(r)
    return pd.DataFrame(rows)

def write_report(table,full,observed,reach,comparison,rel,audit):
    total=full[full.Event_Type.eq('All')].iloc[0]; comp=comparison[comparison.Event_Type.eq('All')].iloc[0]
    lines=['# M07 绝对DO与低DO状态结果及论文补充建议','',f'运行日期：{RUN_DATE}。只读取S0和已有DO/来源标记，不重算Shock，不重跑M01—M06。', '',
           '## 1. 五类事件的绝对浓度','',
           '完整序列采用观测与填补DO。下面两个浓度均为事件等权均值：先求每条事件的平均/最低DO，再对事件汇总。最低DO是日尺度记录最低，不是瞬时最低。','',
           '|类型|有效事件/断面|事件平均DO的均值|事件最低DO的均值|事件最低DO的中位数|<5事件比例|<3事件比例|<2事件比例|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in full.itertuples():
        lines.append(f'|{r.Event_Type}|{r.Event_N}/{r.Site_N}|{r.Event_DO_Mean_Mean:.4f}|{r.Event_DO_Min_Mean:.4f}|{r.Event_DO_Min_Median:.4f}|{r.Below5_Event_Pct:.2f}%|{r.Below3_Event_Pct:.2f}%|{r.Below2_Event_Pct:.2f}%|')
    top5=full[full.Event_Type.ne('All')].sort_values('Below5_Event_Pct',ascending=False).iloc[0]
    top3=full[full.Event_Type.ne('All')].sort_values('Below3_Event_Pct',ascending=False).iloc[0]
    top2=full[full.Event_Type.ne('All')].sort_values('Below2_Event_Pct',ascending=False).iloc[0]
    lines+=['',f'浓度单位mg L⁻¹。描述性<5事件比例最高为{top5.Event_Type}（{top5.Below5_Event_N}/{top5.Event_N}，{top5.Below5_Event_Pct:.2f}%）；<3和<2比例最高分别为{top3.Event_Type}和{top2.Event_Type}。没有检验组间显著性，不能据此声称生态损害最严重。',
             '',f'全部{int(total.Event_N)}事件中，<5为{int(total.Below5_Event_N)}（{total.Below5_Event_Pct:.2f}%），<3为{int(total.Below3_Event_N)}（{total.Below3_Event_Pct:.2f}%），<2为{int(total.Below2_Event_N)}（{total.Below2_Event_Pct:.2f}%）。',
             f'<5日尺度记录合计{int(total.Below5_Days_Total)}天，事件内比例的事件等权平均为{total.DO_Below5_Pct_Mean:.2f}%；合并低值日数/合并有效日数为{total.Below5_Pooled_Day_Pct:.2f}%，二者分母与权重不同。', '',
             '## 2. 河段空间结果','', '|河段|事件/断面|站点等权平均事件DO|站点等权平均事件最低DO|站点等权<5事件比例|站点等权低值日比例均值|',
             '|---|---:|---:|---:|---:|---:|']
    for r in reach[(reach.Subset.eq('Full')) & reach.Weighting.eq('Site_Equal') & reach.Event_Type.eq('All') & reach.Reach.ne('All')].itertuples():
        lines.append(f'|{r.Reach}|{r.Event_N}/{r.Site_N}|{r.Event_DO_Mean_Mean:.4f}|{r.Event_DO_Min_Mean:.4f}|{r.Below5_Event_Pct:.2f}%|{r.DO_Below5_Pct_Mean:.2f}%|')
    lines+=['','河段先求各站事件统计，再对参与站点等权；下游仅5站，类型分组更少，不能把个别低值推广为整个下游普遍生态风险。详见Reach_Summary的有效事件数与站点数。', '',
             '## 3. 相对偏离与绝对低DO','', '|指标|比值类别|最低DO类别|事件数/有效分母|比例|低值组断面数|',
             '|---|---|---|---:|---:|---:|']
    for r in rel.itertuples(): lines.append(f'|{r.Metric}|{r.Ratio_Class}|{r.Minimum_DO_Class}|{r.Event_N}/{r.Eligible_Event_N}|{r.Event_Pct:.2f}%|{r.Site_N}|')
    lines+=['','接近1的辅助定义为|ratio−1|≤0.05；该行分母是接近1的有效事件，四象限各行分母则是该Shock指标全部有效事件。原始指标和样本不变。相对比值较高不保证日尺度DO记录始终达到5 mg/L；这不是生态风险在事件期间发生逆转的证据。', '',
             '## 4. 事件前低值','']
    eligible=table[table.Pre_Window_Eligible]; low=eligible[eligible.Any_DO_Below5]
    pre_n=int(eligible.Pre_DO_Below5_Any.astype(bool).sum()); both=int(low.Pre_DO_Below5_Any.astype(bool).sum())
    lines+=[f'原事件前第9—4天排除S0事件日，并要求至少3个有效DO。{len(eligible)}事件有符合要求的窗口，其中{pre_n}（{100*pre_n/len(eligible):.2f}%）已出现<5记录。在事件期出现<5且有合格前期窗口的{len(low)}事件中，{both}（{100*both/len(low):.2f}%）在事件前已出现<5。前期无合格窗口者保留缺失，不能认定事件前没有低值。这些窗口使用完整DO序列，可能包含填补值。', '',
             '## 5. 观测与填补的影响','',
             f'{int(comp.With_Observed_Event_N)}事件有至少1个实测日，{int(comp.No_Observed_Event_N)}事件没有实测DO，{int(comp.Majority_Imputed_Event_N)}事件超过一半有效日来自填补。无观测事件的观测子集指标保持缺失。',
             f'在同一组{int(comp.With_Observed_Event_N)}条有观测事件中，完整序列与仅观测序列的<5事件比例分别为{comp.Paired_Full_Below5_Event_Pct:.2f}%和{comp.Paired_Observed_Below5_Event_Pct:.2f}%；观测最低值减完整最低值的均差为{comp.Paired_Min_Difference_Mean:.4f} mg L⁻¹，中位差为{comp.Paired_Min_Difference_Median:.4f} mg L⁻¹。',
             f'观测日较少会降低捕捉最低值的机会。全体{len(table)}个事件与观测子集的原始比例还受有效日期差异影响。','',
             '|阈值|观测低值日数|填补低值日数|填补占低值日比例|完整有低值而观测未发现事件|其中无观测事件|',
             '|---|---:|---:|---:|---:|---:|']
    for t in THRESHOLDS:
        no=comp[f'Below{t}_Observed_Days']; ni=comp[f'Below{t}_Imputed_Days']
        lines.append(f'|<{t}|{int(no)}|{int(ni)}|{100*ni/(no+ni):.2f}%|{int(comp[f"Below{t}_Imputed_Only_Event_N"])}|{int(comp[f"Below{t}_NoObserved_Event_N"])}|')
    lines+=['','<2等极低值必须同时报告来源；不能统一称为实测缺氧。原质控保留0 mg/L，原始DO负值、>20或高于1.5倍动态饱和限值被置缺失；本次不重新应用不同阈值。清洗文件、Original_DO掩码及保留观测数值逐日核对通过；低于2的所有记录在Below2_Day_Trace给出。',
             f'按低于2的日记录计，填补值占{100*comp.Below2_Imputed_Days/(comp.Below2_Observed_Days+comp.Below2_Imputed_Days):.2f}%；按事件计，{int(total.Below2_Event_N-comp.Below2_Imputed_Only_Event_N)}条已有观测低值，{int(comp.Below2_Imputed_Only_Event_N)}条低值只见于填补日期（其中{int(comp.Below2_NoObserved_Event_N)}条完全无观测）。因此既不能称极低值全由填补产生，也不能把全部结果当作实测证据。',
             f'研究期已有观测0值日期合计{int(audit.Original_Zero_Days.sum())}（跨站计数）；完整DO负值日期{int(audit.Completed_Negative_Days.sum())}。这些计数不改变原质控。',
             '50%观测覆盖敏感性结果已分开列在Event_Type_Summary，不替换全样本。类别排名和日比例的完整/观测结果需据各自分母阅读。','',
             '## 6. 可用于论文的内容与解释边界','',
             '**§2.4 方法补充。** 基于既有S0事件窗口，从保留观测并填补缺失的逐日DO序列提取事件平均与最低DO，均值与既有Event_DO逐条核验。以DO严格低于5、3、2 mg L⁻¹的有效日期数及其占事件有效DO日期比例量化低DO状态。阈值对应GB3838—2002的III、IV、V类DO浓度参照，不用于单独确定总体水质类别或生物损害。对原始质控观测另行计算同类指标，报告观测/填补覆盖和观测覆盖≥50%的敏感性结果。全流域事件等权，河段断面等权，不新增显著性检验。', '',
             f'**§3.3—3.4 结果补充。** 全部{int(total.Event_N)}事件中，{int(total.Below5_Event_N)}条（{total.Below5_Event_Pct:.2f}%）至少有一条日尺度DO记录低于5 mg L⁻¹，{int(total.Below2_Event_N)}条（{total.Below2_Event_Pct:.2f}%）低于2 mg L⁻¹。相对偏离和绝对浓度不一一对应，四象限和接近1的结果应给出本表中的指标特异分母。低值来源需与观测/填补表共同报告。', '',
             '**§4.2 讨论补充。** 相对背景比值反映参照依赖的偏离，绝对浓度反映另一层水环境状态。相对DO不下降仍可能伴随低于水质DO参照的日记录；部分低值在事件前已存在。日尺度数据无法识别瞬时最低值或每一天24小时的低氧持续时间，模型填补、观测覆盖、事件长度与稀疏空间样本也影响最低值的捕捉。因此这些结果支持结合绝对DO、参照背景和来源标记评价氧状况，不能建立极端事件直接引起生态损害的因果链。', '',
             f'**摘要和结论。** 可以引用{int(total.Below5_Event_N)}/{int(total.Event_N)}等实际低DO事件数量、可靠的来源分解及相对指标与绝对浓度不一致的计数，但应明确“完整DO序列中的日尺度记录”。最低值排名、不同类型低值比例差异、下游少量断面的空间推广及“新增/加重低氧”均仅能探索性解释，不作显著性或因果声明。', '',
             '**题目措辞。** M07补充的是绝对状态及其与相对偏离的对应，不证明冲击因果性。现有V12采用departures较为稳妥；无需仅因本分析再修改题目，也不能说M07证明shocks必然不适用。', '',
             '## 图说明','',
             '图A：完整序列各类型最低DO箱线图（中位数、25—75百分位、1.5IQR须，显示全部离群事件）；另两面板为<5事件比例和事件内低值日比例的事件等权均值，没有误差条或显著性检验。图B保留全部极端值并列出四象限计数。英文Arial、PNG600dpi、PDF矢量文本。','',
             f'阈值来源：[GB3838—2002 官方文本]({STANDARD_URL})。本分析不把低DO日期数解释为完整24小时低氧，也不将日记录最低值称为瞬时最低。']

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    log('仅M07；DO<5/3/2按GB3838—2002浓度参照，不判定整体水质类别或鱼类死亡；'+STANDARD_URL)
    events=load_events(); table,low_trace,audit=calculate(events)
    full=summary(table); observed=summary(table,'_Observed','Observed')
    sensitivity=summary(table[table.Observed_Coverage_Pct.ge(50)],subset='ObservedCoverage_GE50_Full')
    sensitivity_obs=summary(table[table.Observed_Coverage_Pct.ge(50)],'_Observed','ObservedCoverage_GE50_Observed')
    reach=pd.concat([summary(table,spatial=True),summary(table,'_Observed','Observed',spatial=True)],ignore_index=True)
    rel=relationships(table); comparison=observed_comparison(table)
    sheets={'Event_Absolute_DO':table,'Event_Type_Summary':pd.concat([full,observed,sensitivity,sensitivity_obs],ignore_index=True),
            'Reach_Summary':reach,'Observed_Imputed':comparison,'Shock_DO_Relationship':rel,
            'Data_QC':pd.DataFrame(SOURCES),'Daily_Provenance_QC':audit,'Below2_Day_Trace':low_trace,
            'Definitions':pd.DataFrame({'说明':['DO均值mg/L，最低值为日尺度记录最低，不是瞬时最低。','严格DO<5/3/2；阈值对应GB3838—2002 III/IV/V类DO参照，不单独判断整体水质。',
             '事件低DO比例分母为该事件有效DO日数；无观测事件的Observed指标为空。','Event_Equal逐事件等权；Site_Equal先求站内均值再对站等权；Site_Equal中位数为站均值的中位数。',
             'Below5_Pooled_Day_Pct为合并有效日比例，与事件比例均值不同；Site_Equal行仅作另列参考，不能冒充站点等权。','观测是质控后日均值，与Original_DO逐日掩码一致；填补不冒充实测；不重复质控或删除极低值。',
             'Imputed_Only表示完整序列出现低值而可用观测未出现；NoObserved单列，不能据此推断没有实测低值。','前9—4天排除S0事件日，完整DO有效至少3天才给出前期低DO状态。','接近1辅助定义为距1不超过5%；未改变原Shock结果。','50%观测覆盖子集仅敏感性分析；无统计检验、无因果推断。',STANDARD_URL]})}
    table.to_csv(OUT/f'{RUN_DATE}_M07事件绝对DO指标.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    for name,d in sheets.items():
        if name!='Event_Absolute_DO': d.to_csv(OUT/f'{RUN_DATE}_M07_{name}.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    io.excel('M07五类事件对比',{'Event_Type_Summary':sheets['Event_Type_Summary']})
    io.excel('M07河段低DO分析',{'Reach_Summary':reach})
    io.excel('M07观测与重构对比',{'Observed_Imputed':comparison,'Below2_Day_Trace':low_trace,'Daily_Provenance_QC':audit})
    io.excel('M07Shock与低DO关系',{'Shock_DO_Relationship':rel})
    io.excel('M07结果汇总',sheets)
    for source in SOURCES:
        assert hashlib.sha256((ROOT/source['File']).read_bytes()).hexdigest()==source['SHA256'],'原文件改变'
    log('输入SHA256运行前后相同；5工作簿已重读检查。')
    log('五类完整序列汇总：\n'+full[['Event_Type','Event_N','Site_N','Event_DO_Mean_Mean','Event_DO_Min_Mean','Below5_Event_Pct','Below3_Event_Pct','Below2_Event_Pct']].to_string(index=False))
    log('观测/填补对照：\n'+comparison.to_string(index=False))
    write_report(table,full,observed,reach,comparison,rel,audit)

if __name__=='__main__':
    try: main()
    except Exception as exc: log('停止正式汇总：'+repr(exc)+'；需要人工核查，未修改原始结果。'); raise
    finally:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/f'{RUN_DATE}_M07数据核查日志.txt').write_text('\n'.join(LOG),encoding='utf-8')
