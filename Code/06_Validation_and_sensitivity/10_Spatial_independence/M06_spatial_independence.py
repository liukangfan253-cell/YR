# -*- coding: utf-8 -*-
"""Reuse the existing station matching, S0 events and Shock-L values in M06."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import sys
import os
ROOT = Path(__file__).resolve().parents[3]
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# ======================== Configuration ========================
MATCH_FILE = ROOT / 'Data/WQ_Hy_M_DYN_data/YR/Match_Summary_Retained.csv'
REGISTRY_DIR = ROOT / 'Data/Event_F/01_Registry_Data/Overall'
SHOCK_FILE = ROOT / 'Result/Decrease/Overall/FINAL_Shock_All_Events_Combined.csv'
HY_RAW_DIR = ROOT / 'Data/Hydrology_data_P/Hy_data_YR/Original/Retained'
HY_FILLED_DIR = ROOT / 'Data/Hydrology_data_P/Hy_data_YR/Interpolated/Retained'
MERGED_DIR = ROOT / 'Data/WQ_Hy_M_DYN_data/YR'
OUTPUT_BASE = ROOT / 'Result/Data_preprocessing/Hydrological_matching'
START, END = '2021-01-01', '2025-12-31'
TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']
EXPECTED = dict(zip(TYPES, [1192, 1035, 132, 102, 89]))
REACH_MAP = {'UR': 'Upstream', 'MR': 'Midstream', 'LR': 'Downstream'}
REACH_FOLDER = {'Upstream': 'upstream', 'Midstream': 'midstream', 'Downstream': 'downstream'}
RUN_DATE = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')  # Run date in Beijing time
OUT = OUTPUT_BASE / RUN_DATE
MIN_TYPE_SITES = 3
Q_ATOL = 1e-8
# ========================================================
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
LOG, SOURCES = [], []
CALENDAR = pd.date_range(START, END)

def log(s):
    LOG.append(str(s)); print(s, flush=True)

def read(path):
    for enc in ['utf-8-sig', 'gb18030']:
        try:
            d = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError: continue
    else: raise ValueError(f'不能解码：{path}')
    SOURCES.append({'File': str(path.relative_to(ROOT)), 'Rows': len(d), 'Fields': '|'.join(d.columns),
                    'SHA256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return d

def require(d, fields, label):
    missing = set(fields)-set(d)
    if missing: raise ValueError(f'{label}缺字段：{missing}')

def normalize(d, label):
    d = d.rename(columns={'WQS_name': 'Site_ID', 'Region': 'Reach'})
    require(d, ['Site_ID','Event_ID','Reach','Event_Type','Start_Date','End_Date'], label)
    if d[['Site_ID','Event_ID']].isna().any().any() or d.duplicated(['Site_ID','Event_ID']).any():
        raise ValueError(f'{label}键缺失或重复')
    d['Reach'] = d.Reach.map(REACH_MAP)
    if d.Reach.isna().any() or not d.Event_Type.isin(TYPES).all():
        raise ValueError(f'{label}有未知河段或事件类型')
    for c in ['Start_Date','End_Date']: d[c] = pd.to_datetime(d[c], errors='raise')
    if not d.Start_Date.between(START, END).all() or not d.End_Date.between(START, END).all():
        raise ValueError(f'{label}日期超出研究期')
    if (d.End_Date < d.Start_Date).any(): raise ValueError(f'{label}事件日期颠倒')
    return d

def load_events():
    reg = pd.concat([normalize(read(REGISTRY_DIR/f'{t}_01_Registry_Overall.csv'), t) for t in TYPES], ignore_index=True)
    if reg.duplicated(['Site_ID','Event_ID']).any(): raise ValueError('五表之间键重复')
    if reg.groupby('Site_ID').Reach.nunique().gt(1).any(): raise ValueError('同一站点河段冲突')
    src = normalize(read(SHOCK_FILE), 'Shock-L')
    require(src, ['Shock_Ratio_LongTerm'], 'Shock-L')
    check = src.merge(reg[['Site_ID','Event_ID','Reach','Event_Type','Start_Date','End_Date']],
                      on=['Site_ID','Event_ID'], how='outer', suffixes=('_src','_reg'), validate='one_to_one', indicator=True)
    bad = check['_merge'].ne('both')
    for c in ['Reach','Event_Type','Start_Date','End_Date']: bad |= check[c+'_src'].ne(check[c+'_reg'])
    if bad.any():
        check.loc[bad].to_csv(OUT/f'{RUN_DATE}_M06事件匹配冲突.csv',index=False,encoding='utf-8-sig')
        raise ValueError(f'注册表与Shock结果存在{bad.sum()}条缺失/冲突；停止，不随意匹配')
    value = pd.to_numeric(src.Shock_Ratio_LongTerm, errors='coerce')
    if (src.Shock_Ratio_LongTerm.notna() & value.isna()).any(): raise ValueError('Shock-L含非数值')
    src['Shock_L'] = value
    d = reg.merge(src[['Site_ID','Event_ID','Shock_L']], on=['Site_ID','Event_ID'],validate='one_to_one')
    for t,n in EXPECTED.items(): log(f'{t}：实际{int(d.Event_Type.eq(t).sum())}，参考{n}，差值{int(d.Event_Type.eq(t).sum())-n}')
    log(f'S0注册表{len(d)}事件、{d.Site_ID.nunique()}站；关联字段与日期完全一致。')
    return d

def hydro_daily(path, name):
    d = read(path)
    require(d, ['HyS_name','Date','Q'], str(path))
    if set(d.HyS_name.dropna()) != {name}: raise ValueError(f'{path}站名与文件/映射不一致')
    date = pd.to_datetime(d.Date,errors='raise').dt.normalize()
    if date.duplicated().any(): raise ValueError(f'{path}逐日日期重复')
    q = pd.to_numeric(d.Q,errors='coerce')
    if (d.Q.notna() & q.isna()).any(): raise ValueError(f'{path}非数值Q未按既有清洗规则处理')
    if (q.lt(0) | (q.notna() & ~np.isfinite(q))).any():
        raise ValueError(f'{path}负值或无穷Q需人工核查；不自设阈值删除')
    return pd.Series(q.to_numpy(), index=date).reindex(CALENDAR)

def matching(events):
    d = read(MATCH_FILE)
    require(d,['WQS_name','Matched_HS_Name','Region','Distance_km','WQS_Lon','WQS_Lat','Matched_HS_Lon','Matched_HS_Lat','WQS_River','Matched_HS_River'],'映射表')
    if d.WQS_name.isna().any() or d.WQS_name.duplicated().any(): raise ValueError('映射表断面键重复或缺失')
    if d.Matched_HS_Name.isna().any() or d.Matched_HS_Name.isin(['None','']).any(): raise ValueError('映射缺少水文站；停止，不猜测')
    sites = events[['Site_ID','Reach']].drop_duplicates()
    if set(sites.Site_ID)-set(d.WQS_name): raise ValueError('S0断面缺少映射')
    extra=set(d.WQS_name)-set(sites.Site_ID)
    log(f'映射原表{len(d)}行；S0研究站外映射{sorted(extra)}。')
    d=d[d.WQS_name.isin(sites.Site_ID)].copy()
    d['Reach']=d.Region.map(REACH_MAP)
    check=sites.merge(d[['WQS_name','Reach']],left_on='Site_ID',right_on='WQS_name',validate='one_to_one',suffixes=('_event','_map'))
    if check.Reach_event.ne(check.Reach_map).any(): raise ValueError('映射和注册表河段冲突')
    d=d.rename(columns={'WQS_name':'WQS_ID','Matched_HS_Name':'Hydro_ID','WQS_River':'River',
                       'Distance_km':'Geodesic_Distance_km','Matched_HS_River':'Hydro_River'})
    d['WQS_Name']=d.WQS_ID; d['Hydro_Name']=d.Hydro_ID
    d['River_Distance_km']=np.nan
    d['Distance_Method']='已有大圆直线距离，球体R=6371km，源表舍入2位；不是沿河道距离'
    d['Identifier_Note']='沿用拼音站名作ID；未提供官方数字编号或中文名称'
    counts=d.groupby('Hydro_ID').WQS_ID.nunique()
    d['Matched_WQS_Count']=d.Hydro_ID.map(counts)
    d['Shared_Hydro']=d.Matched_WQS_Count.gt(1)
    rates=[]; cache={}
    for h in sorted(d.Hydro_ID.unique()):
        rawpath=HY_RAW_DIR/f'{h}.csv'; fillpath=HY_FILLED_DIR/f'{h}_Interpolated.csv'
        record={'Hydro_ID':h,'Q_Expected_Days':len(CALENDAR)}
        raw=hydro_daily(rawpath,h) if rawpath.exists() else None
        filled=hydro_daily(fillpath,h) if fillpath.exists() else None
        record['Q_Coverage_Raw']=100*raw.notna().sum()/len(CALENDAR) if raw is not None else np.nan
        record['Q_Raw_Valid_Days']=int(raw.notna().sum()) if raw is not None else np.nan
        record['Q_Interpolated_Rate']=np.nan
        record['Q_Interpolated_Days']=np.nan
        record['Q_Final_Coverage']=100*filled.notna().sum()/len(CALENDAR) if filled is not None else np.nan
        record['Q_Check_Status']='可核查' if raw is not None and filled is not None else '缺少原始或插值文件，无法完整核查'
        if raw is not None and filled is not None:
            original_mask=raw.notna()
            if not np.allclose(raw[original_mask],filled[original_mask],atol=Q_ATOL,rtol=0,equal_nan=True):
                raise ValueError(f'{h}插值文件改变已有原始Q；需人工核查')
            record['Q_Interpolated_Days']=int((raw.isna() & filled.notna()).sum())
            record['Q_Interpolated_Rate']=100*record['Q_Interpolated_Days']/len(CALENDAR)
        rates.append(record); cache[h]=filled
    d=d.merge(pd.DataFrame(rates),on='Hydro_ID',validate='many_to_one')
    # Verify the station mapping and discharge in the merged daily data using the existing match.
    for r in d.itertuples():
        path=MERGED_DIR/REACH_FOLDER[r.Reach]/'by_station'/f'{r.WQS_ID}.csv'
        if not path.exists(): raise ValueError(f'缺少最终日数据映射核对文件：{path}')
        daily=read(path)
        require(daily,['WQS_name','Matched_HS_Name','Date','Q'],str(path))
        if set(daily.WQS_name.dropna())!={r.WQS_ID} or set(daily.Matched_HS_Name.dropna())!={r.Hydro_ID}:
            raise ValueError(f'{r.WQS_ID}最终日记录包含不同映射，不能忽略')
        dates=pd.to_datetime(daily.Date,errors='raise').dt.normalize()
        if dates.duplicated().any(): raise ValueError(f'{r.WQS_ID}日数据重复')
        q=pd.Series(pd.to_numeric(daily.Q,errors='raise').to_numpy(),index=dates).reindex(CALENDAR)
        if cache[r.Hydro_ID] is not None and not np.allclose(q,cache[r.Hydro_ID],rtol=0,atol=Q_ATOL,equal_nan=True):
            raise ValueError(f'{r.WQS_ID}合并Q与所映射水文插值Q不符')
    log(f'实际{len(d)}水质断面对应{d.Hydro_ID.nunique()}水文站；共享水文站{int(counts.gt(1).sum())}个，使用共享站的断面{int(d.Shared_Hydro.sum())}个，最大匹配{int(counts.max())}个断面。')
    log('距离沿用源表。原始Q使用Original/Retained的插值前逐日序列；按原脚本数值化及日均值规则，Q=0保留，不新增离群阈值。')
    return d

def summarize_events(events,mapping):
    d=events.merge(mapping[['WQS_ID','Hydro_ID']],left_on='Site_ID',right_on='WQS_ID',validate='many_to_one')
    rows=[]
    for reach in ['All','Upstream','Midstream','Downstream']:
        subset=d if reach=='All' else d[d.Reach.eq(reach)]
        for typ in ['All']+TYPES:
            g=subset if typ=='All' else subset[subset.Event_Type.eq(typ)]
            n=g.Site_ID.nunique()
            rows.append({'Reach':reach,'Event_Type':typ,'Event_N':len(g),'Site_N':n,'Hydro_N':g.Hydro_ID.nunique(),
                         'Events_Per_Site':len(g)/n if n else np.nan})
    return pd.DataFrame(rows)

def site_means(events,mapping):
    valid=np.isfinite(events.Shock_L) & events.Shock_L.gt(0)
    log(f'Shock-L有效{int(valid.sum())}；缺失/非有限/非正{int((~valid).sum())}；不补零、不重算。')
    means=events.loc[valid].groupby(['Site_ID','Reach']).Shock_L.agg(Valid_Event_N='count',Mean_Shock_L='mean').reset_index()
    if set(mapping.WQS_ID)-set(means.Site_ID): log('注意：部分断面无有效Shock-L，不参与均值；见Site_Summary。')
    return means,events.loc[valid].copy()

def loso(valid,typ='All'):
    d=valid if typ=='All' else valid[valid.Event_Type.eq(typ)]
    means=d.groupby(['Site_ID','Reach']).Shock_L.agg(['count','mean']).reset_index()
    ref=means.groupby('Reach')['mean'].mean()
    down=means[means.Reach.eq('Downstream')].sort_values('Site_ID')
    if typ!='All' and len(down)<MIN_TYPE_SITES:
        log(f'{typ}下游仅{len(down)}站、{int(down["count"].sum())}有效事件，跳过类别LOSO。'); return pd.DataFrame()
    rows=[]
    for excluded in ['All Sites']+down.Site_ID.tolist():
        remain=down if excluded=='All Sites' else down[down.Site_ID.ne(excluded)]
        avg=remain['mean'].mean(); up=ref.get('Upstream',np.nan); mid=ref.get('Midstream',np.nan)
        rows.append({'Event_Type':typ,'Excluded_Site':excluded,'Retained_Site_N':len(remain),
                     'Plot_Label':'All Sites' if excluded=='All Sites' else 'Exclude '+chr(65+down.Site_ID.tolist().index(excluded)),
                     'Retained_Event_N':int(remain['count'].sum()),'Downstream_Mean_Shock_L':avg,
                     'Upstream_Mean_Shock_L':up,'Midstream_Mean_Shock_L':mid,
                     'Down_minus_Up':avg-up,'Down_minus_Mid':avg-mid,
                     'Down_Below_Up':bool(avg<up) if np.isfinite(avg+up) else None,
                     'Down_Below_Mid':bool(avg<mid) if np.isfinite(avg+mid) else None,
                     'Change_From_All_Sites':avg-down['mean'].mean(),
                     'Coverage_Note':'下游覆盖有限，仅描述性；不是区域独立事件或显著性检验' if len(down)<=3 else '描述性LOSO；不是显著性检验'})
    return pd.DataFrame(rows)

def excel(name,sheets):
    path=OUT/f'{RUN_DATE}_{name}.xlsx'
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        for title,d in sheets.items(): d.to_excel(writer,sheet_name=title,index=False)
    wb=load_workbook(path)
    for ws in wb:
        ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
        for c in ws[1]: c.font=Font(name='Arial',bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='284B63'); c.alignment=Alignment(wrap_text=True,vertical='center')
        ws.row_dimensions[1].height=44
        for col in ws.columns:
            width=min(48,max(16,max(sum(2 if ord(x)>127 else 1 for x in str(c.value or '')) for c in col)+2))
            ws.column_dimensions[get_column_letter(col[0].column)].width=width
            for c in col[1:]:
                c.font=Font(name='Arial',size=10); c.alignment=Alignment(vertical='top',wrap_text=True)
                if isinstance(c.value,float): c.number_format='0.0000'
        for i in range(2,ws.max_row+1):
            lines=max(int(np.ceil(sum(2 if ord(x)>127 else 1 for x in str(c.value or ''))/max(1,ws.column_dimensions[get_column_letter(c.column)].width-2))) for c in ws[i])
            ws.row_dimensions[i].height=max(24,15*max(1,lines)+6)
    wb.save(path)
    checked=load_workbook(path,data_only=True)
    for title,d in sheets.items():
        assert checked[title].max_row==len(d)+1 and checked[title].max_column==len(d.columns)
    log(f'导出并重读检查：{path.name}')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    events=load_events(); mapping=matching(events); counts=summarize_events(events,mapping)
    means,valid=site_means(events,mapping)
    summary=mapping.merge(means,left_on=['WQS_ID','Reach'],right_on=['Site_ID','Reach'],how='left',validate='one_to_one')
    per_site=pd.crosstab(events.Site_ID,events.Event_Type).reindex(columns=TYPES,fill_value=0).rename_axis('WQS_ID').reset_index()
    summary=summary.merge(per_site,on='WQS_ID',validate='one_to_one')
    overall=loso(valid); type_results=pd.concat([loso(valid,t) for t in TYPES],ignore_index=True)
    if overall.Retained_Site_N.iloc[0]!=5: log(f'下游实际{overall.Retained_Site_N.iloc[0]}站，与参考5不同；按实际执行。')
    hydro=mapping.drop_duplicates('Hydro_ID')
    down=means[means.Reach.eq('Downstream')]
    biggest=overall.iloc[1:].loc[overall.iloc[1:].Change_From_All_Sites.abs().idxmax()]
    log('下游断面均值及有效事件数：\n'+down.to_string(index=False))
    log(f'全部下游基准{overall.Downstream_Mean_Shock_L.iloc[0]:.6f}，上游{overall.Upstream_Mean_Shock_L.iloc[0]:.6f}，中游{overall.Midstream_Mean_Shock_L.iloc[0]:.6f}。')
    log(f'LOSO下游范围{overall.Downstream_Mean_Shock_L.iloc[1:].min():.6f}—{overall.Downstream_Mean_Shock_L.iloc[1:].max():.6f}；每次均低于上中游：{bool(overall.Down_Below_Up.all() and overall.Down_Below_Mid.all())}。')
    log(f'影响最大断面{biggest.Excluded_Site}，排除后变化{biggest.Change_From_All_Sites:+.6f}。')
    log(f'唯一水文站原始Q覆盖率范围{hydro.Q_Coverage_Raw.min():.2f}%—{hydro.Q_Coverage_Raw.max():.2f}%，低于90%共{int(hydro.Q_Coverage_Raw.lt(90).sum())}站，无法核查{int(hydro.Q_Coverage_Raw.isna().sum())}站。')
    log(f'唯一水文站插值比例范围{hydro.Q_Interpolated_Rate.min():.2f}%—{hydro.Q_Interpolated_Rate.max():.2f}%；无法完整核查{int(hydro.Q_Check_Status.ne("可核查").sum())}站。')
    log('各河段五类事件数/唯一断面数/唯一水文站数：\n'+counts[counts.Reach.ne('All')].to_string(index=False))
    for t,g in type_results.groupby('Event_Type',sort=False):
        log(f'{t}类别LOSO方向：下游始终低于上/中游={bool(g.Down_Below_Up.all())}/{bool(g.Down_Below_Mid.all())}，下游有效事件{g.Retained_Event_N.iloc[0]}、断面{g.Retained_Site_N.iloc[0]}；不能用总体LOSO概括每类。')
    log('共享水文站会重复使用Q信息；Site_N/Hydro_N是唯一标识数量，不是统计独立性证明。LOSO仅检验描述性方向；不重跑M01，不推断因果。')
    qc=pd.DataFrame(SOURCES)
    sheets={'Hydro_Matching':mapping,'Event_Site_Counts':counts,'Downstream_LOSO':overall,
            'Site_Summary':summary,'Type_LOSO':type_results,'Data_QC':qc,
            'Definitions':pd.DataFrame({'说明':['原始Q覆盖/插值比例的分母均为2021—2025理论1826日；百分数字段为0—100。','Geodesic_Distance_km为既有球体大圆直线距离；River_Distance_km缺失，不编造。','WQS_ID、Hydro_ID沿用现有拼音站名。','河段汇总先对断面内全部有效事件取均值，再对断面等权。','类别少于3下游断面不做LOSO；3站覆盖有限。','Event_N不是独立区域过程；站点随机截距及LOSO不能证明样本独立。']})}
    for title,d in sheets.items(): d.to_csv(OUT/f'{RUN_DATE}_M06_{title}.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    excel('M06水文站匹配表',{'Hydro_Matching':mapping})
    excel('M06事件与断面数量',{'Event_Site_Counts':counts,'Site_Summary':summary})
    excel('M06下游LOSO结果',{'Downstream_LOSO':overall,'Type_LOSO':type_results})
    excel('M06结果汇总',sheets)
    for source in SOURCES:
        assert hashlib.sha256((ROOT/source['File']).read_bytes()).hexdigest()==source['SHA256'], '源文件改变'
    log('所有读取源文件SHA256运行前后相同；4工作簿重读通过。')
    (OUT/f'{RUN_DATE}_M06结果说明.txt').write_text('\n'.join(LOG),encoding='utf-8')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        log('停止：'+repr(exc)+'；不随意忽略匹配错误，不修改原始文件。'); raise
    finally:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/f'{RUN_DATE}_M06数据核查日志.txt').write_text('\n'.join(LOG),encoding='utf-8')
