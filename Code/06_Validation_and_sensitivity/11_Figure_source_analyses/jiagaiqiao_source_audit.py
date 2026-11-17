# -*- coding: utf-8 -*-
"""Comment4 Jiagaiqiao low-DO provenance and environmental-consistency audit.
Read-only with respect to research inputs. No automatic record deletion, imputation,
hydrological rematching, model refitting or instrument-validity certification.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse,ast,hashlib,json,re,sys,time,traceback
import numpy as np
import pandas as pd
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Code/05_Statistical_analysis/00_Shared_utilities'))
from english_table_output import english_table
SITE='Jiagaiqiao';CHINESE_SITE='贾垓桥'
CAL=pd.date_range('2021-01-01','2025-12-31')
PARAMETERS={'WT':'水温','pH':'pH','EC':'电导率','Turb':'浊度','CODMn':'高锰酸盐指数','NH3_N':'氨氮','TP':'总磷','TN':'总氮'}
DAILY_MAP={'WT':'WT','pH':'pH','EC':'EC','Turb':'Turb','CODMn':'CODₘₙ','NH3_N':'NH₃-N','TP':'TP','TN':'TN'}


def require(test,message):
 if not test:raise ValueError(message)


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()


def read(path):
 for encoding in ['utf-8-sig','gb18030']:
  try:return pd.read_csv(path,encoding=encoding,low_memory=False)
  except UnicodeDecodeError:pass
 raise ValueError(f'Cannot decode {path}')


def csv(out,name,data):
 frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
 english_table(frame).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig',float_format='%.12g')
 return frame


def js(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def daily_file(path):
 d=read(path);d.Date=pd.to_datetime(d.Date);require(not d.Date.duplicated().any(),f'Duplicate dates: {path}')
 require(set(d.Date)==set(CAL),f'Daily calendar mismatch: {path}')
 return d.set_index('Date').reindex(CAL)


def runs(mask):
 ix=np.flatnonzero(np.asarray(mask,dtype=bool))
 return [] if not len(ix) else np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)


def summarize(values):
 x=pd.to_numeric(values,errors='raise');x=x[np.isfinite(x)]
 return dict(N=len(x),Median=x.median(),Q25=x.quantile(.25),Q75=x.quantile(.75),Mean=x.mean(),Min=x.min(),Max=x.max())


def raw_runs(raw,args):
 """Consecutive sampled low records/identical readings, not continuous exposure."""
 rows=[]
 for kind in ['Low_DO_below2','Identical_DO']:
  groups=[];current=[]
  for i in range(len(raw)):
   value=raw.QC_DO.iloc[i];valid=np.isfinite(value) and (value<2 if kind=='Low_DO_below2' else True)
   gap=(raw.Timestamp.iloc[i]-raw.Timestamp.iloc[i-1]).total_seconds()/3600 if i else np.inf
   same=(value==raw.QC_DO.iloc[i-1]) if i else False
   join=bool(current) and gap<=args.max_sample_gap_hours and gap>0 and (kind!='Identical_DO' or same)
   if not valid or not join:
    if current:groups.append(current)
    current=[]
   if valid:current.append(i)
  if current:groups.append(current)
  for ids in groups:
   if kind=='Identical_DO' and len(ids)<args.flatline_min_records:continue
   g=raw.iloc[ids];record=dict(Kind=kind,Start=g.Timestamp.iloc[0],End=g.Timestamp.iloc[-1],Records=len(g),Elapsed_Hours=(g.Timestamp.iloc[-1]-g.Timestamp.iloc[0]).total_seconds()/3600,DO_Min=g.QC_DO.min(),DO_Max=g.QC_DO.max(),Unique_DO=g.QC_DO.nunique(),First_Source_Row=int(g.Source_Row.iloc[0]),Last_Source_Row=int(g.Source_Row.iloc[-1]),All_Values_Below2=bool(g.QC_DO.lt(2).all()))
   for v in ['WT','pH','EC','CODMn']:
    x=g['QC_'+v].dropna();record[v+'_N']=len(x);record[v+'_Range']=x.max()-x.min() if len(x) else np.nan
   rows.append(record)
 return pd.DataFrame(rows)


def execute(args,out):
 started=time.perf_counter()
 candidates=list((ROOT/'Data/WaterQuality_data/WQ_data_YR').glob('*贾*'))
 require(len(candidates)==1,'Need a unique original Jiagaiqiao timestamp CSV')
 inputs={
  'timestamp_source':candidates[0],
  'daily_raw':ROOT/'Data/WaterQuality_data_P/WQ_data_YR/Retained/Raw/Jiagaiqiao.csv',
  'daily_clean':ROOT/'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned/Jiagaiqiao.csv',
  'completed':ROOT/'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/LR/Jiagaiqiao_Imputed.csv',
  'merged_environment':ROOT/'Data/WQ_Hy_M_DYN_data/YR/by_station_all/Jiagaiqiao.csv',
  'era5':ROOT/'Data/Meteorology_data/2021-2025/YR_V1/Jiagaiqiao.csv',
  'qc_script':ROOT/'Code/01_Data_integration/02_Water_quality_QC/1.WQ_Data_P.py',
  'registry':ROOT/'Data/Event_F/Event_Registry_For_Validation.csv',
  'event_DO':ROOT/'Result/Decrease/DO_observation_sensitivity/20261111/20261111_M07事件绝对DO指标.csv',
  'basin_daily':ROOT/'Result/Observation_only_references/20261111/20261111_逐日实测来源.csv'}
 if args.quality_records:
  require(args.quality_records.is_file(),'Specified quality-record file is absent');inputs['user_quality_records']=args.quality_records.resolve()
 hashes={k:sha(p) for k,p in inputs.items()}
 original=read(inputs['timestamp_source']);original['Source_Row']=np.arange(len(original))+2
 original['Timestamp']=pd.to_datetime(original['监测时间'],errors='coerce');original['In_Study']=original.Timestamp.dt.year.between(2021,2025).fillna(False)
 csv(out,'source_rows_outside_study_or_invalid_time',original[~original.In_Study])
 raw=original[original.In_Study].sort_values(['Timestamp','Source_Row']).copy().reset_index(drop=True)
 require(set(raw['站点名称'].dropna())=={CHINESE_SITE},'Raw station identity differs')
 raw['Date']=raw.Timestamp.dt.normalize();raw['DO']=pd.to_numeric(raw['溶解氧'],errors='coerce')
 for v,col in PARAMETERS.items():raw[v]=pd.to_numeric(raw[col],errors='coerce')
 qc_tree=ast.parse(inputs['qc_script'].read_text(encoding='utf-8-sig'))
 calc=[n for n in qc_tree.body if isinstance(n,ast.FunctionDef) and n.name=='calc_do_sat'];require(len(calc)==1,'QC saturation function absent')
 ns={'np':np,'pd':pd};exec(compile(ast.Module(body=calc,type_ignores=[]),'<saved QC saturation>','exec'),ns)
 bounds=None
 for node in qc_tree.body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='quality_bounds' for t in node.targets):bounds=ast.literal_eval(node.value)
 require(bounds is not None,'QC bounds absent')
 era=daily_file(inputs['era5']);raw['SP_ERA5']=raw.Date.map(era.SP)
 raw['DO_sat_Recreated']=np.round(ns['calc_do_sat'](raw.WT.to_numpy(),raw.SP_ERA5.to_numpy()),3)
 raw['DO_1_5_sat_Recreated']=np.round(raw.DO_sat_Recreated*1.5,3)
 invalid=raw.DO.lt(0)|raw.DO.gt(20)|raw.DO.gt(raw.DO_1_5_sat_Recreated)
 raw['DO_Original_QC_OutOfRange']=invalid;raw['QC_DO']=raw.DO.mask(invalid)
 for v,chinese in PARAMETERS.items():
  lo,hi=bounds[chinese];raw['QC_'+v]=raw[v].mask(raw[v].lt(lo)|raw[v].gt(hi))
 raw['Duplicate_Timestamp']=raw.Timestamp.duplicated(False)
 raw['Near_Zero_AtMost01']=raw.QC_DO.le(.1);raw['Source_Situation']=raw['站点'] if '站点' in raw else np.nan
 # Water-quality class is recorded for traceability only. It may depend on DO and
 # is not independent calibration evidence.
 raw['Source_WaterQuality_Class']=raw['水质'] if '水质' in raw else np.nan
 daily_raw=daily_file(inputs['daily_raw']);clean=daily_file(inputs['daily_clean']);final=daily_file(inputs['completed']);merged=daily_file(inputs['merged_environment'])
 checks=[]
 for kind,table,prefix in [('Raw',daily_raw,''),('Cleaned',clean,'QC_')]:
  pairs={'DO':'DO',**DAILY_MAP}
  for variable,daily_col in pairs.items():
   rebuilt=raw.groupby('Date')[prefix+variable].mean().round(3).reindex(CAL);actual=table[daily_col]
   ok=np.isclose(rebuilt,actual,atol=2e-6,rtol=0,equal_nan=True)
   checks.append(dict(Table=kind,Variable=variable,Dates=len(CAL),Matching_Dates=int(ok.sum()),All_Match=bool(ok.all())))
   require(ok.all(),f'Timestamp-to-daily {kind}/{variable} mismatch')
 require(np.isclose(final.Original_DO,clean.DO,atol=2e-6,rtol=0,equal_nan=True).all(),'Final original-DO mask/value differs')
 require(np.isclose(final.DO[clean.DO.notna()],clean.DO.dropna(),atol=2e-6,rtol=0).all(),'Final series changed original observations')
 csv(out,'provenance_checks',checks)
 daily=pd.DataFrame({'Date':CAL,'DO_Raw':daily_raw.DO.to_numpy(),'DO_QC':clean.DO.to_numpy(),'DO_Final':final.DO.to_numpy()})
 daily['DO_Source']=np.where(daily.DO_QC.notna(),'Observed',np.where(daily.DO_Final.notna(),'Imputed','Missing'))
 for v,col in DAILY_MAP.items():daily[v]=clean[col].to_numpy()
 for v in ['Q','WL','Ss']:daily['Assigned_'+v]=merged[v].to_numpy()
 daily['SP_ERA5']=era.SP.to_numpy();daily['DO_sat_Approx']=clean.DO_sat.to_numpy()
 daily['Observed_Saturation_Pct']=100*daily.DO_QC/daily.DO_sat_Approx
 counts=raw.groupby('Date').agg(Source_Rows=('Source_Row','size'),Valid_Source_DO=('QC_DO','count'),Source_DO_Min=('QC_DO','min'),Source_DO_Max=('QC_DO','max'),Source_DO_Median=('QC_DO','median'),Source_Below2=('QC_DO',lambda x:int(x.lt(2).sum())),Source_NearZero=('QC_DO',lambda x:int(x.le(.1).sum())))
 for col in counts:daily[col]=daily.Date.map(counts[col])
 daily['Year']=daily.Date.dt.year;daily['Month']=daily.Date.dt.month;daily['YearMonth']=daily.Date.dt.strftime('%Y-%m')
 registry=read(inputs['registry']);registry.Start_Date=pd.to_datetime(registry.Start_Date);registry.End_Date=pd.to_datetime(registry.End_Date)
 daily['Event_ID']='';daily['Event_Type']='NonEvent'
 for event in registry[registry.Site_ID.eq(SITE)].itertuples():
  use=daily.Date.between(event.Start_Date,event.End_Date);require(daily.loc[use,'Event_ID'].eq('').all(),'Overlapping event windows')
  daily.loc[use,'Event_ID']=event.Event_ID;daily.loc[use,'Event_Type']=event.Event_Type
 daily['Is_Event']=daily.Event_ID.ne('');daily['Observed_Below2']=daily.DO_QC.lt(2);daily['Completed_Below2']=daily.DO_Final.lt(2)
 daily['Adjacent_Observed_Jump']=daily.DO_QC.diff().abs();daily['Jump_Diagnostic']=daily.Adjacent_Observed_Jump.ge(args.jump_threshold)
 daily['Observed_Category']=np.where(daily.DO_QC.isna(),'No_DO_observation',np.where(daily.DO_QC.lt(2),'DO_lt2',np.where(daily.DO_QC.lt(5),'DO_2_to_lt5','DO_ge5')))
 daily['Validity_Status']=np.where(daily.DO_Source.eq('Imputed'),'Model_estimate_not_observation',np.where(daily.Observed_Below2,'Source_traced_low_observation_instrument_check_pending','Source_traced_observation'))
 csv(out,'daily_site_audit',daily);csv(out,'timestamp_source_audit',raw)
 event_low=daily[daily.Is_Event&daily.Completed_Below2]
 basin=read(inputs['basin_daily']);basin['Date']=pd.to_datetime(basin.Date)
 basin_event=basin.Event_ID.notna()&basin.Event_ID.astype(str).ne('');basin_low=basin[basin_event&basin.DO_Final.lt(2)].copy()
 mine=basin_low[basin_low.Site_ID.eq(SITE)].sort_values('Date');our=event_low.sort_values('Date')
 require(len(mine)==len(our) and np.array_equal(mine.Date.to_numpy(),our.Date.to_numpy()) and np.allclose(mine.DO_Final,our.DO_Final,atol=2e-6,rtol=0),'Basin/site event records differ')
 csv(out,'all_observed_low_days',daily[daily.Observed_Below2]);csv(out,'event_observed_low_days',daily[daily.Is_Event&daily.Observed_Below2])
 source_summary=[]
 for group,g in [('Jiagaiqiao',basin_low[basin_low.Site_ID.eq(SITE)]),('Other_sites',basin_low[~basin_low.Site_ID.eq(SITE)])]:
  for source,use in [('Observed',g.DO_QC.notna()),('Imputed',g.DO_QC.isna()&g.DO_Final.notna())]:source_summary.append(dict(Group=group,Source=source,Event_Days=int(use.sum())))
 csv(out,'basin_low_DO_source_counts',source_summary)
 strata=[]
 for name,selection in [('All_dates',pd.Series(True,index=daily.index)),('Event_dates',daily.Is_Event),('NonEvent_dates',~daily.Is_Event)]:
  for threshold in [5,3,2]:
   g=daily[selection]
   for source,col in [('Observed','DO_QC'),('Completed','DO_Final')]:
    use=g[col].notna();low=g[col].lt(threshold)
    strata.append(dict(Context=name,DO_Source=source,Threshold=threshold,Valid_Dates=int(use.sum()),Low_Dates=int(low.sum()),Low_Pct=100*low.sum()/use.sum() if use.sum() else np.nan))
 csv(out,'low_DO_day_summary',strata)
 observed_runs=[]
 for kind,mask in [('Observed_low_days',daily.Observed_Below2),('Completed_low_days',daily.Completed_Below2)]:
  for number,ix in enumerate(runs(mask),1):
   g=daily.iloc[ix];observed_runs.append(dict(Kind=kind,Run_ID=f'{kind}_{number:03d}',Start=g.Date.iloc[0],End=g.Date.iloc[-1],Calendar_Days=len(g),Observed_Days=int(g.DO_Source.eq('Observed').sum()),Imputed_Days=int(g.DO_Source.eq('Imputed').sum()),Event_Days=int(g.Is_Event.sum()),Minimum_Observed=g.DO_QC.min(),Minimum_Completed=g.DO_Final.min()))
 csv(out,'daily_low_runs',observed_runs);rr=raw_runs(raw,args);csv(out,'timestamp_low_and_flatline_runs',rr)
 # Review blocks merge overlapping +/-context-day windows around ALL observed low dates.
 blocks=[]
 for date in daily.loc[daily.Observed_Below2,'Date']:
  left=max(CAL[0],date-pd.Timedelta(days=args.context_days));right=min(CAL[-1],date+pd.Timedelta(days=args.context_days))
  if blocks and left<=blocks[-1]['End']+pd.Timedelta(days=1):blocks[-1]['End']=max(blocks[-1]['End'],right)
  else:blocks.append(dict(Block_ID=f'Block_{len(blocks)+1:02d}',Start=left,End=right))
 for block in blocks:
  g=daily[daily.Date.between(block['Start'],block['End'])]
  block.update(Observed_Low_Days=int(g.Observed_Below2.sum()),Event_Observed_Low_Days=int((g.Observed_Below2&g.Is_Event).sum()),Calendar_Days=len(g))
 require(sum(b['Observed_Low_Days'] for b in blocks)==int(daily.Observed_Below2.sum()),'Review blocks do not cover all observed low dates exactly once')
 csv(out,'review_blocks',blocks)
 # Descriptive associations only. Original/QC observations never filled for this step.
 distribution=[];correlations=[];matched=[]
 for context,selection in [('All_observed',daily.DO_QC.notna()),('Event_observed',daily.DO_QC.notna()&daily.Is_Event)]:
  g=daily[selection]
  for category in ['DO_lt2','DO_2_to_lt5','DO_ge5']:
   z=g[g.Observed_Category.eq(category)]
   for variable in PARAMETERS:distribution.append(dict(Context=context,Category=category,Variable=variable,**summarize(z[variable])))
  for variable in PARAMETERS:
   pairs=g[['DO_QC',variable]].dropna();rho=pairs.corr(method='spearman').iloc[0,1] if len(pairs)>2 and pairs.nunique().min()>1 else np.nan
   correlations.append(dict(Context=context,Variable=variable,Pairs=len(pairs),Spearman_Rho=rho,Interpretation='Descriptive original-observation association; no independence-adjusted inference'))
 for row in daily[daily.Observed_Below2].itertuples():
  background=daily[daily.YearMonth.eq(row.YearMonth)&daily.DO_QC.ge(5)]
  for variable in PARAMETERS:
   values=background[variable].dropna();n=len(values);value=getattr(row,variable)
   good=n>=args.min_reference_days and np.isfinite(value)
   matched.append(dict(Date=row.Date,Is_Event=row.Is_Event,Event_ID=row.Event_ID,YearMonth=row.YearMonth,DO_QC=row.DO_QC,Variable=variable,Observed_Value=value,Reference_Days=n,Reference_Median=values.median() if n else np.nan,Reference_Q25=values.quantile(.25) if n else np.nan,Reference_Q75=values.quantile(.75) if n else np.nan,Difference_To_Reference_Median=value-values.median() if good else np.nan,Status='Available' if good else 'Insufficient_same_year_month_reference_or_missing_variable'))
 csv(out,'observed_chemistry_by_DO_group',distribution);csv(out,'observed_DO_associations',correlations);csv(out,'same_year_month_chemistry_reference',matched)
 monthly=[]
 for ym,g in daily.groupby('YearMonth'):
  monthly.append(dict(YearMonth=ym,Calendar_Days=len(g),Observed_DO_Days=int(g.DO_QC.notna().sum()),Observed_Low_Days=int(g.Observed_Below2.sum()),Observed_Median_DO=g.DO_QC.median(),WT_Median=g.WT.median(),CODMn_Median=g.CODMn.median()))
 csv(out,'monthly_observation_summary',monthly)
 event_data=read(inputs['event_DO']);sensitivity=[]
 for name,excluded in [('All_sites',[]),('Without_Jiagaiqiao',[SITE]),('Without_two_influential_sites',[SITE,'Puyangdahanqiao'])]:
  g=event_data[~event_data.Site_ID.isin(excluded)];p=g[np.isfinite(g.Shock_L)&g.Shock_L.gt(0)&np.isfinite(g.Shock_S)&g.Shock_S.gt(0)];neutral=p.Shock_L.eq(1)|p.Shock_S.eq(1);flip=(p.Shock_L-1)*(p.Shock_S-1)<0;down=g[g.Reach.eq('Downstream')]
  rec=dict(Scheme=name,Events=len(g),Sites=int(g.Site_ID.nunique()),Paired_Events=len(p),Discordant=int(flip.sum()),Discordance_Pct=100*flip.sum()/(len(p)-neutral.sum()),Downstream_Sites=int(down.Site_ID.nunique()),Downstream_Site_Equal_ShockL=down.groupby('Site_ID').Shock_L.mean().mean())
  for threshold in [5,3,2]:rec[f'Below{threshold}_Events']=int(g[f'Any_DO_Below{threshold}'].sum());rec[f'Below{threshold}_Event_Pct']=100*g[f'Any_DO_Below{threshold}'].mean()
  sensitivity.append(rec)
 csv(out,'existing_site_exclusion_sensitivity',sensitivity)
 names=[]
 for p in (ROOT/'Data').rglob('*'):
  if p.is_file() and re.search(r'校准|运维|维护|仪器|质控日志|calibr|maintenance|instrument',p.name,re.I):names.append(str(p.relative_to(ROOT)))
 js(out/'instrument_evidence_inventory.json',dict(Search_Scope='Project Data filenames only; not an exhaustive external-record search',Matching_Files=names,Source_Station_Situation_Nonmissing=int(raw.Source_Situation.notna().sum()),User_Quality_Record=str(args.quality_records) if args.quality_records else None,Conclusion='Instrument validity requires manual assessment; filename search and provenance do not certify calibration.'))
 quality=[dict(Evidence='Timestamp_to_daily_provenance',Status='Verified_for_DO_and_8_original_water_quality_variables',Detail='All 1826 dates match original aggregation and QC rules'),dict(Evidence='Instrument_calibration_and_maintenance',Status='Not_established_from_available_project_evidence',Detail='No populated station-situation field; project filename search is bounded, no automatic instrument validity claim'),dict(Evidence='Environmental_covariation',Status='Descriptive_diagnostics_available',Detail='Original observations only; associations do not establish causal processes or sensor accuracy'),dict(Evidence='Assigned_hydrological_records',Status='Context_only_Comment3_deferred',Detail='No rematching or representativeness validation performed'),dict(Evidence='Record_deletion',Status='None',Detail='Flags are review prompts, not automatic exclusion rules')]
 csv(out,'evidence_status',quality)
 flags=rr[rr.Kind.eq('Identical_DO')&rr.All_Values_Below2] if len(rr) else rr
 summary=dict(Site=SITE,Raw_Total_Rows=len(original),Raw_Study_Rows=len(raw),Excluded_Invalid_or_OutOfStudy=len(original)-len(raw),Duplicate_Timestamp_Rows=int(raw.Duplicate_Timestamp.sum()),Original_QC_DO_OutOfRange_Rows=int(invalid.sum()),Raw_QC_DO_NearZero_AtMost01_Rows=int(raw.Near_Zero_AtMost01.sum()),Observed_Daily_DO=int(daily.DO_QC.notna().sum()),Observed_Low_Days_All=int(daily.Observed_Below2.sum()),Event_Low_Days_Completed=len(event_low),Event_Low_Days_Observed=int(event_low.DO_Source.eq('Observed').sum()),Event_Low_Days_Imputed=int(event_low.DO_Source.eq('Imputed').sum()),Basin_Event_Low_Days=len(basin_low),Site_Share_Of_Basin_Event_Low_Pct=100*len(event_low)/len(basin_low),Observed_Low_Run_Count=sum(r['Kind']=='Observed_low_days' for r in observed_runs),Longest_Observed_Low_Run_Days=max([r['Calendar_Days'] for r in observed_runs if r['Kind']=='Observed_low_days'],default=0),Identical_Low_Record_Run_Count=len(flags),Longest_Identical_Low_Record_Run=int(flags.Records.max()) if len(flags) else 0,Adjacent_Observed_Jump_Days=int(daily.Jump_Diagnostic.sum()),Review_Blocks=len(blocks),Station_Situation_Nonmissing=int(raw.Source_Situation.notna().sum()),Possible_Instrument_Record_Filenames=len(names),Instrument_Validity='UNRESOLVED',Sources_Untouched=True)
 protocol=dict(Script=str(Path(__file__).resolve()),Script_SHA256=sha(__file__),Sources={k:dict(Path=str(p),SHA256=hashes[k]) for k,p in inputs.items()},Parameters=vars(args)|{'output_dir':str(args.output_dir) if args.output_dir else None,'quality_records':str(args.quality_records) if args.quality_records else None},Scope='Station-level diagnostic audit only. Comment3 hydrological equivalence remains deferred.',Exclusions='Source rows outside 2021-2025/invalid timestamps are counted and exported. No valid study observation is silently deleted. QC only recreates existing rules for verification.')
 # Python 3.9 dictionary union is used; Path arguments are converted above.
 js(out/'run_provenance.json',protocol)
 require(all(sha(p)==hashes[k] for k,p in inputs.items()),'Source files changed during audit')
 js(out/'run_status.json',dict(Status='COMPLETED',Seconds=time.perf_counter()-started,Instrument_Validity='UNRESOLVED',Hydrological_Matching_Changed=False,Research_Data_Modified=False))
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True);print(f'COMPLETED: {out}',flush=True)


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path);ap.add_argument('--context-days',type=int,default=7);ap.add_argument('--max-sample-gap-hours',type=float,default=12);ap.add_argument('--flatline-min-records',type=int,default=3);ap.add_argument('--jump-threshold',type=float,default=3);ap.add_argument('--min-reference-days',type=int,default=3);ap.add_argument('--quality-records',type=Path)
 args=ap.parse_args();require(args.context_days>=0 and args.max_sample_gap_hours>0 and args.flatline_min_records>=2 and args.jump_threshold>0 and args.min_reference_days>=1,'Invalid diagnostic settings')
 out=args.output_dir.resolve() if args.output_dir else ROOT/'Result/Comment4'/('Comment4_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_Audit')
 require(not out.exists() or not any(out.iterdir()),f'Nonempty output directory: {out}');out.mkdir(parents=True,exist_ok=True)
 js(out/'run_status.json',dict(Status='RUNNING'))
 try:execute(args,out)
 except Exception as exc:
  js(out/'run_status.json',dict(Status='FAILED',Error_Type=type(exc).__name__,Error=str(exc)));(out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
