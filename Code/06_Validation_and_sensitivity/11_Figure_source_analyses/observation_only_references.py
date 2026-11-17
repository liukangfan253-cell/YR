from pathlib import Path
import pandas as pd, numpy as np, json, hashlib, sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/'Result/Observation_only_references/20261111'
DATE='20261111'; START='2021-01-01'; END='2025-12-31'; MIN_PRE=3
inputs={}
def read(p,**kw):
 p=ROOT/p;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p,**kw)
def season(d):return (d.month%12//3).map({0:'Winter',1:'Spring',2:'Summer',3:'Autumn'})
def valid(v):return np.isfinite(v)&(v>0)
master=read(Path('Result/Statistical_comparisons/Event_Level_Statistical_Data.csv')).rename(columns={'Shock-L':'Shock_L','Shock-S':'Shock_S'})
s0=read(Path('Result/Event/Definition_sensitivity/20261111/20261111_M05_S0事件级Shock结果.csv'))
non=read(Path('Data/Event_F/05_Non_Event_Data/Overall/05_Non_Event_Overall.csv'),usecols=['WQS_name','Date','DO'])
non.Date=pd.to_datetime(non.Date);assert not non.duplicated(['WQS_name','Date']).any()
assert master.Site_ID.nunique()==66 and not master.duplicated(['Site_ID','Event_ID']).any()
for col in ['Start_Date','End_Date']:master[col]=pd.to_datetime(master[col])
ref=master.merge(s0[['Site_ID','Event_ID','Event_DO','Seasonal_Baseline_DO','Pre_Baseline_DO','Pre_Baseline_Valid_Days']],on=['Site_ID','Event_ID'],validate='one_to_one')
assert len(ref) == len(s0) == len(master)
trace=[];daily=[];seasonrows=[]
for site,g in ref.groupby('Site_ID',sort=True):
 frames={}
 for key,p in [('raw',f'Data/WaterQuality_data_P/WQ_data_YR/Retained/Raw/{site}.csv'),('qc',f'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned/{site}.csv'),('final',f'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR/{site}_Imputed.csv')]:
  x=read(Path(p));x.Date=pd.to_datetime(x.Date);assert not x.Date.duplicated().any();frames[key]=x.set_index('Date')
 idx=pd.date_range(START,END);d=pd.DataFrame(index=idx);d.index.name='Date'
 for k in ['raw','qc','final']:d[k]=pd.to_numeric(frames[k].DO,errors='coerce').reindex(idx)
 orig=pd.to_numeric(frames['final'].Original_DO,errors='coerce').reindex(idx)
 assert d.qc.notna().equals(orig.notna()) and np.allclose(d.qc.dropna(),orig.dropna(),atol=2e-6,rtol=0)
 assert np.allclose(d.loc[d.qc.notna(),'final'],d.qc.dropna(),atol=2e-6,rtol=0)
 d['Season']=season(d.index);d['Event_ID']='';d['Event_Type']=''
 for e in g.itertuples():
  win=pd.date_range(e.Start_Date,e.End_Date);assert d.loc[win,'Event_ID'].eq('').all();d.loc[win,'Event_ID']=e.Event_ID;d.loc[win,'Event_Type']=e.Event_Type
 d['Is_Event']=d.Event_ID.ne('');nd=non[non.WQS_name==site].set_index('Date');assert set(nd.index)==set(d.index[~d.Is_Event])
 assert np.allclose(nd.DO.sort_index(),d.loc[nd.index,'final'].sort_index(),atol=2e-6,rtol=0,equal_nan=True)
 sn=d[~d.Is_Event].groupby('Season').agg(Observed_Mean=('qc','mean'),Completed_Mean=('final','mean'),Observed_N=('qc','count'),Completed_N=('final','count'))
 for k,r in sn.iterrows():seasonrows.append(dict(Site_ID=site,Season=k,**r.to_dict()))
 for e in g.itertuples():
  ev=d.loc[e.Start_Date:e.End_Date];pre=d.reindex(pd.date_range(e.Start_Date-pd.Timedelta(days=9),e.Start_Date-pd.Timedelta(days=4)))
  eligible=pre[pre.Is_Event.eq(False)];obs_n=ev.qc.count();obs_mean=ev.qc.mean();long_obs=ev.Season.map(sn.Observed_Mean).mean();long_final=ev.Season.map(sn.Completed_Mean).mean()
  pre_obs=eligible.qc.dropna();pre_final=eligible.final.dropna();short_obs=pre_obs.mean() if len(pre_obs)>=MIN_PRE else np.nan
  assert np.isclose(ev.final.mean(),e.Event_DO,atol=2e-6)
  assert np.isclose(long_final,e.Seasonal_Baseline_DO,atol=2e-6)
  check_s=ev.final.mean()/pre_final.mean() if len(pre_final)>=MIN_PRE and pre_final.mean()>0 else np.nan
  assert np.isclose(check_s,e.Shock_S,atol=2e-6,equal_nan=True)
  c_l=obs_mean/long_obs if long_obs>0 else np.nan;c_s=obs_mean/short_obs if short_obs>0 else np.nan
  selected=sn.loc[ev.Season.unique()]
  trace.append(dict(Site_ID=site,Event_ID=e.Event_ID,Event_Type=e.Event_Type,Reach=e.Reach,Start_Date=e.Start_Date,End_Date=e.End_Date,Event_Days=len(ev),Event_Observed_Days=obs_n,Event_Imputed_Days=int((ev.qc.isna()&ev.final.notna()).sum()),Event_Observed_Pct=100*obs_n/len(ev),Event_DO_Observed=obs_mean,Event_DO_Completed=e.Event_DO,Seasonal_Baseline_Observed=long_obs,Seasonal_Baseline_Completed=e.Seasonal_Baseline_DO,Seasonal_Observed_Days=int(selected.Observed_N.sum()),Seasonal_Completed_Days=int(selected.Completed_N.sum()),Seasonal_Observed_Pct=100*selected.Observed_N.sum()/selected.Completed_N.sum(),Missing_Season_Background_Days=int(ev.Season.map(sn.Observed_Mean).isna().sum()),Pre_Calendar_Days=6,Pre_NonEvent_Days=int(pre.Is_Event.eq(False).sum()),Pre_Outside_Study_Days=int(pre.Is_Event.isna().sum()),Pre_Excluded_Event_Days=int(pre.Is_Event.eq(True).sum()),Pre_Observed_Days=len(pre_obs),Pre_Completed_Days=len(pre_final),Pre_Baseline_Observed=short_obs,Pre_Baseline_Completed=e.Pre_Baseline_DO,A_Shock_L=e.Shock_L,A_Shock_S=e.Shock_S,B_Selected=obs_n==len(ev),C_Shock_L=c_l,C_Shock_S=c_s,C_L_Status='No_event_observations' if obs_n==0 else 'Invalid_seasonal_baseline' if not np.isfinite(long_obs) or long_obs<=0 else 'Valid',C_S_Status='No_event_observations' if obs_n==0 else 'Fewer_than_3_observed_antecedent_days' if len(pre_obs)<MIN_PRE else 'Invalid_antecedent_baseline' if not np.isfinite(short_obs) or short_obs<=0 else 'Valid'))
 d['Site_ID']=site;d['Reach']=g.Reach.iloc[0];d['DO_Source']=np.where(d.qc.notna(),'QC_observed',np.where(d.final.notna(),'Imputed','Missing'));daily.append(d.reset_index().rename(columns={'raw':'DO_Raw','qc':'DO_QC','final':'DO_Final'}))
 print(f'读取并核对 {site}：事件{len(g)}，QC观测{d.qc.count()}/1826')
OUT.mkdir(parents=True, exist_ok=True)
t=pd.DataFrame(trace);t.to_csv(OUT/f'{DATE}_实测来源核对.csv',index=False,encoding='utf-8-sig')
pd.concat(daily).to_csv(OUT/f'{DATE}_逐日实测来源.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(seasonrows).to_csv(OUT/f'{DATE}_季节基线来源.csv',index=False,encoding='utf-8-sig')
rows=[];common=t.B_Selected&valid(t.C_Shock_L)&valid(t.C_Shock_S)&valid(t.A_Shock_L)&valid(t.A_Shock_S)
for sample in ['Available','Common_ABC']:
 for group in ['A','B','C']:
  x=t.copy() if sample=='Available' else t[common].copy()
  if group=='B':x=x[x.B_Selected].copy()
  for met in ['Shock_L','Shock_S']:x[met]=x[('C_' if group=='C' else 'A_')+met]
  parts=[('Overall','All',x)]+[('Reach',str(k),z) for k,z in x.groupby('Reach')]+[('Event_Type',str(k),z) for k,z in x.groupby('Event_Type')]+[('Reach_Event_Type',str(k),z) for k,z in x.groupby(['Reach','Event_Type'])]
  for level,entity,z in parts:
   p=z[valid(z.Shock_L)&valid(z.Shock_S)];neutral=(p.Shock_L.eq(1)|p.Shock_S.eq(1));opp=(p.Shock_L-1)*(p.Shock_S-1)<0;band=(abs(p.Shock_L-1)>.05)&(abs(p.Shock_S-1)>.05);opp5=opp&band
   r=dict(Sample=sample,Group=group,Level=level,Entity=entity,Registry_Events=len(z),Sites=z.Site_ID.nunique(),Paired_N=len(p),Paired_Sites=p.Site_ID.nunique(),Neutral_N=int(neutral.sum()),Discordant_N=int(opp.sum()),Discordance_Pct=100*opp.sum()/(len(p)-neutral.sum()) if len(p)-neutral.sum() else np.nan,Discordance_AllPairs_Pct=100*opp.mean() if len(p) else np.nan,Opposite_Beyond5_N=int(opp5.sum()),Opposite_Beyond5_AllPairs_Pct=100*opp5.mean() if len(p) else np.nan,Beyond5_Classifiable_N=int(band.sum()),Opposite_Beyond5_Classifiable_Pct=100*opp5.sum()/band.sum() if band.sum() else np.nan)
   for met in ['Shock_L','Shock_S']:
    u=z[np.isfinite(z[met])];v=u[met];r.update({met+'_N':len(v),met+'_Sites':u.Site_ID.nunique(),met+'_Mean':v.mean(),met+'_Median':v.median(),met+'_Q25':v.quantile(.25),met+'_Q75':v.quantile(.75),met+'_SiteEqual_Mean':u.groupby('Site_ID')[met].mean().mean()})
   rows.append(r)
res=pd.DataFrame(rows);res.to_csv(OUT/f'{DATE}_纯实测双基线敏感性.csv',index=False,encoding='utf-8-sig')
checks={'Rows':len(t),'Sites':t.Site_ID.nunique(),'Common_ABC_N':int(common.sum()),'C_L_Status':t.C_L_Status.value_counts().to_dict(),'C_S_Status':t.C_S_Status.value_counts().to_dict(),'Missing_Season_Background_Days':int(t.Missing_Season_Background_Days.sum()),'QC_Observed_Daily':int(pd.concat(daily).qc.notna().sum()) if False else int(sum(x.DO_QC.notna().sum() for x in daily)),'Event_Observed_Pct':100*t.Event_Observed_Days.sum()/t.Event_Days.sum(),'Pre_Observed_Pct':100*t.Pre_Observed_Days.sum()/t.Pre_Completed_Days.sum(),'Event_No_Obs':int(t.Event_Observed_Days.eq(0).sum()),'Seasonal_Unique_Observed_Pct':100*pd.DataFrame(seasonrows).Observed_N.sum()/pd.DataFrame(seasonrows).Completed_N.sum()}
print(res.query("Level=='Overall'").to_string(index=False));print(json.dumps(checks,ensure_ascii=False,indent=2))
