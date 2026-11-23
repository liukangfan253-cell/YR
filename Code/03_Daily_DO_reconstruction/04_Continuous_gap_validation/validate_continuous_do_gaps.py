# -*- coding: utf-8 -*-
"""Continuous DO gap validation of the saved V1 model.

Default: all recorded sites, observed 2025 targets, lengths 3/7/14/30 days.
Run in the existing PyTorch environment:
    python Code/03_Daily_DO_reconstruction/04_Continuous_gap_validation/validate_continuous_do_gaps.py
--plan-only inventories candidates without loading the model or running inference.
A --sites subset or --max-windows cap is always marked as a diagnostic subset.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = ROOT/'Result/BiLstm_Att_V1/Saved_Models_V1/DO_Model_Manifest_V1.json'
REGISTRY = ROOT/'Data/Event_F/Event_Registry_For_Validation.csv'
DESIGNS = ['Length_specific', 'Common_anchor']
METHODS = ['Masked_BiLSTM_Attention', 'Unmasked_BiLSTM_Attention', 'Last_observation']
POINT_COLUMNS = ['Design','Window_ID','Site_ID','Reach','Length','Depth','Date','Event_ID','Event_Type','Context','Observed_DO','Masked_BiLSTM_Attention','Unmasked_BiLSTM_Attention','Last_observation']
WINDOW_COLUMNS = ['Design','Window_ID','Site_ID','Reach','Length','Start','End','Event_Days','Observed_Low5_Days','Last_Observation_Date','Baseline_Available','Observed_Mean','Observed_Min','Masked_Mean','Masked_Min','Mean_Bias','Minimum_Bias','MAE','RMSE','Bias']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def locate(path):
    p=Path(path)
    return p if p.is_absolute() else ROOT/p


def require(condition, message):
    if not condition:raise ValueError(message)


def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    keep=np.isfinite(y)&np.isfinite(p);y=y[keep];p=p[keep]
    if not len(y):return dict(N=0,MAE=np.nan,RMSE=np.nan,Bias=np.nan,NSE=np.nan)
    error=p-y;den=((y-y.mean())**2).sum()
    return dict(N=int(len(y)),MAE=float(abs(error).mean()),RMSE=float(np.sqrt((error**2).mean())),
                Bias=float(error.mean()),NSE=float(1-(error**2).sum()/den) if len(y)>1 and den>0 else np.nan)


def contiguous_runs(mask):
    idx=np.flatnonzero(np.asarray(mask,dtype=bool))
    return [] if not len(idx) else np.split(idx,np.flatnonzero(np.diff(idx)>1)+1)


def make_windows(df,lengths,seq_len,year):
    valid=(df.Date.dt.year.eq(year)&df.DO.notna()&np.isfinite(df.DO))
    runs=contiguous_runs(valid.to_numpy());windows=[]
    longest=max(lengths)
    for run in runs:
        start=max(int(run[0]),seq_len);stop=int(run[-1])+1
        for length in lengths:
            for anchor in range(start,stop-length+1,length):
                windows.append(('Length_specific',length,anchor))
        # Only full longest-length observed blocks supply comparable starts.
        for anchor in range(start,stop-longest+1,longest):
            for length in lengths:windows.append(('Common_anchor',length,anchor))
    return windows,valid


def event_labels(df,site,registry):
    ids=np.full(len(df),'',dtype=object);types=np.full(len(df),'NonEvent',dtype=object)
    for r in registry.loc[registry.Site_ID.eq(site)].itertuples():
        use=df.Date.between(r.Start_Date,r.End_Date).to_numpy()
        require(not np.any(ids[use]!=''),f'Overlapping registry windows: {site}')
        ids[use]=str(r.Event_ID);types[use]=str(r.Event_Type)
    return ids,types


def load_data(path):
    df=pd.read_csv(path,low_memory=False)
    require({'Date','DO','WQS_name','Region'}.issubset(df.columns),f'Missing input columns: {path}')
    df.Date=pd.to_datetime(df.Date)
    df=df.sort_values('Date').reset_index(drop=True)
    require(not df.Date.duplicated().any(),f'Duplicate dates: {path}')
    require(df.Date.diff().dropna().eq(pd.Timedelta(days=1)).all(),f'Non-daily dates: {path}')
    require(not np.isinf(pd.to_numeric(df.DO,errors='raise')).any(),f'Infinite DO: {path}')
    require(df.WQS_name.dropna().nunique()==1,f'Multiple sites: {path}')
    return df


def validate_assets(manifest_path):
    m=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    require(m.get('version')=='V1','Expected V1 model manifest')
    require(m.get('training_years')==[2021,2022,2023,2024],'Unexpected final training years')
    require(m['configuration']['test_year']==2025,'Expected held-out year 2025')
    require(m['configuration']['sequence_length']==30,'Unexpected model history length')
    require(len(m['features'])==18 and m['features'][-3:]==['DO_lag1','DO_lag2','DO_lag3'],'Unexpected feature order')
    for key in ['weights','sx','sy']:
        p=locate(m[key]);require(p.is_file(),f'Missing {key}: {p}')
        require(sha(p)==m[key+'_sha256'],f'Asset hash mismatch: {key}')
    source=locate(m['source_snapshot'])
    require(source.is_file() and sha(source)==m['source_sha256'],'Saved V1 source snapshot hash mismatch')
    for record in m['input_files']:
        path=locate(record['path'])
        require(path.is_file() and sha(path)==record['sha256'],f'Input changed since training: {path}')
    return m,source


def build_engine(m,source,requested_device):
    import torch
    import torch.nn as nn
    import joblib
    cfg=m['configuration']
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['BiLSTMAttention','准备特征']]
    require(len(nodes)==2,'Model/preprocessing definitions missing from saved source')
    # Extract definitions only. Never import/run V1.py or any training entry point.
    namespace={'torch':torch,'nn':nn,'np':np,'pd':pd,'BASE_FEATURES':cfg['base_features']}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
    device=torch.device(m['environment']['device'] if requested_device=='recorded' else requested_device)
    if device.type=='cuda':require(torch.cuda.is_available(),'Requested CUDA is unavailable')
    torch.backends.cudnn.deterministic=m['environment']['cudnn_deterministic']
    torch.backends.cudnn.benchmark=m['environment']['cudnn_benchmark']
    model=namespace['BiLSTMAttention'](len(m['features']),hidden_size=cfg['hidden_size'],dropout=cfg['dropout']).to(device)
    model.load_state_dict(torch.load(locate(m['weights']),map_location=device,weights_only=True),strict=True)
    model.eval()
    sx=joblib.load(locate(m['sx']));sy=joblib.load(locate(m['sy']))
    require(sx.n_features_in_==len(m['features']) and sy.n_features_in_==1,'Scaler dimensions differ')
    return model,sx,sy,namespace['准备特征'],device


def predict(model,sy,sequences,device,batch_size):
    import torch
    if not len(sequences):return np.array([],dtype=float)
    output=[]
    with torch.no_grad():
        for start in range(0,len(sequences),batch_size):
            batch=torch.from_numpy(np.asarray(sequences[start:start+batch_size],dtype=np.float32)).to(device)
            output.append(model(batch).cpu().numpy())
    return np.clip(sy.inverse_transform(np.vstack(output)).ravel(),0,None)


def prepare_inputs(df,prepare,sx,features):
    prepared,actual=prepare([df],'DO')
    require(actual==features,'Feature order changed')
    require(np.array_equal(prepared[0].Date.to_numpy(),df.Date.to_numpy()),'Preprocessing changed dates/order')
    x=sx.transform(prepared[0][features].values)
    require(np.isfinite(x).all(),'Nonfinite standardized inputs')
    return x,prepared[0]


def masked_sequences(df,start,length,seq,prepare,sx,features):
    # Work on an independent copy for each artificial gap. No predictions feed lags.
    masked=df.copy();masked.loc[start:start+length-1,'DO']=np.nan
    x,prepared=prepare_inputs(masked,prepare,sx,features)
    require(prepared.DO.iloc[start:start+length].isna().all(),'Mask did not survive preprocessing')
    # Track which original dates can feed each ffilled lag. Hidden dates must never
    # be lag sources in any historical row used by this artificial gap.
    raw=masked.DO;used_rows=np.arange(start-seq,start+length-1)
    for lag in (1,2,3):
        expected=raw.shift(lag).ffill().fillna(0).to_numpy(dtype=float)
        require(np.array_equal(expected,prepared[f'DO_lag{lag}'].to_numpy(dtype=float)),f'Lag{lag} mismatch')
        source=pd.Series(np.where(raw.notna(),np.arange(len(raw)),np.nan)).shift(lag).ffill().to_numpy()
        selected=source[used_rows]
        require(not np.any((selected>=start)&(selected<start+length)),'Hidden DO leaked through a lag')
        finite=np.isfinite(selected)
        require(np.all(selected[finite]<=used_rows[finite]-lag),'Future DO entered a lag')
    return np.asarray([x[i-seq:i] for i in range(start,start+length)],dtype=np.float32)


def summarize(points,args,out):
    rows=[];classifications=[];site_rows=[];depth_rows=[];bootstrap=[]
    rng=np.random.default_rng(args.bootstrap_seed)
    def contexts(frame):
        return [('All',frame),('Event',frame[frame.Context.eq('Event')]),('NonEvent',frame[frame.Context.eq('NonEvent')])]
    for (design,length),block in points.groupby(['Design','Length'],sort=True):
        for context,g in contexts(block):
            for method in METHODS:
                available=g[np.isfinite(g[method])]
                row=dict(Design=design,Length=int(length),Context=context,Method=method,
                         Sites=int(available.Site_ID.nunique()),Windows=int(available.Window_ID.nunique()),**metrics(g.Observed_DO,g[method]))
                rows.append(row)
                for threshold in [5.,3.,2.]:
                    actual=available.Observed_DO.to_numpy()<threshold;pred=available[method].to_numpy()<threshold
                    tp=int((actual&pred).sum());fp=int((~actual&pred).sum());fn=int((actual&~pred).sum());tn=int((~actual&~pred).sum())
                    classifications.append(dict(Design=design,Length=int(length),Context=context,Method=method,
                        Threshold=threshold,N=len(available),Observed_Low=tp+fn,Predicted_Low=tp+fp,TP=tp,FP=fp,FN=fn,TN=tn,
                        Recall=tp/(tp+fn) if tp+fn else np.nan,Precision=tp/(tp+fp) if tp+fp else np.nan,
                        F1=2*tp/(2*tp+fp+fn) if tp+fn else np.nan))
                for site,z in available.groupby('Site_ID'):
                    site_rows.append(dict(Design=design,Length=int(length),Context=context,Method=method,Site_ID=site,**metrics(z.Observed_DO,z[method])))
                for depth,z in available.groupby('Depth'):
                    depth_rows.append(dict(Design=design,Length=int(length),Context=context,Method=method,Depth=int(depth),Sites=int(z.Site_ID.nunique()),**metrics(z.Observed_DO,z[method])))
                # Point-weighted metrics; cluster resampling retains every test point
                # from each sampled site. No rows are treated as independent repeats.
                rec=dict(Design=design,Length=int(length),Context=context,Method=method,Sites=row['Sites'],Bootstrap_Reps=args.bootstrap_reps)
                if args.bootstrap_reps>0 and row['Sites']>=args.min_bootstrap_sites:
                    sums=[]
                    for _,z in available.groupby('Site_ID',sort=True):
                        error=(z[method]-z.Observed_DO).to_numpy()
                        sums.append([len(error),abs(error).sum(),(error**2).sum(),error.sum()])
                    values=np.asarray(sums,dtype=float);n=len(values)
                    draw=rng.integers(0,n,size=(args.bootstrap_reps,n));totals=values[draw].sum(axis=1)
                    estimates={'MAE':totals[:,1]/totals[:,0],'RMSE':np.sqrt(totals[:,2]/totals[:,0]),'Bias':totals[:,3]/totals[:,0]}
                    rec['Status']='Computed'
                    for metric,arr in estimates.items():rec[metric+'_CI_Low'],rec[metric+'_CI_High']=map(float,np.quantile(arr,[.025,.975]))
                else:rec['Status']='Not computed: disabled or insufficient site count'
                bootstrap.append(rec)
    for name,records in [('metric_summary',rows),('low_DO_classification',classifications),('site_metrics',site_rows),('gap_depth_metrics',depth_rows),('site_block_intervals',bootstrap)]:
        pd.DataFrame(records).to_csv(out/(name+'.csv'),index=False,encoding='utf-8-sig')
    # Pair errors within the exact same available test dates for each baseline.
    comparisons=[]
    for (design,length),block in points.groupby(['Design','Length'],sort=True):
        for context,g in contexts(block):
            for baseline in ['Unmasked_BiLSTM_Attention','Last_observation']:
                z=g[np.isfinite(g[baseline])&np.isfinite(g.Masked_BiLSTM_Attention)]
                a=metrics(z.Observed_DO,z.Masked_BiLSTM_Attention);b=metrics(z.Observed_DO,z[baseline])
                comparisons.append(dict(Design=design,Length=int(length),Context=context,Baseline=baseline,N=len(z),Sites=int(z.Site_ID.nunique()),Masked_MAE=a['MAE'],Baseline_MAE=b['MAE'],Masked_minus_Baseline_MAE=a['MAE']-b['MAE'],Masked_RMSE=a['RMSE'],Baseline_RMSE=b['RMSE']))
    pd.DataFrame(comparisons).to_csv(out/'paired_baseline_comparison.csv',index=False,encoding='utf-8-sig')
    return pd.DataFrame(rows)


def run(args,out):
    started=time.perf_counter();m,source=validate_assets(args.manifest)
    registry=pd.read_csv(REGISTRY);registry.Start_Date=pd.to_datetime(registry.Start_Date);registry.End_Date=pd.to_datetime(registry.End_Date)
    require(not registry.duplicated(['Site_ID','Event_ID']).any(),'Duplicate registry keys')
    manifest_hash=sha(args.manifest);registry_hash=sha(REGISTRY)
    require(args.year==m['configuration']['test_year'],'Only the held-out test year is allowed')
    seq=m['configuration']['sequence_length'];all_candidates=[];inventory=[];datasets=[]
    scatter_files=list((ROOT/'Result/BiLstm_Att_V1/Reports/By_Site_Scatter').rglob('*_Scatter.csv'))
    for record in m['input_files']:
        path=locate(record['path']);df=load_data(path);site=str(df.WQS_name.dropna().iloc[0]);reach=str(df.Region.dropna().iloc[0])
        if args.sites and site not in args.sites:continue
        raw_windows,valid=make_windows(df,args.lengths,seq,args.year)
        event_ids,event_types=event_labels(df,site,registry)
        selected=[]
        for design in DESIGNS:
            for length in args.lengths:
                possible=[w for w in raw_windows if w[0]==design and w[1]==length]
                chosen=possible[:args.max_windows_per_design_length] if args.max_windows_per_design_length else possible
                inventory.append(dict(Site_ID=site,Reach=reach,Design=design,Length=length,
                    Available_Windows=len(possible),Selected_Windows=len(chosen),Selected_Days=len(chosen)*length,
                    Available_Test_Observed_Days=int(valid.sum()),Selection='All non-overlapping windows' if not args.max_windows_per_design_length else 'Diagnostic first-N cap'))
                selected.extend(chosen)
        for design,length,start in selected:
            end=start+length;eid=event_ids[start:end];truth=df.DO.iloc[start:end].to_numpy(dtype=float)
            wid=f'{design}_{site}_{df.Date.iloc[start]:%Y%m%d}_{length}'
            all_candidates.append(dict(Design=design,Window_ID=wid,Site_ID=site,Reach=reach,Length=length,Start=df.Date.iloc[start],End=df.Date.iloc[end-1],Event_Days=int((eid!='').sum()),Observed_Low5_Days=int((truth<5).sum())))
        datasets.append((site,reach,df,selected,event_ids,event_types))
    require(len(datasets)>0,'No selected stations')
    if args.sites:require(set(args.sites)=={x[0] for x in datasets},'Some requested sites were not found')
    pd.DataFrame(inventory).to_csv(out/'candidate_availability.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_candidates).to_csv(out/'selected_windows.csv',index=False,encoding='utf-8-sig')
    if args.plan_only:
        print(f'PLAN ONLY: {len(datasets)} sites, {len(all_candidates)} windows. No model inference.',flush=True);return
    require(len(all_candidates)>0,'No eligible consecutive-observation windows')
    model,sx,sy,prepare,device=build_engine(m,source,args.device)
    point_path=out/'masked_point_predictions.csv';window_path=out/'window_metrics.csv'
    pd.DataFrame(columns=POINT_COLUMNS).to_csv(point_path,index=False,encoding='utf-8-sig')
    pd.DataFrame(columns=WINDOW_COLUMNS).to_csv(window_path,index=False,encoding='utf-8-sig')
    checks=[]
    for station_index,(site,reach,df,windows,event_ids,event_types) in enumerate(datasets,1):
        unmasked_x,_=prepare_inputs(df,prepare,sx,m['features'])
        test_idx=np.flatnonzero((df.Date.dt.year.eq(args.year)&df.DO.notna()).to_numpy())
        test_idx=test_idx[test_idx>=seq]
        require(len(test_idx)>0,f'No reload-check observations: {site}')
        test_seq=np.asarray([unmasked_x[i-seq:i] for i in test_idx],dtype=np.float32)
        original_pred=predict(model,sy,test_seq,device,args.batch_size)
        matches=[p for p in scatter_files if p.name==f'{site}_Labeled_Scatter.csv']
        require(len(matches)==1,f'Expected one saved scatter file: {site}')
        saved=pd.read_csv(matches[0]);saved=saved[saved.Dataset.eq('Test(25)')].reset_index(drop=True)
        require(len(saved)==len(test_idx),f'Reload sample count mismatch: {site}')
        observed=df.DO.iloc[test_idx].to_numpy(dtype=float)
        require(np.allclose(saved['Observed DO'],np.round(observed,3),atol=1e-7,rtol=0),f'Saved prediction date/order alignment failed: {site}')
        maxerr=float(abs(original_pred-saved['Simulated DO'].to_numpy()).max())
        require(maxerr<=args.reference_tolerance,f'Reloaded predictions differ from V1 at {site}: {maxerr}')
        checks.append(dict(Site_ID=site,N=len(test_idx),Max_Abs_Difference_To_Rounded_Export=maxerr,Tolerance=args.reference_tolerance,Passed=True,Reference_File=str(matches[0].relative_to(ROOT)),Reference_SHA256=sha(matches[0])))
        pd.DataFrame(checks).to_csv(out/'reload_prediction_checks.csv',index=False,encoding='utf-8-sig')
        unmasked_lookup=dict(zip(test_idx,original_pred))
        arrays=[];records=[]
        for design,length,start in windows:
            arrays.append(masked_sequences(df,start,length,seq,prepare,sx,m['features']))
            truth=df.DO.iloc[start:start+length].to_numpy(dtype=float)
            last=df.DO.iloc[:start].dropna();last_index=int(last.index[-1]) if len(last) else None
            last_value=float(last.iloc[-1]) if len(last) else np.nan
            wid=f'{design}_{site}_{df.Date.iloc[start]:%Y%m%d}_{length}'
            records.append((design,length,start,wid,truth,last_index,last_value))
        if arrays:
            predicted=predict(model,sy,np.concatenate(arrays,axis=0),device,args.batch_size)
            offset=0;point_records=[];window_records=[]
            for design,length,start,wid,truth,last_index,last_value in records:
                pp=predicted[offset:offset+length];offset+=length
                stat=metrics(truth,pp)
                window_records.append(dict(Design=design,Window_ID=wid,Site_ID=site,Reach=reach,Length=length,Start=df.Date.iloc[start],End=df.Date.iloc[start+length-1],Event_Days=int((event_ids[start:start+length]!='').sum()),Observed_Low5_Days=int((truth<5).sum()),Last_Observation_Date=df.Date.iloc[last_index] if last_index is not None else pd.NaT,Baseline_Available=last_index is not None,Observed_Mean=float(truth.mean()),Observed_Min=float(truth.min()),Masked_Mean=float(pp.mean()),Masked_Min=float(pp.min()),Mean_Bias=float(pp.mean()-truth.mean()),Minimum_Bias=float(pp.min()-truth.min()),MAE=stat['MAE'],RMSE=stat['RMSE'],Bias=stat['Bias']))
                for depth,(i,y,p) in enumerate(zip(range(start,start+length),truth,pp),1):
                    point_records.append(dict(Design=design,Window_ID=wid,Site_ID=site,Reach=reach,Length=length,Depth=depth,Date=df.Date.iloc[i],Event_ID=event_ids[i],Event_Type=event_types[i],Context='Event' if event_ids[i]!='' else 'NonEvent',Observed_DO=float(y),Masked_BiLSTM_Attention=float(p),Unmasked_BiLSTM_Attention=float(unmasked_lookup[i]),Last_observation=last_value))
            pd.DataFrame(point_records,columns=POINT_COLUMNS).to_csv(point_path,mode='a',header=False,index=False,encoding='utf-8')
            pd.DataFrame(window_records,columns=WINDOW_COLUMNS).to_csv(window_path,mode='a',header=False,index=False,encoding='utf-8')
        print(f'[{station_index}/{len(datasets)}] {site}: reloaded {len(test_idx)} targets, validated {len(windows)} gaps, elapsed {time.perf_counter()-started:.1f}s',flush=True)
    points=pd.read_csv(point_path,low_memory=False)
    require(not points.duplicated(['Design','Site_ID','Length','Date']).any(),'Overlapping windows within a design/site/length')
    results=summarize(points,args,out)
    require(sha(args.manifest)==manifest_hash and sha(REGISTRY)==registry_hash,'Source changed during run')
    for key in ['weights','sx','sy']:require(sha(locate(m[key]))==m[key+'_sha256'],f'Asset changed: {key}')
    for r in m['input_files']:require(sha(locate(r['path']))==r['sha256'],'Input changed during validation')
    scope = '诊断子样本' if args.sites or args.max_windows_per_design_length else '完整验证方案'
    lengths = '、'.join(map(str, args.lengths))
    text=['# DO连续缺测验证结果','',f'{scope}。{len(datasets)}站，{len(all_candidates)}个人工缺口。','',
          f'验证年份为{args.year}年，缺口长度为{lengths}天，推断设备为{device}。站点重采样设置为{args.bootstrap_reps}次，随机种子为{args.bootstrap_seed}，区间计算至少需要{args.min_bootstrap_sites}个站点。',
          '本次仅使用保存的V1权重和scaler推理，没有重训或修改原完成序列。所有站点先通过2025逐点预测重载对照。',
          'Length_specific在各长度内使用不重叠窗口，但不同长度的样本组成不同；Common_anchor使用最长缺口支持的共同起点。两种设计分别汇总，不能混合分母。',
          'Masked_BiLSTM_Attention为连续遮蔽后预测，Unmasked_BiLSTM_Attention为同日期保留观测历史的预测，Last_observation为缺口前最后一次观测的常数延续。',
          '事件标记来自固定S0事件登记，包含序列事件间隔；不是以输入Event_Label简单替代。',
          '统计按逐日记录加权，站点块区间保留站内所有记录；稀少站点组不提供区间。零低值真阳性支持时召回率和F1保持缺失。',
          '原环境变量预处理边界保持回顾性。误差估计适用于有连续原始观测作为真值的验证窗口。','',
          '## 事件期遮蔽精度','',results[(results.Context=='Event')&(results.Method=='Masked_BiLSTM_Attention')].to_csv(index=False)]
    print(f'COMPLETED: {out}',flush=True)


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--lengths',type=int,nargs='+',default=[3,7,14,30])
    ap.add_argument('--year',type=int,default=2025)
    ap.add_argument('--device',default='recorded',choices=['recorded','cpu','cuda'])
    ap.add_argument('--batch-size',type=int,default=2048)
    ap.add_argument('--reference-tolerance',type=float,default=.0011,help='mg/L tolerance versus saved 3-decimal predictions')
    ap.add_argument('--bootstrap-reps',type=int,default=1000)
    ap.add_argument('--bootstrap-seed',type=int,default=20261025)
    ap.add_argument('--min-bootstrap-sites',type=int,default=10)
    ap.add_argument('--sites',nargs='+')
    ap.add_argument('--max-windows-per-design-length',type=int,default=0,help='0 means all; positive values create a labeled diagnostic subset')
    ap.add_argument('--plan-only',action='store_true')
    args=ap.parse_args();args.lengths=sorted(set(args.lengths))
    require(args.lengths and min(args.lengths)>0,'Gap lengths must be positive')
    require(args.batch_size>0 and args.bootstrap_reps>=0 and args.min_bootstrap_sites>=2,'Invalid batch/bootstrap options')
    require(args.reference_tolerance>0 and args.max_windows_per_design_length>=0,'Invalid tolerance/window cap')
    args.manifest=args.manifest.resolve()
    return args


def main():
    args=parse_args()
    if args.output_dir is None:
        suffix='_Plan' if args.plan_only else '_Validation'
        if args.sites or args.max_windows_per_design_length:suffix+='_DiagnosticSubset'
        out=ROOT/'Result/Continuous_gap_validation'/('V1_Continuous_Gap_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+suffix)
    else:out=args.output_dir.resolve()
    # Each run is immutable. Never reuse a nonempty export directory.
    require(not out.exists() or not any(out.iterdir()),f'Output is not empty; choose a new directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    try:run(args,out)
    except Exception:
        (out/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise

if __name__=='__main__':main()
