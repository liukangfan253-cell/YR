"""Evaluate fixed hyperparameters with an independent site–event group split. Use the existing SHAP analysis as a separate result."""
from pathlib import Path
import ast,json,time,sys,platform,hashlib
import numpy as np,pandas as pd,sklearn,xgboost
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from xgboost import XGBRegressor
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'Result/SHAP_Grouped_Validation'
sys.path.insert(0, str(ROOT / 'Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis'))
from analysis_config import read_analysis_config
OUT.mkdir(parents=True,exist_ok=True)
RUN=True;SEED=42;TEST_FRACTION=.20;JOBS=2
def params(df,model):
 d={}
 for r in df[df.Model.eq(model)].itertuples():
  v=str(r.Value)
  try:v=ast.literal_eval(v)
  except Exception:pass
  if r.Parameter in ['n_estimators','max_depth','min_samples_leaf','min_samples_split','min_child_weight'] and v is not None:v=int(v)
  d[r.Parameter]=v
 return d
estim=[];configs=[];hashes={}
for tag in ['H','F','DC','H5F','F5H']:
 cfg=read_analysis_config(ROOT/f'Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_{tag}_shap.py', ROOT)
 p=Path(cfg['DATA_FILE']);b=cfg['OUTPUT_BASENAME']
 hist=pd.read_csv(ROOT / cfg['OUTPUT_DIR'] / f'{b}_Bayesian_Search_History.csv')
 # Estimate runtime from mean fit time in the saved search. Multiply by 2.5 to allow for parameter and device differences.
 for name,g in hist.groupby('Model'):estim.append(dict(Event=tag,Model=name,History_Mean_Fit_Seconds=g.mean_fit_time.median(),Conservative_Seconds=2.5*g.mean_fit_time.max()))
 configs.append((tag,cfg));hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
pd.DataFrame(estim).to_csv(OUT/'SHAP计算成本估计.csv',index=False,encoding='utf-8-sig')
print('15次固定参数拟合，搜索历史粗估上界秒',sum(x['Conservative_Seconds'] for x in estim),flush=True)
rows=[];splits=[];predictions=[];timings=[]
for tag,cfg in configs:
 d=pd.read_csv(cfg['DATA_FILE']);features=cfg['FEATURE_COLS'];d=d.dropna(subset=['DO']+features).copy().reset_index(names='Source_Row')
 assert d.Event_ID.notna().all() and d.WQS_name.notna().all()
 mapping=d.groupby('Event_ID').WQS_name.nunique();globally_unique=bool(mapping.max()==1)
 # Use explicit composite keys. JSON arrays avoid delimiter collisions.
 groups=pd.Series([json.dumps([str(s),str(e)],ensure_ascii=False) for s,e in zip(d.WQS_name,d.Event_ID)])
 tr,te=next(GroupShuffleSplit(n_splits=1,test_size=TEST_FRACTION,random_state=SEED).split(d[features],d.DO,groups=groups))
 assert not(set(groups.iloc[tr])&set(groups.iloc[te]));gtrain=set(groups.iloc[tr]);gtest=set(groups.iloc[te])
 for i,r in d.iterrows():splits.append(dict(Event_Type=tag,Site_ID=r.WQS_name,Event_ID=r.Event_ID,Date=r.Date,Source_Row=r.Source_Row,Split='Train' if groups.iloc[i] in gtrain else 'Test'))
 stem=cfg['OUTPUT_BASENAME'];result_dir=ROOT / cfg['OUTPUT_DIR']
 parpath=result_dir/f'{stem}_Best_Parameters.csv';comparison=pd.read_csv(result_dir/f'{stem}_Model_Comparison.csv')
 selected=comparison.sort_values(['Best_CV_RMSE','CV_RMSE_Std']).Model.iloc[0];par=pd.read_csv(parpath)
 hashes[str(parpath)]=hashlib.sha256(parpath.read_bytes()).hexdigest()
 for name,cls,base in [('Random Forest',RandomForestRegressor,{'criterion':'squared_error'}),('Extra Trees',ExtraTreesRegressor,{'criterion':'squared_error','bootstrap':False}),('XGBoost',XGBRegressor,{'objective':'reg:squarederror','eval_metric':'rmse','tree_method':'hist','verbosity':0})]:
  kw=params(par,name);model=cls(**base,**kw,random_state=SEED,n_jobs=JOBS);start=time.perf_counter();model.fit(d.iloc[tr][features],d.iloc[tr].DO);yp=model.predict(d.iloc[te][features]);elapsed=time.perf_counter()-start
  y=d.iloc[te].DO.to_numpy();orig=comparison[comparison.Model.eq(name)].iloc[0]
  r=dict(Event_Type=tag,Model=name,Originally_Selected=name==selected,Total_Records=len(d),Train_Records=len(tr),Test_Records=len(te),Total_Events=groups.nunique(),Train_Events=len(gtrain),Test_Events=len(gtest),Train_Test_Event_Overlap=0,Event_ID_Global_Unique=globally_unique,Test_Sites=d.iloc[te].WQS_name.nunique(),Shared_Train_Test_Sites=len(set(d.iloc[tr].WQS_name)&set(d.iloc[te].WQS_name)),R2=r2_score(y,yp),MAE=mean_absolute_error(y,yp),RMSE=np.sqrt(mean_squared_error(y,yp)),Original_RecordSplit_R2=orig.Test_R2,Original_RecordSplit_RMSE=orig.Test_RMSE,Elapsed_Seconds=elapsed,Fixed_Parameters=json.dumps(kw),Input_Path=cfg['DATA_FILE'])
  rows.append(r);timings.append({'Event':tag,'Model':name,'Seconds':elapsed})
  for i,o,p in zip(te,y,yp):predictions.append(dict(Event_Type=tag,Model=name,Site_ID=d.iloc[i].WQS_name,Event_ID=d.iloc[i].Event_ID,Date=d.iloc[i].Date,Observed_DO=o,Predicted_DO=float(p)))
  print(tag,name,'R2',round(r['R2'],4),'RMSE',round(r['RMSE'],4),'秒',round(elapsed,2),flush=True)
pd.DataFrame(rows).to_csv(OUT/'SHAP固定参数分组验证.csv',index=False,encoding='utf-8-sig');pd.DataFrame(predictions).to_csv(OUT/'SHAP分组逐点预测.csv',index=False,encoding='utf-8-sig');pd.DataFrame(splits).to_csv(OUT/'SHAP分组拆分清单.csv',index=False,encoding='utf-8-sig')
environment={'Python':sys.version,'Platform':platform.platform(),'sklearn':sklearn.__version__,'numpy':np.__version__,'xgboost':xgboost.__version__,'Seed':SEED,'Preprocessing':'只删除目标或16特征缺失行，与原shap1相同；无插补/标准化拟合；不改变上游回顾性预处理。','Caveat':'原最佳超参数从原随机记录CV得到，可能已接触本次分组测试事件的信息；这是复用参数的敏感性而非完全独立调参验证。不是气象过程阻断或未见站点测试。','Sources':hashes,'Timing':timings}
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
print(json.dumps(environment, ensure_ascii=False, indent=2))
