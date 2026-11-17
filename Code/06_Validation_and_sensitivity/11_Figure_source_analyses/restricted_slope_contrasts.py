"""Update only the two prespecified restricted-cohort slope comparisons on V5.

Reuse the existing V5 MixedLM fitting/checking functions unchanged. No DO
reconstruction, event identification, baseline calculation or model search.
"""
from pathlib import Path
import importlib.util
import hashlib,json,sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Result/Post_event_eligibility/20261111_Analysis'
INPUT=ROOT/'Result/Post_event_eligibility/20261111_Analysis/event_selection_and_intensity.csv'
MODEL=ROOT/'Code/05_Statistical_analysis/01_Mixed_effects/compare_event_do_responses.py'
FULL=ROOT/'Result/Statistical_comparisons/Core_Contrasts.csv'
spec=importlib.util.spec_from_file_location('v5_fixed_mixed_models',MODEL)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from english_table_output import english_table

def main():
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [INPUT,MODEL,FULL]}
    data=pd.read_csv(INPUT)
    if data.duplicated(['Site_ID','Event_ID']).any():raise ValueError('Duplicate V5 event keys.')
    mask=data.Eligible.astype(str).str.lower().eq('true')
    if not np.isfinite(data.loc[mask,'Recovery_Slope']).all():raise ValueError('Invalid eligible V5 slope.')
    cohorts={'Any_event_observation':mask&data.Event_Observed_Days.gt(0),
             'Every_event_day_observed':mask&data.Event_Observed_Days.eq(data.Duration_Days)}
    rows=[];diagnostics=[];membership=[]
    for name,selected in cohorts.items():
        subset=data.loc[selected].copy()
        if subset[['Site_ID','Reach','Season','Event_Type','Compound_Group']].isna().any().any():raise ValueError('Missing model covariates.')
        subset['Response']=subset.Recovery_Slope
        start=len(module.DIAGNOSTICS)
        print(f'V5 {name}: {len(subset)} events, {subset.Site_ID.nunique()} sites',flush=True)
        result=pd.DataFrame([module.fit_compare(subset,'Recovery_Slope',model) for model in ['A','B']])
        if not result['状态'].eq('可靠').all():raise ValueError(f'Unreliable restricted fit: {name}')
        result['Holm校正p值']=multipletests(result['原始p值'],method='holm')[1]
        result['显著性']=np.where(result['Holm校正p值']<module.ALPHA,'显著','证据不足以支持差异')
        result['Cohort']=name;rows.append(english_table(result))
        d=english_table(pd.DataFrame(module.DIAGNOSTICS[start:]))
        d['Cohort']=name;diagnostics.append(d)
        membership.append(subset[['Site_ID','Event_ID','Event_Type','Reach','Season','Recovery_Slope']].assign(Cohort=name))
    OUT.mkdir(parents=True,exist_ok=True)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'observation_restricted_slope_contrasts.csv',index=False,encoding='utf-8-sig',float_format='%.12g')
    pd.concat(diagnostics,ignore_index=True).to_csv(OUT/'restricted_slope_model_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.concat(membership,ignore_index=True).to_csv(OUT/'restricted_slope_model_membership.csv',index=False,encoding='utf-8-sig')
    for path,digest in source_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    report={'sources':source_hashes,'data_version':'20261111 rerun','existing_model_functions_reused':True,
            'model':'Existing V5 REML MixedLM; reach and onset season; site random intercept.',
            'contrasts':'F5H-H5F and Compound-Single; two per cohort; existing Wald CI/Holm rules.',
            'optimizer_rules':module.OPTIMIZERS,'maxiter':module.MAXITER,
            'reason':'Recomputed on current rerun data.',
            'restricted_cohort_counts':{name:int(mask.sum()) for name,mask in cohorts.items()},
            'original_V5_files_modified':False,'V4_data_used':False}
    (OUT/'restricted_slope_update_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('V5 restricted slope model outputs complete.',flush=True)

if __name__=='__main__':main()
