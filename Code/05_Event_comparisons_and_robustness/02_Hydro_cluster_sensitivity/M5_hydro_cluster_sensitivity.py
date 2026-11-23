"""M5: six site-FE/season OLS models with covariance clustered by shared Q source.

Run with the existing scientific Python environment. Uses the current baseline
event population and keeps the original MixedLM results as a same-sample comparator.
No event identification, DO reconstruction, or MixedLM refitting is performed.
"""
from pathlib import Path
import hashlib
import inspect
import json
import platform
import sys
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import t
import statsmodels
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.sandwich_covariance import cov_cluster

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'Result/Statistical_comparisons/Hydro_cluster/20261111'
EVENTS=ROOT/'Result/Statistical_comparisons/Event_Level_Statistical_Data.csv'
MIXED=ROOT/'Result/Statistical_comparisons/Core_Contrasts.csv'
MAPPING=ROOT/'Config/Study_Station_Mapping.csv'
RESPONSES=['Shock-L','Shock-S','Recovery_Slope']
TYPES=['Heatwave_Only','Flood_Only','Direct_Compound','H5F_Compound','F5H_Compound']
SOURCES={}
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')


def read(path,**kwargs):
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,**kwargs)


def save(frame,name):
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def count_support(frame):
    return dict(n_events=len(frame),n_wq_sites=int(frame.wq_site_id.nunique()),
                n_hydro_clusters=int(frame.hydro_site_id.nunique()))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    SOURCES[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    data=read(EVENTS,parse_dates=['Start_Date','End_Date']).rename(columns={'Site_ID':'wq_site_id'})
    mapping=read(MAPPING)
    original=read(MIXED)
    assert not data.duplicated(['wq_site_id','Event_ID']).any()
    assert not mapping.WQS_name.duplicated().any()
    assert mapping.Matched_HS_Name.notna().all()
    mapping=mapping.rename(columns={'WQS_name':'wq_site_id','Matched_HS_Name':'hydro_site_id'})
    data=data.merge(mapping[['wq_site_id','hydro_site_id']],on='wq_site_id',how='left',validate='many_to_one')
    assert data.hydro_site_id.notna().all()
    assert data.wq_site_id.nunique()==66 and data.hydro_site_id.nunique()==42
    # Verify the actual completed-data Q-source identifiers, not a surrogate ID per WQ station.
    for station in mapping.loc[mapping.wq_site_id.isin(data.wq_site_id.unique())].itertuples():
        p=ROOT/f'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR/{station.wq_site_id}_Imputed.csv'
        completed=read(p,usecols=['WQS_name','Matched_HS_Name'])
        assert completed.WQS_name.eq(station.wq_site_id).all()
        assert completed.Matched_HS_Name.eq(station.hydro_site_id).all()
    source_map=data[['wq_site_id','hydro_site_id']].drop_duplicates()
    shared=source_map.groupby('hydro_site_id').wq_site_id.nunique()
    assert shared.gt(1).sum()==16 and shared.loc[shared.gt(1)].sum()==40
    save(source_map,'hydro_cluster_mapping.csv')
    onset=data.Start_Date.dt.month.map(lambda m:{0:'Winter',1:'Spring',2:'Summer',3:'Autumn'}[(m%12)//3])
    assert onset.eq(data.Season).all()
    assert data.Compound_Group.eq(np.where(data.Event_Type.isin(TYPES[:2]),'Single','Compound')).all()
    support_rows,contrasts,diagnostics,membership=[],[],[],[]
    for response in RESPONSES:
        raw=data[response]
        original_covariates=data[['wq_site_id','Event_Type','Reach','Season']].notna().all(axis=1)
        keep=np.isfinite(raw)&original_covariates
        if response!='Recovery_Slope':keep &= raw.gt(0)
        sample=data.loc[keep].copy()
        sample['Response']=np.log(sample[response]) if response!='Recovery_Slope' else sample[response]
        assert np.isfinite(sample.Response).all()
        # Negative/zero log responses remain in the same original eligible population.
        membership.append(sample[['wq_site_id','hydro_site_id','Event_ID','Event_Type','Season',response,'Response']].rename(columns={response:'original_response'}).assign(response=response))
        for level,group,part in [('Overall','All',sample)]+[('Event_type',x,sample.loc[sample.Event_Type.eq(x)]) for x in TYPES]+[
            ('Compound_indicator',x,sample.loc[sample.Compound_Group.eq(x)]) for x in ['Compound','Single']]:
            support_rows.append(dict(response=response,support_level=level,group=group,**count_support(part),
                hydro_cluster_ids=json.dumps(sorted(part.hydro_site_id.unique())),
                events_per_cluster_max=int(part.groupby('hydro_site_id').size().max()) if len(part) else 0))
        for model,factor,reference,target,label in [('A','Event_Type','H5F_Compound','F5H_Compound','F5H-H5F'),
                ('B','Compound_Group','Single','Compound','compound-single')]:
            formula=f'Response ~ C({factor}, Treatment(reference="{reference}")) + C(wq_site_id) + C(Season)'
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                fitted=smf.ols(formula,sample,missing='raise').fit()
                robust=fitted.get_robustcov_results(cov_type='cluster',groups=sample.hydro_site_id.to_numpy(),
                    use_correction=True,df_correction=True,use_t=True)
            X=fitted.model.exog
            n,k=X.shape
            G=int(sample.hydro_site_id.nunique())
            rank=int(np.linalg.matrix_rank(X))
            assert rank==k and 'Reach' not in formula
            assert bool(robust.use_t) and robust.df_resid_inference==G-1
            term=f'C({factor}, Treatment(reference="{reference}"))[T.{target}]'
            index=fitted.model.exog_names.index(term)
            beta=float(robust.params[index]);se=float(robust.bse[index])
            assert np.isfinite(beta) and np.isfinite(se) and se>0
            df=float(robust.df_resid_inference)
            ci=robust.conf_int(alpha=.05)[index]
            p=float(robust.pvalues[index])
            np.testing.assert_allclose(ci,[beta-t.ppf(.975,df)*se,beta+t.ppf(.975,df)*se],rtol=1e-12,atol=1e-12)
            assert np.isclose(p,2*t.sf(abs(beta/se),df),rtol=1e-12,atol=1e-12)
            target_sample=sample.loc[sample[factor].eq(target)]
            reference_sample=sample.loc[sample[factor].eq(reference)]
            target_clusters=set(target_sample.hydro_site_id);reference_clusters=set(reference_sample.hydro_site_id)
            comparator_label='F5H vs H5F' if model=='A' else 'Compound vs Single'
            previous=original.loc[original.Metric.eq(response)&original.Contrast.eq(comparator_label)].iloc[0]
            assert previous.Model_N==n and previous.Model_Sites==sample.wq_site_id.nunique()
            assert previous.Target_Group_N==len(target_sample) and previous.Reference_Group_N==len(reference_sample)
            original_scale=lambda x:float(np.exp(x)) if response!='Recovery_Slope' else float(x)
            row=dict(response=response,contrast=label,model_parameterization=model,**count_support(sample),
                contrast_support_clusters=len(target_clusters|reference_clusters),
                target_n_events=len(target_sample),reference_n_events=len(reference_sample),
                target_n_wq_sites=int(target_sample.wq_site_id.nunique()),reference_n_wq_sites=int(reference_sample.wq_site_id.nunique()),
                target_n_hydro_clusters=len(target_clusters),reference_n_hydro_clusters=len(reference_clusters),
                shared_support_clusters=len(target_clusters&reference_clusters),
                min_group_support_clusters=min(len(target_clusters),len(reference_clusters)),
                site_adjustment='Water_quality_site_fixed_effects',
                covariance_method='CR1_clustered_by_matched_Q_source',
                effect_original_scale=original_scale(beta),CI95_low=original_scale(ci[0]),CI95_high=original_scale(ci[1]),
                p_raw=p,p_Holm_2=np.nan,original_mixed_effect=float(previous.Adjusted_Effect),
                original_mixed_CI95_low=float(previous.CI_Lower),original_mixed_CI95_high=float(previous.CI_Upper),
                original_mixed_p_raw=float(previous.P_Raw),original_mixed_p_Holm_2=float(previous.P_Holm),
                original_mixed_site_adjustment='WQ_site_random_intercept_plus_reach_and_season',
                t_reference_df=df,t_reference='Student_t_G_minus_1_approximation',
                finite_cluster_correction=G/(G-1)*(n-1)/(n-k),n_parameters=k,
                use_correction=True,df_correction=True,use_t=True,
                n_log_negative=int(sample.Response.lt(0).sum()) if response!='Recovery_Slope' else np.nan,
                n_log_zero=int(sample.Response.eq(0).sum()) if response!='Recovery_Slope' else np.nan,
                fit_status='Estimable',formula=formula,software='statsmodels',software_version=statsmodels.__version__)
            contrasts.append(row)
            for role,group,part in [('Target',target,target_sample),('Reference',reference,reference_sample)]:
                support_rows.append(dict(response=response,support_level='Contrast_group',group=group,contrast=label,
                    role=role,**count_support(part),hydro_cluster_ids=json.dumps(sorted(part.hydro_site_id.unique()))))
            diagnostics.append(dict(response=response,model=model,formula=formula,n=n,k=k,rank=rank,
                G=G,t_reference_df=df,df_residual=float(robust.df_resid),use_t=bool(robust.use_t),
                use_correction=robust.cov_kwds['use_correction'],df_correction=robust.cov_kwds['adjust_df'],
                beta=beta,cluster_SE=se,correction_factor=G/(G-1)*(n-1)/(n-k),
                n_negative_transformed=int(sample.Response.lt(0).sum()),
                warnings=' | '.join(str(w.message) for w in caught)))
        print(response, count_support(sample),flush=True)
    result=pd.DataFrame(contrasts)
    for response in RESPONSES:
        mask=result.response.eq(response)
        assert mask.sum()==2
        result.loc[mask,'p_Holm_2']=multipletests(result.loc[mask,'p_raw'],method='holm')[1]
    save(result,'hydro_cluster_contrasts.csv')
    save(pd.DataFrame(support_rows),'model_support_counts.csv')
    save(pd.DataFrame(diagnostics),'hydro_cluster_model_diagnostics.csv')
    save(pd.concat(membership,ignore_index=True),'model_event_membership.csv')
    # Archive the actual installed implementation underlying the documented options.
    (OUT/'statsmodels_cluster_covariance_implementation.txt').write_text(
        inspect.getsource(cov_cluster)+'\n\n'+inspect.getsource(RegressionResults.get_robustcov_results),encoding='utf-8')
    report=dict(sources=SOURCES,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,statsmodels=statsmodels.__version__,model_n=6,wq_sites=66,hydro_clusters=42,
        shared_Q_clusters=16,wq_sites_using_shared_Q=40,
        cluster_variable='Matched_HS_Name, verified against actual completed-data files',
        sample_rule='Same original response-specific eligible event IDs; finite positive raw ratios then log, no post-log positivity filter',
        finite_correction='G/(G-1) * (N-1)/(N-K), statsmodels use_correction=True',
        inference='use_t=True, df_correction=True; t reference df=G-1 is an approximation',
        contrast_support_clusters_definition='Union of target and reference hydro clusters; each group and their intersection are also explicit',
        interpretation='Changes both site adjustment and covariance; does not remove cross-Q regional weather dependence or imputation uncertainty',
        baseline_population='Current primary DC > F5H > H5F registration; M4 alternative not expanded here')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in SOURCES.items())
    (OUT/'run_provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(result[['response','contrast','effect_original_scale','CI95_low','CI95_high','p_Holm_2',
                  'original_mixed_effect','target_n_hydro_clusters','reference_n_hydro_clusters','t_reference_df']].to_string(index=False))


if __name__=='__main__':main()
