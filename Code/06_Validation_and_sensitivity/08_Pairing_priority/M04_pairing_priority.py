"""M4 pairing-order audit using fixed current inputs and the ED_F classifier.

Default retains DC > F5H > H5F and compares DC > H5F > F5H. Event identity
is site plus the set of post-merge primitive IDs, never sequential Event_ID.
--audit-only stops after the event comparison. No DO model is loaded or fitted.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import warnings
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Result/Event/Pairing_priority/20261111'
TYPE_MAP = {'hw':'Heatwave_Only', 'flood':'Flood_Only', 'direct':'Direct_Compound',
            'indirect_h2f':'H5F_Compound', 'indirect_f2h':'F5H_Compound'}
TYPES = list(TYPE_MAP.values())
SOURCES = {}
if hasattr(sys.stdout,'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def load(path, name):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


def read(path, **kwargs):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,**kwargs)


def save(frame, name):
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')


def registry_rows(site, daily, primitives, classified):
    rows, assigned = [], []
    for i, segment in enumerate(sorted(sum([list(x) for x in classified], []),key=lambda x:(x['start'],x['end']))):
        members = [p['primitive_id'] for p in primitives if p['start'] >= segment['start'] and p['end'] <= segment['end']]
        assert members, (site,segment)
        assigned.extend(members)
        rows.append(dict(Site_ID=site, Event_ID=f'{site}_EV{i+1:03d}',
            Event_Type=TYPE_MAP[segment.get('comp_subtype',segment.get('type'))],
            Start_Date=segment['start'], End_Date=segment['end'],
            Total_Days=(segment['end']-segment['start']).days+1,
            Region=daily.Region.iloc[0], primitive_ids=json.dumps(sorted(members))))
    assert len(assigned)==len(set(assigned))==len(primitives), ('primitive reuse or loss',site)
    return rows


def compare_site(site, primitives, base, alt, candidates):
    bmap = {p:r for r in base for p in json.loads(r['primitive_ids'])}
    amap = {p:r for r in alt for p in json.loads(r['primitive_ids'])}
    rows = []
    for primitive in primitives:
        pid = primitive['primitive_id']
        b,a = bmap[pid],amap[pid]
        competing = [c for c in candidates if c['cross_priority_competing'] and pid in [c['hw_id'],c['flood_id']]]
        rows.append(dict(site_id=site,primitive_ids=json.dumps([pid]),
            baseline_type=b['Event_Type'],alternative_type=a['Event_Type'],
            baseline_start_end=f"{b['Start_Date']:%Y-%m-%d}/{b['End_Date']:%Y-%m-%d}",
            alternative_start_end=f"{a['Start_Date']:%Y-%m-%d}/{a['End_Date']:%Y-%m-%d}",
            competing_candidate_n=len(competing), pair_changed=b['primitive_ids']!=a['primitive_ids'],
            type_changed=b['Event_Type']!=a['Event_Type'],
            window_changed=(b['Start_Date'],b['End_Date'])!=(a['Start_Date'],a['End_Date']),
            baseline_event_primitive_ids=b['primitive_ids'], alternative_event_primitive_ids=a['primitive_ids'],
            core_metric_or_contrast='',baseline_estimate=np.nan,alternative_estimate=np.nan))
    return rows


def recompute_metrics(registries, daily_by_site):
    """Reuse M05/decrease definitions and the recovery.py five-date calculator."""
    m05=load(ROOT/'Code/06_Validation_and_sensitivity/05_Definition_sensitivity/M05_sensitivity.py','m4_metric_definitions')
    departure=load(ROOT/'Code/04_DO_response/02_Departure_metrics/decrease.py','m4_seasons')
    recovery=load(ROOT/'Code/04_DO_response/03_Early_post_event_slopes/recovery.py','m4_slopes')
    calculated={}
    for priority,registry in registries.items():
        print(f'Recalculating all event windows, both references, next-event bounds and slopes: {priority}',flush=True)
        metric_rows,background_rows,non_event_rows=[],[],[]
        for site,g in registry.groupby('Site_ID'):
            daily=daily_by_site[site]
            events=g.sort_values(['Start_Date','End_Date']).to_dict('records')
            rows,bg=m05.shock_for_site(daily,events,departure.get_season)
            by_date=daily.set_index('Date')
            occupied=pd.Series(False,index=daily.index)
            for event in events:
                occupied |= daily.Date.between(event['Start_Date'],event['End_Date'])
            non=daily.loc[~occupied,['Date','DO','Original_DO']].copy()
            non['Site_ID']=site
            non_event_rows.append(non)
            background_rows.extend(bg)
            for i,row in enumerate(rows):
                next_start=events[i+1]['Start_Date'] if i+1<len(events) else pd.NaT
                post_end=row['End_Date']+pd.Timedelta(days=30)
                if pd.notna(next_start):
                    post_end=min(post_end,next_start-pd.Timedelta(days=1))
                post=daily.loc[daily.Date.gt(row['End_Date']) & daily.Date.le(post_end)].copy()
                post['Day_Offset']=(post.Date-row['End_Date']).dt.days
                slope=recovery.early_do_slope(post)
                event=daily.loc[daily.Date.between(row['Start_Date'],row['End_Date'])]
                target_dates=[row['End_Date']+pd.Timedelta(days=x) for x in [1,3,5,7,9]]
                target=by_date.reindex(target_dates)
                row.update(Reach={'UR':'Upstream','MR':'Midstream','LR':'Downstream'}[row['Region']],
                    Season=departure.get_season(row['Start_Date'].month),
                    Compound_Group='Single' if row['Event_Type'] in ['Heatwave_Only','Flood_Only'] else 'Compound',
                    Event_Observed_Days=int(event.Original_DO.notna().sum()),
                    Event_Observed_Pct=100*event.Original_DO.notna().mean(),
                    Next_Event_Start=next_start,Post_Window_End=post_end,
                    Recovery_Slope=round(slope,4) if np.isfinite(slope) else np.nan,
                    Post_Five_Observed_Days=int(target.Original_DO.notna().sum()) if np.isfinite(slope) else np.nan)
                metric_rows.append(row)
        frame=pd.DataFrame(metric_rows).rename(columns={'Shock_Ratio_LongTerm':'Shock-L','Shock_Ratio_Pre':'Shock-S'})
        calculated[priority]=frame
        save(frame,f'{priority}_event_metrics.csv')
        save(pd.DataFrame(background_rows),f'{priority}_seasonal_references.csv')
        save(pd.concat(non_event_rows,ignore_index=True),f'{priority}_non_event_calendar.csv')
    current=read(ROOT/'Result/Statistical_comparisons/Event_Level_Statistical_Data.csv',parse_dates=['Start_Date','End_Date'])
    checked=calculated['baseline'].merge(current,on=['Site_ID','Event_ID'],validate='one_to_one',suffixes=('','_saved'))
    assert len(checked)==len(current)
    metric_checks=[]
    for field in ['Shock-L','Shock-S','Recovery_Slope']:
        a,b=checked[field],checked[field+'_saved']
        np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=True)
        metric_checks.append(dict(metric=field,n=len(a),valid_n=int(a.notna().sum()),max_absolute_difference=float((a-b).abs().max())))
    save(pd.DataFrame(metric_checks),'baseline_metric_reproduction_checks.csv')
    return calculated


def core_results(calculated):
    stat=load(ROOT/'Code/05_Statistical_analysis/01_Mixed_effects/compare_event_do_responses.py','m4_current_mixed_models')
    log_lines=[]
    stat.log=lambda message:log_lines.append(str(message))
    descriptive=[]
    contrast_rows=[]
    try:
        for priority,frame in calculated.items():
            for metric in ['Shock-L','Shock-S','Recovery_Slope']:
                v=stat.valid_data(frame,metric)
                for group,part in [('All',v)]+[(t,v.loc[v.Event_Type.eq(t)]) for t in TYPES]+[
                    ('Compound',v.loc[v.Compound_Group.eq('Compound')]),('Single',v.loc[v.Compound_Group.eq('Single')])]:
                    descriptive.append(dict(priority=priority,core_metric_or_contrast=metric+'_mean',group=group,
                        estimate=part[metric].mean(),n=len(part),sites=part.Site_ID.nunique(),
                        event_observed_days=int(part.Event_Observed_Days.sum()),event_days=int(part.Total_Days.sum())))
                print(f'Fitting existing mixed models: {priority}, {metric}',flush=True)
                stat.DIAGNOSTICS=[]
                results=pd.DataFrame([stat.fit_compare(v,metric,'A'),stat.fit_compare(v,metric,'B')])
                if results['原始p值'].notna().all():
                    results['Holm校正p值']=stat.multipletests(results['原始p值'],method='holm')[1]
                for row in results.to_dict('records'):
                    contrast_rows.append(dict(priority=priority,core_metric_or_contrast=metric+':'+row['比较'],group='Adjusted_contrast',
                        estimate=row['调整后效应量'],n=row['模型样本量'],sites=row['模型断面数'],
                        ci_lower=row['CI下限'],ci_upper=row['CI上限'],p_raw=row['原始p值'],p_holm=row['Holm校正p值'],
                        target_n=row['目标组样本量'],reference_n=row['参照组样本量'],
                        target_sites=row['目标组断面数'],reference_sites=row['参照组断面数'],fit_status=row['状态']))
                save(pd.DataFrame(stat.DIAGNOSTICS),f'{priority}_{metric.replace("-", "_")}_model_diagnostics.csv')
        desc,contrasts=pd.DataFrame(descriptive),pd.DataFrame(contrast_rows)
        save(desc,'priority_descriptive_metrics.csv')
        save(contrasts,'priority_adjusted_contrasts.csv')
        # Preserve a direct numerical comparison with the current published baseline.
        current=read(ROOT/'Result/Statistical_comparisons/Core_Contrasts.csv')
        for r in contrasts.loc[contrasts.priority.eq('baseline')].itertuples():
            metric,contrast=r.core_metric_or_contrast.split(':')
            saved=current.loc[current.Metric.eq(metric)&current.Contrast.eq(contrast)].iloc[0]
            assert np.isclose(r.estimate,saved.Adjusted_Effect,rtol=0,atol=1e-5),(metric,contrast,r.estimate,saved.Adjusted_Effect)
        combined=pd.concat([desc,contrasts],ignore_index=True)
        keys=['core_metric_or_contrast','group']
        b=combined.loc[combined.priority.eq('baseline')].drop(columns='priority')
        a=combined.loc[combined.priority.eq('alternative')].drop(columns='priority')
        merged=b.merge(a,on=keys,how='outer',validate='one_to_one',suffixes=('_baseline','_alternative'))
        merged=merged.rename(columns={'estimate_baseline':'baseline_estimate','estimate_alternative':'alternative_estimate'})
        save(merged,'priority_core_results.csv')
        print(contrasts[['priority','core_metric_or_contrast','estimate','n','target_n','reference_n','p_holm']].to_string(index=False),flush=True)
    finally:
        (OUT/'mixed_model_log.txt').write_text('\n'.join(log_lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    SOURCES[str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    detector=load(ROOT/'Code/04_DO_response/01_Completed_event_registry/ED_F.py','priority_detector')
    old=load(Path(__file__).resolve().with_name('ED_F_pre_M4.py'),'pre_m4_detector')
    original=read(ROOT/'Data/Event_F/Event_Registry_For_Validation.csv',parse_dates=['Start_Date','End_Date'])
    assert detector.COMP_INTERVAL==5 and detector.HW_CONTINUE_DAYS==3 and detector.HW_MERGE_INTERVAL==2
    all_base,all_alt,all_primitives,all_candidates,comparisons=[],[],[],[],[]
    daily_by_site={}
    for site in sorted(original.Site_ID.unique()):
        d=read(ROOT/f'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR/{site}_Imputed.csv',parse_dates=['Date'])
        d=d.sort_values('Date').reset_index(drop=True)
        assert pd.DatetimeIndex(d.Date).equals(pd.date_range('2021-01-01','2025-12-31'))
        assert d[['Tmax','Q','DO']].notna().all().all()
        daily_by_site[site]=d
        h=read(ROOT/f'Data/Meteorology_data/1991-2020/YR/{site}.csv')
        t,q=np.percentile(h.Tmax.dropna(),90),np.percentile(d.Q.dropna(),95)
        hw=detector.get_base_segments(d.copy(),'hw',t,q)
        fl=detector.get_base_segments(d.copy(),'flood',t,q)
        primitives=[]
        for kind,segments in [('HW',hw),('Q',fl)]:
            for index,s in enumerate(segments):
                primitives.append(dict(site_id=site,primitive_id=f"{site}:{kind}:{s['start']:%Y%m%d}:{s['end']:%Y%m%d}",
                    primitive_kind=kind,primitive_index=index,**s,Tmax_P90=t,Q_P95=q))
        audit_b,audit_a={},{}
        baseline=detector.classify_complex_events(hw,fl,audit=audit_b)
        alternative=detector.classify_complex_events(hw,fl,sequence_priority=('H5F','F5H'),audit=audit_a)
        original_code=old.classify_complex_events(hw,fl)
        b=registry_rows(site,d,primitives,baseline)
        a=registry_rows(site,d,primitives,alternative)
        check_old=registry_rows(site,d,primitives,original_code)
        assert b==check_old, ('default differs from pre-M4 classifier',site)
        for rows in [b,a]:
            detector.validate_event_exclusivity(site,rows)
        saved=original.loc[original.Site_ID.eq(site)]
        key=lambda rows:sorted((r['Event_Type'],r['Start_Date'],r['End_Date']) for r in rows)
        assert key(b)==key(saved.to_dict('records')), ('current S0 registry not reproduced',site)
        assert audit_b.get('dc_used_hw',[])==audit_a.get('dc_used_hw',[])
        assert audit_b.get('dc_used_flood',[])==audit_a.get('dc_used_flood',[])
        assert audit_b.get('eligible_candidates',[])==audit_a.get('eligible_candidates',[])
        hw_ids={p['primitive_index']:p['primitive_id'] for p in primitives if p['primitive_kind']=='HW'}
        fl_ids={p['primitive_index']:p['primitive_id'] for p in primitives if p['primitive_kind']=='Q'}
        candidates=[]
        for c in audit_b.get('eligible_candidates',[]):
            selected=lambda audit:any(c==x for x in audit.get('selected_candidates',[]))
            candidates.append(dict(site_id=site,**c,hw_id=hw_ids[c['hw_idx']],flood_id=fl_ids[c['flood_idx']],
                                   baseline_selected=selected(audit_b),alternative_selected=selected(audit_a)))
        for c in candidates:
            members={c['hw_id'],c['flood_id']}
            rivals=[x for x in candidates if x['direction']!=c['direction'] and members & {x['hw_id'],x['flood_id']}]
            same=[x for x in candidates if x is not c and x['direction']==c['direction'] and members & {x['hw_id'],x['flood_id']}]
            c['cross_priority_competing']=bool(rivals)
            c['opposite_direction_rival_n']=len(rivals)
            c['same_direction_rival_n']=len(same)
        comparisons.extend(compare_site(site,primitives,b,a,candidates))
        all_base.extend(b);all_alt.extend(a);all_primitives.extend(primitives);all_candidates.extend(candidates)
    base,alt=pd.DataFrame(all_base),pd.DataFrame(all_alt)
    comparison=pd.DataFrame(comparisons)
    candidates=pd.DataFrame(all_candidates)
    save(base,'baseline_registry.csv');save(alt,'alternative_registry.csv')
    save(pd.DataFrame(all_primitives),'primitive_segments.csv')
    save(candidates,'sequential_candidate_gaps.csv')
    save(comparison,'pairing_priority_comparison.csv')
    count_rows=[dict(event_type=t,baseline_n=int(base.Event_Type.eq(t).sum()),alternative_n=int(alt.Event_Type.eq(t).sum())) for t in TYPES]
    save(pd.DataFrame(count_rows),'priority_event_counts.csv')
    affected=comparison.loc[comparison.pair_changed|comparison.type_changed|comparison.window_changed]
    summary=dict(sites=base.Site_ID.nunique(),primitive_n=len(comparison),baseline_events=len(base),alternative_events=len(alt),
        eligible_candidate_n=len(candidates),cross_priority_competing_candidate_n=int(candidates.cross_priority_competing.sum()),
        cross_priority_competing_primitive_n=len({p for c in all_candidates if c['cross_priority_competing'] for p in [c['hw_id'],c['flood_id']]
            if len({x['direction'] for x in all_candidates if p in [x['hw_id'],x['flood_id']]})>1}),
        pair_changed_primitive_n=int(comparison.pair_changed.sum()),type_changed_primitive_n=int(comparison.type_changed.sum()),
        window_changed_primitive_n=int(comparison.window_changed.sum()),
        baseline_affected_event_n=len(affected[['site_id','baseline_event_primitive_ids']].drop_duplicates()),
        alternative_affected_event_n=len(affected[['site_id','alternative_event_primitive_ids']].drop_duplicates()),
        baseline_code_and_registry_verified=True,dc_components_fixed=True,primitive_partition_and_nonoverlap_verified=True,
        baseline_priority='DC > F5H > H5F',alternative_priority='DC > H5F > F5H',
        competition_definition='Distinct post-DC candidate edges sharing a primitive with an opposite-direction edge; not equal to selected pairs',
        model_training=False)
    (OUT/'priority_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(pd.DataFrame(count_rows).to_string(index=False),flush=True)
    if len(affected) and not args.audit_only:
        calculated=recompute_metrics({'baseline':base,'alternative':alt},daily_by_site)
        core_results(calculated)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in SOURCES.items())
    (OUT/'run_provenance.json').write_text(json.dumps(dict(sources=SOURCES,metrics_needed=bool(len(affected))),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
