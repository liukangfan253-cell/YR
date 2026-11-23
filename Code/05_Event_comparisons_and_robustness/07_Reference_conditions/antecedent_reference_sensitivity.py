# -*- coding: utf-8 -*-
"""Comment 9: antecedent-reference support and window sensitivity.

Run with Python + numpy + pandas. No retraining, new event detection or manuscript edits.
Each run creates a new Result/Comment9/Comment9_<timestamp>_Analysis directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
KEY = ['Site_ID', 'Event_ID']
SOURCES = {
    'events': ROOT / 'Result/Statistical_comparisons/Event_Level_Statistical_Data.csv',
    's0': ROOT / 'Result/Event/Definition_sensitivity/20261111/20261111_M05_S0事件级Shock结果.csv',
    'observed_references': ROOT / 'Result/Observation_only_references/20261111/20261111_实测来源核对.csv',
    'paired': ROOT / 'Result/Decrease/Paired_baselines/20261111/20261111_M04共同事件配对数据.csv',
    'observed_summary': ROOT / 'Result/Observation_only_references/20261111/20261111_纯实测双基线敏感性.csv',
    'daily': ROOT / 'Result/Observation_only_references/20261111/20261111_逐日实测来源.csv',
}
WINDOWS = {'W9_4': (9, 4), 'W7_4': (7, 4), 'W14_4': (14, 4)}
CONFIGS = [
    ('Original_completed', 'W9_4', 3, 'Completed', 'Original'),
    ('Observed_3', 'W9_4', 3, 'Observed', 'Support'),
    ('Observed_4', 'W9_4', 4, 'Observed', 'Support'),
    ('Observed_5', 'W9_4', 5, 'Observed', 'Support'),
    ('Observed_6', 'W9_4', 6, 'Observed', 'Support'),
    ('Short_7_4_obs3', 'W7_4', 3, 'Observed', 'Window'),
    ('Long_14_4_obs6', 'W14_4', 6, 'Observed', 'Window'),
    ('Long_14_4_obs3', 'W14_4', 3, 'Observed', 'Window_support_control'),
]
FAMILIES = ['Fixed_completed_event', 'Observed_event_and_seasonal']
PRIMARY_WINDOWS = ['Observed_3', 'Short_7_4_obs3', 'Long_14_4_obs6']
SAME_MIN_WINDOWS = ['Observed_3', 'Short_7_4_obs3', 'Long_14_4_obs3']
HASHES = {}
LOG = []


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(message):
    LOG.append(str(message))
    print(message, flush=True)


def read(path):
    HASHES[str(path)] = sha(path)
    return pd.read_csv(path, low_memory=False)


def save(out, name, frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    frame.to_csv(out / (name + '.csv'), index=False, encoding='utf-8-sig', float_format='%.12g')
    return frame


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def close(a, b, message):
    require(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                        atol=2e-6, rtol=0, equal_nan=True), message)


def positive(x):
    return np.isfinite(x) & (x > 0)


def classification(L, S):
    """A/B same direction, C/D opposite; exact equality to 1 is Neutral."""
    return np.select([(L < 1) & (S < 1), (L > 1) & (S > 1),
                      (L < 1) & (S > 1), (L > 1) & (S < 1)],
                     ['Both_below', 'Both_above', 'L_below_S_above', 'L_above_S_below'],
                     default='Neutral')


def prepare(out):
    t = {k: read(p) for k, p in SOURCES.items()}
    e, s0, old, daily = (t[k] for k in ['events', 's0', 'observed_references', 'daily'])
    for name, d in [('events', e), ('s0', s0), ('observed_references', old)]:
        require(not d[KEY].isna().any().any() and not d.duplicated(KEY).any(), f'Invalid keys in {name}')
        require(set(map(tuple, d[KEY].values)) == set(map(tuple, e[KEY].values)), f'Event keys differ: {name}')
    require(e.Site_ID.nunique() == 66, 'Unexpected station coverage')
    for c in ['Start_Date', 'End_Date']:
        e[c] = pd.to_datetime(e[c], errors='raise')
    daily.Date = pd.to_datetime(daily.Date, errors='raise')
    require(not daily.duplicated(['Site_ID', 'Date']).any(), 'Duplicate site dates')
    require(daily.Is_Event.isin([True, False]).all(), 'Is_Event must be a Boolean, not a guessed flag')
    require(set(daily.Site_ID) == set(e.Site_ID), 'Daily site set differs')
    require(len(daily) == 66 * 1826, 'Daily study calendar differs')
    require(daily.DO_Source.eq(np.where(daily.DO_QC.notna(), 'QC_observed',
                                      np.where(daily.DO_Final.notna(), 'Imputed', 'Missing'))).all(), 'Source flags differ')
    require(np.isfinite(daily.DO_Final).all(), 'Completed daily DO missing/nonfinite')
    require(np.isfinite(daily.DO_QC.dropna()).all(), 'Original daily DO nonfinite')
    close(daily.loc[daily.DO_QC.notna(), 'DO_Final'], daily.DO_QC.dropna(), 'Retained observations changed')
    e = e.sort_values(KEY).reset_index(drop=True)
    a = s0.set_index(KEY).loc[pd.MultiIndex.from_frame(e[KEY])].reset_index()
    o = old.set_index(KEY).loc[pd.MultiIndex.from_frame(e[KEY])].reset_index()
    for d, name in [(a, 'S0'), (o, 'observed trace')]:
        require(d.Event_Type.eq(e.Event_Type).all() and d.Reach.eq(e.Reach).all(), f'{name} metadata differs')
        for c in ['Start_Date', 'End_Date']:
            require(pd.to_datetime(d[c]).eq(e[c]).all(), f'{name} event dates differ')
    for target, source in [('Shock-L', 'Shock_Ratio_LongTerm'), ('Shock-S', 'Shock_Ratio_Pre')]:
        close(e[target], a[source], 'S0 saved ratios differ')
    close(e['Shock-L'], o.A_Shock_L, 'Original L differs from trace')
    close(e['Shock-S'], o.A_Shock_S, 'Original S differs from trace')
    e['Event_DO_Fixed'] = a.Event_DO
    e['Seasonal_Reference_Fixed'] = a.Seasonal_Baseline_DO
    e['Event_DO_Observed'] = o.Event_DO_Observed
    e['Seasonal_Reference_Observed'] = o.Seasonal_Baseline_Observed
    e['L_Observed'] = o.C_Shock_L
    e['C_S_Existing'] = o.C_Shock_S
    e['Event_Observed_Days'] = o.Event_Observed_Days
    e['Original_Pre_Observed_Days'] = o.Pre_Observed_Days
    e['Original_Pre_Completed_Days'] = o.Pre_Completed_Days
    e['Original_Pre_Reference'] = a.Pre_Baseline_DO
    window_rows, date_rows, checks = [], [], []
    for site, g in e.groupby('Site_ID', sort=True):
        d = daily[daily.Site_ID.eq(site)].sort_values('Date').set_index('Date')
        require(d.index.equals(pd.date_range('2021-01-01', '2025-12-31')), f'Calendar differs: {site}')
        qc_path = ROOT / f'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned/{site}.csv'
        qc = read(qc_path)
        qc.Date = pd.to_datetime(qc.Date, errors='raise')
        require(not qc.Date.duplicated().any(), f'Duplicate original QC dates: {site}')
        qc_values = qc.set_index('Date').DO.reindex(d.index)
        close(d.DO_QC, qc_values, f'Original QC differs from provenance: {site}')
        expected = pd.Series('', index=d.index)
        for ev in g.itertuples():
            event_dates = pd.date_range(ev.Start_Date, ev.End_Date)
            require(event_dates.isin(d.index).all() and expected.loc[event_dates].eq('').all(), f'Event overlap/outside calendar: {ev.Event_ID}')
            expected.loc[event_dates] = ev.Event_ID
            ed = d.loc[event_dates]
            close(ed.DO_Final.mean(), ev.Event_DO_Fixed, 'Event numerator differs')
            close(ed.DO_QC.mean(), ev.Event_DO_Observed, 'Observed event numerator differs')
            require(ed.DO_QC.count() == ev.Event_Observed_Days, 'Event observation count differs')
            for name, (start, end) in WINDOWS.items():
                dates = pd.date_range(ev.Start_Date - pd.Timedelta(days=start), ev.Start_Date - pd.Timedelta(days=end))
                w = d.reindex(dates)
                eligible = w.Is_Event.eq(False)
                obs = w.loc[eligible, 'DO_QC'].dropna()
                complete = w.loc[eligible, 'DO_Final'].dropna()
                n = len(obs)
                observed_mean = obs.mean()
                complete_mean = complete.mean()
                row = dict(Site_ID=site, Event_ID=ev.Event_ID, Window=name,
                           First_Day_Before=start, Last_Day_Before=end, Calendar_Days=len(dates),
                           Outside_Study_Days=int(w.Is_Event.isna().sum()),
                           Excluded_Event_Days=int(w.Is_Event.eq(True).sum()),
                           NonEvent_Days=int(eligible.sum()), Observed_Days=n,
                           Completed_Days=len(complete), Imputed_Days=int((eligible & w.DO_QC.isna() & w.DO_Final.notna()).sum()),
                           Observed_Mean=observed_mean, Completed_Mean=complete_mean,
                           Observed_SD=obs.std(), Observed_Min=obs.min(), Observed_Max=obs.max(),
                           Observed_Median=obs.median(), Observed_Values=';'.join(f'{v:.12g}' for v in obs),
                           Observed_Dates=';'.join(x.strftime('%Y-%m-%d') for x in obs.index))
                window_rows.append(row)
                for date, day in w.iterrows():
                    date_rows.append(dict(Site_ID=site, Event_ID=ev.Event_ID, Window=name, Date=date,
                                          Days_Before=int((ev.Start_Date - date).days),
                                          Status='Outside_study' if pd.isna(day.Is_Event) else 'Excluded_event' if day.Is_Event else 'Eligible_non_event',
                                          DO_QC=day.DO_QC, DO_Final=day.DO_Final,
                                          Used_Observed=bool(eligible.loc[date] and pd.notna(day.DO_QC))))
                if name == 'W9_4':
                    require(n == ev.Original_Pre_Observed_Days and len(complete) == ev.Original_Pre_Completed_Days, 'Original pre counts differ')
                    original_mean = complete_mean if len(complete) >= 3 else np.nan
                    close(original_mean, ev.Original_Pre_Reference, 'Original pre mean differs')
                    original_s = ev.Event_DO_Fixed / original_mean if positive(original_mean) else np.nan
                    close(original_s, g.loc[g.Event_ID.eq(ev.Event_ID), 'Shock-S'].iloc[0], 'Original S not reproduced')
                    old_pre = o.loc[o.Event_ID.eq(ev.Event_ID) & o.Site_ID.eq(site), 'Pre_Baseline_Observed'].iloc[0]
                    close(observed_mean if n >= 3 else np.nan, old_pre, 'Original observed pre mean differs')
        require(d.Is_Event.eq(expected.ne('')).all() and d.Event_ID.fillna('').eq(expected).all(), f'Event mask differs: {site}')
        # Independently reconstruct the observation-only seasonal reference, retaining original calendar-day weighting.
        season_means = d.loc[~d.Is_Event].groupby('Season').DO_QC.mean()
        for ev in g.itertuples():
            season_reference = d.loc[ev.Start_Date:ev.End_Date, 'Season'].map(season_means).mean()
            close(season_reference, ev.Seasonal_Reference_Observed, 'Observed seasonal reference differs')
        checks.append(dict(Site_ID=site, Study_Days=len(d), Events=len(g), QC_Days=int(d.DO_QC.count()), Status='PASS'))
    windows = save(out, 'antecedent_windows', window_rows)
    save(out, 'antecedent_daily_trace', date_rows)
    save(out, 'site_input_checks', checks)
    close(e.Event_DO_Fixed / e.Seasonal_Reference_Fixed, e['Shock-L'], 'Original seasonal ratio not reproduced')
    valid = positive(e['Shock-L']) & positive(e['Shock-S'])
    paired = t['paired']
    require(valid.sum() == len(paired) and (((e.loc[valid, 'Shock-L'] - 1) * (e.loc[valid, 'Shock-S'] - 1)) < 0).sum() == paired.Direction_Flip.sum(),
            'Current M04 paired direction counts differ')
    save(out, 'fixed_events', e)
    log(f'Input checks PASS: {len(e)} events, 66 sites; V5 M04 paired direction counts reproduced.')
    return e, windows


def scenario_data(e, windows, out):
    rows = []
    for config, window, minimum, source, purpose in CONFIGS:
        w = windows[windows.Window.eq(window)]
        base = e.merge(w, on=KEY, validate='one_to_one')
        number = base.Completed_Days if source == 'Completed' else base.Observed_Days
        mean = base.Completed_Mean if source == 'Completed' else base.Observed_Mean
        reference = mean.where(number.ge(minimum))
        for family in FAMILIES:
            d = base.copy()
            d['Family'], d['Scenario'], d['Purpose'] = family, config, purpose
            d['Reference_Source'], d['Minimum_Values'] = source, minimum
            d['Antecedent_Reference'] = reference
            d['Numerator'] = d.Event_DO_Fixed if family == FAMILIES[0] else d.Event_DO_Observed
            d['L'] = d['Shock-L'] if family == FAMILIES[0] else d.L_Observed
            d['S'] = (d.Numerator / d.Antecedent_Reference).where(positive(d.Antecedent_Reference))
            if config == 'Original_completed' and family == FAMILIES[0]:
                close(d.S, d['Shock-S'], 'Control scenario differs')
                d['S'] = d['Shock-S']  # Preserve saved reference control, after independent numerical reproduction.
            d['Paired_Valid'] = positive(d.L) & positive(d.S)
            d['Class'] = classification(d.L, d.S)
            d.loc[~d.Paired_Valid, 'Class'] = 'Unavailable'
            d['Discordant'] = d.Paired_Valid & ((d.L - 1) * (d.S - 1) < 0)
            d['Beyond5'] = d.Paired_Valid & d.L.sub(1).abs().gt(.05) & d.S.sub(1).abs().gt(.05)
            d['Discordant_Beyond5'] = d.Discordant & d.Beyond5
            d['Status'] = np.select([~positive(d.Numerator), number.lt(minimum), ~positive(reference)],
                                    ['No_valid_event_numerator', 'Insufficient_antecedent_values', 'Nonpositive_or_missing_reference'], default='Valid')
            rows.append(d)
    data = pd.concat(rows, ignore_index=True)
    c = data[data.Family.eq(FAMILIES[1]) & data.Scenario.eq('Observed_3')]
    close(c.S, c.C_S_Existing, 'Existing observation-only Shock-S not reproduced')
    observed = read(SOURCES['observed_summary']).query("Sample == 'Available' and Group == 'C' and Level == 'Overall'").iloc[0]
    require(c.Paired_Valid.sum() == observed.Paired_N and c.Discordant.sum() == observed.Discordant_N, 'Current observation-only direction counts differ')
    save(out, 'scenario_event_results', data)
    save(out, 'scenario_definitions', [dict(Scenario=c, Window=w, Minimum_Values=n, Reference_Source=s, Purpose=p,
         Calendar_Days=WINDOWS[w][0]-WINDOWS[w][1]+1) for c, w, n, s, p in CONFIGS])
    return data


class Bootstrap:
    def __init__(self, reps, seed):
        self.reps, self.seed, self.cache = reps, seed, {}

    def weights(self, sites):
        sites = tuple(sites)
        if sites not in self.cache:
            salt = int.from_bytes(hashlib.sha256('|'.join(sites).encode()).digest()[:4], 'little')
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, salt]))
            self.cache[sites] = rng.multinomial(len(sites), np.full(len(sites), 1 / len(sites)), size=self.reps)
        return self.cache[sites]

    def interval(self, d, column):
        counts = d.reset_index(drop=True).groupby('Site_ID', sort=True).agg(N=('Event_ID', 'size'), X=(column, 'sum'))
        if len(counts) < 10:
            return dict(CI_Low=np.nan, CI_High=np.nan, Bootstrap_Valid=0, CI_Status='Fewer_than_10_sites')
        w = self.weights(counts.index)
        den = w @ counts.N.to_numpy()
        draws = 100 * (w @ counts.X.to_numpy(dtype=float)) / den
        return dict(CI_Low=float(np.quantile(draws, .025)), CI_High=float(np.quantile(draws, .975)),
                    Bootstrap_Valid=len(draws), CI_Status='Site_block_percentile')


def levels(d):
    yield 'Overall', 'All', d
    for column in ['Event_Type', 'Reach']:
        for entity, g in d.groupby(column, sort=True):
            yield column, str(entity), g
    for (reach, event), g in d.groupby(['Reach', 'Event_Type'], sort=True):
        yield 'Reach_Event_Type', f'{reach}|{event}', g


def summarize(g, boot):
    p = g[g.Paired_Valid].reset_index(drop=True).copy()
    neutral = p.Class.eq('Neutral')
    r = dict(Candidate_Events=len(g), Paired_N=len(p), Sites=int(p.Site_ID.nunique()),
             Neutral_N=int(neutral.sum()), Discordant_N=int(p.Discordant.sum()),
             Discordance_Pct=100*p.Discordant.mean(),
             Discordance_NonNeutral_Pct=100*p.Discordant.sum()/(len(p)-neutral.sum()) if len(p)-neutral.sum() else np.nan,
             Beyond5_N=int(p.Beyond5.sum()), Discordant_Beyond5_N=int(p.Discordant_Beyond5.sum()),
             Discordant_Beyond5_AllPairs_Pct=100*p.Discordant_Beyond5.mean(),
             Discordant_Beyond5_Conditional_Pct=100*p.Discordant_Beyond5.sum()/p.Beyond5.sum() if p.Beyond5.sum() else np.nan,
             Event_Observed_Days_Mean=p.Event_Observed_Days.mean(), Observed_Pre_Days_Mean=p.Observed_Days.mean(),
             Reference_AtMost_0_5_N=int(p.Antecedent_Reference.le(.5).sum()),
             Reference_AtMost_1_N=int(p.Antecedent_Reference.le(1).sum()))
    for col in ['L', 'S', 'Antecedent_Reference']:
        r.update({col+'_Mean': p[col].mean(), col+'_Median': p[col].median(),
                  col+'_Q25': p[col].quantile(.25), col+'_Q75': p[col].quantile(.75),
                  col+'_SiteEqual_Mean': p.groupby('Site_ID')[col].mean().mean()})
    r.update(boot.interval(p, 'Discordant'))
    return r


def summarize_all(data, boot, out):
    summaries, compositions, selection, compared, transitions = [], [], [], [], []
    for family, f in data.groupby('Family', sort=False):
        cases = {s: g.set_index(KEY, drop=False) for s, g in f.groupby('Scenario', sort=False)}
        def intersection(names):
            index = cases[names[0]].index[cases[names[0]].Paired_Valid]
            for name in names[1:]:
                index = index.intersection(cases[name].index[cases[name].Paired_Valid])
            return index
        common = {
            'Common_support_3_to_6': intersection(['Observed_3', 'Observed_4', 'Observed_5', 'Observed_6']),
            'Common_primary_windows': intersection(PRIMARY_WINDOWS),
            'Common_windows_min3': intersection(SAME_MIN_WINDOWS),
        }
        for sample in ['Available', *common]:
            if sample == 'Available':
                selected_cases = list(cases)
            elif sample == 'Common_support_3_to_6':
                selected_cases = ['Original_completed', 'Observed_3', 'Observed_4', 'Observed_5', 'Observed_6']
            else:
                selected_cases = ['Original_completed'] + (PRIMARY_WINDOWS if sample == 'Common_primary_windows' else SAME_MIN_WINDOWS)
            for scenario in selected_cases:
                d = cases[scenario] if sample == 'Available' else cases[scenario].loc[common[sample]]
                for level, entity, g in levels(d):
                    summaries.append(dict(Family=family, Sample=sample, Scenario=scenario, Level=level, Entity=entity, **summarize(g, boot)))
                if sample == 'Available':
                    for status, g in d.groupby('Status', sort=True):
                        selection.append(dict(Family=family, Scenario=scenario, Status=status, Events=len(g), Sites=g.Site_ID.nunique()))
                    for dimension in ['Event_Type', 'Reach', 'Season']:
                        for entity, g in d[d.Paired_Valid].groupby(dimension, sort=True):
                            compositions.append(dict(Family=family, Scenario=scenario, Dimension=dimension, Entity=entity,
                                                     Events=len(g), Sites=g.Site_ID.nunique(),
                                                     Pct=100*len(g)/d.Paired_Valid.sum(),
                                                     Mean_Duration=(g.End_Date-g.Start_Date).dt.days.mean()+1))
        # Every support threshold uses all available original readings, so estimates must match on the common events.
        idx = common['Common_support_3_to_6']
        for scenario in ['Observed_4', 'Observed_5', 'Observed_6']:
            close(cases[scenario].loc[idx, 'S'], cases['Observed_3'].loc[idx, 'S'], 'Thresholds changed identical-event estimates')
        for scenario in [s for s in cases if s != 'Original_completed']:
            for comparator in ['Original_completed', 'Observed_3']:
                if comparator == scenario:
                    continue
                x, y = cases[comparator], cases[scenario]
                matched = x.Paired_Valid & y.Paired_Valid
                paired = y.loc[matched].copy()
                paired['Delta_Discordant'] = paired.Discordant.astype(int)-x.loc[matched, 'Discordant'].astype(int)
                paired['Comparator_S'] = x.loc[matched, 'S']
                paired['Comparator_Reference'] = x.loc[matched, 'Antecedent_Reference']
                paired['Comparator_Class'] = x.loc[matched, 'Class']
                paired['Comparator_Discordant'] = x.loc[matched, 'Discordant']
                paired['Binary_Changed'] = paired.Discordant.ne(paired.Comparator_Discordant)
                paired['Class_Changed'] = paired.Class.ne(paired.Comparator_Class)
                for level, entity, g in levels(paired):
                    ci = boot.interval(g, 'Delta_Discordant')
                    original_group = x if level == 'Overall' else x[x.Event_Type.eq(entity)] if level == 'Event_Type' else x[x.Reach.eq(entity)] if level == 'Reach' else x[x.Reach.eq(entity.split('|')[0]) & x.Event_Type.eq(entity.split('|')[1])]
                    original_valid = original_group[original_group.Paired_Valid]
                    comparator_all = 100*original_valid.Discordant.mean()
                    restricted = 100*g.Comparator_Discordant.mean()
                    alternative = 100*g.Discordant.mean()
                    compared.append(dict(Family=family, Comparator=comparator, Scenario=scenario, Level=level, Entity=entity,
                        Paired_N=len(g), Sites=g.Site_ID.nunique(), Comparator_All_N=len(original_valid),
                        Comparator_All_Discordance_Pct=comparator_all, Comparator_Matched_Discordance_Pct=restricted,
                        Alternative_Matched_Discordance_Pct=alternative,
                        Selection_Change_pp=restricted-comparator_all, Reference_Change_pp=alternative-restricted,
                        Reference_Change_CI_Low=ci['CI_Low'], Reference_Change_CI_High=ci['CI_High'], CI_Status=ci['CI_Status'],
                        Binary_Changed_N=int(g.Binary_Changed.sum()), Class_Changed_N=int(g.Class_Changed.sum()),
                        Entered_Discordance_N=int((g.Discordant & ~g.Comparator_Discordant).sum()),
                        Left_Discordance_N=int((~g.Discordant & g.Comparator_Discordant).sum()),
                        Median_S_Change=(g.S-g.Comparator_S).median(), Median_Reference_Change=(g.Antecedent_Reference-g.Comparator_Reference).median()))
                paired['Comparator'], paired['Scenario'] = comparator, scenario
                transitions.append(paired.reset_index(drop=True)[KEY+['Family','Comparator','Scenario','Reach','Event_Type',
                    'Comparator_S','S','Comparator_Reference','Antecedent_Reference','Comparator_Class','Class',
                    'Comparator_Discordant','Discordant','Binary_Changed','Class_Changed']])
            control = cases['Original_completed']
            selection.append(dict(Family=family, Scenario=scenario, Status='Valid_alternative_but_original_unavailable',
                                  Events=int((y.Paired_Valid & ~control.Paired_Valid).sum()),
                                  Sites=y.loc[y.Paired_Valid & ~control.Paired_Valid, 'Site_ID'].nunique()))
    summary = save(out, 'sensitivity_summary', summaries)
    comparison = save(out, 'matched_comparisons', compared)
    save(out, 'event_classification_transitions', pd.concat(transitions, ignore_index=True))
    save(out, 'sample_composition', compositions)
    save(out, 'eligibility_counts', selection)
    save(out, 'near_zero_reference_diagnostics', data[data.Paired_Valid & data.Antecedent_Reference.le(1)])
    save(out, 'large_ratio_diagnostics', data[data.Paired_Valid & (data.S.gt(2) | data.S.lt(.5))])
    primary_original = data[data.Family.eq(FAMILIES[0]) & data.Scenario.eq('Original_completed')]
    support = []
    for n in range(7):
        g = primary_original[primary_original.Observed_Days.eq(n)]
        support.append(dict(Original_Observed_Pre_Days=n, Registry_Events=len(g), Registry_Sites=g.Site_ID.nunique(), **summarize(g, boot)))
    save(out, 'original_observation_support', support)
    log('Summaries and paired decomposition completed; support-common estimates identical as required.')
    return summary, comparison


def report(out, summary, comparison, args):
    # Plain Markdown built without an optional tabulate dependency.
    def table(d, columns):
        selected = d[columns]
        lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
        for row in selected.itertuples(index=False, name=None):
            lines.append('| '+' | '.join('NA' if pd.isna(x) else f'{x:.3f}' if isinstance(x, (float,np.floating)) else str(x) for x in row)+' |')
        return '\n'.join(lines)
    overall = summary[summary.Sample.eq('Available') & summary.Level.eq('Overall')]
    primary = overall[overall.Family.eq(FAMILIES[0])]
    paired = comparison[comparison.Family.eq(FAMILIES[0]) & comparison.Comparator.eq('Original_completed') & comparison.Level.eq('Overall')]
    text = ['# Comment 9 前期参考稳定性分析',
        '固定本轮重算的事件、日期及分类。主分析固定完成序列事件期均值和原季节参考，只改变前期参考的来源、原始观测要求或时间窗口。次分析固定原始观测事件均值与原始观测季节参考，单独列报，不与主分析合并。',
        '## 预先定义的方案',
        '原窗口为起始前9至4天，最低原始观测分别为3、4、5、6条。短窗口为前7至4天且至少3条；长窗口为前14至4天且至少6条，另列长窗口至少3条作为观测门槛控制。所有窗口排除同站任何原事件日及研究期外日期；最低数针对原定日历窗口，不因剔除事件日而降低。窗口变化是本次敏感性设定，不声称来自既有文献标准。',
        '## 主分析，各自可用样本',
        table(primary,['Scenario','Paired_N','Sites','Discordant_N','Discordance_Pct','CI_Low','CI_High','Discordant_Beyond5_AllPairs_Pct','S_Median','Antecedent_Reference_Median']),
        '## 相同事件上的参考变化，与原完成参考比较',
        table(paired,['Scenario','Paired_N','Comparator_Matched_Discordance_Pct','Alternative_Matched_Discordance_Pct','Selection_Change_pp','Reference_Change_pp','Reference_Change_CI_Low','Reference_Change_CI_High','Binary_Changed_N']),
        'Selection_Change_pp比较原方案全部有效事件与匹配子集，Reference_Change_pp在同一批事件中比较参考值。若替代窗口新增了原方案不可计算的事件，它们单列在eligibility_counts.csv，不能由上述匹配分解代表。',
        '## 共同样本检查',
        table(summary[summary.Family.eq(FAMILIES[0]) & summary.Level.eq('Overall') & summary.Sample.ne('Available')],
              ['Sample','Scenario','Paired_N','Discordance_Pct','Discordant_Beyond5_AllPairs_Pct','S_Median']),
        '提高最低观测数只改变入选条件，不改变相同事件已有原始观测的均值。因此，3至6条门槛的共同事件结果应完全相同；程序已核对。要评估参考值变化，需与原完成参考或替代窗口作同事件比较，不能将门槛共同样本相同当作测量稳定性的独立证明。',
        '## 次分析，仅原始事件观测与原始季节参考',
        table(overall[overall.Family.eq(FAMILIES[1])],['Scenario','Paired_N','Discordant_N','Discordance_Pct','S_Median']),
        '原始观测前期至少3条的次分析须复现本轮V5实测来源独立核对的配对数与方向改变数；本次已作为强制一致性断言。主分析与次分析的分子和季节参考不同，不能直接解释其差异为前期重构误差。',
        '## 统计与解释边界',
        f'符号不一致为(L−1)(S−1)<0，精确等于1记Neutral；主比例分母为所有有效配对，另列非Neutral分母。5%带要求两个比值均超出1±0.05且方向相反，分别报告全部配对及超幅条件分母。站点块bootstrap {args.bootstrap_reps}次，seed={args.bootstrap_seed}，少于10个站点不提供区间；这些区间没有处理共享区域过程或全部时空依赖，也没有传播测量和填补不确定性。各方案采用同一站点重抽样权重作配对差值，区间为百分位区间，不另做显著性检验。',
        '前期均值≤0.5、≤1 mg/L和Shock-S>2或<0.5仅用于完整诊断清单，不用于剔除或改写数值。原始观测值为既有质控日均值，未新增低值质控规则。所有可计算方案保留，未依据结果选择窗口。',
        '这组分析评估结果对前期参考定义和观测支持的依赖，不能证明前期真实均值、仪器准确性、完全无观测事件响应或事件因果效应。',
        '## 文件',
        'scenario_event_results.csv为两套分子/季节参考、8个方案的事件明细；antecedent_windows.csv与antecedent_daily_trace.csv保留每条前期记录；sensitivity_summary.csv保留总体、类型、河段及河段×类型；matched_comparisons.csv和event_classification_transitions.csv保留配对分解及逐事件分类变化。所有源文件只读，哈希和参数见run_provenance.json。未修改V13、V14或现有分析。']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--bootstrap-reps', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261025)
    args = parser.parse_args()
    require(args.bootstrap_reps >= 100, 'Use at least 100 bootstrap replicates')
    out = args.output_dir or ROOT/'Result/Comment9'/('Comment9_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_Analysis')
    require(not out.exists() or not any(out.iterdir()), 'Refusing to overwrite nonempty output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'run_status.json', dict(Status='RUNNING', Output=str(out)))
    log(f'Output: {out}')
    try:
        e, windows = prepare(out)
        data = scenario_data(e, windows, out)
        boot = Bootstrap(args.bootstrap_reps, args.bootstrap_seed)
        summary, comparison = summarize_all(data, boot, out)
        report(out, summary, comparison, args)
        for path, expected in HASHES.items():
            require(sha(path) == expected, f'Source changed during run: {path}')
        manuscript_hashes = {str(p):sha(p) for p in (ROOT/'Submission_Documents').rglob('*.docx') if not p.name.startswith('~$')}
        write_json(out/'run_provenance.json', dict(Script=str(Path(__file__)),Script_SHA256=sha(__file__),
            Inputs=HASHES, Manuscript_Hashes_At_Completion=manuscript_hashes,
            Windows=WINDOWS, Scenarios=CONFIGS, Families=FAMILIES,
            Bootstrap_Reps=args.bootstrap_reps, Bootstrap_Seed=args.bootstrap_seed,
            Bootstrap_Unit='Site; within-site events kept together; fewer than 10 sites => no CI',
            Python=sys.version, Platform=platform.platform(), Numpy=np.__version__,Pandas=pd.__version__,
            Scope='Completed fixed-event descriptive sensitivity; no model training, LMM refit or manuscript edits'))
        write_json(out/'run_status.json', dict(Status='COMPLETED', Output=str(out),
            Scope='All 8 scenarios, 2 numerator/reference families, available and matched populations'))
        log('COMPLETED. Read the saved CSV results; statistical stability must be assessed from results, not this status.')
    except Exception:
        write_json(out/'run_status.json', dict(Status='FAILED', Error=traceback.format_exc(), Output=str(out)))
        raise
    finally:
        (out/'run_log.txt').write_text('\n'.join(LOG)+'\n', encoding='utf-8')


if __name__ == '__main__':
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
