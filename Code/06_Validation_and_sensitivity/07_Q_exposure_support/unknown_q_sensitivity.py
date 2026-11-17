"""Rebuild Table S4(e–f) from the current discharge and event inputs."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from export_table_s4_q_support import DATES, ROOT, true_runs


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def uncertain_days(heatwaves, possible_flow, missing, gap):
    flows = list(true_runs(possible_flow))
    heat = [(DATES.get_loc(h['start']), DATES.get_loc(h['end']) + 1) for h in heatwaves]
    nodes = flows + heat
    neighbours = [set() for _ in nodes]
    for i, (a, b) in enumerate(flows):
        for j, (c, d) in enumerate(heat, start=len(flows)):
            if max(c - b, a - d) <= gap:
                neighbours[i].add(j)
                neighbours[j].add(i)

    result = np.zeros(len(DATES), dtype=bool)
    visited = set()
    for i, (a, b) in enumerate(flows):
        if i in visited or not missing[a:b].any():
            continue
        pending = [i]
        component = set()
        while pending:
            node = pending.pop()
            if node not in component:
                component.add(node)
                pending.extend(neighbours[node] - component)
        visited.update(component)
        result[min(nodes[j][0] for j in component):max(nodes[j][1] for j in component)] = True
    return result


def event_keys(detector, heatwaves, high_flow):
    flows = [
        {'start': DATES[a], 'end': DATES[b - 1], 'days': b - a, 'type': 'flood'}
        for a, b in true_runs(high_flow)
    ]
    compound, heat, flood = detector.classify_complex_events(heatwaves, flows)
    labels = {'direct': 'Direct_Compound', 'indirect_h2f': 'H5F_Compound',
              'indirect_f2h': 'F5H_Compound'}
    return {(labels[e['comp_subtype']], e['start'], e['end']) for e in compound} | {
        (label, e['start'], e['end'])
        for label, episodes in [('Heatwave_Only', heat), ('Flood_Only', flood)]
        for e in episodes
    }


def bound_assignments(observed_high, missing):
    """Deterministic examples of possible high-flow assignments on unknown days."""
    assignments = {name: np.zeros(len(DATES), dtype=bool) for name in (
        'Alternating_first', 'Alternating_second', 'Endpoints', 'First_half', 'Second_half')}
    for start, stop in true_runs(missing):
        middle = start + (stop - start + 1) // 2
        assignments['Alternating_first'][start:stop:2] = True
        assignments['Alternating_second'][start + 1:stop:2] = True
        assignments['Endpoints'][[start, stop - 1]] = True
        assignments['First_half'][start:middle] = True
        assignments['Second_half'][middle - ((stop - start) % 2):stop] = True
    return {'Observed_lower': observed_high, 'Possible_upper': observed_high | missing,
            **{name: observed_high | values for name, values in assignments.items()}}


def summarize(events, scenario):
    paired = events[['Shock-L', 'Shock-S']].dropna()
    paired = paired[(paired > 0).all(axis=1)]
    discordant = int(((paired['Shock-L'] - 1) * (paired['Shock-S'] - 1) < 0).sum())
    slopes = events.Recovery_Slope.dropna()
    return dict(Scenario=scenario, Events=len(events), Sites=events.Site_ID.nunique(),
                Mean_Shock_L=events['Shock-L'].mean(), Paired_N=len(paired),
                Discordant_N=discordant, Discordant_Pct=100 * discordant / len(paired),
                Slope_N=len(slopes), Positive_Slope_N=int(slopes.gt(0).sum()),
                Positive_Slope_Pct=100 * slopes.gt(0).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q-dir', type=Path, default=ROOT / 'Supplementary_Data/TableS4_Q_Support')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'Result/Q_exposure_support/20261111')
    args = parser.parse_args()
    detector = load_script('Code/04_DO_response/01_Completed_event_registry/ED_F.py', 'q_detector')
    model = load_script('Code/05_Statistical_analysis/01_Mixed_effects/compare_event_do_responses.py', 'q_models')
    events = pd.read_csv(model.OUTPUT_DIR / 'Event_Level_Statistical_Data.csv',
                         parse_dates=['Start_Date', 'End_Date'])
    q_daily = pd.read_csv(args.q_dir / 'S4_Daily_Q_Provenance.csv', parse_dates=['Date'])
    gauges = {g: d.set_index('Date', verify_integrity=True).reindex(DATES)
              for g, d in q_daily.groupby('Hydro_Site_ID')}
    observation = pd.read_csv(ROOT / 'Supplementary_Data/S8/daily_DO_observation_provenance.csv',
                              parse_dates=['Date'])
    dates_by_site = {s: d.set_index('Date', verify_integrity=True).reindex(DATES)
                     for s, d in observation.groupby('Site_ID')}
    mapping = pd.read_csv(ROOT / 'Config/Study_Station_Mapping.csv').set_index(
        'WQS_name', verify_integrity=True).Matched_HS_Name
    observed_reference, event_audit = [], []
    one_day = pd.Timedelta(days=1)
    for site, site_events in events.groupby('Site_ID'):
        gauge = mapping.loc[site]
        q = gauges[gauge]
        missing = q.Q_Original_Missing.to_numpy(dtype=bool)
        observed_high = q.Q_Original.ge(q.P95).to_numpy()
        current = dates_by_site[site]
        merged = pd.read_csv(detector.ORIGINAL_FOLDER / f'{site}_Imputed.csv',
                             usecols=['Date', 'Tmax', 'Q', 'Matched_HS_Name'], parse_dates=['Date'])
        merged = merged.set_index('Date', verify_integrity=True).reindex(DATES)
        if not merged.Matched_HS_Name.eq(gauge).all() or not np.allclose(
                merged.Q, q.Q_Completed, atol=1e-8, rtol=0, equal_nan=True):
            raise ValueError(f'{site}: completed Q or gauge assignment differs from the Q audit')
        history = pd.read_csv(detector.BASE_FOLDER / f'{site}.csv', usecols=['Tmax'])
        threshold = np.percentile(history.Tmax.dropna(), detector.TARGET_HW_P)
        heatwaves = detector.get_base_segments(merged.reset_index(), 'hw', threshold, 0)
        uncertain = uncertain_days(heatwaves, observed_high | missing, missing, detector.COMP_INTERVAL)
        uncertain = pd.Series(uncertain, index=DATES)
        original_ref = ~current.Is_Event
        selected_ref = original_ref & ~uncertain
        old_background = current.loc[original_ref].groupby('Season').DO_Final.mean()
        new_background = current.loc[selected_ref].groupby('Season').DO_Final.mean()
        removed = current.loc[original_ref & uncertain, ['Season', 'DO_Final']].copy()
        removed['Site_ID'] = site
        removed['Hydro_Site_ID'] = gauge
        removed['Q_Original_Missing'] = q.Q_Original_Missing.reindex(removed.index)
        observed_reference.append(removed.reset_index())
        bounds = {name: event_keys(detector, heatwaves, assignment)
                  for name, assignment in bound_assignments(observed_high, missing).items()}
        for _, event in site_events.iterrows():
            start, end = event.Start_Date, event.End_Date
            retained = not uncertain.loc[start:end].any()
            event_key = (event.Event_Type, start, end)
            changed = [name for name, keys in bounds.items() if event_key not in keys]
            if retained and changed:
                raise ValueError(f'{event.Event_ID}: retained event changed under {changed}')
            event_days = current.loc[start:end]
            pre_days = current.loc[start - 9 * one_day:start - 4 * one_day]
            before = pre_days.loc[original_ref.reindex(pre_days.index), 'DO_Final'].dropna()
            after = pre_days.loc[selected_ref.reindex(pre_days.index), 'DO_Final'].dropna()
            mean_do = event_days.DO_Final.mean()
            original_l = event_days.Season.map(old_background).mean()
            revised_l = event_days.Season.map(new_background).mean()
            expected_s = mean_do / before.mean() if len(before) >= 3 and before.mean() > 0 else np.nan
            if not np.allclose([mean_do / original_l, expected_s], [event['Shock-L'], event['Shock-S']],
                               atol=1e-10, rtol=0, equal_nan=True):
                raise ValueError(f'{event.Event_ID}: baseline ratios disagree with daily DO inputs')
            post_uncertain = bool(uncertain.loc[end + one_day:end + 9 * one_day].any())
            response = event.to_dict()
            response.update(
                Hydro_Site_ID=gauge, Event_Retained=retained, Event_Mean_DO=mean_do,
                Original_Shock_L=event['Shock-L'], Original_Shock_S=event['Shock-S'],
                Original_Slope=event.Recovery_Slope,
                Original_Seasonal_Reference=original_l, Sensitivity_Seasonal_Reference=revised_l,
                Original_Pre_Valid_Days=len(before), Sensitivity_Pre_Valid_Days=len(after),
                Post_Days_1_to_9_Uncertain=post_uncertain,
                Original_Q_Missing_Event_Days=int(q.loc[start:end].Q_Original_Missing.sum()),
                Uncertain_Component_Event_Days=int(uncertain.loc[start:end].sum()),
                Changing_Q_Assignments='|'.join(changed),
            )
            response['Shock-L'] = mean_do / revised_l if retained and revised_l > 0 else np.nan
            response['Shock-S'] = mean_do / after.mean() if retained and len(after) >= 3 and after.mean() > 0 else np.nan
            response['Recovery_Slope'] = event.Recovery_Slope if retained and not post_uncertain else np.nan
            event_audit.append(response)

    audit = pd.DataFrame(event_audit)
    screened = audit[audit.Event_Retained]
    ref_audit = pd.concat(observed_reference, ignore_index=True)
    summary = pd.DataFrame([summarize(events, 'Original'), summarize(screened, 'Unknown_Q_screen')])
    print(summary.to_string(index=False), flush=True)
    print(f'Removed reference days: {len(ref_audit)}', flush=True)
    contrasts, diagnostics = model.fit_prespecified_contrasts(screened)
    if not contrasts.Fit_Status.eq('Reliable').all():
        raise ValueError('Unreliable Q exposure sensitivity contrasts')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in {
        'Q_Unknown_Event_Audit.csv': audit,
        'Q_Unknown_Reference_Day_Audit.csv': ref_audit,
        'Q_Unknown_Sample_Summary.csv': summary,
        'Q_Unknown_Core_Contrasts.csv': contrasts,
        'Q_Unknown_Model_Diagnostics.csv': diagnostics,
    }.items():
        table.to_csv(args.output_dir / filename, index=False, date_format='%Y-%m-%d')


if __name__ == '__main__':
    main()
