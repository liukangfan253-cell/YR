"""Rebuild the CSVs for Tables S28–S30 and Supplementary Data S6–S7."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
THRESHOLDS = (0, 25, 50, 75, 100)
KEY = ['Site_ID', 'Event_ID']


def event_membership():
    metrics = pd.read_csv(ROOT / 'Result/Decrease/DO_observation_sensitivity/20261111/20261111_M07事件绝对DO指标.csv')
    selections = pd.read_csv(ROOT / 'Result/Post_event_eligibility/20261111_Analysis/event_selection_and_intensity.csv')
    source = ['Duration_Days', 'Observed_DO_Days', 'Imputed_DO_Days', 'Event_DO_Valid_Days',
              'Event_DO_Mean', 'Event_DO_Min', 'Any_DO_Below5', 'Any_DO_Below3', 'Any_DO_Below2']
    metrics = metrics.rename(columns={'Event_Duration_Days': 'Duration_Days'})
    d = selections[KEY + ['Event_Type', 'Reach', 'Start_Date', 'End_Date', 'Season', 'Compound_Group',
                          'Post_All_Five_Observed', 'Post_Observed_Points',
                          'Shock-L', 'Shock-S', 'Recovery_Slope']].merge(
        metrics[KEY + source], on=KEY, how='outer', validate='one_to_one', indicator=True)
    if not d._merge.eq('both').all():
        raise ValueError('Baseline membership differs between event coverage and slope eligibility')
    d = d.drop(columns='_merge')
    d['Observed_Coverage_Pct'] = d.Observed_DO_Days * 100 / d.Duration_Days
    d['Any_event_observation'] = d.Observed_DO_Days.gt(0)
    d['Every_event_day_observed'] = d.Observed_DO_Days.eq(d.Duration_Days)
    d['Coverage_Class'] = np.select(
        [d.Every_event_day_observed, d.Any_event_observation],
        ['Fully_observed', 'Partially_observed'], default='Entirely_reconstructed')
    for source, label in [('Shock-L', 'Shock_L'), ('Shock-S', 'Shock_S')]:
        d[f'{label}_Eligible'] = np.isfinite(d[source]) & d[source].gt(0)
    d['Paired_Eligible'] = d.Shock_L_Eligible & d.Shock_S_Eligible
    l, s = d['Shock-L'], d['Shock-S']
    d['Paired_Neutral'] = d.Paired_Eligible & (l.eq(1) | s.eq(1))
    d['Paired_Discordant'] = d.Paired_Eligible & ((l - 1) * (s - 1)).lt(0)
    d['Paired_Discordant_Beyond_5pct'] = d.Paired_Eligible & (
        (l.lt(.95) & s.gt(1.05)) | (s.lt(.95) & l.gt(1.05)))
    d['Slope_Eligible'] = np.isfinite(d.Recovery_Slope)
    main_columns = KEY + ['Event_Type', 'Reach', 'Start_Date', 'End_Date', 'Season', 'Compound_Group',
        'Duration_Days', 'Observed_DO_Days', 'Imputed_DO_Days', 'Event_DO_Valid_Days', 'Event_DO_Mean',
        'Event_DO_Min', 'Any_DO_Below5', 'Any_DO_Below3', 'Any_DO_Below2', 'Post_All_Five_Observed',
        'Post_Observed_Points', 'Shock-L', 'Shock-S', 'Recovery_Slope', 'Observed_Coverage_Pct',
        'Any_event_observation', 'Every_event_day_observed', 'Coverage_Class', 'Shock_L_Eligible',
        'Shock_S_Eligible', 'Paired_Eligible', 'Paired_Neutral', 'Paired_Discordant',
        'Paired_Discordant_Beyond_5pct', 'Slope_Eligible']
    return d[main_columns].sort_values(KEY).reset_index(drop=True)


def pct(count, total):
    return 100 * count / total if total else np.nan


def summarize(d):
    pair_count = int(d.Paired_Eligible.sum())
    neutral = int(d.Paired_Neutral.sum())
    discordant = int(d.Paired_Discordant.sum())
    beyond_band = int(d.Paired_Discordant_Beyond_5pct.sum())
    positive = int((d.Slope_Eligible & d.Recovery_Slope.gt(0)).sum())
    out = dict(Events=len(d), Sites=d.Site_ID.nunique(), Median_Duration_Days=d.Duration_Days.median(),
               Paired_Events=pair_count, Neutral_Pairs=neutral, Discordant_Pairs=discordant,
               Discordant_Pct_All_Pairs=pct(discordant, pair_count),
               Discordant_Pct_Nonneutral_Pairs=pct(discordant, pair_count - neutral),
               Discordant_Beyond_5pct_Pairs=beyond_band,
               Discordant_Beyond_5pct_Pct_All_Pairs=pct(beyond_band, pair_count),
               Event_Mean_DO_Mean_mg_L=d.Event_DO_Mean.mean(),
               Event_Minimum_DO_Mean_mg_L=d.Event_DO_Min.mean(),
               Positive_Slopes=positive, Positive_Slopes_Pct=pct(positive, d.Slope_Eligible.sum()),
               Slope_Windows_All_Five_Post_Days_Observed=int((d.Slope_Eligible & d.Post_All_Five_Observed).sum()))
    for cutoff in (5, 3, 2):
        count = int(d[f'Any_DO_Below{cutoff}'].sum())
        out[f'Events_DO_Below_{cutoff}_mg_L'] = count
        out[f'Events_DO_Below_{cutoff}_mg_L_Pct'] = pct(count, len(d))
    for label, column, valid in [('Shock_L', 'Shock-L', 'Shock_L_Eligible'),
                                 ('Shock_S', 'Shock-S', 'Shock_S_Eligible'),
                                 ('Early_Slope', 'Recovery_Slope', 'Slope_Eligible')]:
        eligible = d.loc[d[valid]]
        values = eligible[column]
        stats = dict(N=len(values), Mean=values.mean(), Median=values.median(),
                     Q25=values.quantile(.25), Q75=values.quantile(.75),
                     Site_Equal_Mean=eligible.groupby('Site_ID')[column].mean().mean())
        out.update({f'{label}_{k}': v for k, v in stats.items()})
    return out


def strata(d, source, **labels):
    for reach in ['All', *sorted(source.Reach.unique())]:
        reach_data = d if reach == 'All' else d[d.Reach.eq(reach)]
        for event_type in ['All', *sorted(source.Event_Type.unique())]:
            subset = reach_data if event_type == 'All' else reach_data[reach_data.Event_Type.eq(event_type)]
            yield dict(**labels, Reach=reach, Event_Type=event_type, **summarize(subset))


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'Result/Observation_coverage/20261111')
    args = parser.parse_args()
    d = event_membership()
    threshold_members = d.drop(columns=['Any_event_observation', 'Every_event_day_observed']).copy()
    threshold_rows = []
    for cutoff in THRESHOLDS:
        mask = d.Observed_Coverage_Pct.ge(cutoff)
        threshold_members[f'Eligible_Coverage_GE_{cutoff}pct'] = mask
        threshold_rows.extend(strata(d[mask], d, Minimum_Observed_Coverage_Pct=cutoff))
    nested = {'All_eligible': d, 'Any_event_observation': d[d.Any_event_observation],
              'Every_event_day_observed': d[d.Every_event_day_observed]}
    population_rows = []
    for name, cohort in nested.items():
        population_rows.extend(strata(cohort, d, Grouping='Nested', Population=name))
    for name in ('Fully_observed', 'Partially_observed', 'Entirely_reconstructed'):
        population_rows.extend(strata(
            d[d.Coverage_Class.eq(name)], d, Grouping='Mutually_exclusive', Population=name))

    model = load_script('Code/05_Event_comparisons_and_robustness/01_Mixed_effects/compare_event_do_responses.py',
                        'coverage_models')
    paired_model = load_script('Code/04_DO_conditions_and_post_event_changes/06_Paired_references/M04_paired_baselines.py',
                               'coverage_bootstrap')
    results, diagnostics, intervals = [], [], []
    for name, subset in nested.items():
        print(f'Fitting coverage cohort {name}: {len(subset)} events', flush=True)
        pairs = subset[subset.Paired_Eligible].assign(
            Direction_Class=np.where(subset.loc[subset.Paired_Eligible, 'Paired_Neutral'],
                                     'Neutral', 'Classifiable'),
            Direction_Flip=lambda x: x.Paired_Discordant)
        interval, _ = paired_model.bootstrap(pairs)
        interval = interval.drop(columns=['Group', 'Status', 'Method'])
        interval.insert(0, 'Cohort', name)
        intervals.append(interval)
        estimates, checks = model.fit_prespecified_contrasts(subset)
        if not estimates.Fit_Status.eq('Reliable').all():
            raise ValueError(f'{name}: unreliable prespecified estimates')
        estimates.insert(0, 'Cohort', name)
        checks.insert(0, 'Cohort', name)
        results.append(estimates)
        diagnostics.append(checks)

    for subdir, tables in {
        'S6': {'coverage_threshold_event_eligibility.csv': threshold_members,
               'coverage_threshold_stratified_summary.csv': pd.DataFrame(threshold_rows)},
        'S7': {'coverage_population_event_membership.csv': d,
               'coverage_population_response_summary.csv': pd.DataFrame(population_rows),
               'coverage_population_mixed_model_contrasts.csv': pd.concat(results, ignore_index=True),
               'coverage_population_mixed_model_diagnostics.csv': pd.concat(diagnostics, ignore_index=True)},
    }.items():
        directory = args.output_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(directory / name, index=False)
    pd.concat(intervals, ignore_index=True).to_csv(
        args.output_dir / 'coverage_population_discordance_intervals.csv', index=False)


if __name__ == '__main__':
    main()
