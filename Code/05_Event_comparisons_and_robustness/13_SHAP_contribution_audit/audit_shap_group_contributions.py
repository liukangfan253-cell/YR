"""Audit saved SHAP feature and group percentages and explanatory sample coverage.

This check uses saved outputs and leaves the model, SHAP values and manuscript files intact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis'))
from analysis_config import read_analysis_config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'Result/Statistical_comparisons/SHAP_group_descriptive_audit')
    args = parser.parse_args()
    rows, checks, sources = [], [], {}
    for tag, event in [('H','Heatwave'), ('F','Flood'), ('DC','DC'), ('H5F','H5F'), ('F5H','F5H')]:
        script = ROOT/f'Code/05_Event_comparisons_and_robustness/11_Tree_model_analysis/analyze_{tag}_shap.py'
        cfg = read_analysis_config(script, ROOT)
        data_path = Path(cfg['DATA_FILE'])
        if not data_path.is_absolute(): data_path = ROOT/data_path
        data = pd.read_csv(data_path).dropna(subset=['DO']+cfg['FEATURE_COLS'])
        events = data[['WQS_name','Event_ID']].drop_duplicates()
        stem = ROOT / cfg['OUTPUT_DIR'] / cfg['OUTPUT_BASENAME']
        paths = {name:Path(str(stem)+suffix) for name,suffix in [
            ('feature','_Feature_Importance.csv'), ('group','_Group_Contribution.csv'),
            ('log','_Log.csv'), ('comparison','_Model_Comparison.csv')]}
        feature = pd.read_csv(paths['feature']); group = pd.read_csv(paths['group'])
        log = dict(pd.read_csv(paths['log']).itertuples(index=False, name=None))
        if len(data) != int(log['Cleaned_Sample_Size']): raise ValueError(f'Sample mismatch: {event}')
        for name, features in cfg['GROUP_FEATURES'].items():
            selected = feature[feature.Group == name]
            percent = float(group.loc[group.Group == name,'SHAP_percent'].iloc[0])
            if set(selected.Feature) != set(features) or abs(percent-selected.SHAP_percent.sum()) >= 2e-5:
                raise ValueError(f'Group composition/contribution mismatch: {event}, {name}')
            rows.append(dict(Event=event, Model=log['Selected_Model'], Records=len(data),
                             Events=len(events), Sites=data.WQS_name.nunique(), Group=name,
                             Feature_Count=len(features), Features=', '.join(features),
                             Group_Percent=percent, Per_Feature_Mean_Percent=percent/len(features)))
        if abs(group.SHAP_percent.sum()-100) > 0.001: raise ValueError(f'Percent total mismatch: {event}')
        checks.append(dict(Event=event, Records=len(data), Events=len(events), Sites=data.WQS_name.nunique(),
                           Model=log['Selected_Model'], Group_Sum=float(group.SHAP_percent.sum())))
        for path in [script, data_path, *paths.values()]:
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'SHAP_Group_Descriptive_Audit.csv', index=False, encoding='utf-8-sig')
    report = dict(Status='PASS', Sources=sources, Checks=checks, New_inference=False,
                  Group_definition='Sum of feature mean absolute SHAP contributions; percentages normalized within model.',
                  Per_feature_definition='Group percentage divided by number of features; percentage points.',
                  Sample_counts='Explanatory coverage, not inferential sample sizes.')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(Status='PASS', Event_types=len(checks), Group_rows=len(rows), New_inference=False), indent=2))

if __name__ == '__main__':
    main()
