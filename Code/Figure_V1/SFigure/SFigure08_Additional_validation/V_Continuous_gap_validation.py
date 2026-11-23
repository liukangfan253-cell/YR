"""Plot saved pointwise and continuous-gap validation results; no model inference.

Run: python Code/Figure_V1/SFigure/SFigure08_Additional_validation/V_Continuous_gap_validation.py
Requires numpy, pandas and matplotlib. --output-dir selects a fresh export directory.
All plotted observations come from saved validation summaries, not simulated data.
"""
from __future__ import annotations
import html
import math
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from io import BytesIO
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, str(ROOT/'Code/03_Daily_DO_reconstruction/04_Continuous_gap_validation'))
from validation_labels import normalize_validation_labels
from reconstruction_diagnostic_inputs import (
    POINTWISE_GROUPS, add_pointwise_arguments, read_pointwise, summarize_test_points, test_predictions,
)
from validate_continuous_do_gaps import DESIGNS, metrics as error_metrics, require, sha

VALIDATION_ROOT = ROOT/'Result/Continuous_gap_validation'
BLUE, ORANGE, GRAY = '#447F9A', '#C87953', '#939DA4'


def completed_validation(directory):
    """Recognize complete result tables without a Markdown or JSON marker."""
    required = (
        'candidate_availability', 'selected_windows', 'masked_point_predictions',
        'window_metrics', 'metric_summary', 'low_DO_classification', 'site_metrics',
        'gap_depth_metrics', 'site_block_intervals', 'paired_baseline_comparison',
        'reload_prediction_checks',
    )
    if (directory/'error_traceback.txt').exists() or not all(
            (directory/(name+'.csv')).is_file() for name in required):
        return False
    try:
        checks = pd.read_csv(directory/'reload_prediction_checks.csv')
        selected = pd.read_csv(directory/'selected_windows.csv')
        windows = pd.read_csv(directory/'window_metrics.csv')
        inventory = pd.read_csv(directory/'candidate_availability.csv')
        points = pd.read_csv(directory/'masked_point_predictions.csv', usecols=['Window_ID'])
        counts = points.groupby('Window_ID').size()
        return bool(
            len(checks) and checks.Passed.astype(str).str.lower().eq('true').all()
            and set(checks.Site_ID) == set(inventory.Site_ID)
            and len(selected) and selected.Window_ID.is_unique and windows.Window_ID.is_unique
            and set(selected.Window_ID) == set(windows.Window_ID) == set(counts.index)
            and inventory.Selected_Windows.sum() == len(selected)
            and inventory.Selected_Days.sum() == len(points)
            and counts.reindex(selected.Window_ID).to_numpy().tolist() == selected.Length.tolist()
        )
    except (OSError, ValueError, KeyError, AttributeError):
        return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validation-dir', type=Path)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--pointwise-csv', type=Path, help='Use a prepared pointwise summary instead of aggregating saved predictions')
    ap.add_argument('--site', default='Jiagaiqiao', help='Station to highlight in the pointwise panel')
    ap.add_argument('--site-scatter-csv', type=Path, help='Explicit saved prediction CSV for the highlighted station')
    for option in ('metrics', 'intervals', 'paired', 'low-do'):
        ap.add_argument(f'--{option}-csv', type=Path, help=f'Override the {option} CSV from the validation directory')
    add_pointwise_arguments(ap)
    args=ap.parse_args()
    validation = args.validation_dir
    table_overrides = (args.metrics_csv, args.intervals_csv, args.paired_csv, args.low_do_csv)
    if validation is None and not (all(table_overrides) and args.output_dir):
        completed = sorted(p for p in VALIDATION_ROOT.glob('V1_Continuous_Gap_*_Validation')
                           if completed_validation(p))
        if not completed:
            raise FileNotFoundError(f'No completed validation in {VALIDATION_ROOT}')
        validation = completed[-1]
    out = (args.output_dir or ROOT/'V/Figure_V1/SFigure/SFigure08_Additional_validation/Images'/validation.name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source={
        'metrics': args.metrics_csv or validation/'metric_summary.csv',
        'intervals': args.intervals_csv or validation/'site_block_intervals.csv',
        'paired': args.paired_csv or validation/'paired_baseline_comparison.csv',
        'low_do': args.low_do_csv or validation/'low_DO_classification.csv',
    }
    if args.pointwise_csv:
        source['pointwise'] = args.pointwise_csv
    if args.site_scatter_csv:
        source['site'] = args.site_scatter_csv
    elif args.pointwise_csv:
        matches = list(args.scatter_dir.rglob(f'{args.site}_Labeled_Scatter.csv'))
        require(len(matches) == 1, f'Expected one saved scatter file for {args.site}. Use --site-scatter-csv to specify it.')
        source['site'] = matches[0]
    hashes={str(p.resolve()):sha(p) for p in source.values()}
    if args.pointwise_csv:
        point = pd.read_csv(args.pointwise_csv)
    else:
        predictions, _, point_sources = read_pointwise(args)
        hashes.update(point_sources)
        point = summarize_test_points(predictions)
        if validation is not None and not any(table_overrides):
            checks_path = validation/'reload_prediction_checks.csv'
            if checks_path.is_file():
                hashes[str(checks_path.resolve())] = sha(checks_path)
                checks = pd.read_csv(checks_path)
                by_name = {Path(path).name: digest for path, digest in point_sources.items()
                           if path.endswith('_Labeled_Scatter.csv')}
                require(all(by_name.get(Path(row.Reference_File.replace('\\','/')).name) == row.Reference_SHA256
                            for row in checks.itertuples()),
                        'Saved predictions do not match this gap validation. Choose a matching --validation-dir or --scatter-dir.')
    if 'Subset' in point:
        point = point.loc[point.Subset.eq('All_points')]
    point = point.loc[point.Group.isin(POINTWISE_GROUPS)].set_index('Group')
    require(point.index.is_unique and set(point.index) == set(POINTWISE_GROUPS), 'Expected one row each for All, Event and NonEvent')
    point = point.loc[POINTWISE_GROUPS].reset_index()
    require((point.N >= 0).all() and point.N.iloc[0] == point.N.iloc[1:].sum(), 'Inconsistent pointwise sample counts')
    site = test_predictions(source['site']) if 'site' in source else predictions.loc[predictions.Site_ID.eq(args.site)]
    require(len(site) > 0, f'No test predictions for {args.site}')
    site_row = pd.DataFrame([dict(Group=args.site, **error_metrics(site['Observed DO'], site['Simulated DO']))])
    point=pd.concat([point[['Group','N','MAE','RMSE','Bias']],site_row],ignore_index=True)
    require(np.isfinite(point.loc[point.N.gt(0), ['MAE','RMSE','Bias']]).to_numpy().all() and point.MAE.dropna().ge(0).all(),
            'Invalid pointwise error metrics')
    metrics=normalize_validation_labels(pd.read_csv(source['metrics']))
    masked=metrics[(metrics.Context=='Event') & (metrics.Method=='Masked_BiLSTM_Attention')].copy()
    intervals=normalize_validation_labels(pd.read_csv(source['intervals']))
    intervals=intervals[(intervals.Context=='Event') & (intervals.Method=='Masked_BiLSTM_Attention')]
    masked=masked.merge(intervals.reindex(columns=['Design','Length','MAE_CI_Low','MAE_CI_High']),on=['Design','Length'],validate='one_to_one')
    paired=normalize_validation_labels(pd.read_csv(source['paired']));paired=paired[paired.Context=='Event'].copy()
    low=normalize_validation_labels(pd.read_csv(source['low_do']))
    low=low[(low.Context=='Event') & (low.Method=='Masked_BiLSTM_Attention')].copy()
    lengths = sorted(masked.Length.unique())
    thresholds = sorted(low.Threshold.unique(), reverse=True)
    require(bool(lengths) and bool(thresholds), 'Missing event-date gap lengths or low-DO thresholds')
    for frame, columns, levels in (
        (masked, ['Design','Length'], [DESIGNS,lengths]),
        (paired, ['Design','Length','Baseline'], [DESIGNS,lengths,['Unmasked_BiLSTM_Attention','Last_observation']]),
        (low, ['Design','Length','Threshold'], [DESIGNS,lengths,thresholds]),
    ):
        keys = pd.MultiIndex.from_frame(frame[columns])
        require(keys.is_unique and set(keys) == set(pd.MultiIndex.from_product(levels)), f'Incomplete or duplicate gap results for {columns}')
    require((low.TP + low.FN).eq(low.Observed_Low).all() and
            np.allclose(low.Recall, low.TP / low.Observed_Low, equal_nan=True), 'Inconsistent low-DO recall counts')
    longest = masked.loc[masked.Length.eq(lengths[-1])]
    require(longest.N.nunique() == 1 and np.allclose(longest.MAE, longest.MAE.iloc[0], equal_nan=True),
            'The two designs disagree at their shared longest gap length')
    ticks = np.arange(len(lengths))
    for name, frame in [('source_pointwise.csv', point), ('source_masked_event_errors.csv', masked),
                        ('source_paired_event_comparisons.csv', paired), ('source_masked_event_low_DO.csv', low)]:
        target = out / name
        if str(target.resolve()) not in hashes:
            frame.to_csv(target,index=False,encoding='utf-8-sig')

    # Figure contract: pointwise errors do not establish long-gap accuracy;
    # masking exposes larger event-date errors and limited low-DO detection.
    # Roles: a pointwise reference; b masking stress test; c same-date controls;
    # d low-concentration failure boundary. Quantitative 2x2 grid, 183 x 168 mm.
    # b uses saved 95% site-block percentile intervals (1,000 resamples).
    # a/c/d are descriptive saved aggregates/counts, not independent repeated trials.
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(183/25.4,168/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.91,wspace=.44,hspace=.74)
    for label,ax in zip('abcd',axes.flat):
        ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-29,15),
            textcoords='offset points',fontsize=11,fontweight='bold',va='bottom')
        ax.tick_params(direction='out',length=3)

    ax=axes[0,0]
    positions=np.arange(4)
    ax.bar(positions,point.MAE,width=.58,color=[GRAY,BLUE,GRAY,ORANGE])
    ax.set(xticks=positions,xticklabels=['All','Event','Non-event',args.site],
        ylabel='MAE (mg/L)',ylim=(0,max(1.02, point.MAE.max()*1.16)),title='Saved pointwise tests')
    for x,row in zip(positions,point.itertuples()):
        label = f'{row.MAE:.3f}\nn={int(row.N):,}' if row.N else 'n=0'
        ax.text(x,(row.MAE if row.N else 0)+.025,label,ha='center',va='bottom',fontsize=7)

    ax=axes[0,1]
    for design,color,marker,offset in zip(DESIGNS,[BLUE,ORANGE],['o','s'],[-.06,.06]):
        d=masked[masked.Design==design].set_index('Length').loc[lengths]
        xx=ticks+offset
        ax.plot(xx,d.MAE,color=color,marker=marker,markersize=4,lw=1.2,label='L' if design==DESIGNS[0] else 'C')
        for x, lo, hi in zip(xx, d.MAE_CI_Low, d.MAE_CI_High):
            if np.isfinite([lo,hi]).all():
                ax.vlines(x, lo, hi, color=color, linewidth=1.2)
                ax.plot([x,x], [lo,hi], linestyle='none', marker='_', markersize=4.6, color=color)
    values = masked[['MAE','MAE_CI_Low','MAE_CI_High']]
    ax.set(xticks=ticks,xticklabels=lengths,xlabel='Masked gap length (days)',
        ylabel='Event-date MAE (mg/L)',ylim=(min(.3,values.min().min()*.95),max(1.15,values.max().max()*1.05)),title='Continuous-gap stress test')
    ax.legend(loc='upper left',ncol=2,fontsize=7.5,handlelength=1.2)

    ax=axes[1,0]
    for design,ls,marker in zip(DESIGNS,['-','--'],['o','s']):
        for base,color,name in [('Unmasked_BiLSTM_Attention',GRAY,'Unmasked'),('Last_observation',BLUE,'Persistence')]:
            d=paired[(paired.Design==design)&(paired.Baseline==base)].set_index('Length').loc[lengths]
            ax.plot(ticks,d.Masked_minus_Baseline_MAE,color=color,ls=ls,marker=marker,
                markersize=4,lw=1.2,label=f'{"L" if design==DESIGNS[0] else "C"}: {name}')
    ax.axhline(0,color='#B6BEC2',lw=.7,zorder=0)
    delta = paired.Masked_minus_Baseline_MAE
    ax.set(xticks=ticks,xticklabels=lengths,xlabel='Masked gap length (days)',
        ylabel='Masked − baseline MAE (mg/L)',ylim=(min(-.22,delta.min()*1.1),max(.52,delta.max()*1.1)),title='Same-date baseline comparisons')
    ax.legend(loc='upper left',fontsize=7,ncol=2,columnspacing=.9,handlelength=1.6)

    ax=axes[1,1]
    rows=[];matrix=[];labels=[]
    for design, code in zip(DESIGNS, ['L','C']):
        for threshold in thresholds:
            d=low[(low.Design==design)&(low.Threshold==threshold)].set_index('Length').loc[lengths]
            rows.append(d);matrix.append(d.Recall.to_numpy()*100);labels.append(f'{code}: <{threshold:g}')
    matrix=np.asarray(matrix)
    cmap=LinearSegmentedColormap.from_list('Recall_blue',['#F3F6F8','#AFD0DD','#447F9A'])
    ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=0,vmax=100,interpolation='nearest')
    ax.set(xticks=ticks,xticklabels=lengths,xlabel='Masked gap length (days)',
        yticks=np.arange(len(rows)),yticklabels=labels,
        title='Low-DO recall (%) and TP / low n')
    for i,d in enumerate(rows):
        for j,length in enumerate(lengths):
            row=d.loc[length]
            recall = f'{100*row.Recall:.1f}%' if row.Observed_Low else '—'
            ax.text(j,i,f'{recall}\n{int(row.TP)}/{int(row.Observed_Low)}',
                ha='center',va='center',fontsize=7,color='#102D3A')
    ax.set_xticks(np.arange(len(lengths)+1)-.5,minor=True)
    ax.set_yticks(np.arange(len(rows)+1)-.5,minor=True)
    ax.grid(which='minor',color='white',lw=1.5)
    ax.tick_params(which='minor',length=0)
    for sp in ax.spines.values():sp.set_visible(False)

    stem=out/'Figure_S5_DO_validation_summary'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png',dpi=600)
    with BytesIO() as tiff_buffer:
        fig.savefig(tiff_buffer, format="tiff",dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        Path(str(stem)+'.tiff').write_bytes(tiff_buffer.getbuffer())
    plt.close(fig)
    require(all(sha(path)==digest for path,digest in hashes.items()), 'Figure input changed during export')
    provenance={'Sources':hashes,
        'Backend':'Python matplotlib','Model_training':False,'Inference':False,
        'Figure_scope':'Pointwise errors, event-date masking, paired controls and saved low-DO cutoffs',
        'Test_year':args.test_year,'Highlighted_site':args.site,
        'Source_rows':{'Pointwise_groups':len(point),'Masked_event_groups':len(masked),'Paired_event_groups':len(paired),'Low_DO_groups':len(low)},
        'Other_contexts':'All/non-event masking results remain in Table S11(a) and source files; plots b–d focus on event dates.',
        'Uncertainty':'Panel b shows saved site-block percentile intervals where available.',
        'Shared_records':f'L/C share {lengths[-1]}-day targets. Common-anchor shorter windows use the same starts.',
        'Physical_size_mm':[183,168]}
    print('Figure and source checks PASS:',stem)


# Matplotlib panel alignment and figure export checks.
SCHEMA_VERSION = 1


DEFAULT_TOLERANCE_PT = 1.5


DEFAULT_GUTTER_TOLERANCE_PT = 1.5


class PanelAlignmentError(RuntimeError):
    """Raised when a required in-memory alignment gate does not pass."""


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _bbox(value: Any) -> tuple[float, float, float, float]:
    if isinstance(value, Mapping):
        values = [value.get(key) for key in ("left", "bottom", "right", "top")]
    else:
        values = list(value) if isinstance(value, Sequence) else []
    if len(values) != 4:
        raise ValueError("panel bbox_pt must contain left, bottom, right and top")
    bbox = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError("panel bbox_pt values must be finite")
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("panel bbox_pt must have positive width and height")
    return bbox


def _group_rows(raw: Any, prefix: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    if raw is None:
        return groups
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError(f"{prefix}_groups must be a list")
    for index, item in enumerate(raw, 1):
        if isinstance(item, Mapping):
            panels = item.get("panels", [])
            group_id = str(item.get("id") or f"{prefix}-{index}")
        else:
            panels = item
            group_id = f"{prefix}-{index}"
        if not isinstance(panels, Sequence) or isinstance(panels, (str, bytes)):
            raise ValueError(f"{prefix} group {group_id} must list panel ids")
        panel_ids = [str(panel) for panel in panels]
        if len(panel_ids) < 2:
            raise ValueError(f"{prefix} group {group_id} needs at least two panels")
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError(f"{prefix} group {group_id} repeats a panel id")
        groups.append({"id": group_id, "panels": panel_ids})
    return groups


def _inferred_groups(panels: Sequence[dict[str, Any]], orientation: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for panel in panels:
        if orientation == "row":
            fields = (panel.get("grid_id"), panel.get("row_start"), panel.get("row_stop"))
        else:
            fields = (panel.get("grid_id"), panel.get("col_start"), panel.get("col_stop"))
        if any(value is None for value in fields):
            continue
        grouped[fields].append(str(panel["id"]))
    output: list[dict[str, Any]] = []
    for index, (_key, panel_ids) in enumerate(grouped.items(), 1):
        if len(panel_ids) >= 2:
            output.append({"id": f"inferred-{orientation}-{index}", "panels": panel_ids})
    return output


def _inferred_boundary_groups(panels: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infer shared grid boundaries across panels with unequal spans.

    This covers layouts such as two stacked panels beside one panel that spans
    both rows. Equal-span panels are handled by the ordinary row/column groups,
    so boundary groups are emitted only when at least two different spans meet
    at the same grid boundary.
    """
    specifications = (
        ("top", "row_start", "row_start", "row_stop"),
        ("bottom", "row_stop", "row_start", "row_stop"),
        ("left", "col_start", "col_start", "col_stop"),
        ("right", "col_stop", "col_start", "col_stop"),
    )
    output: list[dict[str, Any]] = []
    for edge, boundary_field, span_start, span_stop in specifications:
        grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for panel in panels:
            grid_id = panel.get("grid_id")
            boundary = panel.get(boundary_field)
            start = panel.get(span_start)
            stop = panel.get(span_stop)
            if any(value is None for value in (grid_id, boundary, start, stop)):
                continue
            grouped[(grid_id, boundary)].append(panel)
        edge_index = 0
        for (_grid_id, _boundary), members in grouped.items():
            spans = {(panel.get(span_start), panel.get(span_stop)) for panel in members}
            if len(members) < 2 or len(spans) < 2:
                continue
            edge_index += 1
            output.append(
                {
                    "id": f"inferred-shared-{edge}-{edge_index}",
                    "edge": edge,
                    "panels": [str(panel["id"]) for panel in members],
                }
            )
    return output


def _validated_layout(manifest: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    figure = manifest.get("figure")
    if not isinstance(figure, Mapping):
        errors.append("figure must be an object with width_pt and height_pt")
        figure = {}
    try:
        width_pt = float(figure.get("width_pt"))
        height_pt = float(figure.get("height_pt"))
        if not math.isfinite(width_pt) or not math.isfinite(height_pt) or width_pt <= 0 or height_pt <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("figure width_pt and height_pt must be positive finite numbers")
        width_pt = height_pt = 0.0

    raw_panels = manifest.get("panels")
    panels: list[dict[str, Any]] = []
    if not isinstance(raw_panels, Sequence) or isinstance(raw_panels, (str, bytes)):
        errors.append("panels must be a list")
        raw_panels = []
    for index, raw_panel in enumerate(raw_panels, 1):
        if not isinstance(raw_panel, Mapping):
            errors.append(f"panel {index} must be an object")
            continue
        panel_id = str(raw_panel.get("id") or "").strip()
        if not panel_id:
            errors.append(f"panel {index} is missing id")
            continue
        try:
            bbox = _bbox(raw_panel.get("bbox_pt"))
        except (TypeError, ValueError) as exc:
            errors.append(f"panel {panel_id}: {exc}")
            continue
        if bbox[0] < -0.01 or bbox[1] < -0.01 or bbox[2] > width_pt + 0.01 or bbox[3] > height_pt + 0.01:
            errors.append(f"panel {panel_id} bbox_pt extends beyond the declared figure")
        panel = dict(raw_panel)
        panel["id"] = panel_id
        panel["bbox_pt"] = list(bbox)
        anchor = raw_panel.get("panel_label_anchor_pt")
        if anchor is not None:
            try:
                anchor_values = [float(value) for value in anchor]
                if len(anchor_values) != 2 or not all(math.isfinite(value) for value in anchor_values):
                    raise ValueError
                panel["panel_label_anchor_pt"] = anchor_values
            except (TypeError, ValueError):
                errors.append(f"panel {panel_id} has invalid panel_label_anchor_pt")
        panels.append(panel)

    panel_ids = [panel["id"] for panel in panels]
    if len(panel_ids) != len(set(panel_ids)):
        errors.append("panel ids must be unique")
    if not panels:
        errors.append("at least one panel rectangle is required")

    try:
        row_groups = _group_rows(manifest.get("row_groups"), "row")
        column_groups = _group_rows(manifest.get("column_groups"), "column")
    except ValueError as exc:
        errors.append(str(exc))
        row_groups = column_groups = []
    if not row_groups:
        row_groups = _inferred_groups(panels, "row")
    if not column_groups:
        column_groups = _inferred_groups(panels, "column")
    boundary_groups = _inferred_boundary_groups(panels)

    known_ids = set(panel_ids)
    for group in [*row_groups, *column_groups]:
        missing = [panel for panel in group["panels"] if panel not in known_ids]
        if missing:
            errors.append(f"group {group['id']} references unknown panels: {', '.join(missing)}")

    exemptions: list[dict[str, Any]] = []
    raw_exemptions = manifest.get("exemptions", [])
    if not isinstance(raw_exemptions, Sequence) or isinstance(raw_exemptions, (str, bytes)):
        errors.append("exemptions must be a list")
        raw_exemptions = []
    for index, exemption in enumerate(raw_exemptions, 1):
        if not isinstance(exemption, Mapping):
            errors.append(f"exemption {index} must be an object")
            continue
        exemption_panels = exemption.get("panels", [])
        checks = exemption.get("checks", [])
        reason = str(exemption.get("reason") or "").strip()
        if isinstance(exemption_panels, str):
            exemption_panels = [exemption_panels]
        if isinstance(checks, str):
            checks = [checks]
        if not exemption_panels or any(str(panel) not in known_ids for panel in exemption_panels):
            errors.append(f"exemption {index} must reference known panels")
        allowed_checks = {
            "row",
            "column",
            "panel-width",
            "horizontal-gutter",
            "vertical-gutter",
            "panel-label",
            "all",
        }
        check_names = [str(check) for check in checks]
        if not check_names or any(check not in allowed_checks for check in check_names):
            errors.append(f"exemption {index} has invalid checks")
        if not reason:
            errors.append(f"exemption {index} needs a non-empty reason")
        exemptions.append(
            {"panels": [str(panel) for panel in exemption_panels], "checks": check_names, "reason": reason}
        )

    if errors:
        return None, errors
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": str(manifest.get("backend") or "unknown"),
        "figure": {"width_pt": width_pt, "height_pt": height_pt},
        "panels": panels,
        "row_groups": row_groups,
        "column_groups": column_groups,
        "boundary_groups": boundary_groups,
        "exemptions": exemptions,
    }, []


def _is_exempt(layout: Mapping[str, Any], panel_id: str, check: str) -> bool:
    for exemption in layout.get("exemptions", []):
        if panel_id in exemption["panels"] and (check in exemption["checks"] or "all" in exemption["checks"]):
            return True
    return False


def _metric_spread(values: Mapping[str, float]) -> float:
    return max(values.values()) - min(values.values()) if values else 0.0


def _finding(
    kind: str,
    group: Mapping[str, Any],
    panels: Sequence[str],
    tolerance_pt: float,
    metrics: Mapping[str, Mapping[str, float]],
    message: str,
    severity: str = "FAIL",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "kind": kind,
        "group": group["id"],
        "panels": list(panels),
        "message": message,
        "tolerance_pt": tolerance_pt,
        "metric_spreads_pt": {
            metric: round(_metric_spread(values), 6) for metric, values in metrics.items()
        },
        "values_pt": {
            metric: {panel: round(value, 6) for panel, value in values.items()}
            for metric, values in metrics.items()
        },
    }


def audit_layout_manifest(
    manifest: Mapping[str, Any],
    *,
    tolerance_pt: float = DEFAULT_TOLERANCE_PT,
    gutter_tolerance_pt: float = DEFAULT_GUTTER_TOLERANCE_PT,
    require_panel_labels: bool = False,
) -> dict[str, Any]:
    """Audit a backend-neutral panel-layout manifest."""
    if tolerance_pt < 0 or gutter_tolerance_pt < 0:
        raise ValueError("alignment tolerances must be non-negative")
    layout, errors = _validated_layout(manifest)
    if layout is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "auditable": False,
            "verdict": "NOT AUDITABLE",
            "summary": {"fail": 0, "warn": 0, "comparisons": 0, "exemptions": 0},
            "errors": errors,
            "findings": [],
        }

    if len(layout["panels"]) == 1:
        return {
            "schema_version": SCHEMA_VERSION,
            "applicable": False,
            "auditable": False,
            "verdict": "NOT APPLICABLE",
            "backend": layout["backend"],
            "summary": {"fail": 0, "warn": 0, "comparisons": 0, "exemptions": 0},
            "tolerances": {
                "alignment_pt": tolerance_pt,
                "gutter_pt": gutter_tolerance_pt,
            },
            "errors": [],
            "findings": [],
            "layout": layout,
        }

    panel_map = {panel["id"]: panel for panel in layout["panels"]}
    findings: list[dict[str, Any]] = []
    comparisons = 0

    for orientation, group_key, check_name in (
        ("row", "row_groups", "row"),
        ("column", "column_groups", "column"),
    ):
        for group in layout[group_key]:
            group_panel_ids = list(group["panels"])
            panel_ids = [
                panel_id for panel_id in group_panel_ids if not _is_exempt(layout, panel_id, check_name)
            ]
            if len(panel_ids) < 2:
                continue
            comparisons += 1
            boxes = {panel_id: _bbox(panel_map[panel_id]["bbox_pt"]) for panel_id in panel_ids}
            if orientation == "row":
                metrics = {
                    "top": {panel: box[3] for panel, box in boxes.items()},
                    "bottom": {panel: box[1] for panel, box in boxes.items()},
                    "height": {panel: box[3] - box[1] for panel, box in boxes.items()},
                }
                kind = "row-axes-misalignment"
                message = "Panels intended for one row do not share top/bottom edges and height"
            else:
                metrics = {
                    "left": {panel: box[0] for panel, box in boxes.items()},
                    "right": {panel: box[2] for panel, box in boxes.items()},
                    "width": {panel: box[2] - box[0] for panel, box in boxes.items()},
                }
                kind = "column-axes-misalignment"
                message = "Panels intended for one column do not share left/right edges and width"
            if max(_metric_spread(values) for values in metrics.values()) > tolerance_pt:
                findings.append(_finding(kind, group, panel_ids, tolerance_pt, metrics, message))

            if orientation == "row" and len(group_panel_ids) >= 3:
                width_candidates = [
                    panel_id
                    for panel_id in group_panel_ids
                    if not _is_exempt(layout, panel_id, "row")
                    and not _is_exempt(layout, panel_id, "panel-width")
                ]
                by_column_span: dict[Any, list[str]] = defaultdict(list)
                for panel_id in width_candidates:
                    panel = panel_map[panel_id]
                    try:
                        column_span = float(panel.get("col_stop")) - float(panel.get("col_start"))
                        if not math.isfinite(column_span) or column_span <= 0:
                            raise ValueError
                        span_key: Any = round(column_span, 9)
                    except (TypeError, ValueError):
                        span_key = "unspecified"
                    by_column_span[span_key].append(panel_id)
                for width_index, width_ids in enumerate(by_column_span.values(), 1):
                    if len(width_ids) < 2:
                        continue
                    comparisons += 1
                    width_metrics = {
                        "width": {
                            panel_id: _bbox(panel_map[panel_id]["bbox_pt"])[2]
                            - _bbox(panel_map[panel_id]["bbox_pt"])[0]
                            for panel_id in width_ids
                        }
                    }
                    if _metric_spread(width_metrics["width"]) > tolerance_pt:
                        findings.append(
                            _finding(
                                "horizontal-panel-width-misalignment",
                                {"id": f"{group['id']}-equal-span-{width_index}"},
                                width_ids,
                                tolerance_pt,
                                width_metrics,
                                "Three-or-more-panel row contains unequal final plot-area widths for equal grid spans",
                            )
                        )

            anchors = {
                panel_id: panel_map[panel_id].get("panel_label_anchor_pt") for panel_id in panel_ids
            }
            present_anchors = {panel: anchor for panel, anchor in anchors.items() if anchor is not None}
            missing_labels = [panel for panel, anchor in anchors.items() if anchor is None]
            if require_panel_labels and missing_labels:
                findings.append(
                    {
                        "severity": "WARN",
                        "kind": "panel-label-not-auditable",
                        "group": group["id"],
                        "panels": missing_labels,
                        "message": "Comparable panels are missing detectable top-left panel-label anchors",
                    }
                )
            if len(present_anchors) >= 2:
                label_check = "panel-label"
                label_ids = [
                    panel for panel in panel_ids if panel in present_anchors and not _is_exempt(layout, panel, label_check)
                ]
                if len(label_ids) >= 2:
                    coordinate = 1 if orientation == "row" else 0
                    metric_name = "label-y" if orientation == "row" else "label-x"
                    metrics = {
                        metric_name: {panel: float(present_anchors[panel][coordinate]) for panel in label_ids}
                    }
                    if _metric_spread(metrics[metric_name]) > tolerance_pt:
                        findings.append(
                            _finding(
                                f"{orientation}-panel-label-misalignment",
                                group,
                                label_ids,
                                tolerance_pt,
                                metrics,
                                "Panel-label anchors are not aligned within their comparable group",
                            )
                        )

            gutter_check = "horizontal-gutter" if orientation == "row" else "vertical-gutter"
            gutter_ids = [panel for panel in panel_ids if not _is_exempt(layout, panel, gutter_check)]
            if len(gutter_ids) >= 2:
                if orientation == "row":
                    ordered = sorted(gutter_ids, key=lambda panel: boxes[panel][0])
                    gaps = {
                        f"{first}->{second}": boxes[second][0] - boxes[first][2]
                        for first, second in zip(ordered, ordered[1:])
                    }
                else:
                    ordered = sorted(gutter_ids, key=lambda panel: boxes[panel][3], reverse=True)
                    gaps = {
                        f"{first}->{second}": boxes[first][1] - boxes[second][3]
                        for first, second in zip(ordered, ordered[1:])
                    }
                if gaps and min(gaps.values()) < -tolerance_pt:
                    findings.append(
                        _finding(
                            "panel-axes-overlap",
                            group,
                            gutter_ids,
                            tolerance_pt,
                            {"gutter": gaps},
                            "Panel plot-area rectangles overlap",
                        )
                    )
                elif len(gaps) >= 2 and _metric_spread(gaps) > gutter_tolerance_pt:
                    findings.append(
                        _finding(
                            f"{gutter_check}-misalignment",
                            group,
                            gutter_ids,
                            gutter_tolerance_pt,
                            {"gutter": gaps},
                            "Comparable inter-panel gutters are not uniform",
                        )
                    )

    edge_coordinates = {"left": 0, "bottom": 1, "right": 2, "top": 3}
    for group in layout["boundary_groups"]:
        edge = str(group["edge"])
        check_name = "row" if edge in {"top", "bottom"} else "column"
        panel_ids = [
            panel_id
            for panel_id in group["panels"]
            if not _is_exempt(layout, panel_id, check_name)
        ]
        if len(panel_ids) < 2:
            continue
        comparisons += 1
        coordinate = edge_coordinates[edge]
        metrics = {
            edge: {
                panel_id: _bbox(panel_map[panel_id]["bbox_pt"])[coordinate]
                for panel_id in panel_ids
            }
        }
        if _metric_spread(metrics[edge]) > tolerance_pt:
            findings.append(
                _finding(
                    f"shared-{edge}-edge-misalignment",
                    group,
                    panel_ids,
                    tolerance_pt,
                    metrics,
                    f"Panels meeting the same grid {edge} boundary are not aligned",
                )
            )

        if edge not in {"top", "left"}:
            continue
        anchors = {
            panel_id: panel_map[panel_id].get("panel_label_anchor_pt")
            for panel_id in panel_ids
        }
        missing_labels = [panel_id for panel_id, anchor in anchors.items() if anchor is None]
        if require_panel_labels and missing_labels:
            findings.append(
                {
                    "severity": "WARN",
                    "kind": "panel-label-not-auditable",
                    "group": group["id"],
                    "panels": missing_labels,
                    "message": "Comparable panels are missing detectable top-left panel-label anchors",
                }
            )
        label_ids = [
            panel_id
            for panel_id, anchor in anchors.items()
            if anchor is not None and not _is_exempt(layout, panel_id, "panel-label")
        ]
        if len(label_ids) < 2:
            continue
        label_coordinate = 1 if edge == "top" else 0
        metric_name = "label-y" if edge == "top" else "label-x"
        label_metrics = {
            metric_name: {
                panel_id: float(anchors[panel_id][label_coordinate])
                for panel_id in label_ids
            }
        }
        if _metric_spread(label_metrics[metric_name]) > tolerance_pt:
            findings.append(
                _finding(
                    f"shared-{edge}-panel-label-misalignment",
                    group,
                    label_ids,
                    tolerance_pt,
                    label_metrics,
                    "Panel-label anchors are not aligned at the shared grid boundary",
                )
            )
    fail_count = sum(finding["severity"] == "FAIL" for finding in findings)
    warn_count = sum(finding["severity"] == "WARN" for finding in findings)
    auditable = comparisons > 0
    verdict = "FIX BEFORE DELIVERY" if fail_count else "REVIEW REQUIRED" if warn_count else "PASS"
    if not auditable:
        verdict = "NOT AUDITABLE"
    return {
        "schema_version": SCHEMA_VERSION,
        "applicable": True,
        "auditable": auditable,
        "verdict": verdict,
        "backend": layout["backend"],
        "summary": {
            "fail": fail_count,
            "warn": warn_count,
            "comparisons": comparisons,
            "exemptions": len(layout["exemptions"]),
        },
        "tolerances": {
            "alignment_pt": tolerance_pt,
            "gutter_pt": gutter_tolerance_pt,
        },
        "layout": layout,
        "findings": findings,
        **(
            {"errors": ["No comparable row, column or shared-boundary groups were declared or inferred"]}
            if not auditable
            else {}
        ),
    }


def exit_code(report: Mapping[str, Any], strict: bool = False) -> int:
    if report.get("verdict") == "NOT APPLICABLE":
        return 0
    if not report.get("auditable") or report.get("verdict") == "NOT AUDITABLE":
        return 2
    if int(report.get("summary", {}).get("fail", 0)):
        return 1
    if strict and int(report.get("summary", {}).get("warn", 0)):
        return 1
    return 0


def render_text(report: Mapping[str, Any], strict: bool = False) -> str:
    summary = report.get("summary", {})
    lines = [
        f"Panel alignment: {report.get('verdict', 'UNKNOWN')}",
        f"  comparisons: {summary.get('comparisons', 0)}",
        f"  fail: {summary.get('fail', 0)}",
        f"  warn: {summary.get('warn', 0)}",
        f"  exemptions: {summary.get('exemptions', 0)}",
    ]
    for error in report.get("errors", []):
        lines.append(f"  ERROR: {error}")
    for finding in report.get("findings", []):
        lines.append(
            f"  {finding['severity']} {finding['kind']} [{finding.get('group', 'n/a')}]: "
            f"{finding['message']} ({', '.join(finding.get('panels', []))})"
        )
    if strict and summary.get("warn", 0):
        lines.append("  strict mode: WARN findings block delivery")
    return "\n".join(lines)


def write_json_report(report: Mapping[str, Any], path: str | Path) -> None:
    _atomic_write_text(Path(path), json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def write_overlay_svg(report: Mapping[str, Any], path: str | Path) -> None:
    layout = report.get("layout")
    if not isinstance(layout, Mapping):
        raise ValueError("an auditable layout is required for the diagnostic SVG")
    width = float(layout["figure"]["width_pt"])
    height = float(layout["figure"]["height_pt"])
    failed_panels = {
        panel for finding in report.get("findings", []) if finding.get("severity") == "FAIL"
        for panel in finding.get("panels", [])
    }
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for panel in layout["panels"]:
        left, bottom, right, top = _bbox(panel["bbox_pt"])
        y = height - top
        color = "#d7191c" if panel["id"] in failed_panels else "#2c7bb6"
        rows.append(
            f'<rect x="{left:.3f}" y="{y:.3f}" width="{right-left:.3f}" height="{top-bottom:.3f}" '
            f'fill="none" stroke="{color}" stroke-width="1.2"/>'
        )
        rows.append(
            f'<text x="{left+2:.3f}" y="{y+10:.3f}" font-family="Arial, sans-serif" font-size="8" '
            f'fill="{color}">{html.escape(str(panel["id"]))}</text>'
        )
    rows.append(
        f'<text x="4" y="{height-5:.3f}" font-family="Arial, sans-serif" font-size="7" fill="#333">'
        f'{report.get("verdict", "UNKNOWN")}: {report.get("summary", {}).get("fail", 0)} fail, '
        f'{report.get("summary", {}).get("warn", 0)} warn</text>'
    )
    rows.append("</svg>")
    _atomic_write_text(Path(path), "\n".join(rows) + "\n")


def _alphabetic_id(index: int) -> str:
    output = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        output = chr(ord("a") + remainder) + output
    return output


def _font_is_bold(weight: Any) -> bool:
    if isinstance(weight, (int, float)):
        return float(weight) >= 600
    return str(weight).lower() in {"bold", "semibold", "demibold", "heavy", "black"}


def _matplotlib_panel_label_anchor(ax: Any) -> tuple[str, list[float]] | None:
    for artist in ax.texts:
        label = artist.get_text().strip()
        if not re.fullmatch(r"[a-z]", label) or not _font_is_bold(artist.get_fontweight()):
            continue
        display = artist.get_transform().transform(artist.get_position())
        axes_xy = ax.transAxes.inverted().transform(display)
        if not (-0.35 <= axes_xy[0] <= 0.2 and 0.8 <= axes_xy[1] <= 1.35):
            continue
        inches = ax.figure.dpi_scale_trans.inverted().transform(display)
        return label, [float(inches[0] * 72), float(inches[1] * 72)]
    return None


def matplotlib_layout_manifest(
    fig: Any,
    *,
    axes: Sequence[Any] | None = None,
    panel_ids: Mapping[Any, str] | Sequence[str] | None = None,
    row_groups: Sequence[Any] | None = None,
    column_groups: Sequence[Any] | None = None,
    exclude_axes: Iterable[Any] = (),
    exemptions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Measure final Matplotlib axes rectangles in physical points."""
    fig.canvas.draw()
    width_in, height_in = (float(value) for value in fig.get_size_inches())
    width_pt, height_pt = width_in * 72, height_in * 72
    excluded = {id(axis) for axis in exclude_axes}
    candidates = list(axes) if axes is not None else list(fig.axes)
    selected: list[Any] = []
    seen_subplot_cells: set[tuple[int, int, int]] = set()
    for axis in candidates:
        if id(axis) in excluded or not axis.get_visible():
            continue
        axis_label = str(axis.get_label() or "")
        if axis_label.startswith("<colorbar"):
            continue
        try:
            subplot_spec = axis.get_subplotspec()
        except AttributeError:
            subplot_spec = None
        if subplot_spec is not None and panel_ids is None:
            subplot_key = (
                id(subplot_spec.get_gridspec()),
                int(subplot_spec.num1),
                int(subplot_spec.num2),
            )
            if subplot_key in seen_subplot_cells:
                continue
            seen_subplot_cells.add(subplot_key)
        selected.append(axis)

    if isinstance(panel_ids, Mapping):
        ids = [str(panel_ids.get(axis) or "") for axis in selected]
    elif panel_ids is not None:
        ids = [str(value) for value in panel_ids]
        if len(ids) != len(selected):
            raise ValueError("panel_ids length must match the selected Matplotlib axes")
    else:
        ids = []
        for index, axis in enumerate(selected):
            candidate = str(axis.get_gid() or axis.get_label() or "").strip()
            ids.append(candidate if candidate and not candidate.startswith("<") else _alphabetic_id(index))

    grid_ids: dict[int, str] = {}
    panels: list[dict[str, Any]] = []
    for axis, panel_id in zip(selected, ids):
        position = axis.get_position(original=False)
        panel: dict[str, Any] = {
            "id": panel_id,
            "bbox_pt": [
                float(position.x0 * width_pt),
                float(position.y0 * height_pt),
                float(position.x1 * width_pt),
                float(position.y1 * height_pt),
            ],
        }
        try:
            subplot_spec = axis.get_subplotspec()
        except AttributeError:
            subplot_spec = None
        if subplot_spec is not None:
            grid_spec = subplot_spec.get_gridspec()
            grid_key = id(grid_spec)
            if grid_key not in grid_ids:
                grid_ids[grid_key] = f"matplotlib-grid-{len(grid_ids) + 1}"
            panel.update(
                {
                    "grid_id": grid_ids[grid_key],
                    "row_start": int(subplot_spec.rowspan.start),
                    "row_stop": int(subplot_spec.rowspan.stop),
                    "col_start": int(subplot_spec.colspan.start),
                    "col_stop": int(subplot_spec.colspan.stop),
                }
            )
        label_anchor = _matplotlib_panel_label_anchor(axis)
        if label_anchor is not None:
            panel["panel_label"] = label_anchor[0]
            panel["panel_label_anchor_pt"] = label_anchor[1]
        panels.append(panel)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": "python-matplotlib",
        "figure": {"width_pt": width_pt, "height_pt": height_pt},
        "panels": panels,
        "exemptions": [dict(exemption) for exemption in exemptions],
    }
    if row_groups is not None:
        manifest["row_groups"] = list(row_groups)
    if column_groups is not None:
        manifest["column_groups"] = list(column_groups)
    return manifest


def require_matplotlib_panel_alignment(
    fig: Any,
    *,
    json_out: str | Path | None = None,
    overlay_svg: str | Path | None = None,
    tolerance_pt: float = DEFAULT_TOLERANCE_PT,
    gutter_tolerance_pt: float = DEFAULT_GUTTER_TOLERANCE_PT,
    require_panel_labels: bool = False,
    strict: bool = False,
    **manifest_options: Any,
) -> dict[str, Any]:
    """Measure and block delivery when a Matplotlib multi-panel layout is misaligned."""
    manifest = matplotlib_layout_manifest(fig, **manifest_options)
    report = audit_layout_manifest(
        manifest,
        tolerance_pt=tolerance_pt,
        gutter_tolerance_pt=gutter_tolerance_pt,
        require_panel_labels=require_panel_labels,
    )
    if json_out is not None:
        write_json_report(report, json_out)
    if overlay_svg is not None and report.get("layout"):
        write_overlay_svg(report, overlay_svg)
    code = exit_code(report, strict=strict)
    if code:
        raise PanelAlignmentError(render_text(report, strict=strict))
    return report


if __name__=='__main__':main()
