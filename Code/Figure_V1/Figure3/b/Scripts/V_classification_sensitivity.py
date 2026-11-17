"""Export Figure3b from existing A/B/C ratios and saved summary CSV."""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[5]
BASE = ROOT / 'V/Figure_V1/Figure3'
SOURCE=ROOT/'Result/Observation_only_references/20261111'
TYPES=['Heatwave_Only','Flood_Only','Direct_Compound','H5F_Compound','F5H_Compound']
LABELS=['Heatwave-only','High-flow-only','DC','H5F','F5H']
COLORS=['#B86749','#438DA5','#50846D','#786C9C','#B16B8B']


def check(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(event_csv,summary_csv):
    events=pd.read_csv(event_csv,encoding='utf-8-sig')
    summary=pd.read_csv(summary_csv,encoding='utf-8-sig')
    required=['Site_ID','Event_ID','Event_Type','Event_Days','Event_Observed_Days',
              'A_Shock_L','A_Shock_S','C_Shock_L','C_Shock_S']
    check(set(required).issubset(events),'Missing event-level ratio or coverage fields.')
    check(len(events)>0 and not events[['Site_ID','Event_ID','Event_Type','Event_Days','Event_Observed_Days']].isna().any().any(),'Missing event keys/coverage.')
    check(not events.duplicated(['Site_ID','Event_ID']).any(),'Duplicate site-event keys.')
    check(events.Event_Type.isin(TYPES).all(),'Unknown event type.')
    days=events.Event_Days;obs=events.Event_Observed_Days
    check(((days>0)&(obs>=0)&(obs<=days)&(days%1==0)&(obs%1==0)).all(),'Invalid event coverage counts.')
    valid=lambda values:np.isfinite(values)&values.gt(0)
    rows=[];paired=None;selection={}
    for group in ['A','B','C']:
        prefix='C' if group=='C' else 'A'
        candidate=events.Event_Observed_Days.eq(events.Event_Days) if group=='B' else pd.Series(True,index=events.index)
        eligible=candidate&valid(events[prefix+'_Shock_L'])&valid(events[prefix+'_Shock_S'])
        data=events.loc[eligible].copy()
        data['L']=data[prefix+'_Shock_L'];data['S']=data[prefix+'_Shock_S']
        check(len(data)>0,f'No eligible {group} pairs.')
        opposite=(data.L.lt(1)&data.S.gt(1))|(data.L.gt(1)&data.S.lt(1))
        beyond=(data.L.lt(.95)&data.S.gt(1.05))|(data.S.lt(.95)&data.L.gt(1.05))
        saved=summary[(summary.Sample=='Available')&(summary.Group==group)&(summary.Level=='Overall')&(summary.Entity=='All')]
        check(len(saved)==1,f'Expected one existing overall summary row for {group}.')
        n=len(data);strict=int(opposite.sum());band=int(beyond.sum())
        check(int(saved.Paired_N.iloc[0])==n and int(saved.Discordant_N.iloc[0])==strict and int(saved.Opposite_Beyond5_N.iloc[0])==band,f'{group} ratios disagree with saved summary; do not mix versions.')
        rows.append(dict(Group=group,Paired_N=n,Opposite_N=strict,Opposite_Pct=100*strict/n,
                         Beyond5_N=band,Beyond5_Pct=100*band/n))
        selection[group]={'input_events':len(events),'coverage_eligible':int(candidate.sum()),
                          'positive_finite_pairs':n,'missing_or_nonpositive_ratio_pairs':int(candidate.sum())-n}
        if group=='A':paired=data
    quadrants=[('L < 1, S < 1',(paired.L<1)&(paired.S<1)),
               ('L > 1, S > 1',(paired.L>1)&(paired.S>1)),
               ('L < 1, S > 1',(paired.L<1)&(paired.S>1)),
               ('L > 1, S < 1',(paired.L>1)&(paired.S<1)),
               ('L = 1 or S = 1',paired.L.eq(1)|paired.S.eq(1))]
    quad=pd.DataFrame([dict(Class=label,N=int(mask.sum()),Percentage=100*mask.mean()) for label,mask in quadrants])
    check(int(quad.N.sum())==len(paired),'Quadrants do not partition A pairs.')
    sources={name:{'path':str(path.resolve()),'sha256':sha(path)} for name,path in [('events',event_csv),('summary',summary_csv)]}
    return paired,pd.DataFrame(rows),quad,sources,selection


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,
        'ytick.labelsize':7.5,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'savefig.facecolor':'white'})


def scatter(ax,data,size):
    for event_type,color in zip(TYPES,COLORS):
        part=data[data.Event_Type.eq(event_type)]
        ax.scatter(part.L,part.S,s=size,c=color,alpha=.55,linewidths=0,rasterized=True)
    ax.axvline(1,color='#7C888F',linestyle='--',linewidth=.7,zorder=1)
    ax.axhline(1,color='#7C888F',linestyle='--',linewidth=.7,zorder=1)


def draw_a(fig,ax,data,quad,*,legend_inside=False):
    scatter(ax,data,7)
    maximum=float(max(data.L.max(),data.S.max(),1))*1.04
    ax.set(xlim=(0,maximum),ylim=(0,maximum),xlabel='Seasonal-reference ratio, L',
           ylabel='Antecedent-reference ratio, S',title='Paired reference ratios')
    ax.set_aspect('equal',adjustable='box',anchor='C' if legend_inside else 'W')
    inset=ax.inset_axes([.49,.48,.45,.45])
    scatter(inset,data,3.5)
    inset.set(xlim=(.6,1.4),ylim=(.6,1.4),xticks=[.6,1,1.4],yticks=[.6,1,1.4])
    inset.tick_params(labelsize=6,length=2,pad=1)
    inset.set_title('Detail: 0.6–1.4',fontsize=7,pad=4)
    inset.set_facecolor('white')
    for spine in inset.spines.values():spine.set_visible(True);spine.set_linewidth(.5)
    text='Whole-sample quadrants\n'+'\n'.join(f'{row.Class}: {row.N} ({row.Percentage:.2f}%)' for row in quad.itertuples() if row.N or row.Class!='L = 1 or S = 1')
    note=ax.text(.98,.43,text,transform=ax.transAxes,ha='right',va='top',fontsize=6.5,linespacing=1.4)
    handles=[Line2D([],[],marker='o',linestyle='none',color=color,markersize=4,
                    label=f'{label} (n={int(data.Event_Type.eq(t).sum())})') for t,label,color in zip(TYPES,LABELS,COLORS)]
    legend=ax.legend(handles=handles,loc='lower right' if legend_inside else 'upper left',
                     bbox_to_anchor=(.98,.03) if legend_inside else (-.04,-.17),
                     ncol=2,fontsize=6.8,handletextpad=.4,columnspacing=1,
                     title=f'Completed series: {len(data)} pairs; {data.Site_ID.nunique()} sites',title_fontsize=7)
    fig.canvas.draw()
    legend_box=legend.get_window_extent()
    if legend_inside:
        check(all(ax.bbox.contains(x,y) for x,y in legend_box.get_points()),
              'Legend must fit inside the full-range axes.')
    xy=ax.transData.transform(data[['L','S']].to_numpy())
    for name,box in [('summary',note.get_window_extent(fig.canvas.get_renderer())),
                     ('detail inset',inset.get_window_extent()),('legend',legend_box.padded(2*fig.dpi/72))]:
        inside=(xy[:,0]>box.x0)&(xy[:,0]<box.x1)&(xy[:,1]>box.y0)&(xy[:,1]<box.y1)
        check(not inside.any(),f'{name} obscures full-range data points.')
    return inset


def diagonal_segments(x,y,width,height,spacing):
    """Return 45-degree strokes ending inside a rectangular display area."""
    segments=[]
    for offset in np.arange(-height+spacing/2,width,spacing):
        start=max(0,-offset)
        end=min(height,width-offset)
        if end>start:
            segments.append(((x+offset+start,y+start),(x+offset+end,y+end)))
    return segments


class DiagonalBarKey(HandlerPatch):
    def create_artists(self,legend,orig_handle,xdescent,ydescent,width,height,fontsize,trans):
        artists=super().create_artists(legend,orig_handle,xdescent,ydescent,width,height,fontsize,trans)
        key=artists[0]
        strokes=diagonal_segments(*key.get_xy(),key.get_width(),key.get_height(),5.6)
        artists.append(LineCollection(strokes,colors=[key.get_edgecolor()],linewidths=.55,transform=trans))
        return artists


def draw_b(ax,table):
    positions=np.arange(len(table))[::-1]
    maximum=float(table[['Opposite_Pct','Beyond5_Pct']].to_numpy().max())
    ax.set(yticks=positions,yticklabels=table.Group,xlim=(0,max(65,maximum*1.55)),
           ylim=(-.6,len(table)-.05),xlabel='Opposite classifications (%)',title='Classification sensitivity')
    bar_groups=[]
    for field,countfield,label,color,offset in [
        ('Opposite_Pct','Opposite_N','Opposite sides of 1','#447F9A',.20),
        ('Beyond5_Pct','Beyond5_N','Opposite beyond ±5%','white',-.20)]:
        bars=ax.barh(positions+offset,table[field],height=.32,facecolor=color,
                     edgecolor='#425864',linewidth=.5,label=label)
        bar_groups.append(bars)
        for y,row in zip(positions+offset,table.itertuples()):
            pct=getattr(row,field);count=getattr(row,countfield)
            ax.text(pct+1.1,y,f'{pct:.2f}%\n{count}/{row.Paired_N}',ha='left',va='center',fontsize=6.5,linespacing=1.15)
    # Use bounded vector strokes instead of a repeated PDF tiling pattern.
    strict_bars,band_bars=bar_groups
    display=ax.transData
    to_data=display.inverted()
    for bar in band_bars:
        corners=display.transform([bar.get_xy(),(bar.get_x()+bar.get_width(),bar.get_y()+bar.get_height())])
        width,height=corners[1]-corners[0]
        strokes=diagonal_segments(*corners[0],width,height,5.6*ax.figure.dpi/72)
        if strokes:
            data_strokes=to_data.transform(np.asarray(strokes).reshape(-1,2)).reshape(-1,2,2)
            ax.add_collection(LineCollection(data_strokes,colors=[bar.get_edgecolor()],
                                             linewidths=.55,capstyle='butt'),autolim=False)
    ax.legend(handles=[strict_bars[0],band_bars[0]],
              labels=[strict_bars.get_label(),band_bars.get_label()],
              handler_map={band_bars[0]:DiagonalBarKey()},
              loc='upper left',bbox_to_anchor=(0,1.0),fontsize=6.8,handlelength=1.5)
    ax.tick_params(direction='out',length=3)


def run(panel):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events-csv',type=Path,default=SOURCE/'20261111_实测来源核对.csv')
    parser.add_argument('--summary-csv',type=Path,default=SOURCE/'20261111_纯实测双基线敏感性.csv')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    data,table,quad,sources,selection=load_data(args.events_csv,args.summary_csv)
    style()
    dimensions=[135,118] if panel=='a' else [89,118]
    width,height=dimensions[0]/25.4,dimensions[1]/25.4
    fig,ax=plt.subplots(figsize=(width,height))
    fig.subplots_adjust(left=.13 if panel=='a' else .14,right=.98,bottom=.14,top=.91)
    if panel=='a':extras=[draw_a(fig,ax,data,quad,legend_inside=True)]
    else:draw_b(ax,table);extras=[]
    output=args.output_dir or BASE/panel/'Images'
    stem='Figure3a_Paired_reference_ratios' if panel=='a' else 'Figure3b_Classification_sensitivity'
    output.mkdir(parents=True,exist_ok=True);qa=output.parent/'QA';qa.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=qa/f'{stem}.alignment.json',
        overlay_svg=qa/f'{stem}.alignment.svg',exclude_axes=extras,tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(output/f'{stem}.pdf',dpi=600)
    fig.savefig(output/f'{stem}.svg',dpi=600)
    fig.savefig(output/f'{stem}.png',dpi=300)
    with BytesIO() as buffer:
        fig.savefig(buffer,format='tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        (output/f'{stem}.tiff').write_bytes(buffer.getbuffer())
    plt.close(fig)
    if panel=='a':
        data.to_csv(output/f'{stem}_paired_source.csv',index=False,encoding='utf-8-sig')
        quad.to_csv(output/f'{stem}_quadrants.csv',index=False,encoding='utf-8-sig')
    if panel=='b':table.to_csv(output/f'{stem}_proportion_source.csv',index=False,encoding='utf-8-sig')
    for source in sources.values():check(sha(source['path'])==source['sha256'],'Input changed during plotting.')
    metadata={'sources':sources,'selection':selection,'dimensions_mm':dimensions,
              'input_rows':selection['A']['input_events'],'a_paired_events':len(data),'a_sites':int(data.Site_ID.nunique()),
              'inset':'Detail is a duplicate view; all eligible A points remain in full-range main scatter.',
              'inset_alignment':'Intentionally smaller detail inset excluded from comparable main-axis groups.',
              'data_exclusion':'Only missing or nonpositive ratio pairs and the stated B coverage condition.',
              'uncertainty':'Descriptive proportions; no CI or significance tests.',
              'models_refitted':False,'summary_crosscheck':'All A/B/C counts agree with matching saved summary CSV.'}
    (qa/f'{stem}.provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output/f'{stem}.pdf')
    print(table.to_string(index=False))


# region Rendered panel-alignment audit (local copy for standalone execution)
import html
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any


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
# endregion


if __name__ == '__main__':
    run('b')
