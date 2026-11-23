"""Export Figure5b with an unchanged common-event population."""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import argparse,hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
BASE = ROOT / 'V/Figure_V1/Figure5'
SOURCE=ROOT/'Result/Observation_only_references/20261111'
REACHES=['Upstream','Midstream','Downstream']
LABELS=['Upper','Middle','Lower']
GROUP_STYLE={'A':('#447F9A','o'),'B':('#B86749','s'),'C':('#50846D','^')}


def check(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sources(args):
    paths={'events':args.events_csv,'summary':args.summary_csv,'daily':args.daily_csv}
    frames={name:pd.read_csv(path,encoding='utf-8-sig') for name,path in paths.items()}
    events=frames['events'];summary=frames['summary'];daily=frames['daily']
    fields=['Site_ID','Event_ID','Reach','Event_Days','Event_Observed_Days','A_Shock_L','A_Shock_S','C_Shock_L','C_Shock_S']
    check(set(fields).issubset(events),'Missing event ratio/coverage fields.')
    check(not events[['Site_ID','Event_ID','Reach','Event_Days','Event_Observed_Days']].isna().any().any(),'Missing event keys or coverage.')
    check(not events.duplicated(['Site_ID','Event_ID']).any(),'Duplicate site-event keys.')
    check(events.Reach.isin(REACHES).all() and events.groupby('Site_ID').Reach.nunique().eq(1).all(),'Invalid or inconsistent site reaches.')
    check(((events.Event_Days>0)&(events.Event_Observed_Days>=0)&(events.Event_Observed_Days<=events.Event_Days)).all(),'Invalid event coverage.')
    valid=lambda series:np.isfinite(series)&series.gt(0)
    full=events.Event_Observed_Days.eq(events.Event_Days)
    common=full.copy()
    for field in ['A_Shock_L','A_Shock_S','C_Shock_L','C_Shock_S']:common &= valid(events[field])
    check(common.any(),'No common eligible events.')
    rows=[];members=[]
    for sample in ['Available','Common_ABC']:
        for group in ['A','B','C']:
            mask=pd.Series(True,index=events.index) if sample=='Available' else common.copy()
            if group=='B':mask &= full
            field=('C' if group=='C' else 'A')+'_Shock_L'
            eligible=mask&valid(events[field])
            used=events.loc[eligible,['Site_ID','Event_ID','Reach',field]].rename(columns={field:'L'}).copy()
            used['Sample']=sample;used['Group']=group;members.append(used)
            for reach in REACHES:
                part=used[used.Reach.eq(reach)]
                check(len(part)>0,'Empty reach estimate.')
                site_means=part.groupby('Site_ID').L.mean()
                mean=float(site_means.mean());count=len(part);sites=len(site_means)
                reference=summary[(summary.Sample==sample)&(summary.Group==group)&(summary.Level=='Reach')&(summary.Entity==reach)]
                check(len(reference)==1,'Missing/duplicate saved reach summary.')
                saved=reference.iloc[0]
                check(int(saved.Shock_L_N)==count and int(saved.Shock_L_Sites)==sites and np.isclose(saved.Shock_L_SiteEqual_Mean,mean,rtol=1e-8,atol=1e-10),'Reach mean/population disagrees with saved summary; check data version.')
                rows.append(dict(Sample=sample,Group=group,Reach=reach,Event_N=count,Site_N=sites,Site_equal_mean_L=mean))
    reach_table=pd.DataFrame(rows)
    for reach in REACHES:
        part=reach_table[(reach_table.Sample=='Common_ABC')&(reach_table.Reach==reach)].set_index('Group')
        check(np.isclose(part.loc['A','Site_equal_mean_L'],part.loc['B','Site_equal_mean_L']) and part.Event_N.nunique()==1 and part.Site_N.nunique()==1,'Common A/B estimates or populations disagree.')
    daily_fields=['Date','DO_QC','DO_Final','DO_Source']
    check(set(daily_fields).issubset(daily),'Missing daily source fields.')
    daily=daily.copy();daily.Date=pd.to_datetime(daily.Date,errors='raise')
    check(daily.Date.notna().all() and not daily.Date.duplicated().any(),'Invalid/duplicate dates.')
    daily=daily.sort_values('Date').reset_index(drop=True)
    calendar=pd.date_range(daily.Date.min(),daily.Date.max(),freq='D')
    check(daily.Date.tolist()==list(calendar),'Daily audit has calendar gaps; do not bridge missing dates.')
    check(daily.DO_Source.isin(['Observed','Imputed']).all(),'Unexpected DO source label.')
    check(np.isfinite(daily.DO_Final).all() and daily.DO_Final.ge(0).all(),'Invalid completed daily values.')
    observed=daily.DO_Source.eq('Observed')
    check(observed.equals(daily.DO_QC.notna()),'Daily source labels disagree with original QC availability.')
    check(np.isfinite(daily.loc[observed,'DO_QC']).all() and daily.loc[observed,'DO_QC'].ge(0).all(),'Invalid original observations.')
    check(np.allclose(daily.loc[observed,'DO_Final'],daily.loc[observed,'DO_QC'],rtol=0,atol=2e-6),'Original daily observations changed in the completed series.')
    low=daily.DO_Final.lt(2)
    if 'Completed_Below2' in daily:
        flags=daily.Completed_Below2.astype(str).str.lower()
        check(flags.isin(['true','false']).all() and flags.eq('true').equals(low),'Saved completed-low flag mismatch.')
    sources={name:{'path':str(path.resolve()),'sha256':sha(path),'input_rows':len(frames[name])} for name,path in paths.items()}
    counts={'event_input_rows':len(events),'common_events':int(common.sum()),'common_sites':int(events.loc[common,'Site_ID'].nunique()),
            'daily_rows':len(daily),'daily_observed':int(observed.sum()),'daily_imputed':int((~observed).sum()),'daily_excluded':0,
            'daily_completed_below2':int(low.sum()),'calendar_start':daily.Date.min().isoformat(),'calendar_end':daily.Date.max().isoformat()}
    return reach_table,pd.concat(members,ignore_index=True),daily,sources,counts


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def axis_limits(table):
    values=table.Site_equal_mean_L
    lo=min(float(values.min()),1);hi=max(float(values.max()),1)
    return np.floor((lo-.02)/.02)*.02,np.ceil((hi+.02)/.02)*.02


def draw_reaches(ax,table,common,limits):
    sample='Common_ABC' if common else 'Available'
    data=table[table.Sample==sample]
    positions=np.arange(len(REACHES))[::-1]
    for group,offset in ([('A',.12),('C',-.12)] if common else [('A',.23),('B',0),('C',-.23)]):
        color,marker=GROUP_STYLE[group]
        part=data[data.Group==group].set_index('Reach').loc[REACHES]
        ax.plot(part.Site_equal_mean_L,positions+offset,linestyle='none',marker=marker,
                markersize=5,color=color,label='A=B' if common and group=='A' else group)
        if not common:
            for y,row in zip(positions+offset,part.itertuples()):
                ax.text(1.035,y,f'{group}  {row.Event_N} / {row.Site_N}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=6.5)
    if common:
        part=data[data.Group=='A'].set_index('Reach').loc[REACHES]
        for y,row in zip(positions,part.itertuples()):
            ax.text(1.035,y,f'{row.Event_N} / {row.Site_N}',transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=6.5)
    ax.axvline(1,color='#858E92',linestyle='--',linewidth=.8,zorder=0)
    ax.set(yticks=positions,yticklabels=LABELS,ylim=(-.65,2.9),xlim=limits,
           xlabel='Site-equal mean seasonal ratio, L',title='Common-event comparison' if common else 'Each eligible event population')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.locator_params(axis='x',nbins=4)
    ax.text(1.035,.945,'Events / sites',transform=ax.transAxes,ha='left',va='center',fontsize=6.5)
    ax.legend(loc='upper left',ncol=2 if common else 3,fontsize=7,handletextpad=.25,columnspacing=.7)
    ax.tick_params(direction='out',length=3)
    ax.tick_params(axis='y',length=0,pad=5)


def low_runs(daily):
    flag=daily.DO_Final.lt(2).to_numpy();change=np.diff(np.r_[False,flag,False].astype(int))
    return [(daily.Date.iloc[start],daily.Date.iloc[stop-1]+pd.Timedelta(days=1)) for start,stop in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1))]


def draw_daily(ax,daily,site_label):
    observed=daily.DO_Source.eq('Observed');imputed=~observed
    for start,end in low_runs(daily):ax.axvspan(start,end,color='#E9C4C6',alpha=.18,linewidth=0,zorder=0)
    ax.plot(daily.Date,daily.DO_Final,color='#AEB6BB',linewidth=.35,zorder=1,rasterized=True)
    ax.scatter(daily.loc[observed,'Date'],daily.loc[observed,'DO_QC'],s=7,color='#447F9A',marker='o',linewidths=0,zorder=3,rasterized=True)
    ax.scatter(daily.loc[imputed,'Date'],daily.loc[imputed,'DO_Final'],s=11,color='#C87953',marker='x',linewidths=.55,zorder=2,rasterized=True)
    for cutoff in [2,3,5]:ax.axhline(cutoff,color='#919A9F',linestyle='--',linewidth=.55,zorder=1)
    handles=[Line2D([],[],marker='o',linestyle='none',color='#447F9A',markersize=3.5,label=f'Original QC (n={int(observed.sum())})'),
             Line2D([],[],marker='x',linestyle='none',color='#C87953',markersize=4,label=f'Imputed (n={int(imputed.sum())})'),
             Patch(facecolor='#E9C4C6',alpha=.5,edgecolor='none',label='Completed DO <2 mg/L'),
             Line2D([],[],color='#919A9F',linestyle='--',linewidth=.6,label='2, 3, 5 mg/L references')]
    ax.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,1.015),ncol=4,fontsize=6.5,
              handletextpad=.35,columnspacing=1,borderaxespad=0)
    padding=pd.Timedelta(days=5)
    ax.set(xlim=(daily.Date.min()-padding,daily.Date.max()+padding),
           ylim=(-.45,float(daily.DO_Final.max())*1.12),ylabel='Daily mean DO (mg L$^{-1}$)',xlabel='Date',
           title=f'{site_label}: daily DO, {daily.Date.min().year}–{daily.Date.max().year}')
    ax.set_title(ax.get_title(),pad=25)
    year_ticks=pd.date_range(f'{daily.Date.min().year}-01-01',f'{daily.Date.max().year}-01-01',freq='YS')
    ax.set_xticks(year_ticks);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
    ax.tick_params(direction='out',length=3)


def run(panel):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events-csv',type=Path,default=SOURCE/'20261111_实测来源核对.csv')
    parser.add_argument('--summary-csv',type=Path,default=SOURCE/'20261111_纯实测双基线敏感性.csv')
    parser.add_argument('--daily-csv',type=Path,default=ROOT/'Result/Jiagaiqiao_source_audit/20261111_Audit/daily_site_audit.csv')
    parser.add_argument('--site-label',default='Jiagaiqiao')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    table,members,daily,sources,counts=read_sources(args)
    style();limits=axis_limits(table)
    dimensions=[183,88] if panel=='c' else [89,95]
    width,height=dimensions[0]/25.4,dimensions[1]/25.4
    fig,ax=plt.subplots(figsize=(width,height))
    fig.subplots_adjust(left=.13 if panel=='c' else .20,right=.97 if panel=='c' else .76,
                        bottom=.20 if panel=='c' else .24,top=.76 if panel=='c' else .86)
    if panel=='c':draw_daily(ax,daily,args.site_label)
    else:
        draw_reaches(ax,table,panel=='b',limits)
        fig.text(.20,.03,'A: completed series\nB: every event day observed\nC: observation-based recalculation',
                 fontsize=6.3,ha='left',va='bottom',linespacing=1.3)
    output=args.output_dir or BASE/panel/'Images'
    stem={'a':'Figure5a_Eligible_populations','b':'Figure5b_Common_events','c':'Figure5c_Jiagaiqiao_daily_DO'}[panel]
    output.mkdir(parents=True,exist_ok=True);qa=output.parent/'QA';qa.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=qa/f'{stem}.alignment.json',overlay_svg=qa/f'{stem}.alignment.svg',
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(output/f'{stem}.pdf',dpi=600);fig.savefig(output/f'{stem}.svg',dpi=600);fig.savefig(output/f'{stem}.png',dpi=300)
    with BytesIO() as buffer:
        fig.savefig(buffer,format='tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        (output/f'{stem}.tiff').write_bytes(buffer.getbuffer())
    plt.close(fig)
    if panel in ['a','b']:
        selected=table[table.Sample.eq('Available' if panel=='a' else 'Common_ABC')]
        selected.to_csv(output/f'{stem}_reach_source.csv',index=False,encoding='utf-8-sig')
        members.to_csv(output/f'{stem}_eligible_event_source.csv',index=False,encoding='utf-8-sig')
    if panel=='c':daily.to_csv(output/f'{stem}_daily_source.csv',index=False,encoding='utf-8-sig')
    for source in sources.values():check(sha(source['path'])==source['sha256'],'Input changed during plotting.')
    metadata={'sources':sources,'counts':counts,'dimensions_mm':dimensions,'reach_xlim':list(limits),
              'weighting':'Mean within each site, then equal mean across sites in each reach.',
              'eligibility':'A finite positive completed L; B same L with all event days observed; C finite positive existing observation-based L. Common intersection additionally requires positive finite A/C S.',
              'crosschecks':'All 18 reach estimates, event counts and site counts agree with saved A/B/C summaries.',
              'uncertainty':'Descriptive site-equal means, no CI; daily source marks do not validate gaps or sensors.',
              'daily_low_runs':len(low_runs(daily)),'daily_excluded':0,'models_or_references_recomputed':False,
              'layout_revision':'Compact 183x158mm; explicit A/B/C count prefixes and definition note; daily shading unchanged in date coverage.',
              'alignment_groups':'Comparable reach panels a/b share a row; c is a different-modality full-width series with the same left edge.'}
    (qa/f'{stem}.provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output/f'{stem}.pdf')


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
