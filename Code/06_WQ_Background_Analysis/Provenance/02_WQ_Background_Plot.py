"""Create submission-style figures from the analysis CSV outputs."""
from pathlib import Path
import importlib.util
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

# Paths and display parameters
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Result/WQ_Background_Analysis"
QUALITY = RESULTS / "01_Data_Quality"
BACKGROUND = RESULTS / "02_Longterm_Background"
EVENT = RESULTS / "03_Event_NonEvent"
TYPES = RESULTS / "04_Event_Type"
LOW = RESULTS / "05_Low_DO"
FIGURES = RESULTS / "06_Figures"
REPORT = RESULTS / "07_Report"
ALIGNMENT_HELPER = Path(os.getenv(
    "WQ_PANEL_ALIGNMENT_HELPER",
    str(Path.home()/".codex/skills/nature-figure/scripts/audit_panel_alignment.py")))
REACH = {"UR":"#4C78A8","MR":"#59A14F","LR":"#E39C45"}
EVENT_ORDER = ["Heatwave only","High-flow only","DC","H5F","F5H"]
EVENT_COLOR = ["#C85D45","#447F9A","#50846D","#79769B","#B06485"]
CHEM = ["NH3_N","TN","TP","CODMn"]
FOCUS = [*CHEM,"WT","DO"]
LABEL = {"NH3_N":r"NH$_3$-N","TN":"TN","TP":"TP","CODMn":"CODMn","WT":"WT","DO":"DO",
         "pH":"pH","EC":"EC","Turb":"Turb"}
DO_GROUPS = ["Very_Low_DO","Low_DO","Higher_DO"]
DO_LABELS = ["<2","2≤DO<5","≥5"]
LOG = logging.getLogger("wq-plot")


def read(folder, filename):
    path = folder/filename
    frame = pd.read_csv(path,encoding="utf-8-sig")
    LOG.info("Read %s (%d rows)",path,len(frame))
    return frame


def configure():
    try:
        findfont(FontProperties(family="Arial"),fallback_to_default=False)
        font="Arial"
    except ValueError:
        font="DejaVu Sans"
        LOG.warning("Arial unavailable. Using %s.",font)
    plt.rcParams.update({"font.family":font,"font.size":8.3,"axes.linewidth":.8,
                         "axes.unicode_minus":True,"xtick.direction":"out","ytick.direction":"out",
                         "xtick.major.size":3,"ytick.major.size":3,"xtick.major.width":.75,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "text.color":"#222222","axes.labelcolor":"#222222",
                         "figure.facecolor":"white","axes.facecolor":"white",
                         "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})


def audit_alignment(fig, filename):
    fig.canvas.draw()
    if len([ax for ax in fig.axes if not getattr(ax, "_colorbar", None)])<2:
        return
    if ALIGNMENT_HELPER.is_file():
        spec=importlib.util.spec_from_file_location("wq_panel_alignment",ALIGNMENT_HELPER)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        qa=FIGURES/"QA"
        qa.mkdir(exist_ok=True)
        module.require_matplotlib_panel_alignment(
            fig,json_out=qa/f"{filename}.alignment.json",require_panel_labels=True)
        return
    LOG.info("Optional panel alignment helper absent, checking grid geometry with Matplotlib")
    areas=[ax.get_position() for ax in fig.axes if ax.get_subplotspec() is not None and
           not getattr(ax, "_colorbar", None)]
    if areas and (np.ptp([p.width for p in areas])*fig.get_figwidth()*72>1.5 or
                  np.ptp([p.height for p in areas])*fig.get_figheight()*72>1.5):
        raise ValueError("Rendered panel dimensions differ by more than 1.5 pt")


def save(fig, filename):
    audit_alignment(fig,filename)
    base = FIGURES/filename
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression":"tiff_lzw"})
    fig.savefig(base.with_suffix(".png"), dpi=300)
    LOG.info("Wrote %s in PDF, SVG, TIFF (600 dpi) and PNG (300 dpi)",base)
    plt.close(fig)


def grid(variables, *, height=5.0, width=None, top=.94, bottom=.13):
    columns=3 if len(variables)==6 else 2
    fig, axes=plt.subplots(2,columns,figsize=(width or (8.9 if columns==3 else 7.7),height))
    fig.subplots_adjust(left=.10,right=.98,bottom=bottom,top=top,wspace=.45 if columns==3 else .31,hspace=.54)
    for index,(ax,var) in enumerate(zip(axes.flat,variables)):
        ax.text(0,1.075,f"({chr(97+index)})",transform=ax.transAxes,ha="left",va="baseline",
                fontsize=10,fontweight="bold",gid=f"panel-{chr(97+index)}")
        ax.text(.145 if columns==3 else .105,1.075,LABEL[var],transform=ax.transAxes,
                ha="left",va="baseline",fontsize=9.4)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        formatter=ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.grid(axis="y",color="#DDE1E3",alpha=.65,linewidth=.45,zorder=0)
        ax.set_axisbelow(True)
    return fig, list(axes.flat)


def unit_label(var):
    return "Temperature (°C)" if var=="WT" else r"Concentration (mg L$^{-1}$)"


def ticks(ax, positions, labels):
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x",pad=4)


def show_distribution(ax, position, sample, color, *, width=.46, points=True, min_box=5):
    sample=np.asarray(sample,dtype=float)
    sample=sample[np.isfinite(sample)]
    if sample.size==0:
        return
    if sample.size>=min_box:
        ax.boxplot([sample],positions=[position],widths=width,manage_ticks=False,patch_artist=True,
                   showfliers=not points,boxprops={"facecolor":color+"33","edgecolor":color,"linewidth":.8},
                   whiskerprops={"color":color,"linewidth":.75},capprops={"color":color,"linewidth":.75},
                   medianprops={"color":color,"linewidth":1.25},
                   flierprops={"markersize":2.6,"marker":"o","markeredgewidth":.6,"markeredgecolor":color})
    else:
        med=np.median(sample)
        ax.plot([position-width*.45,position+width*.45],[med,med],color=color,lw=1.35,zorder=3)
    if points or sample.size<min_box:
        jitter=np.linspace(-width*.28,width*.28,num=sample.size) if sample.size>1 else np.zeros(sample.size)
        ax.scatter(position+jitter,sample,s=13,color=color,edgecolor="white",linewidth=.35,alpha=.88,zorder=4)


def longterm():
    frame=read(BACKGROUND,"A02_Station_5Year_WQ_Statistics.csv")
    reach=read(BACKGROUND,"A02_Reach_5Year_WQ_Statistics.csv")
    counts=reach.loc[reach.Method.eq("M3")].set_index(["Reach","Variable"]).Contributing_Sites
    fig,axes=grid(FOCUS,height=5.25)
    for ax,var in zip(axes,FOCUS):
        for i,(group,color) in enumerate(REACH.items()):
            cells=frame.loc[frame.Variable.eq(var)&frame.Reach.eq(group),"M3"]
            values=cells.dropna()
            excluded_count=len(cells)-len(values)
            LOG.info("A1 %s/%s: %d stations shown, %d ineligible for M3", group,var,len(values),excluded_count)
            if len(values)!=counts.loc[(group,var)]:
                raise ValueError(f"M3 count mismatch {group}/{var}")
            show_distribution(ax,i,values,color,min_box=1)
        ticks(ax,range(3),[f"{r}\n(n={int(counts.loc[(r,var)])})" for r in REACH])
        ax.set(xlim=(-.52,2.52),ylabel=unit_label(var))
    save(fig,"Fig_A1_Longterm_WQ_Background")


def event_non_event():
    data=read(EVENT,"A03_Station_Event_NonEvent_WQ.csv")
    frame=data.loc[data.Comparison.eq("Matched")]
    summary=read(EVENT,"A03_Reach_Matched_WQ_Changes.csv")
    fig,axes=grid(CHEM,top=.87)
    for ax,var in zip(axes,CHEM):
        for i,(reach,color) in enumerate(REACH.items()):
            group=frame.loc[frame.Variable.eq(var)&frame.Reach.eq(reach)]
            pair=group.pivot(index="Site_ID",columns="Period",values="Mean").reindex(
                columns=["NonEvent","Event"]).dropna()
            off=np.linspace(-.055,.055,len(pair))
            if len(pair):
                left=i-.17+off
                right=i+.17+off
                for x1,x2,non,event in zip(left,right,pair.NonEvent,pair.Event):
                    ax.plot([x1,x2],[non,event],color=color,alpha=.28,lw=.65,zorder=2)
                ax.scatter(left,pair.NonEvent,marker="o",s=13,facecolor="white",edgecolor=color,lw=.7,zorder=3)
                ax.scatter(right,pair.Event,marker="s",s=13,color=color,edgecolor="white",lw=.3,zorder=3)
        counts=summary.loc[summary.Variable.eq(var)&summary.Metric.eq("Delta_C")].set_index("Reach").Contributing_Sites
        ticks(ax,range(3),[f"{r}\n(n={int(counts.get(r,0))})" for r in REACH])
        ax.set(xlim=(-.49,2.49),ylabel=unit_label(var))
    fig.legend(handles=[Line2D([],[],marker="o",color="#53595C",markerfacecolor="white",linestyle="none",
                               markersize=4,label="Non-event"),
                        Line2D([],[],marker="s",color="#53595C",linestyle="none",
                               markersize=4,label="Event")],loc="upper center",frameon=False,ncol=2,
               bbox_to_anchor=(.53,.995))
    save(fig,"Fig_A2_Event_NonEvent_WQ")

    fig,axes=grid(CHEM,top=.87)
    relative=summary.loc[summary.Metric.eq("Relative_Change")].set_index(["Reach","Variable"])
    for ax,var in zip(axes,CHEM):
        ax.axhline(0,color="#70787D",linestyle=(0,(3,2)),linewidth=.8,zorder=1)
        labels=[]
        for i,(reach,color) in enumerate(REACH.items()):
            vals=frame.loc[frame.Variable.eq(var)&frame.Reach.eq(reach)&frame.Period.eq("Event"),
                           "Relative_Change"].dropna()
            cell=relative.loc[(reach,var)]
            if len(vals)!=cell.Contributing_Sites:
                raise ValueError(f"Relative change count mismatch {reach}/{var}")
            jitter=np.linspace(-.15,.15,len(vals))
            ax.scatter(i+jitter,vals,s=12,color=color,alpha=.62,linewidth=0,zorder=2)
            if pd.notna(cell.Mean):
                ax.scatter(i,cell.Mean,marker="D",s=31,color=color,edgecolor="#252525",linewidth=.8,zorder=4)
            labels.append(f"{reach}\n(n={len(vals)})")
        ticks(ax,range(3),labels)
        ax.set(xlim=(-.50,2.50),ylabel="Relative change (%)")
    fig.legend(handles=[Line2D([],[],marker="o",color="#687378",linestyle="none",
                               markersize=4,label="Station"),
                        Line2D([],[],marker="D",color="#53595C",markeredgecolor="#252525",
                               linestyle="none",markersize=4.5,label="Reach mean")],
               loc="upper center",frameon=False,ncol=2,bbox_to_anchor=(.53,.995))
    save(fig,"Fig_A3_Matched_Event_WQ_Changes")


def low_do():
    frame=read(LOW,"A05_Station_DO_Group_WQ.csv")
    frame=frame.loc[frame.Context.eq("Event") & frame.Valid_Pairs.gt(0)]
    for lower_only,filename in [(False,"Fig_A4_Low_DO_WQ_Associations"),
                                (True,"Fig_A4b_Downstream_Low_DO_WQ")]:
        groups=["LR"] if lower_only else list(REACH)
        offsets=[0] if lower_only else [-.25,0,.25]
        fig,axes=grid(CHEM,height=5.6 if not lower_only else 5.1,top=.87,bottom=.18)
        for ax,var in zip(axes,CHEM):
            labels=[]
            for i,dogroup in enumerate(DO_GROUPS):
                counts=[]
                for reach,off in zip(groups,offsets):
                    selected=frame.loc[frame.DO_Group.eq(dogroup)&frame.Variable.eq(var)&frame.Reach.eq(reach)]
                    sample=selected.Median.dropna()
                    show_distribution(ax,i+off,sample,REACH[reach],width=.46 if lower_only else .19)
                    letter=f"{reach} " if not lower_only else ""
                    counts.append(f"{letter}{len(sample)}/{int(selected.Valid_Pairs.sum())}")
                labels.append("\n".join([DO_LABELS[i],*counts]))
            ticks(ax,range(3),labels)
            ax.tick_params(axis="x",labelsize=7.2)
            ax.set(xlim=(-.55,2.55),ylabel=unit_label(var))
        fig.legend(handles=[Line2D([],[],marker="o",color=REACH[g],linestyle="none",
                                   markersize=4,label=g) for g in groups],
                   loc="upper center",frameon=False,ncol=len(groups),bbox_to_anchor=(.54,.997))
        fig.text(.54,.022,r"DO group (mg L$^{-1}$)  ·  Counts: sites / paired days",ha="center",fontsize=8)
        save(fig,filename)


def event_types():
    data=read(TYPES,"A04_Event_Level_WQ_Statistics.csv")
    totals=read(TYPES,"A04_EventType_WQ_Statistics.csv")
    counts=totals.loc[totals.Statistic_Unit.eq("Event equal")].set_index(
        ["Event_Type","Variable"]).Contributing_Events
    fig,axes=grid(CHEM,height=5.15,bottom=.15)
    names=["Heatwave\nonly","High-flow\nonly","DC","H5F","F5H"]
    for ax,var in zip(axes,CHEM):
        labels=[]
        for i,(category,color,name) in enumerate(zip(EVENT_ORDER,EVENT_COLOR,names)):
            sample=data.loc[data.Variable.eq(var)&data.Event_Type.eq(category),"Mean"].dropna()
            if len(sample)!=counts.loc[(category,var)]:
                raise ValueError(f"S0 valid event count mismatch {category}/{var}")
            show_distribution(ax,i,sample,color,width=.48,points=False)
            labels.append(f"{name}\n(n={len(sample)})")
        ticks(ax,range(5),labels)
        ax.tick_params(axis="x",labelsize=7)
        ax.set(xlim=(-.48,4.48),ylabel=unit_label(var))
    save(fig,"Fig_A5_EventType_WQ")


def coverage_mesh(ax, matrix, norm):
    rows, columns = matrix.shape
    ax.grid(False)
    image = ax.pcolormesh(np.arange(columns + 1)-.5, np.arange(rows + 1)-.5,
                          matrix.to_numpy()*100, cmap="Blues", norm=norm, shading="flat")
    ax.set_xlim(-.5,columns-.5)
    ax.set_ylim(rows-.5,-.5)
    return image


def observation_coverage():
    data=read(QUALITY,"A01_Station_Variable_Coverage.csv")
    month=read(QUALITY,"A01_Monthly_Coverage.csv")
    sites=pd.concat([data.loc[data.Reach.eq(r),"Site_ID"].drop_duplicates().sort_values() for r in REACH],
                    ignore_index=True)
    counts=[data.loc[data.Reach.eq(r),"Site_ID"].nunique() for r in REACH]
    cut=np.cumsum(counts)
    centers=np.cumsum(counts)-np.asarray(counts)/2-.5
    variable_order=["DO","WT","pH","EC","Turb","CODMn","NH3_N","TP","TN"]
    norm=Normalize(0,100)
    fig,ax=plt.subplots(figsize=(7,11.65))
    fig.subplots_adjust(left=.31,right=.88,top=.97,bottom=.055)
    matrix=data.pivot(index="Site_ID",columns="Variable",values="Coverage").reindex(
        index=sites,columns=variable_order)
    image=coverage_mesh(ax,matrix,norm)
    ticks(ax,range(len(variable_order)),[LABEL[v] for v in variable_order])
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites,fontsize=5.6)
    ax.tick_params(axis="y",length=0)
    for y in cut[:-1]:
        ax.axhline(y-.5,color="#384048",linewidth=1.15)
    for y,label in zip(centers,REACH):
        ax.text(-.46,y,label,transform=ax.get_yaxis_transform(),color=REACH[label],
                fontweight="bold",va="center")
    cax=fig.add_axes([.91,.14,.022,.72])
    colorbar=fig.colorbar(image,cax=cax,label="Original observations (%)",ticks=[0,25,50,75,100])
    colorbar.solids.set_rasterized(False)
    save(fig,"Fig_A6_WQ_Observation_Coverage")

    fig,axes=grid(FOCUS,height=6.1,width=9.35,top=.93,bottom=.095)
    fig.subplots_adjust(right=.88,wspace=.39,hspace=.43)
    months=sorted(month.Year_Month.unique())
    for ax,var in zip(axes,FOCUS):
        m=month.loc[month.Variable.eq(var)].pivot(index="Site_ID",columns="Year_Month",values="Coverage")
        m=m.reindex(index=sites,columns=months)
        image=coverage_mesh(ax,m,norm)
        ticks(ax,np.arange(5)*12+5.5,[x[:4] for x in months[::12]])
        ax.tick_params(axis="x",labelsize=7)
        ax.set_yticks(centers)
        ax.set_yticklabels(REACH)
        ax.tick_params(axis="y",length=0)
        for y in cut[:-1]:
            ax.axhline(y-.5,color="#384048",linewidth=.65)
        for col in (12,24,36,48):
            ax.axvline(col-.5,color="white",linewidth=.6,alpha=.9)
    cax=fig.add_axes([.913,.21,.018,.59])
    colorbar=fig.colorbar(image,cax=cax,label="Original observations (%)",ticks=[0,25,50,75,100])
    colorbar.solids.set_rasterized(False)
    save(fig,"Fig_A6b_Monthly_Coverage")


def main():
    FIGURES.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(REPORT/"Plot_Run.log",mode="w",encoding="utf-8")])
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    configure()
    longterm()
    event_non_event()
    low_do()
    event_types()
    observation_coverage()
    LOG.info("Completed all eight figure sets")


if __name__=="__main__":
    main()
