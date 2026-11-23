# -*- coding: utf-8 -*-
"""Comment 15: audit selection into the unchanged day-1/3/5/7/9 slope cohort.

Run this script after the observation-only references and M05 analyses.
This script reads existing study inputs; it neither retrains nor edits manuscripts.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

TYPES = ["Heatwave_Only", "Flood_Only", "Direct_Compound", "H5F_Compound", "F5H_Compound"]
OFFSETS = [1, 3, 5, 7, 9]
TYPE_MAP = {"hw": TYPES[0], "flood": TYPES[1], "direct": TYPES[2],
            "indirect_h2f": TYPES[3], "indirect_f2h": TYPES[4]}
METRICS = ["Duration_Days", "Heat_Excess_Mean_C", "Heat_Excess_Max_C",
           "Flow_Relative_Excess_Mean", "Flow_Relative_Excess_Max",
           "Heat_Active_Days", "Flow_Active_Days", "Event_Observed_Pct"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.15g")


def read_daily(path):
    d = pd.read_csv(path)
    d.columns = d.columns.str.strip().str.replace("\n", "", regex=False).str.replace("\r", "", regex=False)
    d = d.rename(columns={c: "Date" for c in d if c.lower() == "date"})
    d["Date"] = pd.to_datetime(d.Date, errors="raise")
    assert not d.Date.duplicated().any(), f"Duplicate dates: {path}"
    return d.sort_values("Date").reset_index(drop=True)


def load_detection_functions(path):
    """Load the event classification functions without running the registry."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    names = {"get_base_segments", "classify_complex_events", "pair_remaining_sequences"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = dict(pd=pd, np=np, HW_CONTINUE_DAYS=3, HW_MERGE_INTERVAL=2, COMP_INTERVAL=5)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_base_segments"], namespace["classify_complex_events"]


def numeric_summary(events):
    rows = []
    strata = [("Overall", "All", events)]
    for column in ["Event_Type", "Reach", "Season", "Compound_Group"]:
        strata += [(column, str(value), g) for value, g in events.groupby(column, dropna=False)]
    for dimension, level, g in strata:
        for metric in METRICS:
            vals = {c: g.loc[g.Cohort.eq(c), metric].dropna().to_numpy(float)
                    for c in ["Eligible", "Excluded"]}
            a, b = vals["Eligible"], vals["Excluded"]
            smd = np.nan
            if len(a) > 1 and len(b) > 1:
                denominator = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                smd = (a.mean() - b.mean()) / denominator if denominator > 0 else (0.0 if a.mean() == b.mean() else np.nan)
            for cohort, v in {"All": g[metric].dropna().to_numpy(float), **vals}.items():
                rows.append(dict(Dimension=dimension, Level=level, Metric=metric, Cohort=cohort,
                                 N=len(v), Missing_N=len(g if cohort == "All" else g[g.Cohort.eq(cohort)]) - len(v),
                                 Mean=np.mean(v) if len(v) else np.nan,
                                 SD=np.std(v, ddof=1) if len(v) > 1 else np.nan,
                                 Median=np.median(v) if len(v) else np.nan,
                                 Q25=np.quantile(v, .25) if len(v) else np.nan,
                                 Q75=np.quantile(v, .75) if len(v) else np.nan,
                                 Min=np.min(v) if len(v) else np.nan,
                                 Max=np.max(v) if len(v) else np.nan,
                                 SMD_Eligible_Minus_Excluded=smd))
    return pd.DataFrame(rows)


def categorical_summary(events):
    rows = []
    for col in ["Event_Type", "Reach", "Season", "Year", "Month", "Site_ID", "Compound_Group", "Event_Coverage_Class"]:
        for level, g in events.groupby(col, dropna=False):
            ne, nx = g.Cohort.eq("Eligible").sum(), g.Cohort.eq("Excluded").sum()
            rows.append(dict(Dimension=col, Level=level, All_N=len(g), Eligible_N=ne, Excluded_N=nx,
                             All_Pct=100 * len(g) / len(events),
                             Eligible_Composition_Pct=100 * ne / events.Cohort.eq("Eligible").sum(),
                             Excluded_Composition_Pct=100 * nx / events.Cohort.eq("Excluded").sum(),
                             Eligibility_Pct=100 * ne / len(g),
                             Eligible_Minus_Excluded_Composition_PP=100 * ne / events.Cohort.eq("Eligible").sum() - 100 * nx / events.Cohort.eq("Excluded").sum()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output_dir or root / "Result/Post_event_eligibility" / ("Comment15_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_Analysis")
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "run_status.json", {"Status": "RUNNING"})
    inputs = {}

    def track(path):
        path = Path(path)
        inputs[str(path)] = sha(path)
        return path

    try:
        master_path = track(root / "Result/Statistical_comparisons/Event_Level_Statistical_Data.csv")
        daily_path = track(root / "Result/Observation_only_references/20261111/20261111_逐日实测来源.csv")
        threshold_path = track(root / "Result/Event/Definition_sensitivity/20261111/20261111_M05实际阈值.csv")
        detection_path = track(root / "Code/02_Extreme_event_identification/02_Completed_event_registry/ED_F.py")
        track(root / "Code/04_DO_conditions_and_post_event_changes/03_Early_post_event_slopes/recovery.py")
        protected = {str(p): sha(p) for p in (root / "Submission_Documents").rglob("*.docx") if not p.name.startswith("~$")}
        assert all(sha(p) == h for p, h in protected.items()), "Manuscript hashes changed since latest integration; inspect before proceeding."

        events = pd.read_csv(master_path, parse_dates=["Start_Date", "End_Date"])
        assert not events.duplicated(["Site_ID", "Event_ID"]).any()
        assert set(events.Event_Type) == set(TYPES) and events.Site_ID.nunique() == 66
        events = events.sort_values(["Site_ID", "Start_Date", "End_Date"]).reset_index(drop=True)
        events["Next_Event_Start"] = events.groupby("Site_ID").Start_Date.shift(-1)
        assert ((events.Next_Event_Start > events.End_Date) | events.Next_Event_Start.isna()).all()
        daily = pd.read_csv(daily_path, parse_dates=["Date"])
        assert not daily.duplicated(["Site_ID", "Date"]).any()
        assert set(daily.Site_ID) == set(events.Site_ID)
        assert set(daily.DO_Source) <= {"QC_observed", "Imputed", "Missing"}
        assert np.array_equal(daily.DO_Source.eq("QC_observed"), daily.DO_QC.notna())
        assert np.allclose(daily.loc[daily.DO_QC.notna(), "DO_Final"], daily.DO_QC.dropna(), rtol=0, atol=1e-12)
        thresholds = pd.read_csv(threshold_path).query("Scenario == 'S0'").set_index("Site_ID")
        assert len(thresholds) == 66 and not thresholds.index.duplicated().any()
        basic, classify = load_detection_functions(detection_path)
        registry = pd.concat([pd.read_csv(track(p), parse_dates=["Start_Date", "End_Date"])
                              for p in sorted((root / "Data/Event_F/01_Registry_Data/Overall").glob("*.csv"))], ignore_index=True)
        registry = registry.rename(columns={"WQS_name": "Site_ID"}).set_index(["Site_ID", "Event_ID"])
        assert len(registry) == len(events) and registry.index.is_unique
        post = pd.concat([pd.read_csv(track(p), usecols=["Event_ID", "Event_Type", "WQS_name", "Date", "DO", "Day_Offset"], parse_dates=["Date"])
                          for p in sorted((root / "Data/Event_F/04_Post_Recovery_Data/Overall").glob("*.csv"))], ignore_index=True)
        post = post.rename(columns={"WQS_name": "Site_ID"})
        assert not post.duplicated(["Site_ID", "Event_ID", "Date"]).any()
        post_groups = {(s, e): g.set_index("Date") for (s, e), g in post.groupby(["Site_ID", "Event_ID"])}
        assert set(post_groups) <= set(zip(events.Site_ID, events.Event_ID))
        rows, date_rows, intensity_rows, component_rows, site_rows = [], [], [], [], []
        for number, (site, es) in enumerate(events.groupby("Site_ID"), 1):
            d = daily[daily.Site_ID.eq(site)].sort_values("Date").set_index("Date")
            assert d.index.equals(pd.date_range("2021-01-01", "2025-12-31", name="Date"))
            o = read_daily(track(root / "Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR" / f"{site}_Imputed.csv"))
            b = read_daily(track(root / "Data/Meteorology_data/1991-2020/YR" / f"{site}.csv"))
            q = read_daily(track(root / "Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned" / f"{site}.csv"))
            assert o.Date.equals(pd.Series(d.index, name="Date"))
            assert b.Date.min() == pd.Timestamp("1991-01-01") and b.Date.max() == pd.Timestamp("2020-12-31")
            assert len(b) == len(pd.date_range("1991-01-01", "2020-12-31"))
            assert np.allclose(o.DO, d.DO_Final, rtol=0, atol=1e-12, equal_nan=True)
            raw_qc = q.set_index("Date").DO.reindex(d.index)
            assert np.allclose(raw_qc, d.DO_QC, rtol=0, atol=1e-12, equal_nan=True)
            assert np.array_equal(o.Original_DO.notna(), d.DO_QC.notna())
            assert np.isfinite(o[["Tmax", "Q"]].to_numpy()).all()
            t_th = float(np.percentile(b.Tmax.dropna(), 90))
            q_th = float(np.percentile(o.Q.dropna(), 95))
            assert q_th > 0
            assert abs(t_th - thresholds.loc[site, "Tmax_Annual_P90"]) < 2e-6
            assert abs(q_th - thresholds.loc[site, "Q_Threshold"]) < 2e-6
            h, f = basic(o.copy(), "hw", t_th, q_th), basic(o.copy(), "flood", t_th, q_th)
            c, rh, rf = classify(h, f)
            reconstructed = sorted(c + rh + rf, key=lambda x: (x["start"], x["end"]))
            assert len(reconstructed) == len(es)
            component_assigned = set()
            env = o.set_index("Date")
            for i, (_, event) in enumerate(es.iterrows()):
                site_key = (site, event.Event_ID)
                re = reconstructed[i]
                assert event.Event_ID == f"{site}_EV{i+1:03d}"
                assert event.Start_Date == re["start"] and event.End_Date == re["end"]
                assert event.Event_Type == TYPE_MAP[re.get("comp_subtype", re.get("type"))]
                reg = registry.loc[site_key]
                assert reg.Event_Type == event.Event_Type and reg.Start_Date == event.Start_Date and reg.End_Date == event.End_Date
                nday = (event.End_Date - event.Start_Date).days + 1
                assert reg.Total_Days == nday
                dates = pd.date_range(event.Start_Date, event.End_Date)
                component_masks = {"hw": pd.Series(False, index=dates), "flood": pd.Series(False, index=dates)}
                component_counts = {"hw": 0, "flood": 0}
                for hazard, segments in [("hw", h), ("flood", f)]:
                    for k, seg in enumerate(segments):
                        overlap = not (seg["end"] < event.Start_Date or seg["start"] > event.End_Date)
                        if not overlap:
                            continue
                        assert event.Start_Date <= seg["start"] <= seg["end"] <= event.End_Date
                        assert (hazard, k) not in component_assigned
                        component_assigned.add((hazard, k))
                        component_masks[hazard].loc[seg["start"]:seg["end"]] = True
                        component_counts[hazard] += 1
                        component_rows.append(dict(Site_ID=site, Event_ID=event.Event_ID, Hazard=hazard,
                                                   Component_Start=seg["start"], Component_End=seg["end"], Component_Days=seg["days"]))
                w = env.loc[dates]
                hm = component_masks["hw"].to_numpy() & w.Tmax.ge(t_th).to_numpy()
                fm = component_masks["flood"].to_numpy() & w.Q.ge(q_th).to_numpy()
                assert hm.any() == (event.Event_Type != "Flood_Only")
                assert fm.any() == (event.Event_Type != "Heatwave_Only")
                heat = (w.Tmax.to_numpy() - t_th)[hm]
                flow = (w.Q.to_numpy() / q_th - 1)[fm]
                obs = d.loc[dates].DO_Source.eq("QC_observed")
                assert d.loc[dates].Event_ID.eq(event.Event_ID).all()
                assert d.loc[dates].Event_Type.eq(event.Event_Type).all()
                for j, day in enumerate(dates):
                    intensity_rows.append(dict(Site_ID=site, Event_ID=event.Event_ID, Date=day, Tmax=w.Tmax.iloc[j], Q=w.Q.iloc[j],
                                               Tmax_P90=t_th, Q_P95=q_th, Heat_Component_Member=component_masks["hw"].iloc[j],
                                               Flow_Component_Member=component_masks["flood"].iloc[j], Heat_Active=hm[j], Flow_Active=fm[j],
                                               Heat_Excess_C=w.Tmax.iloc[j] - t_th if hm[j] else np.nan,
                                               Flow_Relative_Excess=w.Q.iloc[j] / q_th - 1 if fm[j] else np.nan))
                targets = event.End_Date + pd.to_timedelta(OFFSETS, unit="D")
                support = d.reindex(targets)
                interrupted = pd.notna(event.Next_Event_Start) and event.Next_Event_Start <= targets[-1]
                end_truncated = targets[-1] > d.index.max()
                record_missing = not targets.isin(d.index).all()
                do_invalid = not np.isfinite(support.DO_Final.to_numpy(float)).all()
                eligible = not (interrupted or end_truncated or record_missing or do_invalid)
                assert eligible == bool(np.isfinite(event.Recovery_Slope))
                slope = np.nan
                if eligible:
                    x = np.array(OFFSETS, dtype=float)
                    slope = float(np.dot(x - x.mean(), support.DO_Final.to_numpy(float)) / np.sum((x - x.mean()) ** 2))
                    assert abs(slope - event.Recovery_Slope) <= 5.00001e-5
                pe = post_groups.get(site_key, pd.DataFrame(columns=["DO", "Day_Offset", "Event_Type"], index=pd.DatetimeIndex([], name="Date")))
                valid_end = min(event.End_Date + pd.Timedelta(days=30), d.index.max())
                if pd.notna(event.Next_Event_Start):
                    valid_end = min(valid_end, event.Next_Event_Start - pd.Timedelta(days=1))
                expected_dates = pd.date_range(event.End_Date + pd.Timedelta(days=1), valid_end)
                assert pe.index.sort_values().equals(expected_dates.rename("Date")), (site_key, len(pe), len(expected_dates), pe.index.tolist(), expected_dates.tolist())
                if len(pe):
                    assert pe.Event_Type.eq(event.Event_Type).all()
                    assert np.allclose(pe.DO, d.loc[pe.index].DO_Final, rtol=0, atol=1e-12, equal_nan=True)
                    assert np.array_equal(pe.Day_Offset, (pe.index - event.End_Date).days)
                reason = "Eligible" if eligible else ("Next_event_on_or_before_day9" if interrupted else
                          "Study_end_before_day9" if end_truncated else "Missing_daily_record" if record_missing else "Invalid_required_DO")
                for offset, day in zip(OFFSETS, targets):
                    present = day in d.index
                    v = support.loc[day]
                    uninterrupted = pd.isna(event.Next_Event_Start) or day < event.Next_Event_Start
                    date_rows.append(dict(Site_ID=site, Event_ID=event.Event_ID, Offset=offset, Date=day,
                                          Daily_Record_Present=present, DO_Final=v.DO_Final, DO_QC=v.DO_QC, DO_Source=v.DO_Source,
                                          Before_Next_Event=uninterrupted, In_Study=present,
                                          Present_In_Original_Post_Slice=day in pe.index, Eligible=eligible))
                row = event.to_dict()
                row.update(Duration_Days=nday, Year=event.Start_Date.year, Month=event.Start_Date.month,
                           Cohort="Eligible" if eligible else "Excluded", Eligible=eligible, Primary_Reason=reason,
                           Next_Event_On_Or_Before_Day9=interrupted, Study_End_Before_Day9=end_truncated,
                           Required_Daily_Record_Missing=record_missing, Required_DO_Invalid=do_invalid,
                           Next_Event_Gap_Days=(event.Next_Event_Start - event.End_Date).days - 1 if pd.notna(event.Next_Event_Start) else np.nan,
                           No_Following_Event=pd.isna(event.Next_Event_Start), Study_Days_Remaining=(d.index.max() - event.End_Date).days,
                           Post_Observed_Points=int(support.DO_Source.eq("QC_observed").sum()),
                           Post_All_Five_Observed=bool(support.DO_Source.eq("QC_observed").all()),
                           Recomputed_Eligible_Slope=slope,
                           Event_Observed_Days=int(obs.sum()), Event_Observed_Pct=100 * obs.mean(),
                           Event_Coverage_Class="Fully_observed" if obs.all() else "Entirely_reconstructed" if not obs.any() else "Partially_observed",
                           Heat_Components=component_counts["hw"], Flow_Components=component_counts["flood"],
                           Heat_Active_Days=int(hm.sum()) if hm.any() else np.nan,
                           Flow_Active_Days=int(fm.sum()) if fm.any() else np.nan,
                           Heat_Excess_Mean_C=float(heat.mean()) if len(heat) else np.nan,
                           Heat_Excess_Max_C=float(heat.max()) if len(heat) else np.nan,
                           Flow_Relative_Excess_Mean=float(flow.mean()) if len(flow) else np.nan,
                           Flow_Relative_Excess_Max=float(flow.max()) if len(flow) else np.nan)
                rows.append(row)
            assert len(component_assigned) == len(h) + len(f)
            site_rows.append(dict(Site_ID=site, Events=len(es), Daily_Rows=len(d), Historical_Rows=len(b),
                                  Tmax_P90=t_th, Q_P95=q_th, Heat_Components=len(h), Flow_Components=len(f),
                                  Original_Event_Definition_Reproduced=True, Original_Post_Slices_Reproduced=True,
                                  Daily_QC_And_Completed_DO_Reproduced=True))
            if number % 15 == 0:
                print(f"Validated {number}/66 sites", flush=True)
        result = pd.DataFrame(rows)
        dates = pd.DataFrame(date_rows)
        intensity = pd.DataFrame(intensity_rows)
        cat, num = categorical_summary(result), numeric_summary(result)
        assert len(result) == len(events)
        assert not result.duplicated(["Site_ID", "Event_ID"]).any()
        eligible = result[result.Eligible]
        positive = int(eligible.Recovery_Slope.gt(0).sum())
        assert positive == int(eligible.Recomputed_Eligible_Slope.gt(0).sum()), "Rounded-vs-unrounded sign mismatch"
        post_support = []
        for level, g in eligible.groupby("Post_All_Five_Observed"):
            post_support.append(dict(Post_All_Five_Observed=level, N=len(g), Positive_N=int(g.Recovery_Slope.gt(0).sum()),
                                     Positive_Pct=100 * g.Recovery_Slope.gt(0).mean()))
        exclusion_flags = ["Next_Event_On_Or_Before_Day9", "Study_End_Before_Day9", "Required_Daily_Record_Missing", "Required_DO_Invalid"]
        summary = dict(All_Events=len(result), Sites=result.Site_ID.nunique(), Eligible=len(eligible),
                       Excluded=int((~result.Eligible).sum()), Eligible_Pct=100 * result.Eligible.mean(),
                       Positive_Slopes=positive, Nonpositive_Slopes=len(eligible)-positive,
                       Positive_Slope_Pct=100 * positive/len(eligible),
                       Primary_Reasons={str(k): int(v) for k,v in result.Primary_Reason.value_counts().items()},
                       Overlapping_Exclusion_Flags={c: int(result[c].sum()) for c in exclusion_flags},
                       Eligible_Post_All_Five_Observed=int(eligible.Post_All_Five_Observed.sum()),
                       Eligible_With_At_Least_One_Imputed_Post_Point=int((~eligible.Post_All_Five_Observed).sum()),
                       Target_Date_Trace_Rows=len(dates), Intensity_Day_Trace_Rows=len(intensity),
                       Original_Post_Slice_Rows=len(post),
                       Slope_Rounding_Tolerance=5.00001e-5,
                       Maximum_Absolute_Slope_Difference=float((eligible.Recomputed_Eligible_Slope - eligible.Recovery_Slope).abs().max()),
                       Scope="Descriptive selection audit; positive slopes are conditional on eligibility, not demonstrated recovery to baseline.")
        for name, df in {"event_selection_and_intensity": result, "required_post_dates": dates,
                         "intensity_day_trace": intensity, "original_hazard_components": pd.DataFrame(component_rows),
                         "site_input_checks": pd.DataFrame(site_rows), "categorical_comparisons": cat,
                         "numeric_comparisons": num, "eligible_slope_by_post_observation_support": pd.DataFrame(post_support)}.items():
            csv(df, out / f"{name}.csv")
        csv(result.groupby(["Site_ID", "Reach", "Event_Type", "Cohort"], dropna=False).size().rename("N").reset_index(), out / "site_type_counts.csv")
        lines = ["# Comment 15：事件后斜率分析的选择性纳入核查", "",
                 f"全部 {len(result):,} 次事件中，纳入 {len(eligible):,} 次（{summary['Eligible_Pct']:.2f}%），排除 {summary['Excluded']:,} 次。",
                 f"纳入事件中正斜率为 {positive}/{len(eligible)}（{summary['Positive_Slope_Pct']:.2f}%）。斜率使用事件结束后第1、3、5、7、9天的DO；未改变原纳入规则。", "",
                 "## 排除原因", ""]
        lines += [f"- {k}：{v} 次。" for k,v in summary["Primary_Reasons"].items()]
        lines += ["", "原因标记可以重叠；Primary_Reason按后续事件、研究期末、缺少日期、无效DO的顺序归类，计数与总数核对。", "", "## 事件类型", "",
                  "|类型|全部|纳入|排除|纳入率|", "|---|---:|---:|---:|---:|"]
        for kind in TYPES:
            r = cat[cat.Dimension.eq("Event_Type") & cat.Level.eq(kind)].iloc[0]
            lines.append(f"|{kind}|{r.All_N}|{r.Eligible_N}|{r.Excluded_N}|{r.Eligibility_Pct:.2f}%|")
        lines += ["", "## 持续时间与暴露强度", "",
                  "SMD=(纳入均值−排除均值)/sqrt[(纳入样本方差+排除样本方差)/2]。仅作分布差异描述，不进行独立事件假设下的显著性检验。", "",
                  "|指标|纳入 n；均值；中位数[Q25,Q75]|排除 n；均值；中位数[Q25,Q75]|SMD|", "|---|---|---|---:|"]
        for metric in METRICS:
            g = num[num.Dimension.eq("Overall") & num.Metric.eq(metric)].set_index("Cohort")
            a,b = g.loc["Eligible"],g.loc["Excluded"]
            fmt=lambda x:f"{int(x.N)}；{x.Mean:.3f}；{x.Median:.3f}[{x.Q25:.3f},{x.Q75:.3f}]"
            lines.append(f"|{metric}|{fmt(a)}|{fmt(b)}|{a.SMD_Eligible_Minus_Excluded:.3f}|")
        lines += ["", "## 定义与解释边界", "",
                  "热强度仅在原热浪组成过程中达到历史1991–2020年全年Tmax第90百分位阈值的日期计算：Tmax−P90，单位°C；合并过程中的低温间隔日不计入强度均值。",
                  "高流量强度仅在原高流量组成过程中达到本站2021–2025年Q第95百分位阈值的日期计算：Q/P95−1，无量纲。非对应灾害的强度留空，不作为0。复合事件两种强度分别报告。",
                  "完全复现原事件识别和配对后提取组成过程，未改变事件窗口。按类型、河段、季节的分层对照见numeric_comparisons.csv；按断面和月份、年份的组成见categorical_comparisons.csv。",
                  f"纳入事件中 {summary['Eligible_Post_All_Five_Observed']} 次五个斜率点均为原始质控观测；{summary['Eligible_With_At_Least_One_Imputed_Post_Point']} 次含插补点。事件期覆盖率与事件后五点的覆盖率是两个不同指标。",
                  f"{summary['Eligible_Pct']:.1f}%描述有足够未受后续事件打断窗口的事件，不应外推为全部事件的恢复比例。正斜率不等于恢复至事件前基线。未给被打断事件拟合所谓无干扰斜率，也未用加权假定消除选择偏差。",
                  "Next_Event_Gap_Days是结束日至下一事件之间的完整间隔日数；下一事件在第9天开始时仍应排除。无下一事件的间隔留空并以No_Following_Event标识，不能当作0天或无限长已观察间隔。",
                  "分析为描述性审查，同断面事件存在相关性；没有把事件当作独立重复进行p值比较。强度的总体对照也受事件类型组成影响，应结合类型内结果解释。",
                  "本次不修改V13/V14，不改变原分析。", ""]
        snapshot = out / "Code_Snapshot"
        snapshot.mkdir()
        for p in sorted(Path(__file__).parent.iterdir()):
            if p.is_file() and p.suffix in {".py", ".md"}:
                shutil.copy2(p, snapshot / p.name)
        assert all(sha(p) == h for p,h in inputs.items()), "An input changed during execution"
        assert all(sha(p) == h for p,h in protected.items()), "Protected manuscript changed during execution"
        dump(out / "run_provenance.json", dict(Inputs_SHA256=inputs, Protected_Manuscripts_SHA256=protected,
                                               Python=sys.version, Executable=sys.executable,
                                               Numpy=np.__version__, Pandas=pd.__version__,
                                               Code_Snapshot_SHA256={p.name:sha(p) for p in snapshot.iterdir()}))
        dump(out / "run_status.json", {"Status": "COMPLETE", "Output": str(out)})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Output: {out}", flush=True)
    except Exception as error:
        dump(out / "run_status.json", {"Status": "FAILED", "Error": repr(error)})
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
