"""Observed-only Yellow River water-quality background and grouped comparisons."""
from pathlib import Path
import logging

import numpy as np
import pandas as pd

# Paths and analysis parameters
ROOT = Path(__file__).resolve().parents[2]
CLEANED = ROOT / "Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned"
STATIONS = ROOT / "Config/Study_Station_Mapping.csv"
REGISTRY = ROOT / "Data/Event_F/Event_Registry_For_Validation.csv"
S0 = ROOT / "Result/Event/Pairing_priority/20261111/baseline_registry.csv"
PROVENANCE = ROOT / "Result/Observation_only_references/20261111/20261111_逐日实测来源.csv"
S27 = ROOT / "Result/Jiagaiqiao_source_audit/20261111_Audit/observed_chemistry_by_DO_group.csv"
OUT = ROOT / "Result/WQ_Background_Analysis"
DIRS = dict(enumerate(("01_Data_Quality", "02_Longterm_Background", "03_Event_NonEvent",
                       "04_Event_Type", "05_Low_DO", "06_Figures", "07_Report"), 1))
DATES = pd.date_range("2021-01-01", "2025-12-31", name="Date")
REACHES = {"UR": 28, "MR": 33, "LR": 5}
REACH_NAMES = {"Upstream": "UR", "Midstream": "MR", "Downstream": "LR"}
EVENT_MAP = {"Heatwave_Only": "Heatwave only", "Flood_Only": "High-flow only",
             "Direct_Compound": "DC", "H5F_Compound": "H5F", "F5H_Compound": "F5H"}
EVENT_COUNTS = dict(zip(EVENT_MAP, (1191, 1035, 132, 102, 88)))
FIELDS = {"DO": "DO", "WT": "WT", "pH": "pH", "EC": "EC", "Turb": "Turb",
          "CODₘₙ": "CODMn", "NH₃-N": "NH3_N", "TP": "TP", "TN": "TN"}
UNITS = {"DO": "mg L⁻¹", "WT": "°C", "pH": "dimensionless", "EC": "µS cm⁻¹",
         "Turb": "NTU", **dict.fromkeys(("CODMn", "NH3_N", "TP", "TN"), "mg L⁻¹")}
CHEM = ["NH3_N", "TN", "TP", "CODMn"]
PRIORITY = [*CHEM, "WT", "DO"]
GROUPS = ["Very_Low_DO", "Low_DO", "Higher_DO"]
LONG_COVERAGE = .5
MIN_YEARS = 3
EVENT_MIN = 10
REFERENCE_MIN = 3
STATION_KEY = ["Reach", "Site_ID", "Variable"]

LOG = logging.getLogger("wq")


def read(path, columns):
    data = pd.read_csv(path, usecols=columns, encoding="utf-8-sig").loc[:, columns]
    LOG.info("Read %s (%d records)", path, len(data))
    return data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(data, key, source):
    require(data[key].notna().all().all() and not data.duplicated(key).any(),
            f"Missing/duplicate key {key}: {source}")


def write(frame, section, filename):
    path = OUT / DIRS[section] / filename
    data = frame.copy()
    if "Variable" in data and "Unit" not in data:
        data.insert(data.columns.get_loc("Variable") + 1, "Unit", data["Variable"].map(UNITS))
    if "Metric" in data:
        for metric, unit in {"Ratio_C": "dimensionless", "Relative_Change": "%"}.items():
            data.loc[data.Metric.eq(metric), "Unit"] = unit
    # Nullable counts retain blank entries where the count is inapplicable.
    for column in data:
        if (column.endswith(("_N", "_Days", "_Sites", "_Events", "_Count", "_Years", "_Months", "_Pairs"))
                and pd.api.types.is_numeric_dtype(data[column])
                and column not in {"Mean_Site_Coverage", "Median_Site_Coverage"}):
            numeric = data[column].dropna()
            if (numeric % 1 == 0).all():
                data[column] = data[column].astype("Int64")
    data.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    LOG.info("Wrote %s (%d rows)", path, len(data))
    return frame


def read_inputs():
    site = read(STATIONS, ["WQS_name", "Region", "Status"])
    require(site["Status"].eq("Retained").all(), "Mapping includes non-retained stations")
    site = site.rename(columns={"WQS_name": "Site_ID", "Region": "Reach"})
    unique(site, ["Site_ID"], STATIONS)
    require(site["Reach"].value_counts().to_dict() == REACHES, "Unexpected reach composition")
    require(len(DATES) == 1826, "Incorrect daily calendar")
    actual_files = {p.stem for p in CLEANED.glob("*.csv")}
    require(actual_files == set(site.Site_ID), "Retained QC file names differ from the station registry")
    parts = []
    for row in site.itertuples(index=False):
        path = CLEANED / f"{row.Site_ID}.csv"
        data = read(path, ["WQS_name", "Date", *FIELDS]).rename(
            columns={"WQS_name": "Site_ID", **FIELDS})
        data["Date"] = pd.to_datetime(data["Date"], errors="raise")
        unique(data, ["Site_ID", "Date"], path)
        require(data.Site_ID.eq(row.Site_ID).all(), f"Station identifier mismatch: {path}")
        require(pd.DatetimeIndex(data.Date.sort_values()).equals(DATES), f"Calendar mismatch: {path}")
        values = data[list(UNITS)].apply(pd.to_numeric, errors="raise")
        require(not np.isinf(values.to_numpy()).any(), f"Non-finite measurement: {path}")
        data[list(UNITS)] = values
        data["Reach"] = row.Reach
        parts.append(data)
    daily = pd.concat(parts, ignore_index=True)
    key = ["Site_ID", "Date"]
    unique(daily, key, CLEANED)
    registry_fields = ["Site_ID", "Event_ID", "Event_Type", "Start_Date", "End_Date", "Total_Days", "Region"]
    events = read(REGISTRY, registry_fields)
    baseline = read(S0, registry_fields)
    require(events.sort_values(registry_fields).reset_index(drop=True).equals(
        baseline.sort_values(registry_fields).reset_index(drop=True)), "Registry differs from S0")
    unique(events, ["Site_ID", "Event_ID"], REGISTRY)
    require(events.Event_Type.value_counts().to_dict() == EVENT_COUNTS, "Unexpected S0 event counts")
    require(events.Region.eq(events.Site_ID.map(site.set_index("Site_ID").Reach)).all(),
            "S0 reach differs from the station table")
    for field in ("Start_Date", "End_Date"):
        events[field] = pd.to_datetime(events[field], errors="raise")
    require(events.Start_Date.ge(DATES[0]).all() and events.End_Date.le(DATES[-1]).all()
            and events.Total_Days.gt(0).all()
            and events.Total_Days.eq((events.End_Date-events.Start_Date).dt.days+1).all(),
            "Invalid S0 windows")
    events = events.rename(columns={"Region": "Reach"})
    events["Event_Type"] = events.Event_Type.map(EVENT_MAP)
    windows = events.loc[events.index.repeat(events.Total_Days),
                         ["Site_ID", "Event_ID", "Event_Type", "Start_Date"]].copy()
    windows["Date"] = windows.Start_Date + pd.to_timedelta(windows.groupby(level=0).cumcount(), unit="D")
    windows = windows.drop(columns="Start_Date")
    unique(windows, key, "expanded S0")
    daily = daily.merge(windows, on=key, how="left", validate="one_to_one")
    daily["Period"] = np.where(daily.Event_ID.notna(), "Event", "NonEvent")
    source = read(PROVENANCE, [*key, "DO_QC", "DO_Source", "Event_ID", "Is_Event", "Reach"])
    source["Date"] = pd.to_datetime(source.Date, errors="raise")
    unique(source, key, PROVENANCE)
    require(len(source) == len(daily), "Provenance calendar size differs")
    check = daily.merge(source, on=key, suffixes=("", "_source"), how="left", validate="one_to_one")
    is_observed = check.DO_Source.eq("QC_observed")
    require(check.DO_Source.isin(["QC_observed", "Imputed"]).all()
            and is_observed.eq(check.DO.notna()).all()
            and np.allclose(check.DO, check.DO_QC, equal_nan=True, atol=1e-10, rtol=0),
            "QC DO does not agree with original-observation labels")
    require(check.Is_Event.eq(check.Period.eq("Event")).all()
            and check.Event_ID.fillna("").eq(check.Event_ID_source.fillna("")).all()
            and check.Reach.eq(check.Reach_source.map(REACH_NAMES)).all(),
            "Source event/reach assignments do not reconcile")
    daily["DO_Source"] = check.DO_Source.to_numpy()
    daily = daily.sort_values(["Site_ID", "Date"]).reset_index(drop=True)
    daily["Year"] = daily.Date.dt.year
    daily["Month"] = daily.Date.dt.month
    daily["Year_Month"] = daily.Date.dt.strftime("%Y-%m")
    daily["Season"] = daily.Month.map(dict(zip(range(1, 13), (
        "Winter", "Winter", "Spring", "Spring", "Spring", "Summer", "Summer", "Summer",
        "Autumn", "Autumn", "Autumn", "Winter"))))
    LOG.info("Validated 66 stations, duplicate pairs 0, 1,826 days (2024 has 366), %s",
             daily.Period.value_counts().to_dict())
    LOG.info("S0 %s; DO provenance %s; %d imputed DO days excluded from observation counts",
             events.Event_Type.value_counts().to_dict(), check.DO_Source.value_counts().to_dict(),
             (~is_observed).sum())
    LOG.info("QC source is upstream of imputation for all variables; missing and genuine zeros: %s / %s",
             daily[list(UNITS)].isna().sum().to_dict(), daily[list(UNITS)].eq(0).sum().to_dict())
    columns = ["Reach", "Site_ID", "Date", "Year", "Month", "Year_Month", "Season",
               "Period", "Event_ID", "Event_Type"]
    long = daily.melt(id_vars=columns, value_vars=list(UNITS), var_name="Variable", value_name="Value")
    return daily, long, events


def coverage(frame):
    frame["Missing_Days"] = frame.Theoretical_Days - frame.Valid_N
    frame["Coverage"] = frame.Valid_N.div(frame.Theoretical_Days.where(frame.Theoretical_Days.gt(0)))
    frame["Missing_Rate"] = 1-frame.Coverage
    return frame


def summary(frame, keys, value="Value", counts=True):
    grouped = frame.groupby(keys, sort=True, observed=True, dropna=False)[value]
    result = grouped.agg(Mean="mean", Median="median", SD="std", Min="min", Max="max", Valid_N="count")
    if counts:
        result["Theoretical_Days"] = grouped.size()
    for name, q in {"P05": .05, "P25": .25, "P75": .75, "P95": .95}.items():
        result[name] = grouped.quantile(q)
    return result.reset_index()


def a01(long):
    def time_coverage(period):
        return coverage(long.groupby(STATION_KEY + period).Value.agg(
            Theoretical_Days="size", Valid_N="count").reset_index())

    annual = time_coverage(["Year"])
    monthly = time_coverage(["Year_Month", "Year", "Month"])
    seasonal = time_coverage(["Season"])
    station = time_coverage([])
    monthly["Eligible_Month"] = monthly.Coverage.ge(LONG_COVERAGE)
    month_count = monthly.groupby(STATION_KEY).Eligible_Month.sum().reset_index(name="Eligible_Months")
    runs = long[STATION_KEY + ["Value"]].copy()
    runs["Observed"] = runs.Value.notna()
    runs["Run"] = runs.groupby(STATION_KEY, sort=False).Observed.cumsum()
    longest = (runs.loc[~runs.Observed].groupby(STATION_KEY+["Run"]).size()
               .groupby(level=[0, 1, 2]).max().rename("Longest_Missing_Run_Days").reset_index())
    station = station.merge(month_count, on=STATION_KEY).merge(longest, on=STATION_KEY, how="left")
    station["Longest_Missing_Run_Days"] = station.Longest_Missing_Run_Days.fillna(0)
    station["Contributing_Sites"] = station.Valid_N.gt(0).astype(int)
    for period in (annual, monthly, seasonal):
        period["Contributing_Sites"] = period.Valid_N.gt(0).astype(int)

    def reach_coverage(frame, label, windows):
        result = (frame.groupby(["Reach", "Variable", *windows])
                  .agg(Theoretical_Days=("Theoretical_Days", "sum"), Valid_N=("Valid_N", "sum"),
                       Contributing_Sites=("Contributing_Sites", "sum"),
                       Total_Sites=("Site_ID", "size"), Mean_Site_Coverage=("Coverage", "mean"),
                       Min_Site_Coverage=("Coverage", "min"), Max_Site_Coverage=("Coverage", "max"))
                  .reset_index())
        return coverage(result.assign(Window=label))
    reach = pd.concat([reach_coverage(station, "All", []), reach_coverage(annual, "Annual", ["Year"]),
                       reach_coverage(seasonal, "Seasonal", ["Season"]),
                       reach_coverage(monthly, "Monthly", ["Year_Month"])], ignore_index=True)
    for frame, name in [(station, "Station_Variable"), (annual, "Annual"), (monthly, "Monthly"),
                        (seasonal, "Seasonal"), (reach, "Reach_Coverage_Summary")]:
        write(frame, 1, f"A01_{name}{'' if name.endswith('Summary') else '_Coverage'}.csv")
    return station, reach


def reach_descriptive(stations, keys, value, counts=None):
    """Summarize independent station estimates and retain their supporting records."""
    out = summary(stations, keys, value=value, counts=False).rename(columns={"Valid_N": "Contributing_Sites"})
    used = stations.loc[stations[value].notna()]
    if counts:
        recorded = used.groupby(keys, dropna=False)[counts].sum().reset_index()
        out = out.merge(recorded, on=keys, how="left")
        out[counts] = out[counts].fillna(0)
    out["Statistic_Unit"] = "Site equal"
    return out


def a02(long):
    station = coverage(summary(long, STATION_KEY))
    annual = coverage(summary(long, STATION_KEY + ["Year"]))
    monthly = coverage(summary(long, STATION_KEY + ["Year_Month", "Year", "Month"]))
    for frame in (station, annual, monthly):
        frame["Contributing_Sites"] = frame.Valid_N.gt(0).astype(int)
    annual["Eligible_Year"] = annual.Coverage.ge(LONG_COVERAGE)
    monthly["Eligible_Month"] = monthly.Coverage.ge(LONG_COVERAGE)
    m2 = (annual.loc[annual.Eligible_Year].groupby(STATION_KEY)
          .agg(M2=("Mean", "mean"), M2_Eligible_Years=("Year", "size"),
               M2_Years=("Year", lambda s: "|".join(map(str, s))),
               M2_Valid_N=("Valid_N", "sum"), M2_Theoretical_Days=("Theoretical_Days", "sum")).reset_index())
    m2["M2"] = m2.M2.where(m2.M2_Eligible_Years.ge(MIN_YEARS))
    by_month = (monthly.loc[monthly.Eligible_Month].groupby(STATION_KEY+["Month"])
                .agg(Balanced_Month_Mean=("Mean", "mean"), Month_Eligible_Years=("Year", "size"),
                     M3_Valid_N=("Valid_N", "sum"), M3_Theoretical_Days=("Theoretical_Days", "sum"))
                .reset_index())
    by_month["Balanced_Month_Qualified"] = by_month.Month_Eligible_Years.ge(MIN_YEARS)
    monthly = monthly.merge(by_month.drop(columns=["M3_Valid_N", "M3_Theoretical_Days"]),
                            on=STATION_KEY+["Month"], how="left")
    monthly["Balanced_Month_Mean"] = monthly.Balanced_Month_Mean.where(
        monthly.Balanced_Month_Qualified.eq(True))
    m3 = (by_month.loc[by_month.Balanced_Month_Qualified].groupby(STATION_KEY)
          .agg(M3=("Balanced_Month_Mean", "mean"), M3_Eligible_Months=("Month", "size"),
               M3_Valid_N=("M3_Valid_N", "sum"), M3_Theoretical_Days=("M3_Theoretical_Days", "sum"))
          .reset_index())
    m3["M3"] = m3.M3.where(m3.M3_Eligible_Months.eq(12))
    station = station.assign(M1=station.Mean).merge(m2, on=STATION_KEY, how="left")
    station = station.merge(m3, on=STATION_KEY, how="left")
    station[["M2_Eligible_Years", "M3_Eligible_Months"]] = (
        station[["M2_Eligible_Years", "M3_Eligible_Months"]].fillna(0))
    station["M3_Available"] = station.M3.notna()
    methods = []
    for method in ("M1", "M2", "M3"):
        x = station[STATION_KEY].assign(
            Method=method, Estimate=station[method],
            Valid_N=station["Valid_N" if method=="M1" else f"{method}_Valid_N"],
            Theoretical_Days=station["Theoretical_Days" if method=="M1" else f"{method}_Theoretical_Days"])
        x[["Valid_N", "Theoretical_Days"]] = x[["Valid_N", "Theoretical_Days"]].fillna(0)
        methods.append(x)
    method_station = coverage(pd.concat(methods, ignore_index=True))
    reach = coverage(reach_descriptive(method_station, ["Reach", "Variable", "Method"], "Estimate",
                                       ["Valid_N", "Theoretical_Days"]))
    reach["Total_Sites"] = reach.Reach.map(REACHES)
    seasonal_station = coverage(summary(long, STATION_KEY+["Season"]))
    reach_season = coverage(reach_descriptive(seasonal_station, ["Reach", "Variable", "Season"], "Mean",
                                              ["Valid_N", "Theoretical_Days"]))
    reach_season["Total_Sites"] = reach_season.Reach.map(REACHES)
    reach_season["Method"] = "Pooled valid daily observations within station and season"
    for name, data in [("Station_5Year", station), ("Station_Annual", annual), ("Station_Monthly", monthly),
                       ("Reach_5Year", reach), ("Reach_Seasonal", reach_season)]:
        write(data, 2, f"A02_{name}_WQ_Statistics.csv")
    write(method_station.assign(Contributing_Sites=method_station.Estimate.notna().astype(int)),
          2, "A02_Longterm_Method_Comparison.csv")
    return reach, station


def a03(long):
    direct = coverage(summary(long, STATION_KEY+["Period"])).assign(Comparison="Direct")
    direct["Contributing_Sites"] = direct.Valid_N.gt(0).astype(int)
    reach_direct = coverage(reach_descriptive(direct, ["Reach", "Variable", "Period"], "Mean",
                                               ["Valid_N", "Theoretical_Days"]))
    reach_direct["Total_Sites"] = reach_direct.Reach.map(REACHES)

    keys = STATION_KEY + ["Year", "Season"]
    part = long.groupby(keys+["Period"]).Value.agg(Mean="mean", Valid_N="count", Theoretical_Days="size")
    pairs = part.unstack("Period")
    pairs.columns = [f"{period}_{measure}" for measure, period in pairs.columns]
    pairs = pairs.reset_index()
    for period in ("Event","NonEvent"):
        counts = [f"{period}_Valid_N",f"{period}_Theoretical_Days"]
        pairs[counts] = pairs[counts].fillna(0)
        pairs[f"{period}_Missing_Rate"] = 1-pairs[counts[0]].div(pairs[counts[1]].replace(0,np.nan))
    event_ok = pairs.Event_Valid_N.ge(EVENT_MIN)
    non_ok = pairs.NonEvent_Valid_N.ge(EVENT_MIN)
    pairs["Matched"] = event_ok & non_ok
    pairs["Eligibility"] = np.select([~event_ok & ~non_ok, ~event_ok, ~non_ok],
                                      ["Both below minimum", "Event below minimum", "NonEvent below minimum"],
                                      default="Matched")
    matched = pairs.loc[pairs.Matched].copy()
    matched["Delta_C"] = matched.Event_Mean-matched.NonEvent_Mean
    matched["Ratio_C"] = matched.Event_Mean.div(matched.NonEvent_Mean.where(
        matched.NonEvent_Mean.ne(0) & matched.Variable.ne("pH")))
    matched["Relative_Change"] = 100*(matched.Ratio_C-1)
    matched["Contributing_Sites"] = 1
    station_pair = (matched.groupby(STATION_KEY)
                    .agg(Event_Mean=("Event_Mean", "mean"), NonEvent_Mean=("NonEvent_Mean", "mean"),
                         Delta_C=("Delta_C", "mean"), Ratio_C=("Ratio_C", "mean"),
                         Relative_Change=("Relative_Change", "mean"),
                         Matched_Strata_N=("Delta_C","size"), Ratio_Strata_N=("Relative_Change","count"),
                         Event_Valid_N=("Event_Valid_N","sum"), NonEvent_Valid_N=("NonEvent_Valid_N","sum"),
                         Event_Theoretical_Days=("Event_Theoretical_Days","sum"),
                         NonEvent_Theoretical_Days=("NonEvent_Theoretical_Days","sum")).reset_index())
    matched_station = []
    for period in ("Event","NonEvent"):
        x = station_pair[STATION_KEY + ["Matched_Strata_N", "Ratio_Strata_N"]].copy()
        x["Mean"] = station_pair[f"{period}_Mean"]
        x["Valid_N"] = station_pair[f"{period}_Valid_N"]
        x["Theoretical_Days"] = station_pair[f"{period}_Theoretical_Days"]
        x["Relative_Change"] = station_pair.Relative_Change if period=="Event" else np.nan
        matched_station.append(coverage(x.assign(Period=period, Comparison="Matched", Contributing_Sites=1)))
    support = ["Matched_Strata_N", "Ratio_Strata_N", "Event_Valid_N", "NonEvent_Valid_N",
               "Event_Theoretical_Days", "NonEvent_Theoretical_Days"]
    reach_match = pd.concat([reach_descriptive(station_pair, ["Reach","Variable"], metric, support)
                             .assign(Metric=metric) for metric in (
                                 "Event_Mean","NonEvent_Mean","Delta_C","Ratio_C","Relative_Change")],
                            ignore_index=True)
    for period in ("Event","NonEvent"):
        reach_match[f"{period}_Missing_Rate"] = 1-reach_match[f"{period}_Valid_N"].div(
            reach_match[f"{period}_Theoretical_Days"].replace(0,np.nan))
    reach_match["Total_Sites"] = reach_match.Reach.map(REACHES)
    output = [("Station_Event_NonEvent_WQ", pd.concat([direct,*matched_station],ignore_index=True)),
              ("Matched_Seasonal_WQ",matched), ("Reach_Event_NonEvent_WQ",reach_direct),
              ("Reach_Matched_WQ_Changes",reach_match), ("Matching_Eligibility",pairs)]
    for name, data in output:
        write(data, 3, f"A03_{name}.csv")
    LOG.info("Matched event/non-event strata by variable: %s", matched.Variable.value_counts().to_dict())
    return reach_match


def a04(long, events):
    daily_events = long.loc[long.Period.eq("Event")]
    level = coverage(summary(daily_events, ["Reach","Site_ID","Event_ID","Event_Type","Variable"]))
    level = level.merge(events[["Site_ID","Event_ID","Start_Date","End_Date","Total_Days"]],
                        on=["Site_ID","Event_ID"],validate="many_to_one")
    require(level.Theoretical_Days.eq(level.Total_Days).all(), "Event length does not reconcile")
    level = level.drop(columns="Total_Days")
    level["Contributing_Sites"] = level.Valid_N.gt(0).astype(int)
    write(level,4,"A04_Event_Level_WQ_Statistics.csv")

    def compare(keys):
        valid = level.loc[level.Valid_N.gt(0)]
        counts = valid.groupby(keys).agg(Contributing_Sites=("Site_ID","nunique"),
                      Contributing_Events=("Event_ID","size"), Mean_Event_Coverage=("Coverage","mean"))
        totals = level.groupby(keys).agg(Total_Events=("Event_ID","size"), Valid_Day_N=("Valid_N","sum"),
                                         Theoretical_Days=("Theoretical_Days","sum")).join(counts)
        totals["Valid_N"] = totals.Valid_Day_N
        totals = coverage(totals.reset_index()).drop(columns="Valid_N")
        site_average = valid.groupby(keys+["Site_ID"],as_index=False).Mean.mean()
        summaries = [summary(level,keys,"Mean",False).assign(Statistic_Unit="Event equal"),
                     summary(site_average,keys,"Mean",False).assign(Statistic_Unit="Site equal")]
        out = pd.concat(summaries,ignore_index=True).merge(totals,on=keys,how="right")
        out = out.rename(columns={"Valid_N":"Estimate_N"})
        return out
    combined = compare(["Event_Type","Variable"])
    by_reach = compare(["Reach","Event_Type","Variable"])
    write(combined,4,"A04_EventType_WQ_Statistics.csv")
    write(by_reach,4,"A04_Reach_EventType_WQ_Statistics.csv")
    return combined


def a05(daily, long):
    observed = daily.loc[daily.DO_Source.eq("QC_observed"), ["Site_ID","Date","DO","DO_Source"]].copy()
    observed["DO_Group"] = pd.cut(observed.DO, [-np.inf,2,5,np.inf], right=False,labels=GROUPS).astype("object")
    chem = long.loc[long.Variable.ne("DO")].merge(observed,on=["Site_ID","Date"],
                                                   how="inner",validate="many_to_one")
    valid = chem.loc[chem.Value.notna()].copy()
    write(valid,5,"A05_DO_Group_Daily_Pairs.csv")
    contexts = ("All","Event","NonEvent")
    outputs = {"":[], "Reach_":[], "Station_":[], "EventType_":[]}
    for context in contexts:
        scope = chem if context=="All" else chem.loc[chem.Period.eq(context)]
        levels = [("",[]), ("Reach_",["Reach"]), ("Station_",["Reach","Site_ID"])]
        if context=="Event":
            levels.append(("EventType_",["Event_Type"]))
        for prefix, dimensions in levels:
            keys = dimensions + ["DO_Group","Variable"]
            support = (scope.loc[scope.Value.notna()].groupby(keys)
                       .agg(Contributing_Sites=("Site_ID","nunique"),
                            Contributing_Events=("Event_ID","nunique")).reset_index())
            day = coverage(summary(scope,keys)).rename(columns={"Valid_N":"Valid_Pairs"})
            day = day.merge(support,on=keys,how="left")
            day[["Contributing_Sites","Contributing_Events"]] = day[
                ["Contributing_Sites","Contributing_Events"]].fillna(0)
            if context!="Event":
                day["Contributing_Events"] = np.nan
            day["Context"] = context
            day["Statistic_Unit"] = "Daily equal"
            if prefix=="Station_":
                outputs[prefix].append(day)
                continue
            station_mean = scope.groupby(keys+["Site_ID"], as_index=False).Value.mean()
            equal = summary(station_mean,keys,counts=False).rename(columns={"Valid_N":"Estimate_N"})
            counts = [*keys,"Valid_Pairs","Theoretical_Days","Missing_Days","Coverage","Missing_Rate",
                      "Contributing_Sites","Contributing_Events","Context"]
            equal = equal.merge(day[counts],on=keys,validate="one_to_one").assign(Statistic_Unit="Site equal")
            outputs[prefix].extend([day.assign(Estimate_N=day.Valid_Pairs),equal])
    stats = {k:pd.concat(v,ignore_index=True) for k,v in outputs.items()}
    for prefix, data in stats.items():
        write(data,5,f"A05_{prefix}DO_Group_{'WQ_Statistics' if not prefix else 'WQ'}.csv")

    reference_keys = STATION_KEY+["Year","Month"]
    high = valid.loc[valid.DO_Group.eq("Higher_DO")].copy()
    high["Reference_Is_Event"] = high.Period.eq("Event")
    high["Reference_Date_Label"] = high.Date.dt.strftime("%Y-%m-%d")
    reference = (high.groupby(reference_keys)
                 .agg(Reference_Median=("Value","median"), Reference_Valid_N=("Value","size"),
                      Reference_Event_Days=("Reference_Is_Event","sum"),
                      Reference_Dates=("Reference_Date_Label","|".join),
                      Reference_Event_Dates=("Reference_Date_Label",
                          lambda dates: "|".join(dates[high.loc[dates.index,"Reference_Is_Event"]])))
                 .reset_index())
    reference["Reference_NonEvent_Days"] = reference.Reference_Valid_N-reference.Reference_Event_Days

    decisions=[]
    low = chem.loc[chem.DO.lt(5)]
    for threshold in (2,5):
        target = low.loc[low.DO.lt(threshold)].merge(reference,on=reference_keys,how="left",validate="many_to_one")
        counts=["Reference_Valid_N","Reference_Event_Days","Reference_NonEvent_Days"]
        target[counts] = target[counts].fillna(0)
        target["Threshold"] = f"DO < {threshold}"
        target["Matched"] = target.Value.notna() & target.Reference_Valid_N.ge(REFERENCE_MIN)
        target["Eligibility"] = np.select([target.Value.isna(),target.Reference_Valid_N.lt(REFERENCE_MIN)],
                                           ["Target variable missing","Below reference minimum"],default="Matched")
        target["Delta_C"] = (target.Value-target.Reference_Median).where(target.Matched)
        decisions.extend([target.assign(Context="All"),
                          target.loc[target.Period.eq("Event")].assign(Context="Event")])
    decisions = pd.concat(decisions,ignore_index=True)
    matched = decisions.loc[decisions.Matched].copy()
    write(matched,5,"A05_Matched_Low_DO_Comparison.csv")
    keys=["Context","Threshold","Reach","Variable"]
    support = (decisions.assign(Target_Variable_Observed=decisions.Value.notna())
               .groupby(keys+["Site_ID"]).agg(Target_Days=("Date","size"),
                   Valid_Target_Days=("Target_Variable_Observed","sum"), Matched_N=("Matched","sum"),
                   Mean_Delta_C=("Delta_C","mean")).reset_index())
    support["Unmatched_N"] = support.Target_Days-support.Matched_N
    support["Match_Rate"] = support.Matched_N/support.Target_Days
    support["Missing_Rate"] = 1-support.Valid_Target_Days/support.Target_Days
    support["Contributing_Sites"] = support.Matched_N.gt(0).astype(int)
    totals = support.groupby(keys,as_index=False)[
        ["Target_Days","Valid_Target_Days","Matched_N","Unmatched_N","Contributing_Sites"]].sum()
    reach_match = summary(support,keys,"Mean_Delta_C",False).drop(columns="Valid_N").merge(
        totals,on=keys,validate="one_to_one").assign(Statistic_Unit="Site equal")
    reach_match["Match_Rate"] = reach_match.Matched_N/reach_match.Target_Days
    reach_match["Missing_Rate"] = 1-reach_match.Valid_Target_Days/reach_match.Target_Days
    reach_match["Total_Sites"] = reach_match.Reach.map(REACHES)
    write(reach_match,5,"A05_Reach_Matched_Low_DO_Summary.csv")
    write(support,5,"A05_Matching_Eligibility.csv")
    LOG.info("Observed DO days %s; low-DO matching eligibility %s", observed.DO_Group.value_counts().to_dict(),
             decisions.groupby(["Context","Threshold","Eligibility"]).size().to_dict())

    historical = read(S27,["Context","Category","Variable","N","Median"])
    historical = historical.loc[historical.Context.eq("Event_observed") & historical.Variable.isin(CHEM+["WT"])
                                & historical.Category.isin(["DO_lt2","DO_ge5"])]
    historical = historical.assign(DO_Group=historical.Category.map(
        {"DO_lt2":"Very_Low_DO","DO_ge5":"Higher_DO"})).rename(
        columns={"N":"S27_Valid_Pairs","Median":"S27_Median"})
    focus = historical[["DO_Group","Variable","S27_Valid_Pairs","S27_Median"]].merge(
        stats["Station_"].loc[lambda x:x.Site_ID.eq("Jiagaiqiao") & x.Context.eq("Event")],
        on=["DO_Group","Variable"],how="left",validate="one_to_one")
    require(len(focus)==10,"S27 source incomplete")
    focus["Median_Difference"] = focus.Median-focus.S27_Median
    focus["Agrees"] = focus.Valid_Pairs.eq(focus.S27_Valid_Pairs) & np.isclose(
        focus.Median,focus.S27_Median,rtol=0,atol=1e-10)
    LOG.info("S27 observed medians + pair counts reconcile for %d of %d cells",focus.Agrees.sum(),len(focus))
    write(focus,5,"A05_Jiagaiqiao_S27_Verification.csv")
    return stats, reach_match, focus


FIGURES = {
    "A1":"Fig_A1_Longterm_WQ_Background", "A2":"Fig_A2_Event_NonEvent_WQ",
    "A3":"Fig_A3_Matched_Event_WQ_Changes", "A4":"Fig_A4_Low_DO_WQ_Associations",
    "A4b":"Fig_A4b_Downstream_Low_DO_WQ", "A5":"Fig_A5_EventType_WQ",
    "A6":"Fig_A6_WQ_Observation_Coverage", "A6b":"Fig_A6b_Monthly_Coverage",
}


def paper_outputs(daily, station_qc, reach_qc, reach_bg, station_bg, reach_event,
                  event_type, do_stats, low_match, check):
    """Write tables and manuscript guidance from computed values."""
    label = {"NH3_N":"NH₃-N", "CODMn":"CODMn", **{k:k for k in ("TN","TP","WT","DO","pH","EC","Turb")}}
    names = {"UR":"上游", "MR":"中游", "LR":"下游"}
    sections = []
    def add(lines):
        sections.append(lines)

    def fmt(value):
        return "—" if pd.isna(value) else f"{value:.4f}"

    def table(headers, rows):
        return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |",
                          *["| "+" | ".join(map(str,row))+" |" for row in rows]])

    def ordered(data, order_field, variables=CHEM):
        rows = data.loc[data.Variable.isin(variables)].copy()
        rows["__variable"] = rows.Variable.map({v:i for i,v in enumerate(variables)})
        order = {v:i for i,v in enumerate(REACHES if order_field=="Reach" else EVENT_MAP.values())}
        rows["__order"] = rows[order_field].map(order)
        return rows.sort_values(["__variable","__order"])

    def links(section, names_):
        return "、".join(f"[{n}](../{DIRS[section]}/{n})" for n in names_)

    add("# 黄河流域长期水质背景与低 DO 的同期水质特征")
    add("## 数据来源与范围")
    inputs = [(CLEANED,"66 站插补前质控日值"), (STATIONS,"正式站点与河段映射"),
              (REGISTRY,"正式事件登记"), (S0,"S0 基准登记交叉核对"),
              (PROVENANCE,"DO 原始观测来源及事件日交叉核对"), (S27,"表 S27 对应既有专项统计")]
    add("\n".join(f"- {role}。[{path.name}](<{path.as_posix()}>)" for path,role in inputs))
    observed = daily.DO.count()
    date_counts = daily.Period.value_counts()
    add(f"2021–2025 年共 {len(DATES):,} 个日历日，2024 年有 366 天。66 站分布为 UR 28、MR 33、LR 5，"
        f"共 {len(daily):,} 个站点–日期键，无重复。S0 有 2,548 条站点–事件记录，覆盖 "
        f"{date_counts['Event']:,} 个站点–日期，其余 {date_counts['NonEvent']:,} 日为非事件。"
        f"质控日值中的 DO 与逐日来源表逐站逐日一致，共 {observed:,} 个原始 DO 日，"
        f"{len(daily)-observed:,} 个插补等对应缺失日未纳入 DO 分组。所有变量使用插补前日值，按变量独立统计，保留真实零值。"
        "同日分析要求原始 DO 与对应变量均有效。全年季节统计按月份划分，冬季为 12、1、2 月。"
        "同年同季节匹配中的年份指各日期的公历年，冬季 12 月归其所在公历年。")

    add("## 水质数据完整性")
    add(table(["变量","上游覆盖率","中游覆盖率","下游覆盖率","66 站最小–最大覆盖率","最长连续缺失"],
        [(label[v], *[f"{reach_qc.loc[(reach_qc.Window=='All') & (reach_qc.Reach==r) & (reach_qc.Variable==v),'Coverage'].iloc[0]:.2%}" for r in REACHES],
          f"{g.Coverage.min():.2%}–{g.Coverage.max():.2%}",f"{int(g.Longest_Missing_Run_Days.max())} d")
         for v in UNITS for g in [station_qc.loc[station_qc.Variable.eq(v)]]]))
    add("覆盖率以理论日数为分母。河段覆盖率合并河段内各站的理论和实际日数。表中极值按站点取值。"
        "A01 提供各站逐年、逐季、逐年月明细、合格年月及连续缺口。")
    add("## 五年长期水质背景")
    main = ordered(reach_bg.loc[reach_bg.Method.eq("M3")], "Reach", PRIORITY)
    add(table(["变量","单位","河段","M3 站点等权均值","站点中位数","站点间 SD","站点数／总数","有效原始日数"],
          [(label[r.Variable],UNITS[r.Variable],names[r.Reach],fmt(r.Mean),fmt(r.Median),fmt(r.SD),
            f"{int(r.Contributing_Sites)}/{int(r.Total_Sites)}",int(r.Valid_N)) for r in main.itertuples()]))
    add("M3 先用覆盖率至少 50% 的年月均值估计各日历月份，要求同一月份至少有 3 个合格年份，"
        "且 12 个月全部合格。随后月份等权、河段内站点等权。M1 使用全部有效日，M2 对覆盖率至少 50% 的合格年份等权，要求至少 3 年。"
        "三种估计分别输出，覆盖率条件为本研究的描述性设定。季节背景单列，每站先在对应季节对有效日求均值。")
    for variable in CHEM:
        z = main.loc[main.Variable.eq(variable)].set_index("Reach")
        lower = z.loc["LR"]
        if z.Mean.notna().all():
            higher = [names[r] for r in ("UR","MR") if lower.Mean>z.loc[r,"Mean"]]
            comparison = f"高于{'和'.join(higher)}" if higher else "未高于另两个河段"
            add(f"- {label[variable]} 的下游 M3 为 {fmt(lower.Mean)} {UNITS[variable]}，{comparison}"
                f"。下游实际参与 {int(lower.Contributing_Sites)}/5 站。")
        else:
            add(f"- {label[variable]} 至少一个河段缺少合格 M3，保留其余可用估计和独立的 M1、M2。")
    if main.Contributing_Sites.lt(3).any():
        add("部分变量和河段的 M3 少于 3 个站点，相关均值代表有限的监测站点。")
    unavailable = station_bg.loc[station_bg.Reach.eq("LR") & station_bg.Variable.isin(PRIORITY)
                                 & station_bg.M3.isna()]
    if len(unavailable):
        excluded = "；".join(
            f"{site} 的" + "、".join(f"{label[r.Variable]}（{int(r.M3_Eligible_Months)}/12 个月）"
                                    for r in g.itertuples())
            for site,g in unavailable.groupby("Site_ID"))
        add(f"下游 M3 无法计算的站点–变量为 {excluded}。这些站点的可用 M1、M2 单独报告。"
            "长期 M3 与事件期分组所覆盖的站点样本可能不同，比较时需同时说明纳入范围。")

    add("## 事件期与非事件期")
    matched = ordered(reach_event.loc[reach_event.Metric.eq("Delta_C")],"Reach")
    ratios = reach_event.loc[reach_event.Metric.eq("Relative_Change")].set_index(["Reach","Variable"])
    add(table(["变量","河段","事件期减非事件期","平均相对变化（%）","匹配站点","配对层数",
               "事件／非事件有效日"],
        [(label[r.Variable],names[r.Reach],fmt(r.Mean),fmt(ratios.loc[(r.Reach,r.Variable),"Mean"]),
          int(r.Contributing_Sites),int(r.Matched_Strata_N),f"{int(r.Event_Valid_N)}/{int(r.NonEvent_Valid_N)}")
         for r in matched.itertuples()]))
    add("每个同站、同公历年、同季节层内，两组至少各有 10 个有效日。绝对差值保留非事件均值为零的合格层。"
        "pH 只计算差值，分母为零的层也不计算比值或相对变化。先对站点内各匹配层等权，再对河段内站点等权。"
        "表中相对变化是各配对层相对变化的分级平均。无匹配限制的直接比较另见 A03。")
    add("下游合格匹配层的四项浓度差值为 " + "、".join(
        f"{label[r.Variable]} {r.Mean:+.4f}"
        for r in matched.loc[matched.Reach.eq("LR")].itertuples()) + " mg L⁻¹。")

    add("## 五类事件的同期水质特征")
    ev = ordered(event_type.loc[event_type.Statistic_Unit.eq("Event equal")],"Event_Type")
    add(table(["变量","事件类型","事件均值的等权平均","事件均值的中位数","有效事件／总事件","站点数","事件平均观测覆盖率"],
        [(label[r.Variable],r.Event_Type,fmt(r.Mean),fmt(r.Median),f"{int(r.Contributing_Events)}/{int(r.Total_Events)}",
          int(r.Contributing_Sites),f"{r.Mean_Event_Coverage:.2%}") for r in ev.itertuples()]))
    add("事件均值至少由一个事件窗口内的原始观测日计算。覆盖率随事件变化。A04 另提供站点等权估计和河段×类型交叉结果。"
        "五类样本涉及的季节、站点及事件长度不同，排序作为描述性比较。")
    add("事件等权均值最高的类型分别为 " + "、".join(
        f"{label[v]} 的 {r.Event_Type}（{fmt(r.Mean)} mg L⁻¹，{int(r.Contributing_Events)} 个有效事件）"
        for v in CHEM for r in [ev.loc[ev.Variable.eq(v)].nlargest(1,"Mean").iloc[0]]) + "。")

    add("## 不同 DO 浓度组的同期水质")
    lower_groups = do_stats["Reach_"].loc[lambda x:x.Reach.eq("LR") & x.Context.eq("Event") &
                  x.Statistic_Unit.eq("Site equal") & x.Variable.isin(CHEM+["WT","pH"])].copy()
    lower_groups["Order"] = lower_groups.DO_Group.map({g:i for i,g in enumerate(GROUPS)})
    add(table(["变量","DO 组","各站配对日均值的等权平均","配对日数","站点数","事件数"],
        [(label[r.Variable],r.DO_Group,fmt(r.Mean),int(r.Valid_Pairs),int(r.Contributing_Sites),int(r.Contributing_Events))
         for r in lower_groups.sort_values(["Variable","Order"]).itertuples()]))
    add("DO 分组为 <2、2≤DO<5、≥5 mg L⁻¹。表展示事件期下游站点均值的等权统计。"
        "图 A4 与 A4b 采用各站组内中位数，标注有效站点和配对日数。样本支持会随变量、分组和河段变化。"
        "全期、事件期、非事件期以及事件类型的逐日等权和站点等权摘要见 A05。")
    contrast = lower_groups.pivot(index="Variable",columns="DO_Group",values="Mean")
    if set(GROUPS) <= set(contrast):
        add("与 DO ≥5 mg L⁻¹ 组相比，下游事件期 DO <2 mg L⁻¹ 组四项同期浓度的描述性差值为 " +
            "、".join(f"{label[v]} {contrast.loc[v,'Very_Low_DO']-contrast.loc[v,'Higher_DO']:+.4f}"
                     for v in CHEM) + " mg L⁻¹，组间站点构成有差异。")

    add("## 同站同年同月的低 DO 匹配")
    em = ordered(low_match.loc[low_match.Context.eq("Event")],"Reach")
    add(table(["阈值","河段","变量","差值的站点等权均值","成功／目标日","成功比例","站点数"],
        [(r.Threshold,names[r.Reach],label[r.Variable],fmt(r.Mean),f"{int(r.Matched_N)}/{int(r.Target_Days)}",
          f"{r.Match_Rate:.2%}",int(r.Contributing_Sites))
         for r in em.itertuples()]))
    add("两个低 DO 阈值分别计算。目标日与参照需同站、同年、同月，参照日具有原始 DO ≥5 mg L⁻¹ "
        "且对应变量有效。每个变量至少 3 个有效参照日。各目标日减去组内参照中位数，再将站点内差值等权平均。"
        "事件期目标的参照可以跨事件状态，明细记录参照日及其事件归属。表内目标日含对应变量缺失而无法配对的日期。"
        "全期匹配结果另见同一 CSV 的 All 行。")
    for threshold in ("DO < 2","DO < 5"):
        lower = em.loc[em.Reach.eq("LR") & em.Threshold.eq(threshold)]
        if len(lower):
            add(f"在下游事件期 {threshold} mg L⁻¹ 目标中，四项差值分别为 " + "、".join(
                f"{label[r.Variable]} {fmt(r.Mean)} mg L⁻¹（{int(r.Matched_N)}/{int(r.Target_Days)} 日，"
                f"{int(r.Contributing_Sites)} 站）" for r in lower.itertuples()) + "。"
                "成功率限制了结论向全部低 DO 日期的外推。")

    add("## Jiagaiqiao 站 Table S27 复核")
    add(table(["变量","DO 组","本次中位数","S27 原始统计中位数","本次／S27 配对日","一致"],
      [(label[r.Variable],r.DO_Group,fmt(r.Median),fmt(r.S27_Median),f"{int(r.Valid_Pairs)}/{int(r.S27_Valid_Pairs)}",
        "是" if r.Agrees else "否") for r in check.itertuples()]))
    add(f"10 个变量组单元中有 {int(check.Agrees.sum())} 个中位数和样本数与 S27 的底层结果一致。"
        "表 S27 印刷值的 WT 24.422 对应底层值 24.4215。站点专项统计适用于该站观测条件。"
        if check.Agrees.all() else
        "存在 S27 不一致单元。详见核验 CSV 的数值、配对数和差值。")

    add("## 可用于论文的表述")
    chemistry = main.loc[main.Variable.isin(CHEM)].set_index(["Variable","Reach"])
    higher_both = [label[v] for v in CHEM if chemistry.loc[(v,"LR"),"Mean"] >
                   max(chemistry.loc[(v,"UR"),"Mean"],chemistry.loc[(v,"MR"),"Mean"])]
    supports = "、".join(higher_both) if higher_both else "无"
    add(f"- **Results 3.6。** 可补充 M3 河段对比和实际站点数。在本期合格监测样本中，"
        f"下游均值同时高于上游和中游的指标为 {supports}。引用同月低 DO 比较时同时报告估计值和配对分母。")
    add("- **Discussion 4.3。** 可讨论部分低 DO 日期与较高营养盐或 CODMn 同时出现的水质背景。引用时同时给出匹配后的方向、"
        "配对成功比例和实际参与站点数。季节及站点构成、流量和其他过程可能共同影响这些观察。"
        "站点浓度差异及同期组间比较的证据范围是描述性水质特征。")
    add("- **Conclusions 第二段。** 可概括河段间水质背景存在差别、部分事件期低 DO 记录伴随水质浓度差异，"
        "并保留下游监测站点有限及匹配样本稀少的限定。本文统计以浓度和同日配对差值为单位。")

    key=[]
    def retain(frame, metric, figure, section, filename, count, method, extra=None):
        for r in frame.to_dict("records"):
            key.append({"Metric":metric,"Variable":r["Variable"],"Unit":"%" if metric=="Relative_Change" else UNITS[r["Variable"]],
                        "Reach":r.get("Reach","All"),"Event_Type":r.get("Event_Type"),"DO_Group":r.get("DO_Group"),
                        "Context":r.get("Context"),"Threshold":r.get("Threshold"),"Estimate":r["Mean"],
                        "Sample_N":r[count],"Contributing_Sites":r.get("Contributing_Sites"),
                        "Target_Days":r.get("Target_Days"),"Valid_N":r.get("Valid_N",r.get("Valid_Pairs")),
                        "Match_Rate":r.get("Match_Rate"),"Method":method, "Data_Source":(
                            "Pre-imputation daily QC; formal S0; observed DO provenance checked"),
                        "Source_CSV":f"../{DIRS[section]}/{filename}",
                        "Figure":f"../{DIRS[6]}/{FIGURES[figure]}.pdf" if figure else None, **(extra or {})})
    retain(main,"M3","A1",2,"A02_Reach_5Year_WQ_Statistics.csv","Contributing_Sites",
           "Months equal within site; sites equal within reach")
    retain(matched,"Delta_C","A2",3,"A03_Reach_Matched_WQ_Changes.csv","Matched_Strata_N",
           "Matched site/year/season; strata then sites equal")
    retain(ratios.reset_index(),"Relative_Change","A3",3,"A03_Reach_Matched_WQ_Changes.csv","Ratio_Strata_N",
           "Mean of eligible relative changes; strata then sites equal")
    retain(ev,"Event_WQ","A5",4,"A04_EventType_WQ_Statistics.csv","Contributing_Events",
           "Daily means within event; event means equally weighted")
    retain(lower_groups,"DO_Group_WQ","A4b",5,"A05_Reach_DO_Group_WQ.csv","Valid_Pairs",
           "Same-day QC pairs; daily means within site; sites equal",
           {"Figure_Note":"Related display uses station medians"})
    retain(em,"Matched_Low_DO_Delta",None,5,"A05_Reach_Matched_Low_DO_Summary.csv","Matched_N",
           "Same-site/year/month reference medians; target days then sites equal")
    write(pd.DataFrame(key),7,"Manuscript_Key_Results.csv")

    add("## 结果文件与图件")
    for i in (1,2,3,4,5):
        files=sorted(p.name for p in (OUT/DIRS[i]).glob("A0*.csv"))
        add(f"- {DIRS[i]}。"+links(i, files))
    add("\n".join(f"- {id_}。 [{name}.pdf](../{DIRS[6]}/{name}.pdf)（同时提供 SVG、600 dpi TIFF 和 300 dpi PNG）"
                  for id_,name in FIGURES.items()))
    add("[论文关键统计汇总](Manuscript_Key_Results.csv)按估计量标明数据和图件出处。图 A3 显示站点相对变化、"
        "零参考线与河段等权均值。图 A4 的站点中位数图和站点等权均值摘要各有明确口径。")
    report=OUT / DIRS[7] / "WQ_Background_Analysis_Report.md"
    report.write_text("\n\n".join(sections)+"\n",encoding="utf-8")
    LOG.info("Wrote %s",report)


def main():
    for name in DIRS.values():
        (OUT/name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                           logging.FileHandler(OUT/DIRS[7]/"Analysis_Run.log",mode="w",encoding="utf-8")])
    LOG.info("Parameters: long-term coverage >= %.0f%%, minimum years %d, matched strata min %d per period,"
             " low-DO same-month reference min %d",LONG_COVERAGE*100,MIN_YEARS,EVENT_MIN,REFERENCE_MIN)
    LOG.info("Field mapping and units: %s / %s",FIELDS,UNITS)
    daily,long,events=read_inputs()
    station_qc,reach_qc=a01(long)
    reach_bg,station_bg=a02(long)
    reach_event=a03(long)
    event_type=a04(long,events)
    do_stats,low_match,check=a05(daily,long)
    paper_outputs(daily,station_qc,reach_qc,reach_bg,station_bg,reach_event,event_type,do_stats,low_match,check)
    LOG.info("A01–A05 complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Analysis failed")
        raise

