# 01_Event_Detection_and_Splitting.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

warnings.filterwarnings('ignore')

# ===================== 1. Paths and parameters =====================
ROOT = Path(__file__).resolve().parents[3]
BASE_FOLDER = ROOT / "Data/Meteorology_data/1991-2020/YR"
ORIGINAL_FOLDER = ROOT / "Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR"
RESULT_FOLDER = ROOT / "Data/Event_F"

TARGET_HW_P, TARGET_FLOOD_P = 90, 95
HW_CONTINUE_DAYS, HW_MERGE_INTERVAL, COMP_INTERVAL = 3, 2, 5
QC_TARGET_COLS = ['DO', 'WT', 'pH', 'EC', 'NH₃-N', 'TP', 'TN', 'CODₘₙ', 'WL', 'Q', 'Ss', 'Turb']
EXPECTED_EVENT_TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']

# Set the source column names for coordinates and static attributes.
LON_COL = 'WQS_Lon'
LAT_COL = 'WQS_Lat'
REGION_COL = 'Region'
RIVER_COL = 'River'

# Append only the four static-attribute fields below.
REGION_DUMMY_COLS = ['Region_Upper', 'Region_Middle', 'Region_Lower']
RIVER_TYPE_COL = 'River_Type'  # Main stem = 1, tributary = 0.
REGISTRY_COLUMNS = [
    'Event_ID', 'WQS_name', LON_COL, LAT_COL, REGION_COL, *REGION_DUMMY_COLS,
    RIVER_COL, RIVER_TYPE_COL, 'Event_Type', 'Start_Date', 'End_Date', 'Total_Days',
]


# ===================== 2. Utilities =====================
def read_all_csv(folder_path, numeric_cols):
    site_data = {}
    for file in [f for f in os.listdir(folder_path) if f.endswith(".csv")]:
        try:
            df = pd.read_csv(os.path.join(folder_path, file), encoding='utf-8')
        except:
            df = pd.read_csv(os.path.join(folder_path, file), encoding='gbk')

        df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')

        # Standardize the date and region column names. Retain the other source names.
        rename_dict = {}
        for col in df.columns:
            if col.lower() == 'date': rename_dict[col] = 'Date'
            if col.lower() == 'r': rename_dict[col] = 'Region'
        df.rename(columns=rename_dict, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

        # Convert only the quality-control target columns to numeric values.
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'WQS_name' in df.columns:
            site_name = str(df['WQS_name'].iloc[0]).strip()
            site_data[site_name] = df
    return site_data


def add_static_attributes(df):
    """
    Add four static model features using the original Region and River columns.

    Region_Upper, Region_Middle and Region_Lower indicate UR, MR and LR, respectively. River_Type is 1 when River is Yellow_river, 0 for other recorded rivers, and NA when River is missing.
    """
    df = df.copy()

    # ---------- One-hot encode upstream, midstream and downstream ----------
    if REGION_COL in df.columns:
        region_code = df[REGION_COL].astype('string').str.strip().str.upper()
    else:
        region_code = pd.Series(pd.NA, index=df.index, dtype='string')

    df['Region_Upper'] = (region_code == 'UR').fillna(False).astype(int)
    df['Region_Middle'] = (region_code == 'MR').fillna(False).astype(int)
    df['Region_Lower'] = (region_code == 'LR').fillna(False).astype(int)

    # ---------- Encode main stem and tributary ----------
    if RIVER_COL in df.columns:
        river_name = df[RIVER_COL].astype('string').str.strip()
    else:
        river_name = pd.Series(pd.NA, index=df.index, dtype='string')

    valid_river = river_name.notna() & river_name.ne('')
    is_mainstem = river_name.str.lower().eq('yellow_river')

    river_type = pd.Series(pd.NA, index=df.index, dtype='Int64')
    river_type.loc[valid_river] = is_mainstem.loc[valid_river].astype(int)
    df[RIVER_TYPE_COL] = river_type

    return df


def get_base_segments(df, event_type, t_thresh, q_thresh):
    df['flag'] = (df['Tmax'] >= t_thresh).astype(int) if event_type == 'hw' else (df['Q'] >= q_thresh).astype(int)
    df['group'] = (df['flag'] != df['flag'].shift(1)).cumsum()
    segments = []
    for _, group_df in df[df['flag'] == 1].groupby('group'):
        start, end = group_df['Date'].min(), group_df['Date'].max()
        days = (end - start).days + 1
        if event_type == 'hw' and days < HW_CONTINUE_DAYS: continue
        if event_type == 'hw' and segments and 0 <= (start - segments[-1]['end']).days - 1 <= HW_MERGE_INTERVAL:
            segments[-1]['end'] = end
            segments[-1]['days'] = (end - segments[-1]['start']).days + 1
            continue
        segments.append({'start': start, 'end': end, 'days': (end - start).days + 1, 'type': event_type})
    return segments


def pair_remaining_sequences(h_df, f_df, used_hw, used_fl,
                             sequence_priority=('F5H', 'H5F'), audit=None):
    """Keep both original tie-breakers; vary only which direction is assigned first.

    Candidates are enumerated after DC and before either sequential assignment.
    Filtering already used primitives in the second pass is equivalent to the
    original second-pass regeneration. Default behavior remains F5H then H5F.
    """
    if tuple(sequence_priority) not in [('F5H', 'H5F'), ('H5F', 'F5H')]:
        raise ValueError('sequence_priority must contain F5H and H5F exactly once')
    un_hw = h_df[~h_df.index.isin(used_hw)]
    un_fl = f_df[~f_df.index.isin(used_fl)]
    candidates = {'F5H': [], 'H5F': []}
    for hi, h in un_hw.iterrows():
        for fi, f in un_fl.iterrows():
            if h['start'] > f['end']:
                direction = 'F5H'
                gap = (h['start'] - f['end']).days - 1
            elif f['start'] > h['end']:
                direction = 'H5F'
                gap = (f['start'] - h['end']).days - 1
            else:
                continue
            if 0 <= gap <= COMP_INTERVAL:
                candidates[direction].append(dict(gap_days=gap, hw_idx=hi, flood_idx=fi,
                    hw_start=h['start'], hw_end=h['end'], flood_start=f['start'], flood_end=f['end']))
    candidates['F5H'].sort(key=lambda x: (x['gap_days'], x['hw_start'], -x['flood_end'].value,
                                          x['flood_idx'], x['hw_idx']))
    candidates['H5F'].sort(key=lambda x: (x['gap_days'], x['flood_start'], -x['hw_end'].value,
                                          x['hw_idx'], x['flood_idx']))
    if audit is not None:
        audit['eligible_candidates'] = [{**c, 'direction': direction} for direction in ['F5H', 'H5F']
                                         for c in candidates[direction]]
        audit['dc_used_hw'] = sorted(used_hw)
        audit['dc_used_flood'] = sorted(used_fl)
        audit['selected_candidates'] = []
    result = []
    for direction in sequence_priority:
        for candidate in candidates[direction]:
            hi, fi = candidate['hw_idx'], candidate['flood_idx']
            if hi in used_hw or fi in used_fl:
                continue
            start = candidate['flood_start'] if direction == 'F5H' else candidate['hw_start']
            end = candidate['hw_end'] if direction == 'F5H' else candidate['flood_end']
            result.append(dict(comp_subtype='indirect_f2h' if direction == 'F5H' else 'indirect_h2f',
                               start=start, end=end, days=(end-start).days+1))
            used_hw.add(hi)
            used_fl.add(fi)
            if audit is not None:
                audit['selected_candidates'].append({**candidate, 'direction': direction})
    return result


def classify_complex_events(hw_segs, flood_segs, sequence_priority=('F5H', 'H5F'), audit=None):
    comp_segs, used_hw, used_fl = [], set(), set()
    h_df, f_df = pd.DataFrame(hw_segs), pd.DataFrame(flood_segs)

    if not h_df.empty and not f_df.empty:
        # Prioritize DC events by merging heatwaves and floods connected by at least one day of overlap.
        adjacency = {}
        for i, h in h_df.iterrows():
            h_node = ('h', i)
            adjacency.setdefault(h_node, set())
            for j, f in f_df.iterrows():
                if not (h['end'] < f['start'] or h['start'] > f['end']):
                    f_node = ('f', j)
                    adjacency[h_node].add(f_node)
                    adjacency.setdefault(f_node, set()).add(h_node)

        visited = set()
        for node in adjacency:
            if node in visited or not adjacency[node]:
                continue

            stack, component = [node], set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adjacency[current] - visited)

            hw_ids = {idx for kind, idx in component if kind == 'h'}
            fl_ids = {idx for kind, idx in component if kind == 'f'}
            if hw_ids and fl_ids:
                starts = [h_df.loc[idx, 'start'] for idx in hw_ids]
                starts.extend(f_df.loc[idx, 'start'] for idx in fl_ids)
                ends = [h_df.loc[idx, 'end'] for idx in hw_ids]
                ends.extend(f_df.loc[idx, 'end'] for idx in fl_ids)
                c_start, c_end = min(starts), max(ends)
                comp_segs.append(
                    {'comp_subtype': 'direct', 'start': c_start, 'end': c_end,
                     'days': (c_end - c_start).days + 1})
                used_hw.update(hw_ids)
                used_fl.update(fl_ids)
        comp_segs.extend(pair_remaining_sequences(h_df, f_df, used_hw, used_fl, sequence_priority, audit))

    rem_hw = [row for i, row in h_df.iterrows() if i not in used_hw]
    rem_fl = [row for j, row in f_df.iterrows() if j not in used_fl]
    return comp_segs, rem_hw, rem_fl


def validate_event_exclusivity(site_name, events):
    """Validate the final events without modifying them. Check that one event cannot overwrite another event's Event_Label."""
    event_ids = [ev['Event_ID'] for ev in events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError(f"站点 {site_name} 存在重复 Event_ID: {event_ids}")

    ordered = sorted(events, key=lambda ev: (ev['Start_Date'], ev['End_Date']))
    occupied_dates = set()
    previous = None
    for current in ordered:
        if previous is not None and current['Start_Date'] <= previous['End_Date']:
            raise RuntimeError(
                f"站点 {site_name} 最终事件时间窗重叠: "
                f"{previous['Event_Type']} {previous['Start_Date']}~{previous['End_Date']}；"
                f"{current['Event_Type']} {current['Start_Date']}~{current['End_Date']}"
            )
        event_dates = set(pd.date_range(current['Start_Date'], current['End_Date']))
        if occupied_dates & event_dates:
            raise RuntimeError(
                f"站点 {site_name} 同一日期属于多个最终事件: "
                f"{current['Event_Type']} {current['Start_Date']}~{current['End_Date']}"
            )
        occupied_dates.update(event_dates)
        previous = current


def write_validation_registry(registry):
    """Build the registry for continuous-gap validation from the current events."""
    validation = registry.rename(columns={'WQS_name': 'Site_ID'})
    if validation.duplicated(['Site_ID', 'Event_ID']).any():
        raise RuntimeError("验证登记表存在重复的 Site_ID 和 Event_ID 组合")

    mapping = pd.read_csv(ROOT / 'Config/Study_Station_Mapping.csv')
    reach = mapping.set_index('WQS_name')['Region'].map(
        {'UR': 'Upstream', 'MR': 'Midstream', 'LR': 'Downstream'}
    )
    validation['Reach'] = validation['Site_ID'].map(reach)
    type_order = {event_type: order for order, event_type in enumerate(EXPECTED_EVENT_TYPES)}
    validation = validation.sort_values('Event_Type', key=lambda values: values.map(type_order), kind='stable')

    output = Path(RESULT_FOLDER) / 'Event_Registry_For_Validation.csv'
    with NamedTemporaryFile(dir=output.parent, prefix=output.stem + '.', suffix='.tmp', delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        validation.to_csv(temp_path, index=False, encoding='utf-8-sig')
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"✅ [连续缺测验证] 事件登记表已写出：{output}（{len(validation)} 个事件）")


# ===================== 3. Main workflow =====================
def main():
    print("🚀 [脚本1] 开始事件识别与切片 (完美保留原表全部列)...")

    # Load reference-period meteorology.
    base_data = read_all_csv(BASE_FOLDER, ['Tmax'])
    # Load the completed data with all columns, converting only the specified columns to numeric types.
    orig_data = read_all_csv(ORIGINAL_FOLDER, QC_TARGET_COLS + ['Tmax'])

    type_map = {'flood': 'Flood_Only', 'hw': 'Heatwave_Only', 'direct': 'Direct_Compound',
                'indirect_h2f': 'H5F_Compound', 'indirect_f2h': 'F5H_Compound'}

    all_registry = []
    all_ev_data = []
    all_pre_data = []
    all_post_data = []
    all_non_event_data = []

    for site in set(base_data.keys()) & set(orig_data.keys()):
        b_df, o_df = base_data[site], orig_data[site]

        # Append static attributes to the output for event, reference, recovery and non-event records.
        o_df = add_static_attributes(o_df)
        # Discard the upstream Event_ID and assign IDs in ED_F to avoid duplicate columns and keep downstream event joins consistent.
        o_df.drop(columns=['Event_ID'], inplace=True, errors='ignore')

        region = o_df[REGION_COL].dropna().iloc[0] if REGION_COL in o_df.columns and not o_df[REGION_COL].dropna().empty else 'Unknown'
        region_upper = int(o_df['Region_Upper'].iloc[0])
        region_middle = int(o_df['Region_Middle'].iloc[0])
        region_lower = int(o_df['Region_Lower'].iloc[0])

        river_raw = o_df[RIVER_COL].dropna().iloc[0] if RIVER_COL in o_df.columns and not o_df[RIVER_COL].dropna().empty else 'Unknown'
        river_type = o_df[RIVER_TYPE_COL].iloc[0]

        # Read WQS_Lon and WQS_Lat.
        lon_val = o_df[LON_COL].dropna().iloc[0] if LON_COL in o_df.columns and not o_df[
            LON_COL].dropna().empty else np.nan
        lat_val = o_df[LAT_COL].dropna().iloc[0] if LAT_COL in o_df.columns and not o_df[
            LAT_COL].dropna().empty else np.nan

        t_th = np.percentile(b_df['Tmax'].dropna(), TARGET_HW_P)
        q_th = np.percentile(o_df['Q'].dropna(), TARGET_FLOOD_P)

        c_segs, h_segs, f_segs = classify_complex_events(get_base_segments(o_df.copy(), 'hw', t_th, q_th),
                                                         get_base_segments(o_df.copy(), 'flood', t_th, q_th))
        all_segs = c_segs + h_segs + f_segs

        # Sort by event start date to find each station's next event.
        all_segs = sorted(all_segs, key=lambda x: (x['start'], x['end']))

        # Event detection uses only Tmax and Q. Both scripts should produce identical events when those inputs match.
        validation_events = [
            {
                'Event_ID': f"{site}_EV{idx + 1:03d}",
                'Event_Type': type_map.get(ev.get('comp_subtype', ev.get('type')), 'Unknown'),
                'Start_Date': ev['start'],
                'End_Date': ev['end']
            }
            for idx, ev in enumerate(all_segs)
        ]
        validate_event_exclusivity(site, validation_events)

        o_df['Event_Label'] = 0

        for idx, ev in enumerate(all_segs):
            ev_id = f"{site}_EV{idx + 1:03d}"
            ev_type = type_map.get(ev.get('comp_subtype', ev.get('type')), 'Unknown')
            if ev_type not in EXPECTED_EVENT_TYPES: continue

            start, end = ev['start'], ev['end']

            # Start date of the next event at the same station
            next_event_start = all_segs[idx + 1]['start'] if idx + 1 < len(all_segs) else pd.NaT

            # Write WQS_Lon and WQS_Lat to the event registry.
            ev_record = {
                'Event_ID': ev_id, 'WQS_name': site,
                LON_COL: lon_val, LAT_COL: lat_val,
                REGION_COL: region,
                'Region_Upper': region_upper,
                'Region_Middle': region_middle,
                'Region_Lower': region_lower,
                RIVER_COL: river_raw,
                RIVER_TYPE_COL: river_type,
                'Event_Type': ev_type,
                'Start_Date': start, 'End_Date': end, 'Total_Days': (end - start).days + 1
            }
            all_registry.append(ev_record)

            mask_ev = (o_df['Date'] >= start) & (o_df['Date'] <= end)
            o_df.loc[mask_ev, 'Event_Label'] = 1
            ev_df = o_df[mask_ev].copy()
            ev_df.insert(0, 'Event_ID', ev_id)
            ev_df['Event_Type'] = ev_type
            all_ev_data.append(ev_df)

            # Use only non-event days in the short-term reference window, 9–4 days before onset, to avoid overlap with nearby extremes.
            mask_pre = (
                (o_df['Date'] >= start - pd.Timedelta(days=9))
                & (o_df['Date'] <= start - pd.Timedelta(days=4))
                & (o_df['Event_Label'] == 0)
            )
            pre_df = o_df[mask_pre].copy()
            pre_df.insert(0, 'Event_ID', ev_id)
            pre_df['Event_Type'] = ev_type
            all_pre_data.append(pre_df)

            # The default recovery window is days 1–30 after event end.
            post_start = end + pd.Timedelta(days=1)
            post_end = end + pd.Timedelta(days=30)

            # If another event starts in the 30-day window, end recovery one day before its onset.
            if pd.notna(next_event_start) and next_event_start <= post_end:
                post_end = next_event_start - pd.Timedelta(days=1)

            # Extract recovery records only when dates remain after truncation.
            if post_end >= post_start:
                mask_post = (o_df['Date'] >= post_start) & (o_df['Date'] <= post_end)
                post_df = o_df[mask_post].copy()

                if not post_df.empty:
                    post_df.insert(0, 'Event_ID', ev_id)
                    post_df['Event_Type'] = ev_type
                    post_df['Day_Offset'] = (post_df['Date'] - end).dt.days
                    all_post_data.append(post_df)

        non_ev_df = o_df[o_df['Event_Label'] == 0].copy()
        all_non_event_data.append(non_ev_df)

    print("\n" + "=" * 60 + "\n正在导出数据 (By_Site -> 事件类型平铺)\n" + "=" * 60)

    # ===================== Five main output directories =====================
    export_dirs = {
        "Registry": os.path.join(RESULT_FOLDER, "01_Registry_Data"),
        "Pre_Baseline": os.path.join(RESULT_FOLDER, "02_Pre_Baseline_Data"),
        "Event_Period": os.path.join(RESULT_FOLDER, "03_Event_Period_Data"),
        "Post_Recovery": os.path.join(RESULT_FOLDER, "04_Post_Recovery_Data"),
        "Non_Event": os.path.join(RESULT_FOLDER, "05_Non_Event_Data")
    }

    for dir_path in export_dirs.values():
        os.makedirs(os.path.join(dir_path, "Overall"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "By_Site"), exist_ok=True)

    # ===================== 1. Export non-event records =====================
    if all_non_event_data:
        df_non = pd.concat(all_non_event_data, ignore_index=True)
        df_non.drop(columns=['Event_Label'], inplace=True, errors='ignore')

        df_non.to_csv(os.path.join(export_dirs["Non_Event"], "Overall", "05_Non_Event_Overall.csv"), index=False,
                      encoding='utf-8-sig')
        for site, group in df_non.groupby('WQS_name'):
            group.to_csv(os.path.join(export_dirs["Non_Event"], "By_Site", f"{site}_05_Non_Event.csv"), index=False,
                         encoding='utf-8-sig')
        print(f"✅ [非事件期] 数据分装完成")

    # ===================== 2. Export event and reference segments =====================
    registry_df = pd.DataFrame.from_records(all_registry, columns=REGISTRY_COLUMNS)
    datasets_to_export = [
        ("Registry", registry_df, "01_Registry"),
        ("Pre_Baseline", pd.concat(all_pre_data, ignore_index=True) if all_pre_data else pd.DataFrame(),
         "02_Pre_Baseline"),
        ("Event_Period", pd.concat(all_ev_data, ignore_index=True) if all_ev_data else pd.DataFrame(),
         "03_Event_Period"),
        ("Post_Recovery", pd.concat(all_post_data, ignore_index=True) if all_post_data else pd.DataFrame(),
         "04_Post_Recovery")
    ]

    for dict_key, df_main, file_prefix in datasets_to_export:
        if df_main.empty: continue

        for ev_type in EXPECTED_EVENT_TYPES:
            df_ev = df_main[df_main['Event_Type'] == ev_type]
            if df_ev.empty: continue

            # Export basin-wide output (Overall).
            overall_path = os.path.join(export_dirs[dict_key], "Overall", f"{ev_type}_{file_prefix}_Overall.csv")
            df_ev.to_csv(overall_path, index=False, encoding='utf-8-sig')

            # Export event records by station and event type (By_Site -> Event_Type).
            ev_dir = os.path.join(export_dirs[dict_key], "By_Site", ev_type)
            os.makedirs(ev_dir, exist_ok=True)

            for site, group in df_ev.groupby('WQS_name'):
                site_path = os.path.join(ev_dir, f"{site}_{ev_type}_{file_prefix}.csv")
                group.to_csv(site_path, index=False, encoding='utf-8-sig')

        print(f"✅ [{file_prefix}] 切片分装完成")

    write_validation_registry(registry_df)

    print("\n--- 事件统计核对 ---")
    print(f"Total events: {len(registry_df)}")
    for ev_type in EXPECTED_EVENT_TYPES:
        count = (registry_df['Event_Type'] == ev_type).sum() if not registry_df.empty else 0
        print(f"{ev_type}: {count}")

    print("\n🎉 脚本一全流程完成！原始数据列已保留，并仅新增 Region_Upper、Region_Middle、Region_Lower 和 River_Type 四个静态属性！")


if __name__ == "__main__":
    main()
