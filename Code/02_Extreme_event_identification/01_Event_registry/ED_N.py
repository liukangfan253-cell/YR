# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ===================== 1. Paths and parameters =====================
ROOT = Path(__file__).resolve().parents[3]
BASE_FOLDER = ROOT / "Data/Meteorology_data/1991-2020/YR"
ORIGINAL_FOLDER = ROOT / "Data/WQ_Hy_M_DYN_data/YR/by_station_all"
RESULT_FOLDER = str(ROOT / 'Data/Event_N')

# Flood and heatwave thresholds use the 95th and 90th percentiles, respectively.
TARGET_HW_P, TARGET_FLOOD_P = 90, 95
# Require at least three heatwave days, merge gaps of at most two days, and use a five-day compound-event window.
HW_CONTINUE_DAYS, HW_MERGE_INTERVAL, COMP_INTERVAL = 3, 2, 5

# Quality-controlled water-quality columns used only to classify event coverage as Complete or Missing.
# WL, Q and Ss are filled during preprocessing and do not determine original water-quality coverage.
QC_TARGET_COLS = ['DO', 'WT', 'pH', 'EC', 'NH₃-N', 'TP', 'TN', 'CODₘₙ', 'Turb']

# Base columns and hydrological variables retained for event detection and labeled output.
# Use Q to calculate the flood threshold and identify flood episodes.
EVENT_REQUIRED_COLS = ['Date', 'WQS_name', 'Region', 'Tmax', 'WL', 'Q', 'Ss']
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ===================== Event detection functions =====================
def read_all_csv(folder_path, required_cols, data_type="base"):
    site_data = {}
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"❌ [{data_type}] 未找到CSV文件：{folder_path}")
        return site_data

    for file in csv_files:
        try:
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                df = pd.read_csv(file_path, encoding='gbk')

            df.columns = df.columns.str.strip().str.replace('\n', '').str.replace('\r', '')
            df.rename(columns={'date': 'Date', 'R': 'Region'}, inplace=True)

            if missing_cols := [col for col in required_cols if col not in df.columns]:
                print(f"⚠️ [{data_type}] 文件缺失列: {file} -> {missing_cols}")
                continue

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
            site_name = str(df['WQS_name'].iloc[0]).strip()

            for col in required_cols:
                if col not in ['Date', 'WQS_name', 'Region']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            site_data[site_name] = df
        except Exception as e:
            print(f"❌ [{data_type}] 文件读取失败: {file}\n   原因: {e}")

    print(f"✅ [{data_type}] 成功读取 {len(site_data)} 个站点")
    return site_data


def mark_threshold_days(df, t_thresh, q_thresh):
    df = df.copy()
    df['hw_flag'] = (df['Tmax'] >= t_thresh).astype(int)
    df['flood_flag'] = (df['Q'] >= q_thresh).astype(int)
    return df


def get_base_segments(df, event_type):
    flag_col = f"{event_type}_flag"
    df = df.copy().reset_index(drop=True)
    df['group'] = (df[flag_col] != df[flag_col].shift(1)).cumsum()

    segments = []
    for _, group_df in df[df[flag_col] == 1].groupby('group'):
        start, end = group_df['Date'].min(), group_df['Date'].max()
        days = (end - start).days + 1

        if event_type == 'hw' and days < HW_CONTINUE_DAYS:
            continue
        if event_type == 'hw' and segments:
            gap_days = (start - segments[-1]['end']).days - 1
            if 0 <= gap_days <= HW_MERGE_INTERVAL:
                segments[-1]['end'] = end
                segments[-1]['days'] = (end - segments[-1]['start']).days + 1
                continue

        segments.append({'start': start, 'end': end, 'days': days, 'type': event_type})
    return segments


def classify_complex_events(hw_segs, flood_segs):
    hw_df, flood_df = pd.DataFrame(hw_segs), pd.DataFrame(flood_segs)
    comp_segs, used_hw, used_fl = [], set(), set()

    if not hw_df.empty and not flood_df.empty:
        # Prioritize DC events by merging heatwaves and floods connected by at least one day of overlap.
        adjacency = {}
        for i, h in hw_df.iterrows():
            h_node = ('h', i)
            adjacency.setdefault(h_node, set())
            for j, f in flood_df.iterrows():
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
                starts = [hw_df.loc[idx, 'start'] for idx in hw_ids]
                starts.extend(flood_df.loc[idx, 'start'] for idx in fl_ids)
                ends = [hw_df.loc[idx, 'end'] for idx in hw_ids]
                ends.extend(flood_df.loc[idx, 'end'] for idx in fl_ids)
                c_start, c_end = min(starts), max(ends)
                comp_segs.append(
                    {'comp_subtype': 'direct', 'start': c_start, 'end': c_end,
                     'days': (c_end - c_start).days + 1})
                used_hw.update(hw_ids)
                used_fl.update(fl_ids)

        un_hw, un_fl = hw_df[~hw_df.index.isin(used_hw)], flood_df[~flood_df.index.isin(used_fl)]

        # For F5H, build all candidates with gaps of 0–5 days, then match distinct pairs in order of shortest gap.
        f5h_candidates = []
        for i, f in un_fl.iterrows():
            for j, h in un_hw.iterrows():
                if h['start'] > f['end']:
                    gap_days = (h['start'] - f['end']).days - 1
                    if 0 <= gap_days <= COMP_INTERVAL:
                        f5h_candidates.append({
                            'gap_days': gap_days, 'flood_idx': i, 'hw_idx': j,
                            'flood_start': f['start'], 'flood_end': f['end'],
                            'hw_start': h['start'], 'hw_end': h['end']
                        })

        f5h_candidates.sort(key=lambda x: (
            x['gap_days'], x['hw_start'], -x['flood_end'].value,
            x['flood_idx'], x['hw_idx']
        ))
        for candidate in f5h_candidates:
            i, j = candidate['flood_idx'], candidate['hw_idx']
            if i in used_fl or j in used_hw:
                continue
            comp_segs.append({
                'comp_subtype': 'indirect_f2h',
                'start': candidate['flood_start'], 'end': candidate['hw_end'],
                'days': (candidate['hw_end'] - candidate['flood_start']).days + 1
            })
            used_fl.add(i)
            used_hw.add(j)

        # Rebuild the unused-event pools after F5H matching so each base event is used at most once.
        un_hw = hw_df[~hw_df.index.isin(used_hw)]
        un_fl = flood_df[~flood_df.index.isin(used_fl)]
        # For H5F, also match distinct pairs in order of the shortest gap among all candidates.
        h5f_candidates = []
        for i, h in un_hw.iterrows():
            for j, f in un_fl.iterrows():
                if f['start'] > h['end']:
                    gap_days = (f['start'] - h['end']).days - 1
                    if 0 <= gap_days <= COMP_INTERVAL:
                        h5f_candidates.append({
                            'gap_days': gap_days, 'hw_idx': i, 'flood_idx': j,
                            'hw_start': h['start'], 'hw_end': h['end'],
                            'flood_start': f['start'], 'flood_end': f['end']
                        })

        h5f_candidates.sort(key=lambda x: (
            x['gap_days'], x['flood_start'], -x['hw_end'].value,
            x['hw_idx'], x['flood_idx']
        ))
        for candidate in h5f_candidates:
            i, j = candidate['hw_idx'], candidate['flood_idx']
            if i in used_hw or j in used_fl:
                continue
            comp_segs.append({
                'comp_subtype': 'indirect_h2f',
                'start': candidate['hw_start'], 'end': candidate['flood_end'],
                'days': (candidate['flood_end'] - candidate['hw_start']).days + 1
            })
            used_hw.add(i)
            used_fl.add(j)

    rem_hw = [row for i, row in hw_df.iterrows() if i not in used_hw]
    rem_fl = [row for j, row in flood_df.iterrows() if j not in used_fl]
    return comp_segs, rem_hw, rem_fl


def validate_event_exclusivity(site_name, events):
    """Validate the final events without modifying them. Check that one event cannot overwrite another event's daily labels."""
    event_ids = [ev['Event_ID'] for ev in events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError(f"站点 {site_name} 存在重复 Event_ID: {event_ids}")

    ordered = sorted(events, key=lambda ev: (pd.to_datetime(ev['Start_Date']), pd.to_datetime(ev['End_Date'])))
    occupied_dates = set()
    previous = None
    for current in ordered:
        current_start = pd.to_datetime(current['Start_Date'])
        current_end = pd.to_datetime(current['End_Date'])
        if previous is not None and current_start <= pd.to_datetime(previous['End_Date']):
            raise RuntimeError(
                f"站点 {site_name} 最终事件时间窗重叠: "
                f"{previous['Event_Type']} {previous['Start_Date']}~{previous['End_Date']}；"
                f"{current['Event_Type']} {current['Start_Date']}~{current['End_Date']}"
            )
        event_dates = set(pd.date_range(current_start, current_end))
        if occupied_dates & event_dates:
            raise RuntimeError(
                f"站点 {site_name} 同一日期属于多个最终事件: "
                f"{current['Event_Type']} {current['Start_Date']}~{current['End_Date']}"
            )
        occupied_dates.update(event_dates)
        previous = current


def check_event_missing_status(events, orig_df, site_name):
    event_records = []
    for ev in events:
        sub_orig = orig_df[(orig_df['Date'] >= ev['start']) & (orig_df['Date'] <= ev['end'])]

        if not sub_orig.empty and all(c in sub_orig.columns for c in QC_TARGET_COLS):
            valid_count = sub_orig[QC_TARGET_COLS].notna().all(axis=1).sum()
        else:
            valid_count = 0

        missing_days = ev['days'] - valid_count
        status = "Complete" if missing_days == 0 else "Missing"

        event_records.append({
            'WQS_name': site_name,
            'Event_Type': ev.get('comp_subtype', ev.get('type', 'single')),
            'Start_Date': ev['start'].strftime('%Y-%m-%d'),
            'End_Date': ev['end'].strftime('%Y-%m-%d'),
            'Total_Days': ev['days'],
            'Valid_Days': valid_count,
            'Missing_Days': missing_days,
            'Missing_Status': status
        })
    return event_records


# ===================== English frequency table utilities =====================
def generate_detailed_counts(df_events, group_col, expected_events):
    counts = df_events.groupby([group_col, 'Event_Type', 'Missing_Status']).size().reset_index(name='Count')
    pivot_counts = counts.pivot_table(index=[group_col, 'Event_Type'], columns='Missing_Status', values='Count',
                                      fill_value=0).reset_index()

    if 'Complete' not in pivot_counts.columns: pivot_counts['Complete'] = 0
    if 'Missing' not in pivot_counts.columns: pivot_counts['Missing'] = 0

    pivot_counts.rename(columns={'Complete': 'Complete_Count', 'Missing': 'Missing_Count'}, inplace=True)
    pivot_counts['Total_Count'] = pivot_counts['Complete_Count'] + pivot_counts['Missing_Count']

    final_df = pivot_counts.pivot(index=group_col, columns='Event_Type',
                                  values=['Complete_Count', 'Missing_Count', 'Total_Count']).fillna(0).astype(int)

    flat_cols = []
    for ev in expected_events:
        for metric in ['Total_Count', 'Complete_Count', 'Missing_Count']:
            col_tuple = (metric, ev)
            if col_tuple in final_df.columns:
                flat_cols.append(col_tuple)

    final_df = final_df[flat_cols]
    final_df.columns = [f"{ev}_{metric}" for metric, ev in flat_cols]

    return final_df.reset_index()


# ===================== Main workflow =====================
def main():
    print(f"🚀 核心事件分类与质控启动 | HW={TARGET_HW_P}% | FL={TARGET_FLOOD_P}%")

    base_data = read_all_csv(BASE_FOLDER, ['Date', 'Tmax', 'WQS_name'], "Baseline_Meteo")

    orig_cols = list(set(EVENT_REQUIRED_COLS + QC_TARGET_COLS))
    orig_data = read_all_csv(ORIGINAL_FOLDER, orig_cols, "Original_Data")

    common_sites = list(set(base_data.keys()) & set(orig_data.keys()))
    if not common_sites:
        print("\n❌ 数据无交集站点，请检查路径！")
        return
    print(f"\n✅ 成功匹配 {len(common_sites)} 个站点，开始扫描极端气候事件...")

    # Folder for labeled station data
    site_labeled_dir = os.path.join(RESULT_FOLDER, "By_Site_Labeled")
    os.makedirs(site_labeled_dir, exist_ok=True)

    all_event_logs = []
    all_event_perfect_data = []
    all_event_missing_data = []
    all_non_event_data = []

    # Map English event labels to numeric IDs.
    label_map_dict = {
        'Flood_Only': 1,
        'Heatwave_Only': 2,
        'Direct_Compound': 3,
        'H5F_Compound': 4,
        'F5H_Compound': 5
    }

    for site in common_sites:
        b_df, o_df = base_data[site], orig_data[site]

        region = str(o_df['Region'].dropna().iloc[0]).strip() if 'Region' in o_df.columns and not o_df[
            'Region'].dropna().empty else 'Unknown'

        t_thresh = np.percentile(b_df['Tmax'].dropna(), TARGET_HW_P)
        q_thresh = np.percentile(o_df['Q'].dropna(), TARGET_FLOOD_P)
        if pd.isna(t_thresh) or pd.isna(q_thresh): continue

        o_marked = mark_threshold_days(o_df, t_thresh, q_thresh)
        c_segs, h_segs, f_segs = classify_complex_events(get_base_segments(o_marked, 'hw'),
                                                         get_base_segments(o_marked, 'flood'))

        site_events = []
        site_events.extend(check_event_missing_status(c_segs, o_df, site))
        site_events.extend(check_event_missing_status(h_segs, o_df, site))
        site_events.extend(check_event_missing_status(f_segs, o_df, site))

        # Initialize event labels.
        # Event_Label = 0 denotes a non-event period.
        # Event_ID identifies one complete event at one station.
        # Use it to group events in the SHAP training/test split.
        o_df.insert(1, 'Event_Label', 0)
        o_df.insert(2, 'Event_ID', pd.NA)

        # For stable Event_ID values and easy verification,
        # sort by event start date, then by end date.
        site_events = sorted(
            site_events,
            key=lambda x: (
                pd.to_datetime(x['Start_Date']),
                pd.to_datetime(x['End_Date'])
            )
        )

        for idx, record in enumerate(site_events):
            type_mapping = {
                'flood': 'Flood_Only', 'hw': 'Heatwave_Only',
                'direct': 'Direct_Compound', 'indirect_h2f': 'H5F_Compound', 'indirect_f2h': 'F5H_Compound'
            }
            record['Event_Type'] = type_mapping.get(record['Event_Type'], record['Event_Type'])
            record['Region'] = region
            record['Event_ID'] = f"{site}_EV{idx + 1:03d}"
        # Event detection uses only Tmax and Q. Both scripts should produce identical events when those inputs match.
        validate_event_exclusivity(site, site_events)

        for record in site_events:
            all_event_logs.append(record)

            start_dt = pd.to_datetime(record['Start_Date'])
            end_dt = pd.to_datetime(record['End_Date'])

            mask = (o_df['Date'] >= start_dt) & (o_df['Date'] <= end_dt)

            # Update numeric labels 0–5.
            o_df.loc[mask, 'Event_Label'] = label_map_dict.get(record['Event_Type'], 0)

            # Event identifiers
            # Assign the same Event_ID to every day of an event.
            # For example, reuse Danzhengxia_EV001 on all five days of a five-day event.
            # Event_ID includes the station name and can identify groups across the basin.
            o_df.loc[mask, 'Event_ID'] = record['Event_ID']

            # Collect records in basin-wide tables by missingness status.
            # Both Perfect and Missing event tables retain Event_Label and Event_ID.
            event_df = o_df[mask].copy()
            if record['Missing_Status'] == 'Complete':
                all_event_perfect_data.append(event_df)
            else:
                all_event_missing_data.append(event_df)

        # Collect non-event records (label 0).
        non_event_mask = o_df['Event_Label'] == 0
        non_event_df = o_df[non_event_mask].copy()
        if not non_event_df.empty:
            all_non_event_data.append(non_event_df)

        # Export the complete set of numerically labeled records for each station.
        site_out_path = os.path.join(site_labeled_dir, f"{site}_Labeled.csv")
        o_df.to_csv(site_out_path, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60 + "\n正在生成汇总表格与明细数据集\n" + "=" * 60)

    # ================== 1. Export event statistics and registries ==================
    if all_event_logs:
        df_events = pd.DataFrame(all_event_logs)

        cols_order = ['Event_ID', 'WQS_name', 'Region', 'Event_Type', 'Start_Date', 'End_Date',
                      'Total_Days', 'Valid_Days', 'Missing_Days', 'Missing_Status']
        df_events = df_events[cols_order]
        event_status_path = os.path.join(RESULT_FOLDER, "All_Events_Missing_Status.csv")
        df_events.to_csv(event_status_path, index=False, encoding='utf-8-sig')
        print(f"✅ 事件详细记录台账已生成: {event_status_path}")

        expected_events = ['Flood_Only', 'Heatwave_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']
        site_base = pd.concat([orig_data[s][['WQS_name', 'Region']].drop_duplicates() for s in common_sites],
                              ignore_index=True)

        # Detailed regional statistics
        region_counts_df = generate_detailed_counts(df_events, 'Region', expected_events)
        site_counts_region = site_base.groupby('Region').size().reset_index(name='Total_Sites')
        region_final_df = pd.merge(site_counts_region, region_counts_df, on='Region', how='left').fillna(0)

        total_cols = [c for c in region_final_df.columns if 'Total_Count' in c]
        region_final_df['All_Events_Total'] = region_final_df[total_cols].sum(axis=1)
        region_final_df['Avg_Events_Per_Site'] = (
                region_final_df['All_Events_Total'] / region_final_df['Total_Sites']).round(2)
        region_final_df.to_csv(os.path.join(RESULT_FOLDER, "Event_Counts_By_Region.csv"), index=False,
                               encoding='utf-8-sig')

        # Detailed station statistics including Region
        site_counts_df = generate_detailed_counts(df_events, 'WQS_name', expected_events)
        site_final_df = pd.merge(site_base[['WQS_name', 'Region']], site_counts_df, on='WQS_name', how='left').fillna(0)

        site_total_cols = [c for c in site_final_df.columns if 'Total_Count' in c]
        site_final_df['All_Events_Total'] = site_final_df[site_total_cols].sum(axis=1)
        site_final_df.to_csv(os.path.join(RESULT_FOLDER, "Site_Event_Counts.csv"), index=False, encoding='utf-8-sig')
        print("✅ 区域及全站点【Complete/Missing】英文频次统计已生成")

        print("\n--- 事件统计核对 ---")
        print(f"Total events: {len(df_events)}")
        for ev_type in expected_events:
            print(f"{ev_type}: {(df_events['Event_Type'] == ev_type).sum()}")
        print(f"Complete: {(df_events['Missing_Status'] == 'Complete').sum()}")
        print(f"Missing: {(df_events['Missing_Status'] == 'Missing').sum()}")

    # ================== 2. Export records by coverage group ==================
    print("\n--- 正在导出时序数据集 (整体 + 按单站点分装) ---")

    # 1. Process non-event data (Non_Events_Data)
    if all_non_event_data:
        non_event_dir = os.path.join(RESULT_FOLDER, "Non_Events_Data")
        non_event_overall_dir = os.path.join(non_event_dir, "Overall")
        non_event_site_dir = os.path.join(non_event_dir, "By_Site")
        os.makedirs(non_event_overall_dir, exist_ok=True)
        os.makedirs(non_event_site_dir, exist_ok=True)

        df_non_event = pd.concat(all_non_event_data, ignore_index=True)
        # Export the combined table.
        df_non_event.to_csv(os.path.join(non_event_overall_dir, "Data_Non_Event_Overall.csv"), index=False,
                            encoding='utf-8-sig')
        # Export records by station.
        for site, group in df_non_event.groupby('WQS_name'):
            group.to_csv(os.path.join(non_event_site_dir, f"{site}_Non_Event.csv"), index=False, encoding='utf-8-sig')

        print(f"✅ 非事件期平稳数据已生成并分装 ({len(df_non_event)} 行)")

    # 2. Process complete events (Perfect_Events_Data)
    if all_event_perfect_data:
        perfect_dir = os.path.join(RESULT_FOLDER, "Perfect_Events_Data")
        perfect_overall_dir = os.path.join(perfect_dir, "Overall")
        perfect_site_dir = os.path.join(perfect_dir, "By_Site")
        os.makedirs(perfect_overall_dir, exist_ok=True)
        os.makedirs(perfect_site_dir, exist_ok=True)

        df_perfect = pd.concat(all_event_perfect_data, ignore_index=True)
        for ev_type in expected_events:
            label_id = label_map_dict[ev_type]
            df_ev = df_perfect[df_perfect['Event_Label'] == label_id]
            if not df_ev.empty:
                # Export the combined table.
                df_ev.to_csv(os.path.join(perfect_overall_dir, f"{ev_type}_Perfect_Overall.csv"), index=False,
                             encoding='utf-8-sig')
                # Export records by station.
                for site, group in df_ev.groupby('WQS_name'):
                    group.to_csv(os.path.join(perfect_site_dir, f"{site}_{ev_type}_Perfect.csv"), index=False,
                                 encoding='utf-8-sig')

    # 3. Process events with missing data (Missing_Events_Data)
    if all_event_missing_data:
        missing_dir = os.path.join(RESULT_FOLDER, "Missing_Events_Data")
        missing_overall_dir = os.path.join(missing_dir, "Overall")
        missing_site_dir = os.path.join(missing_dir, "By_Site")
        os.makedirs(missing_overall_dir, exist_ok=True)
        os.makedirs(missing_site_dir, exist_ok=True)

        df_missing = pd.concat(all_event_missing_data, ignore_index=True)
        for ev_type in expected_events:
            label_id = label_map_dict[ev_type]
            df_ev = df_missing[df_missing['Event_Label'] == label_id]
            if not df_ev.empty:
                # Export the combined table.
                df_ev.to_csv(os.path.join(missing_overall_dir, f"{ev_type}_Missing_Overall.csv"), index=False,
                             encoding='utf-8-sig')
                # Export records by station.
                for site, group in df_ev.groupby('WQS_name'):
                    group.to_csv(os.path.join(missing_site_dir, f"{site}_{ev_type}_Missing.csv"), index=False,
                                 encoding='utf-8-sig')

    print(f"✅ 包含【Event_Label】(0~5 数值型标签) 和【Event_ID】的全量分站文件夹 (By_Site_Labeled) 已生成！")
    print("📌 Event_ID 已同步写入 Perfect/Missing 事件明细，可直接用于后续 GroupShuffleSplit / GroupKFold。")
    print("\n🎉 事件分类、诊断与文件嵌套分装全部完成！")


if __name__ == "__main__":
    main()
