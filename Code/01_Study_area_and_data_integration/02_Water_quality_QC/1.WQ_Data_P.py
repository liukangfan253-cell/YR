import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
from collections import defaultdict
import chardet
from pypinyin import lazy_pinyin

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

warnings.simplefilter(action='ignore', category=FutureWarning)

# ======================
# 1. Batch configuration and physical bounds
# ======================
input_folder = str(ROOT / 'Data/WaterQuality_data/WQ_data_YR')

# Daily ERA5-Land data exported by script 3 for each station. SP is in hPa.
ERA5_SP_FOLDER = str(ROOT / 'Data/Meteorology_data/2021-2025/YR_V1')
MIN_SP_MATCH_RATIO = 0.95

# Separate screened (Cleaned) and unscreened (Raw) output directories.
output_folder_retained_clean = str(ROOT / 'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned')
output_folder_retained_raw = str(ROOT / 'Data/WaterQuality_data_P/WQ_data_YR/Retained/Raw')
output_folder_excluded_clean = str(ROOT / 'Data/WaterQuality_data_P/WQ_data_YR/Excluded/Cleaned')
output_folder_excluded_raw = str(ROOT / 'Data/WaterQuality_data_P/WQ_data_YR/Excluded/Raw')
summary_output_folder = str(ROOT / 'Result/WQ_data_missing/WQ_data_YR_Summary')

# Map Chinese column names to abbreviations
col_map = {
    '站点名称': 'WQS_name', '经度': 'WQS_Lon', '纬度': 'WQS_Lat',
    '河流': 'River',
    '监测时间': 'Date',
    '水温': 'WT', 'pH': 'pH', '溶解氧': 'DO', '电导率': 'EC',
    '浊度': 'Turb',
    '氨氮': 'NH₃-N',
    '总磷': 'TP', '总氮': 'TN',
    '高锰酸盐指数': 'CODₘₙ',
    'DO_sat': 'DO_sat',          # Use consistent column names.
    'DO_1.5_sat': 'DO_1.5_sat'   # Use consistent column names.
}

month_abbr = {
    '1月': 'Jan', '2月': 'Feb', '3月': 'Mar', '4月': 'Apr',
    '5月': 'May', '6月': 'Jun', '7月': 'Jul', '8月': 'Aug',
    '9月': 'Sep', '10月': 'Oct', '11月': 'Nov', '12月': 'Dec'
}

# Include both DO saturation fields in the daily aggregation.
avg_cols = ['水温', 'pH', '溶解氧', '电导率', '浊度', '氨氮', '总磷', '总氮', '高锰酸盐指数', 'DO_sat', 'DO_1.5_sat']

# Static physical bounds (lower and upper limits)
quality_bounds = {
    '水温': (-10, 45),
    'pH': (0, 14),
    '电导率': (0, 999999),
    '浊度': (0, 999999),
    '氨氮': (0, 500),
    '总磷': (0, 100),
    '总氮': (0, 500),
    '高锰酸盐指数': (0, 1000)
}


# ======================
# 2. Main calculations and utilities
# ======================
def calc_do_sat(temp_c, pressure_hpa=None):
    """Estimate freshwater oxygen saturation (mg/L) for DO screening.

    The temperature term uses the Benson–Krause freshwater formula in equation (7) of the USGS (2011) attachment.
    The pressure term uses pressure_hpa / 1013.25 without salinity, water vapor or nonideal-gas corrections.
    Within this calculation, temperatures below 0 °C are treated as 0 °C, missing temperatures stay NaN, and missing pressures default to 1013.25 hPa.
    The caller applies the screening workflow's rounding.
    """
    t = np.where(pd.isna(temp_c), np.nan, temp_c)
    t = np.where(t < 0, 0, t)
    T_k = t + 273.15
    ln_do = -139.34411 + (1.575701e5 / T_k) - (6.642308e7 / (T_k ** 2)) + (1.243800e10 / (T_k ** 3)) - (
                8.621949e11 / (T_k ** 4))

    if pressure_hpa is None:
        p = 1013.25
    else:
        p = np.where(pd.isna(pressure_hpa), 1013.25, pressure_hpa)

    do_sat = np.exp(ln_do) * (p / 1013.25)
    return do_sat


def translate_site_name(site_name):
    if pd.isna(site_name): return site_name
    site_name_str = str(site_name).strip()
    pinyin_list = lazy_pinyin(site_name_str)
    if not pinyin_list: return site_name_str
    return ''.join(pinyin_list).capitalize()


def translate_river(river_name):
    if pd.isna(river_name): return river_name
    river_str = str(river_name).strip()
    if '黄河' in river_str: return 'Yellow_river'
    pinyin_list = lazy_pinyin(river_str)
    if not pinyin_list: return river_str
    return ''.join(pinyin_list).capitalize()


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding']


def read_csv_with_encoding(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030']
    detected_encoding = detect_encoding(file_path)
    if detected_encoding: encodings.insert(0, detected_encoding)

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            return df, encoding
        except:
            continue
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8', errors='backslashreplace')
        return df, 'utf-8-with-errors'
    except:
        return None, None


def load_era5_sp_for_site(site_name):
    """Read the daily ERA5-Land file with the same name as the current water-quality station."""
    era5_path = os.path.join(ERA5_SP_FOLDER, f'{site_name}.csv')
    if not os.path.isfile(era5_path):
        raise FileNotFoundError(f'找不到当前断面的ERA5文件：{era5_path}')

    era5_df, _ = read_csv_with_encoding(era5_path)
    if era5_df is None:
        raise RuntimeError(f'无法读取ERA5文件：{era5_path}')
    missing_cols = [col for col in ['Date', 'SP'] if col not in era5_df.columns]
    if missing_cols:
        raise ValueError(f'ERA5文件 {os.path.basename(era5_path)} 缺少字段：{missing_cols}')

    era5_sp = era5_df[['Date', 'SP']].copy()
    era5_sp['_ERA5_Date'] = pd.to_datetime(
        era5_sp['Date'], errors='coerce'
    ).dt.normalize()
    era5_sp['_SP_ERA5'] = pd.to_numeric(era5_sp['SP'], errors='coerce')
    era5_sp = era5_sp.dropna(subset=['_ERA5_Date'])

    if era5_sp['_ERA5_Date'].duplicated().any():
        raise ValueError(f'ERA5文件 {os.path.basename(era5_path)} 存在重复日期。')
    return era5_sp[['_ERA5_Date', '_SP_ERA5']]


# ======================
# 3. Process one input file
# ======================
def process_single_file(input_path, summary_retained, summary_excluded):
    try:
        filename = os.path.basename(input_path)
        print(f"\n处理: {filename}")

        df_raw, used_encoding = read_csv_with_encoding(input_path)
        if df_raw is None:
            print(f"  无法读取，跳过")
            return

        print(f"  原始: {len(df_raw)}条")

        required_cols = ['监测时间', '站点名称']
        if [col for col in required_cols if col not in df_raw.columns]:
            print(f"  缺少关键列，跳过")
            return

        if '经度' not in df_raw.columns: df_raw['经度'] = np.nan
        if '纬度' not in df_raw.columns: df_raw['纬度'] = np.nan
        if '河流' not in df_raw.columns: df_raw['河流'] = np.nan

        df_raw['监测时间'] = pd.to_datetime(df_raw['监测时间'], errors='coerce')
        df_raw = df_raw[(df_raw['监测时间'].dt.year >= 2021) & (df_raw['监测时间'].dt.year <= 2025)]
        if len(df_raw) == 0:
            print(f"  无有效年份数据，跳过")
            return

        # Keep a normalized date key for joining the daily ERA5-Land SP from script 3.
        # Retain the original observation timestamp format and daily aggregation procedure.
        df_raw['_WQ_Date_Key'] = df_raw['监测时间'].dt.normalize()
        df_raw['监测时间'] = df_raw['监测时间'].dt.strftime('%Y/%m/%d')
        static_cols = ['站点名称', '河流', '经度', '纬度']

        for col in avg_cols:
            if col in df_raw.columns: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        for col in ['经度', '纬度']:
            if col in df_raw.columns: df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

        df_raw['站点名称'] = df_raw['站点名称'].apply(translate_site_name)
        df_raw['河流'] = df_raw['河流'].apply(translate_river)

        # The matching filenames identify the same station. Join the files by date only.
        sites_in_file = df_raw['站点名称'].dropna().unique()
        if len(sites_in_file) != 1:
            raise ValueError(f'水质文件应只包含一个断面，实际为：{sites_in_file.tolist()}')
        site_name = sites_in_file[0]
        era5_sp = load_era5_sp_for_site(site_name)
        df_raw = df_raw.merge(
            era5_sp,
            how='left',
            left_on='_WQ_Date_Key',
            right_on='_ERA5_Date',
            validate='many_to_one',
            sort=False
        )

        matched_sp_records = int(df_raw['_SP_ERA5'].notna().sum())
        fallback_sp_records = int(df_raw['_SP_ERA5'].isna().sum())
        match_ratio = matched_sp_records / len(df_raw) if len(df_raw) else 0.0
        print(f"  [{site_name}] SP匹配率: {match_ratio:.2%}")
        print(
            f"  [{site_name}] SP fallback到1013.25 hPa的记录数: "
            f"{fallback_sp_records}"
        )
        if match_ratio < MIN_SP_MATCH_RATIO:
            raise ValueError(
                f'{site_name} 的SP匹配率 {match_ratio:.2%} 低于 '
                f'{MIN_SP_MATCH_RATIO:.2%}，停止处理。'
            )

        valid_sp = df_raw['_SP_ERA5'].dropna()
        invalid_sp = valid_sp[(valid_sp < 500) | (valid_sp > 1100)]
        if not invalid_sp.empty:
            raise ValueError(
                f'{site_name} 有 {len(invalid_sp)} 条SP不在500–1100 hPa合理范围内'
                f'（最小值={valid_sp.min():.2f}，最大值={valid_sp.max():.2f}）。'
                '请检查ERA5气压单位；程序不会自动转换。'
            )
        print(
            f"  [{site_name}] SP最小值/平均值/最大值: "
            f"{valid_sp.min():.2f}/{valid_sp.mean():.2f}/{valid_sp.max():.2f} hPa"
        )

        # ----------------------------------------------------
        # Compute saturation from temperatures before static screening and round it to three decimals.
        # Multiply saturation by 1.5, round again to three decimals, and save the resulting DO threshold in the Raw table.
        # ----------------------------------------------------
        if '水温' in df_raw.columns:
            df_raw['DO_sat'] = calc_do_sat(
                df_raw['水温'], pressure_hpa=df_raw['_SP_ERA5']
            ).round(3)
            df_raw['DO_1.5_sat'] = (df_raw['DO_sat'] * 1.5).round(3)
        else:
            df_raw['DO_sat'] = np.nan
            df_raw['DO_1.5_sat'] = np.nan

        # ----------------------------------------------------
        # Copy the data for screening after adding the saturation columns.
        # ----------------------------------------------------
        df_clean = df_raw.copy()
        cleaned_count = 0

        # Screen DO using the dynamic upper bound and the absolute limit of 20 mg/L.
        if '溶解氧' in df_clean.columns and 'DO_1.5_sat' in df_clean.columns:
            do_above_pressure_sat = df_clean['溶解氧'] > df_clean['DO_1.5_sat']
            invalid_do = (df_clean['溶解氧'] < 0) | (df_clean['溶解氧'] > 20) | do_above_pressure_sat
            do_cleaned = invalid_do.sum()
            print(
                f"  [{site_name}] 因 DO > 1.5×气压修正DO_sat "
                f"被判异常的记录数: {int(do_above_pressure_sat.sum())}"
            )
            if do_cleaned > 0:
                cleaned_count += do_cleaned
                df_clean.loc[invalid_do, '溶解氧'] = np.nan

        # Apply static physical bounds after screening DO against its saturation threshold.
        # Set out-of-range water temperatures to missing without recalculating DO_sat or screening DO again.
        for col, (min_val, max_val) in quality_bounds.items():
            if col in df_clean.columns:
                invalid_mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
                col_cleaned = invalid_mask.sum()
                if col_cleaned > 0:
                    cleaned_count += col_cleaned
                    df_clean.loc[invalid_mask, col] = np.nan

        if cleaned_count > 0:
            print(f"  ✔ 共清理 {cleaned_count} 个异常数据点 (Raw备份已保留原值)")

        # ----------------------------------------------------
        # Aggregate and align the Raw and Clean tables to the same calendar and row count.
        # ----------------------------------------------------
        def aggregate_and_parse(target_df):
            agg_df = target_df.groupby(['站点名称', '监测时间'], as_index=False).agg(
                {**{col: 'mean' for col in avg_cols},
                 **{col: 'first' for col in static_cols if col != '站点名称'}}
            )
            for col in avg_cols:
                if col in agg_df.columns: agg_df[col] = agg_df[col].round(3)

            agg_df['监测时间'] = agg_df['监测时间'].apply(lambda x: datetime.strptime(x, '%Y/%m/%d') if pd.notna(x) else None)
            return agg_df.dropna(subset=['监测时间'])

        processed_clean = aggregate_and_parse(df_clean)
        processed_raw = aggregate_and_parse(df_raw)

        # Define the common date range.
        start_date, end_date = datetime(2021, 1, 1), datetime(2025, 12, 31)
        full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        total_days = len(full_dates)

        sites = processed_clean['站点名称'].unique()
        site_info = processed_clean.groupby('站点名称').first().reset_index()[static_cols]

        list_clean_all, list_raw_all = [], []
        retained_sites, excluded_sites = [], []

        # ----------------------------------------------------
        # Evaluate missingness and assign retention status. Require at least 30 observed days in 2025.
        # ----------------------------------------------------
        for site in sites:
            site_clean_df = processed_clean[processed_clean['站点名称'] == site]
            years_present = set(d.year for d in site_clean_df['监测时间'])
            required_years = {2021, 2022, 2023, 2024, 2025}

            # Count the valid observation days in 2025.
            days_in_2025 = sum(1 for d in site_clean_df['监测时间'] if d.year == 2025)

            # Exclude stations with an entirely missing year or fewer than 30 valid days in 2025.
            if not required_years.issubset(years_present):
                excluded_sites.append(site)
                target_summary = summary_excluded
                print(f"  [分流 -> 剔除] {site}: 缺失整个必备年份。")
            elif days_in_2025 < 30:
                excluded_sites.append(site)
                target_summary = summary_excluded
                print(f"  [分流 -> 剔除] {site}: 2025年有效数据仅有 {days_in_2025} 天 (未达到最低 30 天红线)！")
            else:
                retained_sites.append(site)
                target_summary = summary_retained

            # Collect counts for the summary files.
            for indicator in avg_cols:
                valid_dates = site_clean_df.dropna(subset=[indicator])['监测时间']
                valid_dates_set = set(valid_dates)
                missing_dates_ind = [date for date in full_dates if date not in valid_dates_set]
                missing_days = len(missing_dates_ind)
                miss_ratio = missing_days / total_days if total_days > 0 else 0

                month_missing, year_missing = defaultdict(int), defaultdict(int)
                for date in missing_dates_ind:
                    month_missing[f"{date.year}年{date.month}月"] += 1
                    year_missing[date.year] += 1

                target_summary[indicator].append({
                    'WQS_name': site, 'TotDays': total_days, 'MissDays': missing_days,
                    'MissPct': round(miss_ratio * 100, 2), 'MissRatio': miss_ratio,
                    'YearMiss2021': year_missing.get(2021, 0), 'YearMiss2022': year_missing.get(2022, 0),
                    'YearMiss2023': year_missing.get(2023, 0), 'YearMiss2024': year_missing.get(2024, 0),
                    'YearMiss2025': year_missing.get(2025, 0), 'MissDist': dict(sorted(month_missing.items()))
                })

        # Align to the study calendar and insert the same missing dates in both tables.
        for site in sites:
            site_details = site_info[site_info['站点名称'] == site].iloc[0]

            def align_dates(site_data):
                site_dates_set = set(site_data['监测时间'])
                missing_dates = [date for date in full_dates if date not in site_dates_set]
                result_list = [site_data]
                if missing_dates:
                    for date in missing_dates:
                        new_row = {col: site_details[col] for col in static_cols}
                        new_row['监测时间'] = date
                        new_row.update({col: None for col in avg_cols})
                        result_list.append(pd.DataFrame([new_row]))
                return pd.concat(result_list, ignore_index=True)

            site_clean_aligned = align_dates(processed_clean[processed_clean['站点名称'] == site])
            site_raw_aligned = align_dates(processed_raw[processed_raw['站点名称'] == site])

            list_clean_all.append(site_clean_aligned)
            list_raw_all.append(site_raw_aligned)

        # Prepare files for export.
        def finalize_and_save(df_list, out_retained, out_excluded):
            if not df_list: return
            df_final = pd.concat(df_list, ignore_index=True)
            df_final = df_final.sort_values(['站点名称', '监测时间']).reset_index(drop=True)
            df_final['监测时间'] = df_final['监测时间'].apply(lambda x: f"{x.year}/{x.month}/{x.day}")

            new_col_order = static_cols + ['监测时间'] + avg_cols
            df_final = df_final[[col for col in new_col_order if col in df_final.columns]]
            df_final.rename(columns=col_map, inplace=True)

            for site in retained_sites:
                site_data = df_final[df_final['WQS_name'] == site]
                safe_name = "".join([c for c in site if c not in r'\/:*?"<>|'])
                site_data.to_csv(os.path.join(out_retained, f"{safe_name}.csv"), index=False, encoding='utf-8-sig')

            for site in excluded_sites:
                site_data = df_final[df_final['WQS_name'] == site]
                safe_name = "".join([c for c in site if c not in r'\/:*?"<>|'])
                site_data.to_csv(os.path.join(out_excluded, f"{safe_name}.csv"), index=False, encoding='utf-8-sig')

        # Export the Clean and Raw tables separately.
        finalize_and_save(list_clean_all, output_folder_retained_clean, output_folder_excluded_clean)
        finalize_and_save(list_raw_all, output_folder_retained_raw, output_folder_excluded_raw)

    except Exception as e:
        print(f"  错误: {str(e)}")


# ======================
# 4. Export missingness summaries
# ======================
def write_summary(summary_list, output_file):
    if not summary_list: return
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('MissRatio', ascending=False).reset_index(drop=True)

    all_months = set()
    for dist_dict in summary_df['MissDist']: all_months.update(dist_dict.keys())
    all_months_sorted = sorted(all_months, key=lambda x: (int(x.split('年')[0]), int(x.split('年')[1].replace('月', ''))))

    for month_cn in all_months_sorted:
        year, month_num = month_cn.split('年')[0], int(month_cn.split('年')[1].replace('月', ''))
        col_name = f"{month_abbr.get(f'{month_num}月', f'M{month_num}')}_{year}"
        summary_df[col_name] = summary_df['MissDist'].apply(lambda x: x.get(month_cn, 0))

    summary_df = summary_df.drop(['MissDist', 'MissRatio'], axis=1)
    summary_df = summary_df.rename(
        columns={'YearMiss2021': '2021', 'YearMiss2022': '2022', 'YearMiss2023': '2023', 'YearMiss2024': '2024',
                 'YearMiss2025': '2025'})
    summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"    📁 已保存: {os.path.basename(output_file)} (平均缺失率: {summary_df['MissPct'].mean():.2f}%)")


# ======================
# 5. Main entry point
# ======================
if __name__ == "__main__":
    # Create the output directory tree.
    for path in [output_folder_retained_clean, output_folder_retained_raw,
                 output_folder_excluded_clean, output_folder_excluded_raw,
                 summary_output_folder]:
        os.makedirs(path, exist_ok=True)

    summary_retained, summary_excluded = {col: [] for col in avg_cols}, {col: [] for col in avg_cols}

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            process_single_file(os.path.join(input_folder, filename), summary_retained, summary_excluded)

    print("\n====== 生成 [保留站点] 缺失汇总 ======")
    for indicator in avg_cols:
        if summary_retained[indicator]:
            write_summary(summary_retained[indicator], os.path.join(summary_output_folder,
                                                                    f"1.{col_map.get(indicator, indicator)}_Retained_Summary.csv"))

    print("\n====== 生成 [剔除站点] 缺失汇总 ======")
    for indicator in avg_cols:
        if summary_excluded[indicator]:
            write_summary(summary_excluded[indicator], os.path.join(summary_output_folder,
                                                                    f"2.{col_map.get(indicator, indicator)}_Excluded_Summary.csv"))

    print("\n✅ 数据预处理第一阶段完成！现在 DO_sat 和 DO_1.5_sat 已经作为原生特征被完美保留。")
