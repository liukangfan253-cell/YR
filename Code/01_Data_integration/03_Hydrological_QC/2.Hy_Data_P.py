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
# 1. Batch configuration
# ======================
input_folder = str(ROOT / 'Data/Hydrology_data/Hy_data_YR')

# Export separate uninterpolated and linearly interpolated daily series and retain station screening results.
output_folder_original = str(ROOT / 'Data/Hydrology_data_P/Hy_data_YR/Original')
output_folder_interpolated = str(ROOT / 'Data/Hydrology_data_P/Hy_data_YR/Interpolated')

summary_output_folder = str(ROOT / 'Result/Hy_data_missing/Hy_data_YR_Summary')

# ======================
# Map source column names to abbreviations.
# ======================
col_map = {
    '水文站站名': 'HyS_name',
    '经度': 'HyS_Lon',
    '纬度': 'HyS_Lat',
    '河流': 'River',
    '区域': 'Region',
    '时间': 'Date',
    '水位': 'WL',
    '流量': 'Q',
    '含沙量': 'Ss'
}

month_abbr = {
    '1月': 'Jan', '2月': 'Feb', '3月': 'Mar', '4月': 'Apr',
    '5月': 'May', '6月': 'Jun', '7月': 'Jul', '8月': 'Aug',
    '9月': 'Sep', '10月': 'Oct', '11月': 'Nov', '12月': 'Dec'
}

avg_cols = ['水位', '流量', '含沙量']


# ======================
# Translation and mapping utilities
# ======================
def translate_hys_name(name):
    """Convert the station name to Pinyin."""
    if pd.isna(name):
        return name

    name_str = str(name).strip()
    pinyin_list = lazy_pinyin(name_str)
    return ''.join(pinyin_list).capitalize() if pinyin_list else name_str


def translate_river(river_name):
    """Map the Yellow River to Yellow_river and transliterate other river names into Pinyin."""
    if pd.isna(river_name):
        return river_name

    river_str = str(river_name).strip()

    if '黄河' in river_str:
        return 'Yellow_river'

    pinyin_list = lazy_pinyin(river_str)
    return ''.join(pinyin_list).capitalize() if pinyin_list else river_str


def translate_region(region_name):
    """Map upstream to UR, midstream to MR, and downstream to LR."""
    if pd.isna(region_name):
        return region_name

    region_str = str(region_name).strip()
    mapping = {'上游': 'UR', '中游': 'MR', '下游': 'LR'}

    for key, val in mapping.items():
        if key in region_str:
            return val

    return region_str


# ======================
# 2. Core processing functions
# ======================
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        return chardet.detect(f.read(10000))['encoding']


def read_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()

    try:
        if ext == '.csv':
            encoding = detect_encoding(file_path) or 'utf-8'
            return pd.read_csv(
                file_path,
                low_memory=False,
                encoding=encoding
            )

        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)

    except Exception as e:
        print(f"  读取失败: {e}")

    return None


def process_single_file(input_path, summary_retained, summary_excluded):
    try:
        filename = os.path.basename(input_path)

        if filename.startswith('~$'):
            return

        print(f"\n处理: {filename}")

        df = read_file(input_path)

        if df is None:
            return

        time_col = '时间' if '时间' in df.columns else '监测时间'
        name_col = '水文站站名'

        if time_col not in df.columns or name_col not in df.columns:
            print("  缺少关键列，跳过")
            return

        # Add required base columns.
        for c in ['经度', '纬度', '河流', '区域']:
            if c not in df.columns:
                df[c] = np.nan

        # Keep dates from 2021 to 2025.
        df[time_col] = pd.to_datetime(
            df[time_col],
            errors='coerce'
        )

        df = df[
            (df[time_col].dt.year >= 2021) &
            (df[time_col].dt.year <= 2025)
        ]

        if len(df) == 0:
            return

        # Translate station metadata.
        df[name_col] = df[name_col].apply(
            translate_hys_name
        )

        df['河流'] = df['河流'].apply(
            translate_river
        )

        df['区域'] = df['区域'].apply(
            translate_region
        )

        # Convert value columns to numeric types.
        for col in avg_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors='coerce'
                )

        # ======================
        # Compute daily means.
        # ======================
        static_cols = [
            name_col,
            '河流',
            '区域',
            '经度',
            '纬度'
        ]

        df['Date_str'] = df[time_col].dt.strftime(
            '%Y/%m/%d'
        )

        processed_df = df.groupby(
            [name_col, 'Date_str'],
            as_index=False
        ).agg(
            {
                **{
                    col: 'mean'
                    for col in avg_cols
                    if col in df.columns
                },
                **{
                    col: 'first'
                    for col in static_cols
                    if col != name_col
                }
            }
        )

        processed_df['监测时间'] = pd.to_datetime(
            processed_df['Date_str']
        )

        # ======================
        # Add all calendar dates from 2021 to 2025.
        # ======================
        full_dates = pd.date_range(
            start='2021-01-01',
            end='2025-12-31',
            freq='D'
        )

        sites = processed_df[name_col].unique()

        # ======================
        # Process each station separately.
        # ======================
        for site in sites:

            site_df = processed_df[
                processed_df[name_col] == site
            ].copy()

            site_info = site_df.iloc[0][static_cols]

            # ==================================
            # Station exclusion criteria
            # ==================================
            tot_days = len(full_dates)

            years_present = set(
                site_df['监测时间'].dt.year
            )

            required_years = {
                2021,
                2022,
                2023,
                2024,
                2025
            }

            missing_years = sorted(
                list(required_years - years_present)
            )

            wl_valid_days = (
                len(
                    set(
                        site_df
                        .dropna(subset=['水位'])
                        ['监测时间']
                    )
                )
                if '水位' in site_df.columns
                else 0
            )

            q_valid_days = (
                len(
                    set(
                        site_df
                        .dropna(subset=['流量'])
                        ['监测时间']
                    )
                )
                if '流量' in site_df.columns
                else 0
            )

            wl_miss_ratio = (
                tot_days - wl_valid_days
            ) / tot_days

            q_miss_ratio = (
                tot_days - q_valid_days
            ) / tot_days

            reasons = []

            if not required_years.issubset(
                years_present
            ):
                reasons.append(
                    f"整年缺失 {missing_years}"
                )

            if wl_miss_ratio > 0.10:
                reasons.append(
                    f"水位缺失超10% "
                    f"(实际为 {wl_miss_ratio * 100:.1f}%)"
                )

            if q_miss_ratio > 0.10:
                reasons.append(
                    f"流量缺失超10% "
                    f"(实际为 {q_miss_ratio * 100:.1f}%)"
                )

            if reasons:
                print(
                    f"  [分类] 站点 {site} "
                    f"分流至【剔除文件夹】，"
                    f"原因: {' | '.join(reasons)}"
                )

                target_summary = summary_excluded
                is_excluded = True

            else:
                target_summary = summary_retained
                is_excluded = False

            # ==================================
            # Summarize missingness in the original series.
            # ==================================
            for indicator in avg_cols:

                valid_dates = set(
                    site_df
                    .dropna(subset=[indicator])
                    ['监测时间']
                )

                missing_dates = [
                    d
                    for d in full_dates
                    if d not in valid_dates
                ]

                year_miss = defaultdict(int)
                month_miss = defaultdict(int)

                for d in missing_dates:
                    year_miss[d.year] += 1

                    month_miss[
                        f"{d.year}年{d.month}月"
                    ] += 1

                target_summary[indicator].append({
                    'HyS_name': site,
                    'River': site_info['河流'],
                    'TotDays': len(full_dates),
                    'MissDays': len(missing_dates),
                    'MissPct': round(
                        len(missing_dates)
                        / len(full_dates)
                        * 100,
                        2
                    ),
                    'YearMiss2021': year_miss[2021],
                    'YearMiss2022': year_miss[2022],
                    'YearMiss2023': year_miss[2023],
                    'YearMiss2024': year_miss[2024],
                    'YearMiss2025': year_miss[2025],
                    'MissDist': dict(
                        sorted(month_miss.items())
                    ),
                    'MissRatio': (
                        len(missing_dates)
                        / len(full_dates)
                    )
                })

            # ==================================
            # Reindex to the complete daily calendar.
            # ==================================
            full_site_df = pd.DataFrame({
                '监测时间': full_dates
            })

            full_site_df = full_site_df.merge(
                site_df,
                on='监测时间',
                how='left'
            )

            # Fill in static station metadata.
            for col in static_cols:
                full_site_df[col] = site_info[col]

            # ==========================================
            # Initial sediment concentration (Ss) processing
            #
            # Missingness rules
            # 1. Keep recorded Ss values.
            # 2. Set missing Ss to zero when both WL and Q are present that day.
            # 3. If Ss and either WL or Q are missing,
            #    leave Ss as NaN.
            #
            # Order of operations
            # Apply this rule before interpolating WL or Q.
            # Use the original daily WL and Q for the decision.
            # ==========================================

            # Ensure that all three hydrological variables are numeric.
            for col in ['水位', '流量', '含沙量']:
                full_site_df[col] = pd.to_numeric(
                    full_site_df[col],
                    errors='coerce'
                )

            # Find the original missing Ss values.
            ss_missing_mask = (
                full_site_df['含沙量'].isna()
            )

            # Identify dates with both original WL and Q.
            wl_q_available_mask = (
                full_site_df['水位'].notna()
                &
                full_site_df['流量'].notna()
            )

            # When Ss is missing and both WL and Q were observed,
            # treat the sediment concentration that day as zero.
            ss_zero_mask = (
                ss_missing_mask
                &
                wl_q_available_mask
            )

            full_site_df.loc[
                ss_zero_mask,
                '含沙量'
            ] = 0.0

            # Summarize the initial missingness treatment.
            original_ss_count = (
                full_site_df['含沙量'].notna().sum()
                - ss_zero_mask.sum()
            )

            ss_zero_count = ss_zero_mask.sum()

            ss_remaining_missing = (
                full_site_df['含沙量'].isna().sum()
            )

            print(
                f"    {site}: Ss 第一步处理 -> "
                f"原始有效值 {original_ss_count} 个；"
                f"WL、Q 均有数据时补 0 共 "
                f"{ss_zero_count} 个；"
                f"其余保持缺失 {ss_remaining_missing} 个"
            )

            # ==================================
            # Rename and arrange columns.
            # ==================================
            full_site_df['监测时间'] = (
                full_site_df['监测时间']
                .dt.strftime('%Y/%m/%d')
            )

            current_time_col = '监测时间'

            final_col_map = {
                **col_map,
                current_time_col: 'Date'
            }

            final_df = full_site_df[
                static_cols
                + [current_time_col]
                + avg_cols
            ].rename(
                columns=final_col_map
            )

            # ==================================
            # Uninterpolated daily series
            #
            # Order of operations
            # The initial Ss rule has already been applied.
            # Missing Ss is zero only on days with both WL and Q.
            # All other missing Ss values remain NaN.
            #
            # WL, Q and Ss are still uninterpolated at this stage.
            # ==================================
            original_final_df = final_df.copy()

            # ==================================
            # Interpolate WL, Q and Ss linearly.
            # ==================================
            interpolated_final_df = final_df.copy()

            for indicator in ['WL', 'Q', 'Ss']:

                if indicator not in (
                    interpolated_final_df.columns
                ):
                    continue

                interpolated_final_df[indicator] = (
                    pd.to_numeric(
                        interpolated_final_df[indicator],
                        errors='coerce'
                    )
                )

                missing_before = (
                    interpolated_final_df[indicator]
                    .isna()
                    .sum()
                )

                if (
                    interpolated_final_df[indicator]
                    .notna()
                    .any()
                ):

                    interpolated_final_df[indicator] = (
                        interpolated_final_df[indicator]
                        .interpolate(
                            method='linear',
                            limit_direction='both'
                        )
                    )

                    missing_after = (
                        interpolated_final_df[indicator]
                        .isna()
                        .sum()
                    )

                    filled_count = (
                        missing_before
                        - missing_after
                    )

                    print(
                        f"    {site}: "
                        f"{indicator} 线性插补 "
                        f"{filled_count} 个缺失日"
                    )

                else:
                    print(
                        f"    {site}: "
                        f"{indicator} 全为空，"
                        f"未进行线性插补"
                    )

            # ==================================
            # Round hydrological variables to three decimal places.
            # ==================================
            for col in ['WL', 'Q', 'Ss']:

                if col in original_final_df.columns:
                    original_final_df[col] = (
                        original_final_df[col]
                        .round(3)
                    )

                if col in interpolated_final_df.columns:
                    interpolated_final_df[col] = (
                        interpolated_final_df[col]
                        .round(3)
                    )

            # ==================================
            # Save uninterpolated and interpolated data in separate directories.
            # ==================================
            safe_name = ''.join(
                c
                for c in str(site)
                if c.isalnum()
                or c in ' _-'
            ).strip()

            subset_folder = (
                'Excluded'
                if is_excluded
                else 'Retained'
            )

            original_site_dir = os.path.join(
                output_folder_original,
                subset_folder
            )

            interpolated_site_dir = os.path.join(
                output_folder_interpolated,
                subset_folder
            )

            os.makedirs(
                original_site_dir,
                exist_ok=True
            )

            os.makedirs(
                interpolated_site_dir,
                exist_ok=True
            )

            original_path = os.path.join(
                original_site_dir,
                f"{safe_name}.csv"
            )

            interpolated_path = os.path.join(
                interpolated_site_dir,
                f"{safe_name}_Interpolated.csv"
            )

            # Save the uninterpolated series.
            original_final_df.to_csv(
                original_path,
                index=False,
                encoding='utf-8-sig'
            )

            # Save the linearly interpolated WL, Q and Ss series.
            interpolated_final_df.to_csv(
                interpolated_path,
                index=False,
                encoding='utf-8-sig'
            )

            print(
                f"  "
                f"[{'剔除' if is_excluded else '保留'}] "
                f"原始序列: {original_path}"
            )

            print(
                f"  "
                f"[{'剔除' if is_excluded else '保留'}] "
                f"插补序列: {interpolated_path}"
            )

    except Exception as e:
        print(f"  发生错误: {e}")


# ======================
# 3. Export missingness summaries
# ======================
def write_summary(summary_list, output_file):

    if not summary_list:
        return

    df = pd.DataFrame(
        summary_list
    ).sort_values(
        'MissRatio',
        ascending=False
    )

    all_months = sorted(
        set().union(
            *(
                d.keys()
                for d in df['MissDist']
            )
        ),
        key=lambda x: (
            int(x.split('年')[0]),
            int(
                x.split('年')[1]
                .replace('月', '')
            )
        )
    )

    for m in all_months:

        year = m.split('年')[0]

        m_idx = (
            m.split('年')[1]
            .replace('月', '')
        )

        col_name = (
            f"{month_abbr[m_idx + '月']}"
            f"_{year}"
        )

        df[col_name] = (
            df['MissDist']
            .apply(
                lambda x: x.get(m, 0)
            )
        )

    base_cols = [
        'HyS_name',
        'River',
        'TotDays',
        'MissDays',
        'MissPct',
        'YearMiss2021',
        'YearMiss2022',
        'YearMiss2023',
        'YearMiss2024',
        'YearMiss2025'
    ]

    month_cols = [
        c
        for c in df.columns
        if c not in (
            base_cols
            + ['MissDist', 'MissRatio']
        )
    ]

    df = df[
        base_cols + month_cols
    ].rename(
        columns={
            'YearMiss2021': '2021',
            'YearMiss2022': '2022',
            'YearMiss2023': '2023',
            'YearMiss2024': '2024',
            'YearMiss2025': '2025'
        }
    )

    df.to_csv(
        output_file,
        index=False,
        encoding='utf-8-sig'
    )

    print(
        f"已保存: "
        f"{os.path.basename(output_file)}"
    )


# ======================
# 4. Main entry point
# ======================
if __name__ == "__main__":

    for p in [
        output_folder_original,
        output_folder_interpolated,
        summary_output_folder
    ]:
        os.makedirs(
            p,
            exist_ok=True
        )

    summary_retained = {
        col: []
        for col in avg_cols
    }

    summary_excluded = {
        col: []
        for col in avg_cols
    }

    print(
        f"开始处理: "
        f"{input_folder}"
    )

    print(
        f"原始连续序列目录: "
        f"{output_folder_original}"
    )

    print(
        f"线性插补序列目录: "
        f"{output_folder_interpolated}"
    )

    print(
        f"汇总表将保存至: "
        f"{summary_output_folder}"
    )

    for f in os.listdir(
        input_folder
    ):

        if f.endswith(
            (
                '.csv',
                '.xlsx',
                '.xls'
            )
        ):
            process_single_file(
                os.path.join(
                    input_folder,
                    f
                ),
                summary_retained,
                summary_excluded
            )

    print(
        "\n"
        + "=" * 40
    )

    print(
        "====== 生成 [保留站点] "
        "缺失汇总 ======"
    )

    for indicator in avg_cols:

        if summary_retained[indicator]:

            write_summary(
                summary_retained[indicator],
                os.path.join(
                    summary_output_folder,
                    f"1."
                    f"{col_map.get(indicator, indicator)}"
                    f"_Retained_Summary.csv"
                )
            )

    print(
        "\n"
        + "=" * 40
    )

    print(
        "====== 生成 [剔除站点] "
        "缺失汇总 ======"
    )

    has_excluded = False

    for indicator in avg_cols:

        if summary_excluded[indicator]:

            has_excluded = True

            write_summary(
                summary_excluded[indicator],
                os.path.join(
                    summary_output_folder,
                    f"2."
                    f"{col_map.get(indicator, indicator)}"
                    f"_Excluded_Summary.csv"
                )
            )

    if not has_excluded:
        print(
            "没有站点因缺失条件超标而被剔除。"
        )

    print(
        "\n水文站数据处理完成！"
    )