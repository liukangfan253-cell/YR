# -*- coding: utf-8 -*-

import ee
import geemap
import pandas as pd
import os
import warnings
import time

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

warnings.filterwarnings('ignore')


# ==========================================
# 1. Initialize Earth Engine and set global options
# ==========================================

GEE_PROJECT = 'yellow-river-kw-483208'

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)


# ==========================================
# 2. Basic configuration
# ==========================================

# Reference years are 1991–2020.
TARGET_YEARS = list(range(1991, 2021))


POINT_FILE = str(ROOT / 'Config/Stations_B.csv')


# ----------------------------------------------------------
# Use a new output directory.
#
# Reason
# A previous progress log may record results aggregated by UTC day.
# Using that directory again may cause the script to skip those stations.
# ----------------------------------------------------------

OUTPUT_FOLDER = str(ROOT / 'Data/Meteorology_data/1991-2020/B')


# ----------------------------------------------------------
# ERA5-Land has a native resolution of approximately 0.1 degrees (about 11 km).
#
# Sampling at the former 1,000 m setting does not produce 1 km meteorological data.
# Use a scale close to the native resolution.
# ----------------------------------------------------------

SCALE = 11132


# ----------------------------------------------------------
# Beijing time (UTC+8)
# ----------------------------------------------------------

UTC_OFFSET = 8


# ==========================================
# 3. Load the water-quality stations
# ==========================================

print("正在加载黄河流域水质断面点位...")


try:

    df_sites = pd.read_csv(
        POINT_FILE,
        encoding='utf-8'
    )

    print(
        f"✅ 成功导入 "
        f"{len(df_sites)} 个水质断面点位"
    )


except Exception:

    try:

        df_sites = pd.read_csv(
            POINT_FILE,
            encoding='gbk'
        )

        print(
            f"✅ 成功导入 "
            f"{len(df_sites)} 个水质断面点位 "
            f"(gbk编码)"
        )


    except Exception as e:

        print(
            f"❌ 无法读取点位文件: "
            f"{str(e)}"
        )

        raise


# ==========================================
# 4. Check required fields
# ==========================================

required_site_cols = [
    'WQS_name',
    'WQS_Lon',
    'WQS_Lat'
]


missing_cols = [
    col
    for col in required_site_cols
    if col not in df_sites.columns
]


if missing_cols:

    raise ValueError(
        f"点位文件缺少字段：{missing_cols}"
    )


# Drop invalid station locations.
df_sites = (
    df_sites
    .dropna(
        subset=required_site_cols
    )
    .drop_duplicates(
        subset=['WQS_name'],
        keep='first'
    )
    .reset_index(drop=True)
)


print(
    f"📍 有效断面数：{len(df_sites)}"
)


print("📋 点位预览（前5行）：")

print(
    df_sites[
        [
            'WQS_name',
            'WQS_Lon',
            'WQS_Lat'
        ]
    ].head()
)


# ==========================================
# 5. Identify leap years
# ==========================================

def is_leap_year(year):

    return (
        year % 4 == 0
        and
        (
            year % 100 != 0
            or year % 400 == 0
        )
    )


# ==========================================
# 6. Build the calendar in Beijing time
# ==========================================

def get_year_date_range(year):
    """
    Build date labels for a complete calendar year in Beijing time.

    ee.Date represents UTC timestamps. Use the returned dates only as local calendar labels.
    Shift each day's extraction interval eight hours earlier in UTC when querying ERA5.
    """

    total_days = (
        366
        if is_leap_year(year)
        else 365
    )


    print(
        f"  📅 {year}年："
        f"共 {total_days} 天 "
        f"({'闰年' if total_days == 366 else '平年'})"
    )


    # ------------------------------------------------------
    # Use this value only as a local calendar date label.
    # ------------------------------------------------------

    local_year_start = ee.Date.fromYMD(
        year,
        1,
        1
    )


    local_next_year_start = ee.Date.fromYMD(
        year + 1,
        1,
        1
    )


    # ------------------------------------------------------
    # Beijing time
    #
    # January 1 at 00:00
    #
    # Equivalent UTC time
    #
    # December 31 of the previous year at 16:00
    # ------------------------------------------------------

    utc_year_start = (
        local_year_start
        .advance(
            -UTC_OFFSET,
            'hour'
        )
    )


    utc_year_end = (
        local_next_year_start
        .advance(
            -UTC_OFFSET,
            'hour'
        )
    )


    # ------------------------------------------------------
    # Build local date labels in Beijing time.
    #
    # January 1
    # January 2
    # ...
    # December 31
    # ------------------------------------------------------

    day_list = (
        ee.List
        .sequence(
            0,
            total_days - 1
        )
        .map(
            lambda d:
            local_year_start.advance(
                d,
                'day'
            )
        )
    )


    return (
        local_year_start,
        local_next_year_start,
        utc_year_start,
        utc_year_end,
        day_list,
        total_days
    )


# ==========================================
# 7. Process daily air temperature for one station
# ==========================================

def process_site_daily_temperature(
        site_name,
        lon,
        lat
):
    """
    Extract daily T, Tmax and Tmin for one station.

    Each day spans 00:00–24:00 in Beijing time (UTC+8).
    """

    print(
        f"\n📍 处理站点："
        f"{site_name} ({lon}, {lat})"
    )


    point = ee.Geometry.Point([
        float(lon),
        float(lat)
    ])


    all_daily_data = []


    # ======================================
    # Process one year at a time.
    # ======================================

    for year in TARGET_YEARS:

        try:

            print(
                f"\n  📅 处理 {year} 年..."
            )


            (
                local_year_start,
                local_next_year_start,
                utc_year_start,
                utc_year_end,
                day_list,
                total_days
            ) = get_year_date_range(
                year
            )


            # ==============================================
            # Hourly ERA5-Land air temperature
            #
            # Shift the annual UTC window back eight hours to match Beijing time.
            #
            # Example for 2021
            #
            # Beijing time
            # 2021-01-01 00:00
            # →
            # 2022-01-01 00:00
            #
            # UTC：
            # 2020-12-31 16:00
            # →
            # 2021-12-31 16:00
            # ==============================================

            era5_hourly = (
                ee.ImageCollection(
                    "ECMWF/ERA5_LAND/HOURLY"
                )
                .select(
                    'temperature_2m'
                )
                .filterDate(
                    utc_year_start,
                    utc_year_end
                )
                .filterBounds(
                    point
                )
            )


            # ==============================================
            # Each local calendar day in Beijing time
            # ==============================================

            def process_day(
                    local_day
            ):

                # ------------------------------------------
                # local_day is used only as a Beijing-time date label.
                # ------------------------------------------

                local_day = ee.Date(
                    local_day
                )


                # ==========================================
                # 00:00 on that day in Beijing time
                # →
                # 16:00 UTC on the previous day
                # ==========================================

                day_start_utc = (
                    local_day
                    .advance(
                        -UTC_OFFSET,
                        'hour'
                    )
                )


                # ==========================================
                # 00:00 the following day in Beijing time
                # ==========================================

                day_end_utc = (
                    day_start_utc
                    .advance(
                        1,
                        'day'
                    )
                )


                # ==========================================
                # Select the 24 ERA5 hours corresponding to this local day.
                # ==========================================

                daily_data = (
                    era5_hourly
                    .filterDate(
                        day_start_utc,
                        day_end_utc
                    )
                )


                # ==========================================
                # Daily mean, maximum and minimum
                # ==========================================

                reducer = (
                    ee.Reducer
                    .mean()
                    .combine(
                        reducer2=
                        ee.Reducer.max(),
                        sharedInputs=True
                    )
                    .combine(
                        reducer2=
                        ee.Reducer.min(),
                        sharedInputs=True
                    )
                )


                daily_stats = (
                    daily_data
                    .reduce(
                        reducer
                    )
                )


                # ==========================================
                # Kelvin → ℃
                # ==========================================

                daily_stats_celsius = (
                    daily_stats
                    .subtract(
                        273.15
                    )
                    .rename(
                        [
                            'T',
                            'Tmax',
                            'Tmin'
                        ]
                    )
                )


                # ==========================================
                # Extract values only for the current station.
                # ==========================================

                value = (
                    daily_stats_celsius
                    .reduceRegion(
                        reducer=
                        ee.Reducer.first(),

                        geometry=
                        point,

                        scale=
                        SCALE,

                        maxPixels=
                        1e6
                    )
                )


                # ==========================================
                # Keep the Beijing-time calendar date as the label.
                # ==========================================

                date_label = (
                    local_day
                    .format(
                        'YYYY-MM-dd'
                    )
                )


                return ee.Feature(
                    None,
                    {
                        'WQS_name':
                            site_name,

                        'WQS_Lon':
                            lon,

                        'WQS_Lat':
                            lat,

                        'date':
                            date_label,

                        'T':
                            value.get(
                                'T'
                            ),

                        'Tmax':
                            value.get(
                                'Tmax'
                            ),

                        'Tmin':
                            value.get(
                                'Tmin'
                            )
                    }
                )


            # ==============================================
            # All calendar days in one year
            # ==============================================

            daily_features = (
                day_list
                .map(
                    process_day
                )
            )


            daily_fc = (
                ee.FeatureCollection(
                    daily_features
                )
            )


            # ==============================================
            # Convert to a local DataFrame.
            # ==============================================

            try:

                df_year = (
                    geemap.ee_to_df(
                        daily_fc
                    )
                )


                if df_year.empty:

                    print(
                        f"    ⚠️ "
                        f"{year} 年无数据"
                    )

                    continue


                # ==========================================
                # Convert to numeric values
                # ==========================================

                for col in [
                    'T',
                    'Tmax',
                    'Tmin'
                ]:

                    df_year[col] = (
                        pd.to_numeric(
                            df_year[col],
                            errors='coerce'
                        )
                    )


                # ==========================================
                # Avoid dropping records silently.
                # Check for missing values first.
                # ==========================================

                missing_count = (
                    df_year[
                        [
                            'T',
                            'Tmax',
                            'Tmin'
                        ]
                    ]
                    .isna()
                    .any(
                        axis=1
                    )
                    .sum()
                )


                if missing_count > 0:

                    print(
                        f"    ⚠️ "
                        f"{year} 年存在 "
                        f"{missing_count} 天"
                        f"气温缺失"
                    )


                # Drop records that could not be retrieved.
                df_year = (
                    df_year
                    .dropna(
                        subset=[
                            'T',
                            'Tmax',
                            'Tmin'
                        ]
                    )
                    .copy()
                )


                # ==========================================
                # Round to three decimal places.
                # ==========================================

                df_year[
                    'T'
                ] = (
                    df_year[
                        'T'
                    ]
                    .round(3)
                )


                df_year[
                    'Tmax'
                ] = (
                    df_year[
                        'Tmax'
                    ]
                    .round(3)
                )


                df_year[
                    'Tmin'
                ] = (
                    df_year[
                        'Tmin'
                    ]
                    .round(3)
                )


                # ==========================================
                # Date
                # ==========================================

                df_year[
                    'date'
                ] = pd.to_datetime(
                    df_year[
                        'date'
                    ],
                    errors='coerce'
                )


                df_year = (
                    df_year
                    .dropna(
                        subset=[
                            'date'
                        ]
                    )
                    .sort_values(
                        'date'
                    )
                    .reset_index(
                        drop=True
                    )
                )


                # ==========================================
                # Check completeness within the year.
                # ==========================================

                actual_days = len(
                    df_year
                )


                expected_days = (
                    total_days
                )


                if (
                    actual_days
                    == expected_days
                ):

                    print(
                        f"    ✅ "
                        f"{year} 年完整："
                        f"{actual_days} 天"
                    )

                else:

                    print(
                        f"    ⚠️ "
                        f"{year} 年理论 "
                        f"{expected_days} 天，"
                        f"实际 "
                        f"{actual_days} 天"
                    )


                # ==========================================
                # Check December 31.
                # ==========================================

                expected_dec31 = (
                    pd.Timestamp(
                        f'{year}-12-31'
                    )
                )


                dec31_exists = (
                    expected_dec31
                    in
                    df_year[
                        'date'
                    ].values
                )


                print(
                    f"       12月31日："
                    f"{'✅ 存在' if dec31_exists else '❌ 缺失'}"
                )


                # ==========================================
                # Convert dates back to strings
                # ==========================================

                df_year[
                    'date'
                ] = (
                    df_year[
                        'date'
                    ]
                    .dt
                    .strftime(
                        '%Y-%m-%d'
                    )
                )


                all_daily_data.append(
                    df_year
                )


            except Exception as e:

                print(
                    f"    ❌ 提取 "
                    f"{year} 年数据失败："
                    f"{str(e)}"
                )

                continue


            # Pause briefly to space out requests.
            time.sleep(2)


        except Exception as e:

            print(
                f"    ❌ "
                f"{year} 年处理失败："
                f"{str(e)}"
            )

            time.sleep(5)

            continue


    # ======================================
    # Combine all years.
    # ======================================

    if not all_daily_data:

        print(
            f"  ⚠️ "
            f"{site_name} "
            f"没有获得任何数据"
        )

        return pd.DataFrame()


    df_site = pd.concat(
        all_daily_data,
        ignore_index=True
    )


    df_site[
        'date'
    ] = pd.to_datetime(
        df_site[
            'date'
        ]
    )


    df_site = (
        df_site
        .sort_values(
            'date'
        )
        .drop_duplicates(
            subset=[
                'date'
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ======================================
    # Calculate the expected total number of days.
    # ======================================

    total_days_expected = sum(
        366
        if is_leap_year(year)
        else 365
        for year in TARGET_YEARS
    )


    actual_days = len(
        df_site
    )


    # ======================================
    # Check that dates are continuous.
    # ======================================

    expected_dates = pd.date_range(
        start=
        f'{TARGET_YEARS[0]}-01-01',

        end=
        f'{TARGET_YEARS[-1]}-12-31',

        freq='D'
    )


    actual_dates = pd.DatetimeIndex(
        df_site[
            'date'
        ]
    )


    missing_dates = (
        expected_dates
        .difference(
            actual_dates
        )
    )


    print(
        f"\n  ✅ "
        f"{site_name} 处理完成"
    )


    print(
        f"     获得 "
        f"{actual_days} 天数据 "
        f"(期望："
        f"{total_days_expected} 天)"
    )


    print(
        f"     时间范围："
        f"{df_site['date'].min().strftime('%Y-%m-%d')} "
        f"到 "
        f"{df_site['date'].max().strftime('%Y-%m-%d')}"
    )


    if len(
        missing_dates
    ) == 0:

        print(
            "     ✅ 日期连续完整"
        )

    else:

        print(
            f"     ⚠️ "
            f"缺失日期数量："
            f"{len(missing_dates)}"
        )

        print(
            "     前10个缺失日期："
            + ', '.join(
                missing_dates
                .strftime(
                    '%Y-%m-%d'
                )
                .tolist()[:10]
            )
        )


    print(
        f"     日平均气温范围："
        f"{df_site['T'].min():.2f}℃ "
        f"~ "
        f"{df_site['T'].max():.2f}℃"
    )


    # ======================================
    # Convert dates back to strings
    # ======================================

    df_site[
        'date'
    ] = (
        df_site[
            'date'
        ]
        .dt
        .strftime(
            '%Y-%m-%d'
        )
    )


    return df_site


# ==========================================
# 8. Main processing function
# ==========================================

def main():

    print(
        "=" * 65
    )


    print(
        "黄河流域水质断面逐日气温数据下载"
    )


    print(
        f"时间范围："
        f"{TARGET_YEARS[0]}-"
        f"{TARGET_YEARS[-1]}年"
    )


    print(
        "时间标准："
        "北京时间 UTC+8，"
        "每日00:00-24:00"
    )


    print(
        "提取变量："
        "T（日平均）、"
        "Tmax（日最高）、"
        "Tmin（日最低）"
    )


    print(
        f"ERA5-Land采样尺度："
        f"{SCALE} m"
    )


    print(
        "=" * 65
    )


    # ======================================
    # Output folder
    # ======================================

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )


    print(
        f"📂 输出文件夹："
        f"{OUTPUT_FOLDER}"
    )


    # ======================================
    # Progress log
    # ======================================

    progress_file = os.path.join(
        OUTPUT_FOLDER,
        "处理进度.txt"
    )


    processed_sites = set()


    if os.path.exists(
        progress_file
    ):

        with open(
            progress_file,
            'r',
            encoding='utf-8'
        ) as f:

            processed_sites = set(
                line.strip()
                for line in f
                if line.strip()
            )


        print(
            f"📊 已处理站点："
            f"{len(processed_sites)} 个"
        )


    successful_sites = []

    failed_sites = []


    print(
        f"\n🏁 开始处理 "
        f"{len(df_sites)} 个水质断面..."
    )


    # ======================================
    # Process one station at a time.
    # ======================================

    for idx, row in (
        df_sites.iterrows()
    ):

        site_name = str(
            row[
                'WQS_name'
            ]
        )


        lon = float(
            row[
                'WQS_Lon'
            ]
        )


        lat = float(
            row[
                'WQS_Lat'
            ]
        )


        if site_name in processed_sites:

            print(
                f"\n"
                f"[{idx + 1}/{len(df_sites)}] "
                f"{site_name} "
                f"已处理，跳过"
            )

            continue


        print(
            f"\n"
            f"[{idx + 1}/{len(df_sites)}] "
            + "=" * 40
        )


        try:

            df_site = (
                process_site_daily_temperature(
                    site_name,
                    lon,
                    lat
                )
            )


            if df_site.empty:

                failed_sites.append(
                    site_name
                )

                print(
                    f"⚠️ "
                    f"{site_name} "
                    f"无数据，跳过保存"
                )

                continue


            # ==================================
            # Make the filename safe.
            # ==================================

            safe_site_name = (
                str(site_name)
                .replace('/', '_')
                .replace('\\', '_')
                .replace(':', '_')
                .replace('*', '_')
                .replace('?', '_')
                .replace('"', '_')
                .replace('<', '_')
                .replace('>', '_')
                .replace('|', '_')
            )


            output_path = os.path.join(
                OUTPUT_FOLDER,
                f"{safe_site_name}.csv"
            )


            # ==================================
            # Output fields
            # ==================================

            required_columns = [
                'WQS_name',
                'WQS_Lon',
                'WQS_Lat',
                'date',
                'T',
                'Tmax',
                'Tmin'
            ]


            df_site = df_site[
                required_columns
            ]


            # ==================================
            # Save output.
            # ==================================

            df_site.to_csv(
                output_path,
                index=False,
                encoding='utf-8-sig',
                float_format='%.3f'
            )


            successful_sites.append(
                site_name
            )


            # ==================================
            # Record progress only after saving successfully.
            # ==================================

            with open(
                progress_file,
                'a',
                encoding='utf-8'
            ) as f:

                f.write(
                    f"{site_name}\n"
                )


            print(
                f"✅ "
                f"{site_name} 数据已保存："
                f"{output_path}"
            )


            print(
                f"   文件大小："
                f"{os.path.getsize(output_path) / 1024:.1f} KB"
            )


            # The previous 30-second pause is unnecessary.
            # Five seconds is sufficient for a single temperature variable.
            wait_time = 5

            print(
                f"⏸️ 站点间暂停 "
                f"{wait_time} 秒..."
            )

            time.sleep(
                wait_time
            )


        except Exception as e:

            print(
                f"❌ "
                f"{site_name} "
                f"处理失败："
                f"{str(e)}"
            )


            failed_sites.append(
                site_name
            )


            time.sleep(
                10
            )


    # ==========================================
    # 9. Completion summary
    # ==========================================

    print(
        "\n"
        + "=" * 65
    )


    print(
        "数据处理完成"
    )


    print(
        "=" * 65
    )


    print(
        f"✅ 本次成功处理站点："
        f"{len(successful_sites)} 个"
    )


    print(
        f"❌ 本次失败站点："
        f"{len(failed_sites)} 个"
    )


    # ======================================
    # Save the list of stations that failed.
    # ======================================

    if failed_sites:

        failed_file = os.path.join(
            OUTPUT_FOLDER,
            "失败站点列表.txt"
        )


        with open(
            failed_file,
            'w',
            encoding='utf-8'
        ) as f:

            for site in failed_sites:

                f.write(
                    f"{site}\n"
                )


        print(
            f"📝 失败站点列表："
            f"{failed_file}"
        )


    # ======================================
    # Expected number of dates over the full period
    # ======================================

    leap_years = [
        year
        for year in TARGET_YEARS
        if is_leap_year(year)
    ]


    total_days_expected = sum(
        366
        if is_leap_year(year)
        else 365
        for year in TARGET_YEARS
    )


    print(
        "\n📊 数据完整性统计："
    )


    print(
        f"   处理年份："
        f"{TARGET_YEARS[0]}-"
        f"{TARGET_YEARS[-1]}"
    )


    print(
        f"   闰年数量："
        f"{len(leap_years)} 个 "
        f"({leap_years})"
    )


    print(
        f"   预期总天数："
        f"{total_days_expected} 天"
    )


    print(
        "\n🎉 "
        "黄河流域水质断面逐日气温数据下载完成！"
    )


    print(
        f"📁 结果保存路径："
        f"{os.path.abspath(OUTPUT_FOLDER)}"
    )


    print(
        "\n📋 输出变量："
    )


    print(
        "   WQS_name：站点名称"
    )

    print(
        "   WQS_Lon：经度"
    )

    print(
        "   WQS_Lat：纬度"
    )

    print(
        "   date：北京时间日期 YYYY-MM-DD"
    )

    print(
        "   T：北京时间日平均气温 (℃)"
    )

    print(
        "   Tmax：北京时间日最高气温 (℃)"
    )

    print(
        "   Tmin：北京时间日最低气温 (℃)"
    )


    print(
        "=" * 65
    )


# ==========================================
# 10. Run the script
# ==========================================

if __name__ == "__main__":

    main()