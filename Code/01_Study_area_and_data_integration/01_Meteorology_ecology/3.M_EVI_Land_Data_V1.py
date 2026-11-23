# -*- coding: utf-8 -*-
"""Aggregate ERA5 snapshots and hourly increments by Beijing-time calendar day.
Retain the EVI, Dynamic World and station-list workflow and write the station-group output.
Running this script starts a new extraction. Script validation did not include a full extraction.
ERA5-Land timestamp conventions are documented at https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation
"""

import ee
import pandas as pd
import numpy as np
import os
import time
import warnings

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

warnings.filterwarnings('ignore')


# =========================================================
# 1. Initialize Earth Engine
# =========================================================

GEE_PROJECT = 'yellow-river-kw-483208'

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)


# =========================================================
# 2. User configuration
# =========================================================

TARGET_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]


POINT_FILE = str(ROOT / 'Config/Stations_M.csv')


# ---------------------------------------------------------
# Output directory
#
# Start with a new directory
# ---------------------------------------------------------

OUTPUT_FOLDER = str(ROOT / 'Data/Meteorology_data/2021-2025/M_V1')


# ---------------------------------------------------------
# ERA5 extraction scale
#
# Reuse the settings verified in the previous run
# ---------------------------------------------------------

SCALE_METEO = 1000


# ---------------------------------------------------------
# MODIS and Dynamic World extraction scales
# Use the original script settings
# ---------------------------------------------------------

SCALE_ECO = 250


# ---------------------------------------------------------
# China Standard Time
# ---------------------------------------------------------

LOCAL_TIMEZONE = 'Asia/Shanghai'


# ---------------------------------------------------------
# Automatically retry failed Earth Engine requests
# ---------------------------------------------------------

MAX_RETRIES = 3

RETRY_WAIT_SECONDS = 5


# ---------------------------------------------------------
# Whether to skip stations with complete output
#
# The previous Bahekou file had empty EVI values.
# Set this to False to regenerate it.
#
# After validating this run,
# set this to True to resume an interrupted run.
# ---------------------------------------------------------

SKIP_VALID_EXISTING = False


# =========================================================
# 3. Read water-quality station metadata
# =========================================================

def load_site_points(point_file):
    """Read the station list. If the summary is missing, recover coordinates from retained water-quality files in the same group. Leave source files intact."""
    columns = ['WQS_name', 'WQS_Lon', 'WQS_Lat']
    if os.path.isfile(point_file):
        points = pd.read_csv(point_file, encoding='utf-8-sig')
        origin = point_file
    else:
        retained = os.path.join(os.path.dirname(point_file), 'Retained', 'Cleaned')
        if not os.path.isdir(retained):
            raise FileNotFoundError(
                f'站点表不存在：{point_file}；可恢复坐标的目录也不存在：{retained}'
            )
        blocks = []
        for name in sorted(os.listdir(retained)):
            if not name.lower().endswith('.csv') or name.startswith('~$'):
                continue
            source = os.path.join(retained, name)
            block = pd.read_csv(source, usecols=columns, encoding='utf-8-sig')
            blocks.append(block.drop_duplicates())
        if not blocks:
            raise RuntimeError(f'保留水质目录中没有可读取的站点文件：{retained}')
        points = pd.concat(blocks, ignore_index=True).drop_duplicates()
        origin = retained
        print(f'ℹ️ 坐标汇总表缺失，已从保留水质文件读取站点坐标：{retained}')
    missing = [col for col in columns if col not in points.columns]
    if missing:
        raise ValueError(f'站点数据缺少字段：{missing}；来源：{origin}')
    points = points[columns].copy()
    points['WQS_name'] = points['WQS_name'].astype('string').str.strip()
    for col in ['WQS_Lon', 'WQS_Lat']:
        points[col] = pd.to_numeric(points[col], errors='coerce')
    points = points.dropna(subset=columns)
    points = points.loc[points['WQS_name'].ne('')].drop_duplicates()
    if points.empty:
        raise RuntimeError(f'没有有效的站名和坐标：{origin}')
    valid = (np.isfinite(points[['WQS_Lon', 'WQS_Lat']].to_numpy(dtype=float)).all(axis=1)
             & points['WQS_Lon'].between(-180, 180).to_numpy()
             & points['WQS_Lat'].between(-90, 90).to_numpy())
    if not valid.all():
        raise ValueError('站点坐标超出合法范围或存在非有限值。')
    if points['WQS_name'].duplicated().any():
        raise ValueError('同一站名出现不同坐标，需核对后再提取。')
    return points.reset_index(drop=True)


df_sites = load_site_points(POINT_FILE)


required_cols = [
    'WQS_name',
    'WQS_Lon',
    'WQS_Lat'
]


missing_cols = [
    col
    for col in required_cols
    if col not in df_sites.columns
]


if missing_cols:
    raise ValueError(
        f"点位文件缺少必要字段：{missing_cols}"
    )


df_sites = (
    df_sites
    .dropna(
        subset=[
            'WQS_name',
            'WQS_Lon',
            'WQS_Lat'
        ]
    )
    .copy()
)


df_sites = (
    df_sites
    .drop_duplicates(
        subset=['WQS_name'],
        keep='first'
    )
    .reset_index(drop=True)
)


print(
    f"✅ 成功导入 {len(df_sites)} 个水质断面点位"
)


# =========================================================
# 4. Study-period dates
# =========================================================

GLOBAL_START = pd.Timestamp(
    f"{TARGET_YEARS[0]}-01-01"
)

GLOBAL_END = pd.Timestamp(
    f"{TARGET_YEARS[-1]}-12-31"
)


FULL_DATE_INDEX = pd.date_range(
    GLOBAL_START,
    GLOBAL_END,
    freq='D'
)


EXPECTED_DAYS = len(
    FULL_DATE_INDEX
)


# =========================================================
# 5. Shared utilities
# =========================================================

def safe_filename(name):

    name = str(name)

    invalid_chars = [
        '/',
        '\\',
        ':',
        '*',
        '?',
        '"',
        '<',
        '>',
        '|'
    ]

    for ch in invalid_chars:

        name = name.replace(
            ch,
            '_'
        )

    return name


def create_site_point(site_row):

    lon = float(
        site_row['WQS_Lon']
    )

    lat = float(
        site_row['WQS_Lat']
    )

    return ee.Geometry.Point([
        lon,
        lat
    ])


def existing_file_is_complete(
        output_path
):

    if not os.path.exists(
        output_path
    ):
        return False


    try:

        df = pd.read_csv(
            output_path
        )


        # -------------------------------------------------
        # Check row count
        # -------------------------------------------------

        if len(df) != EXPECTED_DAYS:

            return False


        # -------------------------------------------------
        # Check dates
        # -------------------------------------------------

        if 'Date' not in df.columns:

            return False


        dates = pd.to_datetime(
            df['Date'],
            errors='coerce'
        )


        if dates.isna().any():

            return False


        if (
            dates.min()
            != GLOBAL_START
        ):

            return False


        if (
            dates.max()
            != GLOBAL_END
        ):

            return False


        # -------------------------------------------------
        # Check EVI coverage
        #
        # Guard against cases where
        # all 1,826 EVI values are missing
        # but the output is classified as complete.
        # -------------------------------------------------

        if 'EVI' not in df.columns:

            return False


        if (
            df['EVI']
            .isna()
            .any()
        ):

            return False


        return True


    except Exception:

        return False


# =========================================================
# 6. ERA5-Land
#
#    GEE：
# Extract hourly point values only
#
#    Python：
# Convert UTC to Beijing time
# Aggregate hourly values to daily values
# =========================================================

ERA5_BANDS = [

    'temperature_2m',

    'surface_pressure',

    'dewpoint_temperature_2m',

    'u_component_of_wind_10m',

    'v_component_of_wind_10m',

    'total_precipitation_hourly',

    'surface_net_solar_radiation_hourly'
]


# =========================================================
# 6.1 ERA5 for one station and one month
# =========================================================

def aggregate_era5_hourly_natural_days(df_hourly, local_start, local_end):
    """Aggregate snapshots and hour-ending increments separately by Beijing-time day.

    local_start and local_end are local midnights.
    Use timestamps in [start, end) for snapshots and end timestamps in (start, end] for increments.
    Use the hourly increments supplied by Earth Engine directly.
    """
    local_start = pd.Timestamp(local_start)
    local_end = pd.Timestamp(local_end)
    local_start = (local_start.tz_localize(LOCAL_TIMEZONE) if local_start.tzinfo is None
                   else local_start.tz_convert(LOCAL_TIMEZONE))
    local_end = (local_end.tz_localize(LOCAL_TIMEZONE) if local_end.tzinfo is None
                 else local_end.tz_convert(LOCAL_TIMEZONE))
    if (local_start != local_start.normalize() or local_end != local_end.normalize()
            or local_end <= local_start):
        raise ValueError('ERA5汇总边界必须是递增的北京时间00:00。')
    missing_cols = [col for col in ['time'] + ERA5_BANDS if col not in df_hourly]
    if missing_cols:
        raise RuntimeError(f'ERA5返回数据缺少字段：{missing_cols}')
    hourly = df_hourly.copy()
    hourly['Datetime_UTC'] = pd.to_datetime(
        hourly['time'], unit='ms', utc=True, errors='coerce'
    )
    if hourly['Datetime_UTC'].isna().any():
        raise RuntimeError('ERA5存在无法解析的UTC小时戳。')
    hourly['Datetime_BJ'] = hourly['Datetime_UTC'].dt.tz_convert(LOCAL_TIMEZONE)
    hourly = hourly.loc[
        hourly['Datetime_BJ'].ge(local_start)
        & hourly['Datetime_BJ'].le(local_end)
    ].sort_values('Datetime_UTC').reset_index(drop=True)
    if hourly['Datetime_UTC'].duplicated().any():
        raise RuntimeError('ERA5存在重复小时戳；请检查站点是否返回多个像元或重复影像。')
    for col in ERA5_BANDS:
        hourly[col] = pd.to_numeric(hourly[col], errors='coerce')

    instant = hourly.loc[hourly['Datetime_BJ'].lt(local_end)].copy()
    increment = hourly.loc[hourly['Datetime_BJ'].gt(local_start)].copy()
    instant_bands = [
        'temperature_2m', 'surface_pressure', 'dewpoint_temperature_2m',
        'u_component_of_wind_10m', 'v_component_of_wind_10m'
    ]
    increment_bands = [
        'total_precipitation_hourly', 'surface_net_solar_radiation_hourly'
    ]
    expected_instant = pd.date_range(
        local_start, local_end, freq='h', inclusive='left'
    )
    expected_increment = pd.date_range(
        local_start + pd.Timedelta(hours=1), local_end, freq='h'
    )
    for label, block, expected, bands in [
        ('瞬时量', instant, expected_instant, instant_bands),
        ('小时增量', increment, expected_increment, increment_bands),
    ]:
        actual = pd.DatetimeIndex(block['Datetime_BJ'])
        if not actual.equals(expected):
            absent = expected.difference(actual)
            unexpected = actual.difference(expected)
            raise RuntimeError(
                f'ERA5{label}小时网格不完整：期望{len(expected)}条，实际{len(actual)}条；'
                f'缺失示例{[str(t) for t in absent[:5]]}；'
                f'非预期示例{[str(t) for t in unexpected[:5]]}'
            )
        if not np.isfinite(block[bands].to_numpy(dtype=float)).all():
            raise RuntimeError(f'ERA5{label}的使用区间内存在缺失或非有限值。')

    instant['Date'] = instant['Datetime_BJ'].dt.strftime('%Y-%m-%d')
    # Each timestamp marks the end of the hour. The increment at 00:00 belongs to 23:00–24:00 of the previous local day.
    increment['Date'] = (
        increment['Datetime_BJ'] - pd.Timedelta(hours=1)
    ).dt.strftime('%Y-%m-%d')
    instant['T_hourly_C'] = instant['temperature_2m'] - 273.15
    instant['Td_hourly_C'] = instant['dewpoint_temperature_2m'] - 273.15
    instant['SP_hourly_hPa'] = instant['surface_pressure'] / 100.0
    instant['WS_hourly'] = np.sqrt(
        instant['u_component_of_wind_10m'] ** 2
        + instant['v_component_of_wind_10m'] ** 2
    )
    increment['TP_hourly_mm'] = increment['total_precipitation_hourly'] * 1000.0
    instant_daily = instant.groupby('Date', as_index=False).agg(
        T=('T_hourly_C', 'mean'),
        Tmax=('T_hourly_C', 'max'),
        Tmin=('T_hourly_C', 'min'),
        SP=('SP_hourly_hPa', 'mean'),
        WS=('WS_hourly', 'mean'),
        Td=('Td_hourly_C', 'mean'),
    )
    increment_daily = increment.groupby('Date', as_index=False).agg(
        SSR=('surface_net_solar_radiation_hourly', 'sum'),
        TP=('TP_hourly_mm', 'sum'),
    )
    daily = instant_daily.merge(
        increment_daily, on='Date', how='outer', validate='one_to_one', indicator=True
    ).sort_values('Date').reset_index(drop=True)
    expected_dates = pd.date_range(
        local_start, local_end, freq='D', inclusive='left'
    ).strftime('%Y-%m-%d').tolist()
    if daily['Date'].tolist() != expected_dates or not daily['_merge'].eq('both').all():
        raise RuntimeError('ERA5自然日汇总日期缺失或瞬时量/增量日期不一致。')
    meteo_cols = ['T', 'Tmax', 'Tmin', 'SP', 'SSR', 'TP', 'WS', 'Td']
    daily[meteo_cols] = daily[meteo_cols].round(5)
    return daily[['Date'] + meteo_cols]


def extract_era5_site_month(site_row, year, month):
    """Extract a complete Beijing-time month, including the increment ending at midnight on the first day of the following month."""
    site_name = str(site_row['WQS_name'])
    site_lon = float(site_row['WQS_Lon'])
    site_lat = float(site_row['WQS_Lat'])
    site_point = create_site_point(site_row)
    local_start = pd.Timestamp(year=year, month=month, day=1, tz=LOCAL_TIMEZONE)
    local_end = local_start + pd.DateOffset(months=1)
    utc_start = local_start.tz_convert('UTC')
    # The filterDate end is exclusive. Extend it by one hour to include the end timestamp at 00:00 local time on the first day of the next month.
    utc_end_exclusive = local_end.tz_convert('UTC') + pd.Timedelta(hours=1)
    raw_data = (
        ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        .filterDate(
            utc_start.strftime('%Y-%m-%dT%H:%M:%SZ'),
            utc_end_exclusive.strftime('%Y-%m-%dT%H:%M:%SZ'),
        )
        .filterBounds(site_point)
        .select(ERA5_BANDS)
        .getRegion(site_point, SCALE_METEO)
        .getInfo()
    )
    if raw_data is None or len(raw_data) <= 1:
        raise RuntimeError('ERA5返回数据为空。')
    hourly = pd.DataFrame(raw_data[1:], columns=raw_data[0])
    daily = aggregate_era5_hourly_natural_days(hourly, local_start, local_end)
    daily.insert(0, 'WQS_name', site_name)
    daily.insert(1, 'WQS_Lon', site_lon)
    daily.insert(2, 'WQS_Lat', site_lat)
    return daily


# =========================================================
# 6.2 Complete ERA5 series for one station, 2021–2025
# =========================================================

def extract_era5_site(
        site_row
):

    site_name = str(
        site_row['WQS_name']
    )


    all_months = []


    for year in TARGET_YEARS:

        print(
            f"      ⛅ {year} 年"
        )


        for month in range(
            1,
            13
        ):

            month_success = False


            for attempt in range(
                1,
                MAX_RETRIES + 1
            ):

                try:

                    df_month = (
                        extract_era5_site_month(
                            site_row,
                            year,
                            month
                        )
                    )


                    if df_month.empty:

                        raise RuntimeError(
                            '返回月数据为空'
                        )


                    all_months.append(
                        df_month
                    )


                    print(
                        f"         "
                        f"{year}-{month:02d} "
                        f"✅ {len(df_month)} 天"
                    )


                    month_success = True

                    break


                except Exception as e:

                    print(
                        f"         "
                        f"{year}-{month:02d} "
                        f"⚠️ 第 "
                        f"{attempt}/"
                        f"{MAX_RETRIES} "
                        f"次失败：{e}"
                    )


                    if (
                        attempt
                        < MAX_RETRIES
                    ):

                        time.sleep(
                            RETRY_WAIT_SECONDS
                        )


            if not month_success:

                print(
                    f"      ❌ "
                    f"{site_name} "
                    f"{year}-{month:02d} "
                    f"ERA5提取失败"
                )

                return pd.DataFrame()


    # =====================================================
    # Combine all 60 months
    # =====================================================

    if not all_months:

        return pd.DataFrame()


    df = pd.concat(
        all_months,
        ignore_index=True
    )


    df[
        'Date'
    ] = pd.to_datetime(
        df['Date']
    )


    # =====================================================
    # Check for duplicate dates
    # =====================================================

    if (
        df['Date']
        .duplicated()
        .any()
    ):

        duplicate_values = (
            df.loc[
                df['Date'].duplicated(),
                'Date'
            ]
            .dt
            .strftime(
                '%Y-%m-%d'
            )
            .tolist()
        )


        raise RuntimeError(
            'ERA5存在重复日期，例如：'
            + str(
                duplicate_values[:10]
            )
        )


    # =====================================================
    # Sort records
    # =====================================================

    df = (
        df
        .sort_values('Date')
        .reset_index(drop=True)
    )


    # =====================================================
    # Check date coverage
    # =====================================================

    if len(df) != EXPECTED_DAYS:

        raise RuntimeError(
            f'ERA5理论应有 '
            f'{EXPECTED_DAYS} 天，'
            f'实际 {len(df)} 天'
        )


    if (
        df['Date'].min()
        != GLOBAL_START
    ):

        raise RuntimeError(
            f"ERA5起始日期错误："
            f"{df['Date'].min()}"
        )


    if (
        df['Date'].max()
        != GLOBAL_END
    ):

        raise RuntimeError(
            f"ERA5结束日期错误："
            f"{df['Date'].max()}"
        )


    actual_dates = pd.DatetimeIndex(
        df['Date']
    )


    missing_dates = (
        FULL_DATE_INDEX
        .difference(
            actual_dates
        )
    )


    if len(missing_dates) > 0:

        raise RuntimeError(
            'ERA5存在缺失日期：'
            + ', '.join(
                missing_dates
                .strftime(
                    '%Y-%m-%d'
                )
                .tolist()[:10]
            )
        )


    print(
        f"      ✅ ERA5完整："
        f"{df['Date'].min().strftime('%Y-%m-%d')} "
        f"→ "
        f"{df['Date'].max().strftime('%Y-%m-%d')}，"
        f"共 {len(df)} 天"
    )


    # =====================================================
    # Convert dates back to strings
    # =====================================================

    df[
        'Date'
    ] = (
        df[
            'Date'
        ]
        .dt
        .strftime(
            '%Y-%m-%d'
        )
    )


    return df


# =========================================================
# 7. MODIS EVI
#
# Revised extraction
#
# Extract only original MOD13Q1 composite dates in Earth Engine.
#
# Date alignment
# Normalize each MODIS composite date to midnight on that date
# without shifting it to 08:00 Beijing time.
# =========================================================

def extract_evi_site_raw(
        site_row
):

    site_name = str(
        site_row['WQS_name']
    )


    site_point = create_site_point(
        site_row
    )


    # =====================================================
    # Study period
    #
    # Extend extraction through 2026-02-01
    # to provide observations for interpolating EVI at the end of 2025.
    # =====================================================

    start_date = (
        f"{TARGET_YEARS[0]}-01-01"
    )


    end_date = (
        f"{TARGET_YEARS[-1] + 1}-02-01"
    )


    modis_col = (
        ee.ImageCollection(
            'MODIS/061/MOD13Q1'
        )
        .filterDate(
            start_date,
            end_date
        )
        .filterBounds(
            site_point
        )
        .select(
            [
                'EVI',
                'SummaryQA'
            ]
        )
    )


    raw_data = (
        modis_col
        .getRegion(
            site_point,
            SCALE_ECO
        )
        .getInfo()
    )


    if (
        raw_data is None
        or len(raw_data) <= 1
    ):

        raise RuntimeError(
            'MODIS EVI返回为空'
        )


    df = pd.DataFrame(
        raw_data[1:],
        columns=raw_data[0]
    )


    if df.empty:

        raise RuntimeError(
            'MODIS EVI DataFrame为空'
        )


    # =====================================================
    # Align MODIS EVI composite dates
    #
    # MOD13Q1 is a 16-day composite product.
    #
    # ERA5 requires hourly aggregation in Beijing time. MOD13Q1 already provides composite dates.
    #
    # Use each Earth Engine timestamp as the MODIS composite date,
    # then normalize the timestamp to midnight on that date.
    #
    # Example
    #
    # Original timestamp
    # 2021-01-01 00:00 UTC
    #
    # Normalized timestamp
    # 2021-01-01 00:00
    #
    # Timestamp to avoid
    # 2021-01-01 08:00
    #
    # This aligns the observations with the daily index.
    # =====================================================

    df[
        'Date'
    ] = (
        pd.to_datetime(
            df['time'],
            unit='ms',
            utc=True
        )
        .dt
        .tz_convert(None)
        .dt
        .normalize()
    )


    # =====================================================
    # Convert to numeric values
    # =====================================================

    df[
        'EVI'
    ] = pd.to_numeric(
        df['EVI'],
        errors='coerce'
    )


    df[
        'SummaryQA'
    ] = pd.to_numeric(
        df['SummaryQA'],
        errors='coerce'
    )


    # =====================================================
    # EVI scale factor
    #
    # MOD13Q1：
    # scale factor = 0.0001
    # =====================================================

    df[
        'EVI'
    ] = (
        df[
            'EVI'
        ]
        * 0.0001
    )


    # =====================================================
    # Keep the required fields
    # =====================================================

    df = (
        df[
            [
                'Date',
                'EVI',
                'SummaryQA'
            ]
        ]
        .drop_duplicates(
            subset=['Date'],
            keep='first'
        )
        .sort_values('Date')
        .reset_index(drop=True)
    )


    # =====================================================
    # Check valid source EVI values
    # =====================================================

    raw_valid = int(
        df[
            'EVI'
        ]
        .notna()
        .sum()
    )


    print(
        f"      ✅ MODIS原始EVI："
        f"{len(df)} 个时相，"
        f"有效EVI {raw_valid} 个"
    )


    if raw_valid == 0:

        raise RuntimeError(
            'MODIS EVI全部为空'
        )


    return df


# =========================================================
# 8. Dynamic World
# Annual land-cover mode for one station
# =========================================================

def extract_landcover_site(
        site_row
):

    site_point = create_site_point(
        site_row
    )


    year_list = ee.List(
        TARGET_YEARS
    )


    def annual_feature(year):

        year = ee.Number(
            year
        ).toInt()


        start = ee.Date.fromYMD(
            year,
            1,
            1
        )


        end = start.advance(
            1,
            'year'
        )


        annual_mode = (
            ee.ImageCollection(
                'GOOGLE/DYNAMICWORLD/V1'
            )
            .filterDate(
                start,
                end
            )
            .filterBounds(
                site_point
            )
            .select(
                'label'
            )
            .mode()
            .rename(
                'LandCover_Class'
            )
        )


        value_dict = (
            annual_mode
            .reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=site_point,
                scale=SCALE_ECO,
                maxPixels=1e6
            )
        )


        return ee.Feature(
            None,
            {
                'Year':
                    year,

                'LandCover_Class':
                    value_dict.get(
                        'LandCover_Class'
                    )
            }
        )


    annual_fc = ee.FeatureCollection(
        year_list.map(
            annual_feature
        )
    )


    info = (
        annual_fc
        .getInfo()
    )


    rows = []


    for feature in info[
        'features'
    ]:

        props = feature[
            'properties'
        ]


        rows.append({

            'Year':
                props.get(
                    'Year'
                ),

            'LandCover_Class':
                props.get(
                    'LandCover_Class'
                )
        })


    df = pd.DataFrame(
        rows
    )


    if df.empty:

        raise RuntimeError(
            'Dynamic World返回为空'
        )


    df[
        'Year'
    ] = pd.to_numeric(
        df['Year'],
        errors='coerce'
    )


    df[
        'LandCover_Class'
    ] = pd.to_numeric(
        df[
            'LandCover_Class'
        ],
        errors='coerce'
    )


    df = (
        df
        .dropna(
            subset=['Year']
        )
        .sort_values('Year')
        .reset_index(drop=True)
    )


    print(
        f"      ✅ Dynamic World："
        f"{len(df)} 个年度结果"
    )


    return df


# =========================================================
# 9. Local daily processing of EVI and land cover
# =========================================================

def process_eco_daily(
        evi_raw_df,
        landcover_df,
        site_row
):

    site_name = str(
        site_row['WQS_name']
    )


    site_lon = float(
        site_row['WQS_Lon']
    )


    site_lat = float(
        site_row['WQS_Lat']
    )


    # =====================================================
    # 1. Apply EVI quality flags
    # =====================================================

    evi = evi_raw_df.copy()


    # -----------------------------------------------------
    # Valid values before quality screening
    # -----------------------------------------------------

    valid_before_qa = int(
        evi[
            'EVI'
        ]
        .notna()
        .sum()
    )


    # -----------------------------------------------------
    # Use the original quality rule
    #
    # SummaryQA:
    #
    # 0 = good
    # 1 = marginal
    # 2 = snow / ice
    # 3 = cloudy
    #
    # Set values with QA of at least 2 to NaN
    # -----------------------------------------------------

    evi.loc[
        evi[
            'SummaryQA'
        ] >= 2,
        'EVI'
    ] = np.nan


    valid_after_qa = int(
        evi[
            'EVI'
        ]
        .notna()
        .sum()
    )


    print(
        f"      EVI QA前有效值："
        f"{valid_before_qa}"
    )


    print(
        f"      EVI QA后有效值："
        f"{valid_after_qa}"
    )


    if valid_after_qa == 0:

        raise RuntimeError(
            'EVI经过QA质控后全部为空'
        )


    # =====================================================
    # Normalize dates again
    #
    # Check for residual time components
    # =====================================================

    evi[
        'Date'
    ] = (
        pd.to_datetime(
            evi[
                'Date'
            ]
        )
        .dt
        .normalize()
    )


    evi = (
        evi
        .drop_duplicates(
            subset=['Date'],
            keep='first'
        )
        .sort_values('Date')
        .set_index('Date')
    )


    # =====================================================
    # Interpolation endpoint
    #
    # Use the extra MODIS observations from early 2026
    # =====================================================

    interpolation_end = max(
        GLOBAL_END,
        evi.index.max()
    )


    interpolation_index = (
        pd.date_range(
            start=GLOBAL_START,
            end=interpolation_end,
            freq='D'
        )
    )


    # =====================================================
    # Reindex to a daily time series
    # =====================================================

    evi_daily = (
        evi[
            ['EVI']
        ]
        .reindex(
            interpolation_index
        )
    )


    # =====================================================
    # Verify that original MODIS observations survive reindexing.
    #
    # This check catches the earlier failure where
    # all 117 composite dates disappeared after reindexing.
    # =====================================================

    retained_observations = int(
        evi_daily[
            'EVI'
        ]
        .notna()
        .sum()
    )


    print(
        f"      日序列中保留的MODIS有效观测："
        f"{retained_observations}"
    )


    if retained_observations == 0:

        raise RuntimeError(
            'EVI日期对齐失败：'
            '日序列中没有保留任何MODIS观测'
        )


    # =====================================================
    # Interpolate day by day with a linear method
    # =====================================================

    evi_daily[
        'EVI'
    ] = (
        evi_daily[
            'EVI'
        ]
        .interpolate(
            method='linear',
            limit_direction='both'
        )
    )


    # =====================================================
    # Restrict output to the study period
    # =====================================================

    evi_daily = (
        evi_daily
        .loc[
            GLOBAL_START:
            GLOBAL_END
        ]
        .copy()
    )


    # =====================================================
    # Check the plausible EVI range
    #
    # MODIS EVI typically ranges from -1 to 1.
    #
    # Retain out-of-range values
    # and report them as warnings.
    # =====================================================

    evi_min = (
        evi_daily[
            'EVI'
        ]
        .min()
    )


    evi_max = (
        evi_daily[
            'EVI'
        ]
        .max()
    )


    print(
        f"      EVI范围："
        f"{evi_min:.5f} "
        f"→ "
        f"{evi_max:.5f}"
    )


    if (
        evi_min < -1
        or evi_max > 1
    ):

        print(
            "      ⚠️ EVI存在超出"
            "[-1, 1]的值，请检查原始数据"
        )


    # =====================================================
    # Check for missing values after interpolation
    # =====================================================

    evi_missing = int(
        evi_daily[
            'EVI'
        ]
        .isna()
        .sum()
    )


    print(
        f"      EVI插值后缺失值："
        f"{evi_missing}"
    )


    if evi_missing > 0:

        raise RuntimeError(
            f'EVI插值后仍有 '
            f'{evi_missing} 个缺失值'
        )


    # =====================================================
    # Round to five decimal places
    # =====================================================

    evi_daily[
        'EVI'
    ] = (
        evi_daily[
            'EVI'
        ]
        .round(5)
    )


    evi_daily.index.name = (
        'Date'
    )


    evi_daily = (
        evi_daily
        .reset_index()
    )


    # =====================================================
    # 2. Annual Dynamic World class
    # =====================================================

    landcover_map = {}


    for _, row in (
        landcover_df.iterrows()
    ):

        if pd.isna(
            row['Year']
        ):

            continue


        year = int(
            row[
                'Year'
            ]
        )


        value = row[
            'LandCover_Class'
        ]


        if pd.isna(value):

            landcover_map[
                year
            ] = -1

        else:

            landcover_map[
                year
            ] = int(
                value
            )


    # =====================================================
    # Assign the annual land-cover class to every day of that year.
    #
    # Land cover is categorical,
    # so linear interpolation does not apply.
    # =====================================================

    evi_daily[
        'Year'
    ] = (
        evi_daily[
            'Date'
        ]
        .dt
        .year
    )


    evi_daily[
        'LandCover_Class'
    ] = (
        evi_daily[
            'Year'
        ]
        .map(
            landcover_map
        )
        .fillna(-1)
        .astype(int)
    )


    # =====================================================
    # Add station metadata
    # =====================================================

    evi_daily[
        'WQS_name'
    ] = site_name


    evi_daily[
        'WQS_Lon'
    ] = site_lon


    evi_daily[
        'WQS_Lat'
    ] = site_lat


    # =====================================================
    # Format dates as strings
    # =====================================================

    evi_daily[
        'Date'
    ] = (
        evi_daily[
            'Date'
        ]
        .dt
        .strftime(
            '%Y-%m-%d'
        )
    )


    # =====================================================
    # Output columns
    # =====================================================

    evi_daily = evi_daily[
        [
            'WQS_name',
            'WQS_Lon',
            'WQS_Lat',
            'Date',

            'EVI',
            'LandCover_Class'
        ]
    ]


    return evi_daily


# =========================================================
# 10. Process all inputs for one station
# =========================================================

def process_one_site(
        site_row,
        site_index,
        total_sites
):

    site_name = str(
        site_row['WQS_name']
    )


    safe_site_name = (
        safe_filename(
            site_name
        )
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        f'{safe_site_name}.csv'
    )


    print(
        "\n"
        "========================================================="
    )


    print(
        f"📍 "
        f"[{site_index}/{total_sites}] "
        f"开始处理断面："
        f"{site_name}"
    )


    print(
        f"   经度："
        f"{site_row['WQS_Lon']}"
    )


    print(
        f"   纬度："
        f"{site_row['WQS_Lat']}"
    )


    print(
        "========================================================="
    )


    # =====================================================
    # Skip stations with complete files
    # =====================================================

    if (
        SKIP_VALID_EXISTING
        and
        existing_file_is_complete(
            output_path
        )
    ):

        print(
            f"⏭️ 已存在完整结果，"
            f"跳过：{safe_site_name}.csv"
        )

        return True


    # =====================================================
    # ① ERA5
    # =====================================================

    print(
        "\n   ① ERA5-Land 气象数据"
    )


    print(
        "      GEE：提取逐小时点值"
    )


    print(
        "      Python：UTC → 北京时间"
    )


    print(
        "      Python：小时 → 日尺度聚合"
    )


    print(
        "      T：小时平均"
    )


    print(
        "      Tmax：小时最大"
    )


    print(
        "      Tmin：小时最小"
    )


    print(
        "      TP：hourly增量日累计"
    )


    print(
        "      SSR：hourly增量日累计"
    )


    print(
        "      WS：逐小时风速后日平均"
    )


    meteo_df = (
        extract_era5_site(
            site_row
        )
    )


    if meteo_df.empty:

        print(
            f"❌ {site_name} "
            f"ERA5提取失败"
        )

        return False


    # =====================================================
    # ② MODIS EVI
    # =====================================================

    print(
        "\n   ② MODIS EVI"
    )


    evi_raw_df = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            evi_raw_df = (
                extract_evi_site_raw(
                    site_row
                )
            )

            break


        except Exception as e:

            print(
                f"      ⚠️ EVI "
                f"第 {attempt}/"
                f"{MAX_RETRIES} "
                f"次失败：{e}"
            )


            if (
                attempt
                < MAX_RETRIES
            ):

                time.sleep(
                    RETRY_WAIT_SECONDS
                )


    if (
        evi_raw_df is None
        or evi_raw_df.empty
    ):

        print(
            f"❌ {site_name} "
            f"EVI提取失败"
        )

        return False


    # =====================================================
    # ③ Dynamic World
    # =====================================================

    print(
        "\n   ③ Dynamic World 年度土地覆盖"
    )


    landcover_df = None


    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            landcover_df = (
                extract_landcover_site(
                    site_row
                )
            )

            break


        except Exception as e:

            print(
                f"      ⚠️ 土地覆盖 "
                f"第 {attempt}/"
                f"{MAX_RETRIES} "
                f"次失败：{e}"
            )


            if (
                attempt
                < MAX_RETRIES
            ):

                time.sleep(
                    RETRY_WAIT_SECONDS
                )


    if (
        landcover_df is None
        or landcover_df.empty
    ):

        print(
            f"❌ {site_name} "
            f"Dynamic World提取失败"
        )

        return False


    # =====================================================
    # 4. Align EVI and land cover to daily dates
    # =====================================================

    print(
        "\n   ④ 生态数据日尺度处理"
    )


    print(
        "      EVI："
        "日期归一化 + QA质控 + 线性插值"
    )


    print(
        "      LandCover："
        "年度众数直接赋值"
    )


    try:

        eco_daily_df = (
            process_eco_daily(
                evi_raw_df,
                landcover_df,
                site_row
            )
        )


    except Exception as e:

        print(
            f"❌ {site_name} "
            f"生态数据处理失败：{e}"
        )

        return False


    if eco_daily_df.empty:

        print(
            f"❌ {site_name} "
            f"生态数据为空"
        )

        return False


    if (
        len(
            eco_daily_df
        )
        != EXPECTED_DAYS
    ):

        print(
            f"❌ 生态数据理论应为 "
            f"{EXPECTED_DAYS} 天，"
            f"实际 "
            f"{len(eco_daily_df)} 天"
        )

        return False


    print(
        f"      ✅ 生态日尺度："
        f"{len(eco_daily_df)} 天"
    )


    # =====================================================
    # 5. Combine data
    # =====================================================

    print(
        "\n   ⑤ 合并 "
        "气象 + EVI + 土地覆盖"
    )


    eco_merge = (
        eco_daily_df
        .drop(
            columns=[
                'WQS_Lon',
                'WQS_Lat'
            ],
            errors='ignore'
        )
    )


    final_df = pd.merge(
        meteo_df,
        eco_merge,
        on=[
            'WQS_name',
            'Date'
        ],
        how='left'
    )


    # =====================================================
    # 6. Verify date coverage
    # =====================================================

    final_df[
        'Date'
    ] = pd.to_datetime(
        final_df[
            'Date'
        ]
    )


    final_df = (
        final_df
        .sort_values(
            'Date'
        )
        .reset_index(
            drop=True
        )
    )


    if (
        final_df[
            'Date'
        ]
        .duplicated()
        .any()
    ):

        print(
            "❌ 最终结果存在重复日期"
        )

        return False


    if len(
        final_df
    ) != EXPECTED_DAYS:

        print(
            f"❌ 最终结果理论应为 "
            f"{EXPECTED_DAYS} 天，"
            f"实际 {len(final_df)} 天"
        )

        return False


    if (
        final_df[
            'Date'
        ]
        .min()
        != GLOBAL_START
    ):

        print(
            "❌ 最终数据起始日期错误"
        )

        return False


    if (
        final_df[
            'Date'
        ]
        .max()
        != GLOBAL_END
    ):

        print(
            "❌ 最终数据结束日期错误"
        )

        return False


    # =====================================================
    # 7. Check for missing values
    # =====================================================

    check_cols = [
        'T',
        'Tmax',
        'Tmin',
        'SP',
        'SSR',
        'TP',
        'WS',
        'Td',
        'EVI',
        'LandCover_Class'
    ]


    print(
        "\n   ⑥ 数据完整性检查"
    )


    print(
        f"      日期范围："
        f"{final_df['Date'].min().strftime('%Y-%m-%d')} "
        f"→ "
        f"{final_df['Date'].max().strftime('%Y-%m-%d')}"
    )


    print(
        f"      日记录数："
        f"{len(final_df)}"
    )


    print(
        "      缺失值："
    )


    for col in check_cols:

        if col in final_df.columns:

            n_missing = int(
                final_df[
                    col
                ]
                .isna()
                .sum()
            )


            print(
                f"         "
                f"{col:<18}"
                f"{n_missing}"
            )


    # =====================================================
    # Meteorological variables must be complete.
    # =====================================================

    meteo_cols = [
        'T',
        'Tmax',
        'Tmin',
        'SP',
        'SSR',
        'TP',
        'WS',
        'Td'
    ]


    meteo_missing = int(
        final_df[
            meteo_cols
        ]
        .isna()
        .sum()
        .sum()
    )


    if meteo_missing > 0:

        print(
            f"❌ 气象数据存在 "
            f"{meteo_missing} 个缺失值，"
            f"当前断面不输出。"
        )

        return False


    # =====================================================
    # EVI must also be complete.
    #
    # Enforce this completeness check for the current output.
    # =====================================================

    evi_missing = int(
        final_df[
            'EVI'
        ]
        .isna()
        .sum()
    )


    if evi_missing > 0:

        print(
            f"❌ EVI仍存在 "
            f"{evi_missing} 个缺失值，"
            f"当前断面不输出。"
        )

        return False


    # =====================================================
    # Use -1 for missing land cover.
    # =====================================================

    final_df[
        'LandCover_Class'
    ] = (
        final_df[
            'LandCover_Class'
        ]
        .fillna(-1)
        .astype(int)
    )


    # =====================================================
    # Convert dates back to strings
    # =====================================================

    final_df[
        'Date'
    ] = (
        final_df[
            'Date'
        ]
        .dt
        .strftime(
            '%Y-%m-%d'
        )
    )


    # =====================================================
    # Final column order
    # =====================================================

    cols_order = [

        'WQS_name',
        'WQS_Lon',
        'WQS_Lat',
        'Date',

        'T',
        'Tmax',
        'Tmin',
        'SP',
        'SSR',
        'TP',
        'WS',
        'Td',

        'EVI',
        'LandCover_Class'
    ]


    final_df = final_df[
        [
            col
            for col in cols_order
            if col in final_df.columns
        ]
    ]


    # =====================================================
    # 8. Save output
    # =====================================================

    final_df.to_csv(
        output_path,
        index=False,
        encoding='utf-8-sig',
        float_format='%.5f'
    )


    print(
        f"\n✅ "
        f"[{site_index}/{total_sites}] "
        f"{safe_site_name}.csv "
        f"保存完成"
    )


    print(
        f"   共 {len(final_df)} 条记录"
    )


    return True


# =========================================================
# 11. Main entry point
# =========================================================

if __name__ == '__main__':

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )


    print(
        "\n"
        "========================================================="
    )


    print(
        "🚀 黄河流域 "
        "ERA5 + EVI + 土地覆盖 "
        "逐断面提取程序"
    )


    print(
        "========================================================="
    )


    print(
        f"📍 水质断面数："
        f"{len(df_sites)}"
    )


    print(
        f"📅 研究年份："
        f"{TARGET_YEARS[0]}"
        f"–"
        f"{TARGET_YEARS[-1]}"
    )


    print(
        f"📆 理论日数："
        f"{EXPECTED_DAYS}"
    )


    print(
        "⏰ ERA5："
        "GEE提取小时点值 "
        "→ Python转换北京时间 "
        "→ Python日聚合"
    )


    print(
        "🌡️ T：小时平均"
    )


    print(
        "🔥 Tmax：小时最大"
    )


    print(
        "❄️ Tmin：小时最小"
    )


    print(
        "🌧️ TP："
        "total_precipitation_hourly "
        "→ 按01:00至次日00:00结束时标累计到北京时间自然日"
    )


    print(
        "☀️ SSR："
        "surface_net_solar_radiation_hourly "
        "→ 按01:00至次日00:00结束时标累计到北京时间自然日"
    )


    print(
        "💨 WS："
        "sqrt(u10² + v10²) "
        "→ 北京时间日平均"
    )


    print(
        "🌿 EVI："
        "MOD13Q1日期归一化 "
        "→ QA质控 "
        "→ 日尺度线性插值"
    )


    print(
        "🗺️ LandCover："
        "Dynamic World年度众数"
    )


    print(
        "========================================================="
    )


    success_sites = []

    failed_sites = []


    total_sites = len(
        df_sites
    )


    # =====================================================
    # Process stations one at a time
    # =====================================================

    for idx, site_row in (
        df_sites.iterrows()
    ):

        site_number = (
            idx + 1
        )


        site_name = str(
            site_row[
                'WQS_name'
            ]
        )


        try:

            success = (
                process_one_site(
                    site_row,
                    site_number,
                    total_sites
                )
            )


            if success:

                success_sites.append(
                    site_name
                )

            else:

                failed_sites.append(
                    site_name
                )


        except KeyboardInterrupt:

            print(
                "\n\n⛔ 用户手动终止程序。"
            )

            break


        except Exception as e:

            print(
                f"\n❌ "
                f"{site_name} "
                f"出现异常："
                f"{e}"
            )


            failed_sites.append(
                site_name
            )


    # =====================================================
    # 12. Final summary
    # =====================================================

    print(
        "\n"
        "========================================================="
    )


    print(
        "📊 全部断面处理总结"
    )


    print(
        "========================================================="
    )


    print(
        f"✅ 成功："
        f"{len(success_sites)} 个"
    )


    print(
        f"❌ 失败："
        f"{len(failed_sites)} 个"
    )


    if failed_sites:

        print(
            "\n失败断面："
        )


        for site in failed_sites:

            print(
                f"   - {site}"
            )


    print(
        "\n💾 输出目录："
    )


    print(
        os.path.abspath(
            OUTPUT_FOLDER
        )
    )


    print(
        "\n🎉 程序运行结束。"
    )