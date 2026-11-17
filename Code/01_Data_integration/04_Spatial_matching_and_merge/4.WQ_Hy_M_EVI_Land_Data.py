"""Match the nearest same-river hydrological station and merge the daily inputs."""

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CALENDAR = pd.date_range('2021-01-01', '2025-12-31')
STUDY_DATES = set(CALENDAR)


def read_daily_table(path):
    try:
        table = pd.read_csv(path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        table = pd.read_csv(path, encoding='gbk')
    table['Date'] = pd.to_datetime(table['Date']).dt.normalize()
    dates = table['Date']
    if dates.duplicated().any() or set(dates) != STUDY_DATES:
        raise ValueError(f'Expected one row per study date: {path}')
    return table.set_index('Date').reindex(CALENDAR).rename_axis('Date')


def read_meteorology_table(path):
    table = read_daily_table(path)
    if 'TP_M' not in table.columns:
        if 'TP' not in table.columns:
            raise ValueError(f'Missing precipitation column (TP or TP_M): {path}')
        table = table.rename(columns={'TP': 'TP_M'})
    if table['TP_M'].lt(0).any():
        raise ValueError(f'Negative daily precipitation after normalization: {path}')
    return table


def station_metadata(table, columns, source):
    metadata = table[columns].drop_duplicates()
    if len(metadata) != 1 or metadata.iloc[0].isna().any():
        raise ValueError(f'Incomplete or inconsistent station metadata: {source}')
    return next(metadata.itertuples(index=False))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in kilometres using the V4 radius."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_latitude = math.radians(lat2 - lat1)
    delta_longitude = math.radians(lon2 - lon1)
    a = math.sin(delta_latitude / 2) ** 2
    a += math.cos(phi1) * math.cos(phi2) * math.sin(delta_longitude / 2) ** 2
    return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def main():
    water_quality_dir = ROOT / 'Data/WaterQuality_data_P/WQ_data_YR/Retained/Cleaned'
    hydrology_dir = ROOT / 'Data/Hydrology_data_P/Hy_data_YR/Interpolated/Retained'
    meteo_dir = ROOT / 'Data/Meteorology_data_P'
    output_dir = ROOT / 'Data/WQ_Hy_M_DYN_data/YR'
    by_station_dir = output_dir / 'by_station_all'
    regions = {'UR': 'upstream', 'MR': 'midstream', 'LR': 'downstream'}
    meteo_columns = ['T', 'Tmax', 'Tmin', 'SP', 'SSR', 'TP_M', 'WS', 'Td', 'EVI', 'LandCover_Class']
    hydro_columns = ['WL', 'Q', 'Ss']
    columns = [
        'WQS_name', 'WQS_Lon', 'WQS_Lat', 'River', 'Date', *meteo_columns,
        'WT', 'pH', 'DO', 'DO_sat', 'DO_1.5_sat', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN', 'CODₘₙ',
        'Matched_HS_Name', 'Matched_HS_Lon', 'Matched_HS_Lat', *hydro_columns, 'Region',
    ]
    numeric_columns = [
        *meteo_columns[:-1], 'WT', 'pH', 'DO', 'DO_sat', 'DO_1.5_sat', 'EC', 'Turb',
        'NH₃-N', 'TP', 'TN', 'CODₘₙ', *hydro_columns,
    ]

    hydro_by_river = {}
    hydro_tables = {}
    hydro_fields = ['HyS_name', 'River', 'Region', 'HyS_Lon', 'HyS_Lat']
    for source in sorted(hydrology_dir.glob('*_Interpolated.csv')):
        table = read_daily_table(source)
        station = station_metadata(table, hydro_fields, source)
        if source.stem != f'{station.HyS_name}_Interpolated':
            raise ValueError(f'Unexpected hydrological station identifier: {source}')
        if station.Region not in regions:
            raise ValueError(f'Unexpected reach for {station.HyS_name}: {station.Region}')
        hydro_tables[station.HyS_name] = table
        hydro_by_river.setdefault(station.River, []).append(station)
    if not hydro_tables:
        raise FileNotFoundError(f'No hydrological station daily files found: {hydrology_dir}')

    water_quality_files = sorted(water_quality_dir.glob('*.csv'))
    if len(water_quality_files) != 66:
        raise ValueError(f'Expected 66 water-quality station daily files, found {len(water_quality_files)}: {water_quality_dir}')
    blocks = []
    matches = []
    water_quality_fields = ['WQS_name', 'River', 'WQS_Lon', 'WQS_Lat']
    for source in water_quality_files:
        daily = read_daily_table(source)
        site = station_metadata(daily, water_quality_fields, source)
        if source.stem != site.WQS_name:
            raise ValueError(f'Unexpected water-quality station identifier: {source}')
        candidates = hydro_by_river.get(site.River)
        if not candidates:
            raise ValueError(f'No same-river hydrological station for {site.WQS_name} ({site.River})')
        distance, station = min(
            (
                (haversine_distance(site.WQS_Lat, site.WQS_Lon, hydro.HyS_Lat, hydro.HyS_Lon), hydro)
                for hydro in candidates
            ),
            key=lambda match: match[0],
        )
        hydrology = hydro_tables[station.HyS_name]
        meteorology = read_meteorology_table(meteo_dir / source.name)
        if set(meteorology.WQS_name.dropna()) != {site.WQS_name}:
            raise ValueError(f'Unexpected meteorological station identifier: {source.name}')

        daily[meteo_columns] = meteorology[meteo_columns]
        daily[hydro_columns] = hydrology[hydro_columns]
        daily['Matched_HS_Name'] = station.HyS_name
        daily['Matched_HS_Lon'] = station.HyS_Lon
        daily['Matched_HS_Lat'] = station.HyS_Lat
        daily['Region'] = station.Region
        for column in numeric_columns:
            daily[column] = pd.to_numeric(daily[column], errors='coerce')
        daily['LandCover_Class'] = daily['LandCover_Class'].fillna(-1).astype(int)
        for column in hydro_columns:
            if daily[column].isna().any():
                daily[column] = daily.groupby(daily.index.year)[column].transform(
                    lambda series: series.interpolate(method='linear', limit_direction='both').ffill().bfill()
                ).round(4)
        daily = daily.reset_index()[columns]
        blocks.append(daily)
        matches.append({
            'WQS_name': site.WQS_name,
            'WQS_Lon': site.WQS_Lon,
            'WQS_Lat': site.WQS_Lat,
            'WQS_River': site.River,
            'Status': 'Retained',
            'Matched_HS_Name': station.HyS_name,
            'Matched_HS_Lon': station.HyS_Lon,
            'Matched_HS_Lat': station.HyS_Lat,
            'Matched_HS_River': station.River,
            'Match_Strategy': f'Matched (Same River: {site.River})',
            'Distance_km': round(distance, 2),
            'Region': station.Region,
        })

    mapping = pd.DataFrame(matches)
    all_days = pd.concat(blocks, ignore_index=True)
    by_station_dir.mkdir(parents=True, exist_ok=True)
    for match, daily in zip(matches, blocks):
        site_file = f"{match['WQS_name']}.csv"
        daily.to_csv(by_station_dir / site_file, index=False, encoding='utf-8-sig')
        region_dir = output_dir / regions[match['Region']] / 'by_station'
        region_dir.mkdir(parents=True, exist_ok=True)
        daily.to_csv(region_dir / site_file, index=False, encoding='utf-8-sig')

    all_days.to_csv(output_dir / 'All_Merged_Data.csv', index=False, encoding='utf-8-sig')
    for region, name in regions.items():
        part = all_days[all_days['Region'].eq(region)]
        region_dir = output_dir / name
        aggregate = region_dir / f'{name}_all_data.csv'
        if part.empty:
            aggregate.unlink(missing_ok=True)
        else:
            part.to_csv(aggregate, index=False, encoding='utf-8-sig')
        reassigned_sites = mapping.loc[mapping['Region'].ne(region), 'WQS_name']
        for site in reassigned_sites:
            (region_dir / 'by_station' / f'{site}.csv').unlink(missing_ok=True)
    mapping.to_csv(output_dir / 'Match_Summary_Retained.csv', index=False, encoding='utf-8-sig')
    print(
        f'Matched and merged {len(mapping)} water-quality stations to '
        f'{mapping.Matched_HS_Name.nunique()} nearest same-river hydrological stations.'
    )


if __name__ == '__main__':
    main()
