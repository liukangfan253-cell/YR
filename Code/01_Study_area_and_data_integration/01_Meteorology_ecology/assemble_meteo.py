"""Validate and collect daily meteorology files from both V1 station groups."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
STUDY_DATES = set(pd.date_range('2021-01-01', '2025-12-31'))


def validate_station_file(source, station_name):
    try:
        table = pd.read_csv(source, encoding='utf-8-sig')
    except UnicodeDecodeError:
        table = pd.read_csv(source, encoding='gbk')
    dates = pd.to_datetime(table['Date']).dt.normalize()
    if dates.duplicated().any() or set(dates) != STUDY_DATES:
        raise ValueError(f'Expected one row per study date: {source}')
    if set(table.WQS_name.dropna()) != {station_name}:
        raise ValueError(f'Unexpected station identifier: {source}')
    precipitation = 'TP_M' if 'TP_M' in table.columns else 'TP'
    if precipitation not in table.columns:
        raise ValueError(f'Missing precipitation column (TP or TP_M): {source}')
    if not np.isfinite(table[['Tmax', 'SP', precipitation, 'SSR']].to_numpy(dtype=float)).all():
        raise ValueError(f'Non-finite meteorology values: {source}')


def main():
    mapping = pd.read_csv(ROOT / 'Config/Study_Station_Mapping.csv')
    if len(mapping) != 66 or mapping.WQS_name.nunique() != 66 or mapping.Matched_HS_Name.nunique() != 42:
        raise ValueError('Expected 66 water-quality sites matched to 42 hydrological stations.')
    meteo_dir = ROOT / 'Data/Meteorology_data/2021-2025'
    output_dir = meteo_dir / 'YR_V1'
    sources = []
    for station in mapping.itertuples():
        group = 'M_V1' if station.WQS_River == 'Yellow_river' else 'B_V1'
        source = meteo_dir / group / f'{station.WQS_name}.csv'
        validate_station_file(source, station.WQS_name)
        sources.append(source)

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, output_dir / source.name)
    print('Assembled 66 corrected V1 daily station files; values not changed.')


if __name__ == '__main__':
    main()
