"""Prepare the daily meteorology tables with negative precipitation set to zero."""

import csv
import io
import math
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[3]
METEO = ROOT / 'Data/Meteorology_data/2021-2025'
INPUT_DIR = METEO / 'YR_V1'
OUTPUT_DIR = ROOT / 'Data/Meteorology_data_P'
STATIONS = ROOT / 'Config/Study_Station_Mapping.csv'


def normalize_station(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        precipitation = {'TP', 'TP_M'}.intersection(columns)
        if len(precipitation) != 1:
            raise ValueError(f'Expected one precipitation column (TP or TP_M): {path}')
        column = precipitation.pop()
        table = io.StringIO(newline='')
        writer = csv.DictWriter(table, columns)
        writer.writeheader()
        changed = 0
        for row in reader:
            value = float(row[column])
            if not math.isfinite(value):
                raise ValueError(f'Non-finite precipitation in {path}, row {reader.line_num}')
            if value < 0:
                row[column] = '0'
                changed += 1
            writer.writerow(row)
        return table.getvalue(), changed


def main():
    with STATIONS.open(encoding='utf-8-sig', newline='') as stream:
        sites = [row['WQS_name'] for row in csv.DictReader(stream)]
    if not sites or len(sites) != len(set(sites)):
        raise ValueError(f'Missing or duplicate study station identifiers: {STATIONS}')

    outputs = {site: normalize_station(INPUT_DIR / f'{site}.csv') for site in sites}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for site, (content, changed) in outputs.items():
        stream = NamedTemporaryFile(mode='w', encoding='utf-8-sig', newline='',
                                    dir=OUTPUT_DIR, delete=False)
        temporary = Path(stream.name)
        try:
            with stream:
                stream.write(content)
            temporary.replace(OUTPUT_DIR / f'{site}.csv')
        finally:
            temporary.unlink(missing_ok=True)
        if changed:
            print(f'{site}: {changed} negative values set to zero')
    total = sum(changed for _, changed in outputs.values())
    print(f'{len(sites)} daily station tables saved to {OUTPUT_DIR}; {total} values set to zero.')


if __name__ == '__main__':
    main()
