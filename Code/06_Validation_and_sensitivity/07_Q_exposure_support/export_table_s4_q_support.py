"""Export the discharge and basic high-flow provenance for Supplementary Table S4."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HYDROLOGY = ROOT / "Data/Hydrology_data_P/Hy_data_YR"
MERGED_SITES = ROOT / "Data/WQ_Hy_M_DYN_data/YR/by_station_all"
MAPPING = ROOT / "Config/Study_Station_Mapping.csv"
DEFAULT_OUTPUT = ROOT / "Supplementary_Data/TableS4_Q_Support"
DATES = pd.date_range("2021-01-01", "2025-12-31", name="Date")
EPISODE_COLUMNS = [
    "Hydro_Site_ID", "Episode_ID", "Start_Date", "End_Date", "Duration_Days", "P95",
    "Observed_High_Flow_Days", "Imputed_High_Flow_Days", "Contains_Imputed_Q",
    "Start_Q_Imputed", "End_Q_Imputed", "Imputed_Bridge_Runs", "Matched_WQ_Sites",
]


def read_daily(path, columns):
    daily = pd.read_csv(path, usecols=["Date", *columns])
    daily["Date"] = pd.to_datetime(daily["Date"], errors="raise")
    daily = daily.set_index("Date", verify_integrity=True)
    missing_dates = DATES.difference(daily.index)
    if len(missing_dates):
        raise ValueError(f"{path}: missing study calendar dates, starting at {missing_dates[0].date()}")
    daily = daily.reindex(DATES)
    daily["Q"] = pd.to_numeric(daily["Q"], errors="raise")
    if np.isinf(daily["Q"]).any():
        raise ValueError(f"{path}: infinite discharge value")
    return daily


def true_runs(flags):
    """Return half-open position intervals of consecutive true days."""
    flags = np.asarray(flags, dtype=np.int8)
    edges = np.diff(flags, prepend=0, append=0)
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def audit_gauge(gauge, matches):
    original = read_daily(HYDROLOGY / f"Original/Retained/{gauge}.csv", ["Q"])["Q"]
    completed = read_daily(HYDROLOGY / f"Interpolated/Retained/{gauge}_Interpolated.csv", ["Q"])["Q"]
    observed = original.notna()
    if not np.array_equal(original[observed], completed[observed]):
        raise ValueError(f"{gauge}: completed discharge differs on observed dates")
    if not completed.notna().any():
        raise ValueError(f"{gauge}: no completed discharge values for P95")

    sites = sorted(matches["WQS_name"])
    for site in sites:
        merged = read_daily(MERGED_SITES / f"{site}.csv", ["Q", "Matched_HS_Name"])
        if not merged["Matched_HS_Name"].eq(gauge).all() or not np.array_equal(
            merged["Q"], completed, equal_nan=True
        ):
            raise ValueError(f"{site}: baseline daily discharge does not match {gauge}")

    # S0 uses the linear sample percentile and includes days equal to P95.
    p95 = np.percentile(completed.dropna(), 95)
    imputed = ~observed & completed.notna()
    high_flow = completed.ge(p95)
    daily = pd.DataFrame({
        "Hydro_Site_ID": gauge,
        "Date": DATES,
        "Q_Original": original.to_numpy(),
        "Q_Completed": completed.to_numpy(),
        "Q_Original_Missing": (~observed).astype(int).to_numpy(),
        "Q_Imputed": imputed.astype(int).to_numpy(),
        "P95": p95,
        "High_Flow_Completed": high_flow.astype(int).to_numpy(),
        "Basic_Episode_ID": "",
    })
    episodes = []
    for number, (start, stop) in enumerate(true_runs(high_flow), start=1):
        episode_id = f"{gauge}_HF{number:03d}"
        flags = imputed.iloc[start:stop]
        imputed_days = int(flags.sum())
        # A bridge must have observed high-flow days on both sides in this episode.
        bridges = sum(lo > 0 and hi < len(flags) for lo, hi in true_runs(flags))
        daily.loc[start:stop - 1, "Basic_Episode_ID"] = episode_id
        episodes.append({
            "Hydro_Site_ID": gauge,
            "Episode_ID": episode_id,
            "Start_Date": DATES[start],
            "End_Date": DATES[stop - 1],
            "Duration_Days": stop - start,
            "P95": p95,
            "Observed_High_Flow_Days": stop - start - imputed_days,
            "Imputed_High_Flow_Days": imputed_days,
            "Contains_Imputed_Q": imputed_days > 0,
            "Start_Q_Imputed": bool(flags.iloc[0]),
            "End_Q_Imputed": bool(flags.iloc[-1]),
            "Imputed_Bridge_Runs": bridges,
            "Matched_WQ_Sites": "|".join(sites),
        })

    metadata = matches[["Matched_HS_Lon", "Matched_HS_Lat", "Matched_HS_River"]].drop_duplicates()
    if len(metadata) != 1 or metadata.isna().any().any():
        raise ValueError(f"{gauge}: inconsistent or missing gauge metadata in {MAPPING}")
    lon, lat, river = metadata.iloc[0]
    summary = {
        "Hydro_Site_ID": gauge,
        "Longitude": lon,
        "Latitude": lat,
        "River_Name": river,
        "Channel_Class": "Mainstem" if river == "Yellow_river" else "Tributary",
        "Matched_WQ_Site_Count": len(sites),
        "Expected_Days": len(DATES),
        "Original_Valid_Days": int(observed.sum()),
        "Completed_Valid_Days": int(completed.notna().sum()),
        "Original_Missing_Days": int((~observed).sum()),
        "Imputed_Days": int(imputed.sum()),
        "Remaining_Missing_Days": int(completed.isna().sum()),
        "Max_Q_Gap_Days": max((hi - lo for lo, hi in true_runs(~observed)), default=0),
        "P95": p95,
        "High_Flow_Days": int(high_flow.sum()),
        "Basic_High_Flow_Episodes": len(episodes),
        "Episodes_With_Imputed_Q": sum(episode["Contains_Imputed_Q"] for episode in episodes),
        "Imputed_High_Flow_Days": int((imputed & high_flow).sum()),
        "Imputed_Bridge_Runs": sum(episode["Imputed_Bridge_Runs"] for episode in episodes),
    }
    return summary, daily, episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output_dir

    mapping = pd.read_csv(MAPPING).query("Status == 'Retained'")
    keys = mapping[["WQS_name", "Matched_HS_Name"]]
    if keys.empty or keys.isna().any().any() or keys["WQS_name"].duplicated().any():
        raise ValueError(f"{MAPPING}: expected one valid gauge assignment per retained site")

    gauge_rows, daily_tables, episode_rows = [], [], []
    for gauge, matches in mapping.groupby("Matched_HS_Name", sort=True):
        summary, daily, episodes = audit_gauge(gauge, matches)
        gauge_rows.append(summary)
        daily_tables.append(daily)
        episode_rows.extend(episodes)

    tables = {
        "S4_Gauge_Q_Summary.csv": pd.DataFrame(gauge_rows),
        "S4_Daily_Q_Provenance.csv": pd.concat(daily_tables, ignore_index=True),
        "S4_Basic_High_Flow_Episodes.csv": pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS),
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        for name, table in tables.items():
            table.to_csv(Path(temp) / name, index=False, date_format="%Y-%m-%d", lineterminator="\n")
        for name, table in tables.items():
            (Path(temp) / name).replace(output / name)
            print(f"{output / name}: {len(table):,} rows")


if __name__ == "__main__":
    main()
