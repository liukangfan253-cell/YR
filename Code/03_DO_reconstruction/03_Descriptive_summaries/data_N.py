import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import warnings

# Suppress run-time warnings.
warnings.filterwarnings('ignore')

# =====================================================================
# Paths
# =====================================================================
INPUT_FOLDER = str(ROOT / 'Data/WQ_Hy_M_DYN_data/YR/by_station_all')
OUTPUT_FOLDER = str(ROOT / 'Result/Mean_N_data')
# =====================================================================


def process_csv_files(folder_path, output_base_dir):
    """
    Read all CSV files in a folder and calculate summary statistics.

    Parameters
    ----------
    folder_path : str
        Directory containing the CSV files.
    output_base_dir : str
        Root directory for the outputs.
    """

    # ================================================================
    # Variables to summarize
    # SD and SWE are excluded.
    # ================================================================
    stats_columns = [
        'T', 'Tmax', 'Tmin', 'SP', 'SSR', 'TP_M', 'WS', 'Td',
        'WT', 'pH', 'DO', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN',
        'CODₘₙ', 'WL', 'Q', 'Ss', 'EVI',
        'DO_1.5_sat', 'DO_sat'
    ]

    # Create the output directory.
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Read all CSV files.
    all_files = []
    csv_files = list(Path(folder_path).glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files")

    for file_path in csv_files:
        try:
            # Read as UTF-8.
            df = pd.read_csv(file_path, encoding='utf-8')

            # Ensure Date is a datetime column.
            df['Date'] = pd.to_datetime(df['Date'])

            # ========================================================
            # Standardize selected variable names.
            # ========================================================
            df.rename(columns={
                'NH3-N': 'NH₃-N',
                'CODMn': 'CODₘₙ',
                'CODmn': 'CODₘₙ'
            }, inplace=True)

            all_files.append(df)

        except Exception as e:
            print(f"Error reading file {file_path.name}: {e}")

    if not all_files:
        print("No files were successfully read")
        return

    # ================================================================
    # Combine all station records.
    # ================================================================
    combined_df = pd.concat(all_files, ignore_index=True)

    # Extract year and month.
    combined_df['Year'] = combined_df['Date'].dt.year
    combined_df['Month'] = combined_df['Date'].dt.month

    print(f"Total data rows: {len(combined_df)}")
    print(
        f"Date range: {combined_df['Date'].min()} "
        f"to {combined_df['Date'].max()}"
    )

    # ================================================================
    # Calculate the daily mean across all stations.
    # ================================================================
    print("\nCalculating daily means across all stations...")

    daily_means = calculate_daily_means(
        combined_df,
        stats_columns
    )

    daily_output = output_base / "All_Stations_Daily_Means.csv"

    daily_means.to_csv(
        daily_output,
        index=False,
        encoding='utf-8-sig'
    )

    print(f"Daily means saved to: {daily_output}")

    # ================================================================
    # Calculate summaries for each variable.
    # ================================================================
    for column in stats_columns:

        col_dir = output_base / column
        col_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {column}...")

        # ------------------------------------------------------------
        # 1. Overall statistics
        # ------------------------------------------------------------
        overall_stats = calculate_overall_stats(
            combined_df,
            column
        )

        overall_stats.to_csv(
            col_dir / f"{column}_overall_stats.csv",
            index=False,
            encoding='utf-8-sig'
        )

        # ------------------------------------------------------------
        # 2. Annual statistics
        # ------------------------------------------------------------
        yearly_stats = calculate_yearly_stats(
            combined_df,
            column
        )

        yearly_stats.to_csv(
            col_dir / f"{column}_yearly_stats.csv",
            index=False,
            encoding='utf-8-sig'
        )

        # ------------------------------------------------------------
        # 3. Monthly statistics
        # ------------------------------------------------------------
        monthly_by_station, monthly_overall = calculate_monthly_stats(
            combined_df,
            column
        )

        # Monthly statistics by station
        monthly_by_station.to_csv(
            col_dir / f"{column}_monthly_by_station.csv",
            index=False,
            encoding='utf-8-sig'
        )

        # Pooled monthly statistics
        monthly_overall.to_csv(
            col_dir / f"{column}_monthly_all_stations.csv",
            index=False,
            encoding='utf-8-sig'
        )

    print(
        f"\nProcessing completed! "
        f"Results saved to: {output_base_dir}"
    )


def calculate_daily_means(df, stats_columns):
    """
    Compute daily means across all stations.
    """

    # Group by Date and calculate the daily mean across stations.
    daily_means = (
        df.groupby('Date')[stats_columns]
        .mean()
        .reset_index()
    )

    # Add calendar information.
    daily_means['Year'] = daily_means['Date'].dt.year
    daily_means['Month'] = daily_means['Date'].dt.month
    daily_means['Day'] = daily_means['Date'].dt.day

    # Arrange columns.
    column_order = (
        ['Date', 'Year', 'Month', 'Day']
        + stats_columns
    )

    daily_means = daily_means[column_order]

    # Round numeric values to four decimal places.
    daily_means[stats_columns] = (
        daily_means[stats_columns]
        .round(4)
    )

    return daily_means


def calculate_overall_stats(df, column):
    """
    Compute overall statistics by station.
    """

    # ================================================================
    # Statistics by station
    # ================================================================
    overall_stats = (
        df.groupby(
            ['WQS_name', 'WQS_Lon', 'WQS_Lat']
        )[column]
        .agg([
            ('Mean', 'mean'),
            ('Max', 'max'),
            ('Min', 'min'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Count', 'count')
        ])
        .reset_index()
    )

    # ================================================================
    # Pooled statistics across stations
    # ================================================================
    overall_row = pd.DataFrame({
        'WQS_name': ['All_Stations'],
        'WQS_Lon': [np.nan],
        'WQS_Lat': [np.nan],
        'Mean': [df[column].mean()],
        'Max': [df[column].max()],
        'Min': [df[column].min()],
        'Median': [df[column].median()],
        'Std': [df[column].std()],
        'Count': [df[column].count()]
    })

    result = pd.concat(
        [overall_stats, overall_row],
        ignore_index=True
    )

    return result.round(4)


def calculate_yearly_stats(df, column):
    """
    Compute annual statistics.
    """

    # ================================================================
    # Annual statistics by station
    # ================================================================
    yearly_stats = (
        df.groupby(
            [
                'WQS_name',
                'WQS_Lon',
                'WQS_Lat',
                'Year'
            ]
        )[column]
        .agg([
            ('Mean', 'mean'),
            ('Max', 'max'),
            ('Min', 'min'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Count', 'count')
        ])
        .reset_index()
    )

    # ================================================================
    # Pooled annual statistics across stations
    # ================================================================
    yearly_overall = (
        df.groupby('Year')[column]
        .agg([
            ('Mean', 'mean'),
            ('Max', 'max'),
            ('Min', 'min'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Count', 'count')
        ])
        .reset_index()
    )

    yearly_overall['WQS_name'] = 'All_Stations'
    yearly_overall['WQS_Lon'] = np.nan
    yearly_overall['WQS_Lat'] = np.nan

    # Arrange columns.
    yearly_overall = yearly_overall[
        [
            'WQS_name',
            'WQS_Lon',
            'WQS_Lat',
            'Year',
            'Mean',
            'Max',
            'Min',
            'Median',
            'Std',
            'Count'
        ]
    ]

    result = pd.concat(
        [yearly_stats, yearly_overall],
        ignore_index=True
    )

    return result.round(4)


def calculate_monthly_stats(df, column):
    """
    Compute monthly statistics.

    Returns
    -------
    monthly_by_station : DataFrame
        Monthly results by station and year.
    monthly_overall : DataFrame
        Pooled monthly results by year.
    """

    # ================================================================
    # Monthly statistics by station
    # ================================================================
    monthly_by_station = (
        df.groupby(
            [
                'WQS_name',
                'WQS_Lon',
                'WQS_Lat',
                'Year',
                'Month'
            ]
        )[column]
        .agg([
            ('Mean', 'mean'),
            ('Max', 'max'),
            ('Min', 'min'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Count', 'count')
        ])
        .reset_index()
    )

    # ================================================================
    # Pooled monthly statistics
    # ================================================================
    monthly_overall = (
        df.groupby(
            ['Year', 'Month']
        )[column]
        .agg([
            ('Mean', 'mean'),
            ('Max', 'max'),
            ('Min', 'min'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Count', 'count')
        ])
        .reset_index()
    )

    monthly_overall['WQS_name'] = 'All_Stations'
    monthly_overall['WQS_Lon'] = np.nan
    monthly_overall['WQS_Lat'] = np.nan

    # Arrange columns.
    monthly_overall = monthly_overall[
        [
            'WQS_name',
            'WQS_Lon',
            'WQS_Lat',
            'Year',
            'Month',
            'Mean',
            'Max',
            'Min',
            'Median',
            'Std',
            'Count'
        ]
    ]

    return (
        monthly_by_station.round(4),
        monthly_overall.round(4)
    )


# =====================================================================
# Main entry point
# =====================================================================
if __name__ == "__main__":

    # Check that the input directory exists.
    if not os.path.exists(INPUT_FOLDER):

        print(
            f"Error: Folder {INPUT_FOLDER} does not exist! "
            f"Please check the path."
        )

    else:

        # Calculate statistics.
        process_csv_files(
            INPUT_FOLDER,
            OUTPUT_FOLDER
        )

        print("\n" + "=" * 50)
        print("Output file structure:")
        print("=" * 50)

        # Display the output directory structure.
        for root, dirs, files in os.walk(OUTPUT_FOLDER):

            level = root.replace(
                OUTPUT_FOLDER,
                ''
            ).count(os.sep)

            indent = ' ' * 2 * level

            print(
                f'{indent}{os.path.basename(root)}/'
            )

            subindent = ' ' * 2 * (level + 1)

            for file in files:

                if file.endswith('.csv'):

                    print(
                        f'{subindent}{file}'
                    )