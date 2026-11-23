import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
import warnings

# Suppress run-time warnings.
warnings.filterwarnings('ignore')


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

    # Select variables using standardized chemical formulas with subscripts.
    stats_columns = ['T', 'Tmax', 'Tmin', 'SP', 'SSR', 'TP_M', 'WS', 'Td',
                     'WT', 'pH', 'DO', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN',
                     'CODₘₙ', 'WL', 'Q', 'Ss', 'EVI']

    # Create the output directory.
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Read all CSV files.
    all_files = []
    csv_files = list(Path(folder_path).glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files")

    for file_path in csv_files:
        try:
            # Read as UTF-8 to preserve Unicode subscripts.
            df = pd.read_csv(file_path, encoding='utf-8')

            # Convert the Date column to datetime.
            df['Date'] = pd.to_datetime(df['Date'])

            # Rename plain-text chemical formulas to use subscripts.
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

    # Combine all data.
    combined_df = pd.concat(all_files, ignore_index=True)

    # Extract year and month.
    combined_df['Year'] = combined_df['Date'].dt.year
    combined_df['Month'] = combined_df['Date'].dt.month

    print(f"Total data rows: {len(combined_df)}")
    print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")

    # Calculate and save daily means across all stations.
    print("\nCalculating daily means across all stations...")
    daily_means = calculate_daily_means(combined_df, stats_columns)
    daily_means.to_csv(output_base / "All_Stations_Daily_Means.csv", index=False)
    print(f"Daily means saved to: {output_base / 'All_Stations_Daily_Means.csv'}")

    # Create an output directory for each variable.
    for column in stats_columns:
        col_dir = output_base / column
        col_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing {column}...")

        # 1. Compute overall summary statistics.
        overall_stats = calculate_overall_stats(combined_df, column)
        overall_stats.to_csv(col_dir / f"{column}_overall_stats.csv", index=False)

        # 2. Compute annual summary statistics.
        yearly_stats = calculate_yearly_stats(combined_df, column)
        yearly_stats.to_csv(col_dir / f"{column}_yearly_stats.csv", index=False)

        # 3. Compute monthly summaries for each station and for all stations pooled.
        monthly_by_station, monthly_overall = calculate_monthly_stats(combined_df, column)

        # Save monthly statistics by station.
        monthly_by_station.to_csv(col_dir / f"{column}_monthly_by_station.csv", index=False)

        # Save the overall monthly statistics.
        monthly_overall.to_csv(col_dir / f"{column}_monthly_all_stations.csv", index=False)

    print(f"\nProcessing completed! Results saved to: {output_base_dir}")


def calculate_daily_means(df, stats_columns):
    """Compute daily means across all stations."""
    # Group by date and average each variable across stations.
    daily_means = df.groupby('Date')[stats_columns].mean().reset_index()

    # Add year and month for later analyses.
    daily_means['Year'] = daily_means['Date'].dt.year
    daily_means['Month'] = daily_means['Date'].dt.month
    daily_means['Day'] = daily_means['Date'].dt.day

    # Put the date-related columns first.
    column_order = ['Date', 'Year', 'Month', 'Day'] + stats_columns
    daily_means = daily_means[column_order]

    # Round to four decimal places.
    daily_means[stats_columns] = daily_means[stats_columns].round(4)

    return daily_means


def calculate_overall_stats(df, column):
    """Compute overall statistics."""
    # Group by WQS_name, WQS_Lon and WQS_Lat.
    overall_stats = df.groupby(['WQS_name', 'WQS_Lon', 'WQS_Lat'])[column].agg([
        ('Mean', 'mean'),
        ('Max', 'max'),
        ('Min', 'min'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Count', 'count')
    ]).reset_index()

    # Add the pooled result as an aggregate row labeled All_Stations.
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

    result = pd.concat([overall_stats, overall_row], ignore_index=True)
    return result.round(4)


def calculate_yearly_stats(df, column):
    """Compute annual statistics."""
    # Group by Year, WQS_name, WQS_Lon and WQS_Lat.
    yearly_stats = df.groupby(['WQS_name', 'WQS_Lon', 'WQS_Lat', 'Year'])[column].agg([
        ('Mean', 'mean'),
        ('Max', 'max'),
        ('Min', 'min'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Count', 'count')
    ]).reset_index()

    # Overall statistics by year
    yearly_overall = df.groupby('Year')[column].agg([
        ('Mean', 'mean'),
        ('Max', 'max'),
        ('Min', 'min'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Count', 'count')
    ]).reset_index()
    yearly_overall['WQS_name'] = 'All_Stations'
    yearly_overall['WQS_Lon'] = np.nan
    yearly_overall['WQS_Lat'] = np.nan

    # Reorder columns.
    yearly_overall = yearly_overall[['WQS_name', 'WQS_Lon', 'WQS_Lat', 'Year',
                                     'Mean', 'Max', 'Min', 'Median', 'Std', 'Count']]

    result = pd.concat([yearly_stats, yearly_overall], ignore_index=True)
    return result.round(4)


def calculate_monthly_stats(df, column):
    """Compute monthly statistics and return separate DataFrames for station-level and pooled results."""
    # Monthly statistics by WQS_name, WQS_Lon and WQS_Lat
    monthly_by_station = df.groupby(['WQS_name', 'WQS_Lon', 'WQS_Lat', 'Year', 'Month'])[column].agg([
        ('Mean', 'mean'),
        ('Max', 'max'),
        ('Min', 'min'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Count', 'count')
    ]).reset_index()

    # Overall monthly statistics across stations
    monthly_overall = df.groupby(['Year', 'Month'])[column].agg([
        ('Mean', 'mean'),
        ('Max', 'max'),
        ('Min', 'min'),
        ('Median', 'median'),
        ('Std', 'std'),
        ('Count', 'count')
    ]).reset_index()
    monthly_overall['WQS_name'] = 'All_Stations'
    monthly_overall['WQS_Lon'] = np.nan
    monthly_overall['WQS_Lat'] = np.nan

    # Reorder columns.
    monthly_overall = monthly_overall[['WQS_name', 'WQS_Lon', 'WQS_Lat', 'Year', 'Month',
                                       'Mean', 'Max', 'Min', 'Median', 'Std', 'Count']]

    # Return the two DataFrames separately.
    return monthly_by_station.round(4), monthly_overall.round(4)


# Usage example
if __name__ == "__main__":
    # Set the directory paths.
    input_folder = str(ROOT / 'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed/YR') # Replace this value with the path to your CSV directory.
    output_folder = str(ROOT / 'Result/Mean_F_data')  # Output folder

    # Make sure the input directory exists.
    if not os.path.exists(input_folder):
        print(f"Error: Folder {input_folder} does not exist!")
    else:
        # Process the CSV files.
        process_csv_files(input_folder, output_folder)

        print("\n" + "=" * 50)
        print("Output file structure:")
        print("=" * 50)

        # Display the output structure.
        for root, dirs, files in os.walk(output_folder):
            level = root.replace(output_folder, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                if file.endswith('.csv'):
                    print(f'{subindent}{file}')