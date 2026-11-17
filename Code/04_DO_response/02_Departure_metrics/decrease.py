# 02_Shock_Assessment.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

# ===================== 1. Input and output configuration =====================
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

DIR_REGISTRY = str(ROOT / 'Data/Event_F/01_Registry_Data/By_Site')
DIR_PRE_BASE = str(ROOT / 'Data/Event_F/02_Pre_Baseline_Data/By_Site')
DIR_EVENT_PERIOD = str(ROOT / 'Data/Event_F/03_Event_Period_Data/By_Site')
DIR_NON_EVENT = str(ROOT / 'Data/Event_F/05_Non_Event_Data/By_Site')

OUTPUT_FOLDER = str(ROOT / 'Result/Decrease')
MIN_PRE_BASELINE_VALID_DAYS = 3
# ============================================================================

EXPECTED_EVENT_TYPES = [
    'Flood_Only',
    'Heatwave_Only',
    'Direct_Compound',
    'H5F_Compound',
    'F5H_Compound'
]


# ===================== 2. Assign seasons =====================
def get_season(month):
    """
    Assign meteorological seasons.

    Spring spans March–May, summer June–August, autumn September–November and winter December–February.
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'


def main():
    print("🚀 [脚本2] 开始脱氧冲击评估（采用站点 × 季节非事件DO基线）...")

    out_overall_dir = os.path.join(OUTPUT_FOLDER, "Overall")
    out_site_dir = os.path.join(OUTPUT_FOLDER, "By_Site")

    os.makedirs(out_overall_dir, exist_ok=True)

    # ========================================================================
    # 1. Precompute non-event DO references for all four seasons at each station.
    # ========================================================================
    # Example structure
    #
    # site_season_do_dict = {
    #     "Station A": {
    #         "Spring": 9.2,
    #         "Summer": 7.5,
    #         "Autumn": 8.3,
    #         "Winter": 10.1
    #     }
    # }
    # ========================================================================

    site_season_do_dict = {}

    if os.path.exists(DIR_NON_EVENT):

        for file in os.listdir(DIR_NON_EVENT):

            if not file.endswith("_05_Non_Event.csv"):
                continue

            site_name = file.replace("_05_Non_Event.csv", "")

            file_path = os.path.join(DIR_NON_EVENT, file)

            try:
                df_non = pd.read_csv(file_path, encoding='utf-8')
            except:
                df_non = pd.read_csv(file_path, encoding='gbk')

            if df_non.empty:
                continue

            if 'Date' not in df_non.columns or 'DO' not in df_non.columns:
                continue

            # Parse dates.
            df_non['Date'] = pd.to_datetime(
                df_non['Date'],
                errors='coerce'
            )

            # Convert DO to numeric values.
            df_non['DO'] = pd.to_numeric(
                df_non['DO'],
                errors='coerce'
            )

            # Drop rows missing a date or DO.
            df_non = df_non.dropna(
                subset=['Date', 'DO']
            ).copy()

            if df_non.empty:
                continue

            # Assign seasons by month.
            df_non['Season'] = df_non['Date'].dt.month.apply(get_season)

            # ------------------------------------------------------------
            # Calculate the following quantity.
            # Mean non-event DO by station and season
            # ------------------------------------------------------------
            seasonal_mean = (
                df_non
                .groupby('Season')['DO']
                .mean()
                .to_dict()
            )

            site_season_do_dict[site_name] = seasonal_mean

    print(f"✅ 已建立 {len(site_season_do_dict)} 个站点的季节性非事件 DO 基线")

    # ========================================================================
    # 2. Calculate metrics for the five event types.
    # ========================================================================

    all_shock_results = []
    all_site_averaged_results = []

    for ev_type in EXPECTED_EVENT_TYPES:

        ev_reg_dir = os.path.join(
            DIR_REGISTRY,
            ev_type
        )

        if not os.path.exists(ev_reg_dir):
            continue

        out_ev_site_dir = os.path.join(
            out_site_dir,
            ev_type
        )

        os.makedirs(
            out_ev_site_dir,
            exist_ok=True
        )

        print(f"\n🔍 正在计算 [{ev_type}] 事件明细与站点均值...")

        ev_type_results = []

        # ====================================================================
        # 2.1 Iterate over stations.
        # ====================================================================

        for file in os.listdir(ev_reg_dir):

            if not file.endswith("_01_Registry.csv"):
                continue

            site = file.replace(
                f"_{ev_type}_01_Registry.csv",
                ""
            )

            path_reg = os.path.join(
                ev_reg_dir,
                file
            )

            path_pre = os.path.join(
                DIR_PRE_BASE,
                ev_type,
                f"{site}_{ev_type}_02_Pre_Baseline.csv"
            )

            path_ev = os.path.join(
                DIR_EVENT_PERIOD,
                ev_type,
                f"{site}_{ev_type}_03_Event_Period.csv"
            )

            # The antecedent reference is used for Shock-S. Calculate Shock-L even when the antecedent reference is missing.
            if not os.path.exists(path_ev):
                continue

            # ------------------------------------------------------------
            # Load data.
            # ------------------------------------------------------------

            reg_df = pd.read_csv(path_reg)
            if os.path.exists(path_pre):
                pre_df = pd.read_csv(path_pre)
            else:
                # If the antecedent reference file is unavailable, leave Shock-S missing and do not impute DO.
                pre_df = pd.DataFrame({
                    'Event_ID': pd.Series(dtype=reg_df['Event_ID'].dtype),
                    'DO': pd.Series(dtype='float64')
                })
            ev_df = pd.read_csv(path_ev)

            # Convert DO consistently to numeric form.
            if 'DO' in pre_df.columns:
                pre_df['DO'] = pd.to_numeric(
                    pre_df['DO'],
                    errors='coerce'
                )

            if 'DO' in ev_df.columns:
                ev_df['DO'] = pd.to_numeric(
                    ev_df['DO'],
                    errors='coerce'
                )

            # =================================================================
            # 2.2 Shock-S uses the non-event reference 9–4 days before onset.
            # Require at least three valid DO days. Otherwise, leave the short-term reference missing.
            # =================================================================

            baseline = (
                pre_df
                .groupby('Event_ID')['DO']
                .agg(['mean', 'count'])
                .reset_index()
                .rename(
                    columns={
                        'mean': 'Pre_Baseline_DO',
                        'count': 'Pre_Baseline_Valid_Days'
                    }
                )
            )
            baseline.loc[
                baseline['Pre_Baseline_Valid_Days'] < MIN_PRE_BASELINE_VALID_DAYS,
                'Pre_Baseline_DO'
            ] = np.nan

            # =================================================================
            # 2.3 Calculate mean DO during the event.
            # =================================================================

            event_do = (
                ev_df
                .groupby('Event_ID')['DO']
                .mean()
                .reset_index()
                .rename(
                    columns={
                        'DO': 'Event_DO'
                    }
                )
            )

            # =================================================================
            # 2.4 Match daily seasonal references.
            # Assign each event day its corresponding seasonal non-event DO reference.
            # =================================================================

            if 'Date' in ev_df.columns:

                ev_df['Date'] = pd.to_datetime(
                    ev_df['Date'],
                    errors='coerce'
                )

                # Find the season of each event day.
                ev_df['Season'] = (
                    ev_df['Date']
                    .dt.month
                    .apply(
                        lambda x: get_season(x)
                        if pd.notna(x)
                        else np.nan
                    )
                )

                # Get the station's four seasonal references.
                season_baseline_map = site_season_do_dict.get(
                    site,
                    {}
                )

                # ------------------------------------------------------------
                # Use the matching seasonal reference for each event day.
                #
                # Example
                # August 30 -> Summer reference
                # August 31 -> Summer reference
                # September 1 -> Autumn reference
                #
                # This also handles events that span multiple seasons.
                # ------------------------------------------------------------

                ev_df['Seasonal_Baseline_DO_Daily'] = (
                    ev_df['Season']
                    .map(season_baseline_map)
                )

                # ------------------------------------------------------------
                # Compute the seasonal reference for each event.
                #
                # For events within one season,
                # all days share the same reference.
                #
                # For events that span seasons,
                # average the references, weighting by the number of event days in each season.
                # ------------------------------------------------------------

                event_season_baseline = (
                    ev_df
                    .groupby('Event_ID')['Seasonal_Baseline_DO_Daily']
                    .mean()
                    .reset_index()
                    .rename(
                        columns={
                            'Seasonal_Baseline_DO_Daily':
                                'Seasonal_Baseline_DO'
                        }
                    )
                )

            else:

                # Event records are expected to include Date.
                # If Date is absent, return an empty reference so processing can continue.
                event_season_baseline = (
                    event_do[['Event_ID']]
                    .copy()
                )

                event_season_baseline[
                    'Seasonal_Baseline_DO'
                ] = np.nan

            # =================================================================
            # 2.5 Merge event metadata.
            # =================================================================

            res_df = (
                reg_df
                .merge(
                    baseline,
                    on='Event_ID',
                    how='left'
                )
                .merge(
                    event_do,
                    on='Event_ID',
                    how='left'
                )
                .merge(
                    event_season_baseline,
                    on='Event_ID',
                    how='left'
                )
            )

            # =================================================================
            # 2.6 Calculate the two DO response ratios.
            # =================================================================

            # ------------------------------------------------------------
            # Shock-S
            #
            # Event_DO divided by mean DO 9–4 days before onset
            #
            # Use the existing short-term reference calculation.
            # ------------------------------------------------------------

            res_df['Shock_Ratio_Pre'] = np.where(
                (
                    res_df['Pre_Baseline_DO'].notna()
                )
                & (
                    res_df['Pre_Baseline_DO'] > 0
                ),
                res_df['Event_DO']
                / res_df['Pre_Baseline_DO'],
                np.nan
            )

            # ------------------------------------------------------------
            # Shock-L
            #
            # Previous seasonal reference
            # Event_DO divided by the annual Non_Event_DO
            #
            # Current seasonal reference
            # Event_DO divided by the corresponding Seasonal_Baseline_DO
            # ------------------------------------------------------------

            res_df['Shock_Ratio_LongTerm'] = np.where(
                (
                    res_df['Seasonal_Baseline_DO'].notna()
                )
                & (
                    res_df['Seasonal_Baseline_DO'] > 0
                ),
                res_df['Event_DO']
                / res_df['Seasonal_Baseline_DO'],
                np.nan
            )

            # =================================================================
            # 2.7 Standardize output fields.
            # =================================================================

            core_cols = [
                'Event_ID',
                'WQS_name',
                'WQS_Lon',
                'WQS_Lat',
                'Region',
                'Event_Type',
                'Start_Date',
                'End_Date',
                'Total_Days',

                # Seasonal long-term reference
                'Seasonal_Baseline_DO',

                # Short-term antecedent reference
                'Pre_Baseline_DO',
                'Pre_Baseline_Valid_Days',

                # Event-period DO
                'Event_DO',

                # The two DO response ratios
                'Shock_Ratio_LongTerm',
                'Shock_Ratio_Pre'
            ]

            res_df = res_df[
                [
                    c for c in core_cols
                    if c in res_df.columns
                ]
            ]

            # =================================================================
            # 2.8 Export event records for one station.
            # =================================================================

            res_df.to_csv(
                os.path.join(
                    out_ev_site_dir,
                    f"{site}_{ev_type}_Shock.csv"
                ),
                index=False,
                encoding='utf-8-sig'
            )

            ev_type_results.append(
                res_df
            )

        # ====================================================================
        # 2.9 Summarize the current event type.
        # ====================================================================

        if ev_type_results:

            df_ev_all = pd.concat(
                ev_type_results,
                ignore_index=True
            )

            # ------------------------------------------------------------
            # Save all individual events of this type.
            # ------------------------------------------------------------

            df_ev_all.to_csv(
                os.path.join(
                    out_overall_dir,
                    f"{ev_type}_Shock_All_Events.csv"
                ),
                index=False,
                encoding='utf-8-sig'
            )

            all_shock_results.append(
                df_ev_all
            )

            # ------------------------------------------------------------
            # Summarize station-level means.
            # ------------------------------------------------------------

            numeric_cols = [
                'Total_Days',
                'Seasonal_Baseline_DO',
                'Pre_Baseline_DO',
                'Pre_Baseline_Valid_Days',
                'Event_DO',
                'Shock_Ratio_LongTerm',
                'Shock_Ratio_Pre'
            ]

            # 1. Means
            avg_df = (
                df_ev_all
                .groupby('WQS_name')[numeric_cols]
                .mean()
                .reset_index()
            )

            # 2. Event counts by station
            count_df = (
                df_ev_all
                .groupby('WQS_name')['Event_ID']
                .count()
                .reset_index()
                .rename(
                    columns={
                        'Event_ID': 'Event_Count'
                    }
                )
            )

            # 3. Static metadata
            static_info = (
                df_ev_all[
                    [
                        'WQS_name',
                        'WQS_Lon',
                        'WQS_Lat',
                        'Region',
                        'Event_Type'
                    ]
                ]
                .drop_duplicates(
                    subset=['WQS_name']
                )
            )

            # 4. Merge the summaries.
            site_avg_df = (
                static_info
                .merge(
                    count_df,
                    on='WQS_name'
                )
                .merge(
                    avg_df,
                    on='WQS_name'
                )
            )

            # Round results to three decimal places.
            for col in numeric_cols:
                site_avg_df[col] = (
                    site_avg_df[col]
                    .round(3)
                )

            # Save station-level means.
            site_avg_df.to_csv(
                os.path.join(
                    out_overall_dir,
                    f"{ev_type}_Shock_Site_Averaged.csv"
                ),
                index=False,
                encoding='utf-8-sig'
            )

            all_site_averaged_results.append(
                site_avg_df
            )

    # ========================================================================
    # 3. Combine the final tables.
    # ========================================================================

    if all_shock_results:

        # ------------------------------------------------------------
        # 3.1 Complete table of individual events
        # ------------------------------------------------------------

        pd.concat(
            all_shock_results,
            ignore_index=True
        ).to_csv(
            os.path.join(
                out_overall_dir,
                "FINAL_Shock_All_Events_Combined.csv"
            ),
            index=False,
            encoding='utf-8-sig'
        )

        # ------------------------------------------------------------
        # 3.2 Complete table of station means
        # ------------------------------------------------------------

        pd.concat(
            all_site_averaged_results,
            ignore_index=True
        ).to_csv(
            os.path.join(
                out_overall_dir,
                "FINAL_Shock_Site_Averaged_Combined.csv"
            ),
            index=False,
            encoding='utf-8-sig'
        )

        print(
            "\n✅ [脚本2] 计算与均值聚合完成！"
            "\n✅ Shock-L 已采用站点特异性的春、夏、秋、冬季节非事件 DO 基线。"
            "\n✅ Shock-S 仍采用事件前 9–4 d DO 基线。"
        )


if __name__ == "__main__":
    main()
