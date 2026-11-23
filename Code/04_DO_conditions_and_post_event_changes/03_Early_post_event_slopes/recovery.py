# 03_Recovery_Tracking.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from scipy.stats import linregress


# ===================== 1. Input and output configuration =====================
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

DIR_REGISTRY = str(ROOT / 'Data/Event_F/01_Registry_Data/By_Site')
DIR_PRE_BASE = str(ROOT / 'Data/Event_F/02_Pre_Baseline_Data/By_Site')
DIR_POST_RECOVERY = str(ROOT / 'Data/Event_F/04_Post_Recovery_Data/By_Site')

OUTPUT_FOLDER = str(ROOT / 'Result/Recovery')
MIN_PRE_BASELINE_VALID_DAYS = 3
# ============================================================================

EXPECTED_EVENT_TYPES = [
    'Flood_Only',
    'Heatwave_Only',
    'Direct_Compound',
    'H5F_Compound',
    'F5H_Compound'
]


def early_do_slope(post_frame, target_days=(1, 3, 5, 7, 9)):
    """Original five-date slope, reusable after rebuilding the post-event window.

    The caller must truncate at the next event. Return the unrounded slope;
    the existing output convention rounds it to four decimals.
    """
    values = []
    for day in target_days:
        selected = post_frame.loc[post_frame['Day_Offset'].eq(day), 'DO'].values
        if not len(selected) or pd.isna(selected[0]):
            return np.nan
        values.append(selected[0])
    # Keep scipy's numpy scalar: Python float round() can choose the other side
    # of a four-decimal half-tie compared with the original numpy-scalar round().
    return linregress(target_days, values).slope


def main():

    print(
        "🚀 [脚本3] 开始灾后恢复追踪"
        "（仅保留第1、3、5、7、9天完整的事件）..."
    )

    out_overall_dir = os.path.join(
        OUTPUT_FOLDER,
        "Overall"
    )

    out_site_dir = os.path.join(
        OUTPUT_FOLDER,
        "By_Site"
    )

    os.makedirs(
        out_overall_dir,
        exist_ok=True
    )

    all_recovery_results = []
    all_site_averaged_results = []

    # ================== 1. Match events by type and station ==================
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

        print(
            f"\n🔍 正在追踪 [{ev_type}] "
            f"事件明细与站点均值..."
        )

        ev_type_results = []

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

            path_post = os.path.join(
                DIR_POST_RECOVERY,
                ev_type,
                f"{site}_{ev_type}_04_Post_Recovery.csv"
            )

            # The slope depends only on post-event DO. Compute it even if the antecedent reference file is missing.
            if not os.path.exists(path_post):
                continue

            reg_df = pd.read_csv(path_reg)
            if os.path.exists(path_pre):
                pre_df = pd.read_csv(path_pre)
            else:
                # Treat an empty reference as missing. Oxygen deficit and time to return to the reference remain NaN.
                pre_df = pd.DataFrame({
                    'Event_ID': pd.Series(dtype=reg_df['Event_ID'].dtype),
                    'DO': pd.Series(dtype='float64')
                })
            post_df = pd.read_csv(path_post)

            pre_df['DO'] = pd.to_numeric(pre_df['DO'], errors='coerce')

            # Calculate non-event reference DO 9–4 days before onset, using at least three valid DO days.
            baseline = (
                pre_df
                .groupby('Event_ID')['DO']
                .agg(['mean', 'count'])
                .reset_index()
                .rename(
                    columns={
                        'mean': 'Target_Baseline_DO',
                        'count': 'Pre_Baseline_Valid_Days'
                    }
                )
            )
            baseline.loc[
                baseline['Pre_Baseline_Valid_Days'] < MIN_PRE_BASELINE_VALID_DAYS,
                'Target_Baseline_DO'
            ] = np.nan

            recovery_records = []

            # Use days 1, 3, 5, 7 and 9 after the event for the recovery slope.
            target_days = [1, 3, 5, 7, 9]

            for _, row in reg_df.iterrows():

                ev_id = row['Event_ID']

                base_val = baseline.loc[
                    baseline['Event_ID'] == ev_id,
                    'Target_Baseline_DO'
                ].values

                valid_days_val = baseline.loc[
                    baseline['Event_ID'] == ev_id,
                    'Pre_Baseline_Valid_Days'
                ].values

                base_do = (
                    base_val[0]
                    if len(base_val) > 0
                    else np.nan
                )

                pre_baseline_valid_days = (
                    int(valid_days_val[0])
                    if len(valid_days_val) > 0
                    else 0
                )

                ev_post = post_df[
                    post_df['Event_ID'] == ev_id
                ]

                do_dict = {}
                rec_x = []
                rec_y = []

                # Extract DO on days 1, 3, 5, 7 and 9 after the event.
                for d in target_days:

                    val = ev_post.loc[
                        ev_post['Day_Offset'] == d,
                        'DO'
                    ].values

                    v = (
                        val[0]
                        if len(val) > 0
                        else np.nan
                    )

                    do_dict[f'Post_{d}d_DO'] = (
                        round(v, 3)
                        if not pd.isna(v)
                        else np.nan
                    )

                    if not pd.isna(v):
                        rec_x.append(d)
                        rec_y.append(v)

                # All five days must have valid DO.
                # Skip the event if any of the five days is missing.
                if len(rec_x) < len(target_days):
                    continue

                # Calculate the early post-event DO slope.
                slope = early_do_slope(ev_post, target_days)

                # ==================================================
                # Compute the initial oxygen deficit one day after the event.
                # Delta_DO1 = antecedent reference DO - DO on the first post-event day.
                # ==================================================
                post_1d_do = rec_y[0]

                delta_do1 = (
                    base_do - post_1d_do
                    if not pd.isna(base_do)
                    else np.nan
                )

                # Use the original time-to-recovery calculation.
                relax = np.nan

                if not pd.isna(base_do):

                    for _, pr in (
                        ev_post
                        .sort_values('Day_Offset')
                        .iterrows()
                    ):

                        if (
                            pd.notna(pr['DO'])
                            and pr['DO'] >= base_do
                        ):

                            relax = pr['Day_Offset']
                            break

                recovery_records.append({
                    'Event_ID': ev_id,

                    'Target_Baseline_DO': (
                        round(base_do, 3)
                        if not pd.isna(base_do)
                        else np.nan
                    ),

                    'Pre_Baseline_Valid_Days': pre_baseline_valid_days,

                    **do_dict,

                    'Delta_DO1': (
                        round(delta_do1, 3)
                        if not pd.isna(delta_do1)
                        else np.nan
                    ),

                    'Recovery_Slope': round(
                        slope,
                        4
                    ),

                    'Relaxation_Time_Days': relax
                })

            # No events at this station have all required time points.
            if not recovery_records:
                continue

            rec_df = pd.DataFrame(
                recovery_records
            )

            # Keep only events with all required observations.
            res_df = reg_df.merge(
                rec_df,
                on='Event_ID',
                how='inner'
            )

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
                'Target_Baseline_DO',
                'Pre_Baseline_Valid_Days',
                'Post_1d_DO',
                'Delta_DO1',
                'Post_3d_DO',
                'Post_5d_DO',
                'Post_7d_DO',
                'Post_9d_DO',
                'Recovery_Slope',
                'Relaxation_Time_Days'
            ]

            res_df = res_df[
                [
                    c for c in core_cols
                    if c in res_df.columns
                ]
            ]

            # Save recovery results for the station.
            res_df.to_csv(
                os.path.join(
                    out_ev_site_dir,
                    f"{site}_{ev_type}_Recovery.csv"
                ),
                index=False,
                encoding='utf-8-sig'
            )

            ev_type_results.append(
                res_df
            )

        # ================== 2. Summarize by event type ==================
        if ev_type_results:

            df_ev_all = pd.concat(
                ev_type_results,
                ignore_index=True
            )

            # Save individual events by type.
            df_ev_all.to_csv(
                os.path.join(
                    out_overall_dir,
                    f"{ev_type}_Recovery_All_Events.csv"
                ),
                index=False,
                encoding='utf-8-sig'
            )

            all_recovery_results.append(
                df_ev_all
            )

            # Calculate and save station means by event type.
            numeric_cols = [
                'Total_Days',
                'Target_Baseline_DO',
                'Pre_Baseline_Valid_Days',
                'Post_1d_DO',
                'Delta_DO1',
                'Post_3d_DO',
                'Post_5d_DO',
                'Post_7d_DO',
                'Post_9d_DO',
                'Recovery_Slope',
                'Relaxation_Time_Days'
            ]

            avg_df = (
                df_ev_all
                .groupby('WQS_name')[numeric_cols]
                .mean()
                .reset_index()
            )

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

            for col in numeric_cols:

                site_avg_df[col] = (
                    site_avg_df[col]
                    .round(3)
                )

            site_avg_df.to_csv(
                os.path.join(
                    out_overall_dir,
                    f"{ev_type}_Recovery_Site_Averaged.csv"
                ),
                index=False,
                encoding='utf-8-sig'
            )

            all_site_averaged_results.append(
                site_avg_df
            )

    # ================== 3. Combine the final tables ==================
    if all_recovery_results:

        pd.concat(
            all_recovery_results,
            ignore_index=True
        ).to_csv(
            os.path.join(
                out_overall_dir,
                "FINAL_Recovery_All_Events_Combined.csv"
            ),
            index=False,
            encoding='utf-8-sig'
        )

        pd.concat(
            all_site_averaged_results,
            ignore_index=True
        ).to_csv(
            os.path.join(
                out_overall_dir,
                "FINAL_Recovery_Site_Averaged_Combined.csv"
            ),
            index=False,
            encoding='utf-8-sig'
        )

        print(
            f"\n✅ [脚本3] 所有站点恢复追踪及均值聚合完成！"
            f"\n仅保留第1、3、5、7、9天DO完整的事件。"
            f"\n结果已存入：{OUTPUT_FOLDER}"
        )


if __name__ == "__main__":
    main()
