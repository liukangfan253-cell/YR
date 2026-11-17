# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
import warnings

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

warnings.filterwarnings('ignore')

# ===================== 1. Paths =====================
# Input folder for labeled station tables created in the previous step
INPUT_FOLDER = str(ROOT / 'Data/Event_N/By_Site_Labeled')
# Output folder for the final LSTM feature tables
OUTPUT_FOLDER = str(ROOT / 'Data/LSTM_Y')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== 2. Lookup tables =====================
# Encode seasons as spring (March–May) = 1, summer (June–August) = 2, autumn (September–November) = 3, and winter (December–February) = 4.
SEASON_MAP = {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 4, 1: 4, 2: 4}

# Encode reaches as upstream = 1, midstream = 2, and downstream = 3.
REGION_MAP = {'UR': 1, 'MR': 2, 'LR': 3}


# ===================== 3. Main workflow =====================
def main():
    # Find all station CSV files.
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    if not csv_files:
        print(f"❌ 未找到输入文件，请检查路径: {INPUT_FOLDER}")
        return

    print(f"🚀 开始为 LSTM 预处理特征工程，共发现 {len(csv_files)} 个站点文件...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(file_path, encoding='gbk')

        # ---------------------------------------------------
        # Step 1. Derive calendar features (month, season and dayofyear).
        # ---------------------------------------------------
        df['Date'] = pd.to_datetime(df['Date'])
        # Sort chronologically before calculating rolling-window features.
        df = df.sort_values('Date').reset_index(drop=True)

        df['month'] = df['Date'].dt.month
        df['dayofyear'] = df['Date'].dt.dayofyear
        df['season'] = df['month'].map(SEASON_MAP)

        # ---------------------------------------------------
        # Step 2. Encode spatial features (Region_Code).
        # ---------------------------------------------------
        if 'Region' in df.columns:
            # Use 2 (midstream) when there is no matching region.
            df['Region_Code'] = df['Region'].map(REGION_MAP).fillna(2).astype(int)
        else:
            df['Region_Code'] = 2

            # ---------------------------------------------------
        # Step 3. Calculate rolling-window features (TP_M_sum3 and Q_mean7).
        # ---------------------------------------------------
        # Create missing input columns with zero values so rolling calculations can proceed.
        if 'TP_M' not in df.columns:
            df['TP_M'] = 0.0
        if 'Q' not in df.columns:
            df['Q'] = np.nan

        # Three-day cumulative precipitation, with min_periods=1 so the first two days have values.
        df['TP_M_sum3'] = df['TP_M'].rolling(window=3, min_periods=1).sum()

        # Seven-day mean discharge
        df['Q_mean7'] = df['Q'].rolling(window=7, min_periods=1).mean()

        # ---------------------------------------------------
        # Step 4. Arrange columns and export.
        # ---------------------------------------------------
        # Move the new columns forward so they are easy to check.
        cols = list(df.columns)
        new_cols = ['Date', 'Event_Label', 'WQS_name', 'Region', 'Region_Code',
                    'month', 'season', 'dayofyear', 'TP_M', 'TP_M_sum3', 'Q', 'Q_mean7']

        # Append the remaining columns.
        remaining_cols = [c for c in cols if c not in new_cols]
        final_cols = new_cols + remaining_cols

        df = df[final_cols]

        # Write the feature table to the new output folder.
        out_path = os.path.join(OUTPUT_FOLDER, file_name)
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"  -> ✅ 成功追加特征并导出: {file_name}")

    print(f"\n🎉 所有站点的数据预处理完成！")
    print(f"📂 LSTM 最终输入文件已存至: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()