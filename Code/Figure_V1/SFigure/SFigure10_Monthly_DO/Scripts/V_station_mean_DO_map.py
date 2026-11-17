# ==========================================
# 05_Spatial_Distribution_Single_Mapping.py
# ==========================================
import argparse
from io import BytesIO
from pathlib import Path
import sys
import matplotlib

ROOT = Path(__file__).resolve().parents[5]

matplotlib.use('Agg')  # On Windows, comment out this line to preview the figure in a window.

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import fiona
import numpy as np
from matplotlib.colors import Normalize

# ========================
# Main configuration
# ========================
# 1. Use Arial throughout.
plt.rcParams['font.size'] = 13
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False  # Use the standard minus sign.
plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})

# 2. File paths
GDB_PATH = str(ROOT / 'Data/GIS/1.gdb')
TARGET_LAYERS = ["Export_Output_1", "Export_Output_2", "Export_Output_3", "Export_Output_4", "Export_Output_5"]

# Specify the full path to the input CSV.
INPUT_CSV_PATH = str(ROOT / 'Result/Mean_F_data/DO/DO_overall_stats.csv')

# Figure output directory
OUTPUT_DIR = str(ROOT / 'V/Figure_V1/SFigure/SFigure10_Monthly_DO/Images')

# Suffix for the output filename, as in Spatial_DO_Mean_Overall.png
OUTPUT_FILENAME_SUFFIX = "DO_Mean_Overall"

# 3. Station columns and mean values
SITE_COLUMN_MAP = {
    "lon": "WQS_Lon",
    "lat": "WQS_Lat",
    "value": "Mean"  # Metric to extract
}

# 4. Custom colorbar label
CUSTOM_COLORBAR_LABEL = r"Mean DO (mg L$^{-1}$)"

# 5. Feature-layer styles for a white or transparent basemap
LAYER_STYLE = {"facecolor": "none", "edgecolor": "#9AABB5", "linewidth": 0.75, "zorder": 1}

# 6. Station marker styles
MARKER_STYLE = 'o'
MIN_MARKER_SIZE = 40
MAX_MARKER_SIZE = 205
MARKER_EDGECOLOR = 'black'
MARKER_EDGEWIDTH = 0.8
COLORMAP = 'viridis'  # Match the color scale of the existing SFigure10a.
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 6.6
DPI = 300


# ========================
# Utilities
# ========================
def load_multiple_gdb_layers(gdb_path, target_layers):
    layer_dict = {}
    try:
        layers_in_gdb = fiona.listlayers(gdb_path)
    except Exception as e:
        print(f"❌ 读取GDB要素列表失败：{e}")
        return None

    for layer_name in target_layers:
        if layer_name in layers_in_gdb:
            try:
                gdf = gpd.read_file(gdb_path, layer=layer_name)
                layer_dict[layer_name] = gdf
                print(f"✅ 成功读取要素类：{layer_name}（共{len(gdf)}个要素）")
            except Exception as e:
                print(f"❌ 读取要素类 {layer_name} 失败：{e}")
        else:
            print(f"⚠️ 警告: GDB 中不存在图层 {layer_name}")
    return layer_dict


# ========================
# Load and plot data from one CSV.
# ========================
def main():
    global INPUT_CSV_PATH, OUTPUT_DIR, GDB_PATH
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='SFigure10a: station mean DO map from saved V5 summaries.')
    parser.add_argument('--input-csv', default=INPUT_CSV_PATH)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--gdb', default=GDB_PATH)
    args = parser.parse_args()
    INPUT_CSV_PATH, OUTPUT_DIR, GDB_PATH = args.input_csv, args.output_dir, args.gdb
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("🚀 开始读取 CSV 文件并生成空间分布图...")

    # 1. Check that the CSV exists.
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 找不到文件，请检查路径是否正确: {INPUT_CSV_PATH}")
        return

    # 2. Read the geodatabase basemap.
    basin_layers = load_multiple_gdb_layers(GDB_PATH, TARGET_LAYERS)
    if not basin_layers:
        print("❌ 未读取到任何GDB要素类，退出程序")
        return

    first_layer_crs = list(basin_layers.values())[0].crs

    # Calculate the basemap bounds before plotting to set an appropriate view.
    all_bounds = []
    for gdf in basin_layers.values():
        all_bounds.extend(gdf.total_bounds)
    x_min, y_min = min(all_bounds[0::4]), min(all_bounds[1::4])
    x_max, y_max = max(all_bounds[2::4]), max(all_bounds[3::4])

    # 3. Read and clean CSV data.
    try:
        try:
            df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8-sig")
        except:
            df = pd.read_csv(INPUT_CSV_PATH, encoding="gbk")

        # Remove the All_Stations summary row.
        if 'WQS_name' in df.columns:
            df = df[df['WQS_name'] != 'All_Stations']

        # Check the required columns.
        required_cols = [SITE_COLUMN_MAP['lon'], SITE_COLUMN_MAP['lat'], SITE_COLUMN_MAP['value']]
        if not all(col in df.columns for col in required_cols):
            print(f"❌ 数据表中缺少必要的列: {required_cols}")
            return

        # Extract the values to plot.
        data = df[required_cols].copy().dropna()
        if len(data) != len(df):
            print(f"站点记录 {len(df)} 条，坐标与均值齐全 {len(data)} 条。")
        data.columns = ["lon", "lat", "value"]

        if len(data) == 0:
            print("❌ 数据表为空或全为缺失值，无法绘图。")
            return

    except Exception as e:
        print(f"❌ 读取或处理 CSV 文件失败: {e}")
        return

    # Use the global minimum and maximum to set a common color scale.
    v_min = data["value"].min()
    v_max = data["value"].max()
    print(f"📊 数据读取成功！共 {len(data)} 个站点。指标最小值为: {v_min:.4f}, 最大值为: {v_max:.4f}")

    global_norm = Normalize(vmin=v_min, vmax=v_max)

    # ========================================================
    # 4. Plot the map.
    # ========================================================
    print(f"🗺️ 正在生成地图...")
    stations_gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["lon"], data["lat"]),
        crs="EPSG:4326"
    ).to_crs(first_layer_crs)

    values = stations_gdf["value"].values

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw the GIS basemap.
    for gdf in basin_layers.values():
        gdf.plot(ax=ax, **LAYER_STYLE)

    # Map values to marker sizes and colors.
    sizes = MIN_MARKER_SIZE + (MAX_MARKER_SIZE - MIN_MARKER_SIZE) * global_norm(values)
    scatter = ax.scatter(
        stations_gdf.geometry.x,
        stations_gdf.geometry.y,
        c=values,
        cmap=COLORMAP,
        s=sizes,
        alpha=0.85,
        edgecolor=MARKER_EDGECOLOR,
        linewidth=MARKER_EDGEWIDTH,
        marker=MARKER_STYLE,
        zorder=3,
        vmin=v_min,
        vmax=v_max
    )

    # Configure the colorbar.
    colorbar_axis = ax.inset_axes([1.06, 0.02, 0.026, 0.96])
    cbar = fig.colorbar(scatter, cax=colorbar_axis)
    cbar.set_label(f'{CUSTOM_COLORBAR_LABEL}', fontsize=21, fontweight='bold',
                   fontfamily='Arial', labelpad=12)
    cbar.ax.tick_params(labelsize=21, labelfontfamily='Arial')
    plt.setp(cbar.ax.get_yticklabels(), fontweight='bold')

    # Style the map with coordinate axes.
    ax.set_xlabel("Longitude (°E)", fontsize=21, fontweight='bold', labelpad=10, fontfamily='Arial')
    ax.set_ylabel("Latitude (°N)", fontsize=21, fontweight='bold', labelpad=10, fontfamily='Arial')
    ax.tick_params(axis='both', labelsize=21, labelfontfamily='Arial')
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.grid(False)

    # Set the map bounds.
    ax.set_xlim(x_min - 1.0, x_max + 1.0)
    ax.set_ylim(y_min - 0.6, y_max + 0.6)

    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.14, top=0.96)

    # Export figures.
    output_png = os.path.join(OUTPUT_DIR, f"Spatial_{OUTPUT_FILENAME_SUFFIX}.png")
    output_pdf = os.path.join(OUTPUT_DIR, f"Spatial_{OUTPUT_FILENAME_SUFFIX}.pdf")

    plt.savefig(output_png, bbox_inches='tight', dpi=DPI, facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(Path(output_pdf).with_suffix('.svg'), bbox_inches='tight', facecolor='white', edgecolor='none')
    with BytesIO() as buffer:
        fig.savefig(buffer, format='tiff', bbox_inches='tight', dpi=600,
                    pil_kwargs={'compression': 'tiff_lzw'})
        Path(output_png).with_suffix('.tiff').write_bytes(buffer.getbuffer())
    plt.close(fig)

    print(f"✅ 成功保存图片至: {output_png}")
    print("🎉 制图任务完成！")


if __name__ == "__main__":
    main()
