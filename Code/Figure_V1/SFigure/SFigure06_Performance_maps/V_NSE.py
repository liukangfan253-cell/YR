# -*- coding: utf-8 -*-
# ==========================================
# Maps of validation metrics (NSE and MAE)
# White-to-dark-red palette, borderless legend and raised violin plots.
# ==========================================
from io import BytesIO
from pathlib import Path
import matplotlib

ROOT = Path(__file__).resolve().parents[4]

matplotlib.use('Agg')  # On Windows, comment out this line to preview the figure in a window.

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os
import fiona
import warnings

warnings.filterwarnings('ignore')

# ========================
# 1. Configuration
# ========================
# Use Arial throughout.
plt.rcParams['font.size'] = 13
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 1.1 File paths
GDB_PATH = str(ROOT / 'Data/GIS/1.gdb')
TARGET_LAYERS = ["Export_Output_1", "Export_Output_2", "Export_Output_3", "Export_Output_4", "Export_Output_5"]

# Evaluation report with station coordinates
EVAL_CSV_PATH = str(ROOT / 'Result/BiLstm_Att_V1/Reports/Evaluation_Details_Report.csv')

# Figure output directory
OUTPUT_DIR = str(ROOT / 'V/Figure_V1/SFigure/SFigure06_Performance_maps/Images/space_NSE')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1.2 Discrete bins and legend labels for the white-to-dark-red palette
# Colors from the Reds scale, from near-white to very dark red.
C_1 = '#FEE5D9'  # Very pale red, nearly white
C_2 = '#FCAE91'  # Light orange-red
C_3 = '#FB6A4A'  # Medium red
C_4 = '#DE2D26'  # Bright red
C_5 = '#A50F15'  # Dark red
C_6 = '#67000D'  # Very dark red

METRIC_CONFIGS = {
    "NSE": {
        # Six color bins for NSE
        "colors": [C_1, C_2, C_3, C_4, C_5, C_6],
        "bounds": [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, np.inf],
        "labels": ["< 0.0", "0.0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "> 0.8"]
    },
    "MAE": {
        # Five color bins for MAE
        "ylabel": r"MAE (mg L$^{-1}$)",
        "inset_left": 0.80,
        "colors": [C_1, C_2, C_3, C_4, C_5],
        "bounds": [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
        "labels": ["< 0.5", "0.5 - 1.0", "1.0 - 1.5", "1.5 - 2.0", "> 2.0"]
    }
}

# 1.3 Basemap and marker styles
LAYER_STYLES = [
    {"facecolor": "none", "edgecolor": "#326db6", "alpha": 0.9, "linewidth": 2.5, "linestyle": "-"},
    {"facecolor": "none", "edgecolor": "#478ecc", "alpha": 0.7, "linewidth": 1.0, "linestyle": "-"},
    {"facecolor": "none", "edgecolor": "#478ecc", "alpha": 0.7, "linewidth": 1.0, "linestyle": "-"},
    {"facecolor": "white", "edgecolor": "#75b5dc", "alpha": 0.8, "linewidth": 1.0},
    {"facecolor": "white", "edgecolor": "#c44438", "alpha": 0.9, "linewidth": 2.5, "linestyle": "-"}
]

MARKER_STYLE = 'o'
MARKER_SIZE = 180
MARKER_EDGECOLOR = 'black'  # Use black outlines so pale and white markers are visible.
MARKER_EDGEWIDTH = 1.0
FIGURE_WIDTH = 12
FIGURE_HEIGHT = 10
DPI = 300


# ========================
# 2. Data-loading utilities
# ========================
def load_multiple_gdb_layers(gdb_path, target_layers):
    layer_dict = {}
    if not os.path.exists(gdb_path):
        print(f"❌ 严重错误: 底图 GDB 文件不存在: {gdb_path}")
        return None

    try:
        layers_in_gdb = fiona.listlayers(gdb_path)
    except Exception as e:
        print(f"❌ 读取GDB要素列表失败：{e}")
        return None

    for layer_name in target_layers:
        if layer_name in layers_in_gdb:
            layer_dict[layer_name] = gpd.read_file(gdb_path, layer=layer_name)
    if not layer_dict and len(layers_in_gdb) > 0:
        for lname in layers_in_gdb[:3]:
            layer_dict[lname] = gpd.read_file(gdb_path, layer=lname)
    return layer_dict


def get_evaluation_data():
    """Extract validation metrics and retain the original distribution."""
    print("🔍 正在读取包含坐标的评估报告...")
    if not os.path.exists(EVAL_CSV_PATH): return pd.DataFrame()
    eval_df = pd.read_csv(EVAL_CSV_PATH)

    val_df = eval_df[eval_df['Dataset'] == 'Test(25)'].copy()
    val_df = val_df.rename(columns={'WQS_Lon': 'lon', 'WQS_Lat': 'lat'})

    # Keep all original values, including extremes.
    print(f"✅ 成功读取 {len(val_df)} 个站点的验证集结果！")
    return val_df


# ========================
# 3. Main workflow
# ========================
def main():
    print("🚀 开始批量生成验证集指标空间图 (白到深红渐变版)...")

    basin_layers = load_multiple_gdb_layers(GDB_PATH, TARGET_LAYERS)
    if not basin_layers: return

    first_layer_crs = list(basin_layers.values())[0].crs
    all_bounds = []
    for gdf in basin_layers.values():
        all_bounds.extend(gdf.total_bounds)
    x_min, y_min = min(all_bounds[0::4]), min(all_bounds[1::4])
    x_max, y_max = max(all_bounds[2::4]), max(all_bounds[3::4])

    merged_data = get_evaluation_data()
    if merged_data.empty: return

    for metric, config in METRIC_CONFIGS.items():
        print(f"\n🗺️ 正在绘制指标: {metric}")
        plot_df = merged_data[['lon', 'lat', metric]].dropna().copy()
        if plot_df.empty: continue

        values = plot_df[metric].values
        stations_gdf = gpd.GeoDataFrame(
            plot_df, geometry=gpd.points_from_xy(plot_df["lon"], plot_df["lat"]), crs="EPSG:4326"
        ).to_crs(first_layer_crs)

        # --------------------------------------------------
        # Assign colors using the specified bin boundaries.
        # --------------------------------------------------
        colors = config["colors"]
        bounds = config["bounds"]

        def get_color(val):
            for i in range(len(bounds) - 1):
                if bounds[i] <= val < bounds[i + 1]:
                    return colors[i]
            return colors[-1]

        # Assign a color to each station.
        point_colors = [get_color(v) for v in values]

        # ---------------- Start plotting ----------------
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Basemap
        for i, (layer_name, gdf) in enumerate(basin_layers.items()):
            style = LAYER_STYLES[i] if i < len(LAYER_STYLES) else {"facecolor": "white", "edgecolor": "black",
                                                                   "alpha": 0.5}
            gdf.plot(ax=ax, **style)

        # Station markers on the main map
        scatter = ax.scatter(
            stations_gdf.geometry.x, stations_gdf.geometry.y,
            c=point_colors, s=MARKER_SIZE, alpha=0.95,
            edgecolor=MARKER_EDGECOLOR, linewidth=MARKER_EDGEWIDTH,
            marker=MARKER_STYLE, zorder=4
        )

        # ==================================================
        # 1. Borderless marker legend in the upper left
        # ==================================================
        legend_handles = []
        for color, label in zip(colors, config["labels"]):
            # Filled circles with black outlines
            patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                               markersize=13, markeredgecolor='black', markeredgewidth=1.0)
            legend_handles.append(patch)

        # Omit the title, background and frame for a floating legend.
        ax.legend(handles=legend_handles, labels=config["labels"],
                  loc='upper left', bbox_to_anchor=(0.02, 0.98),
                  fontsize=15, frameon=False,
                  labelspacing=0.8, borderpad=0.8, handletextpad=0.6)

        # ==================================================
        # 2. Violin panel in the upper right, using the map palette
        # ==================================================
        ax_ins = ax.inset_axes([config.get("inset_left", 0.78), 0.65, 0.18, 0.32])
        ax_ins.set_facecolor('white')
        ax_ins.patch.set_alpha(0.85)

        # A. Draw the pale violin silhouette.
        parts = ax_ins.violinplot(values, vert=True, widths=0.6, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor('#E8E8E8')
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)
            pc.set_linewidth(1.2)
            pc.set_zorder(1)

        # B. Plot the original values using the same white-to-dark-red palette.
        np.random.seed(42)
        jitter = np.random.normal(1, 0.06, size=len(values))
        ax_ins.scatter(
            jitter, values,
            c=point_colors,
            s=35, alpha=0.85, edgecolor='black', linewidth=0.5, zorder=2
        )

        # C. Draw the central box-plot elements.
        quartile1, med, quartile3 = np.percentile(values, [25, 50, 75])
        whiskers_min, whiskers_max = values.min(), values.max()

        ax_ins.vlines(1, whiskers_min, whiskers_max, color='black', linestyle='-', lw=1.5, zorder=3)
        ax_ins.vlines(1, quartile1, quartile3, color='black', linestyle='-', lw=6, zorder=4)
        ax_ins.scatter(1, med, marker='o', color='white', s=45, zorder=5, edgecolor='black', linewidth=1.5)

        # Use limits tailored to the observed values.
        y_margin = max((whiskers_max - whiskers_min) * 0.05, 0.1)
        ax_ins.set_ylim(whiskers_min - y_margin, whiskers_max + y_margin)

        # Draw a full border around the violin panel.
        for spine in ax_ins.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.0)

        ax_ins.set_xticks([])
        # Use the metric name as the y-axis label.
        ax_ins.set_ylabel(config.get("ylabel", metric), fontsize=15, fontweight='bold', fontfamily='Arial')
        ax_ins.tick_params(axis='y', labelsize=15, labelfontfamily='Arial')
        plt.setp(ax_ins.get_yticklabels(), fontweight='bold')

        # ----------------------------------------
        # Style the main map.
        ax.set_xlabel("Longitude (°E)", fontsize=21, fontweight='bold', labelpad=10, fontfamily='Arial')
        ax.set_ylabel("Latitude (°N)", fontsize=21, fontweight='bold', labelpad=10, fontfamily='Arial')
        ax.tick_params(axis='both', labelsize=21, labelfontfamily='Arial')
        plt.setp(ax.get_xticklabels(), fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontweight='bold')
        ax.grid(False)

        ax.set_xlim(x_min - 0.6, x_max + 0.6)
        ax.set_ylim(y_min - 0.6, y_max + 0.6)

        plt.tight_layout()

        # Export figures.
        output_png = os.path.join(OUTPUT_DIR, f"Validation_Metric_{metric}.png")
        plt.savefig(output_png, bbox_inches='tight', dpi=DPI, facecolor='white', edgecolor='none')
        with BytesIO() as tiff_buffer:
            plt.savefig(tiff_buffer, format="tiff", bbox_inches='tight', dpi=600, facecolor='white', edgecolor='none', pil_kwargs={'compression': 'tiff_lzw'})
            Path(os.path.splitext(output_png)[0] + '.tiff').write_bytes(tiff_buffer.getbuffer())
        plt.close(fig)

        print(f"✅ 成功保存: {output_png} (N={len(values)})")

    print("\n🎉 空间分布终极定制出图全部完成！")


if __name__ == "__main__":
    main()
