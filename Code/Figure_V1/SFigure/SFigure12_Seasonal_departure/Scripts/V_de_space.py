# -*- coding: utf-8 -*-
"""Map DO response ratios relative to the seasonal reference (Shock-L)."""
from io import BytesIO
from pathlib import Path
import fiona
import geopandas as gpd
import matplotlib

ROOT = Path(__file__).resolve().parents[5]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# ============================== User configuration ==============================
GDB_PATH = Path(str(ROOT / 'Data/GIS/1.gdb'))
BASE_CSV_DIR = Path(str(ROOT / 'Result/Decrease/Overall'))
OUTPUT_DIR = Path(str(ROOT / 'V/Figure_V1/SFigure/SFigure12_Seasonal_departure/Images'))
TARGET_LAYERS = ["Export_Output_1", "Export_Output_2", "Export_Output_3", "Export_Output_4", "Export_Output_5"]
SITE_COLUMN_MAP = {"lon": "WQS_Lon", "lat": "WQS_Lat", "value": "Shock_Ratio_LongTerm"}
CUSTOM_COLORBAR_LABEL = "Seasonal-relative DO ratio\n(Shock-L)"
X_AXIS_LABEL, Y_AXIS_LABEL = "Longitude (°E)", "Latitude (°N)"
FONT_FAMILY, AXIS_LABEL_SIZE, TICK_SIZE = "Arial", 23, 23
FIGURE_SIZE, OUTPUT_DPI, OUTPUT_PREFIX = (12, 10), 300, "Spatial_L"
EVENT_FILES = {"DC": "Direct_Compound_Shock_Site_Averaged.csv", "F5H": "F5H_Compound_Shock_Site_Averaged.csv", "H5F": "H5F_Compound_Shock_Site_Averaged.csv", "Heatwave": "Heatwave_Only_Shock_Site_Averaged.csv", "Flood": "Flood_Only_Shock_Site_Averaged.csv"}
LAYER_STYLES = [{"facecolor": "none", "edgecolor": "#326db6", "alpha": .9, "linewidth": 2.5}, {"facecolor": "none", "edgecolor": "#478ecc", "alpha": .7, "linewidth": 1}, {"facecolor": "none", "edgecolor": "#478ecc", "alpha": .7, "linewidth": 1}, {"facecolor": "white", "edgecolor": "#75b5dc", "alpha": .8, "linewidth": 1}, {"facecolor": "white", "edgecolor": "#c44438", "alpha": .9, "linewidth": 2.5}]
MARKER_STYLE, MIN_MARKER_SIZE, MAX_MARKER_SIZE = "o", 100, 350
MARKER_EDGECOLOR, MARKER_EDGEWIDTH, COLORMAP, MAP_MARGIN = "black", .8, "RdYlBu_r", .6
# ======================================================================


def configure_font(font_family):
    plt.rcParams.update({"font.family": font_family,
                         "font.sans-serif": [font_family],
                         "axes.unicode_minus": False})


def load_layers(gdb_path, target_layers):
    available = set(fiona.listlayers(gdb_path))
    layers = {}
    for name in target_layers:
        if name not in available:
            print(f"GDB 中不存在图层 {name}，已跳过。")
            continue
        layers[name] = gpd.read_file(gdb_path, layer=name)
    if not layers:
        raise ValueError("未读取到任何 GDB 图层。")
    return layers


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def load_datasets(config):
    datasets = {}
    columns = config["site_column_map"]
    required = [columns["lon"], columns["lat"], columns["value"]]
    for event_name, filename in config["event_files"].items():
        path = Path(config["base_csv_dir"]) / filename
        if not path.is_file():
            print(f"文件不存在，已跳过: {path}")
            continue
        df = read_csv_fallback(path)
        missing = set(required).difference(df.columns)
        if missing:
            print(f"{filename} 缺少列 {sorted(missing)}，已跳过。")
            continue
        data = df[required].copy().dropna()
        data.columns = ["lon", "lat", "value"]
        if not data.empty:
            datasets[event_name] = data
    if not datasets:
        raise ValueError("没有提取到任何有效站点数据。")
    return datasets


def apply_bold_ticks(axis, font_family, font_size):
    axis.set_tick_params(labelsize=font_size)
    for label in axis.get_ticklabels():
        label.set_fontfamily(font_family)
        label.set_fontweight("bold")


def run_spatial(config):
    config = dict(config)
    config["dpi"] = config["output_dpi"]
    config["colorbar_label"] = config["custom_colorbar_label"]
    configure_font(config["font_family"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = load_layers(config["gdb_path"], config["target_layers"])
    datasets = load_datasets(config)
    crs = next(iter(layers.values())).crs
    bounds = np.vstack([layer.total_bounds for layer in layers.values()])
    x_min, y_min = bounds[:, :2].min(axis=0)
    x_max, y_max = bounds[:, 2:].max(axis=0)
    all_values = np.concatenate([data["value"].to_numpy() for data in datasets.values()])
    value_min, value_max = all_values.min(), all_values.max()
    # Shared symmetric color limits and area scale across all five event types.
    max_deviation = float(np.max(np.abs(all_values - 1.0)))
    scale_deviation = max_deviation if max_deviation > 0 else 1.0
    norm = TwoSlopeNorm(vmin=1.0 - scale_deviation, vcenter=1.0,
                        vmax=1.0 + scale_deviation)

    for event_name, data in datasets.items():
        stations = gpd.GeoDataFrame(data,
            geometry=gpd.points_from_xy(data["lon"], data["lat"]),
            crs="EPSG:4326").to_crs(crs)
        values = stations["value"].to_numpy()
        fig, ax = plt.subplots(figsize=config["figure_size"], dpi=config["dpi"])
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        for index, layer in enumerate(layers.values()):
            style = config["layer_styles"][index] if index < len(config["layer_styles"]) else {
                "facecolor": "white", "edgecolor": "black", "alpha": 0.5}
            layer.plot(ax=ax, **style)
        sizes = config["min_marker_size"] + (
            config["max_marker_size"] - config["min_marker_size"]) * (np.abs(values - 1.0) / scale_deviation)
        scatter = ax.scatter(stations.geometry.x, stations.geometry.y, c=values,
            cmap=config["colormap"], s=sizes, alpha=0.85,
            edgecolor=config["marker_edgecolor"], linewidth=config["marker_edgewidth"],
            marker=config["marker_style"], norm=norm)
        # Leave room between the map and the left-side colorbar label.
        colorbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)
        colorbar.ax.yaxis.set_label_position('left')
        colorbar.ax.yaxis.set_ticks_position('right')
        colorbar.set_label(f'{config["colorbar_label"]} ({event_name})',
                           fontsize=config["axis_label_size"], fontweight="bold",
                           fontfamily=config["font_family"], labelpad=12)
        colorbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 5))
        apply_bold_ticks(colorbar.ax.yaxis, config["font_family"], config["tick_size"])
        ax.set_xlabel(config["x_axis_label"], fontsize=config["axis_label_size"],
                      fontweight="bold", fontfamily=config["font_family"], labelpad=10)
        ax.set_ylabel(config["y_axis_label"], fontsize=config["axis_label_size"],
                      fontweight="bold", fontfamily=config["font_family"], labelpad=10)
        apply_bold_ticks(ax.xaxis, config["font_family"], config["tick_size"])
        apply_bold_ticks(ax.yaxis, config["font_family"], config["tick_size"])
        margin = config["map_margin"]
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(False)
        fig.tight_layout()
        stem = f'{config["output_prefix"]}_{event_name}'
        png_path, pdf_path = output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"
        save_options = {"bbox_inches": "tight", "facecolor": "white", "edgecolor": "none"}
        fig.savefig(png_path, dpi=config["dpi"], **save_options)
        with BytesIO() as tiff_buffer:
            fig.savefig(tiff_buffer, format="tiff", dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, **save_options)
            Path(png_path.with_suffix('.tiff')).write_bytes(tiff_buffer.getbuffer())
        fig.savefig(pdf_path, **save_options)
        plt.close(fig)
        print(f"已保存: {png_path} (N={len(values)}, 均值={values.mean():.3f})")


def main():
    run_spatial({name.lower(): value for name, value in globals().copy().items() if name.isupper()})
if __name__ == "__main__":
    main()
