# -*- coding: utf-8 -*-
"""Plot box plots and points for event counts by region and event type."""

from io import BytesIO
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[5]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ============================== User configuration ==============================
INPUT_CSV = Path(str(ROOT / 'Data/Event_N/Site_Event_Counts.csv'))
OUTPUT_DIR = Path(str(ROOT / 'V/Figure_V1/SFigure/SFigure02_Event_frequency/Images'))
OUTPUT_BASENAME = "Event_Boxplot_Scatter_Optimized"

REGION_ORDER = ["UR", "MR", "LR"]
CUSTOM_X_LABELS = ["UR", "MR", "LR"]

FONT_FAMILY = "Arial"
AXIS_LABEL_SIZE = 20
X_TICK_SIZE = 20
Y_TICK_SIZE = 18
LEGEND_TEXT_SIZE = 14
LEGEND_TITLE_SIZE = 16

FIGURE_SIZE = (10, 6)
OUTPUT_DPI = 300
BOX_WIDTH = 0.6
POINT_SIZE = 7
POINT_JITTER = 0.2
PALETTE = {
    "Flood_Only": "#4A90E2",
    "Heatwave_Only": "#D0021B",
    "Compound_Event": "#F5A623",
}
# ======================================================================

EVENT_COLUMNS = list(PALETTE)
FRIENDLY_EVENT_LABELS = ["Flood Only", "Heatwave Only", "Compound Event"]


def configure_plot_style():
    """Apply consistent plot fonts and styling."""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY],
        "axes.unicode_minus": False,
    })


def load_and_prepare_data(input_csv):
    """Load the data, pool compound events and reshape the table to long format."""
    df = pd.read_csv(input_csv)
    required_columns = {"WQS_name", "Region"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"输入数据缺少必要列：{', '.join(sorted(missing_columns))}")

    df["Flood_Only"] = df.get("Flood_Only_Total_Count", 0)
    df["Heatwave_Only"] = df.get("Heatwave_Only_Total_Count", 0)
    df["Compound_Event"] = (
        df.get("Direct_Compound_Total_Count", 0)
        + df.get("H5F_Compound_Total_Count", 0)
        + df.get("F5H_Compound_Total_Count", 0)
    )
    long_df = df.melt(id_vars=["WQS_name", "Region"], value_vars=EVENT_COLUMNS,
                      var_name="Event_Type", value_name="Event_Count")
    long_df["Plot_Pos"] = long_df["Region"] + "_" + long_df["Event_Type"]
    return long_df


def build_x_positions():
    """Set event positions with a gap between neighboring regions."""
    positions = []
    centers = []
    for region_index, region in enumerate(REGION_ORDER):
        start = len(positions)
        positions.extend(f"{region}_{event}" for event in EVENT_COLUMNS)
        centers.append(start + (len(EVENT_COLUMNS) - 1) / 2)
        if region_index < len(REGION_ORDER) - 1:
            positions.append(f"Empty_{region_index + 1}")
    return positions, centers


def style_axis(ax, group_centers):
    """Apply consistent axis labels, ticks and borders."""
    ax.set_xlabel("")
    ax.set_ylabel("Occurrences per Site", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=10)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(CUSTOM_X_LABELS, fontsize=X_TICK_SIZE,
                       fontweight="bold", fontfamily=FONT_FAMILY)
    ax.tick_params(axis="y", labelsize=Y_TICK_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontweight("bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    sns.despine(ax=ax)


def create_chart(long_df):
    """Draw the box plots and points."""
    x_positions, group_centers = build_x_positions()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=OUTPUT_DPI)
    sns.boxplot(data=long_df, x="Plot_Pos", y="Event_Count", hue="Event_Type",
                order=x_positions, palette=PALETTE, dodge=False, width=BOX_WIDTH,
                showfliers=False,
                boxprops={"alpha": 0.5, "edgecolor": "black", "linewidth": 1.2},
                medianprops={"color": "black", "linewidth": 1.5}, ax=ax)
    sns.stripplot(data=long_df, x="Plot_Pos", y="Event_Count", hue="Event_Type",
                  order=x_positions, palette=PALETTE, dodge=False, size=POINT_SIZE,
                  alpha=0.75, jitter=POINT_JITTER, edgecolor="white",
                  linewidth=0.5, ax=ax)

    handles, _ = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[:len(EVENT_COLUMNS)], FRIENDLY_EVENT_LABELS,
        title="Event type", loc="upper right", bbox_to_anchor=(0.99, 0.98),
        frameon=True, facecolor="white", edgecolor="lightgray", framealpha=0.9,
        prop={"family": FONT_FAMILY, "size": LEGEND_TEXT_SIZE},
        title_fontproperties={"family": FONT_FAMILY, "size": LEGEND_TITLE_SIZE,
                              "weight": "bold"},
    )
    style_axis(ax, group_centers)
    fig.tight_layout()
    return fig, legend


def save_chart(fig, legend):
    """Save PNG and PDF output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    options = {"bbox_inches": "tight", "bbox_extra_artists": (legend,)}
    fig.savefig(png_path, dpi=OUTPUT_DPI, **options)
    with BytesIO() as tiff_buffer:
        fig.savefig(tiff_buffer, format="tiff", dpi=600, pil_kwargs={'compression': 'tiff_lzw'}, **options)
        Path(png_path.with_suffix('.tiff')).write_bytes(tiff_buffer.getbuffer())
    fig.savefig(pdf_path, format="pdf", **options)
    plt.close(fig)
    return png_path, pdf_path


def main():
    if len(REGION_ORDER) != len(CUSTOM_X_LABELS):
        raise ValueError("REGION_ORDER 与 CUSTOM_X_LABELS 的数量必须一致。")
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"找不到数据文件：{INPUT_CSV}")
    configure_plot_style()
    figure, legend = create_chart(load_and_prepare_data(INPUT_CSV))
    png_path, pdf_path = save_chart(figure, legend)
    print(f"图表已成功导出：\nPNG: {png_path}\nPDF: {pdf_path}")


if __name__ == "__main__":
    main()
