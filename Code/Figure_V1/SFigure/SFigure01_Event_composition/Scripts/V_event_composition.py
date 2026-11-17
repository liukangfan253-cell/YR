# -*- coding: utf-8 -*-
"""Plot stacked counts of complete and incomplete records by region and event type."""

from pathlib import Path
from io import BytesIO
import argparse

import matplotlib

ROOT = Path(__file__).resolve().parents[5]

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ============================== User configuration ==============================
INPUT_CSV = Path(str(ROOT / 'Data/Event_N/Event_Counts_By_Region.csv'))
OUTPUT_DIR = Path(str(ROOT / 'V/Figure_V1/SFigure/SFigure01_Event_composition/Images'))
OUTPUT_BASENAME = "Total_Events_Stacked_BarChart"

REGION_ORDER = ["UR", "MR", "LR"]

FONT_FAMILY = "Arial"
AXIS_LABEL_SIZE = 20
X_TICK_SIZE = 20
Y_TICK_SIZE = 16
LEGEND_TEXT_SIZE = 16
LEGEND_TITLE_SIZE = 16

FIGURE_SIZE = (10, 6)
OUTPUT_DPI = 300
BAR_TOTAL_WIDTH = 0.7
X_AXIS_LIMITS = (-0.5, 2.9)
COLORS = ["#4A90E2", "#D0021B", "#F5A623"]
# ======================================================================

EVENT_MAPPING = {
    "Flood Only": ("Flood_Only_Complete_Count", "Flood_Only_Missing_Count"),
    "Heatwave Only": ("Heatwave_Only_Complete_Count", "Heatwave_Only_Missing_Count"),
    "Compound Event": ("Compound_Event_Complete_Count", "Compound_Event_Missing_Count"),
}


def configure_plot_style():
    """Apply consistent plot fonts and styling."""
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.sans-serif": [FONT_FAMILY],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_and_prepare_data(input_csv):
    """Load the data and pool the three compound-event types."""
    df = pd.read_csv(input_csv)
    df["Compound_Event_Complete_Count"] = df[
        ["Direct_Compound_Complete_Count", "H5F_Compound_Complete_Count",
         "F5H_Compound_Complete_Count"]
    ].sum(axis=1)
    df["Compound_Event_Missing_Count"] = df[
        ["Direct_Compound_Missing_Count", "H5F_Compound_Missing_Count",
         "F5H_Compound_Missing_Count"]
    ].sum(axis=1)
    return df


def style_axis(ax, region_labels):
    """Apply consistent axis labels, ticks and borders."""
    ax.set_xlabel("")
    ax.set_ylabel("Number of site-event records", fontsize=AXIS_LABEL_SIZE,
                  fontweight="bold", labelpad=10)
    ax.set_xlim(*X_AXIS_LIMITS)
    ax.set_xticks(np.arange(len(REGION_ORDER)))
    ax.set_xticklabels(region_labels, fontsize=X_TICK_SIZE,
                       fontweight="bold", fontfamily=FONT_FAMILY)
    ax.tick_params(axis="y", labelsize=Y_TICK_SIZE)
    for label in ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)
        label.set_fontweight("bold")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    sns.despine(ax=ax)


def create_chart(df):
    """Draw grouped stacked bars."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=OUTPUT_DPI)
    x = np.arange(len(REGION_ORDER))
    bar_width = BAR_TOTAL_WIDTH / len(EVENT_MAPPING)
    offsets = (np.arange(len(EVENT_MAPPING)) * bar_width
               - BAR_TOTAL_WIDTH / 2 + bar_width / 2)

    for region_index, region in enumerate(REGION_ORDER):
        region_data = df.loc[df["Region"] == region]
        if region_data.empty:
            continue
        row = region_data.iloc[0]
        for event_index, (_, columns) in enumerate(EVENT_MAPPING.items()):
            complete_col, missing_col = columns
            complete_value = row[complete_col]
            missing_value = row[missing_col]
            x_position = x[region_index] + offsets[event_index]
            bar_style = {"width": bar_width, "color": COLORS[event_index],
                         "alpha": 0.5, "edgecolor": "black", "linewidth": 1.2}
            ax.bar(x_position, complete_value, **bar_style)
            if missing_value > 0:
                ax.bar(x_position, missing_value, bottom=complete_value,
                       hatch="////", **bar_style)

    color_handles = [
        mpatches.Patch(facecolor=color, alpha=0.5, edgecolor="black", label=name)
        for color, name in zip(COLORS, EVENT_MAPPING)
    ]
    texture_handles = [
        mpatches.Patch(facecolor="gray", alpha=0.5, edgecolor="black", label="Complete"),
        mpatches.Patch(facecolor="gray", alpha=0.5, hatch="////",
                       edgecolor="black", label="Missing"),
    ]
    title_font = {"family": FONT_FAMILY, "size": LEGEND_TITLE_SIZE, "weight": "bold"}
    text_font = {"family": FONT_FAMILY, "size": LEGEND_TEXT_SIZE}
    legend1 = ax.legend(handles=color_handles, title="Event Type", loc="upper left",
                        bbox_to_anchor=(0.82, 0.98), frameon=True, edgecolor="black",
                        facecolor="white", prop=text_font, title_fontproperties=title_font)
    ax.add_artist(legend1)
    ax.legend(handles=texture_handles, title="Data Quality", loc="upper left",
              bbox_to_anchor=(0.82, 0.65), frameon=True, edgecolor="black",
              facecolor="white", prop=text_font, title_fontproperties=title_font)
    site_counts = df.set_index('Region')['Total_Sites']
    style_axis(ax, [f'{region}({int(site_counts.loc[region])})' for region in REGION_ORDER])
    fig.tight_layout()
    return fig, legend1


def save_chart(fig, extra_artist):
    """Save PNG and PDF output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    options = {"bbox_inches": "tight", "bbox_extra_artists": (extra_artist,)}
    fig.savefig(png_path, dpi=OUTPUT_DPI, **options)
    fig.savefig(pdf_path, format="pdf", **options)
    fig.savefig(pdf_path.with_suffix(".svg"), **options)
    with BytesIO() as buffer:
        fig.savefig(buffer, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, **options)
        png_path.with_suffix(".tiff").write_bytes(buffer.getbuffer())
    plt.close(fig)
    return png_path, pdf_path


def main():
    global INPUT_CSV, OUTPUT_DIR
    parser = argparse.ArgumentParser(description='SFigure01: regional event counts by data completeness.')
    parser.add_argument('--input-csv', type=Path, default=INPUT_CSV)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    INPUT_CSV, OUTPUT_DIR = args.input_csv, args.output_dir
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"找不到数据文件：{INPUT_CSV}")
    configure_plot_style()
    figure, legend = create_chart(load_and_prepare_data(INPUT_CSV))
    png_path, pdf_path = save_chart(figure, legend)
    print(f"图表已成功导出：\nPNG: {png_path}\nPDF: {pdf_path}")


if __name__ == "__main__":
    main()
