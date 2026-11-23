# -*- coding: utf-8 -*-
"""Draw a raincloud plot of DO response ratios relative to the seasonal reference (Shock-L)."""
from io import BytesIO
from pathlib import Path
import matplotlib

ROOT = Path(__file__).resolve().parents[4]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================== User configuration ==============================
BASE_INPUT_DIR = Path(str(ROOT / 'Result/Decrease/Overall'))
OUTPUT_PATH = Path(str(ROOT / 'V/Figure_V1/SFigure/SFigure12_Seasonal_departure/Images/DO_Shock_Ratio_LongTerm.png'))
TARGET_COLUMN, Y_AXIS_LABEL = "Shock_Ratio_LongTerm", "Seasonal-relative DO ratio (Shock-L)"
CUSTOM_CATEGORY_LABELS = {"Heatwave": "Heatwave", "Flood": "High-flow-only", "DC": "DC", "H5F": "H5F", "F5H": "F5H"}
FONT_FAMILY = "Arial"
AXIS_LABEL_SIZE, X_TICK_SIZE, Y_TICK_SIZE = 23, 23, 23
FIGURE_SIZE, OUTPUT_DPI = (12, 8), 300
COLORS = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
BOX_WIDTH, VIOLIN_WIDTH, POINT_SIZE, RANDOM_SEED = 0.15, 0.5, 30, 42
FILE_CONFIGS = {name: BASE_INPUT_DIR / filename for name, filename in {
    "Heatwave": "Heatwave_Only_Shock_All_Events.csv", "Flood": "Flood_Only_Shock_All_Events.csv",
    "DC": "Direct_Compound_Shock_All_Events.csv", "H5F": "H5F_Compound_Shock_All_Events.csv",
    "F5H": "F5H_Compound_Shock_All_Events.csv"}.items()}
# ======================================================================


def configure_font(font_family):
    plt.rcParams.update({
        "font.family": font_family,
        "font.sans-serif": [font_family],
        "axes.unicode_minus": False,
    })


def read_csv_fallback(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="gbk")


def load_event_data(file_configs, target_column):
    valid_data = {}
    print(f"开始提取目标指标: [{target_column}]")
    for label, file_path in file_configs.items():
        path = Path(file_path)
        if not path.is_file():
            print(f"文件不存在，已跳过: {path}")
            continue
        try:
            df = read_csv_fallback(path)
            if target_column not in df.columns:
                print(f"{label} 缺少列 [{target_column}]，已跳过。")
                continue
            values = df[target_column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            if values.size:
                valid_data[label] = values
                print(f"{label}: 提取 {values.size} 个有效数据点。")
        except (OSError, pd.errors.ParserError) as error:
            print(f"读取 {label} 失败: {error}")
    if not valid_data:
        raise ValueError("所有输入文件均未提取到有效数据。")
    return valid_data


def draw_raincloud(valid_data, config):
    categories = [name for name in config["category_labels"] if name in valid_data]
    positions = np.arange(len(categories))
    colors = config["colors"][:len(categories)]
    rng = np.random.default_rng(config["random_seed"])
    fig, ax = plt.subplots(figsize=config["figure_size"])

    for index, category in enumerate(categories):
        values = valid_data[category]
        box_position = positions[index] - 0.05
        ax.boxplot(
            values, positions=[box_position], widths=config["box_width"],
            patch_artist=True, showfliers=False, notch=False,
            medianprops={"color": "black", "linewidth": 2},
            boxprops={"facecolor": colors[index], "edgecolor": "black",
                      "linewidth": 1.5, "alpha": 0.8},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
        )
        violin_position = positions[index] + config["box_width"] / 2
        violin = ax.violinplot(values, positions=[violin_position],
                               widths=config["violin_width"], showmeans=False,
                               showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(colors[index])
            body.set_edgecolor("black")
            body.set_alpha(0.6)
            body.set_linewidth(1)
            vertices = body.get_paths()[0].vertices
            vertices[:, 0] = np.maximum(vertices[:, 0], violin_position)
        jitter_x = rng.normal(box_position - config["box_width"], 0.03, values.size)
        ax.scatter(jitter_x, values, color=colors[index], alpha=0.6,
                   s=config["point_size"], edgecolor="white", linewidth=0.5, zorder=2)

    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.2, zorder=0)

    font = config["font_family"]
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{config["category_labels"][name]}\n(n={len(valid_data[name]):,})' for name in categories],
                       fontsize=config["x_tick_size"], fontweight="bold",
                       fontfamily=font)
    ax.set_ylabel(config["y_axis_label"], fontsize=config["axis_label_size"],
                  fontweight="bold", fontfamily=font)
    ax.tick_params(axis="y", labelsize=config["y_tick_size"])
    for label in ax.get_yticklabels():
        label.set_fontfamily(font)
        label.set_fontweight("bold")
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig, categories


def export_statistics(valid_data, categories, csv_path):
    records = []
    for category in categories:
        values = valid_data[category]
        records.append({
            "Event_Type": category,
            "Sample_Size": len(values),
            "Mean": round(np.mean(values), 3),
            "Median": round(np.median(values), 3),
            "25th_Percentile": round(np.percentile(values, 25), 3),
            "75th_Percentile": round(np.percentile(values, 75), 3),
        })
    pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")


def run_raincloud(config):
    configure_font(config["font_family"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_data = load_event_data(config["file_configs"], config["target_column"])
    fig, categories = draw_raincloud(valid_data, config)
    fig.savefig(output_path, dpi=config["dpi"], bbox_inches="tight")
    with BytesIO() as tiff_buffer:
        fig.savefig(tiff_buffer, format="tiff", dpi=600, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
        Path(output_path.with_suffix('.tiff')).write_bytes(tiff_buffer.getbuffer())
    plt.close(fig)
    statistics_path = output_path.with_name(f"{output_path.stem}_Statistics.csv")
    export_statistics(valid_data, categories, statistics_path)
    print(f"图表已保存: {output_path}\n统计摘要已保存: {statistics_path}")


def main():
    run_raincloud({"file_configs": FILE_CONFIGS, "output_path": OUTPUT_PATH, "target_column": TARGET_COLUMN,
        "y_axis_label": Y_AXIS_LABEL, "category_labels": CUSTOM_CATEGORY_LABELS, "font_family": FONT_FAMILY,
        "axis_label_size": AXIS_LABEL_SIZE, "x_tick_size": X_TICK_SIZE, "y_tick_size": Y_TICK_SIZE,
        "figure_size": FIGURE_SIZE, "dpi": OUTPUT_DPI, "colors": COLORS, "box_width": BOX_WIDTH,
        "violin_width": VIOLIN_WIDTH, "point_size": POINT_SIZE, "random_seed": RANDOM_SEED})
if __name__ == "__main__":
    main()
