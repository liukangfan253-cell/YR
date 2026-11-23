# -*- coding: utf-8 -*-
"""
Plot monthly DO before and after gap filling from 2021 to 2025.

Each side combines a violin plot and a heatmap. The panels share an axis and the left panel shows all year labels. Use bold Arial, omit horizontal dashed grid lines and export at 300 DPI.
"""

from io import BytesIO
from pathlib import Path
import matplotlib

ROOT = Path(__file__).resolve().parents[4]

matplotlib.use('Agg')  # Use the noninteractive Agg backend for headless environments.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.colors as mcolors

# ====== Use bold Arial throughout ======
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 25
plt.rcParams['font.weight'] = 'bold'  # Use bold font globally.
plt.rcParams['axes.labelweight'] = 'bold'  # Use bold axis labels.
plt.rcParams['axes.titleweight'] = 'bold'  # Use bold titles.
plt.rcParams['axes.unicode_minus'] = False


def load_csv_data(file_path):
    """Load CSV data."""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df = df[df['Year'].between(2021, 2025)]  # Keep records from 2021 to 2025.
        print(f"已加载数据：{len(df)}行 -> {os.path.basename(file_path)}")
        return df
    except Exception as e:
        print(f"加载文件失败 {file_path}：{e}")
        return None


def get_monthly_means(violin_data):
    """Extract monthly means to set common color limits."""
    means = []
    for month in range(1, 13):
        month_data = violin_data[violin_data['Month'] == month]['Mean'].dropna().values
        if len(month_data) > 0:
            means.append(np.mean(month_data))
    return means


def draw_violin_subplot(ax, violin_data, variable_name, show_ylabel, custom_cmap, global_min_mean, global_max_mean,
                        border_width):
    """Plot one violin panel with a shared color normalization."""
    violin_data_list = []
    month_means = []
    month_indices = []

    for month in range(1, 13):
        month_data = violin_data[violin_data['Month'] == month]['Mean'].dropna().values
        if len(month_data) > 0:
            violin_data_list.append(month_data)
            month_means.append(np.mean(month_data))
            month_indices.append(month)

    if not violin_data_list:
        return

    # Shared normalization
    if global_max_mean == global_min_mean:
        normalized_means = [0.5] * len(month_means)
    else:
        normalized_means = [(mean - global_min_mean) / (global_max_mean - global_min_mean) for mean in month_means]

    colors = [custom_cmap(value) for value in normalized_means]

    violin_parts = ax.violinplot(violin_data_list,
                                 showmeans=False,
                                 showmedians=False,
                                 showextrema=False,
                                 widths=0.8,
                                 bw_method=0.5)

    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)
        pc.set_linewidth(1)

    for i, (month_data, month_idx) in enumerate(zip(violin_data_list, month_indices), 1):
        if len(month_data) > 0:
            mean_val = np.mean(month_data)
            q1 = np.percentile(month_data, 25)
            q3 = np.percentile(month_data, 75)

            ax.hlines(mean_val, i - 0.4, i + 0.4, colors='black', linestyles='--', linewidth=1.5, alpha=0.8)
            ax.hlines(q1, i - 0.3, i + 0.3, colors='black', linestyles=':', linewidth=1.2, alpha=0.6)
            ax.hlines(q3, i - 0.3, i + 0.3, colors='black', linestyles=':', linewidth=1.2, alpha=0.6)

    if show_ylabel:
        ax.set_ylabel(f'{variable_name} Mean', fontsize=29, fontweight='bold')
    else:
        ax.set_ylabel('')

    ax.set_xticks([])
    ax.set_xlim(0.5, 12.5)

    # Enlarge the violin plot's y-axis tick labels by two points and set them in bold.
    ax.tick_params(axis='y', labelsize=27)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    for spine in ax.spines.values():
        spine.set_linewidth(border_width)


def process_heatmap_matrix(heatmap_data):
    """Reindex the heatmap to all months and years from 2021 to 2025 so missing values do not shift the axes."""
    heatmap_matrix = heatmap_data.pivot_table(values='Mean', index='Year', columns='Month', aggfunc='mean')

    for month in range(1, 13):
        if month not in heatmap_matrix.columns:
            heatmap_matrix[month] = np.nan

    for year in [2021, 2022, 2023, 2024, 2025]:
        if year not in heatmap_matrix.index:
            heatmap_matrix.loc[year] = np.nan

    return heatmap_matrix.sort_index(axis=1).sort_index()


def draw_heatmap_subplot(ax, heatmap_matrix, vmin, vmax, custom_cmap, heatmap_text_color, show_ylabel, border_width):
    """Plot one heatmap."""
    cax = ax.imshow(heatmap_matrix.values, cmap=custom_cmap, aspect='auto', interpolation='nearest', vmin=vmin,
                    vmax=vmax)

    ax.set_xticks(np.arange(len(heatmap_matrix.columns)))
    ax.set_yticks(np.arange(len(heatmap_matrix.index)))

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Use bold month labels on the x-axis.
    ax.set_xticklabels(month_labels, fontsize=27, fontweight='bold')

    # Show year labels on the left panel and hide them on the right.
    if show_ylabel:
        year_labels = [str(int(year)) for year in heatmap_matrix.index]
        # Use bold year labels on the y-axis.
        ax.set_yticklabels(year_labels, fontsize=27, fontweight='bold')
        ax.set_ylabel('Year', fontsize=29, fontweight='bold', labelpad=15)
    else:
        # Hide the right panel's y-axis labels because it shares the axis with the left panel.
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylabel('')

    for i in range(len(heatmap_matrix.index)):
        for j in range(len(heatmap_matrix.columns)):
            value = heatmap_matrix.iloc[i, j]
            if not pd.isna(value):
                ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                        color=heatmap_text_color, fontsize=21, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, len(heatmap_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heatmap_matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=0.8)
    ax.tick_params(which="minor", size=0)

    for spine in ax.spines.values():
        spine.set_linewidth(border_width)

    return cax


def create_comparison_plot(before_v, before_h, after_v, after_h, variable_name, output_dir,
                           spacing_ratio=0.05, violin_height_ratio=0.9,
                           heatmap_text_color='black', border_width=1.5):
    """Build a side-by-side comparison figure."""
    custom_colors = ['#f8c2d0', '#facfda', '#fcdde6', '#fde9ef', '#c7e8f9', '#afdbf2', '#89c8ea', '#7ac0e6']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', custom_colors)

    means_before = get_monthly_means(before_v)
    means_after = get_monthly_means(after_v)
    all_means = means_before + means_after
    global_min_mean = min(all_means) if all_means else 0
    global_max_mean = max(all_means) if all_means else 1

    hm_before = process_heatmap_matrix(before_h)
    hm_after = process_heatmap_matrix(after_h)
    vmin = min(hm_before.min().min(), hm_after.min().min())
    vmax = max(hm_before.max().max(), hm_after.max().max())

    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[violin_height_ratio, 1], wspace=0.05, hspace=spacing_ratio)

    ax_v_before = fig.add_subplot(gs[0, 0])
    ax_v_after = fig.add_subplot(gs[0, 1], sharey=ax_v_before)

    # Align the right heatmap with the left and share their y-axis.
    ax_h_before = fig.add_subplot(gs[1, 0])
    ax_h_after = fig.add_subplot(gs[1, 1], sharey=ax_h_before)

    ax_v_before.set_title("Before Imputation", fontsize=31, fontweight='bold', pad=15)
    ax_v_after.set_title("After Imputation", fontsize=31, fontweight='bold', pad=15)

    draw_violin_subplot(ax_v_before, before_v, variable_name, True, custom_cmap, global_min_mean, global_max_mean,
                        border_width)
    draw_violin_subplot(ax_v_after, after_v, variable_name, False, custom_cmap, global_min_mean, global_max_mean,
                        border_width)
    plt.setp(ax_v_after.get_yticklabels(), visible=False)

    cax1 = draw_heatmap_subplot(ax_h_before, hm_before, vmin, vmax, custom_cmap, heatmap_text_color, True, border_width)
    cax2 = draw_heatmap_subplot(ax_h_after, hm_after, vmin, vmax, custom_cmap, heatmap_text_color, False, border_width)

    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
    cbar = fig.colorbar(cax1, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=25)

    # Use bold tick labels below the colorbar.
    for label in cbar.ax.get_xticklabels():
        label.set_fontweight('bold')

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_file = output_dir / f"{variable_name}_Comparison_Arial_Bold_{timestamp}.png"
    pdf_file = output_dir / f"{variable_name}_Comparison_Arial_Bold_{timestamp}.pdf"

    plt.savefig(png_file, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_file, bbox_inches='tight')
    with BytesIO() as tiff_buffer:
        plt.savefig(tiff_buffer, format="tiff", dpi=600, bbox_inches='tight',
                    pil_kwargs={'compression': 'tiff_lzw'})
        Path(png_file.with_suffix('.tiff')).write_bytes(tiff_buffer.getbuffer())
    plt.close(fig)

    print(f"已生成 Arial 加粗版 对比图：{png_file.name}")
    return png_file, pdf_file


def process_variable(variable_name, files_dict, output_dir,
                     spacing_ratio=0.05, violin_height_ratio=0.9,
                     heatmap_text_color='black', border_width=1.5):
    """Process one variable."""
    print(f"\n处理变量：{variable_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    before_v = load_csv_data(files_dict['before_violin'])
    before_h = load_csv_data(files_dict['before_heatmap'])
    after_v = load_csv_data(files_dict['after_violin'])
    after_h = load_csv_data(files_dict['after_heatmap'])

    if any(df is None or df.empty for df in [before_v, before_h, after_v, after_h]):
        print(f"[{variable_name}] 数据加载不完整，跳过生成对比图。")
        return None

    png_file, pdf_file = create_comparison_plot(before_v, before_h, after_v, after_h, variable_name,
                                                output_path, spacing_ratio, violin_height_ratio,
                                                heatmap_text_color, border_width)

    return {'comparison': (png_file, pdf_file)}


def main():
    # ====== Configuration ======
    variables_config = {
        'DO': {
            'before_violin': str(ROOT / 'Result/Mean_N_data/DO/DO_monthly_by_station.csv'),
            'before_heatmap': str(ROOT / 'Result/Mean_N_data/DO/DO_monthly_all_stations.csv'),
            'after_violin': str(ROOT / 'Result/Mean_F_data/DO/DO_monthly_by_station.csv'),
            'after_heatmap': str(ROOT / 'Result/Mean_F_data/DO/DO_monthly_all_stations.csv')
        },
    }

    output_dir = str(ROOT / 'V/Figure_V1/SFigure/SFigure10_Monthly_DO/Images')

    spacing_ratio = 0.02
    violin_height_ratio = 0.9
    heatmap_text_color = 'black'
    border_width = 1.5

    print("=" * 60)
    print("开始生成纯英文 Arial 版可视化图形 (刻度全部加粗)")
    print("=" * 60)

    for var_name, files in variables_config.items():
        process_variable(var_name, files, output_dir, spacing_ratio, violin_height_ratio, heatmap_text_color,
                         border_width)


if __name__ == "__main__":
    main()
