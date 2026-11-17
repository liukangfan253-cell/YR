# -*- coding: utf-8 -*-
from io import BytesIO
from pathlib import Path
import os
import glob
import math
import pandas as pd
import matplotlib

# Use a noninteractive backend.
ROOT = Path(__file__).resolve().parents[5]

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import warnings
from PIL import Image, ImageOps

warnings.filterwarnings('ignore')

# Global font settings
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 21
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. Edit paths to match your local files.
# =========================================================

INPUT_IMPUTED_DIR = str(ROOT / 'Result/BiLstm_Att_V1/Imputed_Data/By_Site_Imputed')
INPUT_SCATTER_DIR = str(ROOT / 'Result/BiLstm_Att_V1/Reports/By_Site_Scatter')
OUTPUT_PLOT_DIR = str(ROOT / 'V/Figure_V1/SFigure/SFigure07_Full_station_series/Images')

# Process one partition at a time. YR already includes all stations in LR, MR and UR.
# Reading those partitions as well would plot each station twice and duplicate it in the page grids.
TARGET_REGION = 'YR'

# Page grid settings from V_space2
MONTAGE_OUTPUT_DIR = os.path.join(OUTPUT_PLOT_DIR, 'Montage_Pages')
MONTAGE_PER_PAGE = 15
MONTAGE_COLS = 3
MONTAGE_ROWS = 5
MONTAGE_TILE_SIZE = (1100, 420)
MONTAGE_PADDING = 20
MONTAGE_BG_COLOR = 'white'


# =========================================================
# 2. Index metrics and assemble pages
# =========================================================
def build_scatter_file_index(scatter_root):
    """Recursively index station metrics in partitions such as LR, MR and UR."""
    pattern = os.path.join(scatter_root, '**', '*_Labeled_Scatter.csv')
    scatter_index = {}

    for scatter_path in glob.glob(pattern, recursive=True):
        filename = os.path.basename(scatter_path)
        if filename in scatter_index:
            print(f"⚠️ 发现重名指标文件，将使用第一个: {filename}")
            continue
        scatter_index[filename] = scatter_path

    return scatter_index


def create_montage_pages(image_paths, output_dir):
    """Arrange station figures in pages of 15, with three columns and five rows."""
    if not image_paths:
        print('⚠️ 没有可用于拼图的图片，跳过拼图输出。')
        return

    os.makedirs(output_dir, exist_ok=True)

    # Remove pages from the previous run so an output with fewer stations has no stale pages.
    for extension in ('png', 'tiff'):
        old_page_pattern = os.path.join(
            output_dir, f'All_Sites_TS_Test2025_Montage_Page_*.{extension}'
        )
        for old_page in glob.glob(old_page_pattern):
            os.remove(old_page)

    total_pages = math.ceil(len(image_paths) / MONTAGE_PER_PAGE)
    print(f'🧩 开始生成整体拼图：共 {len(image_paths)} 张图，输出 {total_pages} 页。')

    for page_idx in range(total_pages):
        start_idx = page_idx * MONTAGE_PER_PAGE
        page_images = image_paths[start_idx:start_idx + MONTAGE_PER_PAGE]
        tiles = []

        for img_path in page_images:
            try:
                with Image.open(img_path) as source_img:
                    img = source_img.convert('RGB')
                    thumb = ImageOps.contain(img, MONTAGE_TILE_SIZE)
                    tile = Image.new('RGB', MONTAGE_TILE_SIZE, MONTAGE_BG_COLOR)
                    x = (MONTAGE_TILE_SIZE[0] - thumb.size[0]) // 2
                    y = (MONTAGE_TILE_SIZE[1] - thumb.size[1]) // 2
                    tile.paste(thumb, (x, y))
                    tiles.append(tile)
            except Exception as e:
                print(f'⚠️ 无法读取图片 {img_path}，已跳过。原因：{e}')

        if not tiles:
            continue

        canvas_w = MONTAGE_COLS * MONTAGE_TILE_SIZE[0] + (MONTAGE_COLS + 1) * MONTAGE_PADDING
        canvas_h = MONTAGE_ROWS * MONTAGE_TILE_SIZE[1] + (MONTAGE_ROWS + 1) * MONTAGE_PADDING
        canvas = Image.new('RGB', (canvas_w, canvas_h), MONTAGE_BG_COLOR)

        for idx, tile in enumerate(tiles):
            row, col = divmod(idx, MONTAGE_COLS)
            x = MONTAGE_PADDING + col * (MONTAGE_TILE_SIZE[0] + MONTAGE_PADDING)
            y = MONTAGE_PADDING + row * (MONTAGE_TILE_SIZE[1] + MONTAGE_PADDING)
            canvas.paste(tile, (x, y))

        out_file = os.path.join(
            output_dir, f'All_Sites_TS_Test2025_Montage_Page_{page_idx + 1}.png'
        )
        canvas.save(out_file, dpi=(300, 300))
        # Preserve montage pixels; changing DPI metadata cannot restore detail.
        with BytesIO() as tiff_buffer:
            canvas.save(tiff_buffer, format='TIFF',
                        compression='tiff_lzw', dpi=(300, 300))
            Path(os.path.splitext(out_file)[0] + '.tiff').write_bytes(tiff_buffer.getbuffer())
        print(f'  - 拼图第 {page_idx + 1} 页已输出: {out_file}')


# =========================================================
# 3. Plot station time series in batches
# =========================================================

def plot_all_sites_timeseries():
    if not os.path.exists(INPUT_IMPUTED_DIR):
        print(f"❌ 错误: 找不到输入目录 {INPUT_IMPUTED_DIR}")
        return

    target_dir = os.path.join(INPUT_IMPUTED_DIR, TARGET_REGION)
    if not os.path.isdir(target_dir):
        print(f"❌ 错误: 找不到指定数据分区 {target_dir}")
        return
    regions = [TARGET_REGION]

    total_plots = 0
    all_output_images = []
    scatter_file_index = build_scatter_file_index(INPUT_SCATTER_DIR)
    print(f"📊 启动绘图引擎，检测到 {len(regions)} 个数据分区...")
    print(f"📈 已索引 {len(scatter_file_index)} 个站点指标文件。")

    for region in regions:
        current_imputed_dir = os.path.join(INPUT_IMPUTED_DIR, region)
        current_plot_dir = os.path.join(OUTPUT_PLOT_DIR, region) if region else OUTPUT_PLOT_DIR
        png_dir = os.path.join(current_plot_dir, 'PNG')
        tiff_dir = os.path.join(current_plot_dir, 'TIFF')
        os.makedirs(png_dir, exist_ok=True)
        os.makedirs(tiff_dir, exist_ok=True)

        site_files = sorted(glob.glob(os.path.join(current_imputed_dir, "*_Imputed.csv")))
        if site_files:
            region_label = region if region else "根目录"
            print(f"\n📂 正在处理 [{region_label}] 区域的站点数据...")

        for file_path in site_files:
            site_name = os.path.basename(file_path).replace('_Imputed.csv', '')

            # -----------------------------------------------------
            # Read metrics from the 2025 test set.
            # -----------------------------------------------------
            scatter_filename = f"{site_name}_Labeled_Scatter.csv"
            scatter_file_path = scatter_file_index.get(scatter_filename)

            test_nse_str, test_mae_str = "N/A", "N/A"
            if scatter_file_path and os.path.exists(scatter_file_path):
                scatter_df = pd.read_csv(scatter_file_path)
                if not scatter_df.empty and 'Test_NSE' in scatter_df.columns and 'Test_MAE' in scatter_df.columns:
                    nse_values = pd.to_numeric(scatter_df['Test_NSE'], errors='coerce').dropna()
                    mae_values = pd.to_numeric(scatter_df['Test_MAE'], errors='coerce').dropna()
                    nse_val = nse_values.iloc[0] if not nse_values.empty else None
                    mae_val = mae_values.iloc[0] if not mae_values.empty else None

                    if nse_val is not None: test_nse_str = f"{float(nse_val):.3f}"
                    if mae_val is not None: test_mae_str = f"{float(mae_val):.3f}"
                if test_nse_str == "N/A" or test_mae_str == "N/A":
                    print(f"⚠️ {site_name} 缺少有效的 2025 测试指标，将显示 N/A，不回退到验证指标。")
            else:
                print(f"⚠️ 未找到 {site_name} 的指标文件，NSE/MAE 将显示为 N/A。")

            # -----------------------------------------------------
            # Load the data and plot.
            # -----------------------------------------------------
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])

            if 'Original_DO' not in df.columns or 'Imputed_DO' not in df.columns:
                print(f"⚠️ {site_name} 数据不完整，跳过。")
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300, gridspec_kw={'width_ratios': [1.8, 1.0]})
            # Reduce horizontal spacing between panels.
            plt.subplots_adjust(wspace=0.12)
            test_start, test_end = pd.to_datetime('2025-01-01'), pd.to_datetime('2025-12-31')

            # ====== Left panel: full series ======
            ax1.plot(df['Date'], df['Original_DO'], color='#E57373', linewidth=1.0)
            ax1.scatter(df['Date'], df['Imputed_DO'], facecolors='black', edgecolors='white', linewidths=0.6, s=25, zorder=3)
            ax1.axvspan(test_start, test_end, color='gray', alpha=0.25)
            ax1.set_ylabel('Dissolved Oxygen (mg/L)', fontsize=21, fontweight='bold')
            ax1.set_xlim(df['Date'].min(), df['Date'].max())
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax1.tick_params(axis='both', which='major', labelsize=21)
            for tick in ax1.get_xticklabels() + ax1.get_yticklabels():
                tick.set_fontweight('bold')
            ax1.grid(False)

            # Place the station name in the upper-left corner of the left panel.
            ax1.text(
                0.02, 0.96, site_name, transform=ax1.transAxes,
                fontsize=21, fontweight='bold', ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor='none', alpha=0.80)
            )

            # ====== Right panel: detail ======
            test_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
            if not test_df.empty:
                ax2.plot(test_df['Date'], test_df['Original_DO'], color='#E57373', linewidth=1.2)
                ax2.scatter(test_df['Date'], test_df['Imputed_DO'], facecolors='black', edgecolors='white', linewidths=1.2, s=70, zorder=3)
                ax2.set_facecolor('#f8f9fa')
                ax2.set_xlim(test_start, test_end)

                ymin = min(test_df['Original_DO'].min(), test_df['Imputed_DO'].min())
                ymax = max(test_df['Original_DO'].max(), test_df['Imputed_DO'].max())
                ax2.set_ylim(ymin - 1, ymax + 1)

                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

                ax2.tick_params(axis='both', which='major', labelsize=21)
                ax2.tick_params(axis='x', which='major', labelsize=17)
                for tick in ax2.get_xticklabels() + ax2.get_yticklabels():
                    tick.set_fontweight('bold')
            ax2.grid(False)

            # Show the 2025 test metrics. The caption identifies the test year.
            test_label = f"NSE: {test_nse_str}\nMAE: {test_mae_str}"
            ax2.text(0.96, 0.04, test_label, transform=ax2.transAxes, fontsize=17, fontweight='bold',
                     horizontalalignment='right', verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            # ====== Arrow between the panels ======
            arrow = patches.FancyArrowPatch(
                posA=(0.595, 0.45), posB=(0.625, 0.45), transform=fig.transFigure,
                color='gray', alpha=0.8, arrowstyle='simple,tail_width=3,head_width=10,head_length=10'
            )
            fig.patches.append(arrow)

            # Save the figure.
            safe_site_name = str(site_name).replace('/', '_').replace('\\', '_')
            out_file = os.path.join(png_dir, f'{safe_site_name}_TS_Test2025.png')
            plt.savefig(out_file, dpi=300, bbox_inches='tight')
            with BytesIO() as tiff_buffer:
                plt.savefig(tiff_buffer, format="tiff", dpi=600,
                            bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
                Path(tiff_dir, f'{safe_site_name}_TS_Test2025.tiff').write_bytes(tiff_buffer.getbuffer())
            plt.close()

            all_output_images.append(out_file)
            total_plots += 1
            print(f"  - 生成成功: {safe_site_name}")

    create_montage_pages(all_output_images, MONTAGE_OUTPUT_DIR)

    print(f"\n✅ 绘图任务圆满完成！共生成 {total_plots} 张高清图表。")
    print(f"📁 单站点图片目录: {OUTPUT_PLOT_DIR}")
    print(f"🧩 拼图分页输出目录: {MONTAGE_OUTPUT_DIR}")


if __name__ == "__main__":
    plot_all_sites_timeseries()
