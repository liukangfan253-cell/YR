from io import BytesIO
from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np
import matplotlib

ROOT = Path(__file__).resolve().parents[5]

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl

# ================== 0. Global font settings ==================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1. Configuration ==================
# Set the input directory. The script reads all CSV files in it.
input_folder = str(ROOT / 'Data/WQ_Hy_M_DYN_data/YR/by_station_all')

output_folder = str(ROOT / 'V/Figure_V1/SFigure/SFigure09_Correlation/Images')
os.makedirs(output_folder, exist_ok=True)

# Exclude snow metrics and include Tmax and Tmin among the meteorological predictors.
raw_csv_vars = [
    'WT', 'pH', 'DO', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN', 'CODₘₙ',
    'T', 'Tmax', 'Tmin', 'SP', 'SSR', 'TP_M', 'WS', 'Td',
    'WL', 'Q', 'Ss', 'EVI'
]

grid_height = 1.3
dpi = 600
# ==============================================

# ================== 2. Load and clean the data ==================
print(f"正在读取并合并 {input_folder} 下的所有 CSV 数据...")

# Find all CSV files in the directory.
csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

df_list = []
if not csv_files:
    raise FileNotFoundError(f"未在输入目录找到 CSV 文件，停止生成相关性图表：{input_folder}")
else:
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
            print(f"  - 成功读取: {os.path.basename(file)} (数据形状: {temp_df.shape})")
        except Exception as e:
            print(f"  ❌ 读取文件 {os.path.basename(file)} 时出错: {e}")

    if df_list:
        # Concatenate all DataFrames.
        df = pd.concat(df_list, ignore_index=True)
        print(f"✅ 合并完成！总数据形状: {df.shape}")
    else:
        raise RuntimeError(f"所有 CSV 文件均读取失败，停止生成相关性图表：{input_folder}")

# Use math mode only for subscripts. The other characters inherit bold Arial.
# Use LaTeX subscripts for Tmax and Tmin labels.
latex_rename_dict = {
    'NH₃-N': 'NH$_3$-N',
    'CODₘₙ': 'COD$_{Mn}$',
    'Tmax': 'T$_{max}$',
    'Tmin': 'T$_{min}$'
}
df.rename(columns=latex_rename_dict, inplace=True)
interest_vars = [latex_rename_dict.get(v, v) for v in raw_csv_vars]

initial_vars = [v for v in interest_vars if v in df.columns]
df_temp = df[initial_vars].copy()

available_vars = []
for col in initial_vars:
    if df_temp[col].nunique() > 1:
        available_vars.append(col)

# Scale values to the interval from zero to one.
for col in available_vars:
    min_val = df_temp[col].min()
    max_val = df_temp[col].max()
    if max_val > min_val:
        df_temp[col] = (df_temp[col] - min_val) / (max_val - min_val)
    else:
        df_temp[col] = 0

# ================== 2.5 Export CSV tables ==================
print("正在计算相关系数和P值矩阵，并导出 CSV 表格...")
corr_df = pd.DataFrame(np.ones((len(available_vars), len(available_vars))), index=available_vars,
                       columns=available_vars)
pval_df = pd.DataFrame(np.zeros((len(available_vars), len(available_vars))), index=available_vars,
                       columns=available_vars)

for i in range(len(available_vars)):
    for j in range(len(available_vars)):
        if i != j:
            pair = df_temp[[available_vars[i], available_vars[j]]].dropna()
            if len(pair) > 2:
                r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                corr_df.iloc[i, j] = r
                pval_df.iloc[i, j] = p
            else:
                corr_df.iloc[i, j] = np.nan
                pval_df.iloc[i, j] = np.nan

# Set the CSV output path.
corr_csv = os.path.join(output_folder, "YR_Merged_correlation_matrix.csv")
pval_csv = os.path.join(output_folder, "YR_Merged_pvalue_matrix.csv")
data_csv = os.path.join(output_folder, "YR_Merged_cleaned_normalized_data.csv")

# Save with a UTF-8 BOM for compatibility with Excel.
corr_df.to_csv(corr_csv, float_format='%.3f', encoding='utf-8-sig')
pval_df.to_csv(pval_csv, float_format='%.3f', encoding='utf-8-sig')
df_temp.to_csv(data_csv, index=False, encoding='utf-8-sig')
print("✅ CSV表格导出成功！共导出 3 个表格（相关矩阵、P值矩阵、归一化清洗数据）。")


# ================== 3. Plotting functions ==================
def corrfunc(x, y, **kwargs):
    ax = plt.gca()
    mask = ~pd.isna(x) & ~pd.isna(y)
    if mask.sum() < 3: return

    r, p = stats.pearsonr(x[mask], y[mask])
    cmap = plt.get_cmap('RdBu_r')
    color = cmap((r + 1) / 2)

    rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=color, zorder=0)
    ax.add_patch(rect)

    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    text_color = "white" if abs(r) > 0.55 else "black"
    annot_text = f"{r:.3f}\n{stars}" if stars else f"{r:.3f}"

    ax.text(0.5, 0.5, annot_text, transform=ax.transAxes,
            ha="center", va="center", color=text_color, fontsize=16, fontweight='bold', fontfamily='Arial')


def scatter_reg_func(x, y, **kwargs):
    ax = plt.gca()
    mask = ~pd.isna(x) & ~pd.isna(y)
    if mask.sum() < 2: return
    x_valid, y_valid = x[mask], y[mask]
    sns.regplot(x=x_valid, y=y_valid, ax=ax,
                scatter_kws={'s': 10, 'alpha': 0.4, 'color': '#D36B58'},
                line_kws={'color': '#C33C23', 'linewidth': 1.8})


def diag_func(x, **kwargs):
    ax = plt.gca()
    x_valid = x.dropna()
    if len(x_valid) > 2 and x_valid.nunique() > 1:
        try:
            kde = stats.gaussian_kde(x_valid)
            x_range = np.linspace(0, 1, 100)
            y_density = kde(x_range)
            y_scaled = y_density / y_density.max() * 0.9
            ax.fill_between(x_range, 0, y_scaled, color="#5B7DB1", alpha=0.5)
            ax.plot(x_range, y_scaled, color="#1F4E79", linewidth=2)
        except:
            pass
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


# ================== 4. Draw the grid ==================
print("正在绘制矩阵图，预计需要 1-2 分钟...")
sns.set_theme(style="white", font_scale=0.8, rc={"font.family": "sans-serif", "font.sans-serif": ["Arial"]})

g = sns.PairGrid(df_temp, height=grid_height)
g.map_diag(diag_func)
g.map_lower(scatter_reg_func)
g.map_upper(corrfunc)

# ================== 5. Style and arrange panels ==================
g.fig.subplots_adjust(wspace=0.03, hspace=0.03)

for i in range(g.axes.shape[0]):
    for j in range(g.axes.shape[1]):
        ax = g.axes[i, j]
        if ax is None: continue
        ax.grid(False)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Hide all x-axis tick labels.
        ax.set_xticks([])

        # Show the y-axis labels in the leftmost column only.
        if j == 0:
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'], fontsize=12, fontweight='bold')
        else:
            ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#808080')
            spine.set_linewidth(1.0)

        if i >= j: ax.set_facecolor('white')

for idx, var_name in enumerate(available_vars):
    g.axes[idx, 0].set_ylabel(var_name, rotation=0, ha='right', va='center', fontsize=20, fontweight='bold',
                              labelpad=18, fontfamily='Arial')
    g.axes[-1, idx].set_xlabel(var_name, rotation=0, ha='center', va='top', fontsize=20, fontweight='bold',
                               labelpad=18, fontfamily='Arial')

g.fig.subplots_adjust(right=0.92, bottom=0.1, left=0.1)

cbar_ax = g.fig.add_axes([0.94, 0.15, 0.015, 0.7])
cmap = plt.get_cmap('RdBu_r')
norm = mpl.colors.Normalize(vmin=-1, vmax=1)
cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
cb.set_label('YR Pearson Correlation', fontsize=24, fontweight='bold', fontfamily='Arial')
cb.ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'], fontsize=18, fontweight='bold')

# ================== 6. Save output ==================
png_file = os.path.join(output_folder, "YR_Merged_Correlation_Matrix.png")
pdf_file = os.path.join(output_folder, "YR_Merged_Correlation_Matrix.pdf")

print("正在保存高清图片...")
plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
with BytesIO() as tiff_buffer:
    plt.savefig(tiff_buffer, format="tiff", dpi=dpi, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
    Path(os.path.splitext(png_file)[0] + '.tiff').write_bytes(tiff_buffer.getbuffer())
plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
plt.close()

print(f"大功告成！\nPNG文件已保存至: {png_file}\nPDF文件已保存至: {pdf_file}")
