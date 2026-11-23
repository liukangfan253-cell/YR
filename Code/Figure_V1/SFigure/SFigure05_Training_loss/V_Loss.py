# -*- coding: utf-8 -*-
"""
Plot BiLSTM-Attention training and validation loss for Figure S5.

Input: DO_Development_Loss_History.csv.
Outputs: submission TIFF at 600 DPI, vector PDF and SVG, and a PNG preview.
"""

from io import BytesIO
from pathlib import Path
import os

import matplotlib
ROOT = Path(__file__).resolve().parents[4]

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd


# =========================================================
# 1. Inputs and outputs
# =========================================================

# Full path to the input CSV
INPUT_FILE = str(ROOT / 'Result/BiLstm_Att_V1/Reports/Loss_History/DO_Development_Loss_History.csv')

# Separate directory for figure outputs
OUTPUT_DIR = str(ROOT / 'V/Figure_V1/SFigure/SFigure05_Training_loss/Images')

# Output filename
OUTPUT_TIF = os.path.join(
    OUTPUT_DIR,
    "Figure_S2_BiLSTM_Attention_Loss_Curve.tiff"
)

OUTPUT_PDF = os.path.join(
    OUTPUT_DIR,
    "Figure_S2_BiLSTM_Attention_Loss_Curve.pdf"
)

OUTPUT_SVG = os.path.join(
    OUTPUT_DIR,
    "Figure_S2_BiLSTM_Attention_Loss_Curve.svg"
)

OUTPUT_PNG = os.path.join(
    OUTPUT_DIR,
    "Figure_S2_BiLSTM_Attention_Loss_Curve.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. Water Research plot style
# =========================================================

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# =========================================================
# 3. Read CSV files
# =========================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"未找到输入文件：\n{INPUT_FILE}"
    )

try:
    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )
except UnicodeDecodeError:
    df = pd.read_csv(
        INPUT_FILE,
        encoding="gbk"
    )


required_columns = [
    "Epoch",
    "Train_Loss",
    "Validation_Loss",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"CSV文件缺少必要字段：{missing_columns}"
    )


for col in required_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

n_before = len(df)
df = (
    df.dropna(subset=required_columns)
    .sort_values("Epoch")
    .reset_index(drop=True)
)

excluded_count = n_before - len(df)
if excluded_count:
    print(f"跳过 {excluded_count} 条轮次或损失值无效的记录")

if df.empty:
    raise ValueError(
        "CSV文件中没有有效的损失数据。"
    )


# =========================================================
# 4. Select the best training epoch
# =========================================================

if "Best_Epoch_Selected_By_Val24" in df.columns:

    best_epoch_series = pd.to_numeric(
        df["Best_Epoch_Selected_By_Val24"],
        errors="coerce"
    ).dropna()

    if not best_epoch_series.empty:
        best_epoch = int(
            best_epoch_series.iloc[0]
        )
    else:
        best_epoch = int(
            df.loc[
                df["Validation_Loss"].idxmin(),
                "Epoch"
            ]
        )

else:
    best_epoch = int(
        df.loc[
            df["Validation_Loss"].idxmin(),
            "Epoch"
        ]
    )


best_row = df[df["Epoch"] == best_epoch]

if best_row.empty:
    best_index = df["Validation_Loss"].idxmin()
    best_epoch = int(
        df.loc[best_index, "Epoch"]
    )
    best_val_loss = float(
        df.loc[best_index, "Validation_Loss"]
    )
else:
    best_val_loss = float(
        best_row["Validation_Loss"].iloc[0]
    )


# =========================================================
# 5. Plot the loss curves
# =========================================================

# A double-column plot is approximately 180 mm wide.
figure_width_mm = 180
figure_height_mm = 110
mm_to_inch = 1 / 25.4

fig, ax = plt.subplots(
    figsize=(
        figure_width_mm * mm_to_inch,
        figure_height_mm * mm_to_inch
    )
)


# Training loss
ax.plot(
    df["Epoch"],
    df["Train_Loss"],
    linestyle="-",
    linewidth=1.6,
    label="Training"
)


# Validation loss
validation_line, = ax.plot(
    df["Epoch"],
    df["Validation_Loss"],
    linestyle="--",
    linewidth=1.6,
    label="Validation"
)


# Best training epoch
ax.axvline(
    x=best_epoch,
    color="0.45",
    linestyle=":",
    linewidth=1.1,
    label=f"Best epoch = {best_epoch}"
)


# Minimum validation loss
ax.scatter(
    best_epoch,
    best_val_loss,
    s=25,
    color=validation_line.get_color(),
    edgecolor="white",
    linewidth=0.6,
    zorder=5
)


# =========================================================
# 6. Configure axes
# =========================================================

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE")

# Journal figures typically omit an in-panel title.
ax.grid(False)


epoch_min = int(df["Epoch"].min())
epoch_max = int(df["Epoch"].max())
tick_locator = MaxNLocator(nbins=5, steps=[1, 2, 5, 10], integer=True)
major_epochs = [
    int(tick)
    for tick in tick_locator.tick_values(epoch_min, epoch_max)
    if epoch_min < tick <= epoch_max
]
ax.set_xticks([epoch_min, *major_epochs])

ax.set_xlim(
    max(0, epoch_min - 1),
    epoch_max + 1
)


all_losses = pd.concat(
    [
        df["Train_Loss"],
        df["Validation_Loss"]
    ],
    ignore_index=True
)

loss_min = float(all_losses.min())
loss_max = float(all_losses.max())
loss_range = loss_max - loss_min

if loss_range <= 0:
    loss_range = max(
        abs(loss_max),
        0.1
    )

ax.set_ylim(
    max(
        0,
        loss_min - 0.12 * loss_range
    ),
    loss_max + 0.10 * loss_range
)


ax.legend(
    loc="best",
    frameon=False,
    handlelength=2.5
)


for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)


ax.tick_params(
    axis="both",
    which="major",
    direction="out"
)

fig.tight_layout()


# =========================================================
# 7. Save figures
# =========================================================

with BytesIO() as tiff_buffer:
    fig.savefig(
        tiff_buffer,
        dpi=600,
        format="tiff",
        bbox_inches="tight",
    )
    Path(OUTPUT_TIF).write_bytes(tiff_buffer.getbuffer())

fig.savefig(
    OUTPUT_PDF,
    format="pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_SVG,
    format="svg",
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    format="png",
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# 8. Report outputs
# =========================================================

print("=" * 60)
print("Figure S5绘制完成")
print(f"实际训练轮数：{epoch_max}")
print(f"最佳训练轮次：{best_epoch}")
print(f"最低验证损失：{best_val_loss:.6f}")
print("-" * 60)
print(f"图片输出目录：{OUTPUT_DIR}")
print(f"TIFF：{OUTPUT_TIF}")
print(f"PDF ：{OUTPUT_PDF}")
print(f"SVG ：{OUTPUT_SVG}")
print(f"PNG ：{OUTPUT_PNG}")
print("=" * 60)
