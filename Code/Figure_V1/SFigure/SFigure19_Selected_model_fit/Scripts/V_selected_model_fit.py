"""Draw SFigure19 from previously exported SHAP plotting inputs."""
import argparse
import gc
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[5]


EVENTS = ('Heatwave', 'Flood', 'DC', 'H5F', 'F5H')


TARGET_COL = 'DO'


COLOR_TRAIN = '#6F9FBB'


COLOR_TEST = '#E89DA0'


FIT_DOT_SIZE = 42


FONT_SIZE_TICK = 14


FONT_SIZE_LABEL = 17


def configure_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['mathtext.fontset'] = 'dejavusans'
    sns.set_theme(
        style='ticks',
        rc={
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'axes.linewidth': 1.1,
            'axes.labelcolor': '#2F2F2F',
            'xtick.color': '#2F2F2F',
            'ytick.color': '#2F2F2F'
        }
    )


def save_figure(fig, path, event):
    """Save figures with bounded canvas sizes and reduce raster DPI if memory is insufficient."""
    for text_obj in fig.findobj(match=matplotlib.text.Text):
        text_obj.set_fontweight('bold')
    path.parent.mkdir(parents=True, exist_ok=True)
    save_path = path
    filename = path.name
    RASTER_DPI = 300 if event == 'DC' else 600
    RASTER_FALLBACK_DPI = 180
    MAX_RASTER_PIXELS = 40_000_000 if event == 'DC' else float('inf')
    extension = path.suffix.lower()

    # PDF and SVG are vector formats and do not need an exceptionally high raster DPI.
    if extension in {'.pdf', '.svg'}:
        fig.savefig(save_path, bbox_inches='tight')
        return

    width_in, height_in = fig.get_size_inches()
    pixel_limited_dpi = (int(np.sqrt(MAX_RASTER_PIXELS / max(width_in * height_in, 1.0)))
                         if event == 'DC' else RASTER_DPI)
    save_dpi = max(72, min(RASTER_DPI, pixel_limited_dpi))

    def save_raster(dpi, bbox_inches):
        if extension in {'.tif', '.tiff'}:
            with BytesIO() as buffer:
                fig.savefig(buffer, format='tiff', dpi=dpi, bbox_inches=bbox_inches,
                            pil_kwargs={'compression': 'tiff_lzw'})
                with open(save_path, 'wb') as output:
                    output.write(buffer.getbuffer())
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)

    try:
        save_raster(save_dpi, 'tight')
    except MemoryError:
        # Clear objects left by the failed render, then retry with lower DPI and fixed bounds.
        gc.collect()
        retry_dpi = min(RASTER_FALLBACK_DPI, save_dpi)
        print(
            f'警告：保存 {filename} 时内存不足，'
            f'正以 {retry_dpi} DPI 重试。'
        )
        save_raster(retry_dpi, None)


def fit_figure(data, comparison, event):
    y_train = data['y_train']
    y_test = data['y_test']
    y_train_pred = data['train_pred']
    y_test_pred = data['test_pred']
    selected_model_name = str(data['selected_model'].item())
    row = comparison.loc[comparison.Model.eq(selected_model_name)].iloc[0]
    r2_train, rmse_train = row.Train_R2, row.Train_RMSE
    r2_test, rmse_test = row.Test_R2, row.Test_RMSE
    print('正在生成模型拟合图...')
    fig_fit = plt.figure(figsize=(8, 7))
    gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)
    ax_main = fig_fit.add_subplot(gs[0, 0])
    ax_right = fig_fit.add_subplot(gs[0, 1], sharey=ax_main)

    ax_main.scatter(y_train, y_train_pred, c=COLOR_TRAIN, alpha=0.74, edgecolors='white',
                    linewidths=0.35, s=FIT_DOT_SIZE, label='Training set')
    ax_main.scatter(y_test, y_test_pred, c=COLOR_TEST, alpha=0.82, edgecolors='white',
                    linewidths=0.35, s=FIT_DOT_SIZE, label='Test set')

    min_val = min(y_train.min(), y_train_pred.min(), y_test.min(), y_test_pred.min())
    max_val = max(y_train.max(), y_train_pred.max(), y_test.max(), y_test_pred.max())
    margin = (max_val - min_val) * 0.05
    lim_min, lim_max = min_val - margin, max_val + margin
    ax_main.set(xlim=(lim_min, lim_max), ylim=(lim_min, lim_max))
    ax_main.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='#3D3D3D', lw=1.5, label='1:1 line')

    z = np.polyfit(np.r_[y_train, y_test], np.r_[y_train_pred, y_test_pred], 1)
    ax_main.plot([lim_min, lim_max], np.poly1d(z)([lim_min, lim_max]), color='#242424', lw=1.8,
                 label='Fitted line')
    ax_main.set_xlabel(f'Observed {TARGET_COL} (mg L$^{{-1}}$)', fontsize=FONT_SIZE_LABEL)
    ax_main.set_ylabel(f'Predicted {TARGET_COL} (mg L$^{{-1}}$)', fontsize=FONT_SIZE_LABEL)
    ax_main.set_title(f'Selected model: {selected_model_name} ({event})', fontsize=FONT_SIZE_LABEL, loc='left', pad=10, fontweight='bold')
    ax_main.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
    ax_main.text(
        0.04, 0.95,
        f'Train $R^2$ = {r2_train:.4f}, RMSE = {rmse_train:.2f}\n'
        f'Test $R^2$ = {r2_test:.4f}, RMSE = {rmse_test:.2f}',
        transform=ax_main.transAxes, fontsize=FONT_SIZE_TICK, color='#3D3D3D', va='top'
    )
    ax_main.legend(loc='lower right', fontsize=10, frameon=False)

    sns.histplot(y=y_train_pred, ax=ax_right, color=COLOR_TRAIN, kde=True, bins=35,
                 alpha=0.55, stat='density', edgecolor='white', linewidth=0.35)
    sns.histplot(y=y_test_pred, ax=ax_right, color=COLOR_TEST, kde=True, bins=35,
                 alpha=0.55, stat='density', edgecolor='white', linewidth=0.35)
    ax_right.axis('off')
    sns.despine(ax=ax_main)

    return fig_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'Result/shap')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'V/Figure_V1/SFigure/SFigure19_Selected_model_fit/Images')
    parser.add_argument('--event', choices=EVENTS, nargs='+', default=EVENTS)
    args = parser.parse_args()
    configure_style()
    for event in args.event:
        basename = f'{event}_ThreeModel_SHAP_Analysis'
        stem = args.input_dir / basename
        source = Path(str(stem) + '_Plot_Data.npz')
        if not source.is_file():
            raise FileNotFoundError(f'{source}. Run the matching analyze_*_shap.py to prepare plot data.')
        with np.load(source, allow_pickle=False) as data:
            comparison = pd.read_csv(str(stem) + '_Model_Comparison.csv')
            fig = fit_figure(data, comparison, event)
            basename += '_Model_Fitting'
            for extension in ('png', 'pdf', 'tiff'):
                save_figure(fig, args.output_dir / f'{basename}.{extension}', event)
            plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()
