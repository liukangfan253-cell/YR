"""Draw SFigure20 from previously exported SHAP plotting inputs."""
import argparse
import gc
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import shap


ROOT = Path(__file__).resolve().parents[5]


EVENTS = ('Heatwave', 'Flood', 'DC', 'H5F', 'F5H')


GROUP_FEATURES = {
    'Water quality': ['pH', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN', 'CODₘₙ'],
    'Thermal-meteo': ['WT', 'Tmax', 'SSR', 'TP_M', 'WS'],
    'Hydro-sediment': ['WL', 'Q', 'Ss'],
    'Ecology': ['EVI']
}


FEATURE_LABELS = {
    'pH': 'pH', 'EC': 'EC', 'Turb': 'Turb', 'NH₃-N': r'NH$_3$-N',
    'TP': 'TP', 'TN': 'TN', 'CODₘₙ': r'COD$_{Mn}$', 'WT': 'WT',
    'Tmax': r'T$_{max}$', 'SSR': 'SSR', 'TP_M': 'TP_M', 'WS': 'WS',
    'WL': 'WL', 'Q': 'Q', 'Ss': 'Ss', 'EVI': 'EVI'
}


COLOR_WATER_QUALITY = '#88CEE6'


COLOR_THERMAL_METEO = '#E89DA0'


COLOR_HYDRO_SEDIMENT = '#F6C8A8'


COLOR_ECOLOGICAL = '#B2D3A4'


GROUP_COLORS = {
    'Water quality': COLOR_WATER_QUALITY,
    'Thermal-meteo': COLOR_THERMAL_METEO,
    'Hydro-sediment': COLOR_HYDRO_SEDIMENT,
    'Ecology': COLOR_ECOLOGICAL
}


SHAP_CMAP = 'RdBu_r'


BAR_ALPHA = 0.96


SHAP_DOT_SIZE = 22


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


def explanation_figure(data, event):
    FEATURE_COLS = data['feature_names'].tolist()
    X_shap = pd.DataFrame(data['feature_values'], columns=FEATURE_COLS)
    shap_values = data['shap_values']
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_percent = mean_abs_shap / mean_abs_shap.sum() * 100
    feature_group = {
        feature: group
        for group, features in GROUP_FEATURES.items()
        for feature in features
    }
    shap_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Feature_label': [FEATURE_LABELS.get(feature, feature) for feature in FEATURE_COLS],
        'Group': [feature_group.get(feature, 'Other') for feature in FEATURE_COLS],
        'SHAP_percent': shap_percent
    }).sort_values('SHAP_percent', ascending=True)

    if (shap_df['Group'] == 'Other').any():
        ungrouped = shap_df.loc[shap_df['Group'] == 'Other', 'Feature'].tolist()
        raise ValueError(f'下列特征未分组，请在 GROUP_FEATURES 中补充：{ungrouped}')

    shap_df['Color'] = shap_df['Group'].map(GROUP_COLORS)
    group_shap_df = (
        shap_df.groupby('Group', sort=False)['SHAP_percent'].sum()
        .reindex(GROUP_FEATURES.keys())
        .dropna()
    )
    print('正在生成四类别 SHAP 双面板图...')
    fig_shap, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1.04, 1.20]}
    )

    bars = ax1.barh(shap_df['Feature_label'], shap_df['SHAP_percent'], color=shap_df['Color'],
                    alpha=BAR_ALPHA, height=0.68)
    ax1.set_xlabel('Relative SHAP contribution (%)', fontsize=FONT_SIZE_LABEL)
    ax1.set_title(f'{event} Feature importance', fontsize=FONT_SIZE_LABEL + 1, loc='left', pad=12,
                  fontweight='bold')
    ax1.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
    ax1.set_xlim(0, shap_df['SHAP_percent'].max() * 1.30)
    sns.despine(ax=ax1)

    legend_handles = [
        mpatches.Patch(color=GROUP_COLORS[group], alpha=BAR_ALPHA, label=group)
        for group in GROUP_FEATURES if group in group_shap_df.index
    ]
    # Put the legend in the space above the pie chart to avoid covering the tallest importance bar and its value label.
    ax1.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.76, 0.56),
               frameon=False, fontsize=12, handlelength=1.2, labelspacing=0.55,
               borderaxespad=0.0)

    axins = inset_axes(ax1, width='39%', height='39%', loc='lower right', borderpad=1.1)
    pie_labels = [label.replace('\n', ' ') for label in group_shap_df.index]
    wedges, _ = axins.pie(
        group_shap_df.values,
        colors=[GROUP_COLORS[group] for group in group_shap_df.index],
        startangle=90,
        counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.1}
    )
    axins.set_title('Relative group contribution (%)', fontsize=12, pad=4)
    axins.axis('equal')

    # Pass the standard variable names to the SHAP plot to keep labels consistent across panels.
    X_shap_display = X_shap.rename(columns=FEATURE_LABELS)
    plt.sca(ax2)
    shap.summary_plot(
        shap_values, X_shap_display, cmap=SHAP_CMAP, show=False,
        plot_size=None, max_display=len(FEATURE_COLS)
    )
    ax2.set_title(f'{event} SHAP value distribution', fontsize=FONT_SIZE_LABEL + 1, loc='left', pad=12,
                  fontweight='bold')
    ax2.set_xlabel('SHAP value (mg L$^{-1}$)', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('')
    ax2.tick_params(axis='both', labelsize=FONT_SIZE_TICK)

    for collection in ax2.collections:
        if isinstance(collection, matplotlib.collections.PathCollection):
            collection.set_sizes([SHAP_DOT_SIZE] * len(collection.get_offsets()))

    if len(fig_shap.axes) > 2:
        cbar_ax = fig_shap.axes[-1]
        cbar_ax.tick_params(labelsize=FONT_SIZE_TICK - 1)
        cbar_ax.set_ylabel('Feature value', fontsize=FONT_SIZE_TICK)

    # Narrow the gap between panels while reserving room for the y-axis labels in panel (b).
    fig_shap.suptitle('SHAP explanation: all retained records; selected training-set model',
                      fontsize=FONT_SIZE_TICK, y=0.99)
    fig_shap.subplots_adjust(wspace=0.16, left=0.08, right=0.93, top=0.91, bottom=0.10)
    return fig_shap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'Result/shap')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'V/Figure_V1/SFigure/SFigure20_SHAP_exploratory/Images')
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
            fig = explanation_figure(data, event)
            for extension in ('png', 'pdf', 'tiff'):
                save_figure(fig, args.output_dir / f'{basename}.{extension}', event)
            plt.close(fig)
    print(args.output_dir)


if __name__ == "__main__":
    main()
