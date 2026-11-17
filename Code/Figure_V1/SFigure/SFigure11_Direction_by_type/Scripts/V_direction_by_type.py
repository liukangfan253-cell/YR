"""SFigure11: direction disagreement by event type from saved M04 results."""
import argparse
from io import BytesIO
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[5]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVENT_TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']
COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
LABELS = ['Heatwave only', 'Flood only', 'DC', 'H5F', 'F5H']

def save_figure(fig, name, figure_dir, run_date):
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f'{run_date}_{name}.png', dpi=600)
    fig.savefig(figure_dir / f'{run_date}_{name}.pdf')
    fig.savefig(figure_dir / f'{run_date}_{name}.svg')
    with BytesIO() as buffer:
        fig.savefig(buffer, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
        (figure_dir / f'{run_date}_{name}.tiff').write_bytes(buffer.getbuffer())

def draw(flips, figure_dir, run_date):
    # The bars show descriptive proportions by event type. No type-specific confidence intervals or significance tests are shown.
    rows = flips.loc[flips.Level.eq('Event_Type')].set_index('Event_Type').reindex(EVENT_TYPES)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(np.arange(5), rows.Flip_Rate_pct, color=COLORS, width=.6)
    ax.set_xticks(np.arange(5), LABELS)
    ax.set_ylabel('Direction flip rate (%)')
    ax.set_ylim(0, 100)
    for bar, (_, row) in zip(bars, rows.iterrows()):
        if np.isfinite(row.Flip_Rate_pct):
            ax.annotate(f'{row.Flip_Rate_pct:.1f}%', (bar.get_x()+bar.get_width()/2, bar.get_height()),
                         xytext=(0, 5), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.set_xticklabels([f'{label}\n(n={int(r.n_decidable)}, sites={int(r.n_sites_decidable)})'
                        for label, (_, r) in zip(LABELS, rows.iterrows())], fontsize=8)
    fig.text(.10, .015, 'n: classifiable paired events (Neutral excluded); descriptive rates, no type-specific CI.', fontsize=8)
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.94)
    save_figure(fig, 'M04事件类型方向改变比例', figure_dir, run_date)
    plt.close(fig)


def main():
    args = plot_parser(__doc__).parse_args()
    date, (source,) = result_tables(
        ROOT / 'Result/Decrease/Paired_baselines', ('_M04偏离方向改变比例.csv',), args.input_dir
    )
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    out = args.output_dir or ROOT / 'V/Figure_V1/SFigure/SFigure11_Direction_by_type/Images' / date
    draw(pd.read_csv(source), out, date)
    print(out)


# Select a complete saved result batch.
def plot_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input-dir', type=Path, help='Saved result batch. Defaults to the latest complete batch.')
    parser.add_argument('--output-dir', type=Path, help='Override the default figure output directory.')
    return parser


def result_tables(base, suffixes, input_dir=None):
    directories = [input_dir] if input_dir is not None else sorted(
        (path for path in base.iterdir() if path.is_dir()), reverse=True
    )
    for directory in directories:
        for first in sorted(directory.glob('*' + suffixes[0]), reverse=True):
            prefix = first.name.removesuffix(suffixes[0])
            tables = [directory / (prefix + suffix) for suffix in suffixes]
            if all(path.is_file() for path in tables):
                return prefix, tables
    raise FileNotFoundError(f'No complete result batch for {suffixes} in {input_dir or base}')


if __name__ == '__main__':
    main()
