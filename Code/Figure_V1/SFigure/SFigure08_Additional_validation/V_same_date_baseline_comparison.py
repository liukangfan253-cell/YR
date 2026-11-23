"""Export the former Figure S5c from saved V5 event-date comparisons."""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parent
ROOT = FIGURE_DIR.parents[3]
sys.path.insert(0, str(ROOT / 'Code/03_Daily_DO_reconstruction/04_Continuous_gap_validation'))
from validation_labels import normalize_validation_labels
DEFAULT_SOURCE = ROOT / 'Result/Continuous_gap_validation/V1_Continuous_Gap_20261111_Validation/paired_baseline_comparison.csv'
STEM = 'FigureS8_Same_date_baseline_comparison'
DESIGNS = {
    'Length_specific': ('L', '-', 3.6),
    'Common_anchor': ('C', '--', 5.2),
}
BASELINES = {
    'Unmasked_BiLSTM_Attention': ('Unmasked', '#656D73', ('o', 's')),
    'Last_observation': ('Persistence', '#326B86', ('^', 'D')),
}
DELTA = 'Masked_minus_Baseline_MAE'
KEYS = ['Design', 'Length', 'Baseline']
NUMERIC = ['Length', 'N', 'Sites', 'Masked_MAE', 'Baseline_MAE', DELTA]


def load_comparisons(path):
    frame = normalize_validation_labels(pd.read_csv(path, encoding='utf-8-sig'))
    required = {*KEYS, *NUMERIC, 'Context'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing comparison columns: {", ".join(sorted(missing))}')
    frame = frame.loc[frame['Context'].eq('Event')].copy()
    if frame.empty:
        raise ValueError('Expected paired comparisons for event dates only.')

    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric, errors='raise')
    counts = frame[['Length', 'N', 'Sites']]
    if (not np.isfinite(frame[NUMERIC].to_numpy()).all()
            or not ((counts > 0) & (counts % 1 == 0)).to_numpy().all()
            or (frame['Sites'] > frame['N']).any()
            or (frame[['Masked_MAE', 'Baseline_MAE']] < 0).to_numpy().any()):
        raise ValueError('Expected finite MAEs and positive integer lengths and counts.')
    lengths = sorted(frame['Length'].astype(int).unique())
    indexed = frame.set_index(KEYS)
    expected = pd.MultiIndex.from_product([DESIGNS, lengths, BASELINES], names=KEYS)
    if not indexed.index.is_unique or set(indexed.index) != set(expected):
        raise ValueError('Expected one row per L/C design, gap length and baseline.')
    if not np.allclose(frame[DELTA], frame['Masked_MAE'] - frame['Baseline_MAE'],
                       rtol=0, atol=1e-9):
        raise ValueError('Saved difference must equal masked MAE minus baseline MAE.')

    matched = frame.groupby(['Design', 'Length'])[['N', 'Sites', 'Masked_MAE']].nunique()
    if (matched != 1).to_numpy().any():
        raise ValueError('Both baselines must use the same masked targets within each design and length.')
    longest = frame.loc[frame['Length'].eq(lengths[-1])]
    shared = longest.groupby('Baseline')[['N', 'Sites', 'Masked_MAE', 'Baseline_MAE', DELTA]].nunique()
    if (shared != 1).to_numpy().any():
        raise ValueError('The longest-gap comparison must share targets and results across L and C.')
    return indexed, lengths


def draw_comparisons(frame, lengths):
    # Four descriptive series, with a shared zero line and no newly estimated intervals.
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 0.7, 'legend.frameon': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })
    width_mm = 89
    height_mm = 78
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.21, right=0.96, bottom=0.17, top=0.73)
    fig.text(0.5, 0.94, 'Same-date baseline comparisons', ha='center', color='black',
             fontsize=9, weight='bold')

    ax.axhline(0, color='#AAB2B7', linewidth=0.7, zorder=1)
    legend_handles = []
    for baseline, (name, color, markers) in BASELINES.items():
        for (design, (code, line, size)), marker in zip(DESIGNS.items(), markers):
            values = frame.xs((design, baseline), level=('Design', 'Baseline')).loc[lengths, DELTA]
            # The larger hollow C marker leaves the L marker visible at the shared length.
            ax.plot(lengths, values, color=color, linestyle=line, marker=marker,
                    markersize=size, markerfacecolor=color if code == 'L' else 'none',
                    markeredgewidth=0.85, linewidth=1.0)
            legend_handles.append(Line2D(
                [], [], color='black', linestyle=line, marker=marker, markersize=size,
                markerfacecolor='black' if code == 'L' else 'none',
                markeredgewidth=0.85, linewidth=1.0, label=f'{code} · {name}',
            ))

    values = frame[DELTA]
    lower, upper = min(0, values.min()), max(0, values.max())
    pad = max((upper - lower) * 0.12, 0.015)
    x_pad = max((lengths[-1] - lengths[0]) * 0.065, 0.5)
    ax.set(xlabel='Masked gap length (days)', ylabel='Masked − baseline MAE (mg/L)',
           xticks=lengths, xlim=(lengths[0] - x_pad, lengths[-1] + x_pad),
           ylim=(lower - pad, upper + pad))
    ax.tick_params(direction='out', length=3)
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.755),
               ncol=2, borderaxespad=0, handlelength=1.9, columnspacing=1.3,
               labelcolor='black')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'V/Figure_V1/SFigure/SFigure08_Additional_validation/Images')
    args = parser.parse_args()

    frame, lengths = load_comparisons(args.input_csv)
    fig = draw_comparisons(frame, lengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / f'{STEM}.pdf')
    fig.savefig(args.output_dir / f'{STEM}.svg')
    with BytesIO() as tiff:
        fig.savefig(tiff, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
        (args.output_dir / f'{STEM}.tiff').write_bytes(tiff.getvalue())
    fig.savefig(args.output_dir / f'{STEM}.png', dpi=300)
    plt.close(fig)
    print(f'Exported {STEM} from {len(frame)} saved paired comparisons.')


if __name__ == '__main__':
    main()
