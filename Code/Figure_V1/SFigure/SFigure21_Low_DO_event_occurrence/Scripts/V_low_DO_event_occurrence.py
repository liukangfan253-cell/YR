"""Draw the former S7b panel from the saved V5 M07 event-type summary."""

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parents[1]
ROOT = FIGURE_DIR.parents[3]
DEFAULT_SOURCE = (ROOT / 'Result/Decrease/DO_observation_sensitivity/20261111'
                  / '20261111_M07_Event_Type_Summary.csv')
STEM = 'FigureS21_Low_DO_event_occurrence'
EVENTS = {
    'Heatwave_Only': 'Heatwave',
    'Flood_Only': 'High-flow',
    'Direct_Compound': 'DC',
    'H5F_Compound': 'H5F',
    'F5H_Compound': 'F5H',
}
LOW_COUNT = 'Below5_Event_N'
ELIGIBLE = 'Below5_Eligible_Event_N'
PERCENT = 'Below5_Event_Pct'


def load_occurrence(path):
    frame = pd.read_csv(path, encoding='utf-8-sig')
    required = {'Subset', 'Reach', 'Weighting', 'Event_Type', LOW_COUNT, ELIGIBLE, PERCENT}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing M07 columns: {", ".join(sorted(missing))}')
    full = frame.loc[frame['Subset'].eq('Full') & frame['Reach'].eq('All')
                     & frame['Weighting'].eq('Event_Equal')].set_index('Event_Type')
    if not full.index.is_unique or set(full.index) != {'All', *EVENTS}:
        raise ValueError('Expected the five event types and the all-event row in the full-series summary.')

    counts = full[[LOW_COUNT, ELIGIBLE]].apply(pd.to_numeric, errors='raise')
    if (not np.isfinite(counts.to_numpy()).all() or (counts % 1 != 0).to_numpy().any()
            or not counts[ELIGIBLE].gt(0).all()
            or not counts[LOW_COUNT].between(0, counts[ELIGIBLE]).all()):
        raise ValueError('Low-DO counts must be integers between zero and the eligible event count.')
    counts = counts.astype(int)
    groups = counts.loc[list(EVENTS)].copy()
    if not counts.loc['All'].eq(groups.sum()).all():
        raise ValueError('Event-type counts do not sum to the all-event counts.')
    computed = 100 * counts[LOW_COUNT] / counts[ELIGIBLE]
    if not np.allclose(full[PERCENT], computed, rtol=0, atol=1e-8):
        raise ValueError('Saved percentages disagree with the low-DO and eligible counts.')
    groups[PERCENT] = computed.loc[groups.index]
    return groups


def draw_occurrence(groups):
    # One descriptive panel showing how many completed-series events cross the daily DO cutoff.
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.labelsize': 8, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    })
    width_mm = 89
    height_mm = 70
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.89)
    x = np.arange(len(groups))
    ax.bar(x, groups[PERCENT], width=0.60, color='#447F9A')
    pad = max(groups[PERCENT].max() * 0.04, 0.8)
    for position, (_, row) in zip(x, groups.iterrows()):
        count = f'{int(row[LOW_COUNT])}/{int(row[ELIGIBLE])}'
        ax.text(position, row[PERCENT] + pad, f'{count}\n{row[PERCENT]:.1f}%',
                ha='center', va='bottom', fontsize=7)
    ax.set(xlabel='Event type', ylabel='Events with daily DO < 5 mg/L (%)',
           xticks=x, xticklabels=[EVENTS[key] for key in groups.index],
           xlim=(-0.55, len(groups) - 0.45),
           ylim=(0, min(100, max(groups[PERCENT].max() + 6 * pad, 10))))
    ax.tick_params(direction='out', length=3)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'V/Figure_V1/SFigure/SFigure21_Low_DO_event_occurrence/Images')
    args = parser.parse_args()

    groups = load_occurrence(args.input_csv)
    fig = draw_occurrence(groups)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / f'{STEM}.pdf')
    fig.savefig(args.output_dir / f'{STEM}.svg')
    with BytesIO() as tiff:
        fig.savefig(tiff, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
        (args.output_dir / f'{STEM}.tiff').write_bytes(tiff.getvalue())
    fig.savefig(args.output_dir / f'{STEM}.png', dpi=300)
    plt.close(fig)
    print(groups.to_string())


if __name__ == '__main__':
    main()
