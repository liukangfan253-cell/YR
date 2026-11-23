"""Plot Figure S3 event counts from the saved V5 M05 CSVs."""
from pathlib import Path
import argparse
import hashlib
import os
from io import BytesIO
import numpy as np
import pandas as pd


FIGURE_DIR = Path(__file__).resolve().parent
ROOT = FIGURE_DIR.parents[3]
DEFAULT_INPUT = ROOT / 'Result/Event/Definition_sensitivity/20261111'
SCENARIOS = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']
EVENT_TYPES = ['Heatwave_Only', 'Flood_Only', 'Direct_Compound', 'H5F_Compound', 'F5H_Compound']


def save_tiff(fig, path):
    # A memory stream avoids Windows libtiff failures on Chinese path names.
    stream = BytesIO()
    fig.savefig(stream, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    Path(path).write_bytes(stream.getvalue())


def draw(counts, output_dir, run_date):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(6)
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
    labels = ['Heatwave only', 'Flood only', 'DC', 'H5F', 'F5H']
    c = counts.set_index('Scenario').reindex(SCENARIOS)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for event, label, color in zip(EVENT_TYPES, labels, colors):
        ax.plot(x, c[event], marker='o', color=color, label=label, linewidth=1.2, markersize=4)
    ax.set_xticks(x, list(SCENARIOS))
    ax.set(xlabel='Scenario', ylabel='Site-event records')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    fig.subplots_adjust(left=.12, right=.75, bottom=.17, top=.95)
    fig.savefig(output_dir / f'{run_date}_M05六方案事件数量图.png', dpi=600)
    fig.savefig(output_dir / f'{run_date}_M05六方案事件数量图.pdf')
    fig.savefig(output_dir / f'{run_date}_M05六方案事件数量图.svg')
    save_tiff(fig, output_dir / f'{run_date}_M05六方案事件数量图.tiff')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    inputs = args.input_dir.resolve()

    def locate(suffix):
        matches = list(inputs.glob(f'*_M05{suffix}.csv'))
        if len(matches) != 1:
            raise ValueError(f'Expected one saved {suffix} CSV in {inputs}; found {len(matches)}')
        return matches[0]

    paths = [locate('六方案事件数量对比')]
    counts = pd.read_csv(paths[0])
    if counts.Scenario.duplicated().any() or set(counts.Scenario) != set(SCENARIOS):
        raise ValueError('Counts must contain exactly S0-S5')
    prefixes = {path.name.split('_M05')[0] for path in paths}
    if len(prefixes) != 1:
        raise ValueError('Mixed M05 analysis dates')
    run_date = prefixes.pop()
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output_dir = (args.output_dir or ROOT / 'V/Figure_V1/SFigure/SFigure03_Definition_counts/Images' / inputs.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    draw(counts, output_dir, run_date)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        raise ValueError('Input CSV changed during export')
    print(output_dir)


if __name__ == '__main__':
    main()
