"""Plot Figure S4 response contrasts from the saved V5 M05 CSVs."""
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


def save_tiff(fig, path):
    # A memory stream avoids Windows libtiff failures on Chinese path names.
    stream = BytesIO()
    fig.savefig(stream, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    Path(path).write_bytes(stream.getvalue())


def draw(comparisons, output_dir, run_date):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    x = np.arange(6)
    for comparison, filename in [('Compound - Single', '复合与单一'), ('F5H - H5F', '事件顺序')]:
        data = comparisons.loc[(comparisons.Comparison == comparison) & (comparisons.Metric == 'Shock_Ratio_LongTerm')].set_index('Scenario').reindex(SCENARIOS)
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        ax.plot(x, data.Mean_Difference, marker='o', color='#24618c', linewidth=1.3)
        ax.axhline(0, color='0.5', linestyle='--', linewidth=.8)
        ax.set_xticks(x, list(SCENARIOS))
        ax.set(xlabel='Scenario', ylabel='Mean DO ratio difference (Shock-L)', title=comparison)
        fig.subplots_adjust(left=.16, right=.97, bottom=.21, top=.86)
        fig.savefig(output_dir / f'{run_date}_M05{filename}响应敏感性.png', dpi=600)
        fig.savefig(output_dir / f'{run_date}_M05{filename}响应敏感性.pdf')
        fig.savefig(output_dir / f'{run_date}_M05{filename}响应敏感性.svg')
        save_tiff(fig, output_dir / f'{run_date}_M05{filename}响应敏感性.tiff')
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

    paths = [locate('复合与单一响应对比'), locate('事件顺序响应对比')]
    comparisons = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for label in ['Compound - Single', 'F5H - H5F']:
        data = comparisons.loc[comparisons.Comparison.eq(label) & comparisons.Metric.eq('Shock_Ratio_LongTerm')]
        if data.Scenario.duplicated().any() or set(data.Scenario) != set(SCENARIOS):
            raise ValueError(f'Incomplete or duplicated Shock-L comparison: {label}')
    prefixes = {path.name.split('_M05')[0] for path in paths}
    if len(prefixes) != 1:
        raise ValueError('Mixed M05 analysis dates')
    run_date = prefixes.pop()
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    output_dir = (args.output_dir or ROOT / 'V/Figure_V1/SFigure/SFigure04_Definition_responses/Images' / inputs.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    draw(comparisons, output_dir, run_date)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
        raise ValueError('Input CSV changed during export')
    print(output_dir)


if __name__ == '__main__':
    main()
