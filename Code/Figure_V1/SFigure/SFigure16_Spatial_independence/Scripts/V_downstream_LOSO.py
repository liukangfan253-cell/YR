"""SFigure16: downstream leave-one-site-out sensitivity from saved M06 results."""
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


def draw(result, figure_dir, run_date):
    plt.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7.2,4.2)); fig.subplots_adjust(left=.12,right=.97,bottom=.28,top=.87)
    x=np.arange(len(result)); vals=result.Downstream_Mean_Shock_L
    ax.plot(x,vals,'o-',color='#24728C',linewidth=1.5,label='Downstream LOSO')
    for value,label,color,style in [(vals.iloc[0],'All downstream sites','#24728C',':'),
                                  (result.Upstream_Mean_Shock_L.iloc[0],'Upstream (unchanged)','#C77C36','--'),
                                  (result.Midstream_Mean_Shock_L.iloc[0],'Midstream (unchanged)','#6C657C','-.')]:
        ax.axhline(value,color=color,linestyle=style,linewidth=1.2,label=label)
    labels=result.Plot_Label.tolist()
    ax.set_xticks(x,labels); ax.set_ylabel('Site-equal mean Shock-L')
    ax.set_xlabel('Downstream site exclusion'); ax.set_ylim(0,max(1.08,1.1*result.Upstream_Mean_Shock_L.max(),1.1*result.Midstream_Mean_Shock_L.max()))
    ax.legend(loc='lower left',fontsize=8,frameon=False)
    fig.suptitle('Downstream leave-one-site-out sensitivity',fontsize=11)
    figure_dir.mkdir(parents=True,exist_ok=True)
    stem=figure_dir/f'{run_date}_M06下游LOSO分析图'
    fig.savefig(str(stem)+'.png',dpi=600); fig.savefig(str(stem)+'.pdf')
    with BytesIO() as buffer:
        fig.savefig(buffer, format='tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
        Path(str(stem) + '.tiff').write_bytes(buffer.getbuffer())
    fig.savefig(figure_dir/'M06_QA_preview.png',dpi=150); plt.close(fig)


def main():
    args = plot_parser(__doc__).parse_args()
    date, (source,) = result_tables(
        ROOT / 'Result/Data_preprocessing/Hydrological_matching', ('_M06_Downstream_LOSO.csv',), args.input_dir
    )
    out = args.output_dir or ROOT / 'V/Figure_V1/SFigure/SFigure16_Spatial_independence/Images' / date
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
