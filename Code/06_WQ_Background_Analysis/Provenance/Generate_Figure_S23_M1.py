from pathlib import Path
import importlib.util, shutil
ROOT=Path('E:/20261111_YR_DO')
OUT=ROOT/'20261122_WQ_Background_Revision'
# Render M1 from existing CSVs with the original style and all 66 points.
spec=importlib.util.spec_from_file_location('wqplot',ROOT/'Code/WQ_Background_Analysis/02_WQ_Background_Plot.py')
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
plot.FIGURES=OUT/'03_Supplementary_Data'/'Supplementary_Data_S10'/'06_Figures'
plot.FIGURES.mkdir(parents=True,exist_ok=True)
original_read=plot.read
def m1_read(folder,name):
 d=original_read(folder,name)
 if name=='A02_Station_5Year_WQ_Statistics.csv':d['M3']=d['M1']
 elif name=='A02_Reach_5Year_WQ_Statistics.csv':
  d=d.loc[d.Method.eq('M1')].copy();d['Method']='M3'
 return d
plot.read=m1_read
original_save=plot.save
plot.save=lambda fig,name: original_save(fig,'Figure_S23_Longterm_WQ_Background_M1')
plot.configure();plot.longterm()
for n,old in [(24,'Fig_A3_Matched_Event_WQ_Changes'),(25,'Fig_A4_Low_DO_WQ_Associations'),(26,'Fig_A4b_Downstream_Low_DO_WQ')]:
 for ext in ['pdf','svg','tiff','png']:
  shutil.copy2(ROOT/'Result/WQ_Background_Analysis/06_Figures'/f'{old}.{ext}',plot.FIGURES/f'Figure_S{n}.{ext}')

