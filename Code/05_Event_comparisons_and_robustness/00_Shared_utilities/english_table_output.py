"""English publication-table serialization, without changing analytical data.

Apply at export boundaries: Chinese internal keys remain available to existing
analysis code. Unknown Chinese text raises an error rather than being dropped or
guessed. Chinese characters in literal source paths use reversible Unicode
escapes; the underlying source files and provenance identifiers are not renamed.
"""
from __future__ import annotations
import csv
import io
import re

CJK = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
HEADERS = {
    '省份':'Province','城市':'City','流域':'Basin','河流':'River',
    '站点名称':'Station_Name','经度':'Longitude','纬度':'Latitude',
    '监测时间':'Monitoring_Time','水温':'Water_Temperature',
    '溶解氧':'Dissolved_Oxygen','电导率':'Electrical_Conductivity',
    '浊度':'Turbidity','高锰酸盐指数':'Permanganate_Index','氨氮':'Ammonia_Nitrogen',
    '总有机碳':'Total_Organic_Carbon','总磷':'Total_Phosphorus','总氮':'Total_Nitrogen',
    '叶绿素α':'Chlorophyll_a','藻密度':'Algal_Density',
    '水质':'Water_Quality_Class','站点':'Station_Situation',
    '指标':'Metric','比较':'Contrast','模型':'Model','模型样本量':'Model_N',
    '模型断面数':'Model_Sites','效应类型':'Effect_Type','单位':'Unit',
    '调整后效应量':'Adjusted_Effect','CI下限':'CI_Lower','CI上限':'CI_Upper',
    '原始p值':'P_Raw','Holm校正p值':'P_Holm','相对变化百分比':'Relative_Change_Pct',
    '显著性':'Significance','状态':'Fit_Status',
    '目标组':'Target_Group','目标组样本量':'Target_Group_N',
    '目标组断面数':'Target_Group_Sites','目标组原始均值':'Target_Group_Unadjusted_Mean',
    '目标组原始中位数':'Target_Group_Unadjusted_Median',
    '参照组':'Reference_Group','参照组样本量':'Reference_Group_N',
    '参照组断面数':'Reference_Group_Sites','参照组原始均值':'Reference_Group_Unadjusted_Mean',
    '参照组原始中位数':'Reference_Group_Unadjusted_Median',
    'log尺度差值或原始斜率差':'Log_Scale_Contrast_or_Slope_Difference',
    '标准误':'SE','随机截距方差':'Random_Intercept_Variance',
    '公式':'Formula','断面数':'Sites','估计方法':'Estimation_Method',
    'CI/p方法':'CI_P_Method','优化器':'Optimizer','收敛':'Converged',
    '残差方差':'Residual_Variance','近零方差':'Near_Zero_Variance','警告':'Warnings',
    '可靠':'Reliable','错误':'Error',
    '完整参数协方差最小特征值':'Minimum_Eigenvalue_Full_Parameter_Covariance',
    'Powell核验收敛':'Powell_Check_Converged',
    'Powell核验随机方差':'Powell_Check_Random_Intercept_Variance',
    'Powell核验最大固定系数差':'Powell_Check_Max_Fixed_Coefficient_Difference',
    'Powell核验logLik差':'Powell_Check_LogLik_Difference',
    'Powell核验警告':'Powell_Check_Warnings',
    '分组方式':'Grouping_Factor','组':'Group','有效样本量':'Valid_N',
    '原始均值':'Unadjusted_Mean','原始中位数':'Unadjusted_Median',
    '标准差':'SD','最小值':'Minimum','最大值':'Maximum',
    '文件':'File','记录数':'Records','字段':'Fields','断面':'Site_IDs','断面识别':'Site_Field',
    '估计':'Estimate','双侧p':'P_Two_Sided',
}
VALUES = {
    '河南省':'Henan','濮阳市':'Puyang','黄河流域':'Yellow River Basin',
    '金堤河':'Jindi River','贾垓桥':'Jiagaiqiao',
    '(无量纲)':'(dimensionless)','无量纲':'dimensionless',
    '类别':'Class','情况':'Situation',
    '劣Ⅴ':'Worse than Class V','劣Ⅴ类':'Worse than Class V',
    '劣V':'Worse than Class V','劣V类':'Worse than Class V',
    '事件数<10':'Events <10','仅1–2断面':'Only 1–2 sites',
    '响应比':'Response_ratio','斜率差':'Slope_difference',
    '证据不足以支持差异':'Insufficient_evidence_of_difference',
    '可靠':'Reliable','失败':'Failed','显著':'Significant',
    '无法可靠估计':'Not_reliably_estimable','等待Holm校正':'Pending_Holm_adjustment',
    '双侧渐近正态Wald；95%CI未作多重校正':
        'Two-sided asymptotic normal Wald; 95% CI not multiplicity-adjusted',
    '断面不足、比较组缺失或五类模型有事件类型缺失':
        'Insufficient sites, missing comparison groups or missing event types in the five-type model',
    '固定效应设计矩阵秩不足':'Rank-deficient fixed-effects design matrix',
    '比较参数标准误无法估计':'Contrast standard error could not be estimated',
    '无明确字段；不推断文件名':'No explicit site field; not inferred from filename',
}
FILE_NAMES = {
    'CSV结构清单':'CSV_Inventory','事件级统计分析数据':'Event_Level_Statistical_Data',
    '输入来源与校验':'Input_Provenance','三指标描述性统计':'Descriptive_Statistics',
    '两组核心比较统计结果':'Core_Contrasts','模型诊断明细':'Model_Diagnostics',
    '统计分析结果汇总':'Statistical_Analysis_Results',
    '20261016_V13_新增敏感性分析':'20261016_V13_Observation_Coverage_Sensitivity',
}
SHEETS = {'核心比较':'Core_Contrasts','描述性统计':'Descriptive_Statistics',
          '事件级数据':'Event_Level_Data','模型诊断':'Model_Diagnostics','输入来源':'Input_Provenance'}

def english_header(value):
    if not isinstance(value,str) or not CJK.search(value):return value
    if value in HEADERS:return HEADERS[value]
    # Reproduction tables append _Old/_Current/_Abs_Difference to model fields.
    for suffix in ('_Abs_Difference','_Current','_Old'):
        if value.endswith(suffix) and value[:-len(suffix)] in HEADERS:
            return HEADERS[value[:-len(suffix)]]+suffix
    raise ValueError(f'Unmapped Chinese table header: {value!r}')

def english_value(value):
    if not isinstance(value,str) or not CJK.search(value):return value
    if value in VALUES:return VALUES[value]
    # Literal source paths are provenance, not prose to be translated or renamed.
    if re.match(r'^(?:[A-Za-z]:[\\/]|(?:Data|Result|Submission|Code)[\\/])',value):
        return CJK.sub(lambda m:'\\u'+format(ord(m[0]),'04x'),value)
    translated=value
    for old,new in sorted(VALUES.items(),key=lambda item:-len(item[0])):
        translated=translated.replace(old,new)
    if CJK.search(translated):raise ValueError(f'Unmapped Chinese table value: {value!r}')
    return translated

def english_table(frame):
    """Return an English-labelled copy; never mutate the caller's frame/dtypes."""
    out=frame.copy(deep=True)
    labels=[english_header(x) for x in frame.columns]
    if len(labels)!=len(set(labels)):raise ValueError('English column labels collide')
    for i in range(len(frame.columns)):
        column=frame.iloc[:,i]
        if column.map(lambda x:isinstance(x,str) and bool(CJK.search(x))).any():
            out.isetitem(i,column.map(english_value))
    out.columns=labels
    return out

def english_csv_bytes(raw):
    """Translate text cells while keeping every numeric/date token verbatim.

    Unchanged CSVs are returned byte-for-byte. Changed CSVs retain their UTF-8
    BOM choice, delimiter, row order, column order and original newline convention.
    """
    bom=raw.startswith(b'\xef\xbb\xbf');text=raw.decode('utf-8-sig')
    rows=list(csv.reader(io.StringIO(text,newline='')))
    if not rows:return raw,0
    count=0;changed=[]
    for i,row in enumerate(rows):
        new=[english_header(x) if i==0 else english_value(x) for x in row]
        count+=sum(a!=b for a,b in zip(row,new));changed.append(new)
    if not count:return raw,0
    target=io.StringIO(newline='')
    csv.writer(target,lineterminator='\r\n' if '\r\n' in text else '\n').writerows(changed)
    return target.getvalue().encode('utf-8-sig' if bom else 'utf-8'),count
