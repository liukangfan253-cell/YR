# -*- coding: utf-8 -*-
"""
Compare random forest, XGBoost and Extra Trees with Bayesian tuning and use SHAP for the selected model.

Feature groups
1. Water quality: pH, EC, Turb, NH₃-N, TP, TN and CODₘₙ
2. Thermal and meteorological conditions: WT, Tmax, SSR, TP_M and WS
3. Hydrology and sediment: WL, Q and Ss
4. Ecology: EVI

DO is the response. T, Tmin, Td, SP, SD and SWE are excluded as separate predictors because they duplicate retained information or lack a clear basis in the process grouping.
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

warnings.filterwarnings('ignore')


try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        '未安装 xgboost，请运行: pip install xgboost'
    ) from exc

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
except ImportError:
    raise ImportError('未安装 scikit-optimize，请运行: pip install scikit-optimize')


# ===================== 1. User configuration =====================
DATA_FILE = str(ROOT / 'Data/Event_N/Perfect_Events_Data/Overall/Direct_Compound_Perfect_Overall.csv')
TARGET_COL = 'DO'
OUTPUT_DIR = str(ROOT / 'Result/shap')
OUTPUT_BASENAME = 'DC_ThreeModel_SHAP_Analysis'
# At 600 DPI, a 20 × 9 inch SHAP figure has roughly 130 million pixels.
# RendererAgg can consume substantial memory at that size. Use 300 DPI for the raster export.

# RF-SHAP inputs retained after correlation screening and classification by process.
FEATURE_COLS = [
    'pH', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN', 'CODₘₙ',       # Water quality
    'WT', 'Tmax', 'SSR', 'TP_M', 'WS',                      # Thermal and meteorological conditions
    'WL', 'Q', 'Ss',                                        # Hydrology and sediment
    'EVI'                                                   # Ecology
]

# Use the same order for the four process groups in the bar-chart legend and pie chart.
GROUP_FEATURES = {
    'Water quality': ['pH', 'EC', 'Turb', 'NH₃-N', 'TP', 'TN', 'CODₘₙ'],
    'Thermal-meteo': ['WT', 'Tmax', 'SSR', 'TP_M', 'WS'],
    'Hydro-sediment': ['WL', 'Q', 'Ss'],
    'Ecology': ['EVI']
}

# Standard variable names used in figures.
FEATURE_LABELS = {
    'pH': 'pH', 'EC': 'EC', 'Turb': 'Turb', 'NH₃-N': r'NH$_3$-N',
    'TP': 'TP', 'TN': 'TN', 'CODₘₙ': r'COD$_{Mn}$', 'WT': 'WT',
    'Tmax': r'T$_{max}$', 'SSR': 'SSR', 'TP_M': 'TP_M', 'WS': 'WS',
    'WL': 'WL', 'Q': 'Q', 'Ss': 'Ss', 'EVI': 'EVI'
}

MODEL_SPACES = {
    'Random Forest': {
        'n_estimators': Integer(150, 500),
        'max_depth': Integer(3, 10),
        'min_samples_split': Integer(10, 50),
        'min_samples_leaf': Integer(5, 30),
        'max_features': Categorical(
            ['sqrt', 'log2', 0.5, 0.7, 1.0]
        ),
        'max_samples': Real(0.6, 0.9)
    },
    'Extra Trees': {
        'n_estimators': Integer(150, 500),
        'max_depth': Integer(3, 12),
        'min_samples_split': Integer(2, 50),
        'min_samples_leaf': Integer(1, 30),
        'max_features': Categorical(
            ['sqrt', 'log2', 0.5, 0.7, 1.0]
        )
    },
    'XGBoost': {
        'n_estimators': Integer(200, 1000),
        'learning_rate': Real(
            0.01, 0.15, prior='log-uniform'
        ),
        'max_depth': Integer(2, 8),
        'min_child_weight': Integer(1, 20),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(0.0, 2.0),
        'reg_lambda': Real(
            0.1, 20.0, prior='log-uniform'
        ),
        'gamma': Real(0.0, 1.0)
    }
}

N_BAYESIAN_ITER = 50
CV_FOLDS = 5
RANDOM_STATE = 42

# Select the best model by training-set cross-validation RMSE.
# Reserve the independent test set for final performance reporting.
MODEL_SELECTION_METRIC = 'CV_RMSE'

# Calculate SHAP values for all records that pass cleaning.
SHAP_USE_ALL_DATA = True



# ===================== 2. Data output =====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'分析数据将保存至: {OUTPUT_DIR}')


# ===================== 3. Load and clean the data =====================
print('\n正在加载数据...')
try:
    df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(DATA_FILE, encoding='gbk')

available_features = [feature for feature in FEATURE_COLS if feature in df.columns]
missing_features = [feature for feature in FEATURE_COLS if feature not in df.columns]
if missing_features:
    print(f'警告：下列预设特征在数据中不存在，已跳过：{missing_features}')
if not available_features:
    raise ValueError('数据中没有任何可用的预设特征，请检查 FEATURE_COLS 和列名。')
if TARGET_COL not in df.columns:
    raise ValueError(f'数据中不存在目标变量列：{TARGET_COL}')

FEATURE_COLS = available_features
initial_len = len(df)
analysis_df = df[[TARGET_COL] + FEATURE_COLS].dropna().copy()
X = analysis_df[FEATURE_COLS]
y = analysis_df[TARGET_COL]
print(f'删除目标变量和特征空值后样本数: {len(X)} (原始 {initial_len})')


# ===================== 4. Train, tune and evaluate the three models =====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)
print(
    f'随机划分完成：训练集 {len(X_train)} 个，'
    f'测试集 {len(X_test)} 个'
)


def create_estimators():
    """Use the same data split and cross-validation folds for all three models."""
    return {
        'Random Forest': RandomForestRegressor(
            criterion='squared_error',
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        'Extra Trees': ExtraTreesRegressor(
            criterion='squared_error',
            bootstrap=False,
            random_state=RANDOM_STATE,
            n_jobs=1
        ),
        'XGBoost': XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0
        )
    }


from shap_plot_data import prepare_saved_plot_data, save_plot_data

parser = argparse.ArgumentParser(description='Fit the three SHAP models and save analysis and plotting data.')
parser.add_argument('--prepare-saved-figures', action='store_true',
                    help='Replay just the saved winning fit and validate it against saved outputs.')
args = parser.parse_args()
plot_stem = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME)
if args.prepare_saved_figures:
    prepare_saved_plot_data(plot_stem, create_estimators(), X, X_train, X_test, y_train, y_test)
    raise SystemExit(0)

model_results = {}
comparison_rows = []
search_history_frames = []

for model_name, estimator in create_estimators().items():
    print(
        f'\n正在优化 {model_name}：'
        f'{N_BAYESIAN_ITER} 次贝叶斯搜索...'
    )

    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=MODEL_SPACES[model_name],
        n_iter=N_BAYESIAN_ITER,
        cv=CV_FOLDS,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=False
    )
    search.fit(X_train, y_train)

    model = search.best_estimator_
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )
    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    test_r2 = r2_score(y_test, test_pred)
    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )
    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )

    best_cv_rmse = -float(
        search.best_score_
    )
    cv_rmse_std = float(
        search.cv_results_[
            'std_test_score'
        ][search.best_index_]
    )

    model_results[model_name] = {
        'model': model,
        'best_params': dict(
            search.best_params_
        ),
        'best_cv_rmse': best_cv_rmse,
        'cv_rmse_std': cv_rmse_std,
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_r2': train_r2,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse
    }

    comparison_rows.append({
        'Model': model_name,
        'Best_CV_RMSE': best_cv_rmse,
        'CV_RMSE_Std': cv_rmse_std,
        'Train_R2': train_r2,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Test_R2': test_r2,
        'Test_MAE': test_mae,
        'Test_RMSE': test_rmse,
        'Train_Test_R2_Gap':
            train_r2 - test_r2
    })

    history_df = pd.DataFrame(
        search.cv_results_
    )
    history_df.insert(
        0,
        'Model',
        model_name
    )
    search_history_frames.append(
        history_df
    )

    print(
        f'{model_name}: '
        f'CV RMSE={best_cv_rmse:.4f}, '
        f'Test R2={test_r2:.4f}, '
        f'Test RMSE={test_rmse:.4f}'
    )


comparison_df = pd.DataFrame(
    comparison_rows
).sort_values(
    by=[
        'Best_CV_RMSE',
        'CV_RMSE_Std'
    ],
    ascending=[
        True,
        True
    ]
).reset_index(drop=True)

comparison_df[
    'Selection_Rank'
] = np.arange(
    1,
    len(comparison_df) + 1
)

selected_model_name = comparison_df.loc[
    0,
    'Model'
]

selected_result = model_results[
    selected_model_name
]

best_model = selected_result['model']
best_params = selected_result[
    'best_params'
]

y_train_pred = selected_result[
    'train_pred'
]
y_test_pred = selected_result[
    'test_pred'
]

r2_train = selected_result['train_r2']
mae_train = selected_result['train_mae']
rmse_train = selected_result[
    'train_rmse'
]

r2_test = selected_result['test_r2']
mae_test = selected_result['test_mae']
rmse_test = selected_result[
    'test_rmse'
]

print('\n三模型比较：')
print(
    comparison_df[
        [
            'Selection_Rank',
            'Model',
            'Best_CV_RMSE',
            'CV_RMSE_Std',
            'Test_R2',
            'Test_MAE',
            'Test_RMSE'
        ]
    ].to_string(index=False)
)

print(
    f'\n最终模型：{selected_model_name}'
)
print(
    '选择标准：训练集内部五折 CV-RMSE 最低；'
    '测试集仅用于最终评价。'
)
print(
    f'Train: R2 = {r2_train:.4f}, '
    f'MAE = {mae_train:.4f}, '
    f'RMSE = {rmse_train:.4f}'
)
print(
    f'Test : R2 = {r2_test:.4f}, '
    f'MAE = {mae_test:.4f}, '
    f'RMSE = {rmse_test:.4f}'
)


# ===================== 6. Compute SHAP values and summarize by group =====================
print(
    f'正在计算 {selected_model_name} 的 SHAP 值...'
)
print(
    f'SHAP解释对象：全部清洗后数据，共 {len(X)} 条记录。'
)
explainer = shap.TreeExplainer(
    best_model
)
X_shap = X.copy()
shap_values = explainer.shap_values(
    X_shap
)

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

group_shap_df = (
    shap_df.groupby('Group', sort=False)['SHAP_percent'].sum()
    .reindex(GROUP_FEATURES.keys())
    .dropna()
)


# ===================== 8. Export results =====================
shap_df[['Feature', 'Feature_label', 'Group', 'SHAP_percent']].sort_values(
    'SHAP_percent', ascending=False
).to_csv(
    os.path.join(OUTPUT_DIR, f'{OUTPUT_BASENAME}_Feature_Importance.csv'),
    index=False, encoding='utf-8-sig'
)
group_shap_df.rename('SHAP_percent').reset_index().rename(columns={'index': 'Group'}).to_csv(
    os.path.join(OUTPUT_DIR, f'{OUTPUT_BASENAME}_Group_Contribution.csv'),
    index=False, encoding='utf-8-sig'
)

comparison_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        f'{OUTPUT_BASENAME}_Model_Comparison.csv'
    ),
    index=False,
    encoding='utf-8-sig',
    float_format='%.6f'
)

best_parameter_rows = []
for model_name, result in model_results.items():
    for parameter, value in result[
        'best_params'
    ].items():
        best_parameter_rows.append({
            'Model': model_name,
            'Parameter': parameter,
            'Value': value
        })

pd.DataFrame(
    best_parameter_rows
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        f'{OUTPUT_BASENAME}_Best_Parameters.csv'
    ),
    index=False,
    encoding='utf-8-sig'
)

pd.concat(
    search_history_frames,
    ignore_index=True
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        f'{OUTPUT_BASENAME}_Bayesian_Search_History.csv'
    ),
    index=False,
    encoding='utf-8-sig'
)

log_rows = [
    ('Original_Sample_Size', initial_len),
    ('Cleaned_Sample_Size', len(X)),
    ('Selected_Model', selected_model_name),
    (
        'Model_Selection_Criterion',
        'Minimum five-fold CV RMSE'
    ),
    ('SHAP_Data', 'All cleaned samples'),
    ('Train_R2', f'{r2_train:.4f}'),
    ('Train_MAE', f'{mae_train:.4f}'),
    ('Train_RMSE', f'{rmse_train:.4f}'),
    ('Test_R2', f'{r2_test:.4f}'),
    ('Test_MAE', f'{mae_test:.4f}'),
    ('Test_RMSE', f'{rmse_test:.4f}')
]

for key, value in best_params.items():
    log_rows.append(
        (
            f'Selected_Best_{key}',
            value
        )
    )

pd.DataFrame(
    log_rows,
    columns=[
        'Metric_or_Parameter',
        'Value'
    ]
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        f'{OUTPUT_BASENAME}_Log.csv'
    ),
    index=False,
    encoding='utf-8-sig'
)

print(f'任务完成。最终模型：{selected_model_name}；SHAP解释全部数据。')

save_plot_data(plot_stem, selected_model_name, y_train, y_test, y_train_pred, y_test_pred,
               X_shap, shap_values)
