"""Save the numerical inputs required by SFigure19 and SFigure20."""
import ast
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_plot_data(stem, selected_model, y_train, y_test, train_pred, test_pred,
                   feature_values, shap_values):
    target = Path(str(stem) + '_Plot_Data.npz')
    with NamedTemporaryFile(dir=target.parent, suffix='.npz', delete=False) as handle:
        tmp = Path(handle.name)
        try:
            np.savez_compressed(
                handle, selected_model=np.asarray(selected_model, dtype=str),
                y_train=np.asarray(y_train), y_test=np.asarray(y_test),
                train_pred=np.asarray(train_pred), test_pred=np.asarray(test_pred),
                feature_names=np.asarray(feature_values.columns, dtype=str),
                feature_values=feature_values.to_numpy(), shap_values=np.asarray(shap_values),
            )
        except Exception:
            handle.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    print(f'Plot data saved: {target}', flush=True)


def prepare_saved_plot_data(stem, estimators, X, X_train, X_test, y_train, y_test):
    """Replay just the saved winning fit and require agreement with its saved tables."""
    comparison = pd.read_csv(str(stem) + '_Model_Comparison.csv')
    best = comparison.sort_values(['Best_CV_RMSE', 'CV_RMSE_Std']).iloc[0]
    model_name = best.Model
    parameters = pd.read_csv(str(stem) + '_Best_Parameters.csv', dtype={'Value': str})
    options = {}
    integer_parameters = {'n_estimators', 'max_depth', 'min_samples_leaf',
                          'min_samples_split', 'min_child_weight'}
    for row in parameters.loc[parameters.Model.eq(model_name)].itertuples():
        try:
            value = ast.literal_eval(row.Value)
        except (ValueError, SyntaxError):
            value = row.Value
        options[row.Parameter] = int(value) if row.Parameter in integer_parameters else value
    if not options:
        raise ValueError(f'No saved parameters for {model_name}')
    model = estimators[model_name].set_params(**options)
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    for split, actual, prediction in [('Train', y_train, train_pred), ('Test', y_test, test_pred)]:
        metrics = [r2_score(actual, prediction), mean_absolute_error(actual, prediction),
                   np.sqrt(mean_squared_error(actual, prediction))]
        saved = [best[f'{split}_{metric}'] for metric in ('R2', 'MAE', 'RMSE')]
        if not np.allclose(metrics, saved, rtol=0, atol=6e-7):
            raise ValueError(f'{stem}: replayed {split} metrics differ: {metrics} versus {saved}')
    values = shap.TreeExplainer(model).shap_values(X)
    importance = np.abs(values).mean(axis=0)
    contribution = importance / importance.sum() * 100
    saved = pd.read_csv(str(stem) + '_Feature_Importance.csv').set_index('Feature')
    saved = saved.loc[X.columns, 'SHAP_percent'].to_numpy()
    if not np.allclose(contribution, saved, rtol=0, atol=6e-7):
        raise ValueError(f'{stem}: replayed SHAP percentages differ from saved values')
    save_plot_data(stem, model_name, y_train, y_test, train_pred, test_pred, X, values)
