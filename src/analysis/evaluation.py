"""
Some handy analysis functions
"""
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Optional, Union
from src.utils import get_ds_mask


def spatial_rmse(true_da: xr.DataArray,
                 pred_da: xr.DataArray) -> xr.DataArray:
    """Calculate the RMSE collapsing the time dimension returning
    a DataArray of the rmse values (spatially)
    """
    true_da_shape = (true_da.lat.shape[0], true_da.lon.shape[0])
    pred_da_shape = (pred_da.lat.shape[0], pred_da.lon.shape[0])
    assert true_da_shape == pred_da_shape

    vals = np.sqrt(
        np.nansum((true_da.values - pred_da.values)**2, axis=0) / pred_da.shape[0]
    )

    da = xr.ones_like(pred_da).isel(time=0)
    da.values = vals

    # reapply the mask
    da = da.where(~get_ds_mask(pred_da))
    return da


def spatial_r2(true_da: xr.DataArray,
               pred_da: xr.DataArray) -> xr.DataArray:
    true_da_shape = (true_da.lat.shape[0], true_da.lon.shape[0])
    pred_da_shape = (pred_da.lat.shape[0], pred_da.lon.shape[0])
    assert true_da_shape == pred_da_shape

    r2_vals = 1 - (
        np.nansum((true_da.values - pred_da.values)**2, axis=0)
    ) / (
        np.nansum((true_da.values - np.nanmean(pred_da.values))**2, axis=0)
    )

    da = xr.ones_like(pred_da).isel(time=0)
    da.values = r2_vals

    # reapply the mask
    da = da.where(~get_ds_mask(pred_da))
    return da


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    So that RMSE can be selected as an evaluation metric
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def annual_scores(models: List[str],
                  metrics: Optional[List[str]] = None,
                  experiment='one_month_forecast',
                  data_path: Path = Path('data'),
                  pred_year: int = 2018,
                  target_var: str = 'VCI',
                  verbose: bool = True,
                  to_dataframe: bool = False,
                  ) -> Union[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """
    Aggregates monthly R2 scores over a `pred_year` of data
    """
    if metrics is None:
        # if None, use all
        metrics = ['rmse', 'r2']
    monthly_scores: Dict[str, Dict[str, List[float]]] = {}
    for metric in metrics:
        monthly_scores[metric] = {'month': []}
        for model in models:
            monthly_scores[metric][model] = []

    for month in range(1, 13):
        scores = monthly_score(month=month, metrics=metrics, models=models,
                               data_path=data_path, pred_year=pred_year,
                               experiment=experiment, target_var=target_var,
                               verbose=verbose)

        for model, metric_scores in scores.items():
            for metric, score in metric_scores.items():
                monthly_scores[metric][model].append(score)
        for metric in metrics:
            monthly_scores[metric]['month'].append(month)

    if to_dataframe:
        return annual_scores_to_dataframe(monthly_scores)
    else:
        return monthly_scores


def annual_scores_to_dataframe(monthly_scores: Dict) -> pd.DataFrame:
    """Convert the dictionary from annual_scores to a pd.DataFrame
    """
    df = pd.DataFrame(monthly_scores)

    metric_dfs = []
    # rename columns by metric
    for metric in monthly_scores.keys():
        metric_df = df[metric].apply(pd.Series).T
        metric_df['metric'] = metric
        metric_dfs.append(metric_df)

    # join columns into one dataframe
    df = pd.concat(metric_dfs)

    return df


def read_pred_data(model: str,
                   data_dir: Path = Path('data'),
                   experiment: str = 'one_month_forecast') -> Union[xr.Dataset, xr.DataArray]:
    model_pred_dir = (data_dir / 'models' / experiment / model)
    pred_ds = xr.open_mfdataset((model_pred_dir / '*.nc').as_posix())
    pred_ds = pred_ds.sortby('time')
    pred_da = pred_ds.preds
    pred_da = pred_da.transpose('time', 'lat', 'lon')

    return pred_ds, pred_da


def monthly_score(month: int,
                  models: List[str],
                  metrics: List[str],
                  experiment='one_month_forecast',
                  data_path: Path = Path('data'),
                  pred_year: int = 2018,
                  target_var: str = 'VCI',
                  verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """
    Calculate the monthly R^2 (or RMSE) score of the model. R^2 is the same metric used by the
    [Kenya's operational drought monitoring](https://www.mdpi.com/2072-4292/11/9/1099)

    This function assumes prediction has been saved into the data directory by the model

    Arguments
    ----------
    month: the month of data being evaluated
    models: A list of models to evaluate. These names must match the model.name attributes
    experiment: The experiment being run, one of {'one_month_forecast', 'nowcast'}
    metrics: A list of metrics to calculate. If None, all (rmse, r2) are calculated.
    data_path: The location of the data directory
    pred_year: The year being predicted
    target_var: a str name of the target variable. Default: 'VCI'
    verbose: bool, if True prints out scores as they are calculated

    Returns:
    ----------
    output_score: A dict {model_name: {metric: score}} for that month's data
    """
    metric2function = {
        'r2': r2_score,
        'rmse': rmse
    }

    model_files: Dict[str, xr.Dataset] = {}
    for model in models:
        pred_path = data_path / f'models/{experiment}/{model}/preds_{pred_year}_{month}.nc'
        model_files[model] = xr.open_dataset(pred_path).isel(time=0)

    true_data = xr.open_dataset(data_path / f'features/{experiment}/test'
                                f'/{pred_year}_{month}/y.nc').isel(time=0)

    output_score: Dict[str, Dict[str, float]] = {}

    for model, preds in model_files.items():
        diff = (true_data[target_var] - preds.preds)
        notnan = ~np.isnan(diff.values)
        joined = true_data.merge(preds, join='inner')
        true_np = joined[target_var].values[notnan].flatten()
        preds_np = joined.preds.values[notnan].flatten()

        for metric in metrics:
            score = metric2function[metric](true_np, preds_np)

            if model not in output_score:
                output_score[model] = {}

            output_score[model][metric] = score

            if verbose:
                print(f'For month {month}, model {model} has {metric} score {score}')
    return output_score


def plot_predictions(pred_month: int, model: str,
                     target_var: str = 'VCI',
                     pred_year: int = 2018,
                     data_path: Path = Path('data'),
                     experiment: str = 'one_month_forecast'):

    true = xr.open_dataset(data_path / f'features/{experiment}/test'
                           f'/{pred_year}_{pred_month}/y.nc').\
        rename({target_var: 'preds'}).isel(time=0)

    model_ds = xr.open_dataset(data_path / f'models/{experiment}/{model}/preds'
                               f'_{pred_year}_{pred_month}.nc')

    model_err = (model_ds - true).preds.values
    model_err = model_err[~np.isnan(model_err)]
    model_err = np.sqrt(model_err ** 2).mean()

    print(f'For month {pred_month}, {model} error: {model_err}')

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    true.preds.plot.hist(ax=ax, label='true', histtype='stepfilled', color='r', alpha=0.3)
    model_ds.preds.plot.hist(ax=ax, label=model, histtype='step', color='black')
    fig.legend()
    plt.show()

    plt.clf()

    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    true.preds.plot(vmin=0, vmax=100, ax=ax[0], add_colorbar=False)
    ax[0].set_title('True')
    model_ds.preds.plot(vmin=0, vmax=100, ax=ax[1], add_colorbar=False)
    ax[1].set_title(model)
    plt.show()
