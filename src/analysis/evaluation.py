"""
Some handy analysis functions
"""
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from typing import Dict, List


def annual_r2_scores(models: List[str],
                     experiment='one_month_forecast',
                     data_path: Path = Path('data'),
                     pred_year: int = 2018) -> Dict[str, List[float]]:
    """
    Aggregates monthly R2 scores over a `pred_year` of data
    """
    monthly_scores: Dict[str, List[float]] = {'month': []}
    for model in models:
        monthly_scores[model] = []

    for month in range(1, 13):
        scores = monthly_r2_score(month=month, models=models, data_path=data_path,
                                  pred_year=pred_year, experiment=experiment)

        for model, score in scores.items():
            monthly_scores[model].append(score)
        monthly_scores['month'].append(month)

    return monthly_scores


def monthly_r2_score(month: int,
                     models: List[str],
                     experiment='one_month_forecast',
                     data_path: Path = Path('data'),
                     pred_year: int = 2018) -> Dict[str, float]:
    """
    Calculate the monthly R^2 score of the model. This is the same metric used by the
    [Kenya's operational drought monitoring](https://www.mdpi.com/2072-4292/11/9/1099)

    This function assumes prediction has been saved into the data directory by the model

    Arguments
    ----------
    month: the month of data being evaluated
    models: A list of models to evaluate. These names must match the model.name attributes
    experiment: The experiment being run, one of {'one_month_forecast', 'nowcast'}
    data_path: The location of the data directory
    pred_year: The year being predicted

    Returns:
    ----------
    output_score: A dict {model_name: score} for that month's data
    """
    model_files: Dict[str, xr.Dataset] = {}
    for model in models:
        pred_path = data_path / f'models/{experiment}/{model}/preds_{pred_year}_{month}.nc'
        model_files[model] = xr.open_dataset(pred_path)

    true_data = xr.open_dataset(data_path / f'features/{experiment}/test'
                                f'/{pred_year}_{month}/y.nc').isel(time=0)

    output_score: Dict[str, float] = {}

    for model, preds in model_files.items():
        diff = (true_data.VHI - preds.preds)
        notnan = ~np.isnan(diff.values)
        joined = true_data.merge(preds, join='inner')
        true_np = joined.VHI.values[notnan].flatten()
        preds_np = joined.preds.values[notnan].flatten()
        score = r2_score(true_np, preds_np)
        print(f'For month {month}, model {model} has r2 score {score}')
        output_score[model] = score
    return output_score


def plot_predictions(pred_month: int, model: str,
                     target_var: str = 'VHI',
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
