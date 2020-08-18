"""
Some handy analysis functions
"""
import xarray as xr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, List, Optional, Union, Tuple
from src.utils import get_ds_mask, _sort_lat_lons


def spatial_rmse(true_da: xr.DataArray, pred_da: xr.DataArray) -> xr.DataArray:
    """Calculate the RMSE collapsing the time dimension returning
    a DataArray of the rmse values (spatially)
    """
    true_da_shape = (true_da.lat.shape[0], true_da.lon.shape[0])
    pred_da_shape = (pred_da.lat.shape[0], pred_da.lon.shape[0])
    assert true_da_shape == pred_da_shape
    assert tuple(true_da.dims) == tuple(pred_da.dims), (
        f"Expect"
        "the dimensions to be the same. Currently: "
        f"True: {tuple(true_da.dims)} Preds: {tuple(pred_da.dims)}. "
        'Have you tried da.transpose("time", "lat", "lon")'
    )

    # sort the lat/lons correctly just to be sure
    pred_da = _sort_lat_lons(pred_da)
    true_da = _sort_lat_lons(true_da)

    vals = np.sqrt(
        np.nansum((true_da.values - pred_da.values) ** 2, axis=0) / pred_da.shape[0]
    )

    da = xr.ones_like(pred_da).isel(time=0)
    da.values = vals

    # reapply the mask
    da = da.where(~get_ds_mask(pred_da))
    return da


def spatial_r2(true_da: xr.DataArray, pred_da: xr.DataArray) -> xr.DataArray:
    true_da_shape = (true_da.lat.shape[0], true_da.lon.shape[0])
    pred_da_shape = (pred_da.lat.shape[0], pred_da.lon.shape[0])
    assert true_da_shape == pred_da_shape

    # sort the dimensions so that no inversions (e.g. lat)
    true_da = true_da.sortby(["time", "lat", "lon"])
    pred_da = pred_da.sortby(["time", "lat", "lon"])

    r2_vals = 1 - (np.nansum((true_da.values - pred_da.values) ** 2, axis=0)) / (
        np.nansum((true_da.values - np.nanmean(pred_da.values)) ** 2, axis=0)
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


def annual_scores(
    models: List[str],
    metrics: Optional[List[str]] = None,
    experiment="one_month_forecast",
    true_data_experiment: Optional[str] = None,
    data_path: Path = Path("data"),
    pred_years: List[int] = [2018],
    target_var: str = "VCI",
    verbose: bool = True,
    to_dataframe: bool = False,
) -> Union[Dict[str, Dict[str, List[float]]], pd.DataFrame]:
    """
    Aggregates monthly R2 scores over a `pred_year` of data
    """
    if metrics is None:
        # if None, use all
        metrics = ["rmse", "r2"]
    monthly_scores: Dict[str, Dict[str, List[float]]] = {}
    for metric in metrics:
        monthly_scores[metric] = {"month": [], "year": []}
        for model in models:
            monthly_scores[metric][model] = []

    out_dict = dict()
    for pred_year in pred_years:
        for month in range(1, 13):
            scores = monthly_score(
                month=month,
                metrics=metrics,
                models=models,
                data_path=data_path,
                pred_year=pred_year,
                experiment=experiment,
                true_data_experiment=true_data_experiment,
                target_var=target_var,
                verbose=verbose,
            )

            for model, metric_scores in scores.items():
                for metric, score in metric_scores.items():
                    monthly_scores[metric][model].append(score)
            for metric in metrics:
                monthly_scores[metric]["month"].append(month)
                monthly_scores[metric]["year"].append(pred_year)

        if to_dataframe:
            out_dict[pred_year] = annual_scores_to_dataframe(monthly_scores)
        else:
            out_dict[pred_year] = monthly_scores

    if to_dataframe:
        out_df = pd.DataFrame()
        for k in out_dict.keys():
            out_df = pd.concat([out_df, out_dict[k]])

        out_df["time"] = out_df.apply(
            lambda row: pd.to_datetime(f"{int(row.month)}-{int(row.year)}"), axis=1
        )

        return out_df
    else:
        return out_dict


def annual_scores_to_dataframe(monthly_scores: Dict) -> pd.DataFrame:
    """Convert the dictionary from annual_scores to a pd.DataFrame
    """
    df = pd.DataFrame(monthly_scores)

    metric_dfs = []
    # rename columns by metric
    for metric in monthly_scores.keys():
        metric_df = df[metric].apply(pd.Series).T
        metric_df["metric"] = metric
        metric_dfs.append(metric_df)

    # join columns into one dataframe
    df = pd.concat(metric_dfs)

    return df


def _read_multi_data_paths(data_paths: List[Path]) -> xr.Dataset:
    try:
        train_ds = (
            xr.open_mfdataset(data_paths, combine="nested", concat_dim="time")
            .sortby("time")
            .compute()
        )
    except ValueError:
        train_ds = xr.concat([xr.open_dataset(d) for d in data_paths], dim="time")
    train_ds = train_ds.transpose("time", "lat", "lon")

    return train_ds


def read_pred_data(
    model: str, data_dir: Path = Path("data"), experiment: str = "one_month_forecast"
) -> Union[xr.Dataset, xr.DataArray]:
    model_pred_dir = data_dir / "models" / experiment / model
    # Open files
    # pred_ds = xr.open_mfdataset((model_pred_dir / "*.nc").as_posix())
    ncfiles = list(model_pred_dir.glob("*.nc"))
    ds_list = [xr.open_dataset(f) for f in ncfiles]
    pred_ds = xr.combine_by_coords(ds_list)

    # set format
    pred_ds = pred_ds.sortby("time")
    pred_da = pred_ds.preds
    pred_da = pred_da.transpose("time", "lat", "lon")

    return pred_ds, pred_da


def read_true_data(
    data_dir: Path = Path("data"), variable: str = "VCI"
) -> Union[xr.Dataset, xr.DataArray]:
    """Read the true test data from the data directory and
    return the joined DataArray.

    (Joined on the `time` dimension).
    """
    true_paths = [
        f
        for f in (data_dir / "features" / "one_month_forecast" / "test").glob("*/y.nc")
    ]
    true_ds = _read_multi_data_paths(true_paths)
    true_da = true_ds[variable]
    return true_da


def monthly_score(
    month: int,
    models: List[str],
    metrics: List[str],
    experiment="one_month_forecast",
    true_data_experiment: Optional[str] = None,
    data_path: Path = Path("data"),
    pred_year: int = 2018,
    target_var: str = "VCI",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate the monthly R^2 (or RMSE) score of the model. R^2 is the same metric used by the
    [Kenya's operational drought monitoring](https://www.mdpi.com/2072-4292/11/9/1099)

    This function assumes prediction has been saved into the data directory by the model

    Arguments
    ----------
    month: the month of data being evaluated
    models: A list of models to evaluate. These names must match the model.name attributes
    experiment: The experiment being run, usually one of {'one_month_forecast', 'nowcast'}.
    true_data_experiment: the name of the experiment (for one run of the Engineer),
    one of {'one_month_forecast', 'nowcast'}. Defaults to the same as the `experiment` arg
    metrics: A list of metrics to calculate. If None, all (rmse, r2) are calculated.
    data_path: The location of the data directory
    pred_year: The year being predicted
    target_var: a str name of the target variable. Default: 'VCI'
    verbose: bool, if True prints out scores as they are calculated

    Returns:
    ----------
    output_score: A dict {model_name: {metric: score}} for that month's data
    """
    metric2function = {"r2": r2_score, "rmse": rmse}

    model_files: Dict[str, xr.Dataset] = {}
    for model in models:
        pred_path = (
            data_path / f"models/{experiment}/{model}/preds_{pred_year}_{month}.nc"
        )
        model_files[model] = xr.open_dataset(pred_path).isel(time=0)

    if true_data_experiment is None:
        true_data_path = (
            data_path / f"features/{experiment}/test" f"/{pred_year}_{month}/y.nc"
        )
    else:
        true_data_path = (
            data_path / f"features/{true_data_experiment}/test"
            f"/{pred_year}_{month}/y.nc"
        )
    true_data = xr.open_dataset(true_data_path).isel(time=0)

    output_score: Dict[str, Dict[str, float]] = {}

    for model, preds in model_files.items():
        diff = true_data[target_var] - preds.preds
        notnan = ~np.isnan(diff.values)
        joined = true_data.merge(preds, join="inner")
        true_np = joined[target_var].values[notnan].flatten()
        preds_np = joined.preds.values[notnan].flatten()

        for metric in metrics:
            score = metric2function[metric](true_np, preds_np)

            if model not in output_score:
                output_score[model] = {}

            output_score[model][metric] = score

            if verbose:
                print(f"For month {month}, model {model} has {metric} score {score}")
    return output_score


def plot_predictions(
    pred_month: int,
    model: str,
    target_var: str = "VCI",
    pred_year: int = 2018,
    data_path: Path = Path("data"),
    experiment: str = "one_month_forecast",
    **spatial_plot_kwargs,
):

    true = (
        xr.open_dataset(
            data_path / f"features/{experiment}/test" f"/{pred_year}_{pred_month}/y.nc"
        )
        .rename({target_var: "preds"})
        .isel(time=0)
    )

    model_ds = xr.open_dataset(
        data_path / f"models/{experiment}/{model}/preds" f"_{pred_year}_{pred_month}.nc"
    )

    model_err = (model_ds - true).preds.values
    model_err = model_err[~np.isnan(model_err)]
    model_err = np.sqrt(model_err ** 2).mean()

    print(f"For month {pred_month}, {model} error: {model_err}")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    true.preds.plot.hist(
        ax=ax, label="true", histtype="stepfilled", color="r", alpha=0.3
    )
    model_ds.preds.plot.hist(ax=ax, label=model, histtype="step", color="black")
    fig.legend()
    plt.show()

    plt.clf()

    if "vmin" not in spatial_plot_kwargs:
        print("You have not provided a **kwargs dict with vmin / vmax")
        print("Are you sure?")
    fig, ax = plt.subplots(1, 2, figsize=(7, 3))
    true.preds.plot(ax=ax[0], add_colorbar=False, **spatial_plot_kwargs)
    ax[0].set_title("True")
    ax[0].set_axis_off()
    model_ds.preds.plot(ax=ax[1], add_colorbar=False, **spatial_plot_kwargs)
    ax[1].set_title(model)
    ax[1].set_axis_off()
    plt.show()


def _read_data(
    data_dir: Path = Path("data"),
    train_or_test: str = "test",
    remove_duplicates: bool = True,
    experiment: str = "one_month_forecast",
) -> Tuple[xr.Dataset, xr.Dataset]:
    # LOAD the y files
    y_data_paths = [
        f for f in (data_dir / "features" / experiment / train_or_test).glob("*/y.nc")
    ]
    y_ds = _read_multi_data_paths(y_data_paths)

    # LOAD the X files
    X_data_paths = [
        f for f in (data_dir / "features" / experiment / train_or_test).glob("*/x.nc")
    ]
    X_ds = _read_multi_data_paths(X_data_paths)

    if remove_duplicates:
        # remove duplicate times from the X ds
        # https://stackoverflow.com/a/51077784/9940782
        _, index = np.unique(X_ds["time"], return_index=True)
        X_ds = X_ds.isel(time=index)

    return X_ds, y_ds


def read_train_data(
    data_dir: Path = Path("data"),
    remove_duplicates: bool = True,
    experiment: str = "one_month_forecast",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Read the training data from the data directory and return the joined DataArray.

    (Joined on the `time` dimension).

    Return:
    ------
    X_train: xr.Dataset
    y_train: xr.Dataset
    """
    train_X_ds, train_y_ds = _read_data(
        data_dir,
        train_or_test="train",
        remove_duplicates=remove_duplicates,
        experiment=experiment,
    )
    return train_X_ds, train_y_ds


def read_test_data(
    data_dir: Path = Path("data"),
    remove_duplicates: bool = True,
    experiment: str = "one_month_forecast",
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Read the test data from the data directory and return the joined DataArray.

    (Joined on the `time` dimension).

    Return:
    ------
    X_test: xr.Dataset
    y_test: xr.Dataset
    """
    test_X_ds, test_y_ds = _read_data(
        data_dir,
        train_or_test="test",
        remove_duplicates=remove_duplicates,
        experiment=experiment,
    )

    return test_X_ds, test_y_ds
