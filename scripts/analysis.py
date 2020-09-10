import xarray as xr
from pathlib import Path
import numpy as np
import sys
from typing import Tuple, List, Union, Dict
import geopandas as gpd
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import json

sys.path.append("..")

from src.analysis import AdministrativeRegionAnalysis
from src.analysis import LandcoverRegionAnalysis
from scripts.utils import get_data_path
from src.analysis import read_train_data, read_test_data, read_pred_data
from src.utils import get_ds_mask
from src.models import load_model
from src.models.neural_networks.base import NNBase
from src.analysis import spatial_rmse, spatial_r2, group_rmse, group_r2


def extract_json_results_dict(
    data_dir: Optional[Path] = None,
    model: str = "rnn",
    experiment: str = "one_month_forecast"
) -> Tuple[float, pd.DataFrame]:
    """Extract the results saved as a dictionary object (`.json`)

    Args:
        model (str, optional): string of the model {rnn ealstm linear_nn}. Defaults to "rnn".
        experiment (str, optional): [description]. Defaults to "one_month_forecast".

    Returns:
        Tuple[float, pd.DataFrame]: [description]
    """
    if data_dir is None:
        data_dir = get_data_path()

    try:
        results = json.load(
            open(data_dir / f"models/{experiment}/{model}/results.json", "rb")
        )
    except FileNotFoundError as e:
        assert False, (
            f"File not found. Try one of the following models for experiment {experiment}: ",
            [d.name for d in (data_dir / f"models/{experiment}").iterdir()],
        )

    df = pd.DataFrame(
        {
            f"{model}_rmse": [v for (k, v) in results.items() if k != "total"],
            "date_str": [k for k in results.keys() if k != "total"],
        },
    )
    df["date"] = pd.to_datetime(d["date_str"], format="%Y_%m") + MonthEnd()

    return results["total"], df


def create_region_lookup_dict(region_mask: xr.DataArray) -> Dict[int, str]:
    region_lookup = dict(
        zip(
            [int(k.lstrip().rstrip()) for k in region_mask.attrs["keys"].split(",")],
            [k.lstrip().rstrip() for k in region_mask.attrs["values"].split(",")],
        )
    )
    return region_lookup


def get_mean_timeseries_per_region(level: int = 1) -> pd.DataFrame:
    data_dir = get_data_path()
    assert level in [1, 2]

    # Calculate mean timeseries for the predictions found in `data_dir/models`
    #  using the boundaries found in `data/analysis/boundaries_preprocessed`
    analyzer = AdministrativeRegionAnalysis(data_dir)

    # ONLY run the analyzer for the region level you are interested in
    # e.g. Level 1 = State; Level 2 = District
    analyzer.region_data_paths = [
        p for p in analyzer.region_data_paths if f"_l{level}_" in p.name
    ]
    assert (
        analyzer.region_data_paths != []
    ), f"Has the boundaries_preprocessor been run for Level {level}?"

    #
    print(f"Starting the Analyzer for Level: {level} ...")
    analyzer.analyze()
    region_df = analyzer.df

    return region_df


def calculate_mean_predictions(
    level: int, region_gdf: gpd.GeoDataFrame, gdf_name_col: str
) -> gpd.GeoDataFrame:
    region_df = get_mean_timeseries_per_region(level)

    # join the mean dataframe to the geometry columns from the gdf object
    ts_gdf = gpd.GeoDataFrame(
        region_df.sort_values(["model", "datetime"])
        .set_index("region_name")
        .join(region_gdf[[gdf_name_col, "geometry"]].set_index(gdf_name_col))
        .reset_index()
    )

    return ts_gdf


def calculate_error_of_mean_predictions(
    region_mean_df: pd.DataFrame,
    region_gdf: gpd.GeoDataFrame,
    gdf_name_col: str,
    monthly_averages: bool = False,
) -> gpd.GeoDataFrame:
    """Calculate the errors of the MEAN PREDICTIONS for each region.

    Either collapse time completely (create a mean over all test times), or
    collapse time from n_months -> an average for each month. Controlled with
    the `monthly_averages` parameter.

    Assume the calculation of the region_mean_df from the
    `get_mean_timeseries_per_region()` function

    Args:
        region_mean_df (pd.DataFrame): The output of the `get_mean_timeseries_per_region()`
            function, for a given administrative level.
        region_gdf (gpd.GeoDataFrame): The GeoDataFrame (read from a shapefile)
            defining shapes
        gdf_name_col (str): The region_name column in the region_gdf
            (shapefile obj.)

    Returns:
        gpd.GeoDataFrame: [description]
    """
    # dropnans
    isnan = (region_mean_df["predicted_mean_value"].isnull()) | (
        region_mean_df["true_mean_value"].isnull()
    )
    region_mean_df = region_mean_df.loc[(~isnan)]

    # calculate scores per region (on the mean outputs)
    if monthly_averages:
        region_mean_df["datetime"] = pd.to_datetime(region_mean_df["datetime"])
        model_rmse = region_mean_df.groupby(
            [region_mean_df["datetime"].dt.strftime("%m"), "region_name", "model"]
        ).apply(group_rmse)
        model_r2 = region_mean_df.groupby(
            [region_mean_df["datetime"].dt.strftime("%m"), "region_name", "model"]
        ).apply(group_r2)

    else:
        model_rmse = region_mean_df.groupby(["region_name", "model"]).apply(group_rmse)
        model_r2 = region_mean_df.groupby(["region_name", "model"]).apply(group_r2)

    model_rmse.name = "rmse"
    model_r2.name = "r2_score"
    model_scores = pd.DataFrame(model_rmse).join(model_r2)
    model_scores = gpd.GeoDataFrame(
        model_scores.reset_index()
        .set_index("region_name")
        .join(region_gdf.set_index(gdf_name_col)[["geometry"]])
    )

    return model_scores


def create_pixel_errors_per_month_df() -> pd.DataFrame:

    return metric_df


def create_mean_pixel_errors_gdf(
    metric_dict: Dict[str, xr.DataArray],
    region_gdf: gpd.GeoDataFrame,
    gdf_name_col: str,
    region_mask: xr.DataArray,
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with the mean error metric inside each region.
    METHOD: caclulating the mean of the error metric for each pixel.
        - $ mean( err_{pixel}(pred_{pixel}) ) $
    Alternative: calculate the mean prediction, and get the error for that mean prediction.
        - $ err( mean(prediction_{pixel}) ) $

    Args:
        metric_dict (Dict[str, xr.DataArray]): Metric Dictionary created by `create_all_error_metrics`
        region_gdf (gpd.GeoDataFrame): The GeoDataFrame (read from a shapefile) defining shapes
        gdf_name_col (str): The region_name column in the region_gdf (shapefile obj.)
        region_lookup (Dict[int, str]): Created from the region_mask object
        region_mask (xr.DataArray): [description]

    Returns:
        [type]: [description]
    """
    region_lookup: Dict[int, str] = create_region_lookup_dict(region_mask)
    dfs = []
    for model in metric_dict.keys():
        metric_da = metric_dict[model]

        # method == "mean_of_error_in_space":
        # create the mean metric for pixels inside region bounds
        _dict = {}
        for region_key, region_name in region_lookup.items():
            region_data = metric_da.where(region_mask == region_key)
            _dict[region_name] = float(
                region_data.mean()[[v for v in region_data.data_vars][0]]
            )

        d = (
            pd.DataFrame(_dict, index=[model])
            .T.reset_index()
            .rename(dict(index="region_name"), axis=1)
        )
        dfs.append(d.set_index("region_name"))

    df = pd.concat(dfs, axis=1)
    gdf = gpd.GeoDataFrame(
        df.join(
            region_gdf[[gdf_name_col, "geometry"]].set_index(gdf_name_col)
        ).reset_index()
    )

    return gdf


def run_administrative_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    data_path = get_data_path()

    assert [
        f for f in (data_path / "features").glob("*/test/*/*.nc")
    ] != [], "There is no true data (has the pipeline been run?)"
    assert [
        f for f in (data_path / "models").glob("*/*/*.nc")
    ] != [], "There is no model data (has the pipeline been run?)"
    assert [f for f in (data_path / "analysis").glob("*/*.nc")] != [], (
        "There are no processed regions. " "Has the pipeline been run?"
    )

    analyzer = AdministrativeRegionAnalysis(data_path)
    analyzer.analyze()
    print(analyzer.regional_mean_metrics)


def run_landcover_region_analysis():
    # if the working directory is alread ml_drought don't need ../data
    data_path = get_data_path()

    assert [
        f for f in (data_path / "features").glob("*/test/*/*.nc")
    ] != [], "There is no true data (has the pipeline been run?)"
    assert [
        f for f in (data_path / "models").glob("*/*/*.nc")
    ] != [], "There is no model data (has the pipeline been run?)"
    assert [
        f.name
        for f in (
            data_path / "interim" / "static" / "esa_cci_landcover_preprocessed"
        ).glob("*.nc")
    ] != [], ("There is no landcover data. " "Has the pipeline been run?")

    analyzer = LandcoverRegionAnalysis(data_path)
    analyzer.analyze()
    print(analyzer.regional_mean_metrics)


def read_all_data(
    data_dir: Path, experiment="one_month_forecast", static: bool = False
) -> Tuple[xr.Dataset]:
    X_train, y_train = read_train_data(data_dir, experiment=experiment)
    X_test, y_test = read_test_data(data_dir, experiment=experiment)

    if static:
        static_ds = xr.open_dataset(data_dir / "features/static/data.nc")
    return (X_train, y_train, X_test, y_test)


def read_all_available_pred_data(
    data_dir: Path, experiment: str = "one_month_forecast"
) -> List[xr.Dataset]:
    model_dir = data_dir / f"models/{experiment}"
    models = np.array([d.name for d in model_dir.iterdir()])

    # drop the models that haven't written .nc files
    models_run_bool = [
        any([".nc" in file.name for file in (model_dir / model).iterdir()])
        for model in models
    ]
    models = models[models_run_bool]

    return {
        model: read_pred_data(model, data_dir, experiment=experiment)[-1]
        for model in models
    }


def load_nn(
    data_dir: Path, model_str: str, experiment: str = "one_month_forecast"
) -> NNBase:
    return load_model(data_dir / f"models/{experiment}/{model_str}/model.pt")


def create_all_error_metrics(
    predictions: Dict[str, xr.DataArray], y_test: xr.DataArray
) -> Tuple[Dict[str, xr.DataArray]]:
    """Pixel Based Error Metrics collapsing time.

    Args:
        predictions (Dict[str, xr.DataArray]): [description]
        y_test (xr.DataArray): [description]

    Returns:
        Tuple[Dict[str, xr.DataArray]]: [description]
    """
    rmse_dict = {}
    r2_dict = {}

    for model in [m for m in predictions.keys()]:
        model_preds = predictions[model]
        model_rmse = spatial_rmse(
            y_test.transpose("time", "lat", "lon"),
            model_preds.transpose("time", "lat", "lon"),
        )
        model_rmse.name = "rmse"
        rmse_dict[model] = model_rmse
        model_r2 = spatial_r2(
            y_test.transpose("time", "lat", "lon"),
            model_preds.transpose("time", "lat", "lon"),
        )
        model_r2.name = "r2"
        r2_dict[model] = model_r2

    return rmse_dict, r2_dict


def create_all_temporal_error_metrics(
    predictions: Dict[str, xr.DataArray], y_test: xr.DataArray
) -> Tuple[Dict[str, xr.DataArray]]:
    """Create the Monthly Mean Pixel-Errors.

    (i.e. we are answering the question, what is the model
    performance each month at the pixel scale ...?)

    Args:
        predictions (Dict[str, xr.DataArray]): [description]
        y_test (xr.DataArray): [description]

    Returns:
        Tuple[Dict[str, xr.DataArray]]: [description]
    """
    rmse_dict = {}
    r2_dict = {}

    for model in [m for m in predictions.keys()]:
        model_preds = predictions[model]
        assert model_preds.shape == y_test.shape

        # create month attribute
        model_preds["month"] = model_preds["time.month"]
        y_test["month"] = y_test["time.month"]

        # For EACH MONTH-Pixel create a new R2/RMSE score
        all_rmse = []
        all_r2 = []
        for mth in range(1, 13):
            month_preds = model_preds.where(model_preds["month"] == mth, drop=True)
            month_obs = y_test.where(y_test["month"] == mth, drop=True)

            # add new time dims (labelled 'time'), add dimensions
            month_rmse = (
                spatial_rmse(month_obs, month_preds)
                .expand_dims("month")
                .drop("time")
                .rename({"month": "time"})
            )
            month_r2 = (
                spatial_r2(month_obs, month_preds)
                .expand_dims("month")
                .drop("time")
                .rename({"month": "time"})
            )
            all_rmse.append(month_rmse)
            all_r2.append(month_r2)

        pixel_rmse_monthly = xr.combine_nested(all_rmse, concat_dim="time").rename(
            "rmse"
        )
        pixel_r2_monthly = xr.combine_nested(all_r2, concat_dim="time").rename("r2")

        # pixel_error_monthly = xr.merge([pixel_rmse_monthly, pixel_r2_monthly])
        # append each object to the dictionary
        rmse_dict[model] = pixel_rmse_monthly
        r2_dict[model] = pixel_r2_monthly

    return rmse_dict, r2_dict


def plot_predictions():

    return


if __name__ == "__main__":
    # run_administrative_region_analysis()
    # run_landcover_region_analysis()

    data_dir = get_data_path()
    experiment = "one_month_forecast"

    # ---------------------------------------- #
    # Read in the data #
    # ---------------------------------------- #
    # read train/test data
    X_train, y_train, X_test, y_test = read_all_data(
        data_dir, static=False, experiment=experiment
    )
    mask = get_ds_mask(X_train.VCI)
    test_da = y_test[list(y_test.data_vars)[0]]

    # read the predicted data
    predictions = read_all_available_pred_data(data_dir, experiment=experiment)
    #  check that the shapes of all predictions are the same
    assert set([predictions[m].shape for m in predictions.keys()]).__len__() == 1
    preds = predictions[list(m for m in predictions.keys())[0]]

    # get the matching shapes from predictions and test data
    test_da = test_da.sel(lat=preds.lat, lon=preds.lon, time=preds.time)

    # read in the models (pytorch models)
    models_str = [m for m in predictions.keys()]
    models = {}
    for m in [m for m in models_str if m != "previous_month"]:
        models[m] = load_nn(data_dir, m, experiment=experiment)

    # Calculate error metrics
    rmse_dict, r2_dict = create_all_error_metrics(predictions, test_da)

    # ---------------------------------------- #
    # Make the plots #
    # ---------------------------------------- #
    # 1. Spatial Plots of RMSE
    for m in models_str:
        fig, ax = plt.subplots()
        # fig.savefig()
    # 1. Spatial Plots of R2
    # 1. Scatter Plots of Observed vs. Predicted
    # 1. Confusion matrix of the Vegetation Deficit Index scores
