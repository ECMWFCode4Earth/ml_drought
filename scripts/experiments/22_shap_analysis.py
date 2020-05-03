"""
# from drought TO RUNOFF
mv interim interim_; mv features features_; mv features__ features; mv interim__ interim

# from runoff TO DROUGHT
mv features features__; mv interim interim__; mv interim_ interim ; mv features_ features

"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Union, Any, Tuple, Dict, Optional
import sys
import calendar
from pandas.tseries.offsets import MonthEnd
from collections import namedtuple

sys.path.append("../..")

from scripts.utils import get_data_path
from src.models.data import TrainData, ModelArrays, DataLoader
from src.models.neural_networks.base import NNBase
from src.models import load_model

EXPERIMENT = "one_month_forecast"
TRUE_EXPERIMENT = "one_month_forecast"
MODEL = "ealstm"


# ------------------------------------------------------------------------
# Run the SHAP Explainer
# ------------------------------------------------------------------------
def run_shap(
    model: NNBase,
    input_arrays: ModelArrays,
    var_names: List[str],
    num_pixels: int,
    start_idx: int = 0,
) -> TrainData:
    """Explain the NN model using shap.DeepExplainer """
    explanations, baseline = model.explain(
        x=input_arrays.x,
        var_names=var_names,
        save_explanations=False,
        background_size=100,
        start_idx=start_idx,
        num_inputs=num_pixels,
        method="shap",
    )

    return explanations


# ------------------------------------------------------------------------
# Convert from TrainData -> .nc files
# ------------------------------------------------------------------------
def dynamic_data_to_xarray(
    shap_data: TrainData,
    latlons: np.ndarray,
    var_names: List[str],
    times: List[pd.Timestamp],
) -> xr.Dataset:
    """convert numpy array of shape (latlons, time, n_features) to xarray
    """
    all_times = []
    for t_ix, time in enumerate(times):
        # pixel, [time], n_features
        array = shap_data.historical[:, t_ix, :]
        all_times.append(single_time_to_xarray(array, latlons, var_names, time=time))
    shap_historical_ds = xr.combine_by_coords(all_times)
    return shap_historical_ds


def single_time_to_xarray(
    array: np.ndarray, latlons: np.ndarray, var_names: List[str], time: pd.Timestamp
) -> xr.Dataset:
    """"""
    assert len(latlons) == array.shape[0]
    # create list of each variable
    all_dfs = [
        pd.DataFrame(
            data={
                var_name: array[:, var_ix],
                "lat": latlons[:, 0],
                "lon": latlons[:, 1],
                "time": [time for _ in range(len(latlons))],
            }
        ).set_index(["lat", "lon", "time"])
        for (var_ix, var_name) in enumerate(var_names)
    ]

    # join all variables together
    df = all_dfs[0]
    for d in all_dfs[1:]:
        df = df.join(d)

    return df.to_xarray()


def static_data_to_xarray(
    shap_data: TrainData, latlons: np.ndarray, var_names: List[str]
) -> xr.Dataset:
    return pixel_data_to_xarray(shap_data.static, latlons, var_names)


def ohe_month_data_to_xarray(shap_data: TrainData, latlons: np.ndarray) -> xr.Dataset:
    return pixel_data_to_xarray(
        shap_data.pred_months,
        latlons,
        var_names=[m for m in calendar.month_abbr if m != ""],
    )


def pixel_data_to_xarray(
    array: np.ndarray, latlons: np.ndarray, var_names: List[str]
) -> xr.Dataset:
    """"""
    assert len(latlons) == array.shape[0]
    # create list of each variable
    all_dfs = [
        pd.DataFrame(
            data={
                var_name: array[:, var_ix],
                "lat": latlons[:, 0],
                "lon": latlons[:, 1],
            }
        ).set_index(["lat", "lon"])
        for (var_ix, var_name) in enumerate(var_names)
    ]

    # join all variables together
    df = all_dfs[0]
    for d in all_dfs[1:]:
        df = df.join(d)

    return df.to_xarray()


# ------------------------------------------------------------------------
# Save helpers
# ------------------------------------------------------------------------
def _make_analysis_folder(model) -> Path:
    analysis_folder = model.model_dir / "analysis"
    if not analysis_folder.exists():
        analysis_folder.mkdir()
    return analysis_folder


def create_folder_date_str(ts: pd.Timestamp) -> str:
    return f"{ts.year}_{ts.month}"


def get_timestep_from_date_str(date_str) -> pd.Timestamp:
    return pd.to_datetime(date_str, format="%Y_%m") + MonthEnd()


def drop_vals(ds: xr.Dataset, ignore_vars: List[str]):
    ignore_vars = [v for v in ignore_vars if v in [var_ for var_ in ds.data_vars]]
    return ds.drop(ignore_vars)


# ------------------------------------------------------------------------
# Main functions
# ------------------------------------------------------------------------
def save_shap(
    dataloader: DataLoader,
    shap_data: TrainData,
    input_arrays: ModelArrays,
    model: NNBase,
) -> Union[Tuple[xr.Dataset, xr.Dataset, xr.Dataset], None]:
    """Convert shap data in TrainData object to xarray objects"""
    # 1. get lats, lons, time
    latlons = input_arrays.latlons
    times = np.array(input_arrays.historical_times)
    target_time = input_arrays.target_time
    date_str = create_folder_date_str(target_time)

    # 2. get static_vars, dynamic_vars
    dynamic_vars = input_arrays.x_vars
    static_vars = list(dataloader.static.data_vars)  # type: ignore

    # 3. check they haven't been calculated before
    analysis_folder = _make_analysis_folder(model)
    if (analysis_folder / date_str).exists():
        print(f"---- Skipping {date_str} ----")
        return None
    else:
        (analysis_folder / date_str).mkdir(exist_ok=True, parents=True)

    # check that the shapes are correct
    assert shap_data.historical.shape == (
        len(latlons),  # type: ignore
        len(times),
        len(dynamic_vars),
    )  # type: ignore
    assert shap_data.static.shape == (len(latlons), len(static_vars))  # type: ignore

    # 3. calculate the xarray objects
    shap_historical_ds = dynamic_data_to_xarray(shap_data, latlons, dynamic_vars, times)
    shap_static = static_data_to_xarray(shap_data, latlons, static_vars)
    shap_pred_month = ohe_month_data_to_xarray(shap_data, latlons)
    return_arrays = (shap_historical_ds, shap_static, shap_pred_month)

    # 4. save the xarray objects
    shap_historical_ds.to_netcdf(analysis_folder / date_str / "shap_historical_ds.nc")
    shap_static.to_netcdf(analysis_folder / date_str / "shap_static.nc")
    shap_pred_month.to_netcdf(analysis_folder / date_str / "shap_pred_month.nc")

    return return_arrays


def run_shap_for_folder(data_dir: Path, test_folder: Path, model: NNBase) -> None:
    """for the test_folder create .nc files
    containing the shap values for the inputs to the model
    """
    # 1. get the Test data to run shap analysis
    dataloader = model.get_dataloader(
        data_path=data_dir,
        mode="test",
        batch_file_size=1,
        to_tensor=True,
        shuffle_data=False,
    )
    dataloader.data_files = [test_folder]
    key, input_arrays = list(next(iter(dataloader)).items())[0]
    num_pixels = input_arrays.x.historical.shape[0]
    start_idx = 0
    var_names = input_arrays.x_vars

    # 2. run shap
    shap_data = run_shap(
        model=model,
        input_arrays=input_arrays,
        var_names=var_names,
        num_pixels=num_pixels,
        start_idx=start_idx,
    )

    # 3. convert to xarray and save to .nc
    _ = save_shap(
        dataloader=dataloader,
        shap_data=shap_data,
        input_arrays=input_arrays,
        model=model,
    )


def open_shap_analysis(model) -> Dict[str, namedtuple]:  # type: ignore
    """Read the data from the SHAP analysis run in the other functions"""
    ShapValues = namedtuple(
        "ShapValues", ["date_str", "target_time", "historical", "pred_month", "static"]
    )

    analysis_dir = model.model_dir / "analysis"
    dirs = [d for d in analysis_dir.iterdir() if len(list(d.glob("*.nc"))) > 0]

    out_dict = {}
    for shap_analysis_dir in dirs:
        shap = ShapValues(
            date_str=shap_analysis_dir.name,
            target_time=get_timestep_from_date_str(shap_analysis_dir.name),
            historical=xr.open_dataset(shap_analysis_dir / "shap_historical_ds.nc"),
            pred_month=xr.open_dataset(shap_analysis_dir / "shap_pred_month.nc"),
            static=xr.open_dataset(shap_analysis_dir / "shap_static.nc"),
        )
        out_dict[shap_analysis_dir.name] = shap

    return out_dict


def main() -> None:
    print(f"Running DeepLIFT for {EXPERIMENT}")
    data_dir = get_data_path()

    # 1. open the model
    model = load_model(data_dir / "models" / EXPERIMENT / MODEL / "model.pt")
    model.models_dir = data_dir / "models" / EXPERIMENT
    model.experiment = TRUE_EXPERIMENT

    # 2. get all the TEST timesteps in the test directory
    test_folders = [d for d in (data_dir / f"features/{EXPERIMENT}/test").iterdir()]
    #  TODO: remove this test
    # test_folders = test_folders[:2]

    #  3. run the shap analysis for each test timestep
    for test_folder in test_folders:
        print(f"\n\n** Working on {test_folder.name} **\n\n")
        run_shap_for_folder(data_dir, test_folder, model)  # type: ignore


if __name__ == "__main__":
    main()
