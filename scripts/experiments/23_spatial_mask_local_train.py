import pandas as pd
import xarray as xr
from typing import List, Dict, Tuple
import tqdm
import sys

sys.path.append("../..")

from src.analysis import spatial_rmse, spatial_r2
from src.models import RecurrentNetwork
from src.analysis import read_pred_data
from scripts.analysis import extract_json_results_dict
from scripts.utils import _rename_directory, get_data_path
from scripts.models import get_forecast_vars, get_ignore_static_vars


def open_the_experiment_csv_files() -> pd.DataFrame:
    data_dir = get_data_path()
    df = pd.concat(
        [
            pd.read_csv(d).drop("Unnamed: 0", axis=1)
            for d in (data_dir / "models/region_expt/").glob("*.csv")
        ]
    )
    return df


def open_region_specific_xr() -> xr.DataArray:
    region_lstms = [d.name for d in (data_dir / "models/region_expt").glob("*_rnn")]
    dataarrays = {}

    for lstm in region_lstms:
        region_name = lstm.split("_rnn")[0].split(":")[-1][7:]
        _, da = read_pred_data(data_dir=data_dir, experiment="region_expt", model=lstm)
        dataarrays[region_name] = da

    return dataarrays


def get_matching_dims(ref_da, other_da) -> Tuple[xr.DataArray]:
    lats = ref_da.lat.values
    lons = ref_da.lon.values
    times = ref_da.time.values

    other_da = other_da.sel(lat=lats, lon=lons, time=times)

    # also mask out missing values
    ref_mask = ref_da.isnull()
    other_da = other_da.where(~ref_mask)
    return ref_da, other_da


def join_region_das_into_one_xr_obj(
    dataarrays: Dict[str, xr.DataArray]
) -> xr.DataArray:
    df = pd.concat(
        [da.to_dataframe() for da in [d for d in dataarrays.values()]], axis=0
    )
    # drop duplicate rows!
    df = df.groupby(level=df.index.names).mean()
    # df[~df.index.duplicated()].to_xarray()
    da = df.to_xarray()["preds"]
    return da


def create_local_global_errors(local_da, global_da, test_da) -> Tuple[Dict[str, xr.DataArray]]:
    """Create xr objects with {RMSE, R2_score} for {local, global} tests.

    Args:
        local_da ([type]): The predictions from the 'local' experiment
        global_da ([type]): The predictions from the 'global' experiment
        test_da ([type]): The observed values to calculate errors

    Returns:
        Tuple[Dict[str, xr.DataArray]]: Two dictionary objects (RMSE, R2)
            for both experiments {}
    """
    _region_da, _test_da = get_matching_dims(local_da, test_da)
    _, _global_da = get_matching_dims(local_da, global_da)

    r2_dict = {}
    rmse_dict = {}

    for experiment, pred_da in zip(["local", "global"], [_region_da, _global_da]):
        experiment_rmse = spatial_rmse(
            _test_da.transpose("time", "lat", "lon"),
            pred_da.transpose("time", "lat", "lon"),
        )
        experiment_rmse.name = "rmse"
        rmse_dict[experiment] = experiment_rmse

        experiment_r2 = spatial_r2(
            _test_da.transpose("time", "lat", "lon"),
            pred_da.transpose("time", "lat", "lon"),
        )
        experiment_r2.name = "r2"
        r2_dict[experiment] = experiment_r2

    return r2_dict, rmse_dict


def lstm(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    ignore_vars=None,
    pretrained=False,
    static="features",
    spatial_mask=None,
):
    data_path = get_data_path()
    predictor = RecurrentNetwork(
        hidden_size=128,
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        ignore_vars=ignore_vars,
        static=static,
        spatial_mask=spatial_mask,
    )
    return predictor


### MARGINALIA
def create_region_preds_xrs(dataarrays, test_da) -> Dict[str, xr.DataArray]:
    das = open_region_specific_xr()
    # output dictionaries
    rmse_dict = {}
    r2_dict = {}
    true_dict = {}

    for (region, region_da) in tqdm.tqdm(das.items()):
        # get the matching shapes
        region_da, temp_da = get_matching_dims(region_da, test_da)

        # get the RMSE and R2 objects
        region_rmse = spatial_rmse(
            temp_da.transpose("time", "lat", "lon"),
            region_da.transpose("time", "lat", "lon"),
        )
        region_rmse.name = "rmse"
        rmse_dict[region] = region_rmse
        region_r2 = spatial_r2(
            temp_da.transpose("time", "lat", "lon"),
            region_da.transpose("time", "lat", "lon"),
        )
        region_r2.name = "r2"
        r2_dict[region] = region_r2

        true_dict[region] = temp_da
        break
    return


if __name__ == "__main__":
    # create region experiment data directory
    data_dir = get_data_path()
    expt_dir = data_dir / "models/region_expt"
    if not expt_dir.exists():
        expt_dir.mkdir(exist_ok=True, parents=True)

    # ignore vars
    forecast_vars = get_forecast_vars()
    ignore_static_vars = get_ignore_static_vars()
    ignore_vars = forecast_vars + ignore_static_vars

    # load in the region netcdf file
    region_nc = xr.open_dataset(
        data_dir / "analysis/boundaries_preprocessed/state_l1_india.nc"
    )

    keys = [int(i) for i in region_nc.attrs["keys"].split(",")]
    state_names = [r.rstrip().lstrip() for r in region_nc.attrs["values"].split(",")]

    # After Lakshadweep onwards (broke with np.nan - no data)
    state_names = state_names[-17:]

    # init list to store 'total_rmse' for each state
    total_rmse = []
    run_names = []

    # --- TRAIN AND EVALUATE THE MODEL FOR EACH STATE (48 times) --- #
    for state_name, state_key in zip(state_names, keys):
        print(f"** Starting Training on {state_name} **")
        save_name = "_".join(state_name.lower().split(" "))
        nanmask = region_nc.where(region_nc["state_l1"] == state_key)["state_l1"]
        mask = nanmask.isnull()

        # initialise model
        model = lstm(spatial_mask=mask)

        # train the model on that spatial subset
        model.train(num_epochs=50, early_stopping=5, verbose=True)
        model.evaluate(save_preds=True)
        model.save_model()

        total, df = extract_json_results_dict(
            data_dir=data_dir, model="rnn", experiment="one_month_forecast"
        )
        df["region"] = [state_name for _ in range(len(df))]

        # append values to lists
        total_rmse.append(total)
        run_names.append(state_name)

        df.to_csv(expt_dir / f"{save_name}_rnn_area_specific.csv")

        to_path = expt_dir / f"{save_name}_{model.model_dir.name}"
        _rename_directory(
            from_path=model.model_dir, to_path=to_path, with_datetime=True
        )

        print(
            f"** LSTM Trained and Evaluated for state: {state_name}. RMSE: {total_rmse} **"
        )
        print()

    total_df = pd.DataFrame(total_rmse, index=run_names).rename({0: "rmse"}, axis=1)
    total_df.to_csv(expt_dir / "region_results.csv")
