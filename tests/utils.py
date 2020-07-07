import numpy as np
import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
from src.models.data import DataLoader
from argparse import Namespace

Point = None
gpd = None


def _make_dataset(
    size,
    variable_name="VHI",
    lonmin=-180.0,
    lonmax=180.0,
    latmin=-55.152,
    latmax=75.024,
    add_times=True,
    const=False,
    start_date="1999-01-01",
    end_date="2001-12-31",
):

    lat_len, lon_len = size
    # create the vector
    longitudes = np.linspace(lonmin, lonmax, lon_len)
    latitudes = np.linspace(latmin, latmax, lat_len)

    dims = ["lat", "lon"]
    coords = {"lat": latitudes, "lon": longitudes}

    if add_times:
        times = pd.date_range(start_date, end_date, name="time", freq="M")
        size = (len(times), size[0], size[1])
        dims.insert(0, "time")
        coords["time"] = times
    var = np.random.randint(100, size=size)
    if const:
        var *= 0
        var += 1

    ds = xr.Dataset({variable_name: (dims, var)}, coords=coords)

    return ds, (lonmin, lonmax), (latmin, latmax)


def _create_dummy_precip_data(tmp_path, start_date="1999-01-01", end_date="2001-12-31"):
    data_dir = tmp_path / "data" / "interim" / "chirps_preprocessed"
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    precip, _, _ = _make_dataset(
        (30, 30), variable_name="precip", start_date=start_date, end_date=end_date
    )
    precip.to_netcdf(data_dir / "data_kenya.nc")

    return data_dir


class CreateSHPFile:
    def __init__(self):
        # import Point and Geopandas
        global Point
        if Point is None:
            from shapely.geometry import Point

        global gpd
        if gpd is None:
            import geopandas as gpd

    @staticmethod
    def create_demo_shapefile(filepath):
        df = pd.DataFrame({"PROVID": [10, 20], "PROVINCE": ["NAIROBI", "KIAMBU"]})

        p1 = Point((34.27795473150634, 0.3094489371060183))
        p2 = Point((35.45785473150634, 0.0118489371060182))

        gdf = gpd.GeoDataFrame(df, geometry=[p1, p2])
        gdf["geometry"] = gdf.buffer(0.2)
        gdf.to_file(driver="ESRI Shapefile", filename=filepath)


def _create_features_dir(
    tmp_path, train=False, start_date="1999-01-01", end_date="2001-12-31"
):
    if train:
        features_dir = tmp_path / "features" / "one_month_forecast" / "train"
    else:
        features_dir = tmp_path / "features" / "one_month_forecast" / "test"

    if not features_dir.exists():
        features_dir.mkdir(parents=True, exist_ok=True)

    daterange = pd.date_range(start=start_date, end=end_date, freq="M")
    dates = [f"{d.year}-{d.month}-{d.day}" for d in daterange]
    dir_names = [f"{d.year}_{d.month}" for d in daterange]

    # TARGET variable target time
    for date, dir_name in zip(dates, dir_names):
        (features_dir / dir_name).mkdir(exist_ok=True, parents=True)
        vci, _, _ = _make_dataset(
            (30, 30), variable_name="vci", start_date=date, end_date=date
        )
        vci.to_netcdf(features_dir / dir_name / "y.nc")

    # INPUT variables (previous time)
    for date in dates:
        # get the previous time
        prev_dt = pd.to_datetime(date) - pd.DateOffset(months=1)
        prev_dt = pd.Series([0], index=[prev_dt]).resample("M").mean().index
        dir_name = f"{prev_dt.year[0]}_{prev_dt.month[0]}"
        prev_date = f"{prev_dt.year[0]}-{prev_dt.month[0]:02}-{prev_dt.day[0]:02}"

        # make the directory
        (features_dir / dir_name).mkdir(exist_ok=True, parents=True)

        # create input data (autoregressive + other)
        vci, _, _ = _make_dataset(
            (30, 30), variable_name="vci", start_date=prev_date, end_date=prev_date
        )
        precip, _, _ = _make_dataset(
            (30, 30), variable_name="precip", start_date=prev_date, end_date=prev_date
        )
        ds = xr.auto_combine([vci, precip])
        # if dir_name != '1998_12':
        # assert False

        ds.to_netcdf(features_dir / dir_name / "x.nc")


def _make_dynamic_data(tmp_path, dates, mode) -> Tuple[xr.Dataset, ...]:
    for date in dates:
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        date_str = f"{date.year}_{date.month}"
        test_features = tmp_path / f"features/one_month_forecast/{mode}/{date_str}"
        test_features.mkdir(parents=True)

        norm_dict = {"VHI": {"mean": 0, "std": 1}}
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "wb"
        ) as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

    return x, y


def make_drought_test_data(tmp_path: Path, len_dates: int = 1, test: bool = False) -> Tuple[xr.Dataset, ...]:
    if len_dates == 1:
        dates = [pd.to_datetime("1980-01-01")]
    else:
        dates = pd.date_range("1980-01-01", freq="M", periods=len_dates)

    if test:
        mode = "test"
        x_static = None
    else:
        mode = "train"

        # make static data
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / "data.nc")

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

    x, y = _make_dynamic_data(tmp_path, dates, mode)

    return x, y, x_static


def get_dataloader(
    mode: str, hparams: Namespace, shuffle_data: bool = False, **kwargs
) -> DataLoader:
    """
    Return the correct dataloader for this model
    """

    default_args: Dict[str, Any] = {
        "data_path": hparams.data_path,
        "batch_file_size": hparams.batch_size,
        "shuffle_data": shuffle_data,
        "mode": mode,
        "mask": None,
        "experiment": hparams.experiment,
        "pred_months": hparams.pred_months,
        "to_tensor": True,
        "ignore_vars": hparams.ignore_vars,
        "monthly_aggs": hparams.include_monthly_aggs,
        "surrounding_pixels": hparams.surrounding_pixels,
        "static": hparams.static,
        "device": "cpu",
        "clear_nans": True,
        "normalize": True,
        "predict_delta": hparams.predict_delta,
        "spatial_mask": hparams.spatial_mask,
        "normalize_y": hparams.normalize_y,
    }

    for key, val in kwargs.items():
        # override the default args
        default_args[key] = val

    dl = DataLoader(**default_args)

    return dl
