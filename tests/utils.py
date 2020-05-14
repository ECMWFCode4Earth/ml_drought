import numpy as np
import xarray as xr
import pickle
from collections import defaultdict
import pandas as pd
from pathlib import Path
from typing import Tuple

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


def _make_features_directory(tmp_path, train=False, expt_name="one_month_forecast") -> Path:
    if train:
        features_dir = tmp_path / "features" / expt_name / "train"
    else:
        features_dir = tmp_path / "features" / expt_name / "test"

    if not features_dir.exists():
        features_dir.mkdir(parents=True, exist_ok=True)

    return features_dir


def _create_features_dir(
    tmp_path, train=False, start_date="1999-01-01", end_date="2001-12-31"
):
    features_dir = _make_features_directory(tmp_path, train=train)

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


def _make_runoff_data(
    start_date="2000-01", end_date="2001-01"
) -> Tuple[xr.Dataset, xr.Dataset]:
    # create dims / coords
    times = pd.date_range(start_date, end_date, freq="D")
    station_ids = np.arange(0, 10)
    dims = ["station_id", "time"]
    coords = {"station_id": station_ids, "time": times}
    shape = (len(station_ids), len(times))

    # create random data
    precip = np.random.random(shape)
    discharge = np.random.random(shape)
    pet = np.random.random(shape)
    datasets = [precip, discharge, pet]
    variables = ["precip", "discharge", "pet"]

    ds = xr.Dataset(
        {variable: (dims, dataset) for variable, dataset in zip(variables, datasets)},
        coords=coords,
    )

    # create static data
    gauge_elev = np.random.random(size=shape[0])
    q_mean = np.random.random(size=shape[0])
    area = np.random.random(size=shape[0])
    datasets = [gauge_elev, q_mean, area]
    variables = ["gauge_elev", "q_mean", "area"]
    dims = ["station_id"]
    coords = {"station_id": station_ids}

    static = xr.Dataset(
        {variable: (dims, dataset) for variable, dataset in zip(variables, datasets)},
        coords=coords,
    )

    return ds, static


def _ds_to_features_dirs(
    tmp_path: Path,
    data: xr.Dataset,
    date_range: pd.DatetimeIndex,
    train: bool = False,
    x: bool = True,
    expt_name = "one_timestep_forecast",
):
    # create features directory setup
    # e.g. features/train/2000_1/x.nc
    features_dir = _make_features_directory(tmp_path, train=train, expt_name=expt_name)

    dates = [f"{d.year}-{d.month}-{d.day}" for d in date_range]
    dir_names = [f"{d.year}_{d.month}" for d in date_range]

    for date, dir_name in zip(dates, dir_names):
        (features_dir / dir_name).mkdir(exist_ok=True, parents=True)
        data.to_netcdf(features_dir / dir_name / "x.nc" if x else "y.nc")


def _calculate_normalization_dict(train_ds: xr.Dataset, static: bool):
    normalization_dict = defaultdict(dict)
    if static:
        dims = [c for c in train_ds.coords]
    else:  # dynamic
        assert (
            len([c for c in train_ds.coords if c != "time"]) == 1
        ), "Only works with one dimension"
        dimension_name = [c for c in train_ds.coords][0]
        dims = [dimension_name, "time"]

    for var in train_ds.data_vars:
        if var.endswith("one_hot"):
            mean = 0
            std = 1

        mean = float(train_ds[var].mean(dim=dims, skipna=True).values)
        std = float(train_ds[var].std(dim=dims, skipna=True).values)
        normalization_dict[var]["mean"] = mean
        normalization_dict[var]["std"] = std

    return normalization_dict


def _create_normalization_dict(tmp_path, X_data, static):
    static_normalizing_dict = _calculate_normalization_dict(static, static=True)
    normalizing_dict = _calculate_normalization_dict(X_data, static=False)
    (tmp_path / "features/one_timestep_forecast").mkdir(exist_ok=True, parents=True)
    static_savepath = (
        tmp_path / "features/static") / "normalizing_dict.pkl"
    dynamic_savepath = (
        tmp_path / "features/one_timestep_forecast") / "normalizing_dict.pkl"

    with dynamic_savepath.open("wb") as f:
        pickle.dump(normalizing_dict, f)
    with static_savepath.open("wb") as f:
        pickle.dump(static_normalizing_dict, f)


def _create_runoff_features_dir(
    tmp_path, train=False, start_date="1999-01-01", end_date="2001-12-31"
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    ds, static = _make_runoff_data(start_date=start_date, end_date=end_date)

    # TARGET variable target time
    target_time = pd.to_datetime(ds.isel(time=-1).time.values)
    y_daterange = pd.DatetimeIndex([target_time])
    y_data = ds.sel(time=target_time)[["discharge"]]

    # non-target data
    X_data = ds.isel(time=slice(0, -1))

    x_y_separate = False
    if x_y_separate:
        _ds_to_features_dirs(
            tmp_path, data=y_data, date_range=y_daterange, train=True, x=False
        )

        # non-target data
        X_data = ds.isel(time=slice(0, -1))

        X_daterange = pd.date_range(
            X_data.time.min().values, X_data.time.max().values, freq="D"
        )

        _ds_to_features_dirs(
            tmp_path, data=X_data, date_range=X_daterange, train=True, x=True
        )

    if not (tmp_path / "features/one_timestep_forecast").exists():
         (tmp_path / "features/one_timestep_forecast").mkdir(exist_ok=True, parents=True)
         ds.to_netcdf(tmp_path / "features/one_timestep_forecast/data.nc")

    # static_data
    if not (tmp_path / "features/static").exists():
        (tmp_path / "features/static").mkdir(exist_ok=True, parents=True)
    static.to_netcdf(tmp_path / "features/static/data.nc")

    # Calculate normalizing_dict
    _create_normalization_dict(tmp_path, X_data, static)

    return X_data, y_data, static
