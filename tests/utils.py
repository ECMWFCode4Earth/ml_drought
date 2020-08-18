import numpy as np
import xarray as xr
from typing import Optional
import pandas as pd

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
    random_nan: Optional[int] = None,
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

    if random_nan:
        indices = [i for i in np.ndindex(var.shape)]
        ix = np.random.choice(range(len(indices)), random_nan, replace=False)
        index = indices[int(ix)]
        # have to convert to float if have nans
        var = var.astype("float64")
        var[index] = np.nan

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
