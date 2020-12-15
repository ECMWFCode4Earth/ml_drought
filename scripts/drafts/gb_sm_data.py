import xarray as xr
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np

import sys

sys.path.insert(2, "/home/tommy/ml_drought")
from src.utils import Region, get_gb
from scripts.utils import get_data_path
from scripts.drafts.all_stations_id_list import ALL_STATIONS_ID_LIST


def read_raw_gb_sm_data(data_dir: Path) -> xr.Dataset:
    return xr.open_dataset(data_dir / "RUNOFF/gb_soil_moisture_1993_2020.nc")


def _melt_data_to_long_format(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = (
        df.melt(id_vars="time")
        .astype({"value": "float64", "variable": "int64"})
        .rename(
            {
                "value": f'swvl{path.name.split("Level_")[-1].replace(".csv", "")}',
                "variable": "station_id",
            },
            axis=1,
        )
        .set_index(["time", "station_id"])
    )
    return df


def _read_one_csv_file(path: Path) -> pd.DataFrame:
    # read in data
    df = (
        pd.read_table(path, sep=";", decimal=",")
        .drop("Unnamed: 0", axis=1)
        .rename({"Date": "time"}, axis=1)
        .astype({"time": "datetime64[ns]"})
    )

    #  TODO: REMOVE / CHECK THIS HACK ...
    #  why are the columns integers and not the station ids. Are we mapping correctly?
    if "671" in [str(c) for c in df.columns]:
        #  REMAP integers to station_ids
        assert len(df.columns) == len(ALL_STATIONS_ID_LIST) + 1
        df.columns = ["time"] + [str(idx) for idx in ALL_STATIONS_ID_LIST]

    # create index from station, time, rename column to soil level volume
    df = _melt_data_to_long_format(df, path)
    return df


def _read_csv_to_xr(sm_data_dir: Path = Path("/cats/datastore/data/RUNOFF/sm_data")):
    all_sm_ds = []
    for ix, path in enumerate(
        tqdm(list(sm_data_dir.glob("*Level*.csv")), desc="Reading SM Level",)
    ):
        df = _read_one_csv_file(path)
        # convert to xarray
        ds = df.to_xarray()
        all_sm_ds.append(ds)

    ds = xr.combine_by_coords(all_sm_ds)
    return ds


def read_gb_sm_data(
    data_dir: Path,
    reload_nc: bool = True,
    sm_data_folder: str = "sm_data",
    save_output: bool = True,
) -> xr.Dataset:
    out_data_path = data_dir / f"RUNOFF/{sm_data_folder}/gb_sm_catchments_1993_2020.nc"

    #  if file exists AND we want to reload it
    if reload_nc and out_data_path.exists():
        ds = xr.open_dataset(out_data_path)

    else:
        all_sm_ds = []
        sm_data_dir = data_dir / "RUNOFF" / sm_data_folder
        assert sm_data_dir.exists()
        ds = _read_csv_to_xr(sm_data_dir)
        if save_output:
            try:
                ds.to_netcdf(out_data_path)
            except PermissionError:
                print("Run out of memory OR open by another process")
                pass

    return ds


def old_read_obs_sm() -> xr.Dataset:
    # read in data
    obs_sm = pd.read_table(
        data_dir / "RUNOFF/Soil_Moisture_Catchments.csv", sep=";", decimal=","
    ).drop("Unnamed: 0", axis=1)

    # convert to xarray
    sm_df = (
        obs_sm.set_index("Date")
        .stack()
        .reset_index()
        .rename({0: "soil_moisture", "level_1": "station_id", "Date": "time"}, axis=1)
    )
    sm_df["time"] = pd.to_datetime(sm_df["time"])
    sm_df = sm_df.sort_values(["station_id", "time"]).set_index(["time", "station_id"])
    sm = sm_df.to_xarray()

    return sm


def upsample_xarray(
    ds: xr.Dataset,
    gb_region: Region,
    _lat_buffer: float = 0.1,
    _lon_buffer: float = 0.1,
    grid_factor: float = 2,
) -> xr.Dataset:
    """[summary]

    I don't use any interpolation method here(see method="zero"), which means that the new cells
    get the exact same value as the original cell they are in.

    gb_region:
    lon_min	        lat_min       lon_max        lat_max
    ----------------------------------------------------------
    -7.57216793459, 49.959999905, 1.68153079591, 58.6350001085

    Args:
        ds (xr.Dataset): ds is the grid data set(e.g. here ECMWF ERA5) with coordinate names "latitude" and "longitude"
        gb_region (Region): extract y_max, y_min, x_max, x_min = the min/max extends in lat/lon direction, from Region
        _lat_buffer and _lon_buffer are, as the name suggests, buffer that are used to crop the basin with some buffer
            at all sides. Be careful with the signs, they might need to be adapted depending on where you are
            (northern, southern hemisphere, and west/east of Greenwhich).
        _lat_buffer (float): [description] Defaults to 0.1.
        _lon_buffer (float): [description] Defaults to 0.1.
        grid_factor is dividing factor for a single grid cell in the original data set.
            E.g. to devide a single ERA5 cell(~30km x 30km) into roughly 1km x 1km grid cells, use a grid factor of 30.
        grid_factor (float, optional): Increase in resolution factor. Defaults to 2.

    Returns:
        xr.Dataset: [description]
    """
    y_max = gb_region.latmax
    y_min = gb_region.latmin
    x_max = gb_region.lonmax
    x_min = gb_region.lonmin

    # crop corresponding part from the xarray
    xr_basin = ds.sel(
        lat=slice(y_max + 0.1, y_min - 0.1), lon=slice(x_min - 0.1, x_max + 0.1),
    )
    data_var = list(xr_basin.data_vars)[0]
    assert all(
        [s != 0 for s in xr_basin[data_var].shape]
    ), "Expect none of the dims to be equal to zero"

    # create finer grid of lat/long coordinates used for regridding
    new_lon = np.arange(
        xr_basin.lon[0],
        xr_basin.lon[-1],
        (xr_basin.lon[1] - xr_basin.lon[0]) / grid_factor,
    )
    new_lat = np.arange(
        xr_basin.lat[0],
        xr_basin.lat[-1],
        (xr_basin.lat[1] - xr_basin.lat[0]) / grid_factor,
    )

    # regrid xarray to finer grid and fill cells without interpolation
    method = "zero"  #  TODO: FAILS!
    method = "nearest"
    xr_regrid = xr_basin.interp(lat=new_lat, lon=new_lon, method=method)

    return xr_regrid


if __name__ == "__main__":
    UPSAMPLE = False
    data_dir = Path("/cats/datastore/data")
    assert data_dir.exists()

    sm_data_folder = "GB_SM_catchments"  #  "sm_data"
    sm = read_gb_sm_data(data_dir, reload_nc=False, sm_data_folder=sm_data_folder)
    print(sm)

    if UPSAMPLE:
        # read in soil moisture data as xarray Dataset
        sm = read_raw_gb_sm_data(data_dir)
        gb_region = get_gb()

        #  increase spatial resolution
        sm_hr = upsample_xarray(sm, gb_region, grid_factor=3)
        print(sm_hr)

        sm_hr.to_netcdf(data_dir / "RUNOFF/gb_soil_moisture_1993_2020_HR.nc")