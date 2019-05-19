import time
import xarray as xr
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any

from xarray import Dataset, DataArray
from pandas._libs.tslibs.timestamps import Timestamp

# ------------------------------------------------------------------------------
# 1. assign time stamp
# ------------------------------------------------------------------------------


def extract_timestamp(ds: Dataset,
                      netcdf_filepath: str,
                      use_filepath: bool = True,
                      time_begin: bool = True) -> Timestamp:
    """from the `attrs` or filename create a datetime object for acquisition time.

    NOTE: the acquisition date is SOMEWHERE in this time range (satuday-friday)

    USE THE FILENAME
    """
    year = ds.attrs['YEAR']

    if use_filepath:  # use the weeknumber in filename
        # https://stackoverflow.com/a/22789330/9940782
        YYYYWWW = netcdf_filepath.split('P')[-1].split('.')[0]
        year = YYYYWWW[:4]
        week = YYYYWWW[5:7]
        atime = time.asctime(
            time.strptime('{} {} 1'.format(year, week), '%Y %W %w')
        )

    else:
        if time_begin:
            day_num = ds.attrs['DATE_BEGIN']
        else:  # time_end
            day_num = ds.attrs['DATE_END']

        atime = time.asctime(
            time.strptime('{} {}'.format(year, day_num), '%Y %j')
        )

    date = pd.to_datetime(atime)
    return date

# ------------------------------------------------------------------------------
# 1. assign lat lon
# ------------------------------------------------------------------------------


def create_lat_lon_vectors(ds: Dataset) -> Tuple[Any, Any]:
    """ read the `ds.attrs` and create new latitude, longitude vectors """
    assert ds.WIDTH.size == 10000, f"We are hardcoding the lat/lon \
        values so we need to ensure that all dims are the same. \
        WIDTH != 10000, == {ds.WIDTH.size}"
    assert ds.HEIGHT.size == 3616, f"We are hardcoding the lat/lon \
        values so we need to ensure that all dims are the same. \
        HEIGHT != 3616, == {ds.HEIGHT.size}"

    # lonmax = ds.attrs['END_LONGITUDE_RANGE']
    # lonmin = ds.attrs['START_LONGITUDE_RANGE']
    # latmin = ds.attrs['END_LATITUDE_RANGE']
    # latmax = ds.attrs['START_LATITUDE_RANGE']
    # NOTE: hardcoded for the VHI data (some files don't have the attrs)
    lonmax = 180
    lonmin = -180.0
    latmin = -55.152
    latmax = 75.024

    # extract the size of the lat/lon coords
    lat_len = ds.HEIGHT.shape[0]
    lon_len = ds.WIDTH.shape[0]

    # create the vector
    longitudes = np.linspace(lonmin, lonmax, lon_len)
    latitudes = np.linspace(latmin, latmax, lat_len)

    return longitudes, latitudes


# ------------------------------------------------------------------------------
# 1. create new dataset with the lat/lon and time dimensions
# ------------------------------------------------------------------------------


def create_new_dataarray(ds: Dataset,
                         variable: str,
                         longitudes: np.ndarray,
                         latitudes: np.ndarray,
                         timestamp: Timestamp) -> DataArray:
    """ Create a new dataarray for the `variable` from `ds` with geocoding and timestamp """
    # Assert statements - to a test function?
    assert variable in [v for v in ds.variables.keys()], f"variable: \
        {variable} need to be a variable in the ds! \
        Currently {[v for v in ds.variables.keys()]}"
    dims = [dim for dim in ds.dims]
    assert(
        (ds[dims[0]].size == longitudes.size) or (ds[dims[1]].size == longitudes.size)
    ), f"Size of dimensions {dims} should be equal either to \
        the size of longitudes. \n Currently longitude: \
        {longitudes.size}. {ds[dims[0]]}: {ds[dims[0]].size} \
        / {ds[dims[1]]}: {ds[dims[1]].size}"
    assert (
        (ds[dims[0]].size == latitudes.size) or (ds[dims[1]].size == latitudes.size)
    ), f"Size of dimensions {dims} should be equal either to the size of \
        latitudes. \n Currently latitude: {latitudes.size}. {ds[dims[0]]}: \
        {ds[dims[0]].size} / {ds[dims[1]]}: {ds[dims[1]].size}"
    assert np.array(timestamp).size == 1, f"The function only currently \
        works with SINGLE TIMESTEPS."

    da = xr.DataArray(
        [ds[variable].values],
        dims=['time', 'latitude', 'longitude'],
        coords={'longitude': longitudes,
                'latitude': latitudes,
                'time': [timestamp]}
    )
    da.name = variable
    return da


def create_new_dataset(ds: Dataset,
                       longitudes: np.ndarray,
                       latitudes: np.ndarray,
                       timestamp: Timestamp,
                       all_vars: bool = False) -> Dataset:
    """ Create a new dataset from ALL the variables in `ds` with the dims"""
    # initialise the list
    da_list = []

    # for each variable create a new data array and append to list
    if all_vars:
        for variable in [v for v in ds.variables.keys()]:
            da_list.append(
                create_new_dataarray(
                    ds,
                    variable,
                    longitudes,
                    latitudes,
                    timestamp
                )
            )
    else:
        # only export the VHI data
        da_list.append(
            create_new_dataarray(
                ds,
                "VHI",
                longitudes,
                latitudes,
                timestamp
            )
        )

    # merge all of the variables into one dataset
    new_ds = xr.merge(da_list)
    new_ds.attrs = ds.attrs

    return new_ds


# ------------------------------------------------------------------------------
# 1. Save the output file to new folder
# ------------------------------------------------------------------------------


def create_filename(t: Timestamp,
                    netcdf_filepath: str,
                    subset: bool = False,
                    subset_name: Optional[str] = None):
    """ create a sensible output filename (HARDCODED for this problem)
    Arguments:
    ---------
    t : pandas._libs.tslibs.timestamps.Timestamp, datetime.datetime
        timestamp of this netcdf file

    Example Output:
    --------------
    STAR_VHP.G04.C07.NN.P_20110101_VH.nc
    VHP.G04.C07.NJ.P1996027.VH.nc
    """
    substr = netcdf_filepath.split('/')[-1].split('.P')[0]
    if subset:
        assert subset_name is not None, "If you have set subset=True then you \
            need to assign a subset name"
        new_filename = f"STAR_{substr}_{t.year}_{t.month}_{t.day}_{subset_name}_VH.nc"
    else:
        new_filename = f"STAR_{substr}_{t.year}_{t.month}_{t.day}_VH.nc"
    return new_filename
