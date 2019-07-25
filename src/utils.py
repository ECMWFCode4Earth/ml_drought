"""
See all the bounding boxes (NOTE COMMENTS OF ERRORS) in this gist
    https://gist.github.com/graydon/11198540
"""

from collections import namedtuple
import calendar
from datetime import date
import xarray as xr
import numpy as np
from scipy import stats

from typing import Optional, Tuple


Region = namedtuple('Region', ['name', 'lonmin', 'lonmax', 'latmin', 'latmax'])


def get_kenya() -> Region:
    """This pipeline is focused on drought prediction in Kenya.
    This function allows Kenya's bounding box to be easily accessed
    by all exporters.
    """
    return Region(name='kenya', lonmin=33.501, lonmax=42.283,
                  latmin=-5.202, latmax=6.002)


def get_ethiopia() -> Region:
    return Region(name='ethiopia', lonmin=32.9975838, lonmax=47.9823797,
                  latmin=3.397448, latmax=14.8940537)


def get_east_africa() -> Region:
    return Region(name='east_africa', lonmin=21, lonmax=51.8,
                  latmin=-11, latmax=23)


def minus_months(cur_year: int, cur_month: int, diff_months: int,
                 to_endmonth_datetime: bool = True) -> Tuple[int, int, Optional[date]]:
    """Given a year-month pair (e.g. 2018, 1), and a number of months subtracted
    from that `diff_months` (e.g. 2), return the new year-month pair (e.g. 2017, 11).

    Optionally, a date object representing the end of that month can be returned too
    """

    new_month = cur_month - diff_months
    if new_month < 1:
        new_month += 12
        new_year = cur_year - 1
    else:
        new_year = cur_year

    if to_endmonth_datetime:
        newdate: Optional[date] = date(new_year, new_month,
                                       calendar.monthrange(new_year, new_month)[-1])
    else:
        newdate = None
    return new_year, new_month, newdate


def get_ds_mask(ds: xr.Dataset) -> xr.Dataset:
    """ Return a boolean Dataset which is a mask of the first timestep in `ds`
    NOTE:
        assumes that all of the null values from `ds` are valid null values (e.g.
        water bodies). Could also be invalid nulls due to poor data processing /
        lack of satellite input data for a pixel!
    """
    mask = ds.isnull().isel(time=0).drop('time')
    mask.name = 'mask'

    return mask


def create_shape_aligned_climatology(ds, clim, variable, time_period):
    """match the time dimension of `clim` to the shape of `ds` so that can
    perform simple calculations / arithmetic on the values of clim

    Arguments:
    ---------
    ds : xr.Dataset
        the dataset with the raw values that you are comparing to climatology

    clim: xr.Dataset
        the climatology values for a given `time_period`

    variable: str
        name of the variable that you are comparing to the climatology

    time_period: str
        the period string used to calculate the climatology
         time_period = {'dayofyear', 'season', 'month'}

    Notes:
        1. assumes that `lat` and `lon` are the
        coord names in ds

    """
    for coord in ['lat', 'lon']:
        assert coord in [c for c in ds.coords]

    ds[time_period] = ds[f'time.{time_period}']

    values = clim[variable].values
    keys = clim[time_period].values
    # map the `time_period` to the `values` of the climatology (threshold or mean)
    lookup_dict = dict(zip(keys, values))

    # extract an array of the `time_period` values from the `ds`
    timevals = ds[time_period].values

    # use the lat lon arrays (climatology values) in `lookup_dict` corresponding
    #  to time_period values defined in `timevals` and stack into new array
    new_clim_vals = np.stack([lookup_dict[timevals[i]] for i in range(len(timevals))])

    assert new_clim_vals.shape == ds[variable].shape, f"\
        Shapes for new_clim_vals and ds must match! \
         new_clim_vals.shape: {new_clim_vals.shape} \
         ds.shape: {ds[variable].shape}"

    # copy that forward in time
    new_clim = xr.Dataset(
        {variable: (['time', 'lat', 'lon'], new_clim_vals)},
        coords={
            'lat': clim.lat,
            'lon': clim.lon,
            'time': ds.time,
        }
    )

    return new_clim


def get_modal_value_across_time(da: xr.DataArray) -> xr.DataArray:
    """Get the modal value along the time dimension
    (produce a 2D spatial array with each pixel being the
    modal value)

    Arguments:
    ---------
    ds: xr.DataArray
        the DataArray that you want to calculate the mode for

    TODO: Why do these not work?
    # stacked = ds.stack(pixels=['lat', 'lon'])
    # # xr.apply_ufunc(stats.mode, stacked)
    # mode = stacked.reduce(, dim='time')
    # mode = mode.unstack('pixel')
    """
    print("Extracting the data to numpy array")
    data = da.values

    print("calculating the mode across the time dimension")
    # NOTE: assuming that time is dim=0
    mode = stats.mode(data, axis=0)[0]

    mode_da = xr.ones_like(da).isel(time=slice(0, 1))
    mode_da.values = mode

    return mode_da


def drop_nans_and_flatten(dataArray: xr.DataArray) -> np.ndarray:
    """flatten the array and drop nans from that array. Useful for plotting histograms.

    Arguments:
    ---------
    : dataArray (xr.DataArray)
        the DataArray of your value you want to flatten
    """
    # drop NaNs and flatten
    return dataArray.values[~np.isnan(dataArray.values)]


# dictionary lookup of regions
region_lookup = {
    "kenya": get_kenya(),
    "ethiopia": get_ethiopia(),
    "east_africa": get_east_africa(),
}
