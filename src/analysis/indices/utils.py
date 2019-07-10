import xarray as xr
from typing import Dict, Any
import numpy as np


def rolling_cumsum(ds: xr.Dataset,
                   rolling_window: int = 3) -> xr.Dataset:

    ds_window = (
        ds.rolling(time=rolling_window, center=True)
        .sum()
        .dropna(dim='time', how='all')
    )

    return ds_window


def apply_over_period(da: xr.Dataset,
                      func,
                      in_variable: str,
                      out_variable: str,
                      time_str: str = 'month',
                      **kwargs: Dict[Any, Any]) -> xr.Dataset:
    kwargs['dim'] = 'time'  # type: ignore
    return (
        da.groupby(f'time.{time_str}')
        .apply(func, args=(), **kwargs)
        .rename({in_variable: out_variable})
    )


def create_shape_aligned_climatology(ds: xr.Dataset,
                                     clim: xr.Dataset,
                                     variable: str,
                                     time_period: str = 'month') -> xr.Dataset:
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
