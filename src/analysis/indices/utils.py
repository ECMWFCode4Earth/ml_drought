import xarray as xr
from typing import Any
import numpy as np
from pathlib import Path


def rolling_cumsum(ds: xr.Dataset, rolling_window: int = 3) -> xr.Dataset:

    ds_window = (
        ds.rolling(time=rolling_window, center=False)
        .sum()
        .dropna(dim="time", how="all")
    )

    return ds_window


def rolling_mean(ds: xr.Dataset, rolling_window: int = 3) -> xr.Dataset:
    ds_window = (
        ds.rolling(time=rolling_window, center=False)
        .mean()
        .dropna(dim="time", how="all")
    )

    return ds_window.sortby("lat")


def apply_over_period(
    da: xr.Dataset,
    func,
    in_variable: str,
    out_variable: str,
    time_str: str = "month",
    **kwargs: Any,
) -> xr.Dataset:
    kwargs["dim"] = "time"
    return (
        da.groupby(f"time.{time_str}")
        .apply(func, args=(), **kwargs)
        .rename({in_variable: out_variable})
    )


def create_shape_aligned_climatology(
    ds: xr.Dataset, clim: xr.Dataset, variable: str, time_period: str = "month"
) -> xr.Dataset:
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
    ds[time_period] = ds[f"time.{time_period}"]
    values = clim[variable].values
    keys = clim[time_period].values
    # map the `time_period` to the `values` of the climatology (threshold or mean)
    lookup_dict = dict(zip(keys, values))
    # extract an array of the `time_period` values from the `ds`
    timevals = ds[time_period].values
    # use the lat lon arrays (climatology values) in `lookup_dict` corresponding
    #  to time_period values defined in `timevals` and stack into new array
    new_clim_vals = np.stack([lookup_dict[timevals[i]] for i in range(len(timevals))])
    assert (
        new_clim_vals.shape == ds[variable].shape
    ), f"\
        Shapes for new_clim_vals and ds must match! \
         new_clim_vals.shape: {new_clim_vals.shape} \
         ds.shape: {ds[variable].shape}"
    # copy that forward in time
    new_clim = xr.Dataset(
        {variable: (["time", "lat", "lon"], new_clim_vals)},
        coords={"lat": clim.lat, "lon": clim.lon, "time": ds.time},
    )
    return new_clim


def fit_all_indices(data_path: Path, variable: str = "precip") -> xr.Dataset:
    """ fit all indices and return one `xr.Dataset`

    Arguments:
    ---------
    data_path: Path
        path to netcdf folder to fit indices to

    variable: str = 'precip'
        the name of the variable in `data_path`

    Note:
    - This is a utility method that fits all the indices
        using default parameters.
    """
    from src.analysis.indices import (
        ZScoreIndex,
        PercentNormalIndex,
        DroughtSeverityIndex,
        ChinaZIndex,
        DecileIndex,
        AnomalyIndex,
        SPI,
    )

    indices = (
        ZScoreIndex,
        PercentNormalIndex,
        DroughtSeverityIndex,
        ChinaZIndex,
        DecileIndex,
        AnomalyIndex,
        SPI,
    )

    # fit each index
    out = {}
    for index in indices:
        i = index(data_path)
        if i.name == "china_z_index":
            i.fit(variable=variable)
            out[index.name] = i  # type: ignore
            # fit modifiedCZI
            i = index(data_path)
            i.fit(variable=variable, modified=True)
            out[index.name + "_modified"] = i  # type: ignore
        else:
            i.fit(variable=variable)
            out[index.name] = i  # type: ignore
    print([k for k in out.keys()])

    # join all indices -> 1 dataset
    print("Joining all variables into one `xr.dataset`")
    ds_objs = [index.index for index in out.values()]
    ds = xr.merge(ds_objs)
    ds = ds.drop(["month", "precip_cumsum"]).isel(time=slice(2, -1))

    return ds
