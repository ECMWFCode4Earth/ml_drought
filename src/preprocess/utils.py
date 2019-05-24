import xarray as xr

from .base import Region


def select_bounding_box(ds: xr.Dataset,
                        region: Region,
                        inverse_lat: bool = False,
                        inverse_lon: bool = False) -> xr.Dataset:
    """ using the Region namedtuple defined in engineering.regions.py select
    the subset of the dataset that you have defined that region for.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the data (usually from netcdf file) that you want to subset a bounding
         box from
    : region (Region)
        namedtuple object defined in engineering/regions.py
    : inverse_lat (bool) = False
        Whether to inverse the minimum and maximum latitudes
    : inverse_lon (bool) = False
        Whether to inverse the minimum and maximum longitudes

    Returns:
    -------
    : ds (xr.DataSet)
        Dataset with a subset of the whol region defined by the Region object
    """
    print(f"selecting region: {region.name} from ds")
    assert isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray), f"ds. " \
        f"Must be an xarray object! currently: {type(ds)}"

    dims = [dim for dim in ds.dims.keys()]
    variables = [var for var in ds.variables if var not in dims]

    latmin, latmax, lonmin, lonmax = region.latmin, region.latmax, region.lonmin, region.lonmax

    if 'latitude' in dims and 'longitude' in dims:
        ds_slice = ds.sel(
            latitude=slice(latmax, latmin) if inverse_lat else slice(latmin, latmax),
            longitude=slice(lonmax, lonmin) if inverse_lon else slice(lonmin, lonmax))
    elif 'lat' in dims and 'lon' in dims:
        ds_slice = ds.sel(
            lat=slice(latmax, latmin) if inverse_lat else slice(latmin, latmax),
            lon=slice(lonmax, lonmin) if inverse_lon else slice(lonmin, lonmax))
    else:
        raise ValueError(f'Your `xr.ds` does not have lon / longitude in the '
                         f'dimensions. Currently: {[dim for dim in ds.dims.keys()]}')

    for variable in variables:
        assert ds_slice[variable].values.size != 0, f"Your slice has returned NO values. " \
            f"Sometimes this means that the latmin, latmax are the wrong way around. " \
            f"Try switching the order of latmin, latmax"
    return ds_slice
