import xarray as xr

from .base import Region


def select_bounding_box(ds: xr.Dataset, region: Region) -> xr.Dataset:
    """ using the Region namedtuple defined in engineering.regions.py select
    the subset of the dataset that you have defined that region for.

    Arguments:
    ---------
    : ds (xr.Dataset)
        the data (usually from netcdf file) that you want to subset a bounding
         box from
    : region (Region)
        namedtuple object defined in engineering/regions.py

    Returns:
    -------
    : ds (xr.DataSet)
        Dataset with a subset of the whol region defined by the Region object
    """
    print(f"selecting region: {region.name} from ds")
    assert isinstance(ds, xr.Dataset) or isinstance(ds, xr.DataArray), f"ds \
        Must be an xarray object! currently: {type(ds)}"
    lonmin = region.lonmin
    lonmax = region.lonmax
    latmin = region.latmin
    latmax = region.latmax

    dims = [dim for dim in ds.dims.keys()]
    variables = [var for var in ds.variables if var not in dims]

    if 'latitude' in dims and 'longitude' in dims:
        ds_slice = ds.sel(latitude=slice(latmin, latmax), longitude=slice(lonmin, lonmax))
    elif 'lat' in dims and 'lon' in dims:
        ds_slice = ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
    else:
        raise ValueError(f'Your `xr.ds` does not have lon / longitude in the \
            dimensions. Currently: {[dim for dim in ds.dims.keys()]}')

    assert ds_slice[variables[0]].values.size != 0, f"Your slice has \
        returned NO values. Sometimes this means that the latmin, \
        latmax are the wrong way around. Try switching the order of \
        latmin, latmax"
    return ds_slice
