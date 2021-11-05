import xarray as xr
from pathlib import Path
import xesmf as xe


def convert_to_same_grid(reference_ds: xr.Dataset, ds: xr.Dataset, method: str="nearest_s2d") -> xr.Dataset:
    """ Use xEMSF package to regrid ds to the same grid as reference_ds """
    assert ("lat" in reference_ds.dims) & (
        "lon" in reference_ds.dims
    ), f"Need (lat,lon) in reference_ds dims Currently: {reference_ds.dims}"
    assert ("lat" in ds.dims) & (
        "lon" in ds.dims
    ), f"Need (lat,lon) in ds dims Currently: {ds.dims}"

    # create the grid you want to convert TO (from reference_ds)
    ds_out = xr.Dataset(
        {"lat": (["lat"], reference_ds.lat), "lon": (["lon"], reference_ds.lon)}
    )

    # create the regridder object
    # xe.Regridder(grid_in, grid_out, method='bilinear')
    regridder = xe.Regridder(ds, ds_out, method, reuse_weights=False)

    # IF it's a dataarray just do the original transformations
    if isinstance(ds, xr.core.dataarray.DataArray):
        ds = regridder(ds)
    # OTHERWISE loop through each of the variables, regrid the datarray then recombine into dataset
    elif isinstance(ds, xr.core.dataset.Dataset):
        vars = [i for i in ds.var().variables]
        if len(vars) == 1:
            ds = regridder(ds)
        else:
            output_dict = {}
            # LOOP over each variable and append to dict
            for var in vars:
                print(f"- regridding var {var} -")
                da = ds[var]
                da = regridder(da)
                output_dict[var] = da
            # REBUILD
            ds = xr.Dataset(output_dict)
    else:
        assert False, "This function only works with xarray dataset / dataarray objects"

    print(
        f"Regridded from {(regridder.shape_in)} to {(regridder.shape_out)}"
    )

    return ds


if __name__ == "__main__":
    # Which regridding algorithm?
    # https://xesmf.readthedocs.io/en/latest/notebooks/Compare_algorithms.html
    method: str = "bilinear"

    data_dir = Path("/cats/datastore/data")
    modis = xr.open_dataset(data_dir / "TOMMY/kenya_modis_10day_data.nc")
    era5 = xr.open_dataset(data_dir / "TOMMY/kenya_era5_land_daily.nc")

    reference_ds = era5[[v for v in era5.data_vars][0]].isel(time=0)

    modis_regrid = convert_to_same_grid(reference_ds, modis, method=method)
    modis_regrid.to_netcdf(data_dir / f"TOMMY/modis_10day_kenya_era5Grid_{method}.nc")

    ds = era5.merge(modis_regrid)
    ds = ds.where(~ds["modis_vci"].isnull(), drop=True)
    ds.to_netcdf(data_dir/ f"TOMMY/ALL_kenya_era5_grid.nc")