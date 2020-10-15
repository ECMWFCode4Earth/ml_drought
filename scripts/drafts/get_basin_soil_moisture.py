import sys
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Optional, Union, List
from itertools import product

sys.path.append("..")

from scripts.preprocess import process_era5_land
from scripts.export import export_era5_land
from scripts.utils import get_data_path


def load_reference_nc(reference_nc_filepath: Path) -> xr.DataArray:
    target_ds = xr.ones_like(xr.open_dataset(reference_nc_filepath))
    data_var = [d for d in target_ds.data_vars][0]
    da = target_ds[data_var]

    return da


def extract_time_series_of_soil_moisture() -> xr.Dataset:
    data_dir = get_data_path()
    # load in shapefile
    # convert shapefile to xarray
    shp_filepath = Path(
        "/soge-home/projects/crop_yield/CAMELS/CAMELS_GB_DATASET"
        "/Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"
    )
    var_name = "swvl1"
    reference_nc_filepath = data_dir / "interim/reanalysis-era5-land_interim/.nc"
    # MUST have a target dataset to create the same shape
    target_ds = xr.ones_like(xr.open_dataset(reference_nc_filepath))
    data_var = [d for d in target_ds.data_vars][0]
    da = target_ds[data_var]

    # turn the shapefile into a categorical variable (like landcover)
    shp_to_nc = SHPtoXarray()
    ds = shp_to_nc.shapefile_to_xarray(
        da=da, shp_path=shp_filepath, var_name=var_name, lookup_colname=lookup_colname,
    )

    # ensure shapefile is same shape as era5land
    # for each variable (swvl1, swvl2, swvl3, swvl4)
    # for each basin extract timeseries
    #  save as xarray object with dims (time, basin)
    pass


def export_preprocess_one_year(
    year: int,
    variable: str,
    cleanup: bool = False
) -> None:
    # Download ERA5-Land
    export_era5_land(
        region_str=subset_str,
        year=[year],
        variables=[variable],
    )
    #  Preprocess ERA5-Land (?)
    process_era5_land(
        subset_str=subset_str,
        monmean=False,
        resample_time="D",
        years=[year],
        cleanup=False,
        with_merge=False,
    )

    assert (
        data_dir / f"raw/reanalysis-era5-land/{variable}/{str(year)}/01_12.nc"
    ).exists()

    assert (data_dir / f"interim/reanalysis-era5-land_interim/{}")
    raw_nc_file = data_dir / \
        f"raw/reanalysis-era5-land/{variable}/{str(year)}/01_12.nc"
    if cleanup:
        # unlink the raw hourly file
        raw_nc_file.unlink()

    print(f"\n-- Downloaded and preprocessed {variable} {year} --\n")


if __name__ == "__main__":
    subset_str = "great_britain"
    data_dir = get_data_path()
    years = np.arange(2004, 2016)
    variables = [
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
    ]

    # Due to memory constraints process hourly data into daily
    # after every Variable/Year combination
    for year, variable in product(years, variables):
        export_preprocess_one_year(year=year, variable=variable, cleanup=True)

    # Extract time series for each basin (defined in shapefile)
    extract_time_series_of_soil_moisture()
