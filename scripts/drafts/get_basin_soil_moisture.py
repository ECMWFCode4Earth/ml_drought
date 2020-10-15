import sys
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Optional, Union, List

sys.path.append("..")

from scripts.preprocess import process_era5_land
from scripts.export import export_era5_land
from scripts.utils import get_data_path


def extract_time_series_of_soil_moisture() -> xr.Dataset:
    data_dir = get_data_path()
    reference_nc_filepath = data_dir / "interim/"
    # load in shapefile
    shp_path = Path(
        "/soge-home/projects/crop_yield/CAMELS/CAMELS_GB_DATASET"
        "/Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"
    )
    #  convert shapefile to xarray
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


if __name__ == "__main__":
    subset_str = "great_britain"

    # Download ERA5-Land
    export_era5_land(region_str=subset_str)
    #  Preprocess ERA5-Land (?)
    process_era5_land(
        subset_str=subset_str, monmean=False, resample_time=None,
    )

    # Extract
    extract_time_series_of_soil_moisture()
