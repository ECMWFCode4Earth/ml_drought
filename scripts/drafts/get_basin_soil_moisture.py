import sys
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Optional, Union, List

sys.path.append("..")

from scripts.preprocess import process_era5_land
from scripts.export import export_era5_land


def extract_time_series_of_soil_moisture() -> xr.Dataset:
    # load in shapefile
    # convert shapefile to xarray
    # ensure shapefile is same shape as era5land
    #
    pass


if __name__ == "__main__":
    subset_str = "great_britain"

    # Download ERA5-Land
    export_era5_land(region_str=subset_str)
    # Preprocess ERA5-Land (?)
    process_era5_land(
        subset_str=subset_str,
        monmean=False,
        resample_time=None,
    )

    # Extract
    extract_time_series_of_soil_moisture()