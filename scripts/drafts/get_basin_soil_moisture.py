import sys
from pathlib import Path
import numpy as np
import xarray as xr
from typing import Optional, Union, List
from itertools import product

sys.path.append("../..")

from scripts.preprocess import process_era5_land
from scripts.export import export_era5_land
from scripts.utils import get_data_path, _rename_directory
from src.preprocess import ERA5LandPreprocessor


def load_reference_nc(reference_nc_filepath: Path) -> xr.DataArray:
    target_ds = xr.ones_like(xr.open_dataset(reference_nc_filepath))
    data_var = [d for d in target_ds.data_vars][0]
    da = target_ds[data_var]

    return da


def extract_time_series_of_soil_moisture() -> xr.Dataset:
    data_dir = get_data_path()
    # load in shapefile
    #  convert shapefile to xarray
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
    cleanup: bool = False,
    subset_str: str = "great_britain",
) -> None:
    # Download ERA5-Land
    export_era5_land(
        region_str=subset_str,
        years=[year],
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
        resample_before_merge=True,
    )

    # -- Check that files correctly exported/processed -- #
    data_dir = get_data_path()
    # has the raw file been downloaded?
    assert (
        data_dir / f"raw/reanalysis-era5-land/{variable}/{str(year)}/01_12.nc"
    ).exists()
    # has the preprocessed file been created?
    fname = f"{year}_01_12_{variable}_great_britain.nc"
    assert (
        data_dir / f"interim/reanalysis-era5-land_interim/{variable}/{fname}"
    ).exists()

    # -- Remove the raw file -- #
    raw_nc_file = data_dir / f"raw/reanalysis-era5-land/{variable}/{str(year)}/01_12.nc"
    if cleanup:
        # delete the raw hourly file
        raw_nc_file.unlink()
        print(f"Removed File: {raw_nc_file}")

    print(f"\n-- Downloaded and preprocessed {variable} {year} --\n")


def merge_files(variable: str, subset_str: str = "great_britain") -> None:
    data_dir = get_data_path()
    processor = ERA5LandPreprocessor(data_dir)
    filename = (
        f'{variable}_data{"_" + subset_str if subset_str is not None else ""}.nc'
    )

    processor.merge_files(
        subset_str=subset_str,
        resample_time="D",
        upsampling=False,
        filename=filename,
    )

    # move all of the interim files
    from_paths = [f for f in (data_dir / "interim/reanalysis-era5-land_interim").glob("*.nc")]
    to_paths = [data_dir / f"interim/reanalysis-era5-land_OLD/{path.name}" for path in from_paths]
    to_paths[0].parents[0].mkdir(exist_ok=True, parents=True)

    for fp, tp in zip(from_paths, to_paths):
        _rename_directory(
            from_path=fp,
            to_path=tp,
        )


if __name__ == "__main__":
    subset_str = "great_britain"
    variables = [
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
    ]
    years = np.arange(2004, 2016)

    # Due to memory constraints process hourly data into daily
    # after every Variable/Year then merge all of the variable files
    for variable in variables:
        for year in years:
            export_preprocess_one_year(year=year, variable=variable, cleanup=True)

        # merge all of these daily files into one NETCDF file
        merge_files(variable, subset_str=subset_str)

        # Do we need to unlink the interim files ???

    # Extract time series for each basin (defined in shapefile)
    # TODO: need to get this working
    # extract_time_series_of_soil_moisture()
