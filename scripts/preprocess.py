import sys
from pathlib import Path
import numpy as np
from typing import Optional, Union, List

sys.path.append("..")
from src.preprocess import (
    VHIPreprocessor,
    CHIRPSPreprocessor,
    PlanetOSPreprocessor,
    GLEAMPreprocessor,
    S5Preprocessor,
    ESACCIPreprocessor,
    SRTMPreprocessor,
    ERA5MonthlyMeanPreprocessor,
    ERA5HourlyPreprocessor,
    BokuNDVIPreprocessor,
    KenyaASALMask,
    ERA5LandPreprocessor,
    ERA5LandMonthlyMeansPreprocessor,
    IndiaAdminProcessor,
)

from src.preprocess.admin_boundaries import KenyaAdminPreprocessor

from scripts.utils import get_data_path


def process_vci(subset_str: str = "kenya"):
    data_path = get_data_path()
    processor = VHIPreprocessor(get_data_path(), "VCI")
    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor.preprocess(
        subset_str=subset_str, resample_time="M", upsampling=False, regrid=regrid_path
    )


def process_precip_2018(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = CHIRPSPreprocessor(data_path)

    processor.preprocess(subset_str=subset_str, regrid=regrid_path, parallel=False)


def process_era5POS_2018(subset_str: str = "kenya"):
    data_path = get_data_path()
    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = PlanetOSPreprocessor(data_path)

    processor.preprocess(
        subset_str=subset_str,
        regrid=regrid_path,
        parallel=False,
        resample_time="M",
        upsampling=False,
    )


def preprocess_era5_land(
    variables: Optional[Union[List, str]] = [
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "potential_evaporation",
            "total_precipitation",
            "2m_temperature",
            "total_evaporation"
    ],
    subset_str: str = "kenya",
    monmean: bool = True,
):
    data_path = get_data_path()
    if monmean:
        dataset = "reanalysis-era5-land-monthly-means"
    else:
        dataset = "reanalysis-era5-land"
    # Check all the provided variables exist
    if variables is None:
        variables = [d.name for d in (data_path / f"raw/{dataset}").iterdir()]
        assert (
            variables != []
        ), f"Expecting to find some variables in: {(data_path / 'raw/')}"
    else:
        if isinstance(variables, str):
            variables = [variables]
            assert variables in [
                d.name for d in (data_path / f"raw/{dataset}").iterdir()
            ], f"Expect to find {variables} in {(data_path / f'raw/{dataset}')}"
        else:
            assert all(
                np.isin(
                    variables,
                    [
                        d.name
                        for d in (data_path / f"raw/{dataset}").iterdir()
                    ],
                )
            ), f"Expected to find {variables}"

    # regrid_path = data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    # assert regrid_path.exists(), f"{regrid_path} not available"
    regrid_path = None

    for variable in variables:

        if monmean:
            processor = ERA5LandMonthlyMeansPreprocessor(data_path)
        else:
            processor = ERA5LandPreprocessor(data_path)

        processor.preprocess(
            subset_str=subset_str,
            regrid=None,
            resample_time="M",
            upsampling=False,
            variable=variable,
        )


def process_gleam(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = GLEAMPreprocessor(data_path)

    processor.preprocess(
        subset_str=subset_str, regrid=regrid_path, resample_time="M", upsampling=False
    )


def process_seas5(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = data_path / "interim/chirps_preprocessed/chirps_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = S5Preprocessor(data_path)
    processor.preprocess(
        subset_str=subset_str, regrid=regrid_path, resample_time="M", upsampling=False
    )


def process_esa_cci_landcover(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(subset_str=subset_str, regrid=regrid_path)


def preprocess_srtm(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    print(
        "Warning: regridding with CDO using the VCI preprocessor data fails because"
        "CDO reads the grid type as generic instead of latlon. This can be fixed "
        "just by changing the grid type to latlon in the grid definition file."
    )

    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str=subset_str, regrid=regrid_path)


def preprocess_kenya_boundaries(selection: str = "level_1"):
    assert selection in [
        f"level_{i}" for i in range(1, 6)
    ], f'selection must be one of {[f"level_{i}" for i in range(1,6)]}'

    data_path = get_data_path()

    regrid_path = data_path / "interim/chirps_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = KenyaAdminPreprocessor(data_path)
    processor.preprocess(reference_nc_filepath=regrid_path, selection=selection)


def preprocess_asal_mask():
    data_path = get_data_path()

    regrid_path = data_path / "interim/chirps_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = KenyaASALMask(data_path)
    processor.preprocess(reference_nc_filepath=regrid_path)


def preprocess_era5(subset_str: str = "kenya"):
    data_path = get_data_path()

    # regrid_path = data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    # assert regrid_path.exists(), f"{regrid_path} not available"
    regrid_path = None

    processor = ERA5MonthlyMeanPreprocessor(data_path)
    processor.preprocess(subset_str=subset_str, regrid=regrid_path)


def preprocess_era5_hourly(subset_str: str = "kenya"):
    data_path = get_data_path()

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = ERA5HourlyPreprocessor(data_path)

    # W-MON is weekly each monday (the same as the NDVI data from Atzberger)
    processor.preprocess(subset_str=subset_str, resample_time="W-MON")
    # processor.merge_files(subset_str='W-MON')


def preprocess_boku_ndvi(subset_str: str = "kenya"):
    data_path = get_data_path()
    processor = BokuNDVIPreprocessor(data_path)

    regrid_path = (
        data_path / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor.preprocess(
        subset_str=subset_str, resample_time="W-MON", regrid=regrid_path
    )


def preprocess_india_boundaries(selection: str = "level_1"):
    data_path = get_data_path()
    subset_str = "india"

    regrid_path = (
        data_path / f"interim/VCI_preprocessed/data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = IndiaAdminProcessor(data_path)
    processor.preprocess(reference_nc_filepath=regrid_path, selection=selection)


def process_boundaries(subset_str: str):
    if subset_str == "kenya":
        preprocess_kenya_boundaries(selection="level_1")
        preprocess_kenya_boundaries(selection="level_2")
        preprocess_kenya_boundaries(selection="level_3")
    if subset_str == "india":
        preprocess_india_boundaries(selection="level_1")
        preprocess_india_boundaries(selection="level_2")


if __name__ == "__main__":
    subset_str = "india"
    # preprocess_era5(subset_str=subset_str)
    # preprocess_era5_land(subset_str=subset_str, monmean=True)
    # process_vci(subset_str=subset_str)
    # process_precip_2018(subset_str=subset_str)
    # process_era5POS_2018(subset_str=subset_str)
    # process_gleam(subset_str=subset_str)
    # process_esa_cci_landcover(subset_str=subset_str)
    preprocess_srtm(subset_str=subset_str)
    # preprocess_era5_hourly(subset_str=subset_str)
    # preprocess_boku_ndvi(subset_str=subset_str)
    # preprocess_asal_mask(subset_str=subset_str)
    # process_boundaries(subset_str)