import sys

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
    KenyaASALMask,
)

from src.preprocess.admin_boundaries import KenyaAdminPreprocessor

from scripts.utils import get_data_path


def process_vci_2018():

    processor = VHIPreprocessor(get_data_path(), "VCI")

    processor.preprocess(subset_str="kenya", resample_time="M", upsampling=False)


def process_precip_2018():
    data_path = get_data_path()

    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = CHIRPSPreprocessor(data_path)

    processor.preprocess(subset_str="kenya", regrid=regrid_path, parallel=False)


def process_era5POS_2018():
    data_path = get_data_path()
    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = PlanetOSPreprocessor(data_path)

    processor.preprocess(
        subset_str="kenya",
        regrid=regrid_path,
        parallel=False,
        resample_time="M",
        upsampling=False,
    )


def process_gleam():
    data_path = get_data_path()

    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = GLEAMPreprocessor(data_path)

    processor.preprocess(
        subset_str="kenya", regrid=regrid_path, resample_time="M", upsampling=False
    )


def process_seas5():
    data_path = get_data_path()

    regrid_path = data_path / "interim/chirps_preprocessed/chirps_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = S5Preprocessor(data_path)
    processor.preprocess(
        subset_str="kenya", regrid=regrid_path, resample_time="M", upsampling=False
    )


def process_esa_cci_landcover():
    data_path = get_data_path()

    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(subset_str="kenya", regrid=regrid_path)


def preprocess_srtm():
    data_path = get_data_path()

    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    print(
        "Warning: regridding with CDO using the VCI preprocessor data fails because"
        "CDO reads the grid type as generic instead of latlon. This can be fixed "
        "just by changing the grid type to latlon in the grid definition file."
    )

    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str="kenya", regrid=regrid_path)


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


def preprocess_era5():
    data_path = get_data_path()

    regrid_path = data_path / "interim/VCI_preprocessed/data_kenya.nc"
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = ERA5MonthlyMeanPreprocessor(data_path)
    processor.preprocess(subset_str="kenya", regrid=regrid_path)


if __name__ == "__main__":
    # process_vci_2018()
    # process_precip_2018()
    # process_era5POS_2018()
    # process_gleam()
    # process_esa_cci_landcover()
    # preprocess_srtm()
    # preprocess_era5()
    # preprocess_kenya_boundaries(selection="level_1")
    # preprocess_kenya_boundaries(selection="level_2")
    # preprocess_kenya_boundaries(selection="level_3")
    preprocess_asal_mask()
