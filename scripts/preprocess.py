from pathlib import Path

import sys
sys.path.append('..')
from src.preprocess import (VHIPreprocessor, CHIRPSPreprocesser,
                            PlanetOSPreprocessor, GLEAMPreprocessor,
                            ESACCIPreprocessor, SRTMPreprocessor)


def process_vci_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    processor = VHIPreprocessor(data_path, 'VCI')

    processor.preprocess(subset_str='kenya',
                         parallel=False, resample_time='M', upsampling=False)


def process_precip_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    regrid_path = data_path / 'interim/VCI_preprocessed/VCI_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = CHIRPSPreprocesser(data_path)

    processor.preprocess(subset_str='kenya',
                         regrid=None,
                         parallel=False)


def process_era5POS_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = PlanetOSPreprocessor(data_path)

    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         parallel=False, resample_time='M', upsampling=False)


def process_gleam():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = GLEAMPreprocessor(data_path)

    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         resample_time='M', upsampling=False)


def process_esa_cci_landcover():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


def preprocess_srtm():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


if __name__ == '__main__':
    process_vci_2018()
    process_precip_2018()
    process_era5POS_2018()
    process_gleam()
    process_esa_cci_landcover()
    preprocess_srtm()
