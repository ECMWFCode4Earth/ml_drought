from pathlib import Path

import sys
sys.path.append('..')
from src.preprocess import (VHIPreprocessor, CHIRPSPreprocesser,
                            PlanetOSPreprocessor, GLEAMPreprocessor,
                            NDVIPreprocessor,)


def process_precip_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    processor = CHIRPSPreprocesser(data_path)

    processor.preprocess(subset_str='kenya',
                         regrid=None,
                         parallel=False)


def process_vhi_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = VHIPreprocessor(data_path)

    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         parallel=False, resample_time='M', upsampling=False)


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


if __name__ == '__main__':
    process_precip_2018()
    process_vhi_2018()
    process_era5POS_2018()
    process_gleam()
