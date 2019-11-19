"""
ERA5 path:
data_dir / 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'

VCI path:
data_dir / 'interim/VCI_preprocessed/data_kenya.nc'

CHIRPS path:
data_dir / 'interim/chirps_preprocessed/data_kenya.nc'
"""

from pathlib import Path

import sys
sys.path.append('..')
from src.preprocess import (VHIPreprocessor, CHIRPSPreprocessor,
                            PlanetOSPreprocessor, GLEAMPreprocessor,
                            S5Preprocessor, NDVIPreprocessor,
                            ESACCIPreprocessor, SRTMPreprocessor,
                            ERA5MonthlyMeanPreprocessor)

from src.preprocess.admin_boundaries import KenyaAdminPreprocessor


def process_vci_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    processor = VHIPreprocessor(data_path, 'VCI')

    processor.preprocess(subset_str='kenya',
                         resample_time='M', upsampling=False)


def process_precip_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    regrid_path = data_path / 'interim/VCI_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = CHIRPSPreprocessor(data_path)

    processor.preprocess(subset_str='kenya',
                         regrid=regrid_path,
                         parallel=False)


def process_era5POS_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/VCI_preprocessed/data_kenya.nc'
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
    regrid_path = data_path / 'interim/VCI_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = GLEAMPreprocessor(data_path)

    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         resample_time='M', upsampling=False)

def process_seas5():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = S5Preprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path,
                         resample_time='M', upsampling=False)


def process_esa_cci_landcover():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/VCI_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


def preprocess_srtm():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/VCI_preprocessed/data_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    print('Warning: regridding with CDO using the VCI preprocessor data fails because'
          'CDO reads the grid type as generic instead of latlon. This can be fixed '
          'just by changing the grid type to latlon in the grid definition file.')

    processor = SRTMPreprocessor(data_path)
    processor.preprocess(subset_str='kenya', regrid=regrid_path)


def preprocess_kenya_boundaries(selection: str = 'level_1'):
    assert selection in [f'level_{i}' for i in range(1,6)], \
        f'selection must be one of {[f"level_{i}" for i in range(1,6)]}'

    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = data_path / 'interim/chirps_preprocessed/chirps_kenya.nc'
    assert regrid_path.exists(), f'{regrid_path} not available'

    processor = KenyaAdminPreprocessor(data_path)
    processor.preprocess(
        reference_nc_filepath=regrid_path, selection=selection
    )


def preprocess_era5():
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    regrid_path = None
    # regrid_path = data_path / 'interim/VCI_preprocessed/data_kenya.nc'
    # assert regrid_path.exists(), f'{regrid_path} not available'

    processor = ERA5MonthlyMeanPreprocessor(data_path)
    processor.preprocess(subset_str='kenya')


def process_ndvi(
    years=None,
    regrid=True,
    ignore_timesteps=['2018-05-28', '2016-08-26'],
    resample_time=None,
):
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    if regrid:
        regrid_path = data_path / \
            'interim/precip/precip_doy.nc'
            # 'interim/reanalysis-era5-single-levels-monthly-means_preprocessed/data_kenya.nc'
        assert regrid_path.exists(), f'{regrid_path} not available'
    else:
        regrid_path = None

    processor = NDVIPreprocessor(data_path)
    processor.preprocess(subset_str='africa', regrid=regrid_path,
                        resample_time=resample_time, upsampling=False,
                        ignore_timesteps=ignore_timesteps,
                        years_to_process=years)


if __name__ == '__main__':
    # process_vci_2018()
    # process_precip_2018()
    # process_era5POS_2018()
    # process_gleam()
    # process_esa_cci_landcover()
    # preprocess_srtm()
    preprocess_era5()
    # preprocess_kenya_boundaries(selection='level_1')
    # preprocess_kenya_boundaries(selection='level_2')
    # preprocess_kenya_boundaries(selection='level_3')
    # process_ndvi(
    #     regrid=True, resample_time=None,
    #     years=[y for y in range(2010, 2019)]
    # )
