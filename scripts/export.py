from pathlib import Path
import numpy as np

import sys
sys.path.append('..')
from src.exporters import (ERA5Exporter, VHIExporter,
                           CHIRPSExporter, ERA5ExporterPOS,
                           GLEAMExporter, ESACCIExporter,
                           S5Exporter, SRTMExporter, NDVIExporter)


def export_era5():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    exporter = ERA5Exporter(data_path)

    exporter.export(variable='soil_temperature_level_1', granularity='monthly',
                    selection_request={'time': '00:00'})


def export_vhi():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    exporter = VHIExporter(data_path)

    exporter.export()


def export_chirps():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    exporter = CHIRPSExporter(data_path)

    exporter.export(years=None, region='global', period='monthly')


def export_era5POS():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    exporter = ERA5ExporterPOS(data_path)

    exporter.export(variable='precipitation_amount_1hour_Accumulation')


def export_gleam():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    exporter = GLEAMExporter(data_folder=data_path)
    exporter.export(['E', 'SMroot', 'SMsurf'], 'monthly')


<<<<<<< HEAD
def export_ndvi():
=======
def export_srtm():
    # if the working directory is alread ml_drought don't need ../data
>>>>>>> 6f09dd93e13f0708272d6c47d484b08796ab1984
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

<<<<<<< HEAD
    exporter = NDVIExporter(data_path)
    exporter.export()



def export_s5(
    granularity='hourly', pressure_level=False,
    variable='total_precipitation', min_year=1993,
    max_year=1994, min_month=1, max_month=12,
):
=======
    exporter = SRTMExporter(data_folder=data_path)
    exporter.export()


def export_esa():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

    exporter = ESACCIExporter(data_folder=data_path)
    exporter.export()


def export_s5():
    # if the working directory is alread ml_drought don't need ../data
>>>>>>> 6f09dd93e13f0708272d6c47d484b08796ab1984
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')

<<<<<<< HEAD
=======
    granularity = 'hourly'
    pressure_level = False

>>>>>>> 6f09dd93e13f0708272d6c47d484b08796ab1984
    exporter = S5Exporter(
        data_folder=data_path,
        granularity=granularity,
        pressure_level=pressure_level,
    )
<<<<<<< HEAD
    max_leadtime = None
    pressure_levels = [200, 500, 925]
    selection_request = None
    n_parallel_requests = 1
=======
    variable = 'total_precipitation'
    min_year = 1993
    max_year = 2014
    min_month = 1
    max_month = 12
    max_leadtime = None
    pressure_levels = [200, 500, 925]
    n_parallel_requests = 20
>>>>>>> 6f09dd93e13f0708272d6c47d484b08796ab1984

    exporter.export(
        variable=variable,
        min_year=min_year,
        max_year=max_year,
        min_month=min_month,
        max_month=max_month,
        max_leadtime=max_leadtime,
        pressure_levels=pressure_levels,
        n_parallel_requests=n_parallel_requests,
    )

<<<<<<< HEAD
=======

>>>>>>> 6f09dd93e13f0708272d6c47d484b08796ab1984
if __name__ == '__main__':
    export_era5()
    export_vhi()
    export_chirps()
    export_era5POS()
    export_gleam()
<<<<<<< HEAD
    export_ndvi()
    # export_s5()
=======
    export_esa()
    export_s5()
>>>>>>> 6f09dd93e13f0708272d6c47d484b08796ab1984
