from pathlib import Path

import sys
sys.path.append('..')
from src.exporters import (ERA5Exporter, VHIExporter,
                           CHIRPSExporter, ERA5ExporterPOS,
                           GLEAMExporter)


def export_precip_2018():
    # if the working directory is alread ml_drought don't need ../data
    if Path('.').absolute().as_posix().split('/')[-1] == 'ml_drought':
        data_path = Path('data')
    else:
        data_path = Path('../data')
    exporter = ERA5Exporter(data_path)

    selection_request = {
        'year': ['2018'],
        'month': ['01'],
        'day': ['01'],
        'time': ['00:00']
    }

    exporter.export(variable='total_precipitation',
                    selection_request=selection_request)


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


if __name__ == '__main__':
    export_vhi()
    export_chirps()
    export_era5POS()
    export_gleam()
