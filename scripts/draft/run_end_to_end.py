from pathlib import Path
import sys

# get the project directory
if Path('.').absolute().name == 'ml_drought':
    sys.path.append('.')
elif Path('.').absolute().parents[0].name == 'ml_drought':
    sys.path.append('..')
elif Path('.').absolute().parents[1].name == 'ml_drought':
    sys.path.append('../..')
else:
    assert False, 'We cant find the base project directory `ml_drought`'

# 1. exporters
from src.exporters import (ERA5Exporter, VHIExporter,
                           CHIRPSExporter, ERA5ExporterPOS,
                           GLEAMExporter, ESACCIExporter)
from src.exporters.srtm import SRTMExporter

# 2. preprocessors
from src.preprocess import (VHIPreprocessor, CHIRPSPreprocesser,
                            PlanetOSPreprocessor, GLEAMPreprocessor,
                            ESACCIPreprocessor)
from src.preprocess.srtm import SRTMPreprocessor

# 3. engineer
from src.engineer import Engineer

# 4. model
from src.models import (Persistence, LinearRegression,
                        LinearNetwork, RecurrentNetwork,
                        EARecurrentNetwork)
from src.models.data import DataLoader

# 5. analysis
from src.analysis import plot_shap_values

# ----------------------------------------------------------------
# Export (download the data)
# ----------------------------------------------------------------

def export_data(data_path):
    # target variable
    print('** Exporting VHI **')
    exporter = VHIExporter(data_path)
    exporter.export()
    del exporter

    # precip
    print('** Exporting CHIRPS Precip **')
    exporter = CHIRPSExporter(data_path)
    exporter.export(years=None, region='global', period='monthly')
    del exporter

    # evaporation
    print('** Exporting GLEAM Evaporation **')
    exporter = GLEAMExporter(data_folder=data_path)
    exporter.export(['E'], 'monthly')
    del exporter

    # topography
    print('** Exporting SRTM Topography **')
    exporter = SRTMExporter(data_folder=data_path)
    exporter.export()
    del exporter

    # landcover
    print('** Exporting Landcover **')
    exporter = ESACCIExporter(data_folder=data_path)
    exporter.export()
    del exporter


# ----------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------


def preprocess_data(data_path):
    # preprocess VHI
    print('** Preprocessing VHI **')
    processor = VHIPreprocessor(data_path)
    processor.preprocess(
        subset_str='kenya', regrid=regrid_path,
         n_parallel_processes=1, resample_time='M',
         upsampling=False
    )

    regrid_path = data_path / 'interim' / 'vhi_preprocessed' / 'vhi_kenya.nc'

    # preprocess CHIRPS Rainfall
    print('** Preprocessing CHIRPS Precipitation **')
    processor = CHIRPSPreprocesser(data_path)
    processor.preprocess(
        subset_str='kenya', regrid=regrid_path,
        n_parallel_processes=1
    )

    # preprocess GLEAM evaporation
    print('** Preprocessing GLEAM Evaporation **')
    processor = GLEAMPreprocessor(data_path)
    processor.preprocess(
        subset_str='kenya', regrid=regrid_path,
        resample_time='M', upsampling=False
    )

    # preprocess SRTM Topography
    print('** Preprocessing SRTM Topography **')
    processor = SRTMPreprocessor(data_path)
    processor.preprocess(
        subset_str='kenya', regrid=regrid_path
    )

    # preprocess ESA CCI Landcover
    print('** Preprocessing ESA CCI Landcover **')
    processor = ESACCIPreprocessor(data_path)
    processor.preprocess(
        subset_str='kenya', regrid=regrid_path,
        resample_time='M', upsampling=False
    )


# ----------------------------------------------------------------
# Engineer (train/test split for each experiment)
# ----------------------------------------------------------------


def engineer(data_path, experiment='one_month_forecast', process_static=True,
             pred_months=12, expected_length=12):
    engineer = Engineer(data_path, experiment=experiment, process_static=process_static)
    engineer.engineer(
        test_year=2018, target_variable='VHI',
        pred_months=pred_months, expected_length=pred_months,
    )


def engineer_data(data_path):
    print('** Engineering data for one_month_forecast experiment **')
    engineer(experiment='one_month_forecast')

    print('** Engineering data for nowcast experiment **')
    engineer(experiment='nowcast')


if __name__ == '__main__':
    # get the data path
    if Path('.').absolute().name == 'ml_drought':
        data_path = Path('data')
    elif Path('.').absolute().parents[0].name == 'ml_drought':
        data_path = Path('../data')
    elif Path('.').absolute().parents[1].name == 'ml_drought':
        data_path = Path('../../data')
    else:
        assert False, 'We cant find the base package directory `ml_drought`'

    # export
    print("** Running Exporters **")
    export_data(data_path)

    # preprocess
    print("** Running Preprocessors **")
    preprocess_data(data_path)

    # engineer
    print("** Running Engineer **")
    engineer_data(data_path)
