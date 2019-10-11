from pathlib import Path

from typing import Dict

from src.exporters import (ERA5Exporter, VHIExporter, ERA5ExporterPOS, GLEAMExporter,
                           CHIRPSExporter)
from src.preprocess import (VHIPreprocessor, ERA5MonthlyMeanPreprocessor,
                            GLEAMPreprocessor, CHIRPSPreprocesser)
from src.engineer import Engineer
import src.models


class DictWithDefaults:

    def __init__(self, user_config: Dict, default_config: Dict) -> None:
        self.user_config = user_config
        self.default_config = default_config
        self._check_keys()

    def _check_keys(self) -> None:

        # To be updated as the pipeline grows
        expected_keys = {'data', 'export', 'preprocess', 'engineer'}

        for key in expected_keys:
            try:
                self.user_config[key]
            except KeyError:
                try:
                    self.default_config[key]
                except KeyError:
                    assert False, f'{key} is not defined ' \
                        f'in the user config or the default config. Try using ' \
                        f'the default config in pipeline_config/(minimal, full).json'

    def __getitem__(self, key: str):

        try:
            return self.user_config[key]
        except KeyError:
            return self.default_config[key]


class Run:
    """Run the pipeline end to end

    Attributes
    ----------
    data: pathlib.Path
        The path to the data folder
    """
    def __init__(self, data: Path) -> None:
        self.data = data

    def export(self, export_args: Dict) -> None:
        """Export the data

        Arguments
        ----------
        export_args: dict
            A dictionary of format {dataset: [{variable_1 arguments}, {variable_2 arguments]}
            for all the variables to be exported
        """

        dataset2exporter = {
            'era5': ERA5Exporter,
            'vhi': VHIExporter,
            'era5POS': ERA5ExporterPOS,
            'gleam': GLEAMExporter,
            'chirps': CHIRPSExporter
        }

        for dataset, variables in export_args.items():

            # check the format is as we expected
            assert dataset in dataset2exporter, \
                f'{dataset} is not supported! Supported datasets are {dataset2exporter.keys()}'

            assert type(variables) is list, \
                f'Expected {dataset} values to be a list. Got {type(variables)} instead'

            exporter = dataset2exporter[dataset](self.data)

            for variable in variables:
                _ = exporter.export(**variable)  # type: ignore

    def process(self, preprocess_args: Dict) -> None:
        """preprocess the data

        subset; assign coordinates (latitude/longitude/time); regrid;
        resample timesteps;
        """
        dataset2preprocessor = {
            'vhi': VHIPreprocessor,
            'gleam': GLEAMPreprocessor,
            'reanalysis-era5-single-levels-monthly-means': ERA5MonthlyMeanPreprocessor,
            'chirps': CHIRPSPreprocesser
        }

        for dataset, variables in preprocess_args.items():

            # check the format is as we expected
            assert dataset in dataset2preprocessor, \
                f'{dataset} is not supported! Supported datasets are {dataset2preprocessor.keys()}'

            assert type(variables) is list, \
                f'Expected {dataset} values to be a list. Got {type(variables)} instead'

            preprocessor = dataset2preprocessor[dataset](self.data)

            for variable in variables:
                preprocessor.preprocess(**variable)

    def engineer(self, engineer_args: Dict) -> None:
        """Run the engineer on the data
        """
        engineer = Engineer(**engineer_args['init_args'])
        engineer.engineer(**engineer_args['run_args'])

    def train_models(self, model_args: Dict) -> None:

        for model_name, args in model_args.dict():

            try:
                model_class = getattr(src.models, model_name)
            except AttributeError:
                print(f'{model_name} not a model class! Skipping')
                continue
            model = model_class(**args['init_args'])
            model.train(**args['train_args'])

            if 'evaluate_args' in args:
                model.evaluate(**args['evaluate_args'])
            model.save_model()

    def run(self, config: DictWithDefaults) -> None:

        self.export(config['export'])
        self.process(config['preprocess'])
        self.engineer(config['engineer'])
        self.train_models(config['models'])
