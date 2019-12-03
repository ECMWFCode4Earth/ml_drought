from pathlib import Path
from collections import OrderedDict
from itertools import dropwhile

from src.exporters import (
    ERA5Exporter,
    VHIExporter,
    ERA5ExporterPOS,
    GLEAMExporter,
    CHIRPSExporter,
    SRTMExporter,
)
from src.preprocess import (
    VHIPreprocessor,
    ERA5MonthlyMeanPreprocessor,
    GLEAMPreprocessor,
    CHIRPSPreprocessor,
    SRTMPreprocessor,
)
from src.engineer import Engineer
import src.models

from typing import Dict, Optional


class DictWithDefaults:
    def __init__(self, user_config: Dict, default_config: Dict) -> None:
        self.user_config = user_config
        self.default_config = default_config
        self._check_keys()

    def _check_keys(self) -> None:

        # To be updated as the pipeline grows
        expected_keys = {"data", "export", "preprocess", "engineer"}

        for key in expected_keys:
            try:
                self.user_config[key]
            except KeyError:
                try:
                    self.default_config[key]
                except KeyError:
                    assert False, (
                        f"{key} is not defined "
                        f"in the user config or the default config. Try using "
                        f"the default config in pipeline_config/(minimal, full).json"
                    )

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
            "era5": ERA5Exporter,
            "vhi": VHIExporter,
            "era5POS": ERA5ExporterPOS,
            "gleam": GLEAMExporter,
            "chirps": CHIRPSExporter,
            "srtm": SRTMExporter,
        }

        for dataset, variables in export_args.items():

            self._check_dataset(dataset, dataset2exporter)

            assert (
                type(variables) is list
            ), f"Expected {dataset} values to be a list. Got {type(variables)} instead"

            try:
                exporter = dataset2exporter[dataset](self.data)

                for variable in variables:
                    _ = exporter.export(**variable)  # type: ignore
            except Exception as e:
                print(f"Exception {e} raised for {dataset}")

    def process(self, preprocess_args: Dict) -> None:
        """preprocess the data

        subset; assign coordinates (latitude/longitude/time); regrid;
        resample timesteps;
        """
        dataset2preprocessor = {
            "vhi": VHIPreprocessor,
            "gleam": GLEAMPreprocessor,
            "reanalysis-era5-single-levels-monthly-means": ERA5MonthlyMeanPreprocessor,
            "chirps": CHIRPSPreprocessor,
            "srtm": SRTMPreprocessor,
        }

        def process_dataset(data: Path, dataset: str, args: Dict) -> None:
            args["init_args"]["data_folder"] = data
            preprocessor = dataset2preprocessor[dataset](**args["init_args"])
            preprocessor.preprocess(**args["run_args"])  # type: ignore

        try:
            regrid_dataset = preprocess_args.pop("regrid_dataset")
            self._check_dataset(regrid_dataset, dataset2preprocessor)
            dataset_args = preprocess_args.pop(regrid_dataset)

            process_dataset(self.data, regrid_dataset, dataset_args)

            regrid_folder = self.data / f"interim/{regrid_dataset}/"
            regrid_file: Optional[Path] = list(regrid_folder.glob("*.nc"))[0]
        except KeyError:
            regrid_file = None

        for dataset, args in preprocess_args.items():
            try:
                self._check_dataset(dataset, dataset2preprocessor)
                args["run_args"]["regrid"] = regrid_file
                process_dataset(self.data, dataset, args)
            except Exception as e:
                print(f"Exception {e} raised for {dataset}")

    def engineer(self, engineer_args: Dict) -> None:
        """Run the engineer on the data
        """
        engineer_args["init_args"]["data_folder"] = self.data
        engineer = Engineer(**engineer_args["init_args"])
        engineer.engineer(**engineer_args["run_args"])

    def train_models(self, model_args: Dict) -> None:

        for model_name, args in model_args.items():

            try:
                model_class = getattr(src.models, model_name)
            except AttributeError:
                print(f"{model_name} not a model class! Skipping")
                continue
            args["init_args"]["data_folder"] = self.data
            model = model_class(**args["init_args"])
            model.train(**args["train_args"])

            if "evaluate_args" in args:
                model.evaluate(**args["evaluate_args"])
            model.save_model()

    def _check_dataset(self, dataset: str, dataset_dict: Dict) -> None:
        # check the format is as we expected
        assert (
            dataset in dataset_dict
        ), f"{dataset} is not supported! Supported datasets are {dataset_dict.keys()}"

    def run(self, config: DictWithDefaults, run_from: str) -> None:

        run_steps = OrderedDict(
            {
                "export": self.export,
                "preprocess": self.process,
                "engineer": self.engineer,
                "models": self.train_models,
            }
        )

        for key in dropwhile(lambda k: k != run_from, run_steps):
            run_func = run_steps[key]
            run_func(config[key])
