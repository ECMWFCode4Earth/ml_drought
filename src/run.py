from pathlib import Path

from typing import Dict

from src.exporters import ERA5Exporter, VHIExporter, ERA5ExporterPOS
from src.preprocess import VHIPreprocessor


class DictWithDefaults:
    def __init__(self, user_config: Dict, default_config: Dict) -> None:
        self.user_config = user_config
        self.default_config = default_config
        self._check_keys()

    def _check_keys(self) -> None:

        # To be updated as the pipeline grows
        expected_keys = {"data", "export", "preprocess"}

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
        }

        for dataset, variables in export_args.items():

            # check the format is as we expected
            assert (
                dataset in dataset2exporter
            ), f"{dataset} is not supported! Supported datasets are {dataset2exporter.keys()}"

            assert (
                type(variables) is list
            ), f"Expected {dataset} values to be a list. Got {type(variables)} instead"

            exporter = dataset2exporter[dataset](self.data)

            for variable in variables:
                _ = exporter.export(**variable)  # type: ignore

    def process(self, preprocess_args: Dict) -> None:
        """preprocess the data

        subset; assign coordinates (latitude/longitude/time); regrid;
        resample timesteps;

        Arguments
        --------
        preprocess_args:
        """
        dataset2preprocessor = {"vhi": VHIPreprocessor}

        for dataset, variables in preprocess_args.items():

            # check the format is as we expected
            assert (
                dataset in dataset2preprocessor
            ), f"{dataset} is not supported! Supported datasets are {dataset2preprocessor.keys()}"

            assert (
                type(variables) is list
            ), f"Expected {dataset} values to be a list. Got {type(variables)} instead"

            preprocessor = dataset2preprocessor[dataset](self.data)

            for variable in variables:
                preprocessor.preprocess(**variable)

    def run(self, config: DictWithDefaults) -> None:

        self.export(config["export"])
