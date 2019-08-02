from pathlib import Path

from typing import Union, List, Optional

from .nowcast import _NowcastEngineer
from .one_month_forecast import _OneMonthForecastEngineer
from .base import _EngineerBase


class Engineer:
    r"""The engineer is responsible for taking all the data from the preprocessors,
    and saving a single training netcdf file to be ingested by machine learning models.

    Training and test sets are defined here, to ensure different machine learning models have
    access to the same data.

    :param data_folder: The location of the data folder.
    :param process_static: Whether to process static data
    :param experiment: One of `{'one_month_forecast', 'nowcast'}, defines the experiment for which
        the dataset is created
    """

    def __init__(self, data_folder: Path = Path('data_path'),
                 process_static: bool = True,
                 experiment: str = 'one_month_forecast') -> None:

        assert experiment in {'one_month_forecast', 'nowcast'},\
            'Experiment not recognized! Must be one of {nowcast, one_month_forecast}'

        engineer: _EngineerBase
        if experiment == 'one_month_forecast':
            engineer = _OneMonthForecastEngineer(data_folder, process_static)
        elif experiment == 'nowcast':
            engineer = _NowcastEngineer(data_folder, process_static)
        self.engineer_class = engineer

    def engineer(self, test_year: Union[int, List[int]],
                 target_variable: str = 'VHI',
                 pred_months: int = 12,
                 expected_length: Optional[int] = 12,
                 ) -> None:
        """
        Take all the preprocessed data generated by the preprocessing classes, and turn it
        into a single training file to be ingested by the machine learning models.

        :param test_year: Data to be used for testing. No data earlier than the earliest test year
            will be used for training. If a list is passed, a file for each year will be saved.
        :param target_variable: The variable to be predicted. Only this variable will be saved in
            the test netcdf files
        :param pred_months: The amount of months of data to feed as input to the model for
            it to make its prediction
        :param expected_length: The expected length of the x data along its time-dimension.
            If this is not None and an x array has a different time dimension size, the array
            is ignored. This differs from pred_months if the preprocessors are run with a
            time granularity different from `'M'`
        """
        self.engineer_class.engineer(test_year, target_variable, pred_months, expected_length)

    @staticmethod
    def engineer_static_only(data_folder: Path = Path('data')):
        """
        Only process static data (i.e. data in interim/static).

        :param data_folder: The location of the data folder.
        """
        engineer = _EngineerBase(data_folder, process_static=True)
        engineer._process_static()
