import numpy as np
from collections import defaultdict
import calendar
from datetime import datetime, date
from pathlib import Path
import xarray as xr

from typing import cast, Dict, List, Optional, Union, Tuple
from typing import DefaultDict as DDict

from ..utils import minus_months


class Engineer:
    """The engineer is responsible for taking all the data from the preprocessors,
    and saving a single training netcdf file to be ingested by machine learning models.

    Training and test sets are defined here, to ensure different machine learning models have
    access to the same data.

    Attributes:
    ----------
    data_folder: Path, default: Path('data')
        The location of the data folder.
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:

        self.data_folder = data_folder

        self.interim_folder = data_folder / 'interim'
        assert self.interim_folder.exists(), \
            f'{data_folder / "interim"} does not exist. Has the preprocesser been run?'

        self.output_folder = data_folder / 'features'
        if not self.output_folder.exists():
            self.output_folder.mkdir()

    def engineer(self, test_year: Union[int, List[int]],
                 target_variable: str = 'VHI',
                 pred_months: int = 11,
                 expected_length: Optional[int] = 11
                 ):
        """
        Take all the preprocessed data generated by the preprocessing classes, and turn it
        into a single training file to be ingested by the machine learning models.

        Arguments
        ----------
        test_year: Union[int, List[int]]
            Data to be used for testing. No data earlier than the earliest test year
            will be used for training.
            If a list is passed, a file for each year will be saved.
        target_variable: str = 'VHI'
            The variable to be predicted. Only this variable will be saved in the test
            netcdf files
        pred_months: int = 11
            The amount of months of data to feed as input to the model for it to make its
            prediction
        expected_length: Optional[int] = 11
            The expected length of the x data along its time-dimension. If this is not None
            and an x array has a different time dimension size, the array is ignored
        """
        data = self._make_dataset()

        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        train_ds, test_dict = self._train_test_split(data, cast(List, test_year), target_variable,
                                                     pred_months, expected_length)

        train_dict = self._stratify_training_data(train_ds, target_variable, pred_months,
                                                  expected_length)
        self._save(train_dict, 'train')
        self._save(test_dict, 'test')

    def _get_preprocessed_files(self) -> List[Path]:
        processed_files = []
        for subfolder in self.interim_folder.iterdir():
            if str(subfolder).endswith('_preprocessed') and subfolder.is_dir():
                processed_files.extend(list(subfolder.glob('*.nc')))
        return processed_files

    def _make_dataset(self) -> xr.Dataset:

        datasets = []
        dims = ['lon', 'lat']
        coords = {}
        for idx, file in enumerate(self._get_preprocessed_files()):
            print(f'Processing {file}')
            datasets.append(xr.open_dataset(file))

            if idx == 0:
                for dim in dims:
                    coords[dim] = datasets[idx][dim].values
            else:
                for dim in dims:
                    assert (datasets[idx][dim].values == coords[dim]).all(), \
                        f'{dim} is different! Was this run using the preprocessor?'

        main_dataset = datasets[0]
        for dataset in datasets[1:]:
            main_dataset = main_dataset.merge(dataset, join='inner')

        return main_dataset

    def _stratify_training_data(self, train_ds: xr.Dataset,
                                target_variable: str,
                                pred_months: int = 11,
                                expected_length: Optional[int] = 11
                                ) -> DDict[int, DDict[int, Dict[str, xr.Dataset]]]:

        min_date = self.get_datetime(train_ds.time.values.min())
        max_date = self.get_datetime(train_ds.time.values.max())

        cur_pred_year, cur_pred_month = max_date.year, max_date.month

        output_dict: DDict[int, DDict[int, Dict[str, xr.Dataset]]] = \
            defaultdict(lambda: defaultdict(dict))

        cur_min_date = max_date
        while cur_min_date >= min_date:

            arrays, cur_min_date = cast(Tuple[Optional[Dict[str, xr.Dataset]], date],
                                        self.stratify_xy(train_ds, cur_pred_year,
                                                         target_variable, cur_pred_month,
                                                         pred_months, expected_length, True))
            if arrays is not None:
                output_dict[cur_pred_year][cur_pred_month] = arrays
            cur_pred_year, cur_pred_month = cur_min_date.year, cur_min_date.month

        return output_dict

    def _train_test_split(self, ds: xr.Dataset,
                          years: List[int],
                          target_variable: str,
                          pred_months: int = 11,
                          expected_length: Optional[int] = 11,
                          ) -> Tuple[xr.Dataset, DDict[int, DDict[int, Dict[str, xr.Dataset]]]]:
        years.sort()

        output_test_arrays: DDict[int, DDict[int, Dict[str, xr.Dataset]]] = \
            defaultdict(lambda: defaultdict(dict))

        xy_test, min_test_date = self.stratify_xy(ds, years[0], target_variable, 1, pred_months,
                                                  expected_length, True)

        train_dates = ds.time.values <= np.datetime64(str(min_test_date))
        train_ds = ds.isel(time=train_dates)

        if xy_test is not None:
            output_test_arrays[years[0]][1] = xy_test

        for year in years:
            for month in range(1, 13):
                if year > years[0] or month > 1:
                    # prevents the initial test set from being recalculated
                    xy_test, _ = self.stratify_xy(ds, year, target_variable, month, pred_months,
                                                  expected_length)
                    if xy_test is not None:
                        output_test_arrays[year][month] = xy_test

        return train_ds, output_test_arrays

    @staticmethod
    def stratify_xy(ds: xr.Dataset,
                    year: int,
                    target_variable: str,
                    target_month: int,
                    pred_months: int,
                    expected_length: Optional[int],
                    return_min_date: bool = False,
                    ) -> Tuple[Optional[Dict[str, xr.Dataset]], Optional[date]]:

        print(f'Generating test data for year: {year}, target month: {target_month}')

        max_date = date(year, target_month, calendar.monthrange(year, target_month)[-1])
        mx_year, mx_month, max_train_date = minus_months(year, target_month, diff_months=1)
        _, _, min_date = minus_months(mx_year, mx_month, diff_months=pred_months)

        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))
        max_train_date_np = np.datetime64(str(max_train_date))

        print(f'Max date: {str(max_date)}, max train date: {str(max_train_date)}, '
              f'min train date: {str(min_date)}')

        if return_min_date:
            output_min_date: Optional[date] = min_date
        else:
            output_min_date = None

        x = ((ds.time.values > min_date_np) & (ds.time.values <= max_train_date_np))
        y = ((ds.time.values > max_train_date_np) & (ds.time.values <= max_date_np))

        if sum(y) == 0:
            print('Wrong number of test y values! Got 0; returning None')
            return None, output_min_date

        if expected_length is not None:
            if sum(x) != expected_length:
                print(f'Wrong number of test x values! Got {sum(x)} Returning None')

                return None, output_min_date

        x_dataset = ds.isel(time=x)
        y_dataset = ds.isel(time=y)[target_variable].to_dataset()

        return {'x': x_dataset, 'y': y_dataset}, max_train_date

    @staticmethod
    def get_datetime(time: np.datetime64) -> date:
        return datetime.strptime(time.astype(str)[:10], '%Y-%m-%d').date()

    def _save(self, ds: DDict[int, DDict[int, Dict[str, xr.Dataset]]],
              dataset_type: str) -> None:

        save_folder = self.output_folder / dataset_type
        save_folder.mkdir(exist_ok=True)

        for year_key, val in ds.items():

            for month_key, test_dict in val.items():

                output_location = save_folder / f'{year_key}_{month_key}'
                output_location.mkdir()

                for x_or_y, output_ds in test_dict.items():
                    output_ds.to_netcdf(output_location / f'{x_or_y}.nc')
