import numpy as np
import calendar
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
import pickle
import xarray as xr

from typing import cast, DefaultDict, Dict, List, Optional, Union, Tuple

from ..utils import minus_months
from .engineer import Engineer


class Nowcast(Engineer):
    """Engineer the preprocessed `.nc` files into `/train`, `/test` `{x, y}.nc`


    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        super().__init__(data_folder)

        self.output_folder = data_folder / 'features' / 'nowcast'
        if not self.output_folder.exists():
            self.output_folder.mkdir()

    def engineer(self,
                 test_year: Union[int, List[int]],
                 target_variable: str = 'VHI',
                 pred_months: int = 11,
                 expected_length: Optional[int] = 11,
                 experiment: str = '1month_forecast',
                 ):
        # read in all the data from interim/{var}_preprocessed
        data = self._make_dataset()

        # ensure test_year is List[int]
        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        # save test data (x, y) and return the train_ds (subset of `data`)
        train_ds = self._train_test_split(
            data, cast(List, test_year), target_variable,
            pred_months, expected_length
        )

        # train_ds into x, y for each year-month in trianing period
        self._stratify_training_data(train_ds, target_variable, pred_months,
                                     expected_length)

        for var in self.normalization_values.keys():
            self.normalization_values[var]['mean'] /= self.num_normalization_values
            self.normalization_values[var]['std'] /= self.num_normalization_values

        savepath = self.output_folder / 'normalizing_dict.pkl'
        with savepath.open('wb') as f:
            pickle.dump(self.normalization_values, f)


        return

    def _stratify_training_data(self, train_ds: xr.Dataset,
                                target_variable: str,
                                pred_months: int = 11,
                                ) -> None:
        """split `train_ds` into x, y and save the outputs to
        self.output_folder (data/features) """
        return

    def _train_test_split(self, ds: xr.Dataset,
                          years: List[int],
                          target_variable: str,
                          pred_months: int = 11,
                          ) -> xr.Dataset:

        years.sort()

        # for the first `year` Jan calculate the xy_test dictionary and min date
        xy_test, min_test_date = self.stratify_xy(
            ds=ds, year=years[0], target_variable=target_variable,
            target_month=1, pred_months=pred_months
        )

        # the train_ds MUST BE from before minimum test date
        train_dates = ds.time.values < np.datetime64(str(min_test_date))
        train_ds = ds.isel(time=train_dates)

        # save the xy_test dictionary
        if xy_test is not None:
            self._save(xy_test, year=years[0], month=1,
                       dataset_type='test')

        # each month in test_year produce an x,y pair for testing
        for year in years:
            for month in range(1, 13):
                if year > years[0] or month > 1:
                    # prevents the initial test set from being recalculated
                    xy_test, _ = self.stratify_xy(
                        ds=ds, year=year, target_variable=target_variable,
                        target_month=month, pred_months=pred_months,
                    )
                    if xy_test is not None:
                        self._save(xy_test, year=year, month=month,
                                   dataset_type='test')
        return train_ds


        return

    @staticmethod
    def stratify_xy(ds: xr.Dataset,
                    year: int,
                    target_variable: str,
                    target_month: int,
                    pred_months: int,
                    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:
        """
        The nowcasting experiment has different lengths for the
         `target` variable vs. the `non_target` variables.

        e.g. if I set the `pred_months = 11`

        `x_target_variable` = 11 timesteps
        `x_non_target_variable` = 12 timesteps (`pred_months + 1`).

        We overcome this by creating an extra timestep with all nan values
        in the `x_dataset`. This way the `x_dataset` contains the `y_dataset`
        timestep but the `target_variable` is an array of all `np.nan` for that
        target timestep. This prevents model leakage.
        """
        print(f'Generating data for year: {year}, target month: {target_month}')

        # get the test datetime
        max_date = date(year, target_month, calendar.monthrange(year, target_month)[-1])
        mx_year, mx_month, max_train_date = minus_months(year, target_month, diff_months=1)
        _, _, min_date = minus_months(mx_year, mx_month, diff_months=pred_months)

        # convert to numpy datetime
        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))
        max_train_date_np = np.datetime64(str(max_train_date))

        print(f'Max date: {str(max_date)}, max input date: {str(max_train_date)}, '
              f'min input date: {str(min_date)}')

        # boolean array indexing the TARGET VARIABLE timestamps to filter `ds`
        x_target = (
            (ds.time.values > min_date_np) & (ds.time.values <= max_train_date_np)
        )
        y_target = (
            (ds.time.values > max_train_date_np) & (ds.time.values <= max_date_np)
        )

        # boolean array indexing the other variables
        x_non_target = (
            (ds.time.values > min_date_np) & (ds.time.values <= max_date_np)
        )

        # only expect ONE y timestamp
        if sum(y) != 1:
            print(f'Wrong number of y values! Expected 1, got {sum(y)}; returning None')
            return None, cast(date, max_train_date)

        # create the target dataset `y_dataset` & the `x_non_target_dataset`
        y_dataset = ds.isel(time=y_target)[target_variable].to_dataset(name=target_variable)
        x_non_target_dataset = ds.drop(target_variable).sel(time=x_non_target)

        # create the x_target_dataset with all nans at target time
        nan_target_variable = make_nan_dataset(y_dataset)
        x_target_dataset = (
            ds[target_variable].isel(time=x_target).to_dataset(name=target_variable)
        )
        x_target_dataset = x_target_dataset.merge(nan_target_variable)

        # merge the x_non_target_dataset + x_target_dataset -> x_dataset
        x_dataset = x_non_target_dataset.merge(x_target_dataset)

        return {'x': x_dataset, 'y': y_dataset}, cast(date, max_train_date)

    @staticmethod
    def make_nan_dataset(ds: Union[xr.Dataset, xr.DataArray],
                        ) -> Union[xr.Dataset, xr.DataArray]:
        nan_ds = xr.full_like(ds, np.nan)
        return nan_ds
#
