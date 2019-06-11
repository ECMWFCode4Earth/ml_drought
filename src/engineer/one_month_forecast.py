import numpy as np
import calendar
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
import pickle
import xarray as xr
import warnings

from typing import cast, DefaultDict, Dict, List, Optional, Union, Tuple

from ..utils import minus_months
from .engineer import Engineer

class OneMonthForecast(Engineer):
    def __init__(self, data_folder: Path = Path('data')) -> None:
        self.name: str = 'one_month_forecast'

        super().__init__(self.name, data_folder)

    @staticmethod
    def stratify_xy(ds: xr.Dataset,
                    year: int,
                    target_variable: str,
                    target_month: int,
                    pred_months: int,
                    expected_length: Optional[int]
                    ) -> Tuple[Optional[Dict[str, xr.Dataset]], date]:

        print(f'Generating data for year: {year}, target month: {target_month}')

        max_date = date(year, target_month, calendar.monthrange(year, target_month)[-1])
        mx_year, mx_month, max_train_date = minus_months(year, target_month, diff_months=1)
        _, _, min_date = minus_months(mx_year, mx_month, diff_months=pred_months)

        # `max_date` is the date to be predicted;
        # `max_train_date` is one timestep before;
        min_date_np = np.datetime64(str(min_date))
        max_date_np = np.datetime64(str(max_date))
        max_train_date_np = np.datetime64(str(max_train_date))

        print(f'Max date: {str(max_date)}, max input date: {str(max_train_date)}, '
              f'min input date: {str(min_date)}')

        # boolean array indexing the timestamps to filter `ds`
        x = ((ds.time.values > min_date_np) & (ds.time.values <= max_train_date_np))
        y = ((ds.time.values > max_train_date_np) & (ds.time.values <= max_date_np))

        # only expect ONE y timestamp
        if sum(y) != 1:
            print(f'Wrong number of y values! Expected 1, got {sum(y)}; returning None')
            return None, cast(date, max_train_date)

        if expected_length is not None:
            if sum(x) != expected_length:
                print(f'Wrong number of x values! Got {sum(x)} Returning None')

                return None, cast(date, max_train_date)

        # filter the dataset
        x_dataset = ds.isel(time=x)
        y_dataset = ds.isel(time=y)[target_variable].to_dataset(name=target_variable)

        if x_dataset.time.size != pred_months:
            # catch the errors as we get closer to the MINIMUM year
            warnings.warn('For the `nowcast` experiment we expect the' +
                          f' number of timesteps to be: {pred_months}.' +
                          f' Currently: {x_dataset.time.size}')
            return None, cast(date, max_train_date)


        return {'x': x_dataset, 'y': y_dataset}, cast(date, max_train_date)
