import numpy as np
import pandas as pd
import calendar
from datetime import date
import xarray as xr
import warnings
import pickle

from typing import cast, Dict, Optional, Tuple, List, Union

from ..utils import minus_months
from .one_month_forecast import _OneMonthForecastEngineer


class _DifferentTrainingPeriodsEngineer(_OneMonthForecastEngineer):
    name = "one_month_forecast"

    @staticmethod
    def check_data_leakage(
        train_ds: xr.Dataset, xy_test: Dict[str, xr.Dataset]
    ) -> None:
        # CHECK DATA LEAKAGE
        train_dts = [pd.to_datetime(t) for t in train_ds.time.values]
        test_dt = pd.to_datetime(xy_test["y"].time.values)
        assert test_dt not in train_dts, (
            "Data Leakage!" f"{test_dt} found in train_ds datetimes: \n {train_ds}"
        )

    def engineer(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
        train_years: Optional[List[int]] = None,
    ) -> None:

        self._process_dynamic(
            test_year, target_variable, pred_months, expected_length, train_years
        )
        if self.process_static:
            self._process_static()

    def _process_dynamic(
        self,
        test_year: Union[int, List[int]],
        target_variable: str = "VHI",
        pred_months: int = 12,
        expected_length: Optional[int] = 12,
        train_years: Optional[List[int]] = None,
    ) -> None:
        if expected_length is None:
            warnings.warn(
                "** `expected_length` is None. This means that \
            missing data will not be skipped. Are you sure? **"
            )

        # read in all the data from interim/{var}_preprocessed
        data = self._make_dataset(static=False)

        # ensure test_year is List[int]
        if type(test_year) is int:
            test_year = [cast(int, test_year)]

        # save test data (x, y) and return the train_ds (subset of `data`)
        train_ds = self._train_test_split(
            ds=data,
            test_years=cast(List, test_year),
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
            train_years=train_years,
        )
        assert train_ds.time.shape[0] > 0, (
            "Expect the train_ds to have" f"`time` dimension. \n{train_ds}"
        )

        if train_years is not None:
            # select only the TRAINING years in train_years
            ds_years = train_ds["time.year"].values
            bool_train_years = [y in train_years for y in ds_years]
            train_ds = train_ds.sel(time=bool_train_years)

        normalization_values = self._calculate_normalization_values(train_ds)

        # split train_ds into x, y for each year-month before `test_year` & save
        self._stratify_training_data(
            train_ds=train_ds,
            target_variable=target_variable,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        savepath = self.output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    def _train_test_split(
        self,
        ds: xr.Dataset,
        test_years: List[int],
        target_variable: str,
        pred_months: int,
        expected_length: Optional[int],
        train_years: Optional[List[int]] = None,
    ) -> xr.Dataset:
        """save the test data and return the training dataset"""

        test_years.sort()

        months = ds["time.month"].values

        if ds["time.year"].min() == test_years[0]:
            # if we have the FIRST year in our dataset as a test year
            # then minimum target month should be `pred_months + 1`
            # e.g. x = 1984 Jan/Feb/Mar, y = 1984 April
            init_target_month = pred_months + months[0]
        else:
            init_target_month = 1

        # NOTE: the test year is
        # for the first `year` calculate the xy_test dictionary and min date
        xy_test, min_test_date = self._stratify_xy(
            ds=ds,
            year=test_years[0],
            target_variable=target_variable,
            target_month=init_target_month,
            pred_months=pred_months,
            expected_length=expected_length,
        )

        # NOTE: relax assumption that train_ds MUST BEFORE before test date
        if train_years is None:
            # get the years BEFORE the test_year
            train_dates = ds.time.values <= np.datetime64(str(min_test_date))
        else:
            # train on the years defined
            train_dates = [
                int(y.values) in train_years for y in ds["time.year"]]
        train_ds = ds.isel(time=train_dates)

        # save the xy_test dictionary
        if xy_test is not None:
            self._save(
                xy_test,
                year=test_years[0],
                month=init_target_month,
                dataset_type="test",
            )

            # check for data leakage
            self.check_data_leakage(train_ds, xy_test)

        # each month in test_year produce an x,y pair for testing
        for year in test_years:
            for month in range(init_target_month, 13):
                if month > init_target_month:
                    # prevents the initial test set from being recalculated
                    xy_test, _ = self._stratify_xy(
                        ds=ds,
                        year=year,
                        target_variable=target_variable,
                        target_month=month,
                        pred_months=pred_months,
                        expected_length=expected_length,
                    )
                    if xy_test is not None:
                        # check for data leakage
                        self.check_data_leakage(train_ds, xy_test)

                        self._save(xy_test, year=year, month=month, dataset_type="test")
        return train_ds
