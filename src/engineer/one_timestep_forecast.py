import numpy as np
import calendar
from datetime import date
import xarray as xr
import warnings
import pandas as pd
from pandas.tseries.offsets import Day
from collections import defaultdict
import pickle
from collections.abc import Iterable

from typing import cast, Dict, Optional, Tuple, DefaultDict, List

from ..utils import minus_timesteps
from .base import _EngineerBase


class _OneTimestepForecastEngineer(_EngineerBase):
    name = "one_timestep_forecast"

    def stratify_xy(
        self,
        ds: xr.Dataset,
        target_var: str,
        target_time: np.datetime64,
        seq_length: int,
        min_ds_date: pd.Timestamp,
        train: bool = True,
        resolution: str = "D",
    ) -> Tuple[Optional[Dict[str, xr.Dataset]], pd.Timestamp]:
        # convert to pandas datetime (easier to work with)
        target_time = pd.to_datetime(target_time)

        # min/max date for X data
        max_X_date = minus_timesteps(target_time, 1, "D")
        min_X_date = minus_timesteps(target_time, seq_length, "D")
        print(f"Min: {min_X_date} Max: {max_X_date}")

        # check whether enough data
        if min_X_date < min_ds_date:
            print(f"Not enough input timesteps for {target_time}")
            return None, target_time

        print(f"Generating data for target time: {target_time}")
        # split into x, y
        X_dataset = ds.sel(time=slice(min_X_date, max_X_date))
        y_dataset = ds[[target_var]].sel(time=target_time)

        ds_dict = {"x": X_dataset, "y": y_dataset}

        dataset_type = "train" if train else "test"
        self._save(
            ds_dict, target_time, dataset_type=dataset_type, resolution=resolution
        )

        return ds_dict, target_time

    def _process_static(self) -> None:
        """This function assumes only one 'dimension_name' """
        output_file = self.static_output_folder / "data.nc"
        if output_file.exists():
            warnings.warn("A static data file already exists!")
            return None

        # NOTE: HARDCODED the dataset to open
        static_ds = xr.open_dataset(self.interim_folder / "static/data.nc")

        normalization_values: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        assert len([c for c in static_ds.coords]) == 1, "Only works with one dimension"
        dimension_name = [c for c in static_ds.coords][0]

        # TODO: ONE HOT ENCODE THE RELEVANT FEATURES
        # --------------------------------------------------
        # removing alot of information here ...
        # only keep the float variables
        # TODO: remove the features that are mostly missing!
        # remove features with >100 missing values
        float_vars = [v for v in static_ds.data_vars if static_ds[v].dtype == float]
        non_missing_float_vars = [
            v
            for v in static_ds[float_vars].data_vars
            if static_ds[float_vars][v].isnull().sum() < 100
        ]
        dropped_vars = [
            v for v in static_ds.data_vars if v not in non_missing_float_vars
        ]
        print(f"Dropping the following non-float vars:\n{dropped_vars}")
        static_ds = static_ds[non_missing_float_vars]
        # --------------------------------------------------

        for var in static_ds.data_vars:
            if var.endswith("one_hot"):
                mean = 0.0
                std = 1.0
            else:
                mean = float(
                    static_ds[var].mean(dim=[dimension_name], skipna=True).values
                )
                std = float(
                    static_ds[var].std(dim=[dimension_name], skipna=True).values
                )

            normalization_values[var]["mean"] = mean
            normalization_values[var]["std"] = std

        static_ds.to_netcdf(self.static_output_folder / "data.nc")
        savepath = self.static_output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

    # TODO: fix this to generalise / work with base.py
    # currently throwing a mypy error
    # Signature of "_process_dynamic" incompatible with supertype "_EngineerBase"
    def _process_dynamic(  # type: ignore
        self,
        test_year: str,
        target_variable: str = "discharge_vol",
        seq_length: int = 365,
        resolution: str = "D",
        expected_length: Optional[int] = 365,
        latlons: bool = False,
    ) -> None:
        """
        Arguments:
        ---------
        test_year: str,
            the date from which testing begins
        target_variable: str = "discharge_vol"
            what is the y_var
        seq_length: int = 365,
            how many timesteps to include in the X data?
        resolution: str = 'D',
            what is the temporal resolution of the data?
        expected_length: Optional[int] = 365,
            how many timesteps to include
        latlons: bool = False
            is this pixel data (latlon?) or 1 dimensional (stations)

        """
        # NOTE: HARDCODED the dataset to open
        ds = xr.open_dataset(self.interim_folder / "camels_preprocessed/data.nc")
        ds = ds.sortby("time")

        # 1. SPLIT TRAIN - TEST
        # NOTE: need to change if not selecting time-ordered train-test splits
        # GET train/test timesteps
        if isinstance(test_year, Iterable):
            test_year = min(test_year)

        min_test_date = pd.to_datetime(f"{test_year}-01-01")
        max_test_date = pd.to_datetime(ds.time.max().values) + Day(1)
        max_train_date = min_test_date - Day(1)
        min_ds_date = pd.to_datetime(ds.time.min().values)

        print(
            f"Generating data.\nTrain: {min_ds_date}-{max_train_date}"
            f"\nTest:  {min_test_date}-{max_test_date} "
        )
        test_ds = ds.sel(time=slice(min_test_date, max_test_date))
        train_ds = ds.sel(time=slice(min_ds_date, max_train_date))

        # check no model leakage train-test
        assert train_ds.time.min().values < test_ds.time.min().values

        # 2. calculate & save the normalisation values
        normalization_values = self._calculate_normalization_values(
            train_ds, latlon=False
        )

        savepath = self.output_folder / "normalizing_dict.pkl"
        with savepath.open("wb") as f:
            pickle.dump(normalization_values, f)

        # 3. stratify X, y for train/test (UN-NORMALISED - normalised in DataLoader)
        for train in [True, False]:
            print(f"\n** Generating {'Training' if train else 'Test'} Data **\n")

            # target_times for train/test
            target_times = train_ds.time.values if train else test_ds.time.values

            for target_time in target_times:
                self.stratify_xy(
                    ds=ds,
                    target_time=target_time,
                    seq_length=seq_length,
                    min_ds_date=min_ds_date,
                    target_var=target_variable,
                    train=train,
                    resolution=resolution,
                )

    # TODO: use this format to generalize the code in `base.py`
    # Argument 3 of "_save" incompatible with supertype "_EngineerBase"
    # from (year: int, month: int) -> (target_time: pd.Timestamp,)
    def _save(  # type: ignore
        self,
        ds_dict: Dict[str, xr.Dataset],
        target_time: pd.Timestamp,
        dataset_type: str,
        resolution: str,
    ) -> None:
        """
        Arguments:
        ---------
        ds_dict: Dict[str, xr.Dataset]
            {'x': xr.Dataset, 'y': xr.Dataset}
        target_time: pd.Timestamp
            the timestamp of the y variable
        dataset_type:
            one of ['test', 'train']
        resolution:
            the resolution of the data used to create directories
        """
        assert dataset_type in ["train", "test"]

        # parent of output folder
        save_folder = self.output_folder / dataset_type
        save_folder.mkdir(exist_ok=True)

        # output folder name
        if resolution == "M":
            output_location = save_folder / f"{target_time.year}_{target_time.month}"
        elif resolution == "D":
            output_location = (
                save_folder
                / f"{target_time.year}_{target_time.month}_{target_time.day}"
            )
        else:
            assert False, "no other resolutions (hrs, mins) etc. have been implemented"
        output_location.mkdir(exist_ok=True)

        for x_or_y, output_ds in ds_dict.items():
            print(f"Saving data to {output_location.as_posix()}/{x_or_y}.nc")
            output_ds.to_netcdf(output_location / f"{x_or_y}.nc")
