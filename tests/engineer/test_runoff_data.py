import pytest
import pickle
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd

from src.engineer import _OneMonthForecastEngineer as OneMonthForecastEngineer

from ..utils import _make_dataset
from .test_base import _setup

class TestRunoffEngineer:
    @staticmethod
    def _create_input_data() -> xr.Dataset:
        # create dims / coords
        times = pd.date_range('2000-01', '2000-03', freq='D')
        station_ids = np.arange(0, 10)
        dims = ['station_id', 'time']
        coords = {'station_id': station_ids, 'time': times}
        shape = (len(station_ids), len(times))

        # create random data
        precip = np.random.random(shape)
        discharge = np.random.random(shape)
        pet = np.random.random(shape)
        datasets = [precip, discharge, pet]
        variables = ['precip', 'discharge', 'pet']

        ds = xr.Dataset(
            {
                variable: (dims, dataset)
                for variable, dataset in zip(variables, datasets)
            }, coords=coords
        )
        return ds

    def _save_runoff_input_data(self, tmp_path):
        ds = _make_runoff_data()

        if not (tmp_path / 'interim/runoff_preprocessed').exists:
            (tmp_path / 'interim/runoff_preprocessed').mkdir(
                exist_ok=True, parents=True
            )
        # save to netcdf
        ds.to_netcdf(
            tmp_path / 'interim/runoff_preprocessed/data.nc'
        )

    def test_train_test_split(self, tmp_path):
        self._save_runoff_input_data()
        engineer = Engineer(tmp_path)