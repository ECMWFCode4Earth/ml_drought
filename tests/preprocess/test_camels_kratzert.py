from src.preprocess.camels_kratzert import (
    CalculateNormalizationParams,
    # reshape_data,
    CamelsCSV,
    get_basins,
    RunoffEngineer,
    CamelsDataLoader,
)
from src.preprocess import CAMELSGBPreprocessor
from ..utils import _copy_runoff_data_to_tmp_path
import pandas as pd
import numpy as np
import h5py
import pytest
from torch.utils.data import DataLoader


class TestCamelsCSV:
    def test(self, tmp_path):
        _copy_runoff_data_to_tmp_path(tmp_path)

        processsor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processsor.preprocess()

        # SETTINGS
        train_dates = [2000, 2010]
        target_var = "discharge_spec"
        x_variables = ["precipitation", "peti"]
        static_variables = ["pet_mean", "aridity", "p_seasonality"]
        seq_length = 10
        with_static = True
        is_train = True
        concat_static = False

        # DERIVED Values
        n_times = len(
            pd.date_range(
                f"{train_dates[0]}-01-01", f"{train_dates[-1]}-12-31", freq="D"
            )
        )
        n_features = len(x_variables)
        n_stations = 2
        n_static_features = len(static_variables)

        normalization_dict = CalculateNormalizationParams(
            data_dir=tmp_path,
            train_dates=train_dates,
            target_var=target_var,
            x_variables=x_variables,
            static_variables=static_variables,
        ).normalization_dict

        assert len([stn for stn in get_basins(tmp_path)]) == n_stations

        for basin in get_basins(tmp_path):
            dataset = CamelsCSV(
                data_dir=tmp_path,
                basin=basin,
                train_dates=train_dates,
                normalization_dict=normalization_dict,
                is_train=is_train,
                target_var=target_var,
                x_variables=x_variables,
                static_variables=static_variables,
                seq_length=seq_length,
                with_static=with_static,
                concat_static=concat_static,
            )
            x = dataset.x
            y = dataset.y
            static = dataset.attributes
            scaler = dataset.normalization_dict

            assert x.shape == (n_times, seq_length, n_features)
            assert y.shape == (n_times, 1)
            assert static.shape == (1, n_static_features)

            expected = [
                "static_means",
                "static_stds",
                "target_mean",
                "target_std",
                "dynamic_stds",
                "dynamic_means",
                "x_variables",
                "target_var",
                "static_variables",
            ]
            assert all(
                np.isin([k for k in scaler.keys()], expected)
            ), f"Expected: {expected} Got: {[k for k in scaler.keys()]}"


class TestRunoffEngineer:
    def _initialise_data(self, tmp_path):
        _copy_runoff_data_to_tmp_path(tmp_path)
        processsor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processsor.preprocess()

    def test(self, tmp_path):
        self._initialise_data(tmp_path)

        # SETTINGS
        train_dates = [2000, 2010]
        target_var = "discharge_spec"
        x_variables = ["precipitation", "peti"]
        static_variables = ["pet_mean", "aridity", "p_seasonality"]
        seq_length = 10
        with_static = True
        concat_static = False
        basins = get_basins(tmp_path)
        with_basin_str = True

        # INITIALIZE
        engineer = RunoffEngineer(
            data_dir=tmp_path,
            basins=basins,
            train_dates=train_dates,
            with_basin_str=with_basin_str,
            target_var=target_var,
            x_variables=x_variables,
            static_variables=static_variables,
            ignore_static_vars=None,
            seq_length=seq_length,
            with_static=with_static,
            concat_static=concat_static,
        )

        engineer.create_training_data()
        h5_file = engineer.out_file

        assert h5_file.exists()
        with h5py.File(h5_file, "r") as f:
            x = f["input_data"][:]
            y = f["target_data"][:]
            str_arr = f["sample_2_basin"][:]
            str_arr = [x.decode("ascii") for x in str_arr]
            q_stds = f["q_stds"][:]

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(str_arr, list)
        assert isinstance(q_stds, np.ndarray)

        assert len(np.unique(q_stds)) == 2
        assert len(np.unique(str_arr)) == 2
        assert x[0].shape == (seq_length, len(x_variables))
        assert len(x) == len(y)


class TestH5Train:
    def _initialise_data(self, tmp_path):
        _copy_runoff_data_to_tmp_path(tmp_path)
        processsor = CAMELSGBPreprocessor(tmp_path, open_shapefile=False)
        processsor.preprocess()

    @pytest.mark.parametrize(
        "with_static,concat_static", [(True, False), (True, True), (False, False)]
    )
    def test(self, tmp_path, with_static, concat_static):
        self._initialise_data(tmp_path)

        # SETTINGS
        with_basin_str = True
        train_dates = [2000, 2002]
        target_var = "discharge_spec"
        x_variables = ["precipitation", "peti"]
        static_variables = ["pet_mean", "aridity", "p_seasonality"]
        seq_length = 5
        with_static = with_static
        concat_static = concat_static
        basins = get_basins(tmp_path)

        # EXPECTED
        out_file = tmp_path / "features/features.h5"
        static_data_path = tmp_path / "interim/static/data.nc"
        n_variables = len(x_variables)
        n_static_features = len(static_variables)

        # INITIALIZE
        engineer = RunoffEngineer(
            data_dir=tmp_path,
            basins=basins,
            train_dates=train_dates,
            with_basin_str=with_basin_str,
            target_var=target_var,
            x_variables=x_variables,
            static_variables=static_variables,
            ignore_static_vars=None,
            seq_length=seq_length,
            with_static=with_static,
            concat_static=concat_static,
        )

        engineer.create_training_data()

        data = CamelsDataLoader(
            data_dir=tmp_path,
            basins=basins,
            concat_static=concat_static,
            cache=True,
            with_static=with_static,
            train_dates=train_dates,
        )

        iterate = [d for d in data]
        assert (
            len(iterate) == ((3 * 365) + 1) * 2
        ), "Should be 3 years (365 days) + 1 leap days, for two basins"

        assert data.h5_file == out_file
        assert data.static_data_path == static_data_path

        for index in [0, -1]:
            x = data[index][0]
            q_stds = data[index][-2]
            y = data[index][-1]

            assert q_stds.numpy().shape == (1,)
            assert y.numpy().shape == (1,)

            if (with_static) & (not concat_static):
                static = data[index][1]
                assert len(data[index]) == 4
                assert x.shape == (seq_length, n_variables)
                assert static.shape == (1, n_static_features)

            if (with_static) & (concat_static):
                assert len(data[index]) == 3
                assert x.shape == (seq_length, n_variables + n_static_features)

            if not with_static:
                assert len(data[index]) == 3
                assert x.shape == (seq_length, n_variables)

        assert data.static_variables == static_variables
        assert data.target_var == target_var

        loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=1)
