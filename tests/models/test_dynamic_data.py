"""

[d.name for d in (tmp_path / 'features').iterdir()]
"""

import numpy as np
import pandas as pd
from src.models.dynamic_data import DynamicDataLoader
import xarray as xr
from ..utils import _create_runoff_features_dir
from src.utils import minus_timesteps
from src.models.data import ModelArrays


class TestUtils:
    def test_minus_timesteps(self):
        time = pd.to_datetime("2000-01-31")
        new_time = minus_timesteps(time, 1, freq="M")
        assert all([new_time.year == 1999, new_time.month == 12, new_time.day == 31])

        new_time = minus_timesteps(time, 1, freq="D")
        assert all([new_time.year == 2000, new_time.month == 1, new_time.day == 30])

        new_time = minus_timesteps(time, 0, freq="D")
        assert all([new_time.year == 2000, new_time.month == 1, new_time.day == 31])

        new_time = minus_timesteps(time, 10, freq="D")
        assert all([new_time.year == 2000, new_time.month == 1, new_time.day == 21])


class TestDynamicDataLoader:
    def test_dynamic_dataloader(self, tmp_path):
        ds, static = _create_runoff_features_dir(tmp_path)
        # initialise the data as an OUTPUT of the engineers
        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["discharge"]
        target_var = "discharge"
        seq_length = 5
        test_years = [2001]
        forecast_horizon = 0
        batch_file_size = 1

        # initialize the DATALOADER object
        # dl = DynamicDataLoader(target_var=target_var,test_years=np.arange(2011, 2016),data_path=tmp_path,seq_length=seq_length,static_ignore_vars=static_ignore_vars,dynamic_ignore_vars=dynamic_ignore_vars,)
        dl = DynamicDataLoader(
            target_var=target_var,
            test_years=test_years,
            data_path=tmp_path,
            seq_length=seq_length,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            forecast_horizon=forecast_horizon,
            batch_file_size=batch_file_size,
        )

        ## check that dealing with spatial variable correctly
        assert "station_id" in dl.reducing_dims

        ## check the normalisations
        assert all(
            np.isin(
                ["gauge_elev", "q_mean", "area"],
                [k for k in dl.static_normalizing_dict.keys()],
            )
        )
        assert all(
            np.isin(
                ["precip", "discharge", "pet"], [k for k in dl.normalizing_dict.keys()]
            )
        )

        ## check the correct train/test times chosen
        assert (
            len(dl.valid_test_times) == 365
        ), f"1 year worth of data chosen for TESTING. Got: {len(dl.valid_test_times)} days"
        assert (
            len(dl.valid_train_times) == 731
        ), f"2 years worth of data chosen for TRAINING. Got {len(dl.valid_train_times)} days"

        ## check that not leaking target_var information
        assert "target_var_original" in dl.ignore_vars

        ## check that data correctly loaded
        assert isinstance(dl.static_ds, xr.Dataset)
        assert all(
            dl.static_ds == static
        ), "Expect data to be the same as created in `_create_runoff_features_dir`"
        assert isinstance(dl.dynamic_ds, xr.Dataset)
        assert all(
            dl.dynamic_ds == ds
        ), "Expect data to be the same as created in `_create_runoff_features_dir`"
        assert (
            len(dl) == 731 // batch_file_size
        ), "Expect 2 years of daily data. 730 + 1 (leap year in 2000). Should be batched."

    def test_get_sample_from_dynamic_data(self, tmp_path):
        ds, static = _create_runoff_features_dir(tmp_path)
        # initialise the data as an OUTPUT of the dynamic engineer
        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["discharge"]
        target_var = "discharge"
        seq_length = 5
        test_years = [2001]
        forecast_horizon = 0
        batch_file_size = 1

        # initialize the object
        dl = DynamicDataLoader(
            target_var=target_var,
            test_years=test_years,
            data_path=tmp_path,
            seq_length=seq_length,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            forecast_horizon=forecast_horizon,
            batch_file_size=batch_file_size,
        )

        #  ------------------------------
        #  Test the iterators
        #  ------------------------------
        resolution = "D"
        iterator = dl.__iter__()
        X, y = dl.__iter__().__next__()

        # which timestep? (because they're randomly shuffled ...)
        station_ids, times = np.where(ds["discharge"].values == y)
        target_time = pd.to_datetime(ds.isel(time=times[0]).time.values)
        assert target_time.year in [2000, 1999], "Expect to iterate through TRAIN times"

        # test get sample from data
        (X_dataset, y_dataset), tt2 = iterator.get_sample_from_dynamic_data(
            target_var=target_var,
            target_time=target_time,
            seq_length=seq_length,
            dynamic_ignore_vars=dynamic_ignore_vars,
            forecast_horizon=forecast_horizon,
            train=True,
            resolution=resolution,
        )
        max_X_date = minus_timesteps(target_time, forecast_horizon, resolution)
        min_X_date = minus_timesteps(
            target_time,
            seq_length - 1 if forecast_horizon == 0 else seq_length,
            resolution,
        )

        assert tt2 == target_time

        # check the X data
        assert isinstance(X_dataset, xr.Dataset)
        assert not all(
            np.isin(dynamic_ignore_vars, list(X_dataset.data_vars))
        ), f"Expected not to find {dynamic_ignore_vars} in {list(X_dataset.data_vars)}"

        # check the y data
        assert isinstance(y_dataset, xr.Dataset)
        assert target_var in list(y_dataset.data_vars)

        # check the forecast horizon info
        if forecast_horizon == 0:
            assert max_X_date == target_time, "Forecast horizon is zero!"

        assert isinstance(min_X_date, pd.Timestamp)
        assert max_X_date - min_X_date == pd.Timedelta(
            f"{seq_length - 1 if forecast_horizon == 0 else seq_length} days"
        )

    def test_ds_sample_to_np(self, tmp_path):
        ds, static = _create_runoff_features_dir(tmp_path)
        # initialise the data as an OUTPUT of the engineers
        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["discharge"]
        target_var = "discharge"
        seq_length = 5
        test_years = [2001]
        forecast_horizon = 0
        batch_file_size = 1

        # initialize the object
        dl = DynamicDataLoader(
            target_var=target_var,
            test_years=test_years,
            data_path=tmp_path,
            seq_length=seq_length,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            forecast_horizon=forecast_horizon,
            batch_file_size=batch_file_size,
        )

        #  ------------------------------
        #  Test the created ModelArrays
        #  ------------------------------
        resolution = "D"
        iterator = dl.__iter__()
        X, y = dl.__iter__().__next__()

        # which timestep? (because they're randomly shuffled ...)
        station_ids, times = np.where(ds["discharge"].values == y)
        target_time = pd.to_datetime(ds.isel(time=times[0]).time.values)
        assert target_time.year in [2000, 1999], "Expect to iterate through TRAIN times"

        # GET THE SAMPLE
        xy_sample, tt2 = iterator.get_sample_from_dynamic_data(
            target_var=target_var,
            target_time=target_time,
            seq_length=seq_length,
            dynamic_ignore_vars=dynamic_ignore_vars,
            forecast_horizon=forecast_horizon,
            train=True,
            resolution=resolution,
        )

        # TURN INTO NUMPY ARRAYS
        arrays = iterator.ds_sample_to_np(
            xy_sample=xy_sample,  #  (Xdataset, ydataset)
            target_time=target_time,  #  time of y
            clear_nans=iterator.clear_nans,  # True
            to_tensor=iterator.to_tensor,  # False
            reducing_dims=iterator.reducing_dims,  #  station_id
            target_var_std=iterator.target_var_std,  # (10,) array
        )

        assert isinstance(arrays, ModelArrays)

        # unnormalize the data
        unnormalized_data = []
        for ix, var in enumerate(arrays.x_vars):
            data = arrays.x.historical[:, :, ix]
            unnormalized_data.append(
                (data * iterator.normalizing_dict[var]["std"])
                + iterator.normalizing_dict[var]["mean"]
            )
        raw_data = np.stack(unnormalized_data, axis=-1)

        # final index is target_time!
        expected_precip = ds.sel(time=target_time)["precip"].values
        expected_pet = ds.sel(time=target_time)["pet"].values
        assert all(
            expected_precip == raw_data[:, -1, 0]
        ), "Expect the final timestep to be the target time PRECIP"
        assert all(
            expected_pet == raw_data[:, -1, 1]
        ), "Expect the final timestep to be the target time PET"

    def test_returned_data(self, tmp_path):
        ds, static = _create_runoff_features_dir(tmp_path)
        # initialise the data as an OUTPUT of the engineers
        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["discharge"]
        target_var = "discharge"
        seq_length = 5
        test_years = [2001]
        forecast_horizon = 0
        batch_file_size = 1

        # initialize the object
        dl = DynamicDataLoader(
            target_var=target_var,
            test_years=test_years,
            data_path=tmp_path,
            seq_length=seq_length,
            static_ignore_vars=static_ignore_vars,
            dynamic_ignore_vars=dynamic_ignore_vars,
            forecast_horizon=forecast_horizon,
            batch_file_size=batch_file_size,
        )

        #  ------------------------------
        #  Test the iterators
        #  ------------------------------
        # resolution = "D"
        # iterator = dl.__iter__()
        X, y = dl.__iter__().__next__()

        # which timestep? (because they're randomly shuffled ...)
        station_ids, times = np.where(ds["discharge"].values == y)
        target_time = pd.to_datetime(ds.isel(time=times[0]).time.values)
        assert target_time.year in [2000, 1999], "Expect to iterate through TRAIN times"

        # check the shape of the returned data
        # idx_to_input = {
        #     0: "historical",
        #     1: "pred_month",
        #     2: "latlons",
        #     3: "current",
        #     4: "yearly_aggs",
        #     5: "static",
        #     6: "prev_y_var",
        #     7: "target_var_std",
        # }
        assert isinstance(X, tuple)
        assert len(X) == 8, "Should be 8 different data inputs for X"

        # 0 historical
        assert isinstance(X[0], np.ndarray)
        assert (
            X[0].shape[1] == seq_length
        ), f"SEQ LENGTH: Should be (#non-nan stations, seq_length, n_predictors). seq_length: {X[0].shape[-1]}"
        assert (
            X[0].shape[-1] == 2
        ), f"N PREDICTORS: Should be (#non-nan stations, seq_length, n_predictors). NPredictors: {X[0].shape[-1]}"

        ## check that the same target_time as the data_time
        precip_tminus1 = X[0][:, -1, 0]
        precip_mean = dl.normalizing_dict["precip"]["mean"]
        precip_std = dl.normalizing_dict["precip"]["std"]
        precip = (precip_tminus1 * precip_std) + precip_mean
        np.where(ds["pet"].values == precip)

        # 1 pred_month
        assert (
            X[1][0] == target_time.month
        ), f"Wrong month! Expected: {target_time.month} Got: {X[1][0]}"

        # 5 static
        assert X[5].shape == (10, 2), "N Stations, N Variables"

        # 7 target_var_std (required for the NSE loss function)
        assert (
            len(X[7]) == 10
        ), "Should have one STD for each station calculated from the training data"
        expected_stds = (
            ds.discharge.sel(time=slice("1999", "2000")).std(dim="time").values
        )
        assert all(
            X[7] == expected_stds
        ), "Expect std to be calculated from the TRAINING data only"

        # 2 latlons
        assert X[2] is None
        # 3 current
        assert X[3] is None
        # 4 yearly_aggs
        assert X[4] is None
        # 6 prev_y_var
        assert X[6] is None
