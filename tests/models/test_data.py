import torch
import numpy as np
import pytest
import xarray as xr
import pandas as pd

from src.models.data import DataLoader, _BaseIter, TrainData

from ..utils import _make_dataset


class TestBaseIter:
    def test_mask(self, tmp_path):

        for i in range(5):
            (tmp_path / f"features/one_month_forecast/train/{i}").mkdir(parents=True)
            (tmp_path / f"features/one_month_forecast/train/{i}/x.nc").touch()
            (tmp_path / f"features/one_month_forecast/train/{i}/y.nc").touch()

        mask_train = [True, True, False, True, False]
        mask_val = [False, False, True, False, True]

        train_paths = DataLoader._load_datasets(
            tmp_path,
            mode="train",
            experiment="one_month_forecast",
            shuffle_data=True,
            mask=mask_train,
        )
        val_paths = DataLoader._load_datasets(
            tmp_path,
            mode="train",
            experiment="one_month_forecast",
            shuffle_data=True,
            mask=mask_val,
        )
        assert (
            len(set(train_paths).intersection(set(val_paths))) == 0
        ), f"Got the same file in both train and val set!"
        assert len(train_paths) + len(val_paths) == 5, f"Not all files loaded!"

    def test_pred_months(self, tmp_path):
        for i in range(1, 13):
            (tmp_path / f"features/one_month_forecast/train/2018_{i}").mkdir(
                parents=True
            )
            (tmp_path / f"features/one_month_forecast/train/2018_{i}/x.nc").touch()
            (tmp_path / f"features/one_month_forecast/train/2018_{i}/y.nc").touch()

        pred_months = [4, 5, 6]

        train_paths = DataLoader._load_datasets(
            tmp_path,
            mode="train",
            shuffle_data=True,
            pred_months=pred_months,
            experiment="one_month_forecast",
        )

        assert len(train_paths) == len(
            pred_months
        ), f"Got {len(train_paths)} filepaths back, expected {len(pred_months)}"

        for return_file in train_paths:
            subfolder = return_file.parts[-1]
            month = int(str(subfolder)[5:])
            assert (
                month in pred_months
            ), f"{month} not in {pred_months}, got {return_file}"

    @pytest.mark.parametrize(
        "normalize,to_tensor,experiment,surrounding_pixels,predict_delta",
        [
            (True, True, "one_month_forecast", 1, True),
            (True, False, "one_month_forecast", None, True),
            (False, True, "one_month_forecast", 1, True),
            (False, False, "one_month_forecast", None, True),
            (True, True, "nowcast", 1, True),
            (True, False, "nowcast", None, True),
            (False, True, "nowcast", 1, True),
            (False, False, "nowcast", None, True),
            (True, True, "one_month_forecast", 1, False),
            (True, False, "one_month_forecast", None, False),
            (False, True, "one_month_forecast", 1, False),
            (False, False, "one_month_forecast", None, False),
            (True, True, "nowcast", 1, False),
            (True, False, "nowcast", None, False),
            (False, True, "nowcast", 1, False),
            (False, False, "nowcast", None, False),
        ],
    )
    def test_ds_to_np(
        self,
        tmp_path,
        normalize,
        to_tensor,
        experiment,
        surrounding_pixels,
        predict_delta,
    ):
        x_pred, _, _ = _make_dataset(size=(5, 5), const=True)
        x_coeff1, _, _ = _make_dataset(size=(5, 5), variable_name="precip")
        x_coeff2, _, _ = _make_dataset(size=(5, 5), variable_name="soil_moisture")
        x_coeff3, _, _ = _make_dataset(size=(5, 5), variable_name="temp")

        x = xr.merge([x_pred, x_coeff1, x_coeff2, x_coeff3])
        y = x_pred.isel(time=[0])

        data_dir = tmp_path / experiment / "1980_1"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)

        x.to_netcdf(data_dir / "x.nc")
        y.to_netcdf(data_dir / "y.nc")

        norm_dict = {}
        for var in x.data_vars:
            norm_dict[var] = {
                "mean": float(
                    x[var].mean(dim=["lat", "lon", "time"], skipna=True).values
                ),
                # we clip the std because since constant=True, the std=0 for VHI,
                # giving NaNs which mess the tests up
                "std": float(
                    np.clip(
                        a=x[var].std(dim=["lat", "lon", "time"], skipna=True).values,
                        a_min=1,
                        a_max=None,
                    )
                ),
            }

        # build static data
        static1 = x.mean(dim="time").rename({v: f"{v}_pixel_mean" for v in x.data_vars})
        ones = xr.ones_like(x.mean(dim="time"))[[v for v in x.data_vars][0]]
        static2 = x.mean(dim=["lat", "lon", "time"]).rename(
            {v: f"{v}_global_mean" for v in x.data_vars}
        )
        static2 = static2 * ones
        static_ds = xr.auto_combine([static1, static2])

        class MockLoader:
            def __init__(self):
                self.batch_file_size = None
                self.mode = None
                self.shuffle = None
                self.clear_nans = None
                self.data_files = []
                self.normalizing_dict = norm_dict if normalize else None
                self.to_tensor = None
                self.experiment = experiment
                self.surrounding_pixels = surrounding_pixels
                self.predict_delta = predict_delta
                self.ignore_vars = ["precip"]
                self.monthly_aggs = False
                self.device = torch.device("cpu")
                self.incl_yearly_aggs = False
                self.static = static_ds
                self.spatial_mask = None
                self.static_normalizing_dict = None
                self.normalize_y = normalize

        base_iterator = _BaseIter(MockLoader())

        arrays = base_iterator.ds_folder_to_np(data_dir, to_tensor=to_tensor)
        x_train_data, y_np, latlons = (arrays.x, arrays.y, arrays.latlons)

        # ----------------------
        # Test the static data
        # ----------------------
        # check first 3 features are CONSTANT (global means)
        assert all(
            [
                all(arrays.x.static[:, i][1:] == arrays.x.static[:, i][:-1])
                for i in range(4)
            ]
        )
        if not predict_delta:
            # check second 3 features vary (pixel means)
            assert all(
                [
                    all(arrays.x.static[:, i][1:] != arrays.x.static[:, i][:-1])
                    for i in range(4, 6)
                ]
            ), (
                f"static data: \n[,4]\n: {arrays.x.static[:, 4][1:]}\n[,5]"
                f"\n: {arrays.x.static[:, 5][1:]}"
            )

        n_samples = 25 if surrounding_pixels is None else 9
        assert (
            arrays.x.static.shape[0] == n_samples
        ), f"Expect {n_samples} samples because ..."

        assert (
            arrays.x.static.shape[-1] == 6
        ), "Expect 6 static features because ignore 'precip' variables in the static data"

        # ----------------------
        # Test the TrainData
        # ----------------------
        assert isinstance(x_train_data, TrainData)
        if not to_tensor:
            assert isinstance(y_np, np.ndarray)

        expected_features = 3 if surrounding_pixels is None else 3 * 9
        assert x_train_data.historical.shape[-1] == expected_features, (
            f"There should be 4 historical variables "
            f"(the final dimension): {x_train_data.historical.shape}"
        )

        if experiment == "nowcast":
            expected_shape = (25, 2) if surrounding_pixels is None else (9, 2 * 9)
            assert x_train_data.current.shape == expected_shape, (
                f"Expecting multiple vars in the current timestep. "
                f"Expect: (25, 5) Got: {x_train_data.current.shape}"
            )

        expected_latlons = 25 if surrounding_pixels is None else 9

        assert latlons.shape == (expected_latlons, 2), (
            "The shape of "
            "latlons should not change"
            f"Got: {latlons.shape}. Expecting: (25, 2)"
        )
        assert x_train_data.latlons.shape == (expected_latlons, 2), (
            "The shape of "
            "latlons should not change"
            f"Got: {latlons.shape}. Expecting: (25, 2)"
        )

        if normalize and (experiment == "nowcast") and (not to_tensor):
            assert x_train_data.current.max() < 6, (
                f"The current data should be"
                f" normalized. Currently: {x_train_data.current.flatten()}"
            )

        if to_tensor:
            assert (type(x_train_data.historical) == torch.Tensor) and (
                type(y_np) == torch.Tensor
            )
        else:
            assert (type(x_train_data.historical) == np.ndarray) and (
                type(y_np) == np.ndarray
            )

        if (not normalize) and (experiment == "nowcast") and (not to_tensor):
            assert x_train_data.historical.shape[0] == x_train_data.current.shape[0], (
                "The 0th dimension (latlons) should be equal in the "
                f"historical ({x_train_data.historical.shape[0]}) and "
                f"current ({x_train_data.current.shape[0]}) arrays."
            )

            expected = (
                x[["soil_moisture", "temp"]]
                .sel(time=y.time)
                .stack(dims=["lat", "lon"])
                .to_array()
                .values.T[:, 0, :]
            )
            got = x_train_data.current

            if surrounding_pixels is None:
                assert expected.shape == got.shape, (
                    "should have stacked latlon"
                    " vars as the first dimension in the current array."
                )

                assert (expected == got).all(), (
                    ""
                    "Expected to find the target timesetep of `precip` values"
                    "(the non-target variable for the target timestep: "
                    f"({pd.to_datetime(y.time.values).strftime('%Y-%m-%d')[0]})."
                    f"Expected: {expected[:5]}. \nGot: {got[:5]}"
                )

        for idx in range(latlons.shape[0]):
            lat, lon = latlons[idx, 0], latlons[idx, 1]
            for time in range(x_train_data.historical.shape[1]):
                target = x.isel(time=time).sel(lat=lat).sel(lon=lon).VHI.values

                if (not normalize) and (not to_tensor):
                    assert target == x_train_data.historical[idx, time, 0], (
                        "Got different x values for time idx:"
                        f"{time}, lat: {lat}, lon: {lon} Expected {target}, "
                        f"got {x_train_data.historical[idx, time, 0]}"
                    )

        if (
            (not normalize)
            and (experiment == "nowcast")
            and (surrounding_pixels is None)
        ):
            # test that we are getting the right `current` data
            relevant_features = ["soil_moisture", "temp"]
            target_time = y.time
            expected = (
                x[relevant_features]  # all vars except target_var and the ignored var
                .sel(time=target_time)  # select the target_time
                .stack(
                    dims=["lat", "lon"]
                )  # stack lat,lon so shape = (lat*lon, time, dims)
                .to_array()
                .values[:, 0, :]
                .T  # extract numpy array, transpose and drop dim
            )

            assert np.all(x_train_data.current == expected), (
                f"Expected to " "find the target_time data for the non target variables"
            )

        if x_train_data.yearly_aggs is not None:
            # n_variables should be 3 because `ignoring` precip
            assert x_train_data.yearly_aggs.shape[1] == 3

        if (not normalize) and (not to_tensor):
            mean_temp = x_coeff3.temp.mean(dim=["time", "lat", "lon"]).values
            if x_train_data.yearly_aggs is not None:
                assert (mean_temp == x_train_data.yearly_aggs).any()

        if predict_delta:
            assert (
                y_np == 0
            ).all(), "The derivatives should be 0 for a constant input."
            assert (
                base_iterator.predict_delta
            ), "should have set model_ derivative to True"

    @pytest.mark.parametrize(
        "surrounding_pixels,monthly_agg",
        [(1, True), (1, False), (None, True), (None, False)],
    )
    def test_additional_dims_pixels(self, surrounding_pixels, monthly_agg):
        x, _, _ = _make_dataset(size=(10, 10))
        org_vars = list(x.data_vars)

        x_with_more = _BaseIter._add_extra_dims(x, surrounding_pixels, monthly_agg)
        shifted_agg_vars = x_with_more.data_vars

        for data_var in org_vars:
            if surrounding_pixels is not None:
                for lat in [-1, 0, 1]:
                    for lon in [-1, 0, 1]:
                        if lat == lon == 0:
                            assert (
                                f"lat_{lat}_lon_{lon}_{data_var}"
                                not in shifted_agg_vars
                            ), (
                                f"lat_{lat}_lon_{lon}_{data_var} should not "
                                f"be in the shifted variables"
                            )
                        else:
                            shifted_var_name = f"lat_{lat}_lon_{lon}_{data_var}"
                            assert (
                                shifted_var_name in shifted_agg_vars
                            ), f"{shifted_var_name} is not in the shifted variables"

                            org = x_with_more.VHI.isel(time=0, lon=5, lat=5).values
                            shifted = (
                                x_with_more[shifted_var_name]
                                .isel(time=0, lon=5 + lon, lat=5 + lat)
                                .values
                            )
                            assert org == shifted, f"Shifted values don't match!"

            if monthly_agg:
                mean_var_name = f"spatial_mean_{data_var}"
                assert (
                    mean_var_name in shifted_agg_vars
                ), f"{mean_var_name} is not in the shifted variables"

                actual_mean = x[data_var].isel(time=0).mean(dim=["lat", "lon"]).values
                output_mean = (
                    x[f"spatial_mean_{data_var}"].isel(time=0, lon=0, lat=0).values
                )

                assert actual_mean == output_mean, f"Mean values don't match!"
