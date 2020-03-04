import numpy as np
import xarray as xr
import pickle
import pytest

from sklearn import linear_model
from src.models import LinearRegression
from src.models.data import DataLoader

from ..utils import _make_dataset


class TestLinearRegression:
    def test_save(self, tmp_path, monkeypatch):

        coef_array = np.array([1, 1, 1, 1, 1])
        intercept_array = np.array([2])

        def mocktrain(self):
            class MockModel:
                @property
                def coef_(self):
                    return coef_array

                @property
                def intercept_(self):
                    return intercept_array

            self.model = MockModel()

        monkeypatch.setattr(LinearRegression, "train", mocktrain)

        model = LinearRegression(
            tmp_path, experiment="one_month_forecast", normalize_y=False
        )
        model.train()
        model.save_model()

        assert (
            tmp_path / "models/one_month_forecast/linear_regression/model.pkl"
        ).exists(), f"Model not saved!"

        with (tmp_path / "models/one_month_forecast/linear_regression/model.pkl").open(
            "rb"
        ) as f:
            model_dict = pickle.load(f)
        assert np.array_equal(
            coef_array, model_dict["model"]["coef"]
        ), "Different coef array saved!"
        assert np.array_equal(
            intercept_array, model_dict["model"]["intercept"]
        ), "Different intercept array saved!"
        assert (
            model_dict["experiment"] == "one_month_forecast"
        ), "Different experiment saved!"

    @pytest.mark.parametrize(
        "use_pred_months,experiment,monthly_agg,predict_delta",
        [
            (True, "one_month_forecast", True, True),
            (True, "nowcast", False, True),
            (False, "one_month_forecast", False, True),
            (False, "nowcast", True, True),
            (True, "one_month_forecast", True, False),
            (True, "nowcast", False, False),
            (False, "one_month_forecast", False, False),
            (False, "nowcast", True, False),
        ],
    )
    def test_train(
        self, tmp_path, capsys, use_pred_months, experiment, monthly_agg, predict_delta
    ):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        y = x.isel(time=[-1])

        x_add1, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="precip")
        x_add1 = x_add1 * 2
        x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="temp")
        x_add2 = x_add2 * 3
        x = xr.merge([x, x_add1, x_add2])

        norm_dict = {
            "VHI": {"mean": 0, "std": 1},
            "precip": {"mean": 0, "std": 1},
            "temp": {"mean": 0, "std": 1},
        }

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}

        test_features = tmp_path / f"features/{experiment}/train/2001_12"
        test_features.mkdir(parents=True)
        pred_features = tmp_path / f"features/{experiment}/test/2001_12"
        pred_features.mkdir(parents=True)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)

        with (tmp_path / f"features/{experiment}/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(norm_dict, f)

        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

        x.to_netcdf(test_features / "x.nc")
        x.to_netcdf(pred_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")
        y.to_netcdf(pred_features / "y.nc")
        x_static.to_netcdf(static_features / "data.nc")

        model = LinearRegression(
            tmp_path,
            include_pred_month=use_pred_months,
            experiment=experiment,
            include_monthly_aggs=monthly_agg,
            predict_delta=predict_delta,
            normalize_y=True,
        )
        model.train()

        captured = capsys.readouterr()
        expected_stdout = "Epoch 1, train RMSE: "
        assert (
            expected_stdout in captured.out
        ), f"Expected stdout to be {expected_stdout}, got {captured.out}"

        assert (
            type(model.model) == linear_model.SGDRegressor
        ), f"Model attribute not a linear regression!"

        if experiment == "nowcast":
            coef_size = (3 * 35) + 2
        elif experiment == "one_month_forecast":
            coef_size = 3 * 36
        if monthly_agg:
            # doubled including the mean, tripled including the std
            coef_size *= 2
        if use_pred_months:
            coef_size += 12

        coef_size += 3  # for the yearly aggs
        coef_size += 1  # for the static variable
        coef_size += 1  # for the prev_y_var

        assert model.model.coef_.size == coef_size, f"Got unexpected coef size"

        test_arrays_dict, preds_dict = model.predict()
        assert (
            test_arrays_dict["2001_12"]["y"].size == preds_dict["2001_12"].shape[0]
        ), "Expected length of test arrays to be the same as the predictions"

        # test saving the model outputs
        model.evaluate(save_preds=True)

        save_path = model.data_path / "models" / experiment / "linear_regression"
        assert (save_path / "preds_2001_12.nc").exists()
        assert (save_path / "results.json").exists()

        pred_ds = xr.open_dataset(save_path / "preds_2001_12.nc")
        assert np.isin(["lat", "lon", "time"], [c for c in pred_ds.coords]).all()
        assert y.time == pred_ds.time

    def test_big_mean(self, tmp_path, monkeypatch):
        def mockiter(self):
            class MockIterator:
                def __init__(self):
                    self.idx = 0
                    self.max_idx = 10

                def __iter__(self):
                    return self

                def __next__(self):
                    if self.idx < self.max_idx:
                        # batch_size = 10, timesteps = 2, num_features = 1
                        self.idx += 1
                        return (
                            (
                                np.ones((10, 2, 1)),
                                np.ones((10,), dtype=np.int8),
                                np.ones((10, 2)),
                                np.ones((10, 2)),
                                np.ones((10, 2)),
                                np.ones((10, 2)),
                                np.ones((10, 1)),
                            ),
                            None,
                        )
                    else:
                        raise StopIteration()

            return MockIterator()

        def do_nothing(
            self,
            data_path,
            batch_file_size,
            mode,
            shuffle_data,
            clear_nans,
            normalize,
            experiment,
            mask,
            pred_months,
            to_tensor,
            surrounding_pixels,
            ignore_vars,
            monthly_aggs,
            static,
            device,
            predict_delta,
            spatial_mask,
            normalize_y,
            incl_yearly_aggs,
        ):

            pass

        monkeypatch.setattr(DataLoader, "__iter__", mockiter)
        monkeypatch.setattr(DataLoader, "__init__", do_nothing)

        model = LinearRegression(tmp_path, normalize_y=False)
        calculated_mean = model._calculate_big_mean()

        # 1 for the 2 features and for the first month, 0 for the rest
        expected_mean = np.array(
            [
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        )

        # np.isclose because of rounding
        assert np.isclose(calculated_mean, expected_mean).all()
