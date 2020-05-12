import numpy as np
import xarray as xr
import pickle
import pytest

from src.models import GBDT

from ..utils import _make_dataset


class TestGBDT:
    @pytest.mark.xfail(reason="XGBoost not part of the test environment")
    @pytest.mark.parametrize(
        "use_pred_months,experiment,monthly_agg",
        [
            (True, "one_month_forecast", True),
            (True, "nowcast", False),
            (False, "one_month_forecast", False),
            (False, "nowcast", True),
        ],
    )
    def test_train(self, tmp_path, use_pred_months, experiment, monthly_agg):

        import xgboost as xgb

        x, _, _ = _make_dataset(size=(5, 5), const=True)
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        y = x.isel(time=[-1])

        x_add1, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="precip")
        x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="temp")
        x = xr.merge([x, x_add1, x_add2])

        norm_dict = {
            "VHI": {"mean": 0, "std": 1},
            "precip": {"mean": 0, "std": 1},
            "temp": {"mean": 0, "std": 1},
        }

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}

        test_features = tmp_path / f"features/{experiment}/train/1980_1"
        test_features.mkdir(parents=True)
        pred_features = tmp_path / f"features/{experiment}/test/1980_1"
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

        model = GBDT(
            tmp_path,
            include_pred_month=use_pred_months,
            experiment=experiment,
            include_monthly_aggs=monthly_agg,
            normalize_y=False,
        )
        model.train()

        assert (
            type(model.model) == xgb.XGBRegressor
        ), f"Model attribute not a gradient boosted regressor!"

        test_arrays_dict, preds_dict = model.predict()
        assert (
            test_arrays_dict["1980_1"]["y"].size == preds_dict["hello"].shape[0]
        ), "Expected length of test arrays to be the same as the predictions"

        # test saving the model outputs
        model.evaluate(save_preds=True)

        save_path = model.data_path / "models" / experiment / "gbdt"
        assert (save_path / "preds_1980_1.nc").exists()
        assert (save_path / "results.json").exists()

        pred_ds = xr.open_dataset(save_path / "preds_1980_1.nc")
        assert np.isin(["lat", "lon", "time"], [c for c in pred_ds.coords]).all()
        assert y.time == pred_ds.time
