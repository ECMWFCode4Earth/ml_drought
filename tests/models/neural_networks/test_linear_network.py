import pickle
from copy import copy
import pytest
import xarray as xr
import torch

from src.models import LinearNetwork
from src.models.neural_networks.linear_network import LinearModel

from tests.utils import _make_dataset


class TestLinearNetwork:
    def test_save(self, tmp_path, monkeypatch):

        layer_sizes = [10]
        input_layer_sizes = copy(layer_sizes)
        input_size = 10
        dropout = 0.25
        include_pred_month = True
        include_latlons = True
        include_monthly_aggs = True
        surrounding_pixels = 1
        ignore_vars = ["precip"]
        include_yearly_aggs = True
        normalize_y = False

        def mocktrain(self):
            self.model = LinearModel(
                input_size,
                layer_sizes,
                dropout,
                include_pred_month,
                include_latlons,
                include_yearly_aggs,
                include_static=True,
                include_prev_y=True,
            )
            self.input_size = input_size

        monkeypatch.setattr(LinearNetwork, "train", mocktrain)

        model = LinearNetwork(
            data_folder=tmp_path,
            layer_sizes=layer_sizes,
            dropout=dropout,
            experiment="one_month_forecast",
            include_pred_month=include_pred_month,
            include_latlons=include_latlons,
            include_monthly_aggs=include_monthly_aggs,
            include_yearly_aggs=include_yearly_aggs,
            surrounding_pixels=surrounding_pixels,
            ignore_vars=ignore_vars,
            include_prev_y=True,
            normalize_y=normalize_y,
        )
        model.train()
        model.save_model()

        assert (
            tmp_path / "models/one_month_forecast/linear_network/model.pt"
        ).exists(), f"Model not saved!"

        model_dict = torch.load(model.model_dir / "model.pt", map_location="cpu")

        for key, val in model_dict["model"]["state_dict"].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict["dropout"] == dropout
        assert model_dict["layer_sizes"] == input_layer_sizes
        assert model_dict["model"]["input_size"] == input_size
        assert model_dict["include_pred_month"] == include_pred_month
        assert model_dict["include_latlons"] == include_latlons
        assert model_dict["include_monthly_aggs"] == include_monthly_aggs
        assert model_dict["include_yearly_aggs"] == include_yearly_aggs
        assert model_dict["surrounding_pixels"] == surrounding_pixels
        assert model_dict["ignore_vars"] == ignore_vars
        assert model_dict["include_prev_y"] is True
        assert model_dict["normalize_y"] == normalize_y

    @pytest.mark.parametrize(
        "use_pred_months,use_latlons,experiment,monthly_agg,static,predict_delta,check_inversion",
        [
            (True, False, "one_month_forecast", True, False, True, True),
            (False, True, "one_month_forecast", False, True, True, True),
            (False, True, "nowcast", True, False, True, True),
            (True, False, "nowcast", False, True, True, True),
            (True, False, "one_month_forecast", True, False, False, True),
            (False, True, "one_month_forecast", False, True, False, True),
            (False, True, "nowcast", True, False, False, True),
            (True, False, "nowcast", False, True, False, True),
            (True, False, "one_month_forecast", True, False, True, False),
            (False, True, "one_month_forecast", False, True, True, False),
            (False, True, "nowcast", True, False, True, False),
            (True, False, "nowcast", False, True, True, False),
            (True, False, "one_month_forecast", True, False, False, False),
            (False, True, "one_month_forecast", False, True, False, False),
            (False, True, "nowcast", True, False, False, False),
            (True, False, "nowcast", False, True, False, False),
        ],
    )
    def test_train(
        self,
        tmp_path,
        capsys,
        use_pred_months,
        use_latlons,
        experiment,
        monthly_agg,
        static,
        predict_delta,
        check_inversion,
    ):
        # make the x, y data (5*5 latlons, 36 timesteps, 3 features)
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        x_add1, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="precip")
        x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="temp")
        x = xr.merge([x, x_add1, x_add2])

        norm_dict = {
            "VHI": {"mean": 0, "std": 1},
            "precip": {"mean": 0, "std": 1},
            "temp": {"mean": 0, "std": 1},
        }

        test_features = tmp_path / f"features/{experiment}/train/1980_1"
        test_features.mkdir(parents=True, exist_ok=True)

        # make the normalising dictionary
        with (tmp_path / f"features/{experiment}/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        if static:
            x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
            static_features = tmp_path / f"features/static"
            static_features.mkdir(parents=True)
            x_static.to_netcdf(static_features / "data.nc")

            static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
            with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
                pickle.dump(static_norm_dict, f)

        layer_sizes = [10]
        dropout = 0.25

        model = LinearNetwork(
            data_folder=tmp_path,
            layer_sizes=layer_sizes,
            dropout=dropout,
            experiment=experiment,
            include_pred_month=use_pred_months,
            include_latlons=use_latlons,
            include_monthly_aggs=monthly_agg,
            static="embeddings",
            predict_delta=predict_delta,
        )
        model.train(check_inversion=check_inversion)

        captured = capsys.readouterr()
        expected_stdout = "Epoch 1, train smooth L1: "
        assert expected_stdout in captured.out

        assert (
            type(model.model) == LinearModel
        ), f"Model attribute not a linear regression!"

    @pytest.mark.parametrize(
        "use_pred_months,use_latlons,experiment",
        [
            (True, True, "one_month_forecast"),
            (True, False, "one_month_forecast"),
            (False, True, "nowcast"),
            (False, False, "nowcast"),
        ],
    )
    def test_predict(self, tmp_path, use_pred_months, use_latlons, experiment):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / f"features/{experiment}/train/1980_1"
        train_features.mkdir(parents=True)

        test_features = tmp_path / f"features/{experiment}/test/1980_1"
        test_features.mkdir(parents=True)

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / "data.nc")

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

        # if nowcast we need another x feature
        if experiment == "nowcast":
            x_add1, _, _ = _make_dataset(
                size=(5, 5), const=True, variable_name="precip"
            )
            x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name="temp")
            x = xr.merge([x, x_add1, x_add2])

            norm_dict = {
                "VHI": {"mean": 0, "std": 1},
                "precip": {"mean": 0, "std": 1},
                "temp": {"mean": 0, "std": 1},
            }
        else:
            norm_dict = {"VHI": {"mean": 0, "std": 1}}

        with (tmp_path / f"features/{experiment}/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        x.to_netcdf(train_features / "x.nc")
        y.to_netcdf(train_features / "y.nc")

        layer_sizes = [10]
        dropout = 0.25

        model = LinearNetwork(
            data_folder=tmp_path,
            layer_sizes=layer_sizes,
            dropout=dropout,
            experiment=experiment,
            include_pred_month=use_pred_months,
            include_latlons=use_latlons,
        )
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "1980_1" is the only one which should be in the dictionaries
        assert ("1980_1" in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
        assert ("1980_1" in pred_dict.keys()) and (len(pred_dict) == 1)

        # _make_dataset with const=True returns all ones
        assert (test_arrays_dict["1980_1"]["y"] == 1).all()

    def test_get_background(self, tmp_path):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / "features/one_month_forecast/train/1980_1"
        train_features.mkdir(parents=True)

        x.to_netcdf(train_features / "x.nc")
        y.to_netcdf(train_features / "y.nc")

        norm_dict = {"VHI": {"mean": 0, "std": 1}}
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "wb"
        ) as f:
            pickle.dump(norm_dict, f)

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / "data.nc")

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

        model = LinearNetwork(
            data_folder=tmp_path,
            layer_sizes=[100],
            dropout=0.25,
            include_pred_month=True,
        )
        background = model._get_background(sample_size=3)
        assert (
            background[0].shape[0] == 3
        ), f"Got {background[0].shape[0]} samples back, expected 3"
        assert (
            background[1].shape[0] == 3
        ), f"Got {background[1].shape[0]} samples back, expected 3"
        assert (
            len(background[1].shape) == 2
        ), f"Expected 2 dimensions, got {len(background[1].shape)}"
