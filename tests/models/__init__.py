import numpy as np

import pytest

from src.models.neural_networks.rnn import RNN, RecurrentNetwork
from src.models.neural_networks.linear_network import LinearNetwork, LinearModel
from src.models.neural_networks.ealstm import EALSTM, EARecurrentNetwork
from src.models.regression import LinearRegression
from src.models.gbdt import GBDT
from src.models import load_model


class TestLoadModels:
    def test_ealstm(self, tmp_path, monkeypatch):

        features_per_month = 5
        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25
        dense_dropout = 0.25
        include_pred_month = True

        def mocktrain(self):
            self.model = EALSTM(
                features_per_month,
                dense_features,
                hidden_size,
                rnn_dropout,
                dense_dropout,
                include_pred_month,
                experiment="one_month_forecast",
            )
            self.features_per_month = features_per_month

        monkeypatch.setattr(EARecurrentNetwork, "train", mocktrain)

        model = EARecurrentNetwork(
            hidden_size=hidden_size,
            dense_features=dense_features,
            rnn_dropout=rnn_dropout,
            data_folder=tmp_path,
        )
        model.train()
        model.save_model()

        model_path = tmp_path / "models/ealstm/rnn/model.pkl"

        assert model_path.exists(), "Model not saved!"

        new_model = load_model(model_path)

        assert type(new_model) == RecurrentNetwork

        for key, val in new_model.model.state_dict.items():
            assert (model.model.state_dict()[key] == val).all()

        assert new_model.dense_features == model.dense_features
        assert new_model.features_per_month == model.features_per_month
        assert new_model.hidden_size == model.hidden_size
        assert new_model.rnn_dropout == model.rnn_dropout
        assert new_model.include_pred_month == model.include_pred_month
        assert new_model.experiment == model.experiment
        assert new_model.surrounding_pixels == model.surrounding_pixels

    def test_rnn(self, tmp_path, monkeypatch):

        features_per_month = 5
        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25
        dense_dropout = 0.25
        include_pred_month = True

        def mocktrain(self):
            self.model = RNN(
                features_per_month,
                dense_features,
                hidden_size,
                rnn_dropout,
                dense_dropout,
                include_pred_month,
                experiment="one_month_forecast",
            )
            self.features_per_month = features_per_month

        monkeypatch.setattr(RecurrentNetwork, "train", mocktrain)

        model = RecurrentNetwork(
            hidden_size=hidden_size,
            dense_features=dense_features,
            rnn_dropout=rnn_dropout,
            data_folder=tmp_path,
        )
        model.train()
        model.save_model()

        model_path = tmp_path / "models/one_month_forecast/rnn/model.pkl"

        assert model_path.exists(), "Model not saved!"

        new_model = load_model(model_path)

        assert type(new_model) == RecurrentNetwork

        for key, val in new_model.model.state_dict.items():
            assert (model.model.state_dict()[key] == val).all()

        assert new_model.dense_features == model.dense_features
        assert new_model.features_per_month == model.features_per_month
        assert new_model.hidden_size == model.hidden_size
        assert new_model.rnn_dropout == model.rnn_dropout
        assert new_model.include_pred_month == model.include_pred_month
        assert new_model.experiment == model.experiment
        assert new_model.surrounding_pixels == model.surrounding_pixels

    def test_linear_network(self, tmp_path, monkeypatch):
        layer_sizes = [10]
        input_size = 10
        dropout = 0.25
        include_pred_month = True
        surrounding_pixels = 1

        def mocktrain(self):
            self.model = LinearModel(
                input_size, layer_sizes, dropout, include_pred_month
            )
            self.input_size = input_size

        monkeypatch.setattr(LinearNetwork, "train", mocktrain)

        model = LinearNetwork(
            data_folder=tmp_path,
            layer_sizes=layer_sizes,
            dropout=dropout,
            experiment="one_month_forecast",
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
        )
        model.train()
        model.save_model()

        model_path = tmp_path / "models/one_month_forecast/linear_network/model.pkl"

        assert model_path.exists(), "Model not saved!"

        new_model = load_model(model_path)

        assert type(new_model) == LinearNetwork

        for key, val in new_model.model.state_dict.items():
            assert (model.model.state_dict()[key] == val).all()

        assert new_model.dense_features == model.dense_features
        assert new_model.features_per_month == model.input_size
        assert new_model.hidden_size == model.dropout
        assert new_model.include_pred_month == model.include_pred_month
        assert new_model.experiment == model.experiment
        assert new_model.surrounding_pixels == model.surrounding_pixels

    def test_regression(self, tmp_path, monkeypatch):

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

        model = LinearRegression(tmp_path, experiment="one_month_forecast")
        model.train()
        model.save_model()

        model_path = tmp_path / "models/one_month_forecast/linear_regression/model.pkl"
        assert model_path.exists(), f"Model not saved!"

        new_model = load_model(model_path)
        assert type(new_model) == LinearRegression

        assert model.model.coef_ == coef_array
        assert model.model.intercept_ == intercept_array
        assert new_model.include_pred_month == model.include_pred_month
        assert new_model.experiment == model.experiment
        assert new_model.surrounding_pixels == model.surrounding_pixels

    @pytest.mark.xfail(reason="XGB is not installed in the test env")
    def test_xgboost(self, tmp_path, monkeypatch):

        import xgboost as xgb

        def mocktrain(self):
            self.model = xgb.XGBRegressor()

        monkeypatch.setattr(GBDT, "train", mocktrain)

        model = GBDT(tmp_path, experiment="one_month_forecast")
        model.train()
        model.save_model()

        model_path = tmp_path / "models/one_month_forecast/gbdt/model.pkl"
        assert model_path.exists(), f"Model not saved!"

        new_model = load_model(model_path)
        assert type(new_model) == GBDT

        assert new_model.include_pred_month == model.include_pred_month
        assert new_model.experiment == model.experiment
        assert new_model.surrounding_pixels == model.surrounding_pixels
