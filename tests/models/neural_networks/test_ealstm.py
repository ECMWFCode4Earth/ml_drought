import pickle
import pytest
from copy import copy
import numpy as np

import torch
from torch import nn

from src.models.neural_networks.ealstm import EALSTM, EALSTMCell, OrgEALSTMCell
from src.models import EARecurrentNetwork
from src.models.data import TrainData

from tests.utils import _make_dataset
from ._build_static_data import get_static_embedding


class TestEARecurrentNetwork:
    def test_save(self, tmp_path, monkeypatch):

        features_per_month = 5
        dense_features = [10]
        input_dense_features = copy(dense_features)
        hidden_size = 128
        rnn_dropout = 0.25
        include_latlons = True
        include_pred_month = True
        include_yearly_aggs = True
        yearly_agg_size = 3
        include_prev_y = True
        normalize_y = False

        def mocktrain(self):
            self.model = EALSTM(
                features_per_month,
                dense_features,
                hidden_size,
                rnn_dropout,
                include_latlons,
                include_pred_month,
                experiment="one_month_forecast",
                yearly_agg_size=yearly_agg_size,
                include_prev_y=include_prev_y,
            )
            self.features_per_month = features_per_month
            self.yearly_agg_size = yearly_agg_size

        monkeypatch.setattr(EARecurrentNetwork, "train", mocktrain)

        model = EARecurrentNetwork(
            hidden_size=hidden_size,
            dense_features=dense_features,
            include_pred_month=include_pred_month,
            include_latlons=include_latlons,
            rnn_dropout=rnn_dropout,
            data_folder=tmp_path,
            include_yearly_aggs=include_yearly_aggs,
            normalize_y=normalize_y,
        )
        model.train()
        model.save_model()

        assert (
            tmp_path / "models/one_month_forecast/ealstm/model.pt"
        ).exists(), f"Model not saved!"

        model_dict = torch.load(model.model_dir / "model.pt", map_location="cpu")

        for key, val in model_dict["model"]["state_dict"].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict["model"]["features_per_month"] == features_per_month
        assert model_dict["model"]["yearly_agg_size"] == yearly_agg_size
        assert model_dict["hidden_size"] == hidden_size
        assert model_dict["rnn_dropout"] == rnn_dropout
        assert model_dict["dense_features"] == input_dense_features
        assert model_dict["include_pred_month"] == include_pred_month
        assert model_dict["include_latlons"] == include_latlons
        assert model_dict["include_yearly_aggs"] == include_yearly_aggs
        assert model_dict["experiment"] == "one_month_forecast"
        assert model_dict["include_prev_y"] == include_prev_y
        assert model_dict["normalize_y"] == normalize_y

    @pytest.mark.parametrize(
        "use_pred_months,use_static_embedding,static,check_inversion",
        [
            (True, 10, "features", True),
            (False, None, "features", True),
            (True, 10, "features", False),
            (False, None, "features", False),
        ],
    )
    def test_train(
        self,
        tmp_path,
        capsys,
        use_pred_months,
        use_static_embedding,
        static,
        check_inversion,
    ):
        # make directories
        for ts in ["2001_11", "2001_12"]:
            test_features = tmp_path / f"features/one_month_forecast/train/{ts}"
            test_features.mkdir(parents=True)

        norm_dict = {"VHI": {"mean": 0, "std": 1}}
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "wb"
        ) as f:
            pickle.dump(norm_dict, f)

        # save the X, y data pairs
        x, _, _ = _make_dataset(size=(5, 5), const=True)

        for ts in ["2001_11", "2001_12"]:
            if ts == "2001_12":
                y = x.sel(time="2001-12")
                x_save = x.sel(time=slice("2000-12", "2001-11"))
            else:
                y = x.sel(time="2001-11")
                x_save = x.sel(time=slice("2000-11", "2001-10"))
            x_save.to_netcdf(test_features / "x.nc")
            y.to_netcdf(test_features / "y.nc")

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / "data.nc")

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25

        model = EARecurrentNetwork(
            hidden_size=hidden_size,
            dense_features=dense_features,
            rnn_dropout=rnn_dropout,
            data_folder=tmp_path,
            static_embedding_size=use_static_embedding,
            normalize_y=True,
            include_yearly_aggs=False,
            static=static,
        )
        model.train(check_inversion=check_inversion)

        captured = capsys.readouterr()
        expected_stdout = "Epoch 1, train smooth L1: 0."
        assert expected_stdout in captured.out

        assert type(model.model) == EALSTM, f"Model attribute not an EALSTM!"

        # ------------------
        # Check static embedding
        # -------------------
        if use_static_embedding is not None:
            all_e, (all_static_x, all_latlons, all_pred_months) = get_static_embedding(
                ealstm=model
            )
            assert (
                all_e[0].shape[0] == 25
            ), f"Expect 25 latlon values (pixels). Got: {all_e[0].shape}"
            assert (
                all_latlons[0].shape[0] == 25
            ), f"Expect 25 latlon values (pixels). Got: {all_e[0].shape}"

            # Moved the PredMonth OHE to the dynamic data
            assert all_static_x[0].shape == (
                25,
                1,  #  13,
            ), f"Expect 13 static dimensions Got: {all_static_x[0].shape}"
            # assert (
            #     len(set(all_pred_months[0])) == 1
            # ), "Only expect one pred month (12=December)"

            # TODO: why is it only loading one month of data?
            # > [d.name for d in model.get_dataloader('train').data_files[0].parents[0].iterdir()]
            # ['2001_11', '2001_12']

    @pytest.mark.parametrize(
        "use_pred_months,predict_delta",
        [(True, True), (False, True), (True, False), (False, False)],
    )
    def test_predict_and_explain(self, tmp_path, use_pred_months, predict_delta):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / "features/one_month_forecast/train/1980_1"
        train_features.mkdir(parents=True)

        test_features = tmp_path / "features/one_month_forecast/test/1980_1"
        test_features.mkdir(parents=True)

        norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
            "wb"
        ) as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / "x.nc")
        y.to_netcdf(test_features / "y.nc")

        x.to_netcdf(train_features / "x.nc")
        y.to_netcdf(train_features / "y.nc")

        # static
        x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
        static_features = tmp_path / f"features/static"
        static_features.mkdir(parents=True)
        x_static.to_netcdf(static_features / "data.nc")

        static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
        with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
            pickle.dump(static_norm_dict, f)

        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25

        model = EARecurrentNetwork(
            hidden_size=hidden_size,
            dense_features=dense_features,
            rnn_dropout=rnn_dropout,
            data_folder=tmp_path,
            predict_delta=predict_delta,
            normalize_y=True,
        )
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "1980_1" is the only one which should be in the dictionaries
        assert ("1980_1" in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
        assert ("1980_1" in pred_dict.keys()) and (len(pred_dict) == 1)

        if not predict_delta:
            # _make_dataset with const=True returns all ones
            assert (test_arrays_dict["1980_1"]["y"] == 1).all()
        else:
            # _make_dataset with const=True & predict_delta
            # returns a change of 0
            assert (test_arrays_dict["1980_1"]["y"] == 0).all()

        # test the Morris explanation works
        test_dl = next(
            iter(model.get_dataloader(mode="test", to_tensor=True, shuffle_data=False))
        )

        for key, val in test_dl.items():
            output_m, _ = model.explain(val.x, save_explanations=True, method="morris")
            assert type(output_m) is TrainData
            assert (model.model_dir / "analysis/morris_value_historical.npy").exists()

            # TODO fix a bug in shap preventing this from passing
            # output_s = model.explain(val.x, save_explanations=True, method="shap")
            # assert type(output_s) is TrainData
            # assert (model.model_dir / "analysis/shap_value_historical.npy").exists()


class TestEALSTMCell:
    @staticmethod
    def test_ealstm(monkeypatch):
        """
        We implement our own unrolled RNN, so that it can be explained with
        shap. This test makes sure it roughly mirrors the behaviour of the pytorch
        LSTM.
        """

        batch_size, hidden_size, timesteps, dyn_input, static_input = 3, 5, 2, 6, 4

        @staticmethod
        def i_init(layer):
            nn.init.constant_(layer.weight.data, val=1)

        monkeypatch.setattr(EALSTMCell, "_reset_i", i_init)

        def org_init(self):
            """Initialize all learnable parameters of the LSTM"""
            nn.init.constant_(self.weight_ih.data, val=1)
            nn.init.constant_(self.weight_sh, val=1)

            weight_hh_data = torch.eye(self.hidden_size)
            weight_hh_data = weight_hh_data.repeat(1, 3)
            self.weight_hh.data = weight_hh_data

            nn.init.constant_(self.bias.data, val=0)
            nn.init.constant_(self.bias_s.data, val=0)

        monkeypatch.setattr(OrgEALSTMCell, "reset_parameters", org_init)

        org_ealstm = OrgEALSTMCell(
            input_size_dyn=dyn_input,
            input_size_stat=static_input,
            hidden_size=hidden_size,
        )

        our_ealstm = EALSTMCell(
            input_size_dyn=dyn_input,
            input_size_stat=static_input,
            hidden_size=hidden_size,
        )

        static = torch.rand(batch_size, static_input)
        dynamic = torch.rand(batch_size, timesteps, dyn_input)

        with torch.no_grad():
            org_hn, org_cn = org_ealstm(dynamic, static)
            our_hn, our_cn = our_ealstm(dynamic, static)

        assert np.isclose(
            org_hn.numpy(), our_hn.numpy(), 0.01
        ).all(), "Difference in hidden state"
        assert np.isclose(
            org_cn.numpy(), our_cn.numpy(), 0.01
        ).all(), "Difference in cell state"
