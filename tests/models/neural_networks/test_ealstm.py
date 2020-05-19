import pickle
import pytest
from copy import copy
import numpy as np

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from src.models.neural_networks.ealstm import EALSTM, EALSTMCell, OrgEALSTMCell
from src.models import EARecurrentNetwork
from src.models.neural_networks.base import train_val_mask, chunk_array

# from src.models.data import TrainData

from tests.utils import _make_dataset, _create_runoff_features_dir


# class TestEARecurrentNetwork:
#     def test_save(self, tmp_path, monkeypatch):

#         features_per_month = 5
#         dense_features = [10]
#         input_dense_features = copy(dense_features)
#         hidden_size = 128
#         rnn_dropout = 0.25
#         include_latlons = True
#         include_pred_month = True
#         include_yearly_aggs = True
#         yearly_agg_size = 3
#         include_prev_y = True
#         normalize_y = False
#         seq_length = 3

#         def mocktrain(self):
#             self.model = EALSTM(
#                 features_per_month=features_per_month,
#                 dense_features=dense_features,
#                 hidden_size=hidden_size,
#                 rnn_dropout=rnn_dropout,
#                 include_latlons=include_latlons,
#                 experiment="one_month_forecast",
#                 yearly_agg_size=yearly_agg_size,
#                 include_prev_y=include_prev_y,
#                 include_pred_month=include_pred_month,
#                 dropout=rnn_dropout,
#             )
#             self.features_per_month = features_per_month
#             self.yearly_agg_size = yearly_agg_size

#         monkeypatch.setattr(EARecurrentNetwork, "train", mocktrain)

#         model = EARecurrentNetwork(
#             hidden_size=hidden_size,
#             dense_features=dense_features,
#             include_pred_month=include_pred_month,
#             include_latlons=include_latlons,
#             rnn_dropout=rnn_dropout,
#             data_folder=tmp_path,
#             include_yearly_aggs=include_yearly_aggs,
#             normalize_y=normalize_y,
#             seq_length=seq_length,
#         )
#         model.target_var = "y"
#         model.test_years = [2011]
#         model.train()
#         model.save_model()

#         assert (
#             tmp_path / "models/one_month_forecast/ealstm/model.pt"
#         ).exists(), f"Model not saved!"

#         model_dict = torch.load(model.model_dir / "model.pt", map_location="cpu")

#         for key, val in model_dict["model"]["state_dict"].items():
#             assert (model.model.state_dict()[key] == val).all()

#         assert model_dict["model"]["features_per_month"] == features_per_month
#         assert model_dict["model"]["yearly_agg_size"] == yearly_agg_size
#         assert model_dict["hidden_size"] == hidden_size
#         assert model_dict["rnn_dropout"] == rnn_dropout
#         assert model_dict["dense_features"] == input_dense_features
#         assert model_dict["include_pred_month"] == include_pred_month
#         assert model_dict["include_latlons"] == include_latlons
#         assert model_dict["include_yearly_aggs"] == include_yearly_aggs
#         assert model_dict["experiment"] == "one_month_forecast"
#         assert model_dict["include_prev_y"] == include_prev_y
#         assert model_dict["normalize_y"] == normalize_y

#         assert model.seq_length == 3

#     @pytest.mark.parametrize(
#         "use_pred_month,use_static_embedding", [(True, 10), (False, None)]
#     )
#     def test_train(self, tmp_path, capsys, use_pred_month, use_static_embedding):
#         x, _, _ = _make_dataset(size=(5, 5), const=True)
#         y = x.isel(time=[-1])

#         test_features = tmp_path / "features/one_month_forecast/train/1980_1"
#         test_features.mkdir(parents=True)

#         norm_dict = {"VHI": {"mean": 0, "std": 1}}
#         with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
#             "wb"
#         ) as f:
#             pickle.dump(norm_dict, f)

#         x.to_netcdf(test_features / "x.nc")
#         y.to_netcdf(test_features / "y.nc")

#         # static
#         x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
#         static_features = tmp_path / f"features/static"
#         static_features.mkdir(parents=True)
#         x_static.to_netcdf(static_features / "data.nc")

#         static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
#         with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
#             pickle.dump(static_norm_dict, f)

#         dense_features = [10]
#         hidden_size = 128
#         rnn_dropout = 0.25

#         model = EARecurrentNetwork(
#             hidden_size=hidden_size,
#             dense_features=dense_features,
#             rnn_dropout=rnn_dropout,
#             data_folder=tmp_path,
#             static_embedding_size=use_static_embedding,
#             normalize_y=True,
#         )
#         model.train()

#         captured = capsys.readouterr()
#         expected_stdout = "Epoch 1, train smooth L1: 0."
#         assert expected_stdout in captured.out

#         assert type(model.model) == EALSTM, f"Model attribute not an EALSTM!"

#     # @pytest.mark.parametrize(
#     #     "use_pred_month,predict_delta",
#     #     [(True, True), (False, True), (True, False), (False, False)],
#     # )
#     # def test_predict_and_explain(self, tmp_path, use_pred_month, predict_delta):
#     #     x, _, _ = _make_dataset(size=(5, 5), const=True)
#     #     y = x.isel(time=[-1])

#     #     train_features = tmp_path / "features/one_month_forecast/train/1980_1"
#     #     train_features.mkdir(parents=True)

#     #     test_features = tmp_path / "features/one_month_forecast/test/1980_1"
#     #     test_features.mkdir(parents=True)

#     #     norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
#     #     with (tmp_path / "features/one_month_forecast/normalizing_dict.pkl").open(
#     #         "wb"
#     #     ) as f:
#     #         pickle.dump(norm_dict, f)

#     #     x.to_netcdf(test_features / "x.nc")
#     #     y.to_netcdf(test_features / "y.nc")

#     #     x.to_netcdf(train_features / "x.nc")
#     #     y.to_netcdf(train_features / "y.nc")

#     #     # static
#     #     x_static, _, _ = _make_dataset(size=(5, 5), add_times=False)
#     #     static_features = tmp_path / f"features/static"
#     #     static_features.mkdir(parents=True)
#     #     x_static.to_netcdf(static_features / "data.nc")

#     #     static_norm_dict = {"VHI": {"mean": 0.0, "std": 1.0}}
#     #     with (tmp_path / f"features/static/normalizing_dict.pkl").open("wb") as f:
#     #         pickle.dump(static_norm_dict, f)

#     #     dense_features = [10]
#     #     hidden_size = 128
#     #     rnn_dropout = 0.25

#     #     model = EARecurrentNetwork(
#     #         hidden_size=hidden_size,
#     #         dense_features=dense_features,
#     #         rnn_dropout=rnn_dropout,
#     #         data_folder=tmp_path,
#     #         predict_delta=predict_delta,
#     #         normalize_y=True,
#     #     )
#     #     model.train()
#     #     test_arrays_dict, pred_dict = model.predict()

#     #     # the foldername "1980_1" is the only one which should be in the dictionaries
#     #     assert ("1980_1" in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
#     #     assert ("1980_1" in pred_dict.keys()) and (len(pred_dict) == 1)

#     #     if not predict_delta:
#     #         # _make_dataset with const=True returns all ones
#     #         assert (test_arrays_dict["1980_1"]["y"] == 1).all()
#     #     else:
#     #         # _make_dataset with const=True & predict_delta
#     #         # returns a change of 0
#     #         assert (test_arrays_dict["1980_1"]["y"] == 0).all()

#     #     # test the Morris explanation works
#     #     test_dl = next(
#     #         iter(model.get_dataloader(mode="test", to_tensor=True, shuffle_data=False))
#     #     )

#     #     for key, val in test_dl.items():
#     #         output_m = model.explain(val.x, save_explanations=True, method="morris")
#     #         assert type(output_m) is TrainData
#     #         assert (model.model_dir / "analysis/morris_value_historical.npy").exists()

#     #         # TODO fix a bug in shap preventing this from passing
#     #         # output_s = model.explain(val.x, save_explanations=True, method="shap")
#     #         # assert type(output_s) is TrainData
#     #         # assert (model.model_dir / "analysis/shap_value_historical.npy").exists()


# class TestEALSTMCell:
#     @staticmethod
#     def test_ealstm(monkeypatch):
#         """
#         We implement our own unrolled RNN, so that it can be explained with
#         shap. This test makes sure it roughly mirrors the behaviour of the pytorch
#         LSTM.
#         """

#         batch_size, hidden_size, timesteps, dyn_input, static_input = 3, 5, 2, 6, 4

#         @staticmethod
#         def i_init(layer):
#             nn.init.constant_(layer.weight.data, val=1)

#         monkeypatch.setattr(EALSTMCell, "_reset_i", i_init)

#         def org_init(self):
#             """Initialize all learnable parameters of the LSTM"""
#             nn.init.constant_(self.weight_ih.data, val=1)
#             nn.init.constant_(self.weight_sh, val=1)

#             weight_hh_data = torch.eye(self.hidden_size)
#             weight_hh_data = weight_hh_data.repeat(1, 3)
#             self.weight_hh.data = weight_hh_data

#             nn.init.constant_(self.bias.data, val=0)
#             nn.init.constant_(self.bias_s.data, val=0)

#         monkeypatch.setattr(OrgEALSTMCell, "reset_parameters", org_init)

#         org_ealstm = OrgEALSTMCell(
#             input_size_dyn=dyn_input,
#             input_size_stat=static_input,
#             hidden_size=hidden_size,
#         )

#         our_ealstm = EALSTMCell(
#             input_size_dyn=dyn_input,
#             input_size_stat=static_input,
#             hidden_size=hidden_size,
#         )

#         static = torch.rand(batch_size, static_input)
#         dynamic = torch.rand(batch_size, timesteps, dyn_input)

#         with torch.no_grad():
#             org_hn, org_cn = org_ealstm(dynamic, static)
#             our_hn, our_cn = our_ealstm(dynamic, static)

#         assert np.isclose(
#             org_hn.numpy(), our_hn.numpy(), 0.01
#         ).all(), "Difference in hidden state"
#         assert np.isclose(
#             org_cn.numpy(), our_cn.numpy(), 0.01
#         ).all(), "Difference in cell state"


class TestEALSTMDynamic:
    def test_train(self, tmp_path):
        DYNAMIC = True

        ds, static = _create_runoff_features_dir(tmp_path)

        static_ignore_vars = ["area"]
        dynamic_ignore_vars = ["discharge"]
        target_var = "discharge"
        seq_length = 5
        test_years = [2001]
        forecast_horizon = 0
        batch_file_size = 1
        static_embedding_size = 5
        hidden_size = 5
        rnn_dropout = 0.3
        dropout = 0.3
        dense_features = None

        loss_func = "MSE"
        learning_rate = 1e-4

        # Model
        ealstm = EARecurrentNetwork(
            dynamic=DYNAMIC,
            data_folder=tmp_path,
            experiment="one_timestep_forecast",
            dynamic_ignore_vars=dynamic_ignore_vars,
            static_ignore_vars=static_ignore_vars,
            target_var=target_var,
            seq_length=seq_length,
            test_years=test_years,
            forecast_horizon=forecast_horizon,
            batch_size=batch_file_size,
            static_embedding_size=static_embedding_size,
            hidden_size=hidden_size,
            rnn_dropout=rnn_dropout,
            dropout=dropout,
            dense_features=dense_features,
            include_latlons=False,
            include_pred_month=False,
            include_timestep_aggs=False,
            include_yearly_aggs=False,
        )

        assert isinstance(ealstm, EARecurrentNetwork)

        # get the dataloaders
        dl = ealstm.get_dataloader(mode="train")
        len_mask = len(dl.valid_train_times)
        train_mask, val_mask = train_val_mask(len_mask, 0.1)
        train_dataloader = ealstm.get_dataloader(
            mode="train", mask=train_mask, to_tensor=True, shuffle_data=True
        )
        val_dataloader = ealstm.get_dataloader(
            mode="train", mask=val_mask, to_tensor=True, shuffle_data=False
        )
        train_iter = train_dataloader.__iter__()
        X, y = train_iter.__next__()

        # Test a single forward pass of the model
        model = ealstm._initialize_model(X)
        ealstm.model = model

        optimizer = torch.optim.Adam(
            [pam for pam in model.parameters()], lr=learning_rate
        )
        train_rmse = []
        train_losses = []
        val_rmses = []

        for epoch in range(2):
            epoch_rmses = []
            epoch_l1s = []

            #  ------------------------
            # TRAIN model
            # ------------------------
            model.train()
            # chunk the arrays FOR EACH STATION (each station individually)
            x_batch, y_batch = chunk_array(X, y, 10, shuffle=True)[0]
            x = Tensor(x_batch[0])
            x_static = Tensor(x_batch[5])

            optimizer.zero_grad()

            # make predictions with this data
            pred = model(x=x, static=x_static)
            if clip_zeros:
                pred = F.relu(pred)

            # calculate the loss
            loss = F.smooth_l1_loss(pred, y_batch)

            # backpropogate the errors
            loss.backward()
            # update the parameters in the network based on backprop
            optimizer.step()

            # check the rmse for the training data
            with torch.no_grad():
                rmse = F.mse_loss(pred, y_batch)
                epoch_rmses = np.append(epoch_rmses, np.sqrt(rmse.cpu().item()))

            epoch_l1s = np.append(epoch_l1s, loss.item())

            train_rmse.append(np.mean(epoch_rmses))
            train_losses.append(np.mean(epoch_l1s))
            #  ------------------------
            # EVALUATE model
            # ------------------------
            model.eval()
            val_rmse = []
            with torch.no_grad():
                (xval, yval) = val_dataloader.__iter__().__next__()
                valx = Tensor(xval[0])
                valx_static = Tensor(xval[5])
                val_pred_y = model(x=valx, static=xval[5])
                val_loss = F.mse_loss(val_pred_y, y)

                # append the validation losses
                val_rmse = np.append(val_rmse, np.sqrt(val_loss.cpu().item()))

            epoch_val_rmse = np.mean(val_rmse)
            val_rmses.append(epoch_val_rmse)

        assert len(val_rmses) == 2
        assert len(train_rmse) == 2
        assert len(train_losses) == 2

        if clip_zeros:
            assert all(pred >= 0)

        assert False
