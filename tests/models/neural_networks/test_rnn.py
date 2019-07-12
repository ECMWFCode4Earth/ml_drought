import torch
from torch import nn
import numpy as np
import pickle
import pytest

from src.models.neural_networks.rnn import UnrolledRNN, RNN
from src.models import RecurrentNetwork

from tests.utils import _make_dataset


class TestRecurrentNetwork:

    def test_save(self, tmp_path, monkeypatch):

        features_per_month = 5
        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25
        include_pred_month = True

        def mocktrain(self):
            self.model = RNN(features_per_month, dense_features, hidden_size,
                             rnn_dropout, include_pred_month)
            self.features_per_month = features_per_month

        monkeypatch.setattr(RecurrentNetwork, 'train', mocktrain)

        model = RecurrentNetwork(hidden_size=hidden_size, dense_features=dense_features,
                                 rnn_dropout=rnn_dropout, data_folder=tmp_path)
        model.train()
        model.save_model()

        assert (tmp_path / 'models/rnn/model.pkl').exists(), f'Model not saved!'

        model_dict = torch.load(model.model_dir / 'model.pkl')

        for key, val in model_dict['state_dict'].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict['features_per_month'] == features_per_month
        assert model_dict['hidden_size'] == hidden_size
        assert model_dict['rnn_dropout'] == rnn_dropout
        assert model_dict['dense_features'] == dense_features
        assert model_dict['include_pred_month'] == include_pred_month

    @pytest.mark.parametrize('use_pred_months', [True, False])
    def test_train(self, tmp_path, capsys, use_pred_months):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        test_features = tmp_path / 'features/train/hello'
        test_features.mkdir(parents=True)

        norm_dict = {'VHI': {'mean': np.zeros(x.to_array().values.shape[:2]),
                             'std': np.ones(x.to_array().values.shape[:2])}
                     }
        with (tmp_path / 'features/normalizing_dict.pkl').open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25

        model = RecurrentNetwork(hidden_size=hidden_size, dense_features=dense_features,
                                 rnn_dropout=rnn_dropout, data_folder=tmp_path)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train RMSE: 0.'
        assert expected_stdout in captured.out

        assert type(model.model) == RNN, \
            f'Model attribute not an RNN!'

    @pytest.mark.parametrize('use_pred_months', [True, False])
    def test_predict(self, tmp_path, use_pred_months):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / 'features/train/hello'
        train_features.mkdir(parents=True)

        test_features = tmp_path / 'features/test/hello'
        test_features.mkdir(parents=True)

        norm_dict = {'VHI': {'mean': np.zeros(x.to_array().values.shape[:2]),
                             'std': np.ones(x.to_array().values.shape[:2])}
                     }
        with (tmp_path / 'features/normalizing_dict.pkl').open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        x.to_netcdf(train_features / 'x.nc')
        y.to_netcdf(train_features / 'y.nc')

        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25

        model = RecurrentNetwork(hidden_size=hidden_size, dense_features=dense_features,
                                 rnn_dropout=rnn_dropout, data_folder=tmp_path)
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "hello" is the only one which should be in the dictionaries
        assert ('hello' in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
        assert ('hello' in pred_dict.keys()) and (len(pred_dict) == 1)

        # _make_dataset with const=True returns all ones
        assert (test_arrays_dict['hello']['y'] == 1).all()


class TestUnrolledRNN:
    @staticmethod
    def test_rnn():
        """
        We implement our own unrolled RNN, so that it can be explained with
        shap. This test makes sure it roughly mirrors the behaviour of the pytorch
        LSTM.
        """

        batch_size, hidden_size, features_per_month = 32, 124, 6

        x = torch.ones(batch_size, 1, features_per_month)

        hidden_state = torch.zeros(1, x.shape[0], hidden_size)
        cell_state = torch.zeros(1, x.shape[0], hidden_size)

        torch_rnn = nn.LSTM(input_size=features_per_month,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=1)

        our_rnn = UnrolledRNN(input_size=features_per_month,
                              hidden_size=hidden_size,
                              batch_first=True)

        for parameters in torch_rnn.all_weights:
            for pam in parameters:
                nn.init.constant_(pam.data, 1)

        for parameters in our_rnn.parameters():
            for pam in parameters:
                nn.init.constant_(pam.data, 1)

        with torch.no_grad():
            o_out, (o_cell, o_hidden) = our_rnn(x, (hidden_state, cell_state))
            t_out, (t_cell, t_hidden) = torch_rnn(x, (hidden_state, cell_state))

        assert np.isclose(o_out.numpy(), t_out.numpy(), 0.01).all(), "Difference in hidden state"
        assert np.isclose(t_cell.numpy(), o_cell.numpy(), 0.01).all(), "Difference in cell state"
