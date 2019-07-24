import numpy as np
import pickle
import pytest

from src.models.neural_networks.ealstm import EALSTM
from src.models import EARecurrentNetwork

from tests.utils import _make_dataset


class TestEARecurrentNetwork:

    def test_save(self, tmp_path, monkeypatch):

        features_per_month = 5
        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25
        include_latlons = True
        include_pred_month = True
        include_yearly_means = True
        yearly_mean_size = 3

        def mocktrain(self):
            self.model = EALSTM(features_per_month, dense_features, hidden_size,
                                rnn_dropout, include_latlons, include_pred_month,
                                experiment='one_month_forecast', yearly_mean_size=yearly_mean_size)
            self.features_per_month = features_per_month
            self.yearly_mean_size = yearly_mean_size

        monkeypatch.setattr(EARecurrentNetwork, 'train', mocktrain)

        model = EARecurrentNetwork(hidden_size=hidden_size, dense_features=dense_features,
                                   include_pred_month=include_pred_month,
                                   rnn_dropout=rnn_dropout, data_folder=tmp_path,
                                   include_yearly_means=include_yearly_means)
        model.train()
        model.save_model()

        assert (tmp_path / 'models/one_month_forecast/ealstm/model.pkl').exists(), \
            f'Model not saved!'

        with (model.model_dir / 'model.pkl').open('rb') as f:
            model_dict = pickle.load(f)

        for key, val in model_dict['model']['state_dict'].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict['model']['features_per_month'] == features_per_month
        assert model_dict['model']['yearly_mean_size'] == yearly_mean_size
        assert model_dict['hidden_size'] == hidden_size
        assert model_dict['rnn_dropout'] == rnn_dropout
        assert model_dict['dense_features'] == dense_features
        assert model_dict['include_pred_month'] == include_pred_month
        assert model_dict['include_latlons'] == include_latlons
        assert model_dict['include_yearly_means'] == include_yearly_means
        assert model_dict['experiment'] == 'one_month_forecast'

    @pytest.mark.parametrize('use_pred_months', [True, False])
    def test_train(self, tmp_path, capsys, use_pred_months):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        test_features = tmp_path / 'features/one_month_forecast/train/hello'
        test_features.mkdir(parents=True)

        norm_dict = {'VHI': {'mean': np.zeros(x.to_array().values.shape[:2]),
                             'std': np.ones(x.to_array().values.shape[:2])}
                     }
        with (tmp_path / 'features/one_month_forecast/normalizing_dict.pkl').open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25

        model = EARecurrentNetwork(hidden_size=hidden_size, dense_features=dense_features,
                                   rnn_dropout=rnn_dropout, data_folder=tmp_path)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train smooth L1: 0.'
        assert expected_stdout in captured.out

        assert type(model.model) == EALSTM, \
            f'Model attribute not an RNN!'

    @pytest.mark.parametrize('use_pred_months', [True, False])
    def test_predict(self, tmp_path, use_pred_months):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / 'features/one_month_forecast/train/hello'
        train_features.mkdir(parents=True)

        test_features = tmp_path / 'features/one_month_forecast/test/hello'
        test_features.mkdir(parents=True)

        norm_dict = {'VHI': {'mean': np.zeros(x.to_array().values.shape[:2]),
                             'std': np.ones(x.to_array().values.shape[:2])}
                     }
        with (tmp_path / 'features/one_month_forecast/normalizing_dict.pkl').open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        x.to_netcdf(train_features / 'x.nc')
        y.to_netcdf(train_features / 'y.nc')

        dense_features = [10]
        hidden_size = 128
        rnn_dropout = 0.25

        model = EARecurrentNetwork(hidden_size=hidden_size, dense_features=dense_features,
                                   rnn_dropout=rnn_dropout, data_folder=tmp_path)
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "hello" is the only one which should be in the dictionaries
        assert ('hello' in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
        assert ('hello' in pred_dict.keys()) and (len(pred_dict) == 1)

        # _make_dataset with const=True returns all ones
        assert (test_arrays_dict['hello']['y'] == 1).all()
