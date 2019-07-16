import numpy as np
import pickle
import pytest
import torch

from src.models import LinearNetwork
from src.models.neural_networks.linear_network import LinearModel

from tests.utils import _make_dataset


class TestLinearNetwork:

    def test_save(self, tmp_path, monkeypatch):

        layer_sizes = [10]
        input_size = 10
        dropout = 0.25
        include_pred_month = True
        surrounding_pixels = 1

        def mocktrain(self):
            self.model = LinearModel(input_size, layer_sizes, dropout, include_pred_month)
            self.input_size = input_size

        monkeypatch.setattr(LinearNetwork, 'train', mocktrain)

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=layer_sizes,
                              dropout=dropout, include_pred_month=include_pred_month,
                              surrounding_pixels=surrounding_pixels)
        model.train()
        model.save_model()

        assert (tmp_path / 'models/linear_network/model.pkl').exists(), f'Model not saved!'

        model_dict = torch.load(model.model_dir / 'model.pkl')

        for key, val in model_dict['state_dict'].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict['dropout'] == dropout
        assert model_dict['layer_sizes'] == layer_sizes
        assert model_dict['input_size'] == input_size
        assert model_dict['include_pred_month'] == include_pred_month
        assert model_dict['surrounding_pixels'] == surrounding_pixels

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

        layer_sizes = [10]
        dropout = 0.25

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=layer_sizes,
                              dropout=dropout, include_pred_month=use_pred_months)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train RMSE: 0.'
        assert expected_stdout in captured.out

        assert type(model.model) == LinearModel, \
            f'Model attribute not a linear regression!'

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

        layer_sizes = [10]
        dropout = 0.25

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=layer_sizes,
                              dropout=dropout, include_pred_month=use_pred_months)
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "hello" is the only one which should be in the dictionaries
        assert ('hello' in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
        assert ('hello' in pred_dict.keys()) and (len(pred_dict) == 1)

        # _make_dataset with const=True returns all ones
        assert (test_arrays_dict['hello']['y'] == 1).all()

    def test_get_background(self, tmp_path):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        train_features = tmp_path / 'features/train/hello'
        train_features.mkdir(parents=True)

        x.to_netcdf(train_features / 'x.nc')
        y.to_netcdf(train_features / 'y.nc')

        norm_dict = {'VHI': {'mean': np.zeros(x.to_array().values.shape[:2]),
                             'std': np.ones(x.to_array().values.shape[:2])}
                     }
        with (tmp_path / 'features/normalizing_dict.pkl').open('wb') as f:
            pickle.dump(norm_dict, f)

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=[100],
                              dropout=0.25, include_pred_month=True)
        background = model._get_background(sample_size=3)
        assert background[0].shape[0] == 3, \
            f'Got {background[0].shape[0]} samples back, expected 3'
        assert background[1].shape[0] == 3, \
            f'Got {background[1].shape[0]} samples back, expected 3'
        assert len(background[1].shape) == 2, \
            f'Expected 2 dimensions, got {len(background[1].shape)}'
