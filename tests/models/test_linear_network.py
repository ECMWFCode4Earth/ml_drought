import numpy as np
import pickle
import torch

from src.models import LinearNetwork
from src.models.linear_network import LinearModel

from ..utils import _make_dataset


class TestLinearNetwork:

    def test_save(self, tmp_path, monkeypatch):

        layer_sizes = [10]
        input_size = 10
        dropout = 0.25

        def mocktrain(self):
            self.model = LinearModel(input_size, layer_sizes, dropout)
            self.input_size = input_size

        monkeypatch.setattr(LinearNetwork, 'train', mocktrain)

        model = LinearNetwork(data_folder=tmp_path, layer_sizes=layer_sizes,
                              dropout=dropout)
        model.train()
        model.save_model()

        assert (tmp_path / 'models/linear_network/model.pkl').exists(), f'Model not saved!'

        model_dict = torch.load(model.model_dir / 'model.pkl')

        for key, val in model_dict['state_dict'].items():
            assert (model.model.state_dict()[key] == val).all()

        assert model_dict['dropout'] == dropout
        assert model_dict['layer_sizes'] == layer_sizes
        assert model_dict['input_size'] == input_size

    def test_train(self, tmp_path, capsys):
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
                              dropout=dropout)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train RMSE: 0.'
        assert expected_stdout in captured.out

        assert type(model.model) == LinearModel, \
            f'Model attribute not a linear regression!'

    def test_predict(self, tmp_path):
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
                              dropout=dropout)
        model.train()
        test_arrays_dict, pred_dict = model.predict()

        # the foldername "hello" is the only one which should be in the dictionaries
        assert ('hello' in test_arrays_dict.keys()) and (len(test_arrays_dict) == 1)
        assert ('hello' in pred_dict.keys()) and (len(pred_dict) == 1)

        # _make_dataset with const=True returns all ones
        assert (test_arrays_dict['hello']['y'] == 1).all()
