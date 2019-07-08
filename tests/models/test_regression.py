import numpy as np
import pickle
import pytest

from sklearn import linear_model
from src.models import LinearRegression
from src.models.data import DataLoader

from ..utils import _make_dataset


class TestLinearRegression:

    def test_save(self, tmp_path, monkeypatch):

        model_array = np.array([1, 1, 1, 1, 1])

        def mocktrain(self):
            class MockModel:
                @property
                def coef_(self):
                    return model_array

            self.model = MockModel()

        monkeypatch.setattr(LinearRegression, 'train', mocktrain)

        model = LinearRegression(tmp_path)
        model.train()
        model.save_model()

        assert (tmp_path / 'models/linear_regression/model.npy').exists(), f'Model not saved!'

        saved_model = np.load(tmp_path / 'models/linear_regression/model.npy')
        assert np.array_equal(model_array, saved_model), f'Different array saved!'

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

        model = LinearRegression(tmp_path, include_pred_month=use_pred_months)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train RMSE: 0.'
        assert expected_stdout in captured.out, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'

        assert type(model.model) == linear_model.SGDRegressor, \
            f'Model attribute not a linear regression!'

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
                        return (np.ones((10, 2, 1)), np.ones((10, ), dtype=np.int8)), None
                    else:
                        raise StopIteration()
            return MockIterator()

        def do_nothing(self, data_path, batch_file_size, shuffle_data, mode):
            pass

        monkeypatch.setattr(DataLoader, '__iter__', mockiter)
        monkeypatch.setattr(DataLoader, '__init__', do_nothing)

        model = LinearRegression(tmp_path)
        calculated_mean = model._calculate_big_mean()

        # 1 for the 2 features and for the first month, 0 for the rest
        expected_mean = np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # np.isclose because of rounding
        assert np.isclose(calculated_mean, expected_mean).all()
