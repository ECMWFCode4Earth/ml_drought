import numpy as np
import xarray as xr
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

        model = LinearRegression(tmp_path, experiment='one_month_forecast')
        model.train()
        model.save_model()

        assert (
            tmp_path / 'models/one_month_forecast/linear_regression/model.npy'
        ).exists(), f'Model not saved!'

        saved_model = np.load(tmp_path / 'models/one_month_forecast/linear_regression/model.npy')
        assert np.array_equal(model_array, saved_model), f'Different array saved!'

    @pytest.mark.parametrize('use_pred_months,experiment',
                             [(True, 'one_month_forecast'),
                              (True, 'nowcast'),
                              (False, 'one_month_forecast'),
                              (False, 'nowcast')])
    def test_train(self, tmp_path, capsys, use_pred_months, experiment):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        x_add1, _, _ = _make_dataset(size=(5, 5), const=True, variable_name='precip')
        x_add2, _, _ = _make_dataset(size=(5, 5), const=True, variable_name='temp')
        x = xr.merge([x, x_add1, x_add2])

        norm_dict = {'VHI': {'mean': np.zeros((1, x.to_array().values.shape[1])),
                             'std': np.ones((1, x.to_array().values.shape[1]))},
                     'precip': {'mean': np.zeros((1, x.to_array().values.shape[1])),
                                'std': np.ones((1, x.to_array().values.shape[1]))},
                     'temp': {'mean': np.zeros((1, x.to_array().values.shape[1])),
                              'std': np.ones((1, x.to_array().values.shape[1]))}}

        test_features = tmp_path / f'features/{experiment}/train/hello'
        test_features.mkdir(parents=True)
        pred_features = tmp_path / f'features/{experiment}/test/hello'
        pred_features.mkdir(parents=True)

        with (
            tmp_path / f'features/{experiment}/normalizing_dict.pkl'
        ).open('wb') as f:
            pickle.dump(norm_dict, f)

        x.to_netcdf(test_features / 'x.nc')
        x.to_netcdf(pred_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')
        y.to_netcdf(pred_features / 'y.nc')

        model = LinearRegression(
            tmp_path, include_pred_month=use_pred_months, experiment=experiment
        )
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Epoch 1, train RMSE: 0.'
        assert expected_stdout in captured.out, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'

        assert type(model.model) == linear_model.SGDRegressor, \
            f'Model attribute not a linear regression!'

        if (experiment != 'nowcast') and (use_pred_months):
            assert model.model.coef_.size == 122, "Expecting 120 coefficients" \
                "(3 historical vars * 36 months) + 12 pred_months one_hot encoded + 2 latlons"

        # Test Predictions / Evaluations
        test_arrays_dict, preds_dict = model.predict()
        if experiment == 'nowcast':
            assert (
                test_arrays_dict['hello']['y'].size == preds_dict['hello'].shape[0]
            ), "Expected length of test arrays to be the same as the predictions"

            if use_pred_months:
                assert model.model.coef_.size == 121, "Expect to have 119 coefficients" \
                    " (35 tstep x 3 historical) + 12 pred_months_one_hot + 2 current + 2 latlons"
            else:
                assert model.model.coef_.size == 109, "Expect to have 107 coefficients" \
                    " (35 tstep x 3 historical) + 2 current + 2 latlons"

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

        def do_nothing(self, data_path, batch_file_size, shuffle_data, mode, pred_months,
                       surrounding_pixels):
            pass

        monkeypatch.setattr(DataLoader, '__iter__', mockiter)
        monkeypatch.setattr(DataLoader, '__init__', do_nothing)

        model = LinearRegression(tmp_path)
        calculated_mean = model._calculate_big_mean()

        # 1 for the 2 features and for the first month, 0 for the rest
        expected_mean = np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

        # np.isclose because of rounding
        assert np.isclose(calculated_mean, expected_mean).all()
