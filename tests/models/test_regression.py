import numpy as np

from sklearn import linear_model
from src.models import LinearRegression

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
        model.save_model()

        assert (tmp_path / 'models/linear_regression/model.npy').exists(), f'Model not saved!'

        saved_model = np.load(tmp_path / 'models/linear_regression/model.npy')
        assert np.array_equal(model_array, saved_model), f'Different array saved!'

    def test_train(self, tmp_path, capsys):
        x, _, _ = _make_dataset(size=(5, 5), const=True)
        y = x.isel(time=[-1])

        test_features = tmp_path / 'features/train/hello'
        test_features.mkdir(parents=True)

        x.to_netcdf(test_features / 'x.nc')
        y.to_netcdf(test_features / 'y.nc')

        model = LinearRegression(tmp_path)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Train set RMSE: 0.0'
        assert expected_stdout in captured.out, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'

        assert type(model.model) == linear_model.SGDRegressor, \
            f'Model attribute not a linear regression!'
