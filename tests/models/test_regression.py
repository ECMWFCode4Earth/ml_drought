import numpy as np

from sklearn import linear_model
from src.models import LinearRegression


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

    def test_train(self, tmp_path, monkeypatch, capsys):

        y = np.array([1, 1, 1, 1, 1])
        x = np.expand_dims(np.expand_dims(y, 1), 1)

        def mockloadtrain(self):
            return x, y

        monkeypatch.setattr(LinearRegression, 'load_train_arrays', mockloadtrain)

        model = LinearRegression(tmp_path)
        model.train()

        captured = capsys.readouterr()
        expected_stdout = 'Train set RMSE: 0.0'
        assert expected_stdout in captured.out, \
            f'Expected stdout to be {expected_stdout}, got {captured.out}'

        assert type(model.model) == linear_model.LinearRegression, \
            f'Model attribute not a linear regression!'
