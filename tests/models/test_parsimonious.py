import numpy as np

from src.models.parsimonious import Persistence
from src.models.base import ModelArrays


class TestPersistence:

    def test_predict(self, tmp_path, monkeypatch):

        batch_size, timesteps, vars = 5, 6, 2

        def mockreturn(self):
            x = np.ones((batch_size, timesteps, vars))
            print(x.shape)
            x[:, -1, :] *= 2

            x_vars = ['VHI', 'precip']
            y = np.ones((batch_size, 1))
            y_var = 'VHI'

            return {'hello': ModelArrays(x=x, x_vars=x_vars, y=y, y_var=y_var)}

        monkeypatch.setattr(Persistence, 'load_test_arrays', mockreturn)

        predictor = Persistence(tmp_path)

        test_arrays, preds = predictor.predict()

        assert preds['hello'].shape == (batch_size, 1), \
            f'Wrong sized predictions! Got {preds["hello"].shape}, expected ({batch_size}, 1)'

        assert (preds['hello'] == 2).all(), f'Expected all values to be 2!'
