import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from typing import Dict, Tuple

from .base import ModelBase
from .data import DataLoader


class LinearRegression(ModelBase):

    model_name = 'linear_regression'

    def train(self) -> None:
        print(f'Training {self.model_name}')

        train_dataloader = DataLoader(data_path=self.data_path, batch_file_size=self.batch_size,
                                      shuffle_data=True, mode='train')
        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor()

        for x, y in train_dataloader:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

            self.model.partial_fit(x, y.ravel())

            train_pred_y = self.model.predict(x)
            train_rmse = np.sqrt(mean_squared_error(y, train_pred_y))

            print(f'Train set RMSE: {train_rmse}')

    def save_model(self) -> None:

        if self.model is None:
            self.train()
            self.model: linear_model.LinearRegression

        coefs = self.model.coef_
        np.save(self.model_dir / 'model.npy', coefs)

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        test_arrays_loader = DataLoader(data_path=self.data_path, batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='test')

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        if self.model is None:
            self.train()
            self.model: linear_model.SGDRegressor

        for dict in test_arrays_loader:
            for key, val in dict.items():
                preds = self.model.predict(val.x.reshape(val.x.shape[0],
                                                         val.x.shape[1] * val.x.shape[2]))
                preds_dict[key] = preds
                test_arrays_dict[key] = {'y': val.y, 'latlons': val.latlons}

        return test_arrays_dict, preds_dict
