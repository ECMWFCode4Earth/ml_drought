import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from typing import Dict, Tuple

from .base import ModelBase, ModelArrays


class LinearRegression(ModelBase):

    model_name = 'linear_regression'

    def train(self) -> None:
        print(f'Training {self.model_name}')
        x, y = self.load_train_arrays()

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

        self.model: linear_model.LinearRegression = linear_model.LinearRegression()
        self.model.fit(x, y)

        train_pred_y = self.model.predict(x)
        train_rmse = np.sqrt(mean_squared_error(y, train_pred_y))

        print(f'Train set RMSE: {train_rmse}')

    def save_model(self) -> None:

        if self.model is None:
            self.train()
            self.model: linear_model.LinearRegression

        coefs = self.model.coef_
        np.save(self.model_dir / 'model.npy', coefs)

    def predict(self) -> Tuple[Dict[str, ModelArrays], Dict[str, np.ndarray]]:
        test_arrays = self.load_test_arrays()

        preds_dict: Dict[str, np.ndarray] = {}

        if self.model is None:
            self.train()
            self.model: linear_model.LinearRegression

        for key, val in test_arrays.items():
            preds = self.model.predict(val.x.reshape(val.x.shape[0],
                                                     val.x.shape[1] * val.x.shape[2]))
            preds_dict[key] = preds

        return test_arrays, preds_dict
