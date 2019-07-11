import numpy as np
from pathlib import Path
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import shap

from typing import cast, Any, Dict, List, Tuple, Optional, Union

from .base import ModelBase
from .utils import chunk_array
from .data import DataLoader, train_val_mask


class LinearRegression(ModelBase):

    model_name = 'linear_regression'

    def __init__(self, data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True) -> None:
        super().__init__(data_folder, batch_size, pred_months, include_pred_month)

        self.explainer: Optional[shap.LinearExplainer] = None

    def train(self, num_epochs: int = 1,
              early_stopping: Optional[int] = None,
              batch_size: int = 256) -> None:
        print(f'Training {self.model_name}')

        if early_stopping is not None:
            len_mask = len(DataLoader._load_datasets(self.data_path, mode='train',
                                                     shuffle_data=False))
            train_mask, val_mask = train_val_mask(len_mask, 0.3)

            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train', mask=train_mask)
            val_dataloader = DataLoader(data_path=self.data_path,
                                        batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='train', mask=val_mask)
            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train')
        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor()

        for epoch in range(num_epochs):
            train_rmse = []
            for x, y in train_dataloader:
                for batch_x, batch_y in chunk_array(x, y, batch_size, shuffle=True):
                    batch_y = cast(np.ndarray, batch_y)
                    x_in = batch_x[0].reshape(
                        batch_x[0].shape[0],
                        batch_x[0].shape[1] * batch_x[0].shape[2])
                    if self.include_pred_month:
                        pred_months = batch_x[1]
                        # one hot encoding, should be num_classes + 1, but
                        # for us its + 2, since 0 is not a class either
                        pred_months_onehot = np.eye(14)[pred_months][:, 1:-1]
                        x_in = np.concatenate((x_in, pred_months_onehot), axis=-1)
                    self.model.partial_fit(x_in, batch_y.ravel())

                    train_pred_y = self.model.predict(x_in)
                    train_rmse.append(np.sqrt(mean_squared_error(batch_y, train_pred_y)))
            if early_stopping is not None:
                val_rmse = []
                for x, y in val_dataloader:
                    x_in = x[0].reshape(x[0].shape[0], x[0].shape[1] * x[0].shape[2])
                    if self.include_pred_month:
                        pred_months = x[1]
                        # one hot encoding, should be num_classes + 1, but
                        # for us its + 2, since 0 is not a class either
                        pred_months_onehot = np.eye(14)[pred_months][:, 1:-1]
                        x_in = np.concatenate((x_in, pred_months_onehot), axis=-1)
                    val_pred_y = self.model.predict(x_in)
                    val_rmse.append(np.sqrt(mean_squared_error(y, val_pred_y)))

            print(f'Epoch {epoch + 1}, train RMSE: {np.mean(train_rmse)}')

            if early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f'Val RMSE: {epoch_val_rmse}')
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                    best_coef = self.model.coef_
                    best_intercept = self.model.intercept_
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print('Early stopping!')
                        self.model.coef_ = best_coef
                        self.model.intercept_ = best_intercept
                        return None

    def explain(self, x: Any) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        assert self.model is not None, 'Model must be trained!'

        if self.explainer is None:
            mean = self._calculate_big_mean()
            self.explainer: shap.LinearExplainer = shap.LinearExplainer(
                self.model, (mean, None), feature_dependence='independent')

        if self.include_pred_month:
            assert type(x) in (tuple, list), f'Input x must be a tuple or list! Got {type(x)}'
            x, pred_months = x
            pred_months = np.eye(14)[pred_months][:, 1:-1]
        batch, timesteps, dims = x.shape[0], x.shape[1], x.shape[2]
        reshaped_x = x.reshape(batch, timesteps * dims)
        if self.include_pred_month:
            reshaped_x = np.concatenate((reshaped_x, pred_months), axis=-1)
        explanations = self.explainer.shap_values(reshaped_x)

        if not self.include_pred_month:
            return explanations.reshape(batch, timesteps, dims)

        historical = explanations[:, :timesteps * dims]
        additional = explanations[:, timesteps * dims:]

        return historical.reshape(batch, timesteps, dims), additional

    def save_model(self) -> None:

        assert self.model is not None, 'Model must be trained!'

        coefs = self.model.coef_
        np.save(self.model_dir / 'model.npy', coefs)

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

        test_arrays_loader = DataLoader(data_path=self.data_path, batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='test')

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        assert self.model is not None, 'Model must be trained!'

        for dict in test_arrays_loader:
            for key, val in dict.items():
                x = val.x.historical
                x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                if self.include_pred_month:
                    pred_months = val.x.pred_months
                    # one hot encoding, should be num_classes + 1, but
                    # for us its + 2, since 0 is not a class either
                    pred_months_onehot = np.eye(14)[pred_months][:, 1:-1]
                    x = np.concatenate((x, pred_months_onehot), axis=-1)
                preds = self.model.predict(x)
                preds_dict[key] = preds
                test_arrays_dict[key] = {'y': val.y, 'latlons': val.latlons}

        return test_arrays_dict, preds_dict

    def _calculate_big_mean(self) -> np.ndarray:
        """
        Calculate the mean of the training data in batches.

        For now, we don't calculate the covariance matrix, since it wouldn't fit in
        memory either
        """
        print('Calculating the mean of the training data')
        train_dataloader = DataLoader(data_path=self.data_path,
                                      batch_file_size=1,
                                      shuffle_data=False, mode='train')

        means, sizes = [], []
        for x, _ in train_dataloader:
            # first, flatten x
            x_in = x[0].reshape(x[0].shape[0], x[0].shape[1] * x[0].shape[2])
            if self.include_pred_month:
                pred_months = x[1]
                # one hot encoding, should be num_classes + 1, but
                # for us its + 2, since 0 is not a class either
                pred_months_onehot = np.eye(14)[pred_months][:, 1:-1]
                x_in = np.concatenate((x_in, pred_months_onehot), axis=-1)
            sizes.append(x_in.shape[0])
            means.append(x_in.mean(axis=0))

        total_size = sum(sizes)
        weighted_means = [mean * size / total_size for mean, size in zip(means, sizes)]
        return sum(weighted_means)
