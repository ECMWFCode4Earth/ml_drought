import numpy as np
from pathlib import Path
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pickle

import shap

from typing import cast, Dict, List, Union, Tuple, Optional

from .base import ModelBase
from .utils import chunk_array
from .data import DataLoader, train_val_mask, TrainData


class LinearRegression(ModelBase):

    model_name = 'linear_regression'

    def __init__(self, data_folder: Path = Path('data'),
                 experiment: str = 'one_month_forecast',
                 batch_size: int = 1,
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 include_latlons: bool = True,
                 include_monthly_means: bool = True,
                 include_yearly_means: bool = True,
                 surrounding_pixels: Optional[int] = None) -> None:
        super().__init__(data_folder, batch_size, experiment, pred_months,
                         include_pred_month, include_latlons, include_monthly_means,
                         include_yearly_means, surrounding_pixels)

        self.explainer: Optional[shap.LinearExplainer] = None

    def train(self, num_epochs: int = 1,
              early_stopping: Optional[int] = None,
              batch_size: int = 256,
              val_split: float = 0.1) -> None:
        print(f'Training {self.model_name} for experiment {self.experiment}')

        if early_stopping is not None:
            len_mask = len(DataLoader._load_datasets(self.data_path, mode='train',
                                                     shuffle_data=False,
                                                     experiment=self.experiment))
            train_mask, val_mask = train_val_mask(len_mask, val_split)

            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          experiment=self.experiment,
                                          shuffle_data=True, mode='train',
                                          pred_months=self.pred_months,
                                          mask=train_mask,
                                          monthly_means=self.include_monthly_means,
                                          surrounding_pixels=self.surrounding_pixels)
            val_dataloader = DataLoader(data_path=self.data_path,
                                        batch_file_size=self.batch_size,
                                        experiment=self.experiment,
                                        shuffle_data=False, mode='train',
                                        pred_months=self.pred_months, mask=val_mask,
                                        monthly_means=self.include_monthly_means,
                                        surrounding_pixels=self.surrounding_pixels)
            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = DataLoader(data_path=self.data_path,
                                          experiment=self.experiment,
                                          batch_file_size=self.batch_size,
                                          pred_months=self.pred_months,
                                          shuffle_data=True, mode='train',
                                          monthly_means=self.include_monthly_means,
                                          surrounding_pixels=self.surrounding_pixels)
        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor()

        for epoch in range(num_epochs):
            train_rmse = []
            for x, y in train_dataloader:
                for batch_x, batch_y in chunk_array(x, y,
                                                    batch_size,
                                                    shuffle=True):
                    batch_y = cast(np.ndarray, batch_y)
                    x_in = self._concatenate_data(batch_x)

                    # fit the model
                    self.model.partial_fit(x_in, batch_y.ravel())
                    # evaluate the fit
                    train_pred_y = self.model.predict(x_in)
                    train_rmse.append(
                        np.sqrt(mean_squared_error(batch_y, train_pred_y))
                    )
            if early_stopping is not None:
                val_rmse = []
                for x, y in val_dataloader:
                    x_in = self._concatenate_data(x)
                    val_pred_y = self.model.predict(x_in)
                    val_rmse.append(np.sqrt(mean_squared_error(y, val_pred_y)))

            print(f'Epoch {epoch + 1}, train RMSE: {np.mean(train_rmse):.2f}')

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

    def explain(self, data: TrainData) -> Tuple[np.ndarray, ...]:

        assert self.model is not None, 'Model must be trained!'

        if self.explainer is None:
            mean = self._calculate_big_mean()
            self.explainer: shap.LinearExplainer = shap.LinearExplainer(
                self.model, (mean, None), feature_dependence='independent')

        x = data.historical
        batch, timesteps, dims = x.shape[0], x.shape[1], x.shape[2]
        reshaped_x = self._concatenate_data(data)

        explanations = self.explainer.shap_values(reshaped_x)

        historical = explanations[:, :timesteps * dims]
        additional = explanations[:, timesteps * dims:]

        # TODO: properly split the additional arrays

        return historical.reshape(batch, timesteps, dims), additional

    def save_model(self) -> None:

        assert self.model is not None, 'Model must be trained!'

        model_data = {
            'model': {'coef': self.model.coef_,
                      'intercept': self.model.intercept_},
            'experiment': self.experiment,
            'pred_months': self.pred_months,
            'include_pred_month': self.include_pred_month,
            'surrounding_pixels': self.surrounding_pixels,
            'batch_size': self.batch_size,
            'include_monthly_means': self.include_monthly_means,
            'include_yearly_means': self.include_yearly_means
        }

        with (self.model_dir / 'model.pkl').open('wb') as f:
            pickle.dump(model_data, f)

    def load(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        self.model: linear_model.SGDRegressor = linear_model.SGDRegressor()
        self.model.coef_ = coef
        self.model.intercept_ = intercept

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]],
                               Dict[str, np.ndarray]]:
        test_arrays_loader = DataLoader(
            data_path=self.data_path, batch_file_size=self.batch_size,
            experiment=self.experiment, shuffle_data=False, mode='test',
            pred_months=self.pred_months, surrounding_pixels=self.surrounding_pixels,
            monthly_means=self.include_monthly_means)

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        assert self.model is not None, 'Model must be trained!'

        for dict in test_arrays_loader:
            for key, val in dict.items():
                x = self._concatenate_data(val.x)
                preds = self.model.predict(x)
                preds_dict[key] = preds
                test_arrays_dict[key] = {'y': val.y, 'latlons': val.latlons}

        return test_arrays_dict, preds_dict

    def _calculate_big_mean(self) -> np.ndarray:
        """
        Calculate the mean of the training data in batches.
        For now, we don't calculate the covariance matrix,
        since it wouldn't fit in memory either
        """
        print('Calculating the mean of the training data')
        train_dataloader = DataLoader(data_path=self.data_path,
                                      batch_file_size=1,
                                      pred_months=self.pred_months,
                                      shuffle_data=False, mode='train',
                                      surrounding_pixels=self.surrounding_pixels,
                                      monthly_means=self.include_monthly_means)

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
            if self.experiment == 'nowcast':
                current = x[2]
                x_in = np.concatenate((x_in, current), axis=-1)
            sizes.append(x_in.shape[0])
            means.append(x_in.mean(axis=0))

        total_size = sum(sizes)
        weighted_means = [
            mean * size / total_size for mean, size in zip(means, sizes)
        ]
        return sum(weighted_means)

    def _concatenate_data(self, x: Union[Tuple[Optional[np.ndarray], ...],
                                         TrainData]) -> np.ndarray:

        if type(x) is tuple:
            x_his, x_pm, x_latlons, x_cur, x_ym = x  # type: ignore
        elif type(x) == TrainData:
            x_his, x_pm, x_latlons = x.historical, x.pred_months, x.latlons  # type: ignore
            x_cur, x_ym = x.current, x.yearly_means  # type: ignore

        assert x_his is not None, \
            'x[0] should be historical data, and therefore should not be None'
        x_in = x_his.reshape(x_his.shape[0], x_his.shape[1] * x_his.shape[2])

        if self.include_pred_month:
            # one hot encoding, should be num_classes + 1, but
            # for us its + 2, since 0 is not a class either
            pred_months_onehot = np.eye(14)[x_pm][:, 1:-1]
            x_in = np.concatenate(
                (x_in, pred_months_onehot), axis=-1
            )
        if self.include_latlons:
            x_in = np.concatenate((x_latlons, x_latlons), axis=-1)
        if self.experiment == 'nowcast':
            x_in = np.concatenate((x_in, x_cur), axis=-1)
        if self.include_yearly_means:
            x_in = np.concatenate((x_in, x_ym), axis=-1)

        return x_in
