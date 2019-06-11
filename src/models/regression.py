import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from typing import Dict, Tuple, Optional

from .base import ModelBase
from .utils import chunk_array
from .data import DataLoader, train_val_mask


class LinearRegression(ModelBase):

    model_name = 'linear_regression'

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
                for batch_x, batch_y in chunk_array(x, y, batch_size):
                    batch_x = batch_x.reshape(batch_x.shape[0],
                                              batch_x.shape[1] * batch_x.shape[2])
                    self.model.partial_fit(batch_x, batch_y.ravel())

                    train_pred_y = self.model.predict(batch_x)
                    train_rmse.append(np.sqrt(mean_squared_error(batch_y, train_pred_y)))
            if early_stopping is not None:
                val_rmse = []
                for x, y in val_dataloader:
                    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
                    val_pred_y = self.model.predict(x)
                    val_rmse.append(np.sqrt(mean_squared_error(y, val_pred_y)))

            print(f'Epoch {epoch + 1}, train RMSE: {np.mean(train_rmse)}')

            if early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f'Val RMSE: {epoch_val_rmse}')
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print('Early stopping!')
                        return None

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
