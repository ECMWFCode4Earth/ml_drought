import numpy as np
from pathlib import Path
import random
import math

import torch
from torch import nn
from torch.nn import functional as F

import shap

from typing import cast, Any, Dict, List, Optional, Tuple, Union

from .base import ModelBase
from .utils import chunk_array
from .data import DataLoader, train_val_mask


class LinearNetwork(ModelBase):

    model_name = 'linear_network'

    def __init__(self, layer_sizes: Union[int, List[int]],
                 dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 experiment: str = 'one_month_forecast',
                 batch_size: int = 1,
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True) -> None:
        super().__init__(
            data_folder, batch_size, experiment=experiment,
            include_pred_month=include_pred_month,
            pred_months=pred_months
        )

        if type(layer_sizes) is int:
            layer_sizes = cast(List[int], [layer_sizes])

        # to initialize and save the model
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.input_size: Optional[int] = None

        # for reproducibility
        torch.manual_seed(42)

        self.explainer: Optional[shap.DeepExplainer] = None

    def save_model(self):

        assert self.model is not None, 'Model must be trained before it can be saved!'

        model_dict = {
            'state_dict': self.model.state_dict(),
            'layer_sizes': self.layer_sizes,
            'dropout': self.dropout,
            'input_size': self.input_size,
            'include_pred_month': self.include_pred_month
        }

        torch.save(model_dict, self.model_dir / 'model.pkl')

    def explain(self, x: Any) -> np.ndarray:
        assert self.model is not None, 'Model must be trained!'

        if self.explainer is None:
            background_samples = self._get_background(sample_size=100)
            self.explainer: shap.DeepExplainer = shap.DeepExplainer(
                self.model, background_samples)

        if self.include_pred_month:
            assert type(x) == list, \
                'include_pred_month is True, so this model expects a list of tensors as input'
            if len(x[1].shape) == 1:
                x[1] = self._one_hot_months(x[1])

        return self.explainer.shap_values(x)

    def train(self, num_epochs: int = 1,
              early_stopping: Optional[int] = None,
              batch_size: int = 256,
              learning_rate: float = 1e-3) -> None:
        print(f'Training {self.model_name}')

        if early_stopping is not None:
            len_mask = len(DataLoader._load_datasets(self.data_path, mode='train',
                                                     experiment=self.experiment,
                                                     shuffle_data=False,
                                                     pred_months=self.pred_months))
            train_mask, val_mask = train_val_mask(len_mask, 0.3)

            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train',
                                          experiment=self.experiment,
                                          mask=train_mask,
                                          to_tensor=True,
                                          pred_months=self.pred_months)
            val_dataloader = DataLoader(data_path=self.data_path,
                                        batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='train',
                                        experiment=self.experiment,
                                        mask=val_mask,
                                        to_tensor=True,
                                        pred_months=self.pred_months)
            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train',
                                          experiment=self.experiment,
                                          to_tensor=True,
                                          pred_months=self.pred_months)

        # initialize the model
        if self.input_size is None:
            x_ref, _ = next(iter(train_dataloader))
            self.input_size = x_ref[0].contiguous().view(
                x_ref[0].shape[0], -1
            ).shape[1]

        current = None
        if self.experiment == 'nowcast':
            current = x_ref[2]

        self.model: LinearModel = LinearModel(input_size=self.input_size,
                                              layer_sizes=self.layer_sizes,
                                              dropout=self.dropout,
                                              include_pred_month=self.include_pred_month,
                                              experiment=self.experiment,
                                              current=current)

        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                     lr=learning_rate)

        for epoch in range(num_epochs):
            train_rmse = []
            self.model.train()
            for x, y in train_dataloader:
                for x_batch, y_batch in chunk_array(x, y, batch_size, shuffle=True):
                    optimizer.zero_grad()
                    if self.experiment == 'nowcast':
                        current = x_batch[2]
                        pred = self.model(
                            x_batch[0],
                            self._one_hot_months(x_batch[1]),
                            current
                        )
                    else:
                        pred = self.model(
                            x_batch[0],
                            self._one_hot_months(x_batch[1])
                        )
                    loss = F.smooth_l1_loss(pred, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_rmse.append(loss.item())

            if early_stopping is not None:
                self.model.eval()
                val_rmse = []
                with torch.no_grad():
                    for x, y in val_dataloader:
                        val_pred_y = self.model(x[0], self._one_hot_months(x[1]))
                        val_loss = F.mse_loss(val_pred_y, y)

                        val_rmse.append(math.sqrt(val_loss.item()))

            print(f'Epoch {epoch + 1}, train RMSE: {np.mean(train_rmse)}')

            if early_stopping is not None:
                epoch_val_rmse = np.mean(val_rmse)
                print(f'Val RMSE: {epoch_val_rmse}')
                if epoch_val_rmse < best_val_score:
                    batches_without_improvement = 0
                    best_val_score = epoch_val_rmse
                    best_model_dict = self.model.state_dict()
                else:
                    batches_without_improvement += 1
                    if batches_without_improvement == early_stopping:
                        print('Early stopping!')
                        self.model.load_state_dict(best_model_dict)
                        return None

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        test_arrays_loader = DataLoader(
            data_path=self.data_path, batch_file_size=self.batch_size,
            shuffle_data=False, mode='test', to_tensor=True,
            experiment=self.experiment, pred_months=self.pred_months
        )

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        assert self.model is not None, 'Model must be trained before predictions can be generated'

        self.model.eval()
        with torch.no_grad():
            for dict in test_arrays_loader:
                for key, val in dict.items():
                    preds = self.model(
                        val.x.historical, self._one_hot_months(val.x.pred_months),
                        val.x.current
                    )
                    preds_dict[key] = preds.numpy()
                    test_arrays_dict[key] = {'y': val.y.numpy(), 'latlons': val.latlons}

        return test_arrays_dict, preds_dict

    def _get_background(self,
                        sample_size: int = 100) -> Union[torch.Tensor,
                                                         List[torch.Tensor]]:

        print('Extracting a sample of the training data')

        train_dataloader = DataLoader(data_path=self.data_path,
                                      batch_file_size=self.batch_size,
                                      shuffle_data=True, mode='train',
                                      pred_months=self.pred_months,
                                      to_tensor=True)
        output_tensors: List[torch.Tensor] = []
        if self.include_pred_month:
            output_pred_months: List[torch.Tensor] = []
        samples_per_instance = max(1, sample_size // len(train_dataloader))

        for x, _ in train_dataloader:
            while len(output_tensors) < sample_size:
                for _ in range(samples_per_instance):
                    idx = random.randint(0, x[0].shape[0] - 1)
                    output_tensors.append(x[0][idx])
                    if self.include_pred_month:
                        output_pred_months.append(self._one_hot_months(x[1][idx: idx + 1]))
        if self.include_pred_month:
            return [torch.stack(output_tensors), torch.cat(output_pred_months, dim=0)]
        else:
            return torch.stack(output_tensors)

    @staticmethod
    def _one_hot_months(indices: torch.Tensor) -> torch.Tensor:
        return torch.eye(14)[indices.long()][:, 1:-1]


class LinearModel(nn.Module):

    def __init__(
        self, input_size, layer_sizes, dropout, include_pred_month,
        current=None, experiment='one_month_forecast'
    ):
        super().__init__()

        self.include_pred_month = include_pred_month
        self.experiment = experiment

        # change the size of inputs if include_pred_month
        if self.include_pred_month:
            input_size += 12

        # change the size of inputs if experiment == `nowcast`
        if self.experiment == 'nowcast':
            assert current is not None, "`current` argument must not be none" \
                "when the experiment is `nowcast` because we need the " \
                "target timestep non-target feature data"
            input_size += current.shape[-1]

        # first layer is the input layer
        layer_sizes.insert(0, input_size)

        # dense layers from 2nd (1) -> penultimate (-2)
        self.dense_layers = nn.ModuleList([
            LinearBlock(in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i], dropout=dropout) for
            i in range(1, len(layer_sizes))
        ])

        # final layer is producing a scalar
        self.final_dense = nn.Linear(in_features=layer_sizes[-1], out_features=1)

        self.init_weights()

    def init_weights(self):
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.linear.weight.data)

        nn.init.kaiming_uniform_(self.final_dense.weight.data)
        # http://cs231n.github.io/neural-networks-2/#init
        # see: Initializing the biases
        nn.init.constant_(self.final_dense.bias.data, 0)

    def forward(self, x, pred_month=None, current=None):
        # flatten the final 2 dimensions (time / feature)
        x = x.contiguous().view(x.shape[0], -1)

        # concatenate the one_hot_month matrix onto X
        if self.include_pred_month:
            # flatten the array
            pred_month = pred_month.contiguous().view(x.shape[0], -1)
            x = torch.cat((x, pred_month), dim=-1)

        # concatenate the non-target variables onto X
        if self.experiment == 'nowcast':
            assert current is not None
            x = torch.cat((x, current), dim=-1)

        # pass the inputs through the layers
        for layer in self.dense_layers:
            x = layer(x)

        # pass through the final layer for a scalar prediction
        return self.final_dense(x)


class LinearBlock(nn.Module):
    """
    A linear layer followed by batchnorm, a ReLU activation, and dropout
    """

    def __init__(self, in_features, out_features, dropout=0.25):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.batchnorm = nn.BatchNorm1d(num_features=out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.linear(x)))
        return self.dropout(x)
