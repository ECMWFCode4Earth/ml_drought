import numpy as np
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from typing import cast, Dict, List, Optional, Tuple, Union

from .base import ModelBase
from .data import DataLoader, train_val_mask


class LinearNetwork(ModelBase):

    model_name = 'linear_network'

    def __init__(self, layer_sizes: Union[int, List[int]],
                 dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1) -> None:
        super().__init__(data_folder, batch_size)

        if type(layer_sizes) is int:
            layer_sizes = cast(List[int], [layer_sizes])

        self.layer_sizes = layer_sizes
        self.dropout = dropout

    def train(self, num_epochs: int = 1,
              early_stopping: Optional[int] = None,
              learning_rate: float = 1e-3) -> None:
        print(f'Training {self.model_name}')

        if early_stopping is not None:
            len_mask = len(DataLoader._load_datasets(self.data_path, mode='train',
                                                     shuffle_data=False))
            train_mask, val_mask = train_val_mask(len_mask, 0.3)

            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train', mask=train_mask,
                                          to_tensor=True)
            val_dataloader = DataLoader(data_path=self.data_path,
                                        batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='train', mask=val_mask,
                                        to_tensor=True)
            batches_without_improvement = 0
            best_val_score = np.inf
        else:
            train_dataloader = DataLoader(data_path=self.data_path,
                                          batch_file_size=self.batch_size,
                                          shuffle_data=True, mode='train',
                                          to_tensor=True)

        # initialize the model
        x_ref, _ = next(iter(train_dataloader))
        input_size = x_ref.contiguous().view(x_ref.shape[0], -1).shape
        self.model: LinearModel = LinearModel(input_size=input_size[1],
                                              layer_sizes=self.layer_sizes,
                                              dropout=self.dropout)

        optimizer = torch.optim.Adam([pam for pam in self.model.parameters()],
                                     lr=learning_rate)

        for epoch in range(num_epochs):
            train_rmse = []
            self.model.train()
            for x, y in train_dataloader:
                # TODO: break x and y into more batches
                optimizer.zero_grad()
                pred = self.model(x)
                loss = F.smooth_l1_loss(pred, y)
                loss.backward()
                optimizer.step()

                train_rmse.append(loss.item())

            if early_stopping is not None:
                self.model.eval()
                val_rmse = []
                with torch.no_grad():
                    for x, y in val_dataloader:
                        val_pred_y = self.model(x)
                        val_loss = F.mse_loss(val_pred_y, y)

                        val_rmse.append(val_loss.item())

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

    def predict(self) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
        test_arrays_loader = DataLoader(data_path=self.data_path, batch_file_size=self.batch_size,
                                        shuffle_data=False, mode='test', to_tensor=True)

        preds_dict: Dict[str, np.ndarray] = {}
        test_arrays_dict: Dict[str, Dict[str, np.ndarray]] = {}

        if self.model is None:
            self.train()
            self.model: LinearModel

        self.model.eval()
        with torch.no_grad():
            for dict in test_arrays_loader:
                for key, val in dict.items():
                    preds = self.model(val.x)
                    preds_dict[key] = preds.numpy()
                    test_arrays_dict[key] = {'y': val.y.numpy(), 'latlons': val.latlons}

        return test_arrays_dict, preds_dict


class LinearModel(nn.Module):

    def __init__(self, input_size, layer_sizes, dropout):
        super().__init__()
        layer_sizes.insert(0, input_size)

        self.dense_layers = nn.ModuleList([
            LinearBlock(in_features=layer_sizes[i - 1],
                        out_features=layer_sizes[i], dropout=dropout) for
            i in range(1, len(layer_sizes))
        ])

        self.final_dense = nn.Linear(in_features=layer_sizes[-1], out_features=1)

        self.init_weights()

    def init_weights(self):
        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.linear.weight.data)

        nn.init.kaiming_uniform_(self.final_dense.weight.data)
        # http://cs231n.github.io/neural-networks-2/#init
        # see: Initializing the biases
        nn.init.constant_(self.final_dense.bias.data, 0)

    def forward(self, x):
        # flatten
        x = x.contiguous().view(x.shape[0], -1)
        for layer in self.dense_layers:
            x = layer(x)

        return self.final_dense(x)


class LinearBlock(nn.Module):
    """
    A linear layer followed by batchnorm, a ReLU activation, and dropout
    """

    def __init__(self, in_features, out_features, dropout=0.25):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.batchnorm = nn.BatchNorm1d(num_features=out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.linear(x)))
        return self.dropout(x)
