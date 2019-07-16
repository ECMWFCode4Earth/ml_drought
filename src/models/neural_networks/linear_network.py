from pathlib import Path

import torch
from torch import nn
from typing import cast, List, Optional, Tuple, Union

from .base import NNBase


class LinearNetwork(NNBase):

    model_name = 'linear_network'

    def __init__(self, layer_sizes: Union[int, List[int]],
                 dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 surrounding_pixels: Optional[int] = None) -> None:
        super().__init__(data_folder, batch_size, pred_months, include_pred_month,
                         surrounding_pixels)

        if type(layer_sizes) is int:
            layer_sizes = cast(List[int], [layer_sizes])

        # to initialize and save the model
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.input_size: Optional[int] = None

    def save_model(self):

        assert self.model is not None, 'Model must be trained before it can be saved!'

        model_dict = {
            'state_dict': self.model.state_dict(),
            'layer_sizes': self.layer_sizes,
            'dropout': self.dropout,
            'input_size': self.input_size,
            'include_pred_month': self.include_pred_month,
            'surrounding_pixels': self.surrounding_pixels
        }

        torch.save(model_dict, self.model_dir / 'model.pkl')

    def _initialize_model(self, x_ref: Tuple[torch.Tensor, ...]) -> nn.Module:
        self.input_size = x_ref[0].view(x_ref[0].shape[0], -1).shape[1]
        return LinearModel(input_size=self.input_size,
                           layer_sizes=self.layer_sizes,
                           dropout=self.dropout,
                           include_pred_month=self.include_pred_month)


class LinearModel(nn.Module):

    def __init__(self, input_size, layer_sizes, dropout, include_pred_month):
        super().__init__()

        self.include_pred_month = include_pred_month
        if self.include_pred_month:
            input_size += 12
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

    def forward(self, x, pred_month=None):
        # flatten
        x = x.contiguous().view(x.shape[0], -1)
        if self.include_pred_month:
            x = torch.cat((x, pred_month), dim=-1)
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
