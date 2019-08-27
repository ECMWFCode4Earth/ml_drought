from pathlib import Path
from copy import copy

import torch
from torch import nn
from typing import cast, Dict, List, Optional, Tuple, Union

from .base import NNBase


class LinearNetwork(NNBase):

    model_name = 'linear_network'

    def __init__(self, layer_sizes: Union[int, List[int]],
                 dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 experiment: str = 'one_month_forecast',
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 include_latlons: bool = False,
                 include_monthly_aggs: bool = True,
                 include_yearly_aggs: bool = True,
                 surrounding_pixels: Optional[int] = None,
                 ignore_vars: Optional[List[str]] = None,
                 include_static: bool = True,
                 device: str = 'cuda:0') -> None:
        super().__init__(data_folder, batch_size, experiment, pred_months, include_pred_month,
                         include_latlons, include_monthly_aggs, include_yearly_aggs,
                         surrounding_pixels, ignore_vars, include_static, device)

        self.input_layer_sizes = copy(layer_sizes)
        if type(layer_sizes) is int:
            layer_sizes = cast(List[int], [layer_sizes])

        # to initialize and save the model
        self.layer_sizes = layer_sizes
        self.dropout = dropout
        self.input_size: Optional[int] = None

    def save_model(self):

        assert self.model is not None, 'Model must be trained before it can be saved!'

        model_dict = {
            'batch_size': self.batch_size,
            'model': {'state_dict': self.model.state_dict(),
                      'input_size': self.input_size},
            'layer_sizes': self.input_layer_sizes,
            'dropout': self.dropout,
            'include_pred_month': self.include_pred_month,
            'include_latlons': self.include_latlons,
            'surrounding_pixels': self.surrounding_pixels,
            'experiment': self.experiment,
            'ignore_vars': self.ignore_vars,
            'include_monthly_aggs': self.include_monthly_aggs,
            'include_yearly_aggs': self.include_yearly_aggs,
            'include_static': self.include_static,
            'device': self.device
        }

        torch.save(model_dict, self.model_dir / 'model.pt')

    def load(self, state_dict: Dict, input_size: int) -> None:

        self.input_size = input_size
        self.model: LinearModel = LinearModel(input_size=self.input_size,
                                              layer_sizes=self.layer_sizes,
                                              dropout=self.dropout,
                                              include_pred_month=self.include_pred_month,
                                              include_latlons=self.include_latlons,
                                              include_yearly_aggs=self.include_yearly_aggs,
                                              experiment=self.experiment,
                                              include_static=self.include_static)
        self.model.to(torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def _initialize_model(self, x_ref: Optional[Tuple[torch.Tensor, ...]]) -> nn.Module:
        if self.input_size is None:
            assert x_ref is not None, "x_ref can't be None if no input size is defined!"
            input_size = x_ref[0].view(x_ref[0].shape[0], -1).shape[1]
            if self.experiment == 'nowcast':
                current_tensor = x_ref[3]
                input_size += current_tensor.shape[-1]
            if self.include_yearly_aggs:
                ym_tensor = x_ref[4]
                input_size += ym_tensor.shape[-1]
            if self.include_static:
                static_tensor = x_ref[5]
                input_size += static_tensor.shape[-1]
            self.input_size = input_size

        model = LinearModel(input_size=self.input_size,
                            layer_sizes=self.layer_sizes,
                            dropout=self.dropout,
                            include_pred_month=self.include_pred_month,
                            include_latlons=self.include_latlons,
                            include_yearly_aggs=self.include_yearly_aggs,
                            experiment=self.experiment,
                            include_static=self.include_static)
        return model.to(torch.device(self.device))


class LinearModel(nn.Module):

    def __init__(self, input_size, layer_sizes, dropout, include_pred_month,
                 include_latlons, include_yearly_aggs, include_static,
                 experiment='one_month_forecast'):
        super().__init__()

        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_yearly_aggs = include_yearly_aggs
        self.include_static = include_static
        self.experiment = experiment

        # change the size of inputs if include_pred_month
        if self.include_pred_month:
            input_size += 12
        if include_latlons:
            input_size += 2

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

    def forward(self, x, pred_month=None, latlons=None, current=None,
                yearly_aggs=None, static=None):
        # flatten the final 2 dimensions (time / feature)
        x = x.contiguous().view(x.shape[0], -1)

        # concatenate the one_hot_month matrix onto X
        if self.include_pred_month:
            x = torch.cat((x, pred_month), dim=-1)
        if self.include_latlons:
            x = torch.cat((x, latlons), dim=-1)
        # concatenate the non-target variables onto X
        if self.experiment == 'nowcast':
            assert current is not None
            x = torch.cat((x, current), dim=-1)
        if self.include_yearly_aggs:
            x = torch.cat((x, yearly_aggs), dim=-1)
        if self.include_static:
            x = torch.cat((x, static), dim=-1)

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
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.batchnorm = nn.BatchNorm1d(num_features=out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.linear(x)))
        return self.dropout(x)
