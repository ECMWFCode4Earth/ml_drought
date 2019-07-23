from pathlib import Path
import pickle

import torch
from torch import nn
from typing import cast, Dict, List, Optional, Tuple, Union

from .base import NNBase, LinearBlock


class LinearNetwork(NNBase):

    model_name = 'linear_network'

    def __init__(self, layer_sizes: Union[int, List[int]],
                 dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 experiment: str = 'one_month_forecast',
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 surrounding_pixels: Optional[int] = None) -> None:
        super().__init__(data_folder, batch_size, experiment, pred_months, include_pred_month,
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
            'batch_size': self.batch_size,
            'model': {'state_dict': self.model.state_dict(),
                      'input_size': self.input_size},
            'layer_sizes': self.layer_sizes,
            'dropout': self.dropout,
            'include_pred_month': self.include_pred_month,
            'surrounding_pixels': self.surrounding_pixels,
            'experiment': self.experiment
        }

        with (self.model_dir / 'model.pkl').open('wb') as f:
            pickle.dump(model_dict, f)

    def load(self, state_dict: Dict, input_size: int) -> None:

        self.input_size = input_size
        self.model: LinearModel = LinearModel(input_size=self.input_size,
                                              layer_sizes=self.layer_sizes,
                                              dropout=self.dropout,
                                              include_pred_month=self.include_pred_month,
                                              experiment=self.experiment)
        self.model.load_state_dict(state_dict)

    def _initialize_model(self, x_ref: Optional[Tuple[torch.Tensor, ...]]) -> nn.Module:
        if self.input_size is None:
            assert x_ref is not None, "x_ref can't be None if no input size is defined!"
            input_size = x_ref[0].view(x_ref[0].shape[0], -1).shape[1]
            if self.experiment == 'nowcast':
                current_tensor = x_ref[2]
                input_size += current_tensor.shape[-1]
            self.input_size = input_size
        return LinearModel(input_size=self.input_size,
                           layer_sizes=self.layer_sizes,
                           dropout=self.dropout,
                           include_pred_month=self.include_pred_month,
                           experiment=self.experiment)


class LinearModel(nn.Module):

    def __init__(self, input_size, layer_sizes, dropout, include_pred_month,
                 experiment='one_month_forecast'):
        super().__init__()

        self.include_pred_month = include_pred_month
        self.experiment = experiment

        # change the size of inputs if include_pred_month
        if self.include_pred_month:
            input_size += 12

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
