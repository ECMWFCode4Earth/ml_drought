import torch
from torch import nn

from pathlib import Path
import pickle

from typing import Dict, List, Optional, Tuple

from .base import NNBase, LinearBlock


class EARecurrentNetwork(NNBase):

    model_name = 'ealstm'

    def __init__(self, hidden_size: int,
                 dense_features: List[int],
                 rnn_dropout: float = 0.25,
                 dense_dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 experiment: str = 'one_month_forecast',
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 surrounding_pixels: Optional[int] = None) -> None:
        super().__init__(data_folder, batch_size, experiment, pred_months, include_pred_month,
                         True, surrounding_pixels)

        # to initialize and save the model
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.dense_dropout = dense_dropout
        self.dense_features = dense_features
        self.features_per_month: Optional[int] = None
        self.current_size: Optional[int] = None

    def save_model(self):

        assert self.model is not None, 'Model must be trained before it can be saved!'

        model_dict = {
            'model': {'state_dict': self.model.state_dict(),
                      'features_per_month': self.features_per_month,
                      'current_size': self.current_size},
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'rnn_dropout': self.rnn_dropout,
            'dense_dropout': self.dense_dropout,
            'dense_features': self.dense_features,
            'include_pred_month': self.include_pred_month,
            'surrounding_pixels': self.surrounding_pixels,
            'experiment': self.experiment
        }

        with (self.model_dir / 'model.pkl').open('wb') as f:
            pickle.dump(model_dict, f)

    def load(self, state_dict: Dict, features_per_month: int, current_size: int) -> None:
        self.features_per_month = features_per_month
        self.current_size = current_size

        self.model: EALSTM = EALSTM(features_per_month=self.features_per_month,
                                    dense_features=self.dense_features,
                                    hidden_size=self.hidden_size,
                                    rnn_dropout=self.rnn_dropout,
                                    dense_dropout=self.dense_dropout,
                                    include_pred_month=self.include_pred_month,
                                    experiment=self.experiment,
                                    current_size=self.current_size)
        self.model.load_state_dict(state_dict)

    def _initialize_model(self, x_ref: Optional[Tuple[torch.Tensor, ...]]) -> nn.Module:
        if self.features_per_month is None:
            assert x_ref is not None, \
                f"x_ref can't be None if features_per_month or current_size is not defined"
            self.features_per_month = x_ref[0].shape[-1]
        if self.experiment == 'nowcast':
            if self.current_size is None:
                assert x_ref is not None, \
                    f"x_ref can't be None if features_per_month or current_size is not defined"
                self.current_size = x_ref[2].shape[-1]

        return EALSTM(features_per_month=self.features_per_month,
                      dense_features=self.dense_features,
                      hidden_size=self.hidden_size,
                      rnn_dropout=self.rnn_dropout,
                      dense_dropout=self.dense_dropout,
                      include_pred_month=self.include_pred_month,
                      experiment=self.experiment,
                      current_size=self.current_size)


class EALSTM(nn.Module):
    def __init__(self, features_per_month, dense_features, hidden_size,
                 rnn_dropout, dense_dropout, include_pred_month,
                 experiment, current_size=None):
        super().__init__()

        self.experiment = experiment
        self.include_pred_month = include_pred_month

        self.dropout = nn.Dropout(rnn_dropout)
        self.rnn = EALSTMCell(input_size_dyn=features_per_month,
                              input_size_stat=2,  # 2 for latlons, currently
                              hidden_size=hidden_size,
                              batch_first=True)
        self.hidden_size = hidden_size
        self.rnn_dropout = nn.Dropout(rnn_dropout)

        dense_input_size = hidden_size
        if include_pred_month:
            dense_input_size += 12
        if experiment == 'nowcast':
            assert current_size is not None
            dense_input_size += current_size
        dense_features.insert(0, dense_input_size)

        self.dense_layers = nn.ModuleList([
            LinearBlock(in_features=dense_features[i - 1],
                        out_features=dense_features[i],
                        dropout=dense_dropout)
            for i in range(1, len(dense_features))
        ])

        self.final_dense = nn.Linear(in_features=dense_features[-1], out_features=1)

        self.initialize_weights()

    def initialize_weights(self):

        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.linear.weight.data)

        nn.init.kaiming_uniform_(self.final_dense.weight.data)
        nn.init.constant_(self.final_dense.bias.data, 0)

    def forward(self, x, pred_month=None, latlons=None, current=None):

        # has to be optional to respect the positionality of the neural
        # nets
        assert latlons is not None, "Latlons can't be None!"

        hidden_state, cell_state = self.rnn(x, latlons)

        x = self.rnn_dropout(hidden_state[:, -1, :])

        if self.include_pred_month:
            x = torch.cat((x, pred_month), dim=-1)
        if self.experiment == 'nowcast':
            assert current is not None
            x = torch.cat((x, current), dim=-1)

        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
        return self.final_dense(x)


class EALSTMCell(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)

    This code was copied from
    https://github.com/kratzert/ealstm_regional_modeling/blob/master/papercode/ealstm.py

    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super().__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size_dyn,  # type: ignore
                                                        3 * hidden_size))  # type: ignore
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size,  # type: ignore
                                                        3 * hidden_size))  # type: ignore
        self.weight_sh = nn.Parameter(torch.FloatTensor(input_size_stat,  # type: ignore
                                                        hidden_size))  # type: ignore
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))  # type: ignore
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))  # type: ignore

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d, x_s):
        """[summary]
        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.
        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n
