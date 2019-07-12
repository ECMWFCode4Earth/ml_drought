import torch
from torch import nn
import numpy as np

from src.models.neural_networks.rnn import UnrolledRNN


class TestUnrolledRNN:
    @staticmethod
    def test_rnn():
        """
        We implement our own unrolled RNN, so that it can be explained with
        shap. This test makes sure it roughly mirrors the behaviour of the pytorch
        LSTM.
        """

        batch_size, hidden_size, features_per_month = 32, 124, 6

        x = torch.ones(batch_size, 1, features_per_month)

        hidden_state = torch.zeros(1, x.shape[0], hidden_size)
        cell_state = torch.zeros(1, x.shape[0], hidden_size)

        torch_rnn = nn.LSTM(input_size=features_per_month,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=1)

        our_rnn = UnrolledRNN(input_size=features_per_month,
                              hidden_size=hidden_size,
                              batch_first=True)

        for parameters in torch_rnn.all_weights:
            for pam in parameters:
                nn.init.constant_(pam.data, 1)

        for parameters in our_rnn.parameters():
            for pam in parameters:
                nn.init.constant_(pam.data, 1)

        with torch.no_grad():
            o_out, (o_cell, o_hidden) = our_rnn(x, (hidden_state, cell_state))
            t_out, (t_cell, t_hidden) = torch_rnn(x, (hidden_state, cell_state))

        assert np.isclose(o_out.numpy(), t_out.numpy(), 0.01).all(), "Difference in hidden state"
        assert np.isclose(t_cell.numpy(), o_cell.numpy(), 0.01).all(), "Difference in cell state"
