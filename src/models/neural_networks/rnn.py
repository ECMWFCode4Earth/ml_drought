import torch
from torch import nn


class UnrolledRNN(nn.Module):
    """An unrolled RNN. The motivation for this is mainly so that we can explain this model using
    the shap deep explainer, but also because we unroll the RNN anyway to apply dropout.
    """

    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.forget_gate = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Sigmoid()])

        self.update_gate = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Sigmoid()
        ])

        self.update_candidates = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Tanh()
        ])

        self.output_gate = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Sigmoid()
        ])

        self.cell_state_activation = nn.Tanh()

    def forward(self, x, state):
        hidden, cell = state

        if self.batch_first:
            hidden, cell = torch.transpose(hidden, 0, 1), torch.transpose(cell, 0, 1)

        forget_state = self.forget_gate(torch.cat((x, hidden), dim=-1))
        update_state = self.update_gate(torch.cat((x, hidden), dim=-1))
        cell_candidates = self.update_candidates(torch.cat((x, hidden), dim=-1))

        updated_cell = (forget_state * cell) + (update_state * cell_candidates)

        output_state = self.output_gate(torch.cat((x, hidden), dim=-1))
        updated_hidden = output_state * self.cell_state_activation(updated_cell)

        if self.batch_first:
            updated_hidden = torch.transpose(updated_hidden, 0, 1)
            updated_cell = torch.transpose(updated_cell, 0, 1)

        return updated_hidden, (updated_hidden, updated_cell)
