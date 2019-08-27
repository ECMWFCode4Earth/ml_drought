import torch

from src.models.neural_networks.base import OneHotMonthEncoder


class TestOneHotMonthEncoder:

    def test_embedding_likeness(self):

        model = OneHotMonthEncoder(15)

        init_weights = torch.stack([torch.arange(1, 13)] * 15).float()
        init_bias = torch.zeros(15).float()
        model.encoder.weight.data = init_weights
        model.encoder.bias.data = init_bias

        for month in range(1, 13):
            indices = torch.tensor([month, month, month])
            month_tensor = torch.eye(14)[indices.long()][:, 1:-1].float()

            model.eval()
            with torch.no_grad():
                output_vals = model(month_tensor)

            assert (output_vals == month).all()
