import torch
from torch import nn


class VariationalDropout(nn.Module):
    """
    This ensures the same dropout is applied to each timestep,
    as described in https://arxiv.org/pdf/1512.05287.pdf
    """
    def __init__(self, p):
        super().__init__()

        self.p = p
        self.mask = None

    def update_mask(self, x_shape, is_cuda):
        mask = torch.bernoulli(torch.ones(x_shape) * (1 - self.p)) / (1 - self.p)
        if is_cuda: mask = mask.cuda()
        self.mask = mask

    def forward(self, x):
        if not self.training:
            return x

        return self.mask * x
