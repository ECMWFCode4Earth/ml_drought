from pathlib import Path
import pytorch_lightning as pl
import xarray as xr
from torch.nn import functional as F
import torch

from torch.utils.data import Dataloader

from copy import copy

from src.models.data import DataLoader, train_val_mask, TrainData
from src.models.dynamic_data import DynamicDataLoader
from src.models.neural_networks.nseloss import NSELoss


from typing import cast, Any, Dict, Optional, Union, List, Tuple

class LSTM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def train_dataloader(self):
        return DataLoader(MNIST(os.getcwd(), train=True, download=True,
                                transform=transforms.ToTensor()), batch_size=32)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    # from argparse import ArgumentParser
    # parser = ArgumentParser()
    # # arguments to parse
    # parser = pl.Trainer.add_argparse_args(parser)

    trainer = pl.Trainer()
    model = LSTM()

    trainer.fit(model)

