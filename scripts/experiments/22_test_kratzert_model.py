
from pathlib import Path
import xarray as xr
import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
import json
import sys

sys.path.append("../..")

from typing import DefaultDict, Dict, Tuple, Optional, Union, List, Any
from scripts.utils import (
    _rename_directory,
    get_data_path,
    rename_features_dir,
    rename_models_dir,
)
from scripts.experiments._static_ignore_vars import static_ignore_vars

from src.engineer.dynamic_engineer import DynamicEngineer
from src.models import EARecurrentNetwork, RecurrentNetwork
from src.models import load_model


import torch.nn
import torch
from src.models.kratzert.ealstm import EALSTM
from src.models.kratzert.lstm import LSTM
from src.models.neural_networks.nseloss import NSELoss


DEVICE = torch.device("cuda:0") and torch.cuda.is_available() else torch.device("cpu")


def get_dataloader():
    return


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connected layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static

        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)


def train(
    epochs: int = 2,
    entity_aware: bool = True,
    use_mse: bool = True
):
    loader = get_dataloader()

    # create model and optimizer
    input_size_stat = 18 if entity_aware else 0
    input_size_dyn = 2 if entity_aware else 20
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=256,
                  initial_forget_bias=5,
                  dropout=0.0,
                  concat_static=False if entity_aware else True,
                  no_static=False if entity_aware else True).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # define loss function
    if use_mse:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, epochs):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if not entity_aware:
            (x, _, _, _, _, x_s, _, q_stds), y = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif entity_aware:
            (x, _, _, _, _, x_s, _, q_stds), y = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")
