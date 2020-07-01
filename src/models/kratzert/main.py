import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import random
import pickle

from src.engineer.basin import CamelsCSV
from src.engineer.runoff import RunoffEngineer, CamelsDataLoader
from src.engineer.runoff_utils import get_basins, load_static_data

from .nseloss import NSELoss
from . import Model
import tqdm
import pandas as pd
from torch import nn
import sys
import time


###########
# Globals #
###########

# fixed settings for all experiments
# GLOBAL_SETTINGS = {
#     'batch_size': 256,
#     'clip_norm': True,
#     'clip_value': 1,
#     'dropout': 0.4,
#     'epochs': 30,
#     'hidden_size': 256,
#     'initial_forget_gate_bias': 5,
#     'log_interval': 50,
#     'learning_rate': 1e-3,
#     'seq_length': 270,
#     'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
#     'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
#     'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
#     'val_end': pd.to_datetime('30091999', format='%d%m%Y')
# }

# check if GPU is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training the model on:", DEVICE)

###############
# Prepare Run #
###############


def engineer_data(
    data_dir: Path,
    basins: List[int],
    train_dates: List[int],
    with_basin_str: bool = True,
    target_var: str = "discharge_spec",
    x_variables: Optional[List[str]] = ["precipitation", "peti"],
    static_variables: Optional[List[str]] = None,
    ignore_static_vars: Optional[List[str]] = None,
    seq_length: int = 365,
    with_static: bool = True,
    concat_static: bool = False,
):
    engineer = RunoffEngineer(
        data_dir=data_dir,
        basins=basins,
        train_dates=train_dates,
        with_basin_str=with_basin_str,
        target_var=target_var,
        x_variables=x_variables,
        static_variables=static_variables,
        ignore_static_vars=None,
        seq_length=seq_length,
        with_static=with_static,
        concat_static=concat_static,
    )

    engineer.create_training_data()


#############
# Run Model #
#############


def train(
    data_dir: Path,
    basins: List[str],
    train_dates: List[int],
    with_basin_str: bool = True,
    target_var: str = "discharge_spec",
    x_variables: Optional[List[str]] = ["precipitation", "peti"],
    static_variables: Optional[List[str]] = None,
    ignore_static_vars: Optional[List[str]] = None,
    seq_length: int = 365,
    with_static: bool = True,
    concat_static: bool = False,
    seed: int = 10101,
    cache: bool = True,
    batch_size: int = 32,
    num_workers: int = 1,
    hidden_size: int = 256,
    initial_forget_gate_bias: int = 5,
    dropout: float = 0.4,
    use_mse: bool = True,
    learning_rate: float = 1e-3,
    epochs: int = 10,
):
    # TODO: SEPARATE the engineering step from the model training step (maximise GPU)
    # TOMMY: assert False,

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    basins = get_basins(data_dir)

    # engineer the data for this training run
    engineer_data(
        data_dir=data_dir,
        basins=basins,
        train_dates=train_dates,
        with_basin_str=with_basin_str,
        target_var=target_var,
        x_variables=x_variables,
        static_variables=static_variables,
        ignore_static_vars=ignore_static_vars,
        seq_length=seq_length,
        with_static=with_static,
        concat_static=concat_static,
    )
    assert (data_dir / "features/features_train.h5").exists(), "Has the engineer been run?"

    # create dataloader
    data = CamelsDataLoader(
        data_dir=data_dir,
        basins=basins,
        concat_static=concat_static,
        cache=cache,
        with_static=with_static,
    )

    # initialise key parameters of the Model
    input_size_stat = len(data.static_df.columns) if with_static else 0
    dynamic_size = len(data.x_variables)
    if with_static:
        input_size_dyn = (
            dynamic_size + input_size_stat if concat_static else dynamic_size
        )
    else:
        input_size_dyn = dynamic_size

    if data.num_samples == 0:
        raise ValueError(f"No data for train dates: {train_dates}")
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    model = Model(
        input_size_dyn=input_size_dyn,
        input_size_stat=input_size_stat,
        hidden_size=hidden_size,
        initial_forget_bias=initial_forget_gate_bias,
        dropout=dropout,
        concat_static=concat_static,
        no_static=not with_static,  # inverse with_static
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # define loss function
    if use_mse:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()  # type: ignore

    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, epochs + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:  # type: ignore
                param_group["lr"] = learning_rates[epoch]

        # TODO: convert to self.train()
        train_epoch(model, optimizer, loss_func, loader, epoch, use_mse)

        # save the model
        model_str = _get_model_str(with_static=with_static, concat_static=concat_static)
        model_path = data_dir / f"models/model_{model_str}_epoch{epoch}.pt"
        model_path.parents[0].mkdir(exist_ok=True, parents=True)

        model.model_path = model_path

        torch.save(model.state_dict(), str(model_path))

    return model


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_func: nn.Module,
    loader: DataLoader,
    epoch: int,
    use_mse: bool,
    clip_value: float = 1.0,
    clip_norm: bool = True,
):
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
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm.tqdm(loader, file=sys.stdout)
    pbar.set_description(f"# Epoch {epoch}")

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
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

        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def evaluate(
    data_dir: Path,
    model_path: Path,
    input_size_dyn: int,
    input_size_stat: int,
    val_dates: List[int],
    with_static: bool = True,
    static_variables: Optional[List[str]] = None,
    dropout: float = 0.4,
    concat_static: bool = False,
    hidden_size: int = 256,
    target_var: str = "discharge_spec",
    x_variables: Optional[List[str]] = ["precipitation", "peti"],
    seq_length: int = 365,
):
    """Evaluate the model

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config

    """
    basins = get_basins(data_dir)

    # get static data (attributes) means/stds
    static_df = load_static_data(
        data_dir=data_dir,
        basins=basins,
        drop_lat_lon=True,
        static_variables=static_variables,
    )

    means = static_df.mean()
    stds = static_df.std()

    # create model
    model = Model(
        input_size_dyn=input_size_dyn,
        input_size_stat=input_size_stat,
        hidden_size=hidden_size,
        dropout=dropout,
        concat_static=concat_static,
        no_static=not with_static,
    ).to(DEVICE)

    # load trained model
    weight_file = model_path
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    val_dates = np.sort(val_dates)
    results: Dict[pd.DataFrame] = {}

    normalization_dict = pickle.load(
        open(data_dir / "features/normalization_dict.pkl", "rb")
    )

    for basin in tqdm.tqdm(basins):
        ds_test = CamelsCSV(
            data_dir=data_dir,
            basin=basin,
            train_dates=val_dates,
            normalization_dict=normalization_dict,
            is_train=True,
            target_var=target_var,
            x_variables=x_variables,
            static_variables=static_variables,
            seq_length=seq_length,
            with_static=with_static,
            concat_static=concat_static,
        )
        # capture missing timesteps
        valid_date_range = ds_test.date_range

        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=1)

        preds, obs = evaluate_basin(
            model, loader, normalization_dict=normalization_dict
        )

        # check if there is no data for that basin / time
        if ((preds is None) & (obs is None)) & (len(valid_date_range) == 0):
            print(f"No data for basin {basin} in val_period: {val_dates} ")
            continue

        assert len(preds) == len(obs)
        assert len(preds) == len(valid_date_range)

        df = pd.DataFrame(
            data={"qobs": obs.flatten(), "qsim": preds.flatten()},
            index=valid_date_range,
        )

        results[basin] = df

    save_eval_results(
        data_dir=data_dir,
        results=results,
        with_static=with_static,
        concat_static=concat_static,
    )


def _get_model_str(with_static: bool, concat_static: bool) -> str:
    if with_static:
        if concat_static:
            model = "lstm"
        else:
            model = "ealstm"
    else:
        model = "lstm_no_static"
    return model


def save_eval_results(
    data_dir: Path,
    results: Dict[Any, pd.DataFrame],
    with_static: bool,
    concat_static: bool,
) -> None:
    """Save the dictionary of dataframes to a pickle object

    Args:
        data_dir (Path): [description]
        results (Dict[pd.DataFrame]): [description]
        with_static (bool): [description]
        concat_static (bool): [description]
    """
    model = _get_model_str(with_static=with_static, concat_static=concat_static)

    dt = time.gmtime()
    dt_str = f"{dt.tm_year}_{dt.tm_mon:02}_{dt.tm_mday:02}:{dt.tm_hour:02}{dt.tm_min:02}{dt.tm_sec:02}"
    name = dt_str + "_" + f"{model}_results.pkl"

    (data_dir / "models").mkdir(exist_ok=True, parents=True)
    pickle.dump(results, open(data_dir / "models" / name, "wb"))


def rescale_features(
    feature: np.ndarray,
    normalization_dict: Dict[str, np.ndarray],
    variable: str = "target",
) -> np.ndarray:
    if variable == "target":
        feature = (feature * normalization_dict["target_std"]) + normalization_dict[
            "target_mean"
        ]
    else:
        assert False, "Not implemented other rescalings"

    return feature


def evaluate_basin(
    model: nn.Module, loader: DataLoader, normalization_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p = model(x_d, x_s[:, 0, :])[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        if preds is None:
            # assert False, "TODO: why is this going wrong?"
            return None, None

        preds = rescale_features(
            preds.numpy(), variable="target", normalization_dict=normalization_dict
        )
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs
