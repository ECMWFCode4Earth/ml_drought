"""Run Cell State Regression
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd
import sys
import torch
from tqdm import tqdm
import xarray as xr
from typing import Any, Optional, List, Union, Tuple, Dict, DefaultDict
from numpy import AxisError
import numpy as np

from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import random_split, Subset, Dataset, DataLoader

sys.path.insert(2, "/home/tommy/neuralhydrology")
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import RegressionTester

sys.path.insert(2, "/home/tommy/ml_drought")
from scripts.drafts.gb_sm_data import read_gb_sm_data
from scripts.drafts.cell_state_extract import (
    load_normalised_cs_data,
    normalize_xr_by_basin,
    load_ealstm,
    load_config_file,
    load_lstm,
)
from src.analysis.evaluation import spatial_r2, spatial_rmse


# CellStateDataset
class CellStateDataset(Dataset):
    def __init__(
        self,
        input_data: xr.Dataset,
        target_data: xr.DataArray,
        config,
        mean: bool = True,
        set_device: bool = True,
    ):
        assert all(np.isin(["time", "dimension", "station_id"], input_data.dims))
        assert "cell_state" in input_data
        assert all(np.isin(input_data.station_id.values, target_data.station_id.values))

        self.input_data = input_data
        self.set_device = set_device

        #  All times that we have data for
        test_times = pd.date_range(
            config.test_start_date, config.test_end_date, freq="D"
        )
        bool_input_times = np.isin(input_data.time.values, test_times)
        bool_target_times = np.isin(target_data.time.values, test_times)
        self.all_times = list(
            set(target_data.time.values[bool_target_times]).intersection(
                set(input_data.time.values[bool_input_times])
            )
        )
        self.all_times = sorted(self.all_times)

        # get input/target data
        self.input_data = self.input_data.sel(time=self.all_times)
        self.target_data = target_data.sel(time=self.all_times)

        # basins
        self.basins = input_data.station_id.values

        # dimensions
        self.dimensions = len(input_data.dimension.values)

        # create x y pairs
        self.create_samples()

    def __len__(self):
        return len(self.samples)

    def create_samples(self):
        self.samples = []
        self.basin_samples = []
        self.time_samples = []

        for basin in self.basins:
            # read the basin data
            X = (
                self.input_data["cell_state"]
                .sel(station_id=basin)
                .values.astype("float64")
            )
            Y = self.target_data.sel(station_id=basin).values.astype("float64")

            # Ensure time is the 1st (0 index) axis
            X_time_axis = int(
                np.argwhere(~np.array([ax == len(self.all_times) for ax in X.shape]))
            )
            if X_time_axis != 1:
                X = X.transpose(1, 0)

            # drop nans over time (1st axis)
            finite_indices = np.logical_and(np.isfinite(Y), np.isfinite(X).all(axis=1))
            X, Y = X[finite_indices], Y[finite_indices]
            times = self.input_data["time"].values[finite_indices].astype(float)

            # convert to Tensors
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()
            if self.set_device:
                X = X.to("cuda:0")
                Y = Y.to("cuda:0")

            # create unique samples [(64,), (1,)]
            samples = [(x, y.reshape(-1)) for (x, y) in zip(X, Y)]
            self.samples.extend(samples)
            self.basin_samples.extend([basin for _ in range(len(samples))])
            self.time_samples.extend(times)

        #  SORT BY TIME (important for train-test split)
        sort_idx = np.argsort(self.time_samples)
        self.time_samples = np.array(self.time_samples)[sort_idx]
        self.samples = np.array(self.samples)[sort_idx]
        self.basin_samples = np.array(self.basin_samples)[sort_idx]

    def __getitem__(self, item: int) -> Tuple[Tuple[str, Any], Tuple[torch.Tensor]]:
        basin = str(self.basin_samples[item])
        time = self.time_samples[item]
        x, y = self.samples[item]

        return (basin, time), (x, y)


def get_matching_timesteps_input_target(
    config: Config, input_data: xr.Dataset, target_data: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    #  All times that we have data for
    test_times = pd.date_range(config.test_start_date, config.test_end_date, freq="D")
    bool_input_times = np.isin(input_data.time.values, test_times)
    bool_target_times = np.isin(target_data.time.values, test_times)
    all_times = list(
        set(target_data.time.values[bool_target_times]).intersection(
            set(input_data.time.values[bool_input_times])
        )
    )
    all_times = sorted(all_times)

    # get input/target data
    input_data = input_data.sel(time=all_times)
    target_data = target_data.sel(time=all_times)

    return input_data, target_data


def create_raw_input_data(
    config: Config,
    target_data: xr.Dataset,
    basins: List[int],
    return_as_array: bool = True,
    with_static: bool = False,
) -> xr.Dataset:
    """Create input data from raw input features to the LSTMs

    Args:
        config (Config): [description]
        target_data (xr.Dataset): [description]
        basins (List[int]): [description]
        return_as_array (bool, optional):
            Return as DataArray with variables unnamed stored in `dimension` coord.
            Defaults to True.
        with_static (bool, optional): [description].
            Defaults to False.

    Returns:
        xr.Dataset: Raw Data (normalised and preprocessed by modelling pipeline)
    """
    if with_static:
        assert return_as_array, "With static and return_as_array must be run together"
    #  1. recreate time period
    test_times = pd.date_range(config.test_start_date, config.test_end_date, freq="D")
    #  365 input times
    min_time = test_times.min() - pd.Timedelta(config.seq_length, unit="D")
    max_time = test_times.max() - pd.Timedelta(1, unit="D")
    test_times = pd.date_range(min_time, max_time, freq="D")

    # 2. recreate ds
    all_basin_ds = []

    if return_as_array:
        for basin in tqdm(basins, desc="Building Raw xr"):
            basin = str(basin)
            ds = RegressionTester(config, config.run_dir)._get_dataset(basin)
            x_d = ds.x_d[basin]["1D"].numpy()
            if with_static:
                n_times = x_d.shape[0]
                # copy static for each timestep (numpy array)
                static = ds.attributes[basin]
                static = np.tile(static.reshape(-1, 1), n_times)
                # create ONE Dynamic array to copy through time
                x_d = np.append(x_d, static.T, axis=-1)

            n_dims = x_d.shape[-1]
            station_ds = xr.Dataset(
                {
                    "cell_state": (
                        ["time", "dimension", "station_id"],
                        x_d.reshape(-1, n_dims, 1),
                    )
                },
                coords={
                    "time": test_times,
                    "dimension": np.arange(n_dims),
                    "station_id": [basin],
                },
            )
            all_basin_ds.append(station_ds)

    else:
        for basin in tqdm(basins, desc="Building Raw xr"):
            basin = str(basin)
            ds = RegressionTester(config, config.run_dir)._get_dataset(str(basin))
            x_d_tensor = ds.x_d[basin]["1D"]
            station_ds = xr.merge(
                [
                    xr.Dataset(
                        {
                            config.dynamic_inputs[ix]: (
                                ["time", "station_id"],
                                x_d_tensor.numpy()[:, ix].reshape(-1, 1),
                            )
                        },
                        coords={"time": test_times, "station_id": [basin]},
                    )
                    for ix in range(x_d_tensor.shape[-1])
                ]
            )
            all_basin_ds.append(station_ds)

    input_ds = xr.combine_by_coords(all_basin_ds)
    input_ds["station_id"] = [int(sid) for sid in input_ds["station_id"]]

    return input_ds


# train-test split
def get_train_test_dataset(
    dataset: Dataset, test_proportion: float = 0.2
) -> Tuple[Subset, Subset]:
    # SubsetRandomSampler = https://stackoverflow.com/a/50544887
    # Subset = https://stackoverflow.com/a/59414029
    #  random_split = https://stackoverflow.com/a/51768651
    all_data_size = len(dataset)
    train_size = int((1 - test_proportion) * all_data_size)
    test_size = all_data_size - train_size
    test_index = all_data_size - int(np.floor(test_size))

    #  test data is from final_sequence : end
    test_dataset = Subset(dataset, range(test_index, all_data_size))
    # train data is from start : test_index
    train_dataset = Subset(dataset, range(0, test_index))
    assert len(train_dataset) + len(test_dataset) == all_data_size

    return train_dataset, test_dataset


def train_validation_split(
    dataset: Dataset,
    validation_split: float = 0.1,
    shuffle_dataset: bool = True,
    random_seed: int = 42,
) -> Tuple[SubsetRandomSampler]:
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


# initialise model
def create_model(dataset):
    D_in = dataset.dimensions
    model = torch.nn.Sequential(torch.nn.Linear(D_in, 1))
    model = model.to("cuda:0")
    return model


# train model on each soil level
def train_model(
    model,
    train_dataset,
    learning_rate: float = 1e-2,
    n_epochs: int = 5,
    weight_decay: float = 0,
    val_split: bool = False,
    desc: str = "Training",
) -> Tuple[Any, List[float], List[float]]:

    if not val_split:
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    loss_fn = torch.nn.MSELoss(reduction="sum")

    # optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    #  TRAIN
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs), desc=desc):
        #  new train-validation split each epoch
        if val_split:
            #  create a unique test, val set (random) for each ...
            train_sampler, val_sampler = train_validation_split(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=256, sampler=train_sampler
            )
            val_loader = DataLoader(train_dataset, batch_size=256, sampler=val_sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=256)

        for (basin, time), data in train_loader:
            X, y = data
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # train/update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().numpy())

        # VALIDATE
        if val_split:
            model.eval()
            with torch.no_grad():
                for (basin, time), data in val_loader:
                    X, y = data
                    y_pred = model(X)
                    loss = loss_fn(y_pred, y)
                    val_losses.append(loss.detach().cpu().numpy())

    return model, train_losses, val_losses


# test models on each soil level
def to_xarray(predictions: Dict[str, List]) -> xr.Dataset:
    return pd.DataFrame(predictions).set_index(["time", "station_id"]).to_xarray()


def calculate_predictions(model, loader):
    from collections import defaultdict

    predictions = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for (basin, time), data in loader:
            X, y = data
            y_hat = model(X)
            predictions["time"].extend(pd.to_datetime(time))
            predictions["station_id"].extend(basin)
            predictions["y_hat"].extend(y_hat.detach().cpu().numpy().flatten())
            predictions["y"].extend(y.detach().cpu().numpy().flatten())

    return to_xarray(predictions)


#  ALL Training Process
def train_model_loop(
    config: Config,
    input_data: xr.Dataset,
    target_data: xr.Dataset,
    train_test: bool = True,
    train_val: bool = False,
    return_loaders: bool = True,
    desc: str = "",
) -> Tuple[List[float], BaseModel, Optional[Tuple[DataLoader]]]:
    #  1. create dataset (input, target)
    dataset = CellStateDataset(
        input_data=input_data, target_data=target_data, config=config,
    )

    #  2. create train-test split
    if train_test:
        #  build the train, test, validation
        train_dataset, test_dataset = get_train_test_dataset(dataset)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    else:
        train_dataset = dataset
        test_dataset = dataset
        test_loader = DataLoader(dataset, batch_size=256, shuffle=False)

    #  3. initialise the model
    model = create_model(dataset)

    # 4. Run training loop (iterate over batches)
    model, train_losses, val_losses = train_model(
        model,
        train_dataset,
        learning_rate=1e-3,
        n_epochs=20,
        weight_decay=0,
        val_split=train_val,
        desc=desc,
    )

    # 5. Save outputs (model, losses: List, dataloaders)
    if return_loaders:
        return train_losses, model, test_loader
    else:
        return train_losses, model, None


def run_regression_each_soil_level(
    config,
    input_data: xr.Dataset,
    target_data: xr.Dataset,
    train_val: bool = False,
    train_test: bool = True,
) -> Tuple[List[float], List[nn.Linear], List[DataLoader]]:
    assert (
        target_data.station_id.dtype == input_data.station_id.dtype
    ), "Need matching datatypes for input and target data - `input_data['station_id'] = [int(sid) for sid in input_data['station_id']]`"
    losses_list = []
    models = []
    test_loaders = []

    for soil_level in list(target_data.data_vars):
        # target data = SOIL MOISTURE
        target = target_data[soil_level]

        train_losses, model, test_loader = train_model_loop(
            config=config,
            input_data=input_data,
            target_data=target,
            train_test=train_test,
            train_val=train_val,
            desc=soil_level,
            return_loaders=True,
        )
        # store outputs of training process
        losses_list.append(train_losses)
        models.append(model)
        test_loaders.append(test_loader)

    return losses_list, models, test_loaders


# Test on hold-out prediction set
def run_all_soil_level_predictions(
    models: List[torch.nn.Linear],
    test_loaders: List[DataLoader],
    target_data: xr.Dataset,
) -> List[xr.Dataset]:
    all_preds = []
    for ix, model in enumerate(models):
        soil_level = list(target_data.data_vars)[ix]
        # target data = SOIL MOISTURE
        test_loader = test_loaders[ix]

        #  run forward pass and convert to xarray object
        preds = calculate_predictions(model, test_loader)

        all_preds.append(preds)
    return all_preds


# run evaluation
def create_error_datasets(all_preds: xr.Dataset) -> Tuple[xr.Dataset]:
    all_r2s = []
    all_rmses = []
    for ix, preds in enumerate(all_preds):
        variable = f"swvl{ix + 1}"
        all_r2s.append(spatial_r2(preds["y"], preds["y_hat"]).rename(variable))
        all_rmses.append(spatial_rmse(preds["y"], preds["y_hat"]).rename(variable))
    r2s = xr.merge(all_r2s).drop("time")
    rmses = xr.merge(all_rmses).drop("time")

    return r2s, rmses


# interpretation
def get_model_weights(model: torch.nn.Linear) -> Tuple[np.ndarray]:
    parameters = list(model.parameters())
    w = parameters[0].cpu().detach().numpy()
    b = parameters[1].cpu().detach().numpy()
    return w, b


def get_all_models_weights(models: List[torch.nn.Linear]) -> Tuple[np.ndarray]:
    model_outputs = defaultdict(dict)
    for sw_ix in range(len(models)):
        w, b = get_model_weights(models[sw_ix])
        model_outputs[f"swvl{sw_ix+1}"]["w"] = w
        model_outputs[f"swvl{sw_ix+1}"]["b"] = b

    ws = np.stack([model_outputs[swl]["w"] for swl in model_outputs.keys()]).reshape(
        4, 64
    )
    bs = np.stack([model_outputs[swl]["b"] for swl in model_outputs.keys()])
    return ws, bs


if __name__ == "__main__":
    EALSTM: bool = False
    data_dir = Path("/cats/datastore/data/")
    assert data_dir.exists()

    catchment_ids = [
        int(c)
        for c in [
            "12002",
            "15006",
            "27009",
            "27034",
            "27041",
            "39001",
            "39081",
            "43021",
            "47001",
            "54001",
            "54057",
            "71001",
            "84013",
        ]
    ]

    if EALSTM:
        run_dir = (
            data_dir / "runs/ensemble_EALSTM/ealstm_ensemble6_nse_1998_2008_2910_030601"
        )
        config = load_config_file(run_dir)
        model = load_ealstm(config)
    else:
        run_dir = data_dir / "runs/ensemble/lstm_ensemble6_nse_1998_2008_2710_171032"
        config = load_config_file(run_dir)
        model = load_lstm(config)

    TEST_BASINS = [str(id_) for id_ in catchment_ids]
    FINAL_VALUE = True
    TEST_TIMES = pd.date_range(config.test_start_date, config.test_end_date, freq="D")

    # get the normalised input data
    norm_cs_data = load_normalised_cs_data(
        config=config,
        model=model,
        test_basins=TEST_BASINS,
        test_times=TEST_TIMES,
        final_value=FINAL_VALUE,
    )
    norm_cs_data["station_id"] = [int(sid) for sid in norm_cs_data["station_id"]]

    # get the target SM data
    sm = read_gb_sm_data(data_dir)
    norm_sm = normalize_xr_by_basin(sm)
    norm_sm["station_id"] = [int(sid) for sid in norm_sm["station_id"]]

    # create input/target data
    if not FINAL_VALUE:
        # need to collapse seq_length by taking a mean over
        #  repeated "actual time" values
        input_data = norm_cs_data.groupby("time").mean()

    # train model on each soil level
    train_test = True
    train_val = False
    losses_list = []
    models = []
    test_loaders = []

    print("-- Training Models for Soil Levels --")
    losses_list, models, test_loaders = run_regression_each_soil_level(
        config,
        target_data=norm_sm,
        input_data=norm_cs_data,
        train_test=train_test,
        train_val=train_val,
    )

    # test models on each soil level
    print("-- Running Tests on hold-out test set --")
    all_preds = run_all_soil_level_predictions(
        models=models, test_loaders=test_loaders, target_data=norm_sm
    )

    # run evaluation on hold out set
    print("-- Creating Error Metrics: R2/RMSE --")
    r2s, rmses = create_error_datasets(all_preds)

    data = (
        r2s.to_dataframe()
        .reset_index()
        .melt(id_vars="station_id")
        .sort_values("variable")
    )
    print(f"MEAN R2: {data.drop('station_id', axis=1).mean().values[0]:.2f}")

    # extract weights and biases
    print("-- Extracting weights and biases --")
    ws, bs = get_all_models_weights(models)

    #  COMPARE TO RAW data ?
    raw_input_data = create_raw_input_data(
        config=config,
        target_data=norm_sm,
        basins=TEST_BASINS,
        return_as_array=False,
        with_static=False,
    )

    print("-- Training Models for Soil Levels [RAW DATA] --")
    raw_losses_list, raw_models, raw_test_loaders = run_regression_each_soil_level(
        config,
        target_data=norm_sm,
        input_data=raw_input_data,
        train_test=train_test,
        train_val=train_val,
    )

    print("-- Running Tests on hold-out test set [RAW DATA] --")
    raw_preds = run_all_soil_level_predictions(
        models=raw_models, test_loaders=raw_test_loaders, target_data=norm_sm
    )

    print("-- Creating Error Metrics: R2/RMSE --")
    raw_r2s, raw_rmses = create_error_datasets(raw_preds)

    raw_data = (
        raw_r2s.to_dataframe()
        .reset_index()
        .melt(id_vars="station_id")
        .sort_values("variable")
    )
    print(f"MEAN R2: {raw_data.drop('station_id', axis=1).mean().values[0]:.2f}")
