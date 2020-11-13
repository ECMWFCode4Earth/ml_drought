import pickle
import torch
from tqdm import tqdm
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, DefaultDict
from collections import defaultdict

from sklearn.preprocessing import scale, StandardScaler
from torch.utils.data import DataLoader
from ruamel.yaml import YAML

#  NeuralHydrology imports
import sys

sys.path.insert(1, "/home/tommy/neuralhydrology")
sys.path.insert(2, "/home/tommy/ml_drought")

from neuralhydrology.modelzoo.ealstm import EALSTM
from neuralhydrology.modelzoo.cudalstm import CudaLSTM
from neuralhydrology.datasetzoo.camelsgb import CamelsGB
from neuralhydrology.datautils.utils import load_basin_file
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import NoTrainDataError
from neuralhydrology.training.train import BaseTrainer
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.evaluation import RegressionTester
from neuralhydrology.utils.errors import NoTrainDataError

#  mldrought
from scripts.drafts.gb_sm_data import read_gb_sm_data


###
#  1. Load in the model config file
def load_config_file(run_dir: Path) -> Config:
    #  Config file
    config_path = run_dir / "config.yml"
    config = Config(config_path)
    return config


def _load_model_weights(model, config: Config):
    model_path = config.run_dir / "model_epoch030.pt"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


def load_ealstm(config: Config):
    model = EALSTM(config)
    model = _load_model_weights(model, config)
    return model


def load_lstm(config: Config):
    model = CudaLSTM(config)
    model = _load_model_weights(model, config)
    return model


#  3. create Cell State embeddings for each basin we want to test
def get_states_from_forward(
    model: BaseModel,
    loader: DataLoader,
    hidden_state: bool = False,
    final_value: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    all_hidden_states = []
    all_cell_states = []
    #  For all the basin data in Loader
    for basin_data in loader:
        with torch.no_grad():
            predict = model.forward(basin_data)
            #  do we want to save the hidden states? NO!
            if hidden_state:
                all_hidden_states.append(predict["h_n"].detach().numpy())

            if final_value:
                # Return FINAL cell state value (targets, seq_len, hidden_dimensions)
                all_cell_states.append(predict["h_n"].detach().numpy()[:, -1, :])
            else:
                # Return ALL cell state values
                all_cell_states.append(predict["c_n"].detach().numpy())

    basin_cell_states = np.vstack(all_cell_states)
    if hidden_state:
        basin_hidden_states = np.vstack(all_hidden_states)
        return basin_hidden_states, basin_cell_states

    else:
        return basin_cell_states


def run_all_basin_forward_passes(
    test_basins: List[str], config: Config, model: BaseModel, final_value: bool = True,
) -> DefaultDict[str, Dict[str, np.ndarray]]:
    #  1. run the forward passes for each basin
    Tester = RegressionTester(
        cfg=config, run_dir=config.run_dir, period="test", init_model=True
    )

    all_basin_data = defaultdict(dict)
    for ix, basin in enumerate(tqdm(test_basins, desc="Extract Cell State")):
        try:
            # For each basin create a DataLoader
            ds = Tester._get_dataset(basin)
        except NoTrainDataError:
            print(f"{basin} Missing")
            continue

        loader = DataLoader(ds, batch_size=config.batch_size, num_workers=0)
        _, basin_cell_states = get_states_from_forward(model, loader, final_value)

        all_basin_data[basin]["c_s"] = basin_cell_states
        del basin_cell_states

    return all_basin_data


def create_time_arrays(test_times: List) -> Tuple[np.ndarray, np.ndarray]:
    #  CREATE TIME
    times = pd.to_datetime(test_times)
    time_deltas = np.array(
        sorted(pd.to_timedelta(np.arange(366), unit="d")[1:], reverse=True)
    )
    time_vals = []
    for ix in tqdm(range(len(times)), desc="Making True Time:"):
        time_vals.append(times[ix] - time_deltas)

    time_vals = np.array(time_vals)
    return times, time_vals


def convert_time_to_long_format(time_vals, times) -> Tuple[np.ndarray, np.ndarray]:
    #  time_vals = (target_time, seq_length)
    #  times = (target_time)
    #  LONG FORMAT
    actual_time = time_vals.flatten()
    target_time = np.tile(times, 365)
    assert actual_time.shape == target_time.shape

    return actual_time, target_time


def convert_dict_to_one_big_array(
    all_basin_data: DefaultDict[str, Dict[str, np.ndarray]]
) -> Tuple[List[str], np.ndarray]:
    # CONVERT to one big array
    all_cs_data = []
    basins = [k for k in all_basin_data.keys()]
    for basin in tqdm(basins, desc="Basin Dict to numpy:"):
        all_cs_data.append(all_basin_data[basin]["c_s"])

    all_cs_data = np.stack(all_cs_data, axis=-1)
    return basins, all_cs_data


#  4. convert cell state embeddings to xarray format(keep an eye on time)
def convert_dict_to_xarray(
    all_basin_data: DefaultDict[str, Dict[str, np.ndarray]],
    times: List[pd.Timestamp],
    wide_format: bool = False,
    final_value: bool = True,
) -> xr.Dataset:
    if final_value:
        assert (
            not wide_format
        ), "`final_value` and `wide_format` are mutually exclusive. `wide_format` requires time_delta values (seq_len)"
    basins, all_cs_data = convert_dict_to_one_big_array(all_basin_data)

    if wide_format:
        times, time_vals = create_time_arrays(times)
        #  WIDE FORMAT
        cs_data = xr.Dataset(
            {
                f"cell_state": (
                    ["target_time", "time_delta", "dimension", "station_id"],
                    all_cs_data,
                )
            },
            coords={
                "target_time": times.values,
                "time_delta": time_deltas,
                "dimension": np.arange(basin_cs.shape[-1]),
                "station_id": basins,
            },
        )
    else:
        #  LONG FORMAT
        if not final_value:
            times, time_vals = create_time_arrays(times)
            actual_time, target_time = convert_time_to_long_format(time_vals, times)
            long_data = all_cs_data.reshape(
                -1, all_cs_data.shape[2], all_cs_data.shape[3]
            )
        else:
            actual_time = target_time = times
            long_data = all_cs_data
        assert long_data.shape[0] == actual_time.shape[0]

        cs_data = xr.Dataset(
            {
                "cell_state": (["time", "dimension", "station_id"], long_data),
                "target_time": (["time"], target_time),
            },
            coords={
                "time": actual_time,
                "dimension": np.arange(long_data.shape[1]),
                "station_id": basins,
            },
        )

    return cs_data


#  5. normalise cell state data
def normalize_cell_states(cell_state: np.ndarray, desc: str = "Normalize"):
    original_shape = cell_state.shape
    store = []
    s = StandardScaler()
    dimensions = len(cell_state.shape)
    # (target_time, time_delta, dimensions)
    if dimensions == 3:
        for ix in tqdm(range(cell_state.shape[-1]), desc=desc):
            store.append(s.fit_transform(cell_state[:, :, ix]))

        c_state = np.stack(store)
        c_state = c_state.transpose(1, 2, 0)
        assert c_state.shape == original_shape

    elif dimensions == 2:
        for ix in tqdm(range(cell_state.shape[-1]), desc=desc):
            store.append(s.fit_transform(cell_state[:, ix].reshape(-1, 1)))
        c_state = np.stack(store)[:, :, 0]
        c_state = c_state.T
        assert c_state.shape == original_shape

    else:
        raise NotImplementedError

    return c_state


def normalize_xarray_cstate(c_state: xr.Dataset) -> xr.Dataset:
    #  Normalize all station values in cs_data:
    all_normed = []
    for station in c_state.station_id.values:
        norm_state = normalize_cell_states(
            c_state.sel(station_id=station)["cell_state"].values
        )
        all_normed.append(norm_state)

    all_normed_stack = np.stack(all_normed).transpose(1, 2, 0)
    norm_c_state = xr.ones_like(c_state["cell_state"])
    norm_c_state = norm_c_state * all_normed_stack

    return norm_c_state


# NORMALIZE SM
def normalize_dataframe_by_basin(df: pd.DataFrame):
    assert all(np.isin(["time", "station_id"], df.reset_index().columns))
    scaler = StandardScaler()
    norm_ = df.groupby("station_id").apply(lambda x: scaler.fit_transform(x).flatten())

    norm_df = (
        norm_.explode()
        .reset_index()
        .rename({0: "norm"}, axis=1)
        .astype({"norm": "float64"})
    )
    norm_df["time"] = df.reset_index().sort_values(["station_id", "time"])["time"]
    norm_df = norm_df.set_index(["station_id", "time"])

    return norm_df


def normalize_xr_by_basin(ds):
    return (ds - ds.mean(dim="time")) / ds.std(dim="time")


#  6. return cell state data
def check_data_not_duplicated(ds: xr.Dataset, var_name: str = "cell_state"):
    assert all(np.isin(["time", "station_id"], [v for v in ds.dims]))
    #  CHECK THAT DATA IS NOT DUPLICATED
    id0 = cs_data.isel(station_id=0).isel(time=slice(0, 100))[var_name]
    id1 = cs_data.isel(station_id=1).isel(time=slice(0, 100))[var_name]

    print(f"{np.isclose(id0.values, id1.values).mean() * 100} %")
    assert np.isclose(id0.values, id1.values).mean() < 1



# load cs_data
def load_normalised_cs_data(
    config: Config,
    model: BaseModel,
    test_basins: List[str],
    test_times: List[pd.Timestamp],
    final_value: bool = True,
) -> xr.Dataset:
    #  1. run the forward passes for each basin
    all_basin_data = run_all_basin_forward_passes(
        test_basins=test_basins,
        config=config,
        final_value=final_value,
        model=model,
    )

    # 3. create xarray object
    cs_data = convert_dict_to_xarray(
        all_basin_data,
        times=test_times,
        wide_format=False,
        final_value=final_value,
    )

    #  5. normalise cell state data
    norm_cs_data = normalize_xarray_cstate(cs_data)

    print("Model Overlap: ")
    check_data_not_duplicated(norm_cs_data, "cell_state")

    return norm_cs_data



if __name__ == "__main__":
    # TEST stations (13 stations in the test sample)
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

    #  1. Load in the model config file
    data_dir = Path("/cats/datastore/data/")
    assert data_dir.exists()

    run_dir = data_dir / "runs/ensemble_EALSTM/ealstm_ensemble6_nse_1998_2008_2910_030601"
    config = load_config_file(run_dir)
    model = load_ealstm(config)

    # get normalised cs_data
    TEST_BASINS = [str(id_) for id_ in catchment_ids]
    FINAL_VALUE = True
    TEST_TIMES = pd.date_range(config.test_start_date, config.test_end_date, freq="D")

    norm_cs_data = load_normalised_cs_data(
        config=config,
        model=model,
        test_basins=TEST_BASINS,
        test_times=TEST_TIMES,
        final_value=FINAL_VALUE,
    )

    # get the target SM data
    sm = read_gb_sm_data(data_dir)
