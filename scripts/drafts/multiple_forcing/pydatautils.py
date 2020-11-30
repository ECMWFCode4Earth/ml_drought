"""
ipython --pdb main.py evaluate -- --run_dir  /cats/datastore/data/runs/lstm_ALL_vars_2004_2210_1035/
ipython --pdb analysis/datautils.py --run_dir /cats/datastore/data/runs/lstm_less_vars_2004_1507_1028
"""
import xarray as xr
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from typing import Tuple, Dict, Optional


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--epoch", type=int, default=None)
    args = vars(parser.parse_args())

    return args


def get_validation_data(run_dir: Path, epoch: int = None) -> Tuple[xr.Dataset, int]:
    # open validation Dict
    val_paths = [d for d in (run_dir / "test").glob("*")]
    if epoch is None:
        val_path = max(val_paths)
        epoch = int(val_path.name[-3:])
    else:
        val_path = [path for path in val_paths if int(path.name[-3:]) == epoch]
        assert (
            val_path != []
        ), f"Epoch: {epoch} has not been validated. Not in: {val_paths}"
        assert len(val_path) == 1, "Expect one val path to match epoch number"
        val_path = val_path[0]

    val_results: Dict = pickle.load(open(val_path / "test_results.p", "rb"))

    valid_ds = create_validation_dataset(val_results)
    return valid_ds, epoch


def get_train_data(run_dir: Path) -> Tuple[xr.Dataset, ...]:
    # open train Dict
    train_path = [d for d in (run_dir / "train_data").glob("*.p")][0]
    train_data: Dict = pickle.load(open(train_path, "rb"))
    train_ds = create_training_dataset(train_data)

    return train_ds


def create_training_dataset(train_data: Dict) -> xr.Dataset:
    # CREATE TRAINING dataset
    basins = train_data["coords"]["basin"]["data"]
    time = train_data["coords"]["date"]["data"]

    coords = {"station_id": basins, "time": time}
    dims = ("station_id", "time")

    # create the xarray data
    data = {
        variable: (dims, train_data["data_vars"][variable]["data"])
        for variable in [v for v in train_data["data_vars"].keys()]
    }
    train_ds = xr.Dataset(data, coords=coords)

    return train_ds


def create_validation_dataset(val_results: Dict) -> xr.Dataset:
    # create VALIDATION dataset

    station_ids = [stn for stn in val_results.keys()]

    discharge_spec_obs_ALL = []
    discharge_spec_sim_ALL = []

    for stn in station_ids:
        discharge_spec_obs_ALL.append(
            val_results[stn]["xr"]["discharge_spec_obs"].values.flatten()
        )
        discharge_spec_sim_ALL.append(
            val_results[stn]["xr"]["discharge_spec_sim"].values.flatten()
        )

    times = val_results[stn]["xr"]["date"].values
    obs = np.vstack(discharge_spec_obs_ALL)
    sim = np.vstack(discharge_spec_sim_ALL)

    assert obs.shape == sim.shape

    # create xarray object
    coords = {"time": times, "station_id": station_ids}
    data = {"obs": (["station_id", "time"], obs), "sim": (["station_id", "time"], sim)}
    valid_ds = xr.Dataset(data, coords=coords)

    return valid_ds


def create_results_csv(run_dir, epoch: Optional[int] = None) -> pd.DataFrame:

    valid_ds, epoch = get_validation_data(run_dir, epoch=epoch)

    # save to netcdf
    # train_ds.to_netcdf(run_dir / 'train_ds.nc')
    valid_ds.to_netcdf(run_dir / "valid_ds.nc")
    outfile = run_dir / f"results_{run_dir.name}_E{epoch:03}.csv"
    valid_df = valid_ds.to_dataframe()
    valid_df.to_csv(outfile)

    print(f"Results written to {outfile}")

    return valid_df


if __name__ == "__main__":
    args = get_args()
    run_dir = Path(args["run_dir"])
    epoch = args["epoch"]
    assert run_dir.exists()
    assert (epoch is None) or (isinstance(epoch, int))

    create_results_csv(run_dir, epoch=epoch)
