import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


def read_ensemble_results(ensemble_dir: Path) -> xr.Dataset:
    assert (
        ensemble_dir / "data_ENS.csv"
    ).exists(), "Has `scripts/multiple_forcing/read_nh_results.py` been run?"
    df = pd.read_csv(ensemble_dir / "data_ENS.csv").drop("Unnamed: 0", axis=1)
    df["time"] = pd.to_datetime(df["time"])
    preds = df.set_index(["station_id", "time"]).to_xarray()
    preds["station_id"] = [int(sid) for sid in preds["station_id"]]
    return preds


def read_ensemble_member_results(ensemble_dir: Path, ensemble_int: int) -> xr.Dataset:
    assert False, "TODO"
    ensemble_dir.glob(f"*{ensemble_int}*")
    preds = None

    preds["station_id"] = [int(sid) for sid in preds["station_id"]]
    preds = preds.rename({"discharge_spec_obs": "obs", "discharge_spec_sim": "sim"})
    return preds


def fuse_to_nc(raw_fuse_path: Path) -> xr.Dataset:
    all_paths = [
        d for d in (raw_fuse_path / "Timeseries_SimQ_Best/").glob("*_Best_Qsim.txt")
    ]

    if not (raw_fuse_path.parents[0] / "ALL_fuse_ds.nc").exists():
        all_dfs = []
        for txt in tqdm(all_paths):
            df = pd.read_csv(txt, skiprows=3, header=0)
            df.columns = [c.rstrip().lstrip() for c in df.columns]
            df = df.rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
            df["time"] = pd.to_datetime(df[["year", "month", "day"]])
            station_id = int(str(txt).split("/")[-1].split("_")[0])
            df["station_id"] = [station_id for _ in range(len(df))]
            df = df.drop(["year", "month", "day", "HH"], axis=1).set_index(
                ["station_id", "time"]
            )
            all_dfs.append(df)

        fuse_ds = pd.concat(all_dfs).to_xarray()
        fuse_ds.to_netcdf(raw_fuse_path.parents[0] / "ALL_fuse_ds.nc")

    else:
        fuse_ds = xr.open_dataset(raw_fuse_path.parents[0] / "ALL_fuse_ds.nc")
    return fuse_ds


def read_fuse_data(raw_fuse_path: Path, obs: xr.Dataset) -> xr.Dataset:
    fuse_ds = fuse_to_nc(raw_fuse_path)
    # join with observations for stations that exist
    fuse_data = fuse_ds.sel(
        station_id=np.isin(fuse_ds.station_id, obs.station_id)
    ).merge(obs.sel(station_id=np.isin(obs.station_id, fuse_ds.station_id)))
    return fuse_data


if __name__ == "__main__":
    import sys

    sys.path.append("/home/tommy/ml_drought")
    from scripts.drafts.calculate_error_scores import DeltaError

    data_dir = Path("/cats/datastore/data")

    # ealstm_ensemble_dir = data_dir / "runs/ensemble_EALSTM"
    # ealstm_preds = read_ensemble_results(ealstm_ensemble_dir)
    pet_ealstm_ensemble_dir = data_dir / "runs/ensemble_pet_ealstm"
    ealstm_preds = read_ensemble_results(pet_ealstm_ensemble_dir)

    # lstm_ensemble_dir = data_dir / "runs/ensemble"
    lstm_ensemble_dir = data_dir / "runs/ensemble_pet"
    lstm_preds = read_ensemble_results(lstm_ensemble_dir)

    #  fuse data
    raw_fuse_path = data_dir / "RUNOFF/FUSE"
    fuse_data = read_fuse_data(raw_fuse_path, lstm_preds["obs"])

    # get matching stations
    all_stations_lstm = np.isin(lstm_preds.station_id, fuse_data.station_id)
    all_stations_ealstm = np.isin(ealstm_preds.station_id, fuse_data.station_id)
    lstm_preds = lstm_preds.sel(
        station_id=all_stations_lstm, time=np.isin(lstm_preds.time, fuse_data.time)
    )
    ealstm_preds = ealstm_preds.sel(
        station_id=all_stations_ealstm, time=np.isin(ealstm_preds.time, fuse_data.time)
    )

    # calculate all error metrics (including benchmarks)
    ds = xr.open_dataset(data_dir / "RUNOFF/ALL_dynamic_ds.nc")
    ds["station_id"] = ds["station_id"].astype(int)
    processor = DeltaError(
        ealstm_preds,
        lstm_preds,
        fuse_data,
        benchmark_calculation_ds=ds[["discharge_spec"]],
        incl_benchmarks=True,
    )
    all_preds = processor.all_preds
    print(all_preds)

    # calculate all error metrics (excluding benchmarks)
    processor = DeltaError(ealstm_preds, lstm_preds, fuse_data, incl_benchmarks=False)
    all_preds = processor.all_preds
    print(all_preds)
