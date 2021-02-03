import xarray as xr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from src.utils import create_shape_aligned_climatology
from typing import List, Dict, Optional


def join_into_one_ds(
    lstm_preds: xr.Dataset,
    fuse_data: xr.Dataset,
    ealstm_preds: Optional[xr.Dataset] = None,
) -> xr.Dataset:
    all_preds = xr.combine_by_coords(
        [
            lstm_preds.rename({"sim": "LSTM"}),
            (
                fuse_data.rename(
                    dict(
                        zip(
                            [v for v in fuse_data.data_vars],
                            [str(v).replace("SimQ_", "") for v in fuse_data.data_vars],
                        )
                    )
                ).drop("obs")
            ),
        ]
    )
    if ealstm_preds is not None:
        all_preds = all_preds.merge(ealstm_preds.rename({"sim": "EALSTM"}).drop("obs"))

    return all_preds


def read_ensemble_results(ensemble_dir: Path) -> xr.Dataset:
    assert (
        ensemble_dir / "data_ENS.csv"
    ).exists(), "Has `scripts/multiple_forcing/read_nh_results.py` been run?"
    df = pd.read_csv(ensemble_dir / "data_ENS.csv").drop("Unnamed: 0", axis=1)
    df["time"] = pd.to_datetime(df["time"])
    preds = df.set_index(["station_id", "time"]).to_xarray()
    preds["station_id"] = [int(sid) for sid in preds["station_id"]]
    return preds


def read_ensemble_member_results(ensemble_dir: Path) -> xr.Dataset:
    """"""
    # assert False, "TODO"
    # data_dir = Path("/cats/datastore/data")
    # paths = [d for d in (data_dir / "runs/ensemble_LANE").glob("**/*.p")]
    import re
    import pickle

    paths = [d for d in (ensemble_dir).glob("**/*_results.p")]
    ps = [pickle.load(p.open("rb")) for p in paths]

    output_dict = {}
    for i, res_dict in tqdm(enumerate(ps), desc="Loading Ensemble Members"):
        stations = [k for k in res_dict.keys()]
        freq = "1D"
        all_xr_objects: List[xr.Dataset] = []

        # get the ensemble number
        m = re.search("ensemble\d+", paths[i].__str__())
        try:
            name = m.group(0)
        except AttributeError as e:
            print("found ensemble mean")
            name = "mean"

        for station_id in stations:
            #  extract the raw results
            try:
                xr_obj = (
                    res_dict[station_id][freq]["xr"].isel(time_step=0).drop("time_step")
                )
            except ValueError:
                # ensemble mode does not have "time_step" dimension
                xr_obj = res_dict[station_id][freq]["xr"].rename({"datetime": "date"})
            xr_obj = xr_obj.expand_dims({"station_id": [station_id]}).rename(
                {"date": "time"}
            )
            all_xr_objects.append(xr_obj)

        preds = xr.concat(all_xr_objects, dim="station_id")
        preds["station_id"] = [int(sid) for sid in preds["station_id"]]
        preds = preds.rename({"discharge_spec_obs": "obs", "discharge_spec_sim": "sim"})

        output_dict[name] = preds

    # return as one dataset
    all_ds = []
    for key in output_dict.keys():
        all_ds.append(
            output_dict[key].assign_coords({"member": key}).expand_dims("member")
        )
    ds = xr.concat(all_ds, dim="member")

    return ds


def fuse_to_nc(raw_fuse_path: Path, double_check: bool = True) -> xr.Dataset:
    all_paths = [
        d for d in (raw_fuse_path / "Timeseries_SimQ_Best/").glob("*_Best_Qsim.txt")
    ]

    if double_check:
        import re

        p = re.compile("\/[\d]*_")

    if not (raw_fuse_path.parents[0] / "ALL_fuse_ds.nc").exists():
        all_dfs = []
        for txt in tqdm(all_paths):
            df = pd.read_csv(txt, skiprows=3, header=0)
            df.columns = [c.rstrip().lstrip() for c in df.columns]
            df = df.rename(columns={"YYYY": "year", "MM": "month", "DD": "day"})
            df["time"] = pd.to_datetime(df[["year", "month", "day"]])

            # double check the string matching
            station_id = int(str(txt).split("/")[-1].split("_")[0])
            if double_check:
                result = p.search(str(txt))
                check_sid = int(result.group(0).replace("/", "").replace("_", ""))
                assert check_sid == station_id

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


def get_matching_stations():
    pass


def calculate_benchmarks(benchmark_calculation_ds: xr.Dataset):
    benchmark_preds = xr.Dataset()
    # 1) Persistence
    benchmark_preds["persistence"] = benchmark_calculation_ds["discharge_spec"].shift(
        time=1
    )

    #  2) DayofYear Climatology
    climatology_unit = "month"

    climatology_doy = (
        benchmark_calculation_ds["discharge_spec"].groupby("time.dayofyear").mean()
    )
    climatology_doy = create_shape_aligned_climatology(
        benchmark_calculation_ds,
        climatology_doy.to_dataset(),
        variable="discharge_spec",
        time_period="dayofyear",
    )

    climatology_mon = (
        benchmark_calculation_ds["discharge_spec"].groupby("time.month").mean()
    )
    climatology_mon = create_shape_aligned_climatology(
        benchmark_calculation_ds,
        climatology_mon.to_dataset(),
        variable="discharge_spec",
        time_period="month",
    )

    benchmark_preds["climatology_doy"] = climatology_doy["discharge_spec"]
    benchmark_preds["climatology_mon"] = climatology_mon["discharge_spec"]

    return benchmark_preds


if __name__ == "__main__":
    save = True
    import sys

    sys.path.append("/home/tommy/ml_drought")
    from scripts.drafts.delta_error import DeltaError

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

    if save:
        all_preds.to_netcdf(data_dir / "RUNOFF/all_preds.nc")
