from pathlib import Path
import pandas as pd
import xarray as xr
from tqdm import tqdm


def fuse_to_nc(raw_fuse_path: Path) -> xr.Dataset:
    all_paths = [
        d for d in (raw_fuse_path / "Timeseries_SimQ_Best/").glob("*_Best_Qsim.txt")
    ]

    if not (raw_fuse_path.parents[0] / "ALL_fuse_ds.nc").exists():
        all_dfs = []
        for txt in tqdm(all_paths):
            df = pd.read_csv(txt, skiprows=3, header=0)
            df.columns = [c.rstrip().lstrip() for c in df.columns]
            df = df.rename(
                columns={"YYYY": "year", "MM": "month", "DD": "day"})
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
        fuse_ds = xr.open_dataset(
            raw_fuse_path.parents[0] / "ALL_fuse_ds.nc")
    return fuse_ds


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data/")
    assert data_dir.exists()

    raw_fuse_path = data_dir / "RUNOFF/FUSE"
    ds = fuse_to_nc(raw_fuse_path)
