import xarray as xr
from tqdm import tqdm
import pandas as pd
from pathlib import Path


def read_gb_sm_data(data_dir: Path) -> xr.Dataset:
    if not (data_dir / "RUNOFF/gb_sm_catchments_1993_2020.nc").exists():
        all_sm_ds = []
        for ix, path in enumerate(
            tqdm(
                list((data_dir / "RUNOFF/sm_data").glob("*Level*.csv")),
                desc="Reading SM Level",
            )
        ):
            # read in data
            df = (
                pd.read_table(d, sep=";", decimal=",")
                .drop("Unnamed: 0", axis=1)
                .rename({"Date": "time"}, axis=1)
                .astype({"time": "datetime64[ns]"})
            )
            # create index from station, time, rename column to soil level volume
            df = (
                df.melt(id_vars="time")
                .astype({"value": "float64", "variable": "int64"})
                .rename(
                    {
                        "value": f'swvl{path.name.split("Level_")[-1].replace(".csv", "")}',
                        "variable": "station_id",
                    },
                    axis=1,
                )
                .set_index(["time", "station_id"])
            )

            # convert to xarray
            ds = df.to_xarray()
            all_sm_ds.append(ds)

        ds = xr.combine_by_coords(all_sm_ds)
        ds.to_netcdf(data_dir / "RUNOFF/gb_sm_catchments_1993_2020.nc")

    else:
        ds = xr.open_dataset(data_dir / "RUNOFF/gb_sm_catchments_1993_2020.nc")

    return ds


def old_read_obs_sm() -> xr.Dataset:
    # read in data
    obs_sm = pd.read_table(
        data_dir / "RUNOFF/Soil_Moisture_Catchments.csv", sep=";", decimal=","
    ).drop("Unnamed: 0", axis=1)

    # convert to xarray
    sm_df = (
        obs_sm.set_index("Date")
        .stack()
        .reset_index()
        .rename({0: "soil_moisture", "level_1": "station_id", "Date": "time"}, axis=1)
    )
    sm_df["time"] = pd.to_datetime(sm_df["time"])
    sm_df = sm_df.sort_values(["station_id", "time"]).set_index(["time", "station_id"])
    sm = sm_df.to_xarray()

    return sm


if __name__ == "__main__":
    #   read in soil moisture data as xarray Dataset
    sm = read_gb_sm_data()
