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


def upsample_xarray(
    ds: xr.Dataset,
    y_max: float,
    y_min: float,
    x_max: float,
    x_min: float,
    _lat_buffer: float,
    _lon_buffer: float,
    grid_factor: float = 30,
) -> xr.Dataset:
    """[summary]

    I don't use any interpolation method here(see method="zero"), which means that the new cells
    get the exact same value as the original cell they are in.

    Args:
        ds (xr.Dataset): ds is the grid data set(e.g. here ECMWF ERA5) with coordinate names "latitude" and "longitude"
        y_max, y_min, x_max, x_min are the min/max extends in lat/lon direction
        y_max (float): [description]
        y_min (float): [description]
        x_max (float): [description]
        x_min (float): [description]
        self._lat_buffer and self._lon_buffer are, as the name suggests, buffer that are used to crop the basin with some buffer
            at all sides. Be careful with the signs, they might need to be adapted depending on where you are
            (northern, southern hemisphere, and west/east of Greenwhich).
        _lat_buffer (float): [description]
        _lon_buffer (float): [description]
        self.grid_factor is dividing factor for a single grid cell in the original data set.
            E.g. to devide a single ERA5 cell(~30km x 30km) into roughly 1km x 1km grid cells, use a grid factor of 30.
        grid_factor (float, optional): [description]. Defaults to 30.

    Returns:
        xr.Dataset: [description]
    """
    # crop corresponding part from the xarray
    xr_basin = ds.sel(
        latitude=slice(y_max + self._lat_buffer, y_min - self._lat_buffer),
        longitude=slice(x_min - self._lon_buffer, x_max + self._lon_buffer),
    )

    # create finer grid of lat/long coordinates used for regridding
    new_lon = np.arange(
        xr_basin.longitude[0],
        xr_basin.longitude[-1],
        (xr_basin.longitude[1] - xr_basin.longitude[0]) / self.grid_factor,
    )
    new_lat = np.arange(
        xr_basin.latitude[0],
        xr_basin.latitude[-1],
        (xr_basin.latitude[1] - xr_basin.latitude[0]) / self.grid_factor,
    )

    # regrid xarray to finer grid and fill cells without interpolation
    xr_regrid = xr_basin.interp(latitude=new_lat, longitude=new_lon, method="zero")

    return xr_regrid


if __name__ == "__main__":
    #   read in soil moisture data as xarray Dataset
    sm = read_gb_sm_data()
