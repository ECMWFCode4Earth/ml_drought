import xarray as xr
import numpy as np
from datetime import datetime
from pathlib import Path

from dask.diagnostics import ProgressBar
import dask 

import sys; sys.path.append("/home/leest/ml_drought")
from scripts.utils import get_data_path
from src.preprocess.dekad_utils import runningdekad2date
from src.preprocess.base import BasePreProcessor
from src.utils import Region, region_lookup


def add_time_dim(xda: xr.Dataset) -> xr.Dataset:
    # https://stackoverflow.com/a/65416801/9940782 
    xda = xda.expand_dims(time = [datetime.now()])
    return xda


def chop_roi(data: xr.Dataset, region_str: str = "kenya") -> xr.Dataset:
    region = region_lookup[region_str]
    latmin, latmax, lonmin, lonmax = (
        region.latmin,
        region.latmax,
        region.lonmin,
        region.lonmax,
    )
    inverse_lat=False
    inverse_lon=False

    return data.sel(
        lat=slice(latmax, latmin) if inverse_lat else slice(latmin, latmax),
        lon=slice(lonmax, lonmin) if inverse_lon else slice(lonmin, lonmax),
    )


def get_time_from_filestr(file_str: str) -> datetime:
    time = file_str.split(".")[1].replace("t", "")
    year, dekad = int(time[:4]), int(time[4:])
    dtime = runningdekad2date(year, dekad)
    return dtime


if __name__ == "__main__":
    # setup dask client 
    from distributed import Client
    # dask.config.set(scheduler="processes")
    # client = Client()
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB', dashboard_address=':8891')


    data_dir = Path("/DataDrive200/data") ## get_data_path()
    # data_dir = get_data_path()
    subset_str = "kenya"

    raw_folder = data_dir / "raw/modis_ndvi_1000"
    
    files = [f for f in raw_folder.glob("*.nc")]
    data = xr.open_mfdataset(files, preprocess=add_time_dim)

    # create time dimension from dekads 
    times = [get_time_from_filestr(f.as_posix()) for f in files]
    data["time"] = times

    # (lat: 5600, lon: 4480) -> (lat: 1255, lon: 983)
    # chop roi (16G -> ~8GB)
    data = chop_roi(data)

    # fix data encodings (remove fill values)
    # (0.0048 • DN) - 0.2
    data = data.where((data["modis_ndvi"] <= 250))
    data = (0.0048 * data) - 0.200

    # rechunk into per-pixel chunks {"time": len(times), "lat": 1, "lon": 1}
    data = data.chunk({"time": len(times), "lat": 10, "lon": 10})

    # # save to disk 
    # 1) save to zarr 
    zarr_dir = (data_dir / "zarr")
    if zarr_dir.exists():
        zarr_dir = zarr_dir.parent / zarr_dir.name + "_" + f"{np.random.randint(0, 100):}"
    else:
        zarr_dir.mkdir(exist_ok=True)

    data.to_zarr(zarr_dir)

    # open from zarr
    ds = xr.open_zarr(data_dir / "zarr")

    out_folder = data_dir / "interim/modis_ndvi_1000_preprocessed"
    out_folder.mkdir(exist_ok=True)
    out_file = out_folder / f"modis_ndvi_1000_{subset_str}.nc"
    
    print(f"Writing to {out_file}")
    ds = ds.compute()
    ds.to_netcdf(out_file)
