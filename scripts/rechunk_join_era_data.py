from pathlib import Path
import xarray as xr 
import numpy as np
from distributed import Client
from dask.diagnostics import ProgressBar



if __name__ == "__main__":
    client = Client(n_workers=4, threads_per_worker=1, memory_limit='4GB', dashboard_address=':8891')
    print(client.dashboard_link)
    data_dir = Path("/DataDrive200/data")

    # Load the ERA5 data
    chunks = {"time": 1000, "lat": 10, "lon": 10}
    files = [f for f in (data_dir / "interim/reanalysis-era5-land_preprocessed/").glob("*_kenya.nc")]
    era_data = [xr.open_dataset(f, chunks=chunks) for f in files]

    # get reference lats, lons
    pcp = era_data[int(np.argwhere(["precip" in f.name for f in files]))]
    lats = pcp.lat
    lons = pcp.lon

    # resample all by time [2001--2019] and select same lat, lons [kenya bounding box]
    daily_era_data = []
    for ix, f in enumerate(era_data):
        print(f.data_vars)
        f = f.sel(time=slice("2001-01-01", "2019-01-30"), lat=lats, lon=lons)
        daily_era_data.append(f.resample(time="D").mean(dim="time"))

    # merge data
    era_d = xr.merge(daily_era_data)

    # ensure consistent chunks
    era_d["pev"] = era_d["pev"].chunk(era_d["tp"].chunks)

    print(era_d)
    print()

    # save
    if ZARR:
        if (data_dir / "era_zarr").exists():
            import shutil
            shutil.rmtree((data_dir / "era_zarr"))

        (data_dir / "era_zarr").mkdir()
        print("Saving to zarr")
        era_d.to_zarr((data_dir / "era_zarr"))
    else:
        era_paths = [data_dir / "interim/reanalysis-era5-land_preprocessed" / f"{var}_kenya.nc" for var in era_d.data_vars]
        era_ds = [era_d[var].to_dataset() for var in era_d.data_vars]

        xr.save_mfdataset(   
            era_ds,
            era_paths
        )