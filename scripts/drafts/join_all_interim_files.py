from pathlib import Path
from dask.diagnostics import ProgressBar
import xarray as xr
from scripts.utils import get_data_path



if __name__ == "__main__":
    NUM_WORKERS = 2
    data_dir = get_data_path()
    interim_path = data_dir / "interim"
    folder_to_join: str = "reanalysis-era5-land_preprocessed"
    out_path = data_dir / "ALL_kenya_era5_land_daily.nc"

    assert out_path.parent.exists()
    assert (interim_path / folder_to_join).exists()

    # join all the files in the interim folder
    interim_files = sorted(list((interim_path / folder_to_join).glob("*.nc")))
    ds = xr.open_mfdataset(interim_files)

    # process data somehow ... 
    # maybe take the mean of the values for each day?
    ds = ds.resample(time="D").mean()

    delayed_obj = ds.to_netcdf(out_path, compute=False)

    with ProgressBar():
        print(f"\n*** Writing the joined file to netcdf: {out_path} ***\n")
        delayed_obj.compute(num_workers=NUM_WORKERS)