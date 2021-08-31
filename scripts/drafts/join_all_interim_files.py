from pathlib import Path
from dask.diagnostics import ProgressBar
import xarray as xr
from scripts.utils import get_data_path



if __name__ == "__main__":
    NUM_WORKERS = 2
    data_dir = get_data_path()
    interim_path = data_dir / "interim"
    folder_to_join: str = "reanalysis-era5-land_preprocessed"
    out_path = data_dir / "kenya_era5_land_hourly.nc"

    assert out_path.exists()
    assert (interim_path / folder_to_join).exists()

    # join all the files in the interim folder
    interim_files = sorted(list((interim_path / folder_to_join).glob("*.nc")))
    ds = xr.open_mfdataset(interim_files)
    delayed_obj = ds.to_netcdf(out_path, compute=False)

    with ProgressBar():
        print("\n*** Writing the joined file to netcdf ***\n")
        delayed_obj.compute(num_workers=NUM_WORKERS)