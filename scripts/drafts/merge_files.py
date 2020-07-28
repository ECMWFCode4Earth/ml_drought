import xarray as xr
import sys

sys.path.append("../..")

from scripts.utils import get_data_path
from src.preprocess import ERA5LandPreprocessor


if __name__ == "__main__":
    # TODO: generalize for any dataset (use a lookup dict)
    data_dir = get_data_path()
    processor = ERA5LandPreprocessor(data_dir)

    # get variable and subset_str
    paths = list((data_dir / "interim/reanalysis-era5-land_interim").glob("*.nc"))
    subset_str = str(paths[0].name).split("_")[-1].replace(".nc", "")
    variable = str(paths[0].name).split("01_12_")[-1].replace(f"{subset_str}.nc", "")

    filename = f'{variable}_data{"_" + subset_str if subset_str is not None else ""}.nc'
    processor.merge_files(subset_str=subset_str, resample_time="M", filename=filename)
