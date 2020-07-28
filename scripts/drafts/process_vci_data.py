import xarray as xr
import sys

sys.path.append("../..")

from scripts.utils import get_data_path
from src.preprocess.base import BasePreProcessor


if __name__ == "__main__":
    data_dir = get_data_path()

    vci = xr.open_dataset(data_dir / "interim/VCI_preprocessed/data_india.nc")
    regrid_ds = xr.open_dataset(
        data_dir / "interim/reanalysis-era5-land_preprocessed/data_india.nc"
    )

    print("** Begin Regridding **")
    processor = BasePreProcessor(data_dir)
    vci = processor.regrid(ds=vci, reference_ds=regrid_ds)

    print("** Saving file **")
    vci.to_netcdf(data_dir / "interim/VCI_preprocessed/regrid_data_india.nc")
