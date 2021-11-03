from pathlib import Path
import xarray as xr 
from scripts.drafts.regridding import convert_to_same_grid


if __name__ == "__main__":
    data_dir = Path("/DataDrive200/data")

    # Load the data
    era = xr.open_dataset(data_dir / "kenya_era5_land_daily.nc")[["tp"]]
    ds = xr.open_dataset(data_dir / "kenya_HIRES.nc")

    regrid_ds = convert_to_same_grid(era, ds)
    regrid_ds.to_netcdf(data_dir / "modis_vci_era5_grid.nc")