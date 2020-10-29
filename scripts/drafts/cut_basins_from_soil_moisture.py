import sys
sys.path.append("../..")

import pandas as pd
from pathlib import Path
import xarray as xr
from scripts.utils import get_data_path


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data")

    sm_path = data_dir / "RUNOFF/gb_soil_moisture_2000_2020.nc"
    shp_path = data_dir / "CAMELS_GB_DATASET/Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"

    ds = xr.open_dataset(sm_path)
    gr = GroupbyRegion(data_dir, country=shp_path.as_posix())

