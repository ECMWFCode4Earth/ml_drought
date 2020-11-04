import sys
sys.path.append("../..")

import pandas as pd
from pathlib import Path
import xarray as xr
from scripts.utils import get_data_path
from src.preprocess.utils import SHPtoXarray


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data")

    sm_path = data_dir / "RUNOFF/gb_soil_moisture_2000_2020.nc"
    shp_path = data_dir / "CAMELS_GB_DATASET/Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"

    da = xr.open_dataset(sm_path)["swvl1"]

    converter = SHPtoXarray()
    shp_xr = converter.shapefile_to_xarray(da, shp_path, var_name="station_id", lookup_colname="ID_STRING")
