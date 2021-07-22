import sys

sys.path.append("../..")

import pandas as pd
from pathlib import Path
import xarray as xr
import geopandas as gpd


if __name__ == "__main__":
    data_dir = Path("/cats/datastore/data")

    sm_path = data_dir / "gb_soil_moisture.nc"
    shp_path = (
        data_dir
        / "CAMELS_GB_DATASET/Catchment_Boundaries/CAMELS_GB_catchment_boundaries.shp"
    )

    gdf = gpd.read_file(shp_path)
    # gdf.crs = {'init': 'epsg:27700'}
    # gdf.crs = {'init': 'epsg:4326'}

    da = xr.open_dataset(sm_path)["swvl1"]
    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da.rio.write_crs("epsg:4326", inplace=True)

    # calculate all basin timeseries
    for index, row in gdf.iterrows():
        id_ = row["ID_STRING"]
        geom = row["geometry"]
        out = da.rio.clip(geom, gdf.crs, drop=False)
        out.assign_coords(station_id=id_)
    # converter = SHPtoXarray()
    # shp_xr = converter.shapefile_to_xarray(
    #     da, shp_path, var_name="station_id", lookup_colname="ID_STRING"
    # )
