import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial
from typing import Dict, DefaultDict, Tuple, List, Optional
import geopandas as gpd


region_abbrs = {
    "Western Scotland": "WS",
    "Eastern Scotland": "ES",
    "North-east England": "NEE",
    "Severn-Trent": "ST",
    "Anglian": "ANG",
    "Southern England": "SE",
    "South-west England & South Wales": "SWESW",
    "North-west England & North Wales (NWENW)": "NWENW",
}


def get_region_station_within(
    stations: gpd.GeoDataFrame, hydro_regions: gpd.GeoDataFrame
) -> pd.Series:
    #  find the region that a station belongs WITHIN
    #  create a list of strings/nans for each station
    region_dict = {}
    for region, geom in zip(hydro_regions["NAME"], hydro_regions["geometry"]):
        isin_region = [p.within(geom) for p in stations]
        region_dict[region] = [(region if item else np.nan) for item in isin_region]

    region_cols = pd.DataFrame(region_dict)

    #  copy non-null values from the right column into the left and select left
    #  https://stackoverflow.com/a/49908660
    regions_list = (region_cols.bfill(axis=1).iloc[:, 0]).rename("region")
    regions_list.index = points.index

    return regions_list


def assign_region_to_dataframe(
    results_dict: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    for key in results_dict.keys():
        df = results_dict[key]
        try:
            df = df.join(regions_list)
            df["region_abbr"] = df["region"].map(region_abbrs)
        except ValueError:
            "Already applied the 'region' column"
        results_dict[key] = df
    return results_dict


def _get_latlon_points(static: Optional[xr.Dataset] = None):
    if static is None:
        data_dir = Path("/cats/datastore/data")
        all_static = xr.open_dataset(data_dir / f"RUNOFF/interim/static/data.nc")
        all_static["station_id"] = all_static["station_id"].astype(int)
        static = all_static

    return static[["gauge_lat", "gauge_lon"]].to_dataframe()


if __name__ == "__main__":
    save: bool = True
    data_dir = Path("/cats/datastore/data")
    # load static data
    all_static = xr.open_dataset(data_dir / f"RUNOFF/interim/static/data.nc")
    all_static["station_id"] = all_static["station_id"].astype(int)
    static = all_static

    # create lat-lon station points
    d = static[["gauge_lat", "gauge_lon"]].to_dataframe()
    points = gpd.GeoSeries(
        gpd.points_from_xy(d["gauge_lon"], d["gauge_lat"]), index=d.index
    )
    points.name = "geometry"

    # get station names (strs)
    names = static["gauge_name"].to_dataframe()
    pts = gpd.GeoDataFrame(points).join(names)

    # read hydroregions
    hydro_regions = gpd.read_file(
        data_dir
        / "RUNOFF/gis_data_Tommy"
        / "UK_hydroclimate_regions_Harrigan_et_al_2018/UK_Hydro_Regions_ESP_HESS.shp"
    ).to_crs(epsg=4326)
    hydro_regions = hydro_regions.loc[
        ~np.isin(hydro_regions["NAME"], ["Northern Ireland", "Republic of Ireland"])
    ]

    #  join the region as a column to the points GeoDataFrame
    regions_list: pd.Series = get_region_station_within(points, hydro_regions)
    all_points = gpd.GeoDataFrame(points).join(names).join(regions_list)

    if save:
        import pickle

        pickle.dump(regions_list, (data_dir / "RUNOFF/regions_list.pkl").open("wb"))
        pickle.dump(
            all_points, (data_dir / "RUNOFF/all_points.pkl").open("wb")
        )

    #  assign hydro-regions as columns to error metrics
    if False:
        region_metrics = assign_region_to_dataframe(all_metrics)
        region_errors = assign_region_to_dataframe(all_errors)

        region_season_errors = defaultdict(dict)
        for season in ["DJF", "MAM", "JJA", "SON"]:
            region_season_errors[season] = assign_region_to_dataframe(
                seasonal_errors[season]
            )

        region_season_metrics = defaultdict(dict)
        for season in ["DJF", "MAM", "JJA", "SON"]:
            region_season_metrics[season] = assign_region_to_dataframe(
                seasonal_metrics[season]
            )