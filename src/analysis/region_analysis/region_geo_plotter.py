from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Dict, Optional, Tuple
from collections import namedtuple

from src.analysis.region_analysis.groupby_region import (
    KenyaGroupbyRegion,
    GroupbyRegion,
)

gpd = None
GeoDataFrame = None


AdminLevelGeoDF = namedtuple("AdminLevelGeoDF", ["gdf", "gdf_colname", "admin_name"])


PlotMetric = namedtuple("PlotMetric", ["metric", "cmap", "vmin", "vmax"])


class RegionGeoPlotter:
    def __init__(
        self, data_folder: Path = Path("data"), country: str = "kenya"
    ) -> None:

        self.data_folder = data_folder
        # try and import geopandas
        global gpd
        if gpd is None:
            print("The RegionGeoPlotter requires `geopandas` to be installed.")
            import geopandas as gpd

        global GeoDataFrame
        if GeoDataFrame is None:
            from geopandas.geodataframe import GeoDataFrame

        self.country = country

        self.country_region_grouper: GroupbyRegion = self.get_groupby_region(
            country_str=country
        )

        # Setup type hints without assignment
        self.region_gdfs: Dict[str, AdminLevelGeoDF] = {}

        self.gdf: Optional[GeoDataFrame] = None  # type: ignore
        self.all_gdfs: List = []

    def get_groupby_region(self, country_str: str) -> GroupbyRegion:
        # if need a new country boundaries add example here
        country_lookup = {"kenya": KenyaGroupbyRegion}
        keys = [k for k in country_lookup.keys()]
        assert country_str in keys, (
            "Expect " f"the `country_str` argument to be one of: {keys}"
        )

        return country_lookup[country_str](self.data_folder)

    def read_shapefiles(self):
        # read the shapefile to GeoDataFrames for each region
        if self.country == "kenya":
            levels = [
                "level_1",
                "level_2",
                "level_3",
                "level_3_wards",
                "level_4",
                "level_5",
            ]

            region_gdfs = {}

            for level in levels:
                admin_boundary = self.country_region_grouper.get_admin_level(
                    selection=level
                )
                path = admin_boundary.shp_filepath
                if not path.exists():
                    print(
                        f"{admin_boundary.shp_filepath} not found."
                        " Moving to next file"
                    )
                    continue

                print(f"Reading file: {path.name}")
                gdf = gpd.read_file(path)
                key = f"{admin_boundary.var_name}_{self.country}"
                region_gdfs[key] = AdminLevelGeoDF(
                    gdf=gdf, gdf_colname=admin_boundary.lookup_colname, admin_name=key
                )

            self.region_gdfs = region_gdfs
            print("* Read shapefiles and stored in `RegionGeoPlotter.region_gdfs` *")

        else:
            raise NotImplementedError

    def join_model_performances_to_geometry(
        self, model_performance_df: pd.DataFrame, admin_name: str
    ) -> GeoDataFrame:  # type: ignore
        """Join the `geometry` column from the shapefile read in as GeoDataFrame
        to the model performance metrics in model_performance_df. Required to
        make spatial plots of data.

        Arguments:
        ---------
        model_performance_df: pd.DataFrame
            the data showing the model performance for each

        admin_name: str
            the name of the administrative units (shapefile name) stored in
            `self.region_gdfs.keys()`
        """
        assert admin_name in [k for k in self.region_gdfs.keys()], (
            "Invalid "
            f"`admin_name`. Expected one of: {[k for k in self.region_gdfs.keys()]}"
            f" Got: {admin_name}"
        )
        gdf = self.region_gdfs[admin_name].gdf
        gdf_colname = self.region_gdfs[admin_name].gdf_colname
        gdf[gdf_colname] = gdf[gdf_colname].apply(str.rstrip).apply(str.lstrip)

        df_colname = "region_name"
        out_gdf = GeoDataFrame(  # type: ignore
            pd.merge(
                model_performance_df,
                gdf[[gdf_colname, "geometry"]],
                left_on=df_colname,
                right_on=gdf_colname,
            )
        )
        return out_gdf

    def merge_all_model_performances_gdfs(
        self, all_models_df: pd.DataFrame
    ) -> GeoDataFrame:  # type: ignore
        all_gdfs: List[GeoDataFrame] = []  # type: ignore
        assert "admin_level_name" in all_models_df.columns, (
            f"Expect to find admin_region" f"in {all_models_df.columns}"
        )

        # join the geometry columns to make GeoDataFrames
        for admin_name in all_models_df.admin_level_name.unique():
            admin_level_df = all_models_df.loc[
                all_models_df.admin_level_name == admin_name
            ]
            all_gdfs.append(
                self.join_model_performances_to_geometry(
                    model_performance_df=admin_level_df, admin_name=admin_name
                )
            )

        self.gdf = pd.concat(all_gdfs)

        # convert mean model outputs to float
        try:
            self.gdf = self.gdf.astype(  # type: ignore
                {"predicted_mean_value": "float64", "true_mean_value": "float64"}
            )
        except KeyError:
            self.gdf = self.gdf.astype(  # type: ignore
                {"rmse": "float64", "mae": "float64", "r2": "float64"}
            )
        print("* Assigned the complete GeoDataFrame to `RegionGeoPlotter.gdf`")

        if not isinstance(self.gdf, GeoDataFrame):  # type: ignore
            self.gdf = GeoDataFrame(self.gdf)  # type: ignore

        return self.gdf

    @staticmethod
    def get_metric(
        selection: str, gdf: GeoDataFrame, **kwargs  # type: ignore
    ) -> PlotMetric:  # type: ignore
        rmse_vmin = kwargs["rmse_vmin"] if "rmse_vmin" in kwargs else None
        rmse_vmax = (
            kwargs["rmse_vmax"]
            if "rmse_vmax" in kwargs
            else np.nanpercentile(gdf.rmse, q=85)  # type: ignore
        )
        rmse = PlotMetric(metric="rmse", cmap="viridis", vmin=rmse_vmin, vmax=rmse_vmax)
        mae_vmin = kwargs["mae_vmin"] if "mae_vmin" in kwargs else None
        mae_vmax = (
            kwargs["mae_vmax"]
            if "mae_vmax" in kwargs
            else np.nanpercentile(gdf.mae, q=85)  # type: ignore
        )
        mae = PlotMetric(metric="mae", cmap="plasma", vmin=mae_vmin, vmax=mae_vmax)
        r2 = PlotMetric(metric="r2", cmap="inferno_r", vmin=0, vmax=1.0)
        lookup = {"rmse": rmse, "mae": mae, "r2": r2}

        assert selection in [k for k in lookup.keys()], (
            "selection should be one of:" f"{[k for k in lookup.keys()]}"
        )

        return lookup[selection]

    @staticmethod
    def plot_metric(
        ax: Axes, metric: PlotMetric, gdf: GeoDataFrame  # type: ignore
    ) -> Axes:
        # nicely format the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        gdf.plot(  # type: ignore
            metric.metric,
            ax=ax,
            legend=True,
            cmap=metric.cmap,
            vmin=metric.vmin,
            vmax=metric.vmax,
            cax=cax,
        )
        ax.set_title(f"{metric.metric.upper()}")
        return ax

    def plot_all_regional_error_metrics(
        self,
        gdf: GeoDataFrame,  # type: ignore
        title: str = "",
        **kwargs: Dict,
    ) -> Tuple[Figure, List[Axes]]:
        """Plot area-based maps of the scores"""
        assert np.isin(["rmse", "mae", "r2"], gdf.columns).all()  # type: ignore
        gdf = gdf.dropna(subset=["rmse", "mae", "r2"])  # type: ignore

        # get the PlotMetric objects
        rmse = self.get_metric("rmse", gdf, **kwargs)
        mae = self.get_metric("mae", gdf, **kwargs)
        r2 = self.get_metric("r2", gdf, **kwargs)

        # build multi-axis plot
        fig, axs = plt.subplots(1, 3, figsize=(12, 8))
        for i, metric in enumerate([rmse, mae, r2]):
            ax = axs[i]
            ax = self.plot_metric(gdf=gdf, ax=ax, metric=metric)
            ax.axis("off")

        fig.suptitle(title)
        return fig, axs

    def plot_regional_error_metric(
        self,
        gdf: GeoDataFrame,  # type: ignore
        selection: str,
        **kwargs: Dict,
    ) -> Tuple[Figure, Axes]:
        valid_metrics = ["rmse", "mae", "r2"]
        assert selection in valid_metrics, (
            "Expecting selection" f" to be one of: {valid_metrics}"
        )
        gdf = gdf.dropna(subset=valid_metrics)  # type: ignore
        metric = self.get_metric(selection, gdf, **kwargs)
        fig, ax = plt.subplots()
        ax = self.plot_metric(gdf=gdf, ax=ax, metric=metric)

        return fig, ax
