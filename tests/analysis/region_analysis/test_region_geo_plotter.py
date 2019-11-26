import pytest
import numpy as np
import pandas as pd

from src.analysis.region_analysis import RegionGeoPlotter
from src.analysis.region_analysis.groupby_region import KenyaGroupbyRegion
from tests.utils import CreateSHPFile


class TestRegionGeoPlotter:
    @staticmethod
    def make_performance_dataframe():
        predicted_mean_values = np.random.rand(6)
        true_mean_values = np.random.rand(6)
        datetimes = np.tile(pd.date_range("2018-01-31", freq="M", periods=3), 2)
        rmses = np.random.rand(6)
        maes = np.random.rand(6)
        r2s = np.random.rand(6)
        admin_level_names = ["province_l1_kenya" for _ in range(6)]
        models = ["ealstm" for _ in range(6)]
        region_names = np.repeat(["NAIROBI", "KIAMBU"], 3)

        df1 = pd.DataFrame(
            {
                "admin_level_name": admin_level_names,
                "model": models,
                "region_name": region_names,
                "datetimes": datetimes,
                "predicted_mean_values": predicted_mean_values,
                "true_mean_values": true_mean_values,
            }
        )
        df2 = pd.DataFrame(
            {
                "admin_level_name": admin_level_names,
                "model": models,
                "region_name": region_names,
                "rmse": rmses,
                "mae": maes,
                "r2": r2s,
            }
        )

        return df1, df2

    @pytest.mark.xfail(reason="geopandas not part of test environment")
    def test_init(self, tmp_path):
        shp_filepath = (
            tmp_path
            / "raw"
            / "boundaries"
            / "kenya"
            / "Admin2/KEN_admin2_2002_DEPHA.shp"
        )
        shp_filepath.parents[0].mkdir(parents=True, exist_ok=True)

        plotter = RegionGeoPlotter(tmp_path, country="kenya")
        assert isinstance(plotter.country_region_grouper, KenyaGroupbyRegion)

    @pytest.mark.xfail(reason="geopandas not part of test environment")
    def test_gdf_merge(self, tmp_path):
        shp_filepath = (
            tmp_path
            / "raw"
            / "boundaries"
            / "kenya"
            / "Admin2/KEN_admin2_2002_DEPHA.shp"
        )
        shp_filepath.parents[0].mkdir(parents=True, exist_ok=True)
        s = CreateSHPFile()
        s.create_demo_shapefile(shp_filepath)

        df1, df2 = self.make_performance_dataframe()

        plotter = RegionGeoPlotter(tmp_path, country="kenya")

        # check reading shapefiles
        plotter.read_shapefiles()
        assert "province_l1_kenya" in [k for k in plotter.region_gdfs.keys()]

        # check the join with the geometry -> GDF object
        gdf = plotter.merge_all_model_performances_gdfs(all_models_df=df2)

        expected_columns = [
            "admin_level_name",
            "model",
            "region_name",
            "rmse",
            "mae",
            "r2",
            "PROVINCE",
            "geometry",
        ]
        assert np.isin(expected_columns, gdf.columns).all()
        assert (gdf.region_name == gdf.PROVINCE).all()
