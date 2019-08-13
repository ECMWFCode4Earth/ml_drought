from pathlib import Path
import pandas as pd

from typing import Any, List, Dict, Optional
from collections import namedtuple

from src.preprocess import KenyaAdminPreprocessor

gpd = None
GeoDataFrame = None


AdminLevelGeoDF = namedtuple('AdminLevelGeoDF', ['gdf', 'gdf_colname', 'admin_name'])


class RegionGeoPlotter:
    def __init__(self, data_folder: Path = Path('data'),
                 country: str = 'kenya') -> None:

        self.data_folder = data_folder
        print('The RegionGeoPlotter requires `geopandas` to be installed.')
        # try and import geopandas
        global gpd
        if gpd is None:
            import geopandas as gpd

        global GeoDataFrame
        if GeoDataFrame is None:
            from geopandas.geodataframe import GeoDataFrame

        self.country = country

        self.country_preprocessor = self.get_country_preprocessor(country_str=country)

        # Setup type hints without assignment
        self.region_gdfs: Dict[str, AdminLevelGeoDF] = {}

        self.gdf: Optional[GeoDataFrame] = None  # type: ignore
        self.all_gdfs: List = []

    def get_country_preprocessor(self, country_str: str) -> Any:
        # if need a new country boundaries add example here
        country_lookup = {
            'kenya': KenyaAdminPreprocessor,
        }
        keys = [k for k in country_lookup.keys()]
        assert country_str in keys, 'Expect ' \
            f'the `country_str` argument to be one of: {keys}'

        return country_lookup[country_str](self.data_folder)

    def read_shapefiles(self):
        # read the shapefile to GeoDataFrames for each region
        if self.country == 'kenya':
            levels = ['level_1', 'level_2', 'level_3',
                      'level_3_wards', 'level_4', 'level_5']

            region_gdfs = {}

            for level in levels:
                admin_boundary = self.country_preprocessor.get_admin_level(selection=level)
                try:
                    gdf = gpd.read_file(admin_boundary.shp_filepath)
                    key = f'{admin_boundary.var_name}_{self.country}'
                    region_gdfs[key] = AdminLevelGeoDF(
                        gdf=gdf,
                        gdf_colname=admin_boundary.lookup_colname,
                        admin_name=key,
                    )
                except Exception:
                    print(
                        f'{admin_boundary.shp_filepath} not found.'
                        'Moving to next file'
                    )
            self.region_gdfs = region_gdfs
            print('* Read shapefiles and stored in `RegionGeoPlotter.region_gdfs`')

        else:
            raise NotImplementedError

    def join_model_performances_to_geometry(self, model_performance_df: pd.DataFrame,
                                            admin_name: str) -> GeoDataFrame:  # type: ignore
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
        assert admin_name in [k for k in self.region_gdfs.keys()], 'Invalid ' \
            f'`admin_name`. Expected one of: {[k for k in self.region_gdfs.keys()]}' \
            f' Got: {admin_name}'
        gdf = self.region_gdfs[admin_name].gdf
        gdf_colname = self.region_gdfs[admin_name].gdf_colname
        gdf[gdf_colname] = gdf[gdf_colname].apply(str.rstrip).apply(str.lstrip)

        df_colname = 'region_name'
        out_gdf = gpd.GeoDataFrame(  # type: ignore
            pd.merge(
                model_performance_df, gdf[[gdf_colname, 'geometry']],
                left_on=df_colname, right_on=gdf_colname
            )
        )

        return out_gdf

    def merge_all_model_performances_gdfs(self, all_models_df: pd.DataFrame
                                          ) -> GeoDataFrame:  # type: ignore
        all_gdfs: List[GeoDataFrame] = []  # type: ignore
        assert 'admin_level_name' in all_models_df.columns, f'Expect to find admin_region' \
            f'in {all_models_df.columns}'

        # join the geometry columns to make gpd.GeoDataFrames
        for admin_name in all_models_df.admin_level_name.unique():
            admin_level_df = all_models_df.loc[all_models_df.admin_level_name == admin_name]
            all_gdfs.append(self.join_model_performances_to_geometry(
                model_performance_df=admin_level_df, admin_name=admin_name
            ))

        self.gdf = pd.concat(all_gdfs)
        # convert mean model outputs to float
        try:
            self.gdf = self.gdf.astype(  # type: ignore
                {'predicted_mean_value': 'float64', 'true_mean_value': 'float64'}
            )
        except KeyError:
            self.gdf = self.gdf.astype(  # type: ignore
                {'rmse': 'float64', 'mae': 'float64', 'r2': 'float64'}
            )

        print('* Assigned the complete GeoDataFrame to `RegionGeoPlotter.gdf`')

        if not isinstance(self.gdf, GeoDataFrame):  # type: ignore
            self.gdf = GeoDataFrame(self.gdf)  # type: ignore

        return self.gdf
