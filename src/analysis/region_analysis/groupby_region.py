from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import namedtuple

import xarray as xr
import pandas as pd

from src.utils import drop_nans_and_flatten

gpd = None
GeoDataFrame = None


AdminBoundaries = namedtuple(
    "AdminBoundaries",
    ["name", "lookup_colname", "var_name", "shp_filepath", "nc_filepath"],
)


class GroupbyRegion:
    """Convert an xr.DataArray object to a GeoDataFrame
    """

    def __init__(self, data_dir: Path = Path("data"), country: str = "kenya") -> None:
        print("GroupbyRegion requires geopandas to be installed")
        global gpd
        if gpd is None:
            import geopandas as gpd

        global GeoDataFrame
        if GeoDataFrame is None:
            from geopandas import GeoDataFrame

        assert data_dir.exists(), "Expect data_dir to exist!"
        self.data_dir = data_dir
        self.country: str = country

        self.shp_raw_dir = data_dir / "raw" / "boundaries" / self.country
        assert self.shp_raw_dir.exists(), (
            f"{self.shp_raw_dir} does not exist! Have you run the "
            "AdminBoundaries Exporters?"
            f"Existing countries: {d for d in [self.shp_raw_dir.parents[0].iterdir()]}"
        )

        self.out_dir: Path = self.data_dir / "analysis" / "region_analysis"
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self.da: xr.DataArray
        self.selection: str

        # filepaths for data
        self.admin_bound: AdminBoundaries
        self.region_shp_path: Path
        self.region_data_path: Path

        self.gpd: GeoDataFrame  # type: ignore
        self.region_da: xr.DataArray

    @staticmethod
    def load_region_data(region_data_path: Path) -> Tuple[xr.DataArray, Dict, str]:
        """Load the preprocessed `region_data` from the
        `data/analysis/boundaries_preprocessed` directory. This will
        not only return the categorical DataArray but also the
        associated lookup data (stored in the `attrs`).

        Returns:
        -------
        :xr.DataArray
            the categorical xr.DataArray with the region data (same shape)

        :Dict
            the lookup dictionary with the keys referring to the
            values in the xr.DataArray and the values the names
            of the regions (for joining to the shapefile data)

        """
        # LOAD in region lookup DataArray
        assert "analysis" in region_data_path.parts, (
            "Only preprocessed"
            "region files (as netcdf) should be used"
            "`data/analysis`"
        )
        region_group_name: str = region_data_path.name
        region_ds: xr.Dataset = xr.open_dataset(region_data_path)
        region_da: xr.DataArray = region_ds[[v for v in region_ds.data_vars][0]]
        region_lookup: Dict = dict(
            zip(
                [int(k.strip()) for k in region_ds.attrs["keys"].split(",")],
                [v.strip() for v in region_ds.attrs["values"].split(",")],
            )
        )

        return region_da, region_lookup, region_group_name

    def analyze(
        self,
        da: xr.DataArray,
        selection: str,
        mean: bool = True,
        save_data_fname: Optional[str] = None,
    ) -> GeoDataFrame:  # type: ignore
        """group the values in the DataArray
        by region, and return a GeoDataFrame allowing
        for spatial plotting and exploration of statistics
        by region.

        Arguments:
        ---------
        da: xr.DataArray
            the data stored in .nc you want to group
            by region and summarise

        selection: str
            the region level to plot
            {'level_1', ..., 'level_5'}

        mean: bool = True
            whether calculating the mean per region
            or else producing all values (can be a very
            large DataFrame >8mn rows)

        save_data_fname: Optional[str] = None
            if None then don't save, otherwise
            save the file to csv
        """
        # initialise attributes
        self.da = da
        assert type(self.da) == xr.DataArray, (
            "This method only works"
            "with `xr.DataArray` not `xr.Dataset`. Make sure you select"
            "one variable for your analysis (can group later)"
        )

        self.selection = selection
        self.admin_bound = self.get_admin_level(selection)

        # initialise filepaths to data
        self.region_shp_path = self.admin_bound.shp_filepath
        assert (
            self.region_shp_path.exists()
        ), "Have the AdminBoundaries Exporters been run?"

        self.region_data_path = self.admin_bound.nc_filepath
        assert (
            self.region_data_path.exists()
        ), "Has the AdminBoundaries Preprocessor been run?"

        print("* Running the Analyzer *")
        print(f"\t Admin Level: {self.admin_bound.var_name}")
        print(f"\t Calculating statistics per region.")

        # 1. load region data as xarray object
        region_da, region_lookup, region_group_name = self.load_region_data(
            self.region_data_path
        )

        # 2. Calculate mean per region
        print("* Calculating DataFrame of values per Region *")
        admin_level_name = self.admin_bound.var_name
        if mean:
            df = self.calculate_mean_per_region(
                da=self.da,
                region_da=region_da,
                region_lookup=region_lookup,
                admin_level_name=admin_level_name,
            )
        else:
            df = self.get_values_for_region(
                da=self.da,
                region_da=region_da,
                region_lookup=region_lookup,
                admin_level_name=admin_level_name,
            )
        df = df.astype(
            {"mean_value": float, "region_name": "category", "datetime": "datetime64"}
        )
        self.df = df

        # 3. join to GeoDataFrame
        print("* Joining DataFrame and GeoDataFrame *")
        gdf = gpd.read_file(self.admin_bound.shp_filepath)  # type: ignore
        gdf = self.join_dataframe_geodataframe(
            df,
            df_colname="region_name",
            gdf=gdf,
            gdf_colname=self.admin_bound.lookup_colname,
        )

        print("* Joined to GeoDataFrame and saved to `self.gdf`*")
        self.gdf = gdf

        if save_data_fname is not None:
            if save_data_fname[-4:] != ".csv":
                save_data_fname += ".csv"
            # g_ = gdf.drop('geometry')
            gdf.to_csv(self.out_dir / save_data_fname)

        return gdf

    @staticmethod
    def get_values_for_region(
        da: xr.DataArray,
        region_da: xr.DataArray,
        region_lookup: Dict,
        admin_level_name: str,
    ) -> pd.DataFrame:
        """Returns all non-nan values in a region. Can
        then run summary statistics or histograms by region!
        """
        valid_region_ids: List = [k for k in region_lookup.keys()]
        region_names: List = []
        all_values: List = []
        datetimes: List = []
        admin_level_names: List = []

        for valid_region_id in valid_region_ids:
            for time in da.time.values:
                region_names.append(region_lookup[valid_region_id])
                datetimes.append(time)
                admin_level_names.append(admin_level_name)
                # extract all non-nan values for that time-region
                values = drop_nans_and_flatten(
                    da.sel(time=time).where(region_da == valid_region_id)
                )
                all_values.append(values)

        assert len(region_names) == len(all_values) == len(datetimes)
        # note: extract the list in the `all_values` column
        # https://stackoverflow.com/a/53218939/9940782
        df = pd.DataFrame(
            {
                "datetime": datetimes,
                "region_name": region_names,
                "all_value": all_values,
            }
        )
        return df.explode("all_values").rename(columns={"all_values": "values"})

    def get_admin_level(self, selection: str) -> AdminBoundaries:  # type: ignore
        # implemented by the child classes (specific for each country)
        assert NotImplementedError

    @staticmethod
    def calculate_mean_per_region(
        da: xr.DataArray,
        region_da: xr.DataArray,
        region_lookup: Dict,
        admin_level_name: str,
    ) -> pd.DataFrame:
        """For each region in region_da calculate the mean """
        assert region_da.shape == (da.lat.shape[0], da.lon.shape[0]), (
            "Require matching shapes"
            f"region_da.shape: {region_da.shape}. da.shape: {da.shape}."
        )
        assert (region_da.lat == da.lat).all() & (region_da.lon == da.lon).all(), (
            "Only works when latlons match"
            "Have they been run through the same preprocessing?"
        )

        valid_region_ids: List = [k for k in region_lookup.keys()]
        region_names: List = []
        mean_values: List = []
        datetimes: List = []
        admin_level_names: List = []

        # TODO: SLOW code how can we speed up?
        # for each time / region calculate the
        for valid_region_id in valid_region_ids:
            for time in da.time.values:
                region_names.append(region_lookup[valid_region_id])
                datetimes.append(time)
                admin_level_names.append(admin_level_name)
                # calculate the mean value for that time-region
                mean_values.append(
                    da.sel(time=time).where(region_da == valid_region_id).mean().values
                )

        assert len(region_names) == len(mean_values) == len(datetimes)
        return pd.DataFrame(
            {
                "datetime": datetimes,
                "region_name": region_names,
                "mean_value": mean_values,
            }
        )

    @staticmethod
    def join_dataframe_geodataframe(
        df: pd.DataFrame,
        gdf: GeoDataFrame,  # type: ignore
        gdf_colname: str,
        df_colname: str,
    ) -> GeoDataFrame:  # type: ignore
        """Join pd.DataFrame with GeoDataFrame to produce
        a GeoDataFrame.
        """
        out_gdf = GeoDataFrame(  # type: ignore
            pd.merge(
                df,
                gdf[[gdf_colname, "geometry"]],
                left_on=df_colname,
                right_on=gdf_colname,
            )
        )
        return out_gdf


class KenyaGroupbyRegion(GroupbyRegion):
    def __init__(self, data_dir: Path = Path("data")) -> None:
        super().__init__(data_dir=data_dir, country="kenya")

    def get_admin_level(self, selection: str) -> AdminBoundaries:
        level_1 = AdminBoundaries(
            name="level_1",
            lookup_colname="PROVINCE",
            var_name="province_l1",
            shp_filepath=self.shp_raw_dir / "Admin2/KEN_admin2_2002_DEPHA.shp",
            nc_filepath=(
                self.data_dir
                / "analysis"
                / "boundaries_preprocessed"
                / "province_l1_kenya.nc"
            ),
        )

        level_2 = AdminBoundaries(
            name="level_2",
            lookup_colname="DISTNAME",
            var_name="district_l2",
            shp_filepath=self.shp_raw_dir / "Ken_Districts/Ken_Districts.shp",
            nc_filepath=(
                self.data_dir
                / "analysis"
                / "boundaries_preprocessed"
                / "district_l2_kenya.nc"
            ),
        )

        level_3 = AdminBoundaries(
            name="level_3",
            lookup_colname="DIVNAME",
            var_name="division_l3",
            shp_filepath=self.shp_raw_dir / "Ken_Divisions/Ken_Divisions.shp",
            nc_filepath=(
                self.data_dir
                / "analysis"
                / "boundaries_preprocessed"
                / "division_l3_kenya.nc"
            ),
        )

        level_3_wards = AdminBoundaries(
            name="level_3_wards",
            lookup_colname="IEBC_WARDS",
            var_name="wards_l3",
            shp_filepath=self.shp_raw_dir / "Kenya wards.shp",
            nc_filepath=(
                self.data_dir
                / "analysis"
                / "boundaries_preprocessed"
                / "level_3_wards_kenya.nc"
            ),
        )

        level_4 = AdminBoundaries(
            name="level_4",
            lookup_colname="LOCNAME",
            var_name="location_l4",
            shp_filepath=self.shp_raw_dir / "Ken_Locations/Ken_Locations.shp",
            nc_filepath=(
                self.data_dir
                / "analysis"
                / "boundaries_preprocessed"
                / "location_l4_kenya.nc"
            ),
        )

        level_5 = AdminBoundaries(
            name="level_5",
            lookup_colname="SLNAME",
            var_name="sublocation_l5",
            shp_filepath=self.shp_raw_dir / "Ken_Sublocations/Ken_Sublocations.shp",
            nc_filepath=(
                self.data_dir
                / "analysis"
                / "boundaries_preprocessed"
                / "sublocation_l5_kenya.nc"
            ),
        )

        lookup = {
            "level_1": level_1,
            "level_2": level_2,
            "level_3": level_3,
            "level_3_wards": level_3_wards,
            "level_4": level_4,
            "level_5": level_5,
        }

        assert (
            selection in lookup.keys()
        ), f"`selection` must be one of: {list(lookup.keys())}"

        return lookup[selection]
