from pathlib import Path
import xarray as xr
from collections import namedtuple
from .base import BasePreProcessor
from .utils import SHPtoXarray

from typing import Optional, Dict

gpd = None
GeoDataFrame = None


AdminBoundaries = namedtuple(
    "AdminBoundaries", ["name", "lookup_colname", "var_name", "shp_filepath"]
)


class OCHAAdminBoundariesPreprocesser(BasePreProcessor):
    """ Preprocesses the OCHA Admin Boundaries data

    https://data.humdata.org/
    """

    country: str
    dataset = "boundaries"
    analysis = True

    def __init__(self, data_folder: Path = Path("data")):
        super().__init__(data_folder)

        # try and import geopandas
        print("The OCHA AdminBoundaries preprocessor requires the geopandas package")
        global gpd
        if gpd is None:
            import geopandas as gpd
        global GeoDataFrame
        if GeoDataFrame is None:
            from geopandas.geodataframe import GeoDataFrame

    def get_filename(self, var_name: str) -> str:  # type: ignore
        new_filename = f"{var_name}_{self.country}.nc"
        return new_filename

    def _preprocess_single(
        self,
        shp_filepath: Path,
        reference_nc_filepath: Path,
        var_name: str,
        lookup_colname: str,
        save: bool = True,
    ) -> Optional[xr.Dataset]:
        """ Preprocess .shp admin boundary files into an `.nc`
        file with the same shape as reference_nc_filepath.

        Will create categorical .nc file which will specify
        which admin region each pixel is in.

        Arguments
        ----------
        shp_filepath: Path
            The path to the shapefile

        reference_nc_filepath: Path
            The path to the netcdf file with the shape
            (must have been run through Preprocessors prior to using)

        var_name: str
            the name of the Variable in the xr.Dataset and the name
            of the output filename - {var_name}_{self.country}.nc

        lookup_colname: str
            the column name to lookup in the shapefile
            (read in as geopandas.GeoDataFrame)
        """
        filename = self.get_filename(var_name)
        if (self.out_dir / filename).exists():
            print(
                "** Data already preprocessed! **\nIf you need to "
                "process again then move or delete existing file"
                f" at: {(self.out_dir / filename).as_posix()}"
            )
            return None

        assert "interim" in reference_nc_filepath.parts, (
            "Expected " "the target data to have been preprocessed by the pipeline"
        )

        # MUST have a target dataset to create the same shape
        target_ds = xr.ones_like(xr.open_dataset(reference_nc_filepath))
        data_var = [d for d in target_ds.data_vars][0]
        da = target_ds[data_var]

        # turn the shapefile into a categorical variable (like landcover)
        shp_to_nc = SHPtoXarray()
        ds = shp_to_nc.shapefile_to_xarray(
            da=da,
            shp_path=shp_filepath,
            var_name=var_name,
            lookup_colname=lookup_colname,
        )

        # save the data
        if save:
            print(f"Saving to {self.out_dir}")

            if self.analysis is True:
                assert self.out_dir.parts[-2] == "analysis", (
                    "self.analysis should"
                    "be True and the output directory should be analysis"
                )

            ds.to_netcdf(self.out_dir / filename)

            print(f"** {(self.out_dir / filename).as_posix()} saved! **")

            return None
        else:
            return ds


class KenyaAdminPreprocessor(OCHAAdminBoundariesPreprocesser):
    country = "kenya"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)
        self.base_raw_dir = self.raw_folder / self.dataset / self.country

    def get_admin_level(self, selection: str) -> AdminBoundaries:
        level_1 = AdminBoundaries(
            name="level_1",
            lookup_colname="PROVINCE",
            var_name="province_l1",
            shp_filepath=self.base_raw_dir / "Admin2/KEN_admin2_2002_DEPHA.shp",
        )

        level_2 = AdminBoundaries(
            name="level_2",
            lookup_colname="DISTNAME",
            var_name="district_l2",
            shp_filepath=self.base_raw_dir / "Ken_Districts/Ken_Districts.shp",
        )

        level_3 = AdminBoundaries(
            name="level_3",
            lookup_colname="DIVNAME",
            var_name="division_l3",
            shp_filepath=self.base_raw_dir / "Ken_Divisions/Ken_Divisions.shp",
        )

        level_3_wards = AdminBoundaries(
            name="level_3_wards",
            lookup_colname="IEBC_WARDS",
            var_name="wards_l3",
            shp_filepath=self.base_raw_dir / "Kenya wards.shp",
        )

        level_4 = AdminBoundaries(
            name="level_4",
            lookup_colname="LOCNAME",
            var_name="location_l4",
            shp_filepath=self.base_raw_dir / "Ken_Locations/Ken_Locations.shp",
        )

        level_5 = AdminBoundaries(
            name="level_5",
            lookup_colname="SLNAME",
            var_name="sublocation_l5",
            shp_filepath=self.base_raw_dir / "Ken_Sublocations/Ken_Sublocations.shp",
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

    def preprocess(self, reference_nc_filepath: Path, selection: str) -> None:
        """Preprocess Kenya Admin Boundaries shapefiles into xarray objects
        """
        admin_level = self.get_admin_level(selection)
        self._preprocess_single(
            shp_filepath=admin_level.shp_filepath,
            lookup_colname=admin_level.lookup_colname,
            reference_nc_filepath=reference_nc_filepath,
            var_name=admin_level.var_name,
        )


class KenyaASALMask(KenyaAdminPreprocessor):

    analysis = False

    asal_districts = [
        "TURKANA",
        "MANDERA",
        "SAMBURU",
        "GARISSA",
        "MARSABIT",
        "WAJIR",
        "TANA RIVER",
        "WEST POKOT",
        "ISIOLO",
        "KITUI",
        "MOYALE",
    ]

    @staticmethod
    def val_to_key(ds: xr.Dataset) -> Dict[str, int]:

        return dict(
            zip(
                [v.strip() for v in ds.attrs["values"].split(",")],
                [int(k.strip()) for k in ds.attrs["keys"].split(",")],
            )
        )

    def preprocess(
        self, reference_nc_filepath: Path, selection: str = "level_2"
    ) -> None:

        assert selection == "level_2", f"Only level 2 supported, got {selection}"
        district_boundaries = self.get_admin_level("level_2")

        ds = self._preprocess_single(
            shp_filepath=district_boundaries.shp_filepath,
            lookup_colname=district_boundaries.lookup_colname,
            reference_nc_filepath=reference_nc_filepath,
            var_name=district_boundaries.var_name,
            save=False,
        )

        assert isinstance(ds, xr.Dataset)

        val2key = self.val_to_key(ds)
        relevant_keys = [val2key.get(district) for district in self.asal_districts]

        ds["mask"] = ~ds.district_l2.isin(relevant_keys)
        # save
        filename = "kenya_asal_mask.nc"
        ds.to_netcdf(self.out_dir / filename)
        print(f"** {(self.out_dir / filename).as_posix()} saved! **")
