from pathlib import Path
import xarray as xr
from collections import namedtuple
from .base import BasePreProcessor
from .utils import SHPtoXarray

from typing import Optional, Dict

gpd = None
GeoDataFrame = None


class FEWSNetPreprocesser(BasePreProcessor):
    """ Preprocesses the FEWSNetwork shapefile data
    """

    country_code_mapping: Dict = {
        "AF": "Afghanistan",
        "AO": "Angola",
        "BF": "Burkina Faso",
        "BI": "Burundi",
        "CF": "CAR",
        "DJ": "Djibouti",
        "TZ": "Tanzania",
        "ZW": "Zimbabwe",
        "ZM": "Zambia",
        "YE": "Yemen",
        "UG": "Uganda",
        "TD": "Chad",
        "SV": "El Salvador",
        "SN": "Senegal",
        "SO": "Somalia",
        "SL": "Sierra Leone",
        "SD": "Sudan",
        "NI": "Nicaragua",
        "NG": "Nigeria",
        "ET": "Ethiopia",
        "NE": "Niger",
        "GN": "Guinea",
        "MZ": "Mozambique",
        "HN": "Honduras",
        "MW": "Malawi",
        "ML": "Mali",
        "MR": "Mauritania",
        "HT": "Haiti",
        "MG": "Madagascar",
        "LR": "Liberia",
        "KE": "Kenya",
        "LS": "Lesotho",
        "CD": "DR Congo",
        "SS": "South Sudan",
        "RW": "Rwanda",
        "NI": "Nicaragua",
        "TJ": "Tajikistan",
        "GT": "Guatemala",
    }

    dataset: str
    analysis = True

    def __init__(self, data_folder: Path = Path("data")):
        super().__init__(data_folder)

        # try and import geopandas
        print("The FEWSNet preprocessor requires the geopandas package")
        global gpd
        if gpd is None:
            import geopandas as gpd
        global GeoDataFrame
        if GeoDataFrame is None:
            from geopandas.geodataframe import GeoDataFrame

    def get_filename(self, var_name: str, country: str) -> str:  # type: ignore
        new_filename = f"{var_name}_{country}.nc"
        return new_filename


class FEWSNetLivelihoodPreprocessor(FEWSNetPreprocesser):
    dataset = "livelihood_zones"

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)
        self.base_raw_dir = self.raw_folder / "boundaries" / self.dataset

    def _preprocess_single(
        self,
        shp_filepath: Path,
        reference_nc_filepath: Path,
        var_name: str,
        lookup_colname: str,
        save: bool = True,
        country_str: Optional[str] = None,
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

        country_str: Optional[str] = None
            the country you want to preprocess

        """
        assert "interim" in reference_nc_filepath.parts, (
            "Expected " "the target data to have been preprocessed by the pipeline"
        )

        # MUST have a target dataset to create the same shape
        target_ds = xr.ones_like(xr.open_dataset(reference_nc_filepath))
        data_var = [d for d in target_ds.data_vars][0]
        da = target_ds[data_var]

        # turn the shapefile into a categorical variable (like landcover)
        gdf = gpd.read_file(shp_filepath)
        shp_to_nc = SHPtoXarray()

        # if supply a country_str then only create .nc file for that country
        country_lookup = dict(zip(self.country_code_mapping.values(), self.country_code_mapping.keys()))
        if country_str is not None:
            if country_str.capitalize() not in country_lookup.keys():
                assert False, (
                    f"Expecting to have one of: \n{country_lookup.keys()}"
                    f"\nYou supplied: {country_str.capitalize()}"
                    "\nDoes this definitely exist?"
                )
            country_code_list = [country_lookup[country_str.capitalize()]]
        else:
            country_code_list = gdf.COUNTRY.unique()

        for country_code in country_code_list:
            gdf_country = gdf.loc[gdf.COUNTRY == country_code]

            # create a unique filename for each country
            country_str = (
                self.country_code_mapping[country_code].lower().replace(" ", "_")
            )
            filename = self.get_filename(var_name, country_str)
            if (self.out_dir / filename).exists():
                print(
                    "** Data already preprocessed! **\nIf you need to "
                    "process again then move or delete existing file"
                    f" at: {(self.out_dir / filename).as_posix()}"
                )
                continue

            ds = shp_to_nc._to_xarray(
                da=da, gdf=gdf_country, var_name=var_name, lookup_colname=lookup_colname
            )

            # save the data
            print(f"Saving to {self.out_dir}")

            if self.analysis is True:
                assert self.out_dir.parts[-2] == "analysis", (
                    "self.analysis should"
                    "be True and the output directory should be analysis"
                )

            ds.to_netcdf(self.out_dir / filename)
            # save key info columns
            gdf_country[
                ["OBJECTID", "FNID", "LZNUM", "LZCODE", "LZNAMEEN", "CLASS"]
            ].to_csv(self.out_dir / f"{country_str}_lookup_dict.csv")

            print(
                f"** {(self.out_dir / filename).as_posix()} and lookup_dict saved! **"
            )

    def preprocess(self, reference_nc_filepath: Path, country_str: Optional[str] = None) -> None:
        """Preprocess FEWSNet Livelihood Zone shapefiles into xarray objects
        """
        self._preprocess_single(
            shp_filepath=self.base_raw_dir / "FEWS_NET_LH_World.shp",
            lookup_colname="LZCODE",
            reference_nc_filepath=reference_nc_filepath,
            var_name="livelihood_zone",
            country_str=country_str
        )
