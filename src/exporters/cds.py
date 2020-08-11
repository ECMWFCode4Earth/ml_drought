from pathlib import Path
import warnings
import re
from pprint import pprint
import multiprocessing
from datetime import datetime
from typing import Dict, Optional, List, cast

from .base import BaseExporter, Region, region_lookup

cdsapi = None


class CDSExporter(BaseExporter):
    """Exports for the Climate Data Store

    cds.climate.copernicus.eu
    """

    def __init__(self, data_folder: Path = Path("data")) -> None:
        super().__init__(data_folder)

        global cdsapi
        if cdsapi is None:
            import cdsapi

        self.client = cdsapi.Client()  # type: ignore

    @staticmethod
    def _correct_input(value, key):
        if type(value) is str:

            # check the string is correctly formatted
            if key == "time":
                assert re.match(
                    r"\d{2}:0{2}", value
                ), f"Expected time string {value} to be in hour:minute format, \
                    e.g. 01:00. Minutes MUST be `00`"
            return value
        else:
            if key == "year":
                return str(value)
            elif key in {"month", "day"}:
                return "{:02d}".format(value)
            elif key == "time":
                return "{:02d}:00".format(value)
        return str(value)

    @staticmethod
    def _check_iterable(value, key):
        if (key == "time") and (type(value) is str):
            return [value]
        try:
            iter(value)
        except TypeError as te:
            warnings.warn(f"{key}: {te}. Converting to list")
            value = [value]
        return value

    @staticmethod
    def create_area(region: Region) -> str:
        """Create an area string for the CDS API from a Region object

        Parameters
        ----------
        region: Region
            A Region to be exported from the CDS warehouse

        Returns
        ----------
        area: str
            A string representing the region which can be passed to the CDS API

        format = 'North/West/South/East'
        """
        x = [region.latmax, region.lonmin, region.latmin, region.lonmax]

        return "/".join(["{:.3f}".format(x) for x in x])

    @staticmethod
    def get_default_era5_times(granularity: str = "hourly", land: bool = False) -> Dict:
        """Returns the era5 selection request arguments
        for the hourly or monthly data

        Parameters
        ----------
        granularity: str, {'monthly', 'hourly'}, default: 'hourly'
            The granularity of data being pulled

        land: bool = False
            ERA5 or ERA5 land
            Currently ERA5 Land only goes back to 2001

        Returns
        ----------
        selection_dict: dict
            A dictionary with all the time-related arguments of the
            selection dict filled out
        """
        months = ["{:02d}".format(month) for month in range(1, 12 + 1)]
        days = ["{:02d}".format(day) for day in range(1, 31 + 1)]
        times = ["{:02d}:00".format(hour) for hour in range(24)]

        if land:  # era5 land
            years = [str(year) for year in range(1981, datetime.now().year + 1)]
        else:  # era5
            years = [str(year) for year in range(1979, datetime.now().year + 1)]

        selection_dict = {"year": years, "month": months, "time": times}
        if granularity == "hourly":
            selection_dict["day"] = days
        return selection_dict

    @staticmethod
    def _filename_from_selection_request(time_array: List, time: str) -> str:
        if len(time_array) > 1:
            warnings.warn(
                f"More than 1 {time} of data being exported! "
                "Export times may be significant."
            )
            time_array.sort()
            time_array_str = f"{time_array[0]}_{time_array[-1]}"
        else:
            time_array_str = str(time_array[0])
        return time_array_str

    def make_filename(self, dataset: str, selection_request: Dict) -> Path:
        """Makes the appropriate filename for a CDS export.
        If necessary, intermediate folders are also made.

        The format in which data is saved is
        {dataset}/{variable}/{year}/{month}/file
        """
        dataset_folder = self.raw_folder / dataset
        if not dataset_folder.exists():
            dataset_folder.mkdir()

        variables = "_".join(selection_request["variable"])
        variables_folder = dataset_folder / variables
        if not variables_folder.exists():
            variables_folder.mkdir()

        years = self._filename_from_selection_request(selection_request["year"], "year")
        years_folder = variables_folder / years
        if not years_folder.exists():
            years_folder.mkdir()

        months = self._filename_from_selection_request(
            selection_request["month"], "month"
        )
        output_filename = years_folder / f"{months}.nc"
        return output_filename

    @staticmethod
    def _print_api_request(
        dataset: str, selection_request: Dict, output_file: Path
    ) -> None:
        print("------------------------")
        print(f"Dataset: {dataset}")
        print("Selection Request:")
        pprint(selection_request)
        print("------------------------")
        print("Output Filename:")
        print(output_file)
        print("------------------------")

    def _export(
        self,
        dataset: str,
        selection_request: Dict,
        show_api_request: bool = False,
        in_parallel: bool = False,
    ) -> Path:
        """Export CDS data

        Parameters
        ----------
        dataset: str
            The dataset to be exported
        selection_request: dict
            The selection information to be passed to the CDS API

        Returns
        ----------
        output_file: Path
            The location of the exported data
        """

        output_file = self.make_filename(dataset, selection_request)

        if show_api_request:
            self._print_api_request(dataset, selection_request, output_file)

        if not output_file.exists():

            if not in_parallel:
                self.client.retrieve(dataset, selection_request, str(output_file))

            else:  # in parallel create a new Client each time it's called
                client = cdsapi.Client()  # type: ignore
                client.retrieve(dataset, selection_request, str(output_file))

        return output_file


class ERA5Exporter(CDSExporter):
    """Exports ERA5 data from the Climate Data Store

    cds.climate.copernicus.eu
    """

    @staticmethod
    def get_era5_times(granularity: str = "hourly") -> Dict:
        """Returns the era5 selection request arguments
        for the hourly or monthly data

        Parameters
        ----------
        granularity: str, {'monthly', 'hourly'}, default: 'hourly'
            The granularity of data being pulled

        Returns
        ----------
        selection_dict: dict
            A dictionary with all the time-related arguments of the
            selection dict filled out
        """
        years = [str(year) for year in range(1979, 2019 + 1)]
        months = ["{:02d}".format(month) for month in range(1, 12 + 1)]
        days = ["{:02d}".format(day) for day in range(1, 31 + 1)]
        times = ["{:02d}:00".format(hour) for hour in range(24)]

        selection_dict = {"year": years, "month": months, "time": times}
        if granularity == "hourly":
            selection_dict["day"] = days
        return selection_dict

    @staticmethod
    def get_dataset(variable: str, granularity: str = "hourly") -> str:
        pressure_level_variables = {
            "divergence",
            "fraction_of_cloud_cover",
            "geopotential",
            "ozone_mass_mixing_ratio",
            "potential_vorticity",
            "relative_humidity",
            "specific_cloud_ice_water_content",
            "specific_cloud_liquid_water_content",
            "specific_humidity",
            "specific_rain_water_content",
            "specific_snow_water_content",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
            "vorticity",
        }

        if variable in pressure_level_variables:
            dataset_type = "pressure-levels"
        else:
            dataset_type = "single-levels"

        dataset = f"reanalysis-era5-{dataset_type}"
        if granularity == "monthly":
            dataset = f"{dataset}-monthly-means"

        return dataset

    def create_selection_request(
        self,
        variable: str,
        selection_request: Optional[Dict] = None,
        granularity: str = "hourly",
        region_str: str = "kenya",
    ) -> Dict:
        # setup the default selection request
        product_type = (
            f'{"monthly_averaged_" if granularity == "monthly" else ""}reanalysis'
        )
        processed_selection_request = {
            "product_type": product_type,
            "format": "netcdf",
            "variable": [variable],
        }
        for key, val in self.get_default_era5_times(granularity).items():
            processed_selection_request[key] = val

        # by default, we investigate Kenya
        # kenya_region = get_kenya()
        region = region_lookup[region_str]
        processed_selection_request["area"] = self.create_area(region)

        # update with user arguments
        if selection_request is not None:
            for key, val in selection_request.items():
                if key in {"day", "hour"}:
                    warnings.warn(
                        f"Overriding default {key} values. "
                        f"The ERA5Exporter assumes all days "
                        f"and times (the default) are being downloaded"
                    )
                val = self._check_iterable(val, key)
                processed_selection_request[key] = [
                    self._correct_input(x, key) for x in val
                ]
        return processed_selection_request

    def export(
        self,
        variable: str,
        dataset: Optional[str] = None,
        granularity: str = "hourly",
        show_api_request: bool = True,
        selection_request: Optional[Dict] = None,
        break_up: bool = False,
        n_parallel_requests: int = 1,
        region_str: str = "kenya",
    ) -> List[Path]:
        """ Export functionality to prepare the API request and to send it to
        the cdsapi.client() object.

        Arguments:
        ---------
        variable: str
            The variable to be exported
        dataset: Optional[str], default = None
            The dataset from which to pull the variable from. If None, this
            is inferred from the dataset and its granularity
        granularity: str: {'hourly', 'monthly'}, default = 'hourly'
            The temporal resolution of the data to be pulled
        show_api_request: bool = True
            Whether to print the selection dictionary before making the API request
        selection_request: Optional[Dict], default = None
            Selection request arguments to be merged with the defaults. If both a key is
            defined in both the selection_request and the defaults, the value in the
            selection_request takes precedence.
        break_up: bool, default = True
            The best way to download the data is by making many small calls to the CDS
            API. If true, the calls will be broken up into months
        parallel: bool, default = True
            Whether to download data in parallel
        n_parallel_requests:
            How many parallel requests to the CDSAPI to make

        Returns:
        -------
        output_files: List of pathlib.Paths
            paths to the downloaded data
        """

        # create the default template for the selection request
        processed_selection_request = self.create_selection_request(
            variable, selection_request, granularity, region_str
        )

        if dataset is None:
            dataset = self.get_dataset(variable, granularity)

        if n_parallel_requests < 1:
            n_parallel_requests = 1

        # break up by year
        if break_up:
            if n_parallel_requests > 1:  # Run in parallel
                p = multiprocessing.Pool(int(n_parallel_requests))

            output_paths = []
            for year in processed_selection_request["year"]:
                updated_request = processed_selection_request.copy()
                updated_request["year"] = [year]

                if n_parallel_requests > 1:  # Run in parallel
                    # multiprocessing of the paths
                    output_paths.append(
                        p.apply_async(
                            self._export,
                            args=(dataset, updated_request, show_api_request, True),
                        )
                    )
                else:  # run sequentially
                    output_paths.append(  # type: ignore
                        self._export(dataset, updated_request, show_api_request)
                    )
            if n_parallel_requests > 1:
                p.close()
                p.join()
            return cast(List[Path], output_paths)

        return [self._export(dataset, processed_selection_request, show_api_request)]
