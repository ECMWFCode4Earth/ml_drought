import cdsapi
from pathlib import Path
import certifi
import urllib3
import warnings
# import itertools
# import re
import numpy as np
# from pprint import pprint

from typing import Dict, Optional, List
from .all_valid_s5 import datasets as dataset_reference
from .base import Region, get_kenya
from .cds import CDSExporter

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)

class S5Exporter(CDSExporter):

    @staticmethod
    def get_s5_initialisation_times(granularity: str,
                     min_year: int = 1993,
                     max_year: int = 2019,
                     min_month: int = 1,
                     max_month: int = 12,) -> Dict:
        """Returns the SEAS5 initialisation times
        for the hourly or monthly data

        NOTE: these depend on the dataset and the naming convention is after
         the cdsapi monthly = `seasonal-monthly-single-levels` vs.
         hourly = `seasonal-original-single-levels`

        """
        # check that valid requests
        assert granularity in ['monthly', 'hourly'], f"Invalid granularity \
            argument. Expected: {['monthly', 'hourly']} Got: {granularity}"
        assert min_year >= 1993, f"The minimum year is 1993. You asked for:\
            {min_year}"
        assert max_year <= 2019, f"The maximum year is 2019. You asked for:\
            {max_year}"

        # build up list of years
        years = [str(year) for year in range(min_year, max_year + 1)]
        months = [
            '{:02d}'.format(month)
            for month in range(min_month, max_month + 1)
        ]

        selection_dict = {
            'year': years,
            'month': months,
            'day': '01',  # forecast initialised on 1st each month
        }

        return selection_dict

    @staticmethod
    def get_6hrly_leadtimes(max_leadtime: int) -> List[int]:
        # every 6hours up to the max number of days
        hrly_6 = np.array([6, 12, 18, 24])
        all_hrs = np.array(
            [hrly_6 + ((day-1) * 24) for day in range(1, max_leadtime + 1)]
        )
        # prepend the 0th hour to the list
        leadtime_times = np.insert(all_hrs.flatten(), 0, 0)

        return leadtime_times

    def get_s5_leadtimes(self,
                         granularity: str,
                         max_leadtime: int,
                         pressure_level: bool) -> Dict:
        """Get the leadtimes for monthly or hourly data"""
        if granularity == 'monthly':
            assert max_leadtime <= 6, f"The maximum leadtime is 6 months.\
                You asked for: {max_leadtime}"
            leadtime_key = 'leadtime_month'
            leadtime_times = [m for m in range(1, max_leadtime + 1)]

        elif granularity == 'hourly':
            assert max_leadtime <= 215, f"The maximum leadtime is 215 days.\
                You asked for: {max_leadtime}"
            leadtime_key = 'leadtime_hour'

            if pressure_level:
                leadtime_times = [day * 24 for day in range(1, 215 + 1)]
            else:
                leadtime_times = list(self.get_6hrly_leadtimes(max_leadtime))

            assert max(leadtime_times) <= 5160, f"Max leadtime must be less \
                than 215 days (5160hrs)"
        else:
            assert False, f'granularity must be in ["monthly", "hourly"]\
                Currently: {granularity}'

        # convert to strings
        leadtime_times = [str(lt) for lt in leadtime_times]
        selection_dict = {
            leadtime_key: leadtime_times,  # leadtime_key differs for monthly/hourly
        }

        return selection_dict

    def get_product_type(self,
                         product_type: Optional[str] = None) -> Optional[str]:
        """By default download the 'monthly_mean' product for monthly data """
        valid_product_types = self.dataset_reference['product_type']

        if valid_product_types is None:
            f"{self.dataset} has no `product_type` key. Only the monthly datasets have product types!"
            return None

        # if not provided then download monthly_mean
        if product_type is None:
            print("No `product_type` provided. Therefore, downloading `monthly_mean`")
            product_type = 'monthly_mean'

        assert product_type in valid_product_types, f"Invalid `product_type`: \
            {product_type}. Must be one of: {valid_product_types}"

        return product_type

    # @staticmethod
    # def _correct_input(value, key):
    #     if type(value) is str:
    #         # check the string is correctly formatted
    #         if key == 'time':
    #             assert (re.match(r"\d{2}:0{2}", value)), \
    #                 f'Expected time string {value} to be in hour:minute format, \
    #                 e.g. 01:00. Minutes MUST be `00`'
    #         return value
    #     else:
    #         if key == 'year':
    #             return str(value)
    #         elif key in {'month', 'day'}:
    #             return '{:02d}'.format(value)
    #         elif key == 'time':
    #             return '{:02d}:00'.format(value)
    #     return str(value)


    def create_selection_request(self,
                                 variable: str,
                                 max_leadtime: int,
                                 min_year: int = 1993,
                                 max_year: int = 2019,
                                 min_month: int = 1,
                                 max_month: int = 12,
                                 selection_request: Optional[Dict] = None,
                                 ) -> Dict:
        """Build up the selection_request dictionary with defaults

        Returns
        ----------
        selection_dict: dict
            A dictionary with all the time-related arguments of the
            selection dict filled out

        Note:
        ----
        - This is a FULL version of the dictionary that is returned. Will likely
         lead to errors if you don't chunk the data (e.g. by month) before
         sending the requests to cdsapi
        - Some attributes are used to create the default arguments
            (self.product_type, self.granularity, self.dataset, self.pressure_level)
        """
        # check the variable is valid for this dataset
        assert variable in self.dataset_reference['variable'], f"\
            {variable} is not a valid variable for the {self.dataset} dataset.\
            Try one of: {self.dataset_reference['variable']}"

        # check the product_type is valid for this dataset
        assert self.product_type in self.dataset_reference['product_type'], f"\
            {self.product_type} is not a valid variable for the {self.dataset} dataset.\
            Try one of: {self.dataset_reference['product_type']}"

        # setup the default selection request
        processed_selection_request = {
            'format': 'grib',
            'originating_centre': 'ecmwf',
            'system': '5',
            'variable': [variable],
            'product_type': [self.product_type]
        }

        # get the initialisation time information
        init_times_dict = self.get_s5_initialisation_times(
            self.granularity, min_year, max_year, min_month, max_month,
        )
        for key, val in init_times_dict.items():
            processed_selection_request[key] = val

        # get the leadtime information
        leadtimes_dict = self.get_s5_leadtimes(self.granularity, max_leadtime, self.pressure_level)
        for key, val in leadtimes_dict.items():
            processed_selection_request[key] = val

        # by default, we investigate Kenya
        kenya_region = get_kenya()
        processed_selection_request['area'] = self.create_area(kenya_region)

        # TODO: do we want this level of flexibility for the user? why add complexity?
        # # update with user arguments
        # if selection_request is not None:
        #     for key, val in selection_request.items():
        #         if key in {'day', 'hour'}:
        #             warnings.warn(f'Overriding default {key} values. '
        #                           f'The ERA5Exporter assumes all days '
        #                           f'and times (the default) are being downloaded')
        #         val = self._check_iterable(val, key)
        #         processed_selection_request[key] = [self._correct_input(x, key) for x in val]

        return processed_selection_request

    @staticmethod
    def get_dataset(granularity: str,
                    pressure_level: bool) -> str:
        if granularity == 'monthly':
            return 'seasonal-monthly-pressure-levels' if pressure_level else 'seasonal-monthly-single-levels'
        elif granularity == 'hourly':
            return 'seasonal-original-pressure-levels' if pressure_level else 'seasonal-original-single-levels'

    def export(variable: str,
               pressure_level: bool,
               granularity: str = 'monthly',
               product_type: Optional[str] = None,
               dataset: Optional[str] = None,
               min_year: Optional[int] = 2017,
               max_year: Optional[int] = 2018,
               min_month: Optional[int] = 1,
               max_month: Optional[int] = 12,
               max_leadtime: Optional[int] = None,
               selection_request: Optional[Dict] = None,
               ):
        """
        Arguments
        --------
        variable: str
            the variable that you want to download.

        pressure_level: bool
            Do you want data at different atmospheric heights?
            True = yes want data at different `pressure-levels`
            False = no want only `single-level` data

        granularity: str, {'monthly', 'hourly'}, default: 'monthly'
            The granularity of data being pulled.
            'hourly'/'pressure_level' data has a forecast for every 12hrs (12, 24, 36 ...)
            'hourly'/'single_level' data has a forecast for every 6hrs (6, 12, 18, 24 ...)
            'monthly' data has a forecast for every month {1 ... 6}

        product_type : str
            The product type is only valid for monthly datasets. It corresponds
            to the post-processing of the monthly forecasts
                if pressure_level == True:
                 {'ensemble_mean', 'hindcast_climate_mean', 'monthly_mean'}
                else
                 {'ensemble_mean', 'hindcast_climate_mean', 'monthly_mean',
                  'monthly_standard_deviation', 'monthly_maximum', 'monthly_minimum'}

        min_year: Optional[int] default = 2017
            the minimum year of your request

        max_year: Optional[int] default = 2018
            the maximum year of your request

        min_month: Optional[int] default = 1
            the minimum month of your request

        max_month: Optional[int] default = 12
            the maximum month of your request

        max_leadtime: Optional[int]
            the maximum leadtime of your request
                (if granularity is `hourly` then provide in days)
                (elif granularity is `monthly` then provide in months)
            defaults to ~3 months (90 days)

        Note:
        ----
        - All parameters that are assigned to class attributes are fixed for one download
        - Only time will be chunked (by months) to send separate calls to the cdsapi
        """
        self.variable = variable
        self.pressure_level = pressure_level
        self.granularity = granularity

        if dataset is None:
            self.dataset = self.get_dataset(self.granularity, self.pressure_level)
        else:
            self.dataset = dataset

        # get the reference dictionary that corresponds to that dataset
        self.dataset_reference = dataset_reference[self.dataset]

        assert self.variable in self.dataset_reference['variable'], f"\
            Variable: {variable} is not in the valid variables for this \
            dataset. Valid variables: {self.dataset_reference['variable']}"

        # get the product type if it exists
        self.product_type = self.get_product_type(product_type)

        # max_leadtime defaults
        if max_leadtime is None:
            # set the max_leadtime to 3 months
            max_leadtime = 90 if (self.granularity == 'hourly') else 3

    #
    # def create_filename():
    #     """ """
    #     return
    #
    # def export(self,
    #            variable: str,
    #            dataset: Optional[str] = None,
    #            granularity: str = 'hourly',
    #            show_api_request: bool = True,
    #            selection_request: Optional[Dict] = None,
    #            break_up: bool = True) -> List[Path]:
    #
    #     # create the default template for the selection request
    #     processed_selection_request = self.create_selection_request(variable,
    #                                                                 selection_request,
    #                                                                 granularity)
    #     return
