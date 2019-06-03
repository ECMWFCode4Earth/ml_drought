import cdsapi
from pathlib import Path
import certifi
import urllib3
import warnings
import itertools
import re
import numpy as np
from pprint import pprint

from typing import Dict, Optional, List

from .base import BaseExporter, Region, get_kenya
from .cds import CDSExporter

http = urllib3.PoolManager(
    cert_reqs='CERT_REQUIRED',
    ca_certs=certifi.where()
)

class S5Exporter(CDSExporter):

    @staticmethod
    def get_s5_times(granularity: str) -> Dict:
        """Returns the SEAS5 selection request arguments
        for the hourly or monthly data
        Parameters
        ----------
        granularity: str, {'monthly', 'daily'}, default: 'daily'
            The granularity of data being pulled.
            'monthly' data has a forecast every day (24, 48, 72 ...)
            'daily' data has a forecast every 6hrs (6, 12, 18, 24 ...)


        NOTE: these depend on the dataset and the naming convention is after
         the cdsapi monthly = `seasonal-monthly-single-levels` vs.
         daily = `seasonal-original-single-levels`

        Returns
        ----------
        selection_dict: dict
            A dictionary with all the time-related arguments of the
            selection dict filled out
        """
        years = [str(year) for year in range(1993, 2019 + 1)]
        months = ['{:02d}'.format(month) for month in range(1, 12 + 1)]
        days = ['{:02d}'.format(day) for day in range(1, 31 + 1)]
        times = ['{:02d}:00'.format(hour) for hour in range(24)]

        # TODO: decide if we want ALL forecast lead times (0 - 215 days?)
        if granularity == 'monthly':
            leadtime_hours = [day * 24 for day in range(1, 216)]
        elif granularity == 'daily':
            hrly_6 = np.array([6, 12, 18, 24])
            all_hrs = np.array([hrly_6 + ((day-1) * 24) for day in range(1, 216)])
            leadtime_hours = np.insert(all_hrs.flatten(), 0, 0)
            leadtime_hours = list(leadtime_hours)
        else:
            assert False, f'granularity must be in ["monthly", "daily"]\
                Currently: {granularity}'

        assert max(leadtime_hours) <= 5160, f"Max leadtime must be less \
            than 215 days (5160hrs)"

        selection_dict = {
            'year': years,
            'month': months,
            'time': times,
            'day': '01',  # forecast initialised on 1st each month
            'leadtime_hour': leadtime_hours,
        }

        return selection_dict

    @staticmethod
    def get_dataset(variable: str, granularity: str = 'daily') -> str:
        assert False, "unimplemented"
        return

    def create_selection_request(self,
                                 variable: str,
                                 selection_request: Optional[Dict] = None,
                                 granularity: str = 'daily',) -> Dict:
            """ """
            # setup the default selection request
            processed_selection_request = {
                'format': 'grib',
                'variable': [variable]
            }
            for key, val in self.get_era5_times(granularity).items():
                processed_selection_request[key] = val

            # by default, we investigate Kenya
            kenya_region = get_kenya()
            processed_selection_request['area'] = self.create_area(kenya_region)

            # update with user arguments
            if selection_request is not None:
                for key, val in selection_request.items():
                    if key in {'day', 'hour'}:
                        warnings.warn(f'Overriding default {key} values. '
                                      f'The ERA5Exporter assumes all days '
                                      f'and times (the default) are being downloaded')
                    val = self._check_iterable(val, key)
                    processed_selection_request[key] = [self._correct_input(x, key) for x in val]
            return processed_selection_request
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
