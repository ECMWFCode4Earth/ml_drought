from pathlib import Path
from typing import Optional, Dict, List, cast
import warnings
import multiprocessing
import itertools

from .cds import CDSExporter
from .base import get_kenya

VALID_ERA5_LAND_VARS = [
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    '2m_dewpoint_temperature', '2m_temperature',
    'evaporation_from_bare_soil',
    'evaporation_from_open_water_surfaces_excluding_oceans',
    'evaporation_from_the_top_of_canopy',
    'evaporation_from_vegetation_transpiration',
    'evapotranspiration', 'forecast_albedo', 'lake_bottom_temperature',
    'lake_ice_depth', 'lake_ice_temperature', 'lake_mix_layer_depth',
    'lake_mix_layer_temperature', 'lake_shape_factor',
    'lake_total_layer_temperature', 'leaf_area_index_high_vegetation',
    'leaf_area_index_low_vegetation', 'potential_evaporation', 'runoff',
    'skin_reservoir_content', 'skin_temperature', 'snow_albedo', 'snow_cover',
    'snow_density', 'snow_depth', 'snow_depth_water_equivalent',
    'snow_evaporation', 'snowfall', 'snowmelt', 'soil_temperature_level_1',
    'soil_temperature_level_2', 'soil_temperature_level_3',
    'soil_temperature_level_4', 'sub_surface_runoff',
    'surface_latent_heat_flux', 'surface_net_solar_radiation',
    'surface_net_thermal_radiation', 'surface_pressure', 'surface_runoff',
    'surface_sensible_heat_flux', 'surface_solar_radiation_downwards',
    'surface_thermal_radiation_downwards', 'temperature_of_snow_layer',
    'total_precipitation', 'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4'
]


class ERA5LandExporter(CDSExporter):
    dataset = 'reanalysis-era5-land'
    granularity = 'hourly'

    @staticmethod
    def print_valid_vars():
        print(VALID_ERA5_LAND_VARS)

    def create_selection_request(self,
                                 variable: str,
                                 selection_request: Optional[Dict] = None,
                                 granularity: str = 'hourly',) -> Dict:
        # setup the default selection request
        assert variable in VALID_ERA5_LAND_VARS, "Need to select a variable " \
            f"from {VALID_ERA5_LAND_VARS}"

        processed_selection_request = {
            'format': 'netcdf',
            'variable': [variable]
        }
        for key, val in self.get_default_era5_times(granularity, land=True).items():
            processed_selection_request[key] = val

        # by default, we investigate Kenya
        kenya_region = get_kenya()
        processed_selection_request['area'] = self.create_area(kenya_region)

        # update with user arguments
        if selection_request is not None:
            for key, val in selection_request.items():
                if key in {'day', 'hour'}:
                    warnings.warn(f'Overriding default {key} values. '
                                  f'The ERA5LandExporter assumes all days '
                                  f'and times (the default) are being downloaded')
                val = self._check_iterable(val, key)
                processed_selection_request[key] = [self._correct_input(x, key) for x in val]
        return processed_selection_request

    def export(self,
               variable: str,
               show_api_request: bool = True,
               selection_request: Optional[Dict] = None,
               break_up: Optional[str] = 'yearly',
               n_parallel_requests: int = 1) -> List[Path]:
        """ Export functionality to prepare the API request and to send it to
        the cdsapi.client() object.

        Arguments:
        ---------
        variable: str
            The variable to be exported
        show_api_request: bool = True
            Whether to print the selection dictionary before making the API request
        selection_request: Optional[Dict], default = None
            Selection request arguments to be merged with the defaults. If both a key is
            defined in both the selection_request and the defaults, the value in the
            selection_request takes precedence.
        break_up: str: {'yearly', 'monthly'}, default = 'yearly'
            The best way to download the data is by relatively large calls to the CDS
            API. If specified, the calls will be broken up by {'yearly', 'monthly'}
        parallel: bool, default = True
            Whether to download data in parallel
        n_parallel_requests:
            How many parallel requests to the CDSAPI to make

        Returns:
        -------
        output_files: List of pathlib.Paths
            paths to the downloaded data
        """
        dataset = self.dataset
        granularity = self.granularity

        if break_up is not None:
            assert break_up in ['yearly', 'monthly'], f'Only 2 valid ways to ' \
                'break up requests: {"yearly", "monthly"}'

        # create the default template for the selection request
        processed_selection_request = self.create_selection_request(variable,
                                                                    selection_request,
                                                                    granularity)

        if n_parallel_requests < 1: n_parallel_requests = 1

        # break up by month
        if break_up == 'monthly':
            if n_parallel_requests > 1:  # Run in parallel
                p = multiprocessing.Pool(int(n_parallel_requests))

            output_paths = []
            for year, month in itertools.product(processed_selection_request['year'],
                                                 processed_selection_request['month']):
                updated_request = processed_selection_request.copy()
                updated_request['year'] = [year]
                updated_request['month'] = [month]

                if n_parallel_requests > 1:  # Run in parallel
                    # multiprocessing of the paths
                    output_paths.append(
                        p.apply_async(
                            self._export,
                            args=(dataset, updated_request, show_api_request, True)
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

        if break_up == 'yearly':
            if n_parallel_requests > 1:  # Run in parallel
                p = multiprocessing.Pool(int(n_parallel_requests))

            output_paths = []
            for year in processed_selection_request['year']:
                updated_request = processed_selection_request.copy()
                updated_request['year'] = [year]

                if n_parallel_requests > 1:  # Run in parallel
                    # multiprocessing of the paths
                    output_paths.append(
                        p.apply_async(
                            self._export,
                            args=(dataset, updated_request, show_api_request, True)
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

        return [self._export(dataset,
                             processed_selection_request,
                             show_api_request)]
