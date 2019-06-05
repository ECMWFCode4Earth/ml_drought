"""
Define the VALID SEAS5 requests from the cdsapi.

Note:
-----

    • in our schema `hourly` datasets refer to the `original` datasets
        [`seasonal-original-single-levels`, `seasonal-original-pressure-levels`]

    • `monthly` datasets have forecast resolution of months

    • `seasonal-original-single-levels` has leadtime resolution of 6 hours,
        whereas, `seasonal-original-pressure-levels` leadtime resolution = 12 hours

    • `monthly` datasets require a `'product_type'` variable whereas `hourly`
        datasets (12hr/6hr) do not

    • all datasets have fixed [format, originating_centre, system, day] variables

    • all datasets require:
        [
            format, originating_centre, system, day,
            variable, month, year
        ]

    • `monthly` datasets have extra: [product_type, leadtime_month]

    • `hourly` datasets have extra: [leadtime_hour]

    • `monthly` product_type depend on whether looking at single_level or pressure_level
        single_level = ['ensemble_mean', 'hindcast_climate_mean',
                        'monthly_maximum', 'monthly_mean',
                        'monthly_minimum', 'monthly_standard_deviation']
        pressure_level = ['ensemble_mean', 'hindcast_climate_mean', 'monthly_mean']
"""

import numpy as np
from typing import Dict, List

hrly_6 = np.array([6, 12, 18, 24])
all_hrs = np.array([hrly_6 + ((day - 1) * 24) for day in range(1, 215 + 1)])
# prepend the 0th hour to the list
leadtime_hours_6 = np.insert(all_hrs.flatten(), 0, 0)
leadtime_hours_6 = list([str(lt) for lt in leadtime_hours_6])


valid_datasets: List[str] = [
    "seasonal-original-single-levels",
    "seasonal-original-pressure-levels",
    "seasonal-monthly-single-levels",
    "seasonal-monthly-pressure-levels",
]

datasets: Dict = {
    "seasonal-original-single-levels": {
        "format": "grib",
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "10m_wind_gust_since_previous_post_processing",
            "10m_wind_speed",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "east_west_surface_stress_rate_of_accumulation",
            "evaporation",
            "maximum_2m_temperature_in_the_last_24_hours",
            "mean_sea_level_pressure",
            "minimum_2m_temperature_in_the_last_24_hours",
            "north_south_surface_stress_rate_of_accumulation",
            "runoff",
            "sea_ice_cover",
            "sea_surface_temperature",
            "snow_density",
            "snow_depth",
            "snowfall",
            "soil_temperature_level_1",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
            "surface_solar_radiation",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation",
            "surface_thermal_radiation_downwards",
            "top_solar_radiation",
            "top_thermal_radiation",
            "total_cloud_cover",
            "total_precipitation",
        ],
        "pressure_level": None,
        "product_type": None,
        "leadtime_month": None,
        "leadtime_hour": leadtime_hours_6,
        "day": "01",
        "month": [str(m) for m in range(1, 13)],
        "year": [str(y) for y in range(1993, 2020)],
    },
    "seasonal-original-pressure-levels": {
        "format": "grib",
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": [
            "10",
            "30",
            "50",
            "100",
            "200",
            "300",
            "400",
            "500",
            "700",
            "850",
            "925",
        ],
        "product_type": None,
        "leadtime_month": None,
        "leadtime_hour": [str(day * 24) for day in range(1, 215 + 1)],
        "day": "01",
        "month": [str(m) for m in range(1, 13)],
        "year": [str(y) for y in range(1993, 2020)],
    },
    "seasonal-monthly-single-levels": {
        "format": "grib",
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "10m_wind_gust_since_previous_post_processing",
            "10m_wind_speed",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "east_west_surface_stress_rate_of_accumulation",
            "evaporation",
            "maximum_2m_temperature_in_the_last_24_hours",
            "mean_sea_level_pressure",
            "minimum_2m_temperature_in_the_last_24_hours",
            "north_south_surface_stress_rate_of_accumulation",
            "runoff",
            "sea_ice_cover",
            "sea_surface_temperature",
            "snow_density",
            "snow_depth",
            "snowfall",
            "soil_temperature_level_1",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
            "surface_solar_radiation",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation",
            "surface_thermal_radiation_downwards",
            "top_solar_radiation",
            "top_thermal_radiation",
            "total_cloud_cover",
            "total_precipitation",
        ],
        "pressure_level": None,
        "product_type": [
            "ensemble_mean",
            "hindcast_climate_mean",
            "monthly_maximum",
            "monthly_mean",
            "monthly_minimum",
            "monthly_standard_deviation",
        ],
        "leadtime_month": [str(m) for m in range(1, 7)],
        "leadtime_hour": None,
        "day": "01",
        "month": [str(m) for m in range(1, 13)],
        "year": [str(y) for y in range(1993, 2020)],
    },
    "seasonal-monthly-pressure-levels": {
        "format": "grib",
        "originating_centre": "ecmwf",
        "system": "5",
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": [
            "10",
            "30",
            "50",
            "100",
            "200",
            "300",
            "400",
            "500",
            "700",
            "850",
            "925",
        ],
        "product_type": ["ensemble_mean", "hindcast_climate_mean", "monthly_mean"],
        "leadtime_month": [str(m) for m in range(1, 7)],
        "leadtime_hour": None,
        "day": "01",
        "month": [str(m) for m in range(1, 13)],
        "year": [str(y) for y in range(1993, 2020)],
    },
}
