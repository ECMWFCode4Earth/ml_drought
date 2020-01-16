import sys

sys.path.append("../..")

from src.exporters import (
    ERA5Exporter,
    VHIExporter,
    CHIRPSExporter,
    ERA5ExporterPOS,
    GLEAMExporter,
    ESACCIExporter,
    S5Exporter,
    SRTMExporter,
    KenyaAdminExporter,
)

from scripts.utils import get_data_path


def export_era5(variables):
    exporter = ERA5Exporter(get_data_path())

    # The ERA5 exporter downloads the data with wierd names.
    # A mapping of actual variables to the downloaded variable
    # names is recorded here
    name2var = {
        "precip": "precip",
        "total_precipitation": "total_precipitation",
        "evaporation": "e",
        "mean_eastward_turbulent_surface_stress": "metss",
        "mean_northward_turbulent_surface_stress": "mntss",
        "potential_evaporation": "pev",
        "slhf": "surface_latent_heat_flux",
        "sp": "surface_pressure",
        "sshf": "surface_sensible_heat_flux",
        "ssrc": "surface_net_solar_radiation_clear_sky",
        "stl1": "soil_temperature_level_1",
        "strc": "surface_net_thermal_radiation_clear_sky",
        "swvl1": "volumetric_soil_water_layer_1",
        "swvl2": "volumetric_soil_water_layer_2",
        "swvl3": "volumetric_soil_water_layer_3",
        "swvl4": "volumetric_soil_water_layer_4",
        "t2m": "2m_temperature",
        "u10": "10m_u_component_of_wind",
        "v10": "10m_v_component_of_wind",
        "p84.162": "vertical_integral_of_divergence_of_moisture_flux",
        "VCI": "VCI",
    }

    for variable in variables:
        exporter.export(variable=variable, granularity="hourly", break_up=True)


if __name__ == "__main__":
    variables = [
        "2m_temperature",
        # "potential_evaporation",
        # "evaporation",
        # "total_precipitation"
        # "volumetric_soil_water_layer_1",
        # "volumetric_soil_water_layer_2",
        # "volumetric_soil_water_layer_3",
        # "volumetric_soil_water_layer_4",
    ]

    export_era5(variables)
