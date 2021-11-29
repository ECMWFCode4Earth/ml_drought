import sys
import numpy as np

sys.path.append("..")
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
    ERA5LandExporter,
    BokuNDVIExporter,
)

from scripts.utils import get_data_path


def export_era5(region_str="kenya"):
    exporter = ERA5Exporter(get_data_path())

    # The ERA5 exporter downloads the data with wierd names.
    # A mapping of actual variables to the downloaded variable
    # names is recorded here
    name2var = {
        "precip": "precip",
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

    era5_variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
        "surface_pressure",
        "surface_sensible_heat_flux",
        "surface_latent_heat_flux",
        "soil_temperature_level_1",
        "2m_temperature",
        "mean_eastward_turbulent_surface_stress",
        "mean_northward_turbulent_surface_stress",
        "surface_net_solar_radiation_clear_sky",
        "surface_net_thermal_radiation_clear_sky",
        "vertical_integral_of_divergence_of_moisture_flux",
        "potential_evaporation",
        "evaporation",
    ]

    for variable in era5_variables:
        exporter.export(variable=variable, granularity="monthly", region_str=region_str)


def export_era5_land(region_str="kenya", granularity: str = "monthly"):
    exporter = ERA5LandExporter(get_data_path())

    variables = [
        # "total_precipitation",
        # "2m_temperature",
        # "evapotranspiration",
        # "potential_evaporation",
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
    ]
    for variable in variables:
        exporter.export(
            variable=variable,
            break_up="yearly",
            region_str=region_str,
            granularity=granularity,
            selection_request=dict(year=np.arange(2000, 2021)),
        )


def export_era5_single_var(variable="total_precipitation"):
    # if the working directory is alread ml_drought don't need ../data
    if Path(".").absolute().as_posix().split("/")[-1] == "ml_drought":
        data_path = Path("data")
    else:
        data_path = Path("../data")

    exporter = ERA5Exporter(data_path)
    exporter.export(variable=variable, granularity="monthly")


def export_vhi():
    exporter = VHIExporter(get_data_path())

    exporter.export(years=np.arange(2000, 2021))


def export_chirps():
    exporter = CHIRPSExporter(get_data_path())

    exporter.export(years=None, region="global", period="monthly")


def export_era5POS():
    exporter = ERA5ExporterPOS(get_data_path())

    variables = [
        "air_temperature_at_2_metres",
        "precipitation_amount_1hour_Accumulation",
    ]

    for variable in variables:
        exporter.export(variable=variable)


def export_gleam():
    exporter = GLEAMExporter(data_folder=get_data_path())
    exporter.export(["E", "SMroot", "SMsurf"], "monthly")


def export_srtm():
    exporter = SRTMExporter(data_folder=get_data_path())
    exporter.export()


def export_esa():

    exporter = ESACCIExporter(data_folder=get_data_path())
    exporter.export()


def export_s5(region_str="kenya"):

    granularity = "monthly"
    pressure_level = False

    exporter = S5Exporter(
        data_folder=get_data_path(),
        granularity=granularity,
        pressure_level=pressure_level,
    )
    min_year = 1993
    max_leadtime = None
    pressure_levels = None  # [200, 500, 925]
    n_parallel_requests = 1
    for variable in variables:
        print(f"\n\nWORKING ON: {variable}\n\n")
        exporter.export(
            variable=variable,
            min_year=min_year,
            max_year=max_year,
            min_month=min_month,
            max_month=max_month,
            max_leadtime=max_leadtime,
            pressure_levels=pressure_levels,
            n_parallel_requests=n_parallel_requests,
            region_str=region_str,
            break_up=False,
        )


def export_kenya_boundaries():

    exporter = KenyaAdminExporter(get_data_path())
    exporter.export()


def export_boku_ndvi():
    exporter = BokuNDVIExporter(get_data_path())
    exporter.export()


if __name__ == "__main__":
    print(f"Writing data to: {get_data_path()}")
    export_era5_land(region_str="gb")
    # export_era5(region_str="kenya")
    # export_vhi()
    # export_chirps()
    # export_era5POS()
    # export_gleam()
    # export_esa()
    # export_s5(region_str="kenya")
    # export_kenya_boundaries()
    # export_boku_ndvi()
