import sys

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
)

from scripts.utils import get_data_path


def export_s5(variable: str):

    granularity = "hourly"
    pressure_level = False

    exporter = S5Exporter(
        data_folder=get_data_path(),
        granularity=granularity,
        pressure_level=pressure_level,
    )

    min_year = 1993
    max_year = 2019
    min_month = 1
    max_month = 12
    max_leadtime = None
    pressure_levels = [200, 500, 925]
    n_parallel_requests = 1

    exporter.export(
        variable=variable,
        min_year=min_year,
        max_year=max_year,
        min_month=min_month,
        max_month=max_month,
        max_leadtime=max_leadtime,
        pressure_levels=pressure_levels,
        n_parallel_requests=n_parallel_requests,
    )


if __name__ == "__main__":
    export_s5("total_precipitation")
    export_s5("2m_temperature")
    export_s5("evaporation")
