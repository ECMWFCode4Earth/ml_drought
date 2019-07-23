r"""
Exporters are responsible for interacting with different data sources and downloading
data. Currently, exporters have been implemented for the current data stores:

- The `Climate Data Store <https://cds.climate.copernicus.eu/#!/home>`_
    * `ERA5 <https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5>`_
        ERA5 provides hourly estimates of a large number of atmospheric, land and oceanic climate variables
    * `S5 <https://www.ecmwf.int/sites/default/files/medialibrary/2017-10/System5_guide.pdf>`_
        S5 provides forecasts which are created by using computational models to calculate the evolution of
        the atmosphere, ocean and land surface starting from an initial state based on observations of the Earth system.

The exporters are described below.

"""

from .cds import ERA5Exporter
from .vhi import VHIExporter
from .chirps import CHIRPSExporter
from .planetOS import ERA5ExporterPOS
from .seas5.s5 import S5Exporter
from .gleam import GLEAMExporter

__all__ = [
    'ERA5Exporter', 'VHIExporter', 'ERA5ExporterPOS',
    'CHIRPSExporter', 'S5Exporter', 'GLEAMExporter'
]
