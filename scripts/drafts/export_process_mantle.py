import sys
from datetime import datetime
import numpy as np

sys.path.append("..")
from src.exporters import MantleModisExporter
# from src.exporters import MantleModisExporter
from scripts.utils import get_data_path


def export_mantle_modis(year: int):
    exporter = MantleModisExporter(get_data_path())
    exporter.export(
        years=[year],
        remove_tif=True
    )


if __name__ == "__main__":
    for year in np.arange(2001, 2021):
        export_mantle_modis(year)