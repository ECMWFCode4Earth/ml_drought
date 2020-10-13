import sys
from datetime import datetime
import numpy as np

sys.path.append("..")
from src.exporters import MantleModisExporter
from src.preprocessor import MantleModisPreprocessor
from scripts.utils import get_data_path


def export_mantle_modis(year: int):
    exporter = MantleModisExporter(get_data_path())
    exporter.export(years=[year], remove_tif=True)


def preprocess_mantle_modis(subset_str: str = "india"):
    data_path = get_data_path()

    regrid_path = (
        data_path
        / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = MantleModisPreprocessor(data_path)
    #  upsampling from low -> high resolution
    processor.preprocess(
        subset_str=subset_str, regrid=regrid_path, upsampling=False,
    )


if __name__ == "__main__":
    for year in np.arange(2001, 2021):
        export_mantle_modis(year)
