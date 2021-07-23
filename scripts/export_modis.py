from typing import Optional
import sys
import numpy as np

sys.path.append("..")
from src.exporters import MantleModisExporter
from scripts.utils import get_data_path
from src.preprocess import MantleModisPreprocessor


def export_mantle_modis(year: int):
    exporter = MantleModisExporter(get_data_path())
    exporter.export(years=[year], remove_tif=True)


def preprocess_mantle_modis(subset_str: str = "india", resample_time: Optional[str] = None):
    data_path = get_data_path()

    regrid_path = (
        data_path
        / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = MantleModisPreprocessor(data_path)
    processor.preprocess(
        subset_str=subset_str, regrid=regrid_path, upsampling=False, resample_time=resample_time
    )


if __name__ == "__main__":
    years = np.arange(2000, 2021)
    years = 2001

    for year in years:
        export_mantle_modis(year=year)
        # extract india and regrid to era5 grid
        preprocess_mantle_modis()

        assert False

        # mv file so that they aren't deleted
        # delete the files in the raw data directory
