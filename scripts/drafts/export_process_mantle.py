import sys
from datetime import datetime
import numpy as np

sys.path.append("../..")
from src.exporters import MantleModisExporter
from src.preprocess import MantleModisPreprocessor
from scripts.utils import get_data_path


def export_mantle_modis(year: int, level: str = "OF"):
    exporter = MantleModisExporter(get_data_path())
    exporter.export(years=[year], remove_tif=True, level=level)


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
        cleanup=False, with_merge=False
    )


def merge_all_mantle_modis(subset_str: str = "india"):
    # get attributes for processor
    data_path = get_data_path()
    resample_time = "M"
    upsampling = False

    # merge the created files
    processor = MantleModisPreprocessor(data_path)
    processor.merge_files(subset_str, resample_time, upsampling)


if __name__ == "__main__":
    data_dir = get_data_path()
    for year in np.arange(2001, 2021):
        export_mantle_modis(year, level="OF")
        preprocess_mantle_modis(
            subset_str="india"
        )

        # DELETE remaining files (tif/nc)
        tif_files = (data_dir / "raw/mantle_modis").glob("**/*.tif")
        nc_files = (data_dir / "raw/mantle_modis").glob("**/*.nc")
        [f.unlink() for f in tif_files]
        [f.unlink() for f in nc_files]
        print(f"\n-- FINISHED: {year} --\n")

    merge_all_mantle_modis(subset_str="india")
