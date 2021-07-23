from typing import Optional
import sys
import numpy as np
import shutil

sys.path.append("..")
from src.exporters import MantleModisExporter
from scripts.utils import get_data_path
from src.preprocess import MantleModisPreprocessor


def export_mantle_modis(year: int):
    exporter = MantleModisExporter(get_data_path())
    exporter.export(years=[year], remove_tif=True)


def preprocess_mantle_modis(
    subset_str: str = "india", resample_time: Optional[str] = None
):
    data_path = get_data_path()

    regrid_path = (
        data_path
        / f"interim/reanalysis-era5-land-monthly-means_preprocessed/2m_temperature_data_{subset_str}.nc"
    )
    assert regrid_path.exists(), f"{regrid_path} not available"

    processor = MantleModisPreprocessor(data_path)
    processor.preprocess(
        subset_str=subset_str,
        regrid=regrid_path,
        upsampling=False,
        resample_time=resample_time,
    )


if __name__ == "__main__":
    data_dir = get_data_path()
    raw_data_dir = data_dir / "raw/mantle_modis"
    interim_data_dir = data_dir / "interim"
    subset_str = "india" 

    years = np.arange(2000, 2021)
    years = [2001]

    for year in years:
        # export data
        export_mantle_modis(year=year)

        # extract india and regrid to era5 grid
        regrid_path = (
            data_dir
            / "interim/reanalysis-era5-land-monthly-means_preprocessed" \
            f"/2m_temperature_data_{subset_str}.nc"
        )
        assert regrid_path.exists(), f"{regrid_path} not available"
        processor = MantleModisPreprocessor(data_dir)
        # get the filepaths for all of the downloaded data
        nc_files = processor.get_filepaths()
        if regrid_path is not None:
            regrid = processor.load_reference_grid(regrid_path)
        else:
            regrid = None

            for file in nc_files:
                processor._preprocess_single(
                    netcdf_filepath=file, subset_str=subset_str, regrid=regrid
                )

        # # mv file so that they aren't deleted
        # preprocess_file = data_dir / "interim/mantle_modis_preprocessed/data_india.nc"
        # destination_file = (
        #     data_dir / f"interim/ALL_mantle_modis_india/{year}_data_india.nc"
        # )
        # destination_file.parent.mkdir(parents=True, exist_ok=True)
        # shutil.move(src=preprocess_file, dst=destination_file)
        # # delete original preprocess file
        # try:
        #     preprocess_file.unlink()
        # except FileNotFoundError:
        #     print(f"Already moved {preprocess_file} to {destination_file}")

        # delete the nearest_s2d file
        [f.unlink() for f in interim_data_dir.glob("nearest_s2d*.nc")]

        #  delete the files in the raw data directory
        raw_files = list(raw_data_dir.glob("**/*.nc")) + list(
            raw_data_dir.glob("**/*.nc")
        )
        for f in raw_files:
            try:
                f.unlink()
            except FileNotFoundError:
                print(f"Already deleted {f}")

        print("\n\n*******************************")
        print(f"Download and regrid MODIS for {year}")
        print("\n\n*******************************")
